//! AIR for the SHA-256 compression function.
//!
//! # Overview
//!
//! One trace row encodes one compression.
//!
//! No transition constraints: rows are independent, so padding is free.
//!
//! # Compression function recap
//!
//! Inputs:
//! - Chaining state `H[0..8]` (eight 32-bit words).
//! - Message block `W[0..16]` (sixteen 32-bit words).
//!
//! Output:
//! - New chaining state `H'[0..8] = H[i] + state[i] (mod 2^32)`.
//!
//! Pipeline:
//!
//! ```text
//!              +----------------------+
//!      W[0..16]|                      |
//!      ------->|   message schedule   |---> W[0..64]
//!              |                      |
//!              +----------------------+
//!                         |
//!                         v
//!              +----------------------+
//!              |                      |
//!      H[0..8] |  64 compression      |   (a, b, c, d, e, f, g, h)_64
//!      ------->|      rounds          |---------------------------->
//!              |                      |
//!              +----------------------+
//!                         |
//!                         v
//!              +----------------------+
//!              |    add H + state     |---> H'[0..8]
//!              +----------------------+
//! ```
//!
//! # Representation choices
//!
//! - Bitwise operations commit unpacked columns, 32 bits per word.
//! - Modular additions commit packed columns, 2 x 16-bit limbs per word.
//! - Bridge constraints keep the two views consistent whenever both are needed.
//! - The output state `H'` commits unpacked: nothing reads its limbs back.
//!
//! # Constraint degree
//!
//! - Boolean checks: degree 2.
//! - Packing identities: degree 1.
//! - `Ch`: degree 2.
//! - `Maj`, sigma XOR3 identities, add2 / add3 helpers: degree 3.

#![no_std]

extern crate alloc;

mod air;
mod columns;
mod constants;
mod generation;

pub use air::*;
pub use columns::*;
pub use constants::*;
pub use generation::*;

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use p3_air::{AirLayout, BaseAir, check_constraints, get_max_constraint_degree};
    use p3_baby_bear::BabyBear;
    use p3_challenger::{HashChallenger, SerializingChallenger32};
    use p3_commit::ExtensionMmcs;
    use p3_field::extension::BinomialExtensionField;
    use p3_field::{PrimeCharacteristicRing, PrimeField32};
    use p3_fri::{FriParameters, TwoAdicFriPcs};
    use p3_keccak::{Keccak256Hash, KeccakF};
    use p3_merkle_tree::MerkleTreeMmcs;
    use p3_symmetric::{CompressionFunctionFromHasher, PaddingFreeSponge, SerializingHasher};
    use p3_uni_stark::{StarkConfig, prove, verify};
    use proptest::prelude::*;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;

    type F = BabyBear;
    type Challenge = BinomialExtensionField<F, 4>;

    /// Reference oracle: call into the `sha2` crate's raw compression.
    ///
    /// Lets every test compare our generator against a widely-trusted
    /// implementation instead of re-deriving expected values by hand.
    fn sha2_compress_reference(block: [u32; 16], h_in: [u32; 8]) -> [u32; 8] {
        // The reference implementation takes the block as 64 big-endian
        // bytes, so serialize each word accordingly.
        let mut block_bytes = [0u8; 64];
        for (i, word) in block.iter().enumerate() {
            block_bytes[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
        }
        let mut state = h_in;
        // `compress256` operates in place on the chaining state.
        sha2::block_api::compress256(&mut state, core::slice::from_ref(&block_bytes));
        state
    }

    /// Rebuild the 8 output words from a populated row.
    ///
    /// Packs the 32 little-endian bit columns back into a single `u32`.
    fn extract_output(cols: &Sha256Cols<F>) -> [u32; 8] {
        core::array::from_fn(|i| {
            // Fold the bits high to low so bit `j` lands in position `j`.
            cols.h_out[i]
                .iter()
                .rev()
                .fold(0u32, |acc, bit| (acc << 1) | bit.as_canonical_u32())
        })
    }

    /// Reinterpret a raw trace buffer as mutable typed rows.
    ///
    /// Used by the negative tests to corrupt a single column in place.
    fn rows_mut(trace: &mut p3_matrix::dense::RowMajorMatrix<F>) -> &mut [Sha256Cols<F>] {
        // Safe: same layout guarantee as the shared view above.
        let (prefix, rows, suffix) = unsafe { trace.values.align_to_mut::<Sha256Cols<F>>() };
        assert!(prefix.is_empty());
        assert!(suffix.is_empty());
        rows
    }

    /// Build a single-row trace over a fixed input, ready to be tampered with.
    fn tamperable_trace() -> p3_matrix::dense::RowMajorMatrix<F> {
        // Zero block on the canonical IV - any valid input would do.
        generate_trace_rows::<F>(vec![concat_input([0u32; 16], SHA256_IV)], 0)
    }

    /// Expose the raw buffer as typed column rows.
    fn rows(trace: &p3_matrix::dense::RowMajorMatrix<F>) -> &[Sha256Cols<F>] {
        // Safe: the trace was allocated with exactly the column struct layout.
        let (prefix, rows, suffix) = unsafe { trace.values.align_to::<Sha256Cols<F>>() };
        assert!(prefix.is_empty());
        assert!(suffix.is_empty());
        rows
    }

    /// Build a 24-word input from a 16-word block plus an 8-word state.
    fn concat_input(block: [u32; 16], h_in: [u32; 8]) -> [u32; INPUT_WORDS] {
        // Layout: [block (16 words) | state (8 words)].
        let mut out = [0u32; INPUT_WORDS];
        out[..16].copy_from_slice(&block);
        out[16..].copy_from_slice(&h_in);
        out
    }

    // Column layout.

    #[test]
    fn column_layout_is_flat() {
        use core::mem::size_of;
        // Invariant: `repr(C)` packs the row struct with no padding, so the
        // byte size over u8 must equal the declared column count.
        //
        // A mismatch would mean every `align_to::<Sha256Cols<F>>` call would
        // be unsound.
        let expected = NUM_SHA256_COLS;
        let actual = size_of::<Sha256Cols<u8>>();
        assert_eq!(expected, actual);
    }

    #[test]
    fn declared_constraint_degree_matches_symbolic_degree() {
        // Invariant: the hint lets the prover skip symbolic evaluation.
        //
        //     hint >= true degree  ->  sound
        //     hint <  true degree  ->  silently invalid proofs
        //
        // Pin it against the degree symbolic evaluation would infer.
        let air = Sha256Air;
        let layout = AirLayout::from_air::<F>(&air);
        let symbolic = get_max_constraint_degree::<F, _>(&air, layout, 1 << 4);
        assert_eq!(BaseAir::<F>::max_constraint_degree(&air), Some(symbolic));
    }

    // Trace generator correctness.

    #[test]
    fn trace_matches_sha2_reference_fixed_iv() {
        // Canonical SHA-256 starting state paired with an all-zero block.
        //
        //     block   = [0; 16]
        //     h_in    = SHA256_IV   (FIPS 180-4 Section 5.3.3)
        //
        // Why: cross-checks against a fixed test vector that anyone can
        // reproduce from the spec.
        let block = [0u32; 16];
        let h_in = SHA256_IV;

        let inputs = vec![concat_input(block, h_in)];
        let trace = generate_trace_rows::<F>(inputs, 0);
        let rows_view = rows(&trace);

        // Compare our generator output against the sha2 crate.
        let expected = sha2_compress_reference(block, h_in);
        let got = extract_output(&rows_view[0]);
        assert_eq!(got, expected);
    }

    #[test]
    fn trace_matches_sha2_reference_all_ones() {
        // All bits set in both block and chaining state.
        //
        //     block   = [2^32 - 1; 16]
        //     h_in    = [2^32 - 1; 8]
        //
        // Why: drives every addition into the overflow path and exercises
        // the full range of every bit column.
        let block = [u32::MAX; 16];
        let h_in = [u32::MAX; 8];

        let inputs = vec![concat_input(block, h_in)];
        let trace = generate_trace_rows::<F>(inputs, 0);
        let rows_view = rows(&trace);

        let expected = sha2_compress_reference(block, h_in);
        let got = extract_output(&rows_view[0]);
        assert_eq!(got, expected);
    }

    #[test]
    fn trace_matches_sha2_reference_deterministic_random_batch() {
        // Four deterministic-random inputs packed into one trace.
        //
        // Why: checks that multi-row traces are populated correctly and
        // stresses the parallel row filler.

        let mut rng = SmallRng::seed_from_u64(42);
        let inputs: Vec<[u32; INPUT_WORDS]> = (0..4).map(|_| rng.random()).collect();
        let trace = generate_trace_rows::<F>(inputs.clone(), 0);
        let rows_view = rows(&trace);

        // Per-row comparison against the reference implementation.
        for (i, input) in inputs.iter().enumerate() {
            let block: [u32; 16] = core::array::from_fn(|j| input[j]);
            let h_in: [u32; 8] = core::array::from_fn(|j| input[16 + j]);

            let expected = sha2_compress_reference(block, h_in);
            let got = extract_output(&rows_view[i]);
            assert_eq!(got, expected, "Row {i} mismatch against sha2");
        }
    }

    // Constraint satisfaction.

    #[test]
    fn check_constraints_pass_for_random_inputs() {
        // Invariant: every trace our generator emits must be accepted by the
        // AIR constraints.
        //
        // Fixture: 4 deterministic random rows (power of two, exercises SIMD
        // path in the parallel filler).
        let air = Sha256Air;
        let trace = air.generate_trace_rows::<F>(4, 0);
        check_constraints(&air, &trace, &[]);
    }

    #[test]
    fn check_constraints_pass_for_iv_and_zero_block() {
        // Fixture: canonical start state, zero block, replicated to two rows
        // so we stay at a power-of-two row count.
        let block = [0u32; 16];
        let h_in = SHA256_IV;

        let inputs = vec![concat_input(block, h_in); 2];
        let trace = generate_trace_rows::<F>(inputs, 0);
        check_constraints(&Sha256Air, &trace, &[]);
    }

    #[test]
    fn check_constraints_pass_at_boundary_values() {
        // Mix of extreme inputs packed into a single 4-row trace.
        //
        //     row 0: block = 0,       state = 0
        //     row 1: block = 2^32-1,  state = 2^32-1
        //     row 2: block = walking 1-bit pattern, state = SHA256_IV
        //     row 3: block = SHA256_IV padded to 16 words, state = SHA256_IV
        //
        // Why: these patterns saturate every bit-level code path at least
        // once.
        let inputs = vec![
            concat_input([0u32; 16], [0u32; 8]),
            concat_input([u32::MAX; 16], [u32::MAX; 8]),
            concat_input(core::array::from_fn(|i| 1u32 << (i % 32)), SHA256_IV),
            concat_input(
                SHA256_IV
                    .into_iter()
                    .chain(core::iter::repeat(0))
                    .take(16)
                    .collect::<Vec<u32>>()
                    .try_into()
                    .unwrap(),
                SHA256_IV,
            ),
        ];
        let trace = generate_trace_rows::<F>(inputs, 0);
        check_constraints(&Sha256Air, &trace, &[]);
    }

    // Constraint rejection.

    #[test]
    #[should_panic(expected = "constraints not satisfied")]
    fn constraints_reject_non_boolean_output_column() {
        // Invariant: every `h_out` column is boolean.
        //
        // Bounding the columns is what bounds the two repacked limbs.
        //
        // A packed `h_out` would admit a second decomposition instead:
        //
        //     lo -= 2^16    hi += 1
        //
        // Both addition checks accept that shift.
        let mut trace = tamperable_trace();
        rows_mut(&mut trace)[0].h_out[0][0] = F::TWO;
        check_constraints(&Sha256Air, &trace, &[]);
    }

    #[test]
    #[should_panic(expected = "constraints not satisfied")]
    fn constraints_reject_flipped_output_bit() {
        // Flipping any output bit changes the repacked word.
        // The finalization addition must reject it.
        let mut trace = tamperable_trace();
        let bit = &mut rows_mut(&mut trace)[0].h_out[7][31];
        *bit = F::ONE - *bit;
        check_constraints(&Sha256Air, &trace, &[]);
    }

    #[test]
    #[should_panic(expected = "constraints not satisfied")]
    fn constraints_reject_flipped_schedule_bit() {
        // The expanded schedule words are witnessed, not derived.
        // Only the recurrence ties them back to the block.
        let mut trace = tamperable_trace();
        let bit = &mut rows_mut(&mut trace)[0].w[16][0];
        *bit = F::ONE - *bit;
        check_constraints(&Sha256Air, &trace, &[]);
    }

    // End-to-end proof.

    #[test]
    fn prove_and_verify_small_trace() {
        // Strongest integration check: generate a real proof and verify it.
        //
        // Fixture: 2 rows - the smallest power-of-two count that still
        // exercises the LDE domain.

        let air = Sha256Air;

        // Hash primitives used to commit and challenge.
        let byte_hash = Keccak256Hash {};
        let u64_hash = PaddingFreeSponge::<KeccakF, 25, 17, 4>::new(KeccakF {});
        let field_hash = SerializingHasher::new(u64_hash);
        let compress = CompressionFunctionFromHasher::<_, 2, 4>::new(u64_hash);

        // Vector commitment scheme over the base field.
        let val_mmcs = MerkleTreeMmcs::<
            [F; p3_keccak::VECTOR_LEN],
            [u64; p3_keccak::VECTOR_LEN],
            _,
            _,
            2,
            4,
        >::new(field_hash, compress, 3);

        // Vector commitment scheme over the challenge extension.
        let challenge_mmcs = ExtensionMmcs::<F, Challenge, _>::new(val_mmcs.clone());

        // Fiat-Shamir challenger reseeded with an empty state.
        let challenger =
            SerializingChallenger32::<F, HashChallenger<u8, _, 32>>::from_hasher(vec![], byte_hash);

        // FRI benchmark parameters: conservative blowup, no proof-of-work.
        let fri_params = FriParameters::new_benchmark(challenge_mmcs);

        // Build the trace with room for the LDE blowup.
        let trace = air.generate_trace_rows::<F>(2, fri_params.log_blowup);

        // Plumb everything into the uni-stark config.
        let dft = p3_dft::Radix2Bowers;
        let pcs = TwoAdicFriPcs::new(dft, val_mmcs, fri_params);
        let config = StarkConfig::new(pcs, challenger);

        // Prove: build a STARK proof over the committed trace.
        let proof = prove(&config, &air, trace, &[]);
        // Verify: the same config must accept the proof.
        verify(&config, &air, &proof, &[]).expect("Verification failed");
    }

    // Property-based tests.

    proptest! {
        // 16 random cases per property - enough to catch shift-by-one bugs
        // without making the suite sluggish.
        #![proptest_config(ProptestConfig { cases: 16, .. ProptestConfig::default() })]

        #[test]
        fn proptest_generator_matches_sha2(
            block in any::<[u32; 16]>(),
            h_in in any::<[u32; 8]>(),
        ) {
            // For any valid input pair, the generator's output must match the
            // sha2 crate bit-for-bit.
            let inputs = vec![concat_input(block, h_in)];
            let trace = generate_trace_rows::<F>(inputs, 0);
            let rows_view = rows(&trace);
            let got = extract_output(&rows_view[0]);
            let expected = sha2_compress_reference(block, h_in);
            prop_assert_eq!(got, expected);
        }

        #[test]
        fn proptest_constraints_accept_generated_trace(
            block in any::<[u32; 16]>(),
            h_in in any::<[u32; 8]>(),
        ) {
            // Sibling property: every generated trace must pass the AIR
            // constraint checker, regardless of the input.
            let inputs = vec![concat_input(block, h_in)];
            let trace = generate_trace_rows::<F>(inputs, 0);
            check_constraints(&Sha256Air, &trace, &[]);
        }
    }
}
