//! Algebraic Intermediate Representation (AIR) for the Poseidon1 permutation.
//!
//! # Overview
//!
//! This crate arithmetizes the [Poseidon1] hash permutation for use in STARK proof systems.
//! Each row of the execution trace computes **one full permutation** of the state.
//!
//! [Poseidon1]: https://eprint.iacr.org/2019/458
//!
//! # Poseidon1 Permutation Structure
//!
//! Poseidon1 is a substitution-permutation network (SPN) over `GF(p)^t` using the
//! [HADES] design strategy. Each permutation consists of three phases:
//!
//! ```text
//!   ┌──────────────────────┐   ┌──────────────────────┐   ┌──────────────────────┐
//!   │  RF/2 Full Rounds    │──▶│   RP Partial Rounds   │──▶│  RF/2 Full Rounds    │
//!   │  (beginning)         │   │   (middle)            │   │  (ending)            │
//!   └──────────────────────┘   └──────────────────────┘   └──────────────────────┘
//! ```
//!
//! ## Full Rounds
//!
//! Apply the S-box to **every** element of the state, then multiply by the MDS matrix.
//! These rounds provide resistance against statistical attacks (differential, linear).
//!
//! ```text
//!   state → AddRoundConstants → S-box(all elements) → MDS multiply → state'
//! ```
//!
//! ## Partial Rounds
//!
//! Apply the S-box to **only the first** element (`state[0]`), then multiply by the
//! MDS matrix. These rounds are cheaper but still increase the algebraic degree
//! sufficiently to resist algebraic attacks (interpolation, Groebner basis).
//!
//! ```text
//!   state → AddConstant(state[0]) → S-box(state[0]) → MDS multiply → state'
//! ```
//!
//! ## S-box
//!
//! The S-box is the power map `x → x^α`, where `α ≥ 3` is the smallest positive
//! integer satisfying `gcd(α, p - 1) = 1`. Supported degrees: 3, 5, 7, and 11.
//!
//! # Design Characteristics
//!
//! - **No initial linear layer.** The permutation starts directly with full rounds.
//! - **Dense MDS matrix.** A full `t × t` MDS matrix (typically circulant) is used
//!   in full rounds, fully mixing all state elements.
//! - **Karatsuba convolution.** Full-round MDS multiplies use Karatsuba convolution
//!   for supported widths (16, 24), reducing cost from O(t²) to O(t log²t).
//! - **Sparse partial rounds.** Partial rounds use the sparse matrix decomposition
//!   from Appendix B of the Poseidon1 paper, storing only 1 committed value per
//!   round (the S-box output for `state[0]`) instead of the full WIDTH-element
//!   post-state.

#![no_std]

extern crate alloc;

mod air;
mod columns;
mod generation;
mod vectorized;

pub use air::*;
pub use columns::*;
pub use generation::*;
pub use p3_poseidon1::external::FullRoundConstants;
pub use p3_poseidon1::internal::PartialRoundConstants;
pub use vectorized::*;

#[cfg(test)]
mod tests {
    use alloc::vec;
    use core::borrow::Borrow;
    use core::mem::size_of;

    use p3_baby_bear::{
        BABYBEAR_POSEIDON1_HALF_FULL_ROUNDS, BABYBEAR_POSEIDON1_PARTIAL_ROUNDS_16,
        BABYBEAR_POSEIDON1_RC_16, BABYBEAR_S_BOX_DEGREE, BabyBear, MDSBabyBearData,
        default_babybear_poseidon1_16,
    };
    use p3_challenger::{HashChallenger, SerializingChallenger32};
    use p3_commit::ExtensionMmcs;
    use p3_field::extension::BinomialExtensionField;
    use p3_fri::{FriParameters, TwoAdicFriPcs};
    use p3_keccak::{Keccak256Hash, KeccakF};
    use p3_matrix::Matrix;
    use p3_merkle_tree::MerkleTreeMmcs;
    use p3_monty_31::MDSUtils;
    use p3_poseidon1::Poseidon1Constants;
    use p3_symmetric::{
        CompressionFunctionFromHasher, PaddingFreeSponge, Permutation, SerializingHasher,
    };
    use p3_uni_stark::{StarkConfig, prove, verify};

    use crate::columns::{Poseidon1Cols, num_cols};
    use crate::{Poseidon1Air, generate_trace_rows};

    type F = BabyBear;

    // BabyBear Poseidon1 parameters (width 16), imported from `p3-baby-bear`.
    const WIDTH: usize = 16;
    const SBOX_DEGREE: u64 = BABYBEAR_S_BOX_DEGREE;
    const SBOX_REGISTERS: usize = 1;
    const HALF_FULL_ROUNDS: usize = BABYBEAR_POSEIDON1_HALF_FULL_ROUNDS;
    const PARTIAL_ROUNDS: usize = BABYBEAR_POSEIDON1_PARTIAL_ROUNDS_16;

    /// Build the raw Poseidon1 constants for BabyBear width 16.
    fn babybear_poseidon1_constants_16() -> Poseidon1Constants<F, 16> {
        Poseidon1Constants {
            rounds_f: 2 * BABYBEAR_POSEIDON1_HALF_FULL_ROUNDS,
            rounds_p: BABYBEAR_POSEIDON1_PARTIAL_ROUNDS_16,
            mds_circ_col: MDSBabyBearData::MATRIX_CIRC_MDS_16_COL,
            round_constants: BABYBEAR_POSEIDON1_RC_16.to_vec(),
        }
    }

    #[test]
    fn test_column_layout() {
        // num_cols() computes the expected column count from const generics.
        let expected =
            num_cols::<WIDTH, SBOX_DEGREE, SBOX_REGISTERS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>();

        // size_of with T=u8 gives the actual struct size in bytes = number of fields.
        let actual = size_of::<
            Poseidon1Cols<u8, WIDTH, SBOX_DEGREE, SBOX_REGISTERS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>,
        >();

        assert_eq!(expected, actual);
    }

    #[test]
    fn test_poseidon1_air_degree_babybear() {
        use p3_air::BaseAir;
        use p3_air::symbolic::{AirLayout, get_max_constraint_degree};

        let raw = babybear_poseidon1_constants_16();
        let (full, partial) = raw.to_optimized();
        let air: Poseidon1Air<
            F,
            WIDTH,
            SBOX_DEGREE,
            SBOX_REGISTERS,
            HALF_FULL_ROUNDS,
            PARTIAL_ROUNDS,
        > = Poseidon1Air::new(full, partial);

        let layout = AirLayout {
            main_width: air.width(),
            ..Default::default()
        };
        let degree = get_max_constraint_degree::<F, _>(&air, layout);
        assert_eq!(
            degree, 3,
            "Expected constraint degree 3 for BabyBear (7, 1)"
        );
    }

    #[test]
    fn test_known_answer_babybear_16() {
        // Build AIR constants from the raw BabyBear Poseidon1 parameters.
        //
        // This applies the sparse matrix decomposition internally.
        let raw = babybear_poseidon1_constants_16();
        let (full, partial) = raw.to_optimized();

        // Generate a single permutation trace with sequential input [0..15].
        let input: [F; 16] = F::new_array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
        let inputs = vec![input];
        let trace = generate_trace_rows::<
            _,
            WIDTH,
            SBOX_DEGREE,
            SBOX_REGISTERS,
            HALF_FULL_ROUNDS,
            PARTIAL_ROUNDS,
        >(inputs, &full, &partial, 0);

        // One input → one row in the trace.
        assert_eq!(trace.height(), 1);

        // Compute the expected output using the reference permutation.
        let perm = default_babybear_poseidon1_16();
        let mut expected_state = input;
        perm.permute_mut(&mut expected_state);

        // Extract the final post-state from the last ending full round.
        let local = trace.row_slice(0).expect("Trace is empty");
        let row: &Poseidon1Cols<
            F,
            WIDTH,
            SBOX_DEGREE,
            SBOX_REGISTERS,
            HALF_FULL_ROUNDS,
            PARTIAL_ROUNDS,
        > = (*local).borrow();
        let final_post = &row.ending_full_rounds[HALF_FULL_ROUNDS - 1].post;

        // The AIR trace output must match the reference implementation.
        assert_eq!(final_post, &expected_state);
    }

    #[test]
    fn test_constraint_satisfaction_babybear_16() {
        let raw = babybear_poseidon1_constants_16();
        let (full, partial) = raw.to_optimized();

        let air: Poseidon1Air<
            F,
            WIDTH,
            SBOX_DEGREE,
            SBOX_REGISTERS,
            HALF_FULL_ROUNDS,
            PARTIAL_ROUNDS,
        > = Poseidon1Air::new(full, partial);

        // Generate 16 permutations and check every constraint row-by-row.
        let trace = air.generate_trace_rows(16, 0);
        p3_air::check_constraints(&air, &trace, &[]);
    }

    #[test]
    fn test_prove_verify_babybear_16() {
        type Val = BabyBear;
        type Challenge = BinomialExtensionField<Val, 4>;

        let raw = babybear_poseidon1_constants_16();
        let (full, partial) = raw.to_optimized();

        let air: Poseidon1Air<
            Val,
            WIDTH,
            SBOX_DEGREE,
            SBOX_REGISTERS,
            HALF_FULL_ROUNDS,
            PARTIAL_ROUNDS,
        > = Poseidon1Air::new(full, partial);

        // Hash function for Merkle tree leaves and Fiat-Shamir challenger.
        let byte_hash = Keccak256Hash {};

        // Sponge-based hash: absorb 17 u64 lanes from KeccakF, squeeze 4.
        let u64_hash = PaddingFreeSponge::<KeccakF, 25, 17, 4>::new(KeccakF {});

        // Serialize field elements to bytes, then hash.
        let field_hash = SerializingHasher::new(u64_hash);

        // Merkle tree compression: hash two 4-u64 digests into one.
        let compress = CompressionFunctionFromHasher::<_, 2, 4>::new(u64_hash);

        // Merkle tree MMCS: vector commitment over the base field.
        let val_mmcs = MerkleTreeMmcs::<
            [Val; p3_keccak::VECTOR_LEN],
            [u64; p3_keccak::VECTOR_LEN],
            _,
            _,
            2,
            4,
        >::new(field_hash, compress, 3);

        // Extension field MMCS for FRI (degree-4 extension of BabyBear).
        let challenge_mmcs = ExtensionMmcs::<Val, Challenge, _>::new(val_mmcs.clone());

        // Fiat-Shamir challenger seeded with an empty initial state.
        let challenger = SerializingChallenger32::<Val, HashChallenger<u8, _, 32>>::from_hasher(
            vec![],
            byte_hash,
        );

        // FRI parameters (log_blowup determines the LDE blowup factor).
        let fri_params = FriParameters::new_benchmark(challenge_mmcs);

        // Generate the trace with extra capacity for the LDE blowup.
        let trace = air.generate_trace_rows(16, fri_params.log_blowup);

        // Polynomial commitment scheme: FRI over the two-adic domain.
        let dft = p3_dft::Radix2Bowers;
        let pcs = TwoAdicFriPcs::new(dft, val_mmcs, fri_params);
        let config = StarkConfig::new(pcs, challenger);

        // Prove: generate a STARK proof for the Poseidon1 AIR.
        let proof = prove(&config, &air, trace, &[]);

        // Verify: check the proof against the same AIR and config.
        verify(&config, &air, &proof, &[]).expect("Verification failed");
    }
}
