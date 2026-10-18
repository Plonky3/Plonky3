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
//! \[HADES\] design strategy. Each permutation consists of three phases:
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
mod mds_dispatch;
mod vectorized;

pub use air::*;
pub use columns::*;
pub use generation::*;
pub use p3_poseidon1::external::FullRoundConstants;
pub use p3_poseidon1::internal::PartialRoundConstants;
pub use vectorized::*;

#[cfg(test)]
mod tests {
    extern crate std;

    use alloc::vec;
    use core::borrow::{Borrow, BorrowMut};
    use core::mem::{MaybeUninit, size_of};
    use std::panic::{AssertUnwindSafe, catch_unwind};

    use p3_air::BaseAir;
    use p3_air::symbolic::{AirLayout, get_max_constraint_degree};
    use p3_baby_bear::{
        BABYBEAR_POSEIDON1_HALF_FULL_ROUNDS, BABYBEAR_POSEIDON1_PARTIAL_ROUNDS_16,
        BABYBEAR_POSEIDON1_RC_16, BABYBEAR_S_BOX_DEGREE, BabyBear, MDSBabyBearData,
        default_babybear_poseidon1_16,
    };
    use p3_challenger::{HashChallenger, SerializingChallenger32};
    use p3_commit::ExtensionMmcs;
    use p3_field::PrimeCharacteristicRing;
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
    use crate::{
        FullRoundConstants, PartialRoundConstants, Poseidon1Air, VectorizedPoseidon1Air,
        generate_trace_rows, generate_trace_rows_for_perm, generate_vectorized_trace_rows,
    };

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

    type TestAir = Poseidon1Air<F, 2, 7, 1, 1, 2>;
    type TestVectorAir = VectorizedPoseidon1Air<F, 2, 7, 1, 1, 2, 2>;
    type TestCols = Poseidon1Cols<MaybeUninit<F>, 2, 7, 1, 1, 2>;

    fn test_constants() -> (FullRoundConstants<F, 2>, PartialRoundConstants<F, 2>) {
        let identity = [[F::ONE, F::ZERO], [F::ZERO, F::ONE]];
        (
            FullRoundConstants {
                initial: vec![[F::ZERO; 2]],
                terminal: vec![[F::ZERO; 2]],
                dense_mds: identity,
            },
            PartialRoundConstants {
                first_round_constants: [F::ZERO; 2],
                m_i: identity,
                sparse_first_row: vec![[F::ONE, F::ZERO]; 2],
                v: vec![[F::ZERO; 2]; 2],
                round_constants: vec![F::ZERO],
                textbook_scalar_constants: vec![F::ZERO; 2],
                textbook_residual: [F::ZERO; 2],
            },
        )
    }

    fn assert_shape_panic(f: impl FnOnce(), expected: &str) {
        let panic = catch_unwind(AssertUnwindSafe(f)).expect_err("invalid shape was accepted");
        let message = panic
            .downcast_ref::<&str>()
            .copied()
            .or_else(|| {
                panic
                    .downcast_ref::<std::string::String>()
                    .map(|s| s.as_str())
            })
            .expect("panic should describe the invalid shape");
        assert!(message.contains(expected), "unexpected panic: {message}");
    }

    #[test]
    fn rejects_short_and_long_constant_vectors_at_every_entrypoint() {
        for field in [
            "initial",
            "terminal",
            "sparse_first_row",
            "v",
            "round_constants",
        ] {
            for extra in [false, true] {
                let (mut full, mut partial) = test_constants();
                let len = if matches!(field, "sparse_first_row" | "v") {
                    2
                } else {
                    1
                };
                let malformed_len = if extra { len + 1 } else { len - 1 };
                match field {
                    "initial" => full.initial.resize(malformed_len, [F::ZERO; 2]),
                    "terminal" => full.terminal.resize(malformed_len, [F::ZERO; 2]),
                    "sparse_first_row" => {
                        partial.sparse_first_row.resize(malformed_len, [F::ZERO; 2]);
                    }
                    "v" => partial.v.resize(malformed_len, [F::ZERO; 2]),
                    "round_constants" => partial.round_constants.resize(malformed_len, F::ZERO),
                    _ => unreachable!(),
                }

                assert_shape_panic(
                    || {
                        TestAir::new(full.clone(), partial.clone());
                    },
                    field,
                );
                assert_shape_panic(
                    || {
                        TestVectorAir::new(full.clone(), partial.clone());
                    },
                    field,
                );
                // Empty inputs guard against reaching the uninitialized allocation path even
                // if shape validation regresses. The shape must be checked before inputs.
                assert_shape_panic(
                    || {
                        generate_trace_rows::<_, 2, 7, 1, 1, 2>(vec![], &full, &partial, 0);
                    },
                    field,
                );
                assert_shape_panic(
                    || {
                        generate_vectorized_trace_rows::<_, 2, 7, 1, 1, 2, 2>(
                            vec![],
                            &full,
                            &partial,
                            0,
                        );
                    },
                    field,
                );

                // Fully initialize the buffer so missing writes can never cause UB.
                let mut cells = vec![MaybeUninit::new(F::ONE); num_cols::<2, 7, 1, 1, 2>()];
                assert_shape_panic(
                    || {
                        let cols: &mut TestCols = cells.as_mut_slice().borrow_mut();
                        generate_trace_rows_for_perm(
                            cols,
                            [F::ZERO; 2],
                            &full,
                            &partial,
                            &[F::ONE, F::ZERO],
                        );
                    },
                    field,
                );
                assert!(
                    cells
                        .iter()
                        .all(|cell| unsafe { cell.assume_init() } == F::ONE),
                    "invalid shape must be rejected before writing any columns"
                );
            }
        }
    }

    #[test]
    fn rejects_zero_partial_rounds_at_every_entrypoint() {
        let (full, mut partial) = test_constants();
        partial.sparse_first_row.clear();
        partial.v.clear();
        partial.round_constants.clear();
        assert_shape_panic(
            || {
                Poseidon1Air::<_, 2, 7, 1, 1, 0>::new(full.clone(), partial.clone());
            },
            "PARTIAL_ROUNDS",
        );
        assert_shape_panic(
            || {
                VectorizedPoseidon1Air::<_, 2, 7, 1, 1, 0, 2>::new(full.clone(), partial.clone());
            },
            "PARTIAL_ROUNDS",
        );
        assert_shape_panic(
            || {
                generate_trace_rows::<_, 2, 7, 1, 1, 0>(vec![], &full, &partial, 0);
            },
            "PARTIAL_ROUNDS",
        );
        assert_shape_panic(
            || {
                generate_vectorized_trace_rows::<_, 2, 7, 1, 1, 0, 2>(vec![], &full, &partial, 0);
            },
            "PARTIAL_ROUNDS",
        );
        let mut cells = vec![MaybeUninit::new(F::ONE); num_cols::<2, 7, 1, 1, 0>()];
        assert_shape_panic(
            || {
                generate_trace_rows_for_perm::<_, 2, 7, 1, 1, 0>(
                    cells.as_mut_slice().borrow_mut(),
                    [F::ZERO; 2],
                    &full,
                    &partial,
                    &[F::ONE, F::ZERO],
                );
            },
            "PARTIAL_ROUNDS",
        );
    }

    #[test]
    fn generation_overwrites_every_initialized_column() {
        let (full, partial) = test_constants();
        let mut cells = vec![MaybeUninit::new(F::ONE); num_cols::<2, 7, 1, 1, 2>()];
        let cols: &mut TestCols = cells.as_mut_slice().borrow_mut();
        generate_trace_rows_for_perm(cols, [F::ZERO; 2], &full, &partial, &[F::ONE, F::ZERO]);
        // Zero constants and zero inputs keep every intermediate zero. A remaining
        // one is an unwritten cell; reading is safe because every cell started initialized.
        assert!(
            cells
                .iter()
                .all(|cell| unsafe { cell.assume_init() } == F::ZERO)
        );
    }

    #[test]
    fn single_partial_round_accepts_empty_scalar_constants() {
        let (full, mut partial) = test_constants();
        partial.sparse_first_row.truncate(1);
        partial.v.truncate(1);
        partial.round_constants.clear();
        let air = Poseidon1Air::<_, 2, 7, 1, 1, 1>::new(full, partial);
        let trace = air.generate_trace_rows_from_inputs(vec![[F::ZERO; 2]], 0);
        assert!(trace.values.iter().all(|&cell| cell == F::ZERO));
        p3_air::check_constraints(&air, &trace, &[]);
    }

    #[test]
    fn canonical_vectorized_trace_matches_scalar_and_satisfies_constraints() {
        let (full, partial) = babybear_poseidon1_constants_16().to_optimized();
        let inputs = vec![[F::ZERO; WIDTH], [F::ONE; WIDTH]];
        let scalar = generate_trace_rows::<
            _,
            WIDTH,
            SBOX_DEGREE,
            SBOX_REGISTERS,
            HALF_FULL_ROUNDS,
            PARTIAL_ROUNDS,
        >(inputs.clone(), &full, &partial, 0);
        let air = VectorizedPoseidon1Air::<
            _,
            WIDTH,
            SBOX_DEGREE,
            SBOX_REGISTERS,
            HALF_FULL_ROUNDS,
            PARTIAL_ROUNDS,
            2,
        >::new(full, partial);
        let vectorized = air.generate_vectorized_trace_rows_from_inputs(inputs, 0);
        assert_eq!(vectorized.values, scalar.values);
        p3_air::check_constraints(&air, &vectorized, &[]);
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
        let degree = get_max_constraint_degree::<F, _>(&air, layout, 1 << 6);
        assert_eq!(
            degree, 3,
            "Expected constraint degree 3 for BabyBear (7, 1)"
        );
    }

    #[test]
    fn test_poseidon1_air_degree_sbox_11_1() {
        // The x^11 one-register S-box must report a degree bound.
        let raw = babybear_poseidon1_constants_16();
        let (full, partial) = raw.to_optimized();
        let air: Poseidon1Air<F, WIDTH, 11, 1, HALF_FULL_ROUNDS, PARTIAL_ROUNDS> =
            Poseidon1Air::new(full, partial);

        // The reported bound is what the prover queries before evaluation.
        assert_eq!(air.max_constraint_degree(), Some(5));

        // The symbolic walk over the actual constraints must agree with the bound:
        //     commit x^3, constrain x^3 = x * x * x         → degree 3
        //     output (x^3)^3 * x^2, asserted against post   → degree 5
        let layout = AirLayout {
            main_width: air.width(),
            ..Default::default()
        };
        let degree = get_max_constraint_degree::<F, _>(&air, layout, 1 << 6);
        assert_eq!(degree, 5, "Expected constraint degree 5 for (11, 1)");
    }

    #[test]
    fn test_constraint_satisfaction_sbox_11_1() {
        // Trace generation and constraint evaluation implement the (11, 1) S-box independently;
        // Check the two agree on real traces.
        let raw = babybear_poseidon1_constants_16();
        let (full, partial) = raw.to_optimized();
        let air: Poseidon1Air<F, WIDTH, 11, 1, HALF_FULL_ROUNDS, PARTIAL_ROUNDS> =
            Poseidon1Air::new(full, partial);

        // Generate 16 permutations and check every constraint row-by-row.
        let trace = air.generate_trace_rows(16, 0);
        p3_air::check_constraints(&air, &trace, &[]);
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
        let proof = prove(&config, &air, trace, &[]).unwrap();

        // Verify: check the proof against the same AIR and config.
        verify(&config, &air, &proof, &[]).expect("Verification failed");
    }
    fn compare_scalar_and_packed<const DEGREE: u64, const REGISTERS: usize>() {
        use p3_field::PrimeCharacteristicRing;
        let (full, partial) = babybear_poseidon1_constants_16().to_optimized();
        for n in [1, 2, 4, 8, 16, 32] {
            let inputs: alloc::vec::Vec<[F; WIDTH]> = (0..n)
                .map(|i| {
                    core::array::from_fn(|j| match (i + j) % 4 {
                        0 => F::ZERO,
                        1 => F::ONE,
                        2 => F::NEG_ONE,
                        _ => F::from_usize(i * WIDTH + j),
                    })
                })
                .collect();
            let expected: alloc::vec::Vec<F> = inputs
                .iter()
                .flat_map(|input| {
                    generate_trace_rows::<
                        F,
                        WIDTH,
                        DEGREE,
                        REGISTERS,
                        HALF_FULL_ROUNDS,
                        PARTIAL_ROUNDS,
                    >(vec![*input], &full, &partial, 0)
                    .values
                })
                .collect();
            let actual = generate_trace_rows::<
                F,
                WIDTH,
                DEGREE,
                REGISTERS,
                HALF_FULL_ROUNDS,
                PARTIAL_ROUNDS,
            >(inputs.clone(), &full, &partial, 1);
            assert_eq!(
                actual.values, expected,
                "batch {n}, S-box {DEGREE}/{REGISTERS}"
            );
            if n >= 4 {
                let vectorized = crate::generate_vectorized_trace_rows::<
                    F,
                    WIDTH,
                    DEGREE,
                    REGISTERS,
                    HALF_FULL_ROUNDS,
                    PARTIAL_ROUNDS,
                    4,
                >(inputs, &full, &partial, 0);
                assert_eq!(vectorized.values, expected);
            }
        }
    }

    #[test]
    fn packed_witness_matches_every_scalar_column() {
        compare_scalar_and_packed::<3, 0>();
        compare_scalar_and_packed::<5, 0>();
        compare_scalar_and_packed::<7, 0>();
        compare_scalar_and_packed::<5, 1>();
        compare_scalar_and_packed::<7, 1>();
        compare_scalar_and_packed::<11, 1>();
        compare_scalar_and_packed::<11, 2>();
    }

    fn compare_packed_width<const W: usize>(non_circulant: bool) {
        use p3_field::PrimeCharacteristicRing;
        let raw = Poseidon1Constants::<F, W> {
            rounds_f: 2,
            rounds_p: 3,
            // A nonsingular circulant with invertible lower-right blocks.
            mds_circ_col: core::array::from_fn(|i| if i == 0 { 2 } else { 1 }),
            round_constants: (0..5)
                .map(|r| core::array::from_fn(|j| F::from_usize(r * W + j)))
                .collect(),
        };
        let (mut full, partial) = raw.to_optimized();
        if non_circulant {
            full.dense_mds[1][2] += F::ONE;
        }
        let inputs: alloc::vec::Vec<[F; W]> = (0..32)
            .map(|i| core::array::from_fn(|j| F::from_usize(i * W + j)))
            .collect();
        let expected: alloc::vec::Vec<F> = inputs
            .iter()
            .flat_map(|input| {
                generate_trace_rows::<F, W, 7, 1, 1, 3>(vec![*input], &full, &partial, 0).values
            })
            .collect();
        let actual = generate_trace_rows::<F, W, 7, 1, 1, 3>(inputs, &full, &partial, 0);
        assert_eq!(actual.values, expected);
    }

    #[test]
    fn packed_witness_covers_karatsuba_24_and_dense_fallback() {
        compare_packed_width::<24>(false);
        compare_packed_width::<8>(false);
        compare_packed_width::<8>(true);
    }
}
