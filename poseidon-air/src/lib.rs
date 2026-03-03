//! An AIR for the Poseidon permutation.

#![no_std]

extern crate alloc;

mod air;
mod columns;
mod constants;
mod generation;
mod vectorized;

pub use air::*;
pub use columns::*;
pub use constants::*;
pub use generation::*;
pub use vectorized::*;

#[cfg(test)]
mod tests {
    use alloc::vec;
    use core::borrow::Borrow;
    use core::mem::size_of;

    use p3_baby_bear::{BABYBEAR_POSEIDON_RC_16, BabyBear, MDSBabyBearData};
    use p3_challenger::{HashChallenger, SerializingChallenger32};
    use p3_commit::ExtensionMmcs;
    use p3_field::extension::BinomialExtensionField;
    use p3_fri::{TwoAdicFriPcs, create_benchmark_fri_params};
    use p3_keccak::{Keccak256Hash, KeccakF};
    use p3_matrix::Matrix;
    use p3_merkle_tree::MerkleTreeMmcs;
    use p3_monty_31::MDSUtils;
    use p3_poseidon::PoseidonConstants;
    use p3_symmetric::{
        CompressionFunctionFromHasher, PaddingFreeSponge, Permutation, SerializingHasher,
    };
    use p3_uni_stark::{StarkConfig, prove, verify};

    use crate::columns::{PoseidonCols, num_cols};
    use crate::constants::RoundConstants;
    use crate::{PoseidonAir, generate_trace_rows};

    type F = BabyBear;

    const WIDTH: usize = 16;
    const SBOX_DEGREE: u64 = 7;
    const SBOX_REGISTERS: usize = 1;
    const HALF_FULL_ROUNDS: usize = 4;
    const PARTIAL_ROUNDS: usize = 13;

    fn babybear_poseidon_constants_16() -> PoseidonConstants<F, 16> {
        PoseidonConstants {
            rounds_f: 8,
            rounds_p: 13,
            mds_circ_col: MDSBabyBearData::MATRIX_CIRC_MDS_16_COL,
            round_constants: BABYBEAR_POSEIDON_RC_16.to_vec(),
        }
    }

    /// Verify that the column struct layout matches `num_cols()`.
    #[test]
    fn test_column_layout() {
        let expected =
            num_cols::<WIDTH, SBOX_DEGREE, SBOX_REGISTERS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>();
        let actual = size_of::<
            PoseidonCols<u8, WIDTH, SBOX_DEGREE, SBOX_REGISTERS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>,
        >();
        assert_eq!(expected, actual);
    }

    /// Known-answer test: compare trace generation output against the existing
    /// Poseidon1 permutation for BabyBear width 16.
    #[test]
    fn test_known_answer_babybear_16() {
        let raw = babybear_poseidon_constants_16();
        let air_constants: RoundConstants<F, WIDTH, HALF_FULL_ROUNDS, PARTIAL_ROUNDS> =
            RoundConstants::from_poseidon_constants(&raw);

        // Generate a single permutation trace.
        let input: [F; 16] = F::new_array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
        let inputs = vec![input];
        let trace = generate_trace_rows::<
            _,
            WIDTH,
            SBOX_DEGREE,
            SBOX_REGISTERS,
            HALF_FULL_ROUNDS,
            PARTIAL_ROUNDS,
        >(inputs, &air_constants, 0);

        // The trace should have exactly 1 row.
        assert_eq!(trace.height(), 1);

        // Compare the final state (after all rounds) with the known Poseidon1 output.
        let perm = p3_baby_bear::default_babybear_poseidon_16();
        let mut expected_state = input;
        perm.permute_mut(&mut expected_state);

        // The ending full rounds' last round post-state should match the Poseidon output.
        let local = trace.row_slice(0).expect("Trace is empty");
        let row: &PoseidonCols<
            F,
            WIDTH,
            SBOX_DEGREE,
            SBOX_REGISTERS,
            HALF_FULL_ROUNDS,
            PARTIAL_ROUNDS,
        > = (*local).borrow();
        let final_post = &row.ending_full_rounds[HALF_FULL_ROUNDS - 1].post;
        assert_eq!(final_post, &expected_state);
    }

    /// Direct constraint satisfaction check using the debug constraint builder.
    #[test]
    fn test_constraint_satisfaction_babybear_16() {
        use rand::SeedableRng;
        use rand::rngs::SmallRng;

        let mut rng = SmallRng::seed_from_u64(1);
        let air_constants: RoundConstants<F, WIDTH, HALF_FULL_ROUNDS, PARTIAL_ROUNDS> =
            RoundConstants::from_rng(&mut rng);

        let air: PoseidonAir<
            F,
            WIDTH,
            SBOX_DEGREE,
            SBOX_REGISTERS,
            HALF_FULL_ROUNDS,
            PARTIAL_ROUNDS,
        > = PoseidonAir::new(air_constants);

        let trace = air.generate_trace_rows(16, 0);
        p3_air::check_constraints(&air, &trace, &[]);
    }

    /// End-to-end prove-verify round trip using `p3-uni-stark`.
    #[test]
    fn test_prove_verify_babybear_16() {
        type Val = BabyBear;
        type Challenge = BinomialExtensionField<Val, 4>;

        use rand::SeedableRng;
        use rand::rngs::SmallRng;

        let mut rng = SmallRng::seed_from_u64(1);
        let air_constants: RoundConstants<F, WIDTH, HALF_FULL_ROUNDS, PARTIAL_ROUNDS> =
            RoundConstants::from_rng(&mut rng);

        let air: PoseidonAir<
            Val,
            WIDTH,
            SBOX_DEGREE,
            SBOX_REGISTERS,
            HALF_FULL_ROUNDS,
            PARTIAL_ROUNDS,
        > = PoseidonAir::new(air_constants);

        let byte_hash = Keccak256Hash {};
        let u64_hash = PaddingFreeSponge::<KeccakF, 25, 17, 4>::new(KeccakF {});
        let field_hash = SerializingHasher::new(u64_hash);
        let compress = CompressionFunctionFromHasher::<_, 2, 4>::new(u64_hash);
        let val_mmcs = MerkleTreeMmcs::<
            [Val; p3_keccak::VECTOR_LEN],
            [u64; p3_keccak::VECTOR_LEN],
            _,
            _,
            4,
        >::new(field_hash, compress, 3);
        let challenge_mmcs = ExtensionMmcs::<Val, Challenge, _>::new(val_mmcs.clone());
        let challenger = SerializingChallenger32::<Val, HashChallenger<u8, _, 32>>::from_hasher(
            vec![],
            byte_hash,
        );

        let fri_params = create_benchmark_fri_params(challenge_mmcs);

        // Use the AIR's generate_trace_rows to ensure constants match.
        let trace = air.generate_trace_rows(16, fri_params.log_blowup);

        let dft = p3_dft::Radix2Bowers;
        let pcs = TwoAdicFriPcs::new(dft, val_mmcs, fri_params);
        let config = StarkConfig::new(pcs, challenger);

        let proof = prove(&config, &air, trace, &[]);
        verify(&config, &air, &proof, &[]).expect("Verification failed");
    }
}
