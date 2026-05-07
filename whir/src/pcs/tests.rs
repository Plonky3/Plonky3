//! End-to-end tests exercising the WHIR PCS through the multilinear trait.

use alloc::vec::Vec;

use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::DuplexChallenger;
use p3_commit::MultilinearPcs;
use p3_dft::Radix2DFTSmallBatch;
use p3_field::Field;
use p3_field::extension::BinomialExtensionField;
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_multilinear_util::point::Point;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

use crate::parameters::{FoldingFactor, ProtocolParameters, SecurityAssumption, SumcheckStrategy};
use crate::pcs::WhirPcs;

type F = BabyBear;
type EF = BinomialExtensionField<F, 4>;
type Perm = Poseidon2BabyBear<16>;

type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
type MyChallenger = DuplexChallenger<F, Perm, 16, 8>;

type PackedF = <F as Field>::Packing;
type MyMmcs = MerkleTreeMmcs<PackedF, PackedF, MyHash, MyCompress, 2, 8>;

type MyDft = Radix2DFTSmallBatch<F>;
type TestWhirPcs = WhirPcs<EF, F, MyMmcs, MyChallenger, MyDft, 8>;

/// Run a full commit -> open -> verify cycle through the trait interface.
///
/// This exercises the complete WHIR protocol (Construction 5.1):
///
/// 1. Build a random multilinear polynomial f: {0,1}^m -> F.
/// 2. Choose random evaluation points z_1, ..., z_t in F^m.
/// 3. Commit to f and register the opening points.
/// 4. Produce an opening proof via multi-round sumcheck + STIR queries.
/// 5. Verify the proof against the commitment and claimed evaluations.
#[allow(clippy::too_many_arguments)]
fn run_whir_pcs_lifecycle(
    num_variables: usize,
    folding_factor: FoldingFactor,
    num_points: usize,
    soundness_type: SecurityAssumption,
    pow_bits: usize,
    rs_domain_initial_reduction_factor: usize,
    sumcheck_strategy: SumcheckStrategy,
) {
    // Total number of evaluations on the Boolean hypercube: 2^m.
    let num_evaluations = 1 << num_variables;

    // Deterministic RNG for reproducible tests.
    let mut rng = SmallRng::seed_from_u64(1);

    // Build Poseidon2-based hash and compression for the Merkle tree.
    let perm = Perm::new_from_rng_128(&mut rng);
    let merkle_hash = MyHash::new(perm.clone());
    let merkle_compress = MyCompress::new(perm);
    let mmcs = MyMmcs::new(merkle_hash, merkle_compress, 0);

    // Assemble the protocol parameters.
    // Security level 32 keeps the test fast; production would use 100-128.
    let whir_params = ProtocolParameters {
        security_level: 32,
        pow_bits,
        rs_domain_initial_reduction_factor,
        folding_factor,
        mmcs,
        soundness_type,
        starting_log_inv_rate: 1,
    };

    // Instantiate the PCS through the trait.
    let dft = MyDft::default();
    let pcs = TestWhirPcs::new(num_variables, whir_params, dft, sumcheck_strategy);

    // Generate a random multilinear polynomial as a flat evaluation vector.
    let evaluations: Vec<F> = (0..num_evaluations).map(|_| rng.random()).collect();
    // Wrap as a single-column matrix (one polynomial).
    let eval_matrix = RowMajorMatrix::new(evaluations, 1);

    // Sample random extension-field points where we claim to know f(z_i).
    // Each point is expanded from a single univariate seed into m coordinates
    // via pow(z, m) = (z^{2^0}, ..., z^{2^{m-1}}).
    let opening_points: Vec<Point<EF>> = (0..num_points)
        .map(|_| Point::expand_from_univariate(rng.random(), num_variables))
        .collect();

    // Prover side: fresh challenger seeded identically to the verifier's.
    let mut rng2 = SmallRng::seed_from_u64(1);
    let mut prover_challenger = MyChallenger::new(Perm::new_from_rng_128(&mut rng2));

    // Phase 1: commit to the polynomial and register opening points.
    // Internally this runs DFT encoding, Merkle commitment, and OOD sampling.
    let (commitment, prover_data) = pcs.commit(
        eval_matrix,
        core::slice::from_ref(&opening_points),
        &mut prover_challenger,
    );

    // Phase 2: produce the opening proof.
    // Internally this runs M rounds of sumcheck + STIR queries + PoW grinding.
    let (opened_values, proof) = pcs.open(prover_data, &mut prover_challenger);

    // Package the claimed evaluations as (point, value) pairs for the verifier.
    let claims: Vec<(Point<EF>, EF)> = opening_points
        .into_iter()
        .zip(opened_values[0].iter().copied())
        .collect();

    // Verifier side: independent challenger with the same seed.
    let mut rng3 = SmallRng::seed_from_u64(1);
    let mut verifier_challenger = MyChallenger::new(Perm::new_from_rng_128(&mut rng3));

    // Phase 3: verify the proof against the commitment and claims.
    pcs.verify(&commitment, &[claims], &proof, &mut verifier_challenger)
        .expect("verification failed");
}

#[test]
fn test_whir_end_to_end() {
    // Sweep all meaningful parameter combinations.
    //
    // The folding factor k controls how many variables are eliminated per round.
    // WHIR supports constant k or a different k for the first round.
    let folding_factors = [
        FoldingFactor::Constant(1),
        FoldingFactor::Constant(2),
        FoldingFactor::Constant(3),
        FoldingFactor::Constant(4),
        FoldingFactor::ConstantFromSecondRound(2, 1),
        FoldingFactor::ConstantFromSecondRound(3, 1),
        FoldingFactor::ConstantFromSecondRound(3, 2),
        FoldingFactor::ConstantFromSecondRound(5, 2),
    ];

    // Three soundness regimes with different proximity parameters:
    //   - Unique decoding: delta = (1-rho)/2
    //   - Johnson bound:   delta = 1 - sqrt(rho) - eta
    //   - Capacity bound:  delta = 1 - rho - eta  (conjectured)
    let soundness_type = [
        SecurityAssumption::JohnsonBound,
        SecurityAssumption::CapacityBound,
        SecurityAssumption::UniqueDecoding,
    ];

    // Number of evaluation claims to prove (0 = proximity test only).
    let num_points = [0, 1, 2];

    // Proof-of-work difficulty: prevents query manipulation (Section 6.2).
    let pow_bits = [0, 5, 10];

    // Initial domain reduction before the first folding round.
    let rs_domain_initial_reduction_factors = 1..=3;

    for rs_domain_initial_reduction_factor in rs_domain_initial_reduction_factors {
        for folding_factor in folding_factors {
            // Skip configurations where the first-round folding is smaller
            // than the initial domain reduction (would produce an empty domain).
            if folding_factor.at_round(0) < rs_domain_initial_reduction_factor {
                continue;
            }
            // Test polynomial sizes from k to 3k variables, where k is
            // the first-round folding factor. This covers 1-round, 2-round,
            // and 3-round protocol executions.
            let num_variables = folding_factor.at_round(0)..=3 * folding_factor.at_round(0);
            for num_variable in num_variables {
                for num_points in num_points {
                    for soundness_type in soundness_type {
                        for pow_bits in pow_bits {
                            // Test both sumcheck strategies: classic constraint
                            // batching and split-value optimization.
                            run_whir_pcs_lifecycle(
                                num_variable,
                                folding_factor,
                                num_points,
                                soundness_type,
                                pow_bits,
                                rs_domain_initial_reduction_factor,
                                SumcheckStrategy::Svo,
                            );
                            run_whir_pcs_lifecycle(
                                num_variable,
                                folding_factor,
                                num_points,
                                soundness_type,
                                pow_bits,
                                rs_domain_initial_reduction_factor,
                                SumcheckStrategy::Classic,
                            );
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod keccak_tests {
    //! Same lifecycle test using Keccak-based Merkle trees over a different field.

    use alloc::vec;
    use alloc::vec::Vec;

    use p3_challenger::{HashChallenger, SerializingChallenger32};
    use p3_commit::MultilinearPcs;
    use p3_dft::Radix2DFTSmallBatch;
    use p3_field::extension::BinomialExtensionField;
    use p3_keccak::{Keccak256Hash, KeccakF};
    use p3_koala_bear::KoalaBear;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_merkle_tree::MerkleTreeMmcs;
    use p3_multilinear_util::point::Point;
    use p3_symmetric::{CompressionFunctionFromHasher, PaddingFreeSponge, SerializingHasher};
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use crate::parameters::{
        FoldingFactor, ProtocolParameters, SecurityAssumption, SumcheckStrategy,
    };
    use crate::pcs::WhirPcs;

    type F = KoalaBear;
    type EF = BinomialExtensionField<F, 4>;

    // Keccak sponge producing u64 digests for Merkle leaves.
    type U64Hash = PaddingFreeSponge<KeccakF, 25, 17, 4>;
    // Serializing wrapper to hash field elements via the Keccak sponge.
    type KeccakFieldHash = SerializingHasher<U64Hash>;
    // 2-to-1 compression for internal Merkle nodes.
    type KeccakCompress = CompressionFunctionFromHasher<U64Hash, 2, 4>;

    // Byte-based challenger using Keccak-256 for Fiat-Shamir.
    type KeccakChallenger = SerializingChallenger32<F, HashChallenger<u8, Keccak256Hash, 32>>;
    type MyMmcs = MerkleTreeMmcs<F, u64, KeccakFieldHash, KeccakCompress, 2, 4>;

    type MyDft = Radix2DFTSmallBatch<F>;
    // Digest width is 4 (four u64 elements per Keccak Merkle node).
    type TestWhirPcs = WhirPcs<EF, F, MyMmcs, KeccakChallenger, MyDft, 4>;

    #[allow(clippy::too_many_arguments)]
    fn run_whir_pcs_lifecycle_keccak(
        num_variables: usize,
        folding_factor: FoldingFactor,
        num_points: usize,
        soundness_type: SecurityAssumption,
        pow_bits: usize,
        rs_domain_initial_reduction_factor: usize,
        sumcheck_strategy: SumcheckStrategy,
    ) {
        let num_evaluations = 1 << num_variables;

        // Build Keccak-based Merkle tree primitives.
        let u64_hash = U64Hash::new(KeccakF {});
        let merkle_hash = KeccakFieldHash::new(u64_hash);
        let merkle_compress = KeccakCompress::new(u64_hash);
        let mmcs = MyMmcs::new(merkle_hash, merkle_compress, 0);

        // Assemble protocol parameters with Keccak hashing.
        let whir_params = ProtocolParameters {
            security_level: 32,
            pow_bits,
            rs_domain_initial_reduction_factor,
            folding_factor,
            mmcs,
            soundness_type,
            starting_log_inv_rate: 1,
        };

        // Instantiate the PCS through the trait.
        let dft = MyDft::default();
        let pcs = TestWhirPcs::new(num_variables, whir_params, dft, sumcheck_strategy);

        // Random polynomial and opening points.
        let mut rng = SmallRng::seed_from_u64(1);
        let evaluations: Vec<F> = (0..num_evaluations).map(|_| rng.random()).collect();
        let eval_matrix = RowMajorMatrix::new(evaluations, 1);

        let opening_points: Vec<Point<EF>> = (0..num_points)
            .map(|_| Point::expand_from_univariate(rng.random(), num_variables))
            .collect();

        // Prover: commit and open using a Keccak-based challenger.
        let inner = HashChallenger::<u8, Keccak256Hash, 32>::new(vec![], Keccak256Hash {});
        let mut prover_challenger = KeccakChallenger::new(inner);

        let (commitment, prover_data) = pcs.commit(
            eval_matrix,
            core::slice::from_ref(&opening_points),
            &mut prover_challenger,
        );

        let (opened_values, proof) = pcs.open(prover_data, &mut prover_challenger);

        // Package claims for the verifier.
        let claims: Vec<(Point<EF>, EF)> = opening_points
            .into_iter()
            .zip(opened_values[0].iter().copied())
            .collect();

        // Verifier: independent challenger with the same empty initial state.
        let inner = HashChallenger::<u8, Keccak256Hash, 32>::new(vec![], Keccak256Hash {});
        let mut verifier_challenger = KeccakChallenger::new(inner);

        pcs.verify(&commitment, &[claims], &proof, &mut verifier_challenger)
            .expect("keccak verification failed");
    }

    #[test]
    fn test_whir_keccak_end_to_end() {
        // Single representative configuration to verify Keccak compatibility.
        run_whir_pcs_lifecycle_keccak(
            10,
            FoldingFactor::Constant(4),
            2,
            SecurityAssumption::CapacityBound,
            0,
            1,
            SumcheckStrategy::default(),
        );
    }
}
