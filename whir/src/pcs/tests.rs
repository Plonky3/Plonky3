//! End-to-end tests exercising the WHIR PCS through the multilinear trait.

use alloc::vec;

use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::DuplexChallenger;
use p3_commit::MultilinearPcs;
use p3_dft::Radix2DFTSmallBatch;
use p3_field::Field;
use p3_field::extension::BinomialExtensionField;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use rand::SeedableRng;
use rand::rngs::SmallRng;

use crate::fiat_shamir::domain_separator::DomainSeparator;
use crate::parameters::{FoldingFactor, ProtocolParameters, SecurityAssumption, WhirConfig};
use crate::pcs::prover::WhirProver;
use crate::sumcheck::layout::{Layout, PrefixProver, SuffixProver, Witness};
use crate::sumcheck::tests::{random_table_specs, table_specs_to_tables};
use crate::sumcheck::{OpeningProtocol, TableSpec};

type F = BabyBear;
type EF = BinomialExtensionField<F, 4>;
type Perm = Poseidon2BabyBear<16>;

type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
type MyChallenger = DuplexChallenger<F, Perm, 16, 8>;

type PackedF = <F as Field>::Packing;
type MyMmcs = MerkleTreeMmcs<PackedF, PackedF, MyHash, MyCompress, 2, 8>;

type MyDft = Radix2DFTSmallBatch<F>;
type TestWhirPcs<L> = WhirProver<EF, F, MyDft, MyMmcs, MyChallenger, L>;

pub(crate) fn challenger() -> MyChallenger {
    let mut rng = SmallRng::seed_from_u64(1);
    let perm = Perm::new_from_rng_128(&mut rng);
    MyChallenger::new(perm)
}

#[allow(clippy::too_many_arguments)]
fn run_whir_pcs<L: Layout<F, EF>>(
    specs: &[TableSpec],
    folding_factor: FoldingFactor,
    soundness_type: SecurityAssumption,
    pow_bits: usize,
    rs_domain_initial_reduction_factor: usize,
) {
    let folding = folding_factor.at_round(0);
    let tables = table_specs_to_tables(specs);
    let witness = L::new_witness(tables, folding);
    let protocol = OpeningProtocol::new(specs.to_vec()).pad_to_min_num_variables(folding);
    assert_eq!(witness.table_shapes(), protocol.table_shapes());

    run_whir_pcs_lifecycle_with_witness::<L>(
        witness,
        protocol,
        folding_factor,
        soundness_type,
        pow_bits,
        rs_domain_initial_reduction_factor,
    );
}

#[allow(clippy::too_many_arguments)]
fn run_whir_pcs_lifecycle_with_witness<L: Layout<F, EF>>(
    witness: Witness<F>,
    protocol: OpeningProtocol,
    folding_factor: FoldingFactor,
    soundness_type: SecurityAssumption,
    pow_bits: usize,
    rs_domain_initial_reduction_factor: usize,
) {
    // Build Poseidon2-based hash and compression for the Merkle tree.
    let num_variables = witness.num_variables();
    let mut rng = SmallRng::seed_from_u64(1);
    let perm = Perm::new_from_rng_128(&mut rng);
    let merkle_hash = MyHash::new(perm.clone());
    let merkle_compress = MyCompress::new(perm);
    let mmcs = MyMmcs::new(merkle_hash, merkle_compress, 0);

    // Assemble the protocol parameters.
    // Security level 32 keeps the test fast; production would use 100-128.
    let params = ProtocolParameters {
        security_level: 32,
        pow_bits,
        rs_domain_initial_reduction_factor,
        folding_factor,
        soundness_type,
        starting_log_inv_rate: 1,
    };

    // Instantiate the PCS through the trait.
    let dft = MyDft::default();
    let config = WhirConfig::new(num_variables, params);
    let pcs = TestWhirPcs::<L>::new(config, dft, mmcs);

    // Prover
    let (commitment, proof) = {
        let mut challenger = challenger();
        let mut domain_separator = DomainSeparator::new(vec![]);
        pcs.add_domain_separator::<8>(&mut domain_separator);
        domain_separator.observe_domain_separator(&mut challenger);

        let (commitment, prover_data) =
            <TestWhirPcs<L> as MultilinearPcs<EF, MyChallenger>>::commit(
                &pcs,
                witness,
                &mut challenger,
            );
        let proof = <TestWhirPcs<L> as MultilinearPcs<EF, MyChallenger>>::open(
            &pcs,
            prover_data,
            protocol.clone(),
            &mut challenger,
        );
        (commitment, proof)
    };

    // Verifier
    {
        let mut challenger = challenger();
        let mut domain_separator = DomainSeparator::new(vec![]);
        pcs.add_domain_separator::<8>(&mut domain_separator);
        domain_separator.observe_domain_separator(&mut challenger);

        <TestWhirPcs<L> as MultilinearPcs<EF, MyChallenger>>::verify(
            &pcs,
            &commitment,
            &proof,
            &mut challenger,
            protocol,
        )
        .expect("verification failed");
    }
}

#[test]
fn test_whir_end_to_end() {
    const N: usize = 5;

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
    let soundness_type = [
        SecurityAssumption::JohnsonBound,
        SecurityAssumption::CapacityBound,
        SecurityAssumption::UniqueDecoding,
    ];
    let pow_bits = [0, 5, 10];
    let rs_domain_initial_reduction_factors = 1..=3;

    let mut rng = SmallRng::seed_from_u64(7);

    for rs_domain_initial_reduction_factor in rs_domain_initial_reduction_factors {
        for folding_factor in folding_factors {
            // Skip configurations where the first-round folding is smaller
            // than the initial domain reduction (would produce an empty domain).
            if folding_factor.at_round(0) < rs_domain_initial_reduction_factor {
                continue;
            }

            for soundness_type in soundness_type {
                for pow_bits in pow_bits {
                    for _ in 0..N {
                        let specs = random_table_specs(&mut rng, folding_factor.at_round(0));
                        run_whir_pcs::<PrefixProver<F, EF>>(
                            &specs,
                            folding_factor,
                            soundness_type,
                            pow_bits,
                            rs_domain_initial_reduction_factor,
                        );
                        run_whir_pcs::<SuffixProver<F, EF>>(
                            &specs,
                            folding_factor,
                            soundness_type,
                            pow_bits,
                            rs_domain_initial_reduction_factor,
                        );
                    }
                }
            }
        }
    }
}

mod error_variant_tests {
    //! Lock the precise error variant emitted on each opening-shape mismatch.
    use alloc::vec;

    use p3_commit::MultilinearPcs;
    use p3_multilinear_util::poly::Poly;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use super::{
        EF, F, MyChallenger, MyCompress, MyDft, MyHash, MyMmcs, Perm, TestWhirPcs, challenger,
    };
    use crate::fiat_shamir::domain_separator::DomainSeparator;
    use crate::parameters::{FoldingFactor, ProtocolParameters, SecurityAssumption, WhirConfig};
    use crate::pcs::proof::PcsProof;
    use crate::pcs::verifier::errors::VerifierError;
    use crate::sumcheck::layout::{Layout, SuffixProver, Table};
    use crate::sumcheck::{OpeningProtocol, TableShape, TableSpec};

    /// Suffix-mode prover used for every shape-mismatch scenario.
    type L = SuffixProver<F, EF>;

    /// Stacked-polynomial arity: large enough for one intermediate STIR round.
    const NUM_VARIABLES: usize = 12;
    /// Variables eliminated per WHIR fold.
    const FOLDING: usize = 4;

    /// Builds a working PCS plus an honest commitment and proof for two batches.
    ///
    /// # Layout
    ///
    /// - Single table of arity 12 with two columns.
    /// - Two opening batches: first opens both columns; second opens column 0.
    ///
    /// Yields `protocol.num_openings() == 2` and the inner batch sizes
    /// `(2, 1)`. Both axes can be tampered independently.
    #[allow(clippy::type_complexity)]
    fn commit_and_open() -> (
        TestWhirPcs<L>,
        <TestWhirPcs<L> as MultilinearPcs<EF, MyChallenger>>::Commitment,
        PcsProof<F, EF, MyMmcs>,
        OpeningProtocol,
    ) {
        // Random table of two columns; deterministic seed for reproducibility.
        let mut rng = SmallRng::seed_from_u64(1);
        let table = Table::new(vec![
            Poly::<F>::rand(&mut rng, NUM_VARIABLES),
            Poly::<F>::rand(&mut rng, NUM_VARIABLES),
        ]);
        let witness = L::new_witness(vec![table], FOLDING);
        // Two opening batches: (cols [0, 1]) and (col [0]).
        let protocol = OpeningProtocol::new(vec![TableSpec::new(
            TableShape::new(NUM_VARIABLES, 2),
            vec![vec![0, 1], vec![0]],
        )]);

        // Same Poseidon2 seed as elsewhere for byte-for-byte reproducibility.
        let perm = Perm::new_from_rng_128(&mut SmallRng::seed_from_u64(1));
        let merkle_hash = MyHash::new(perm.clone());
        let merkle_compress = MyCompress::new(perm);
        let mmcs = MyMmcs::new(merkle_hash, merkle_compress, 0);

        let params = ProtocolParameters {
            security_level: 32,
            pow_bits: 0,
            rs_domain_initial_reduction_factor: 1,
            folding_factor: FoldingFactor::Constant(FOLDING),
            soundness_type: SecurityAssumption::CapacityBound,
            starting_log_inv_rate: 1,
        };
        let pcs = TestWhirPcs::<L>::new(
            WhirConfig::new(witness.num_variables(), params),
            MyDft::default(),
            mmcs,
        );

        let mut prover_challenger = challenger();
        let mut domain_separator = DomainSeparator::new(vec![]);
        pcs.add_domain_separator::<8>(&mut domain_separator);
        domain_separator.observe_domain_separator(&mut prover_challenger);

        let (commitment, prover_data) =
            <TestWhirPcs<L> as MultilinearPcs<EF, MyChallenger>>::commit(
                &pcs,
                witness,
                &mut prover_challenger,
            );
        let proof = <TestWhirPcs<L> as MultilinearPcs<EF, MyChallenger>>::open(
            &pcs,
            prover_data,
            protocol.clone(),
            &mut prover_challenger,
        );

        (pcs, commitment, proof, protocol)
    }

    /// Replays the verifier on a (possibly tampered) proof and returns the result.
    fn verify(
        pcs: &TestWhirPcs<L>,
        commitment: &<TestWhirPcs<L> as MultilinearPcs<EF, MyChallenger>>::Commitment,
        proof: &PcsProof<F, EF, MyMmcs>,
        protocol: OpeningProtocol,
    ) -> Result<(), VerifierError> {
        // Verifier needs the same transcript prefix the prover absorbed.
        let mut verifier_challenger = challenger();
        let mut domain_separator = DomainSeparator::new(vec![]);
        pcs.add_domain_separator::<8>(&mut domain_separator);
        domain_separator.observe_domain_separator(&mut verifier_challenger);

        <TestWhirPcs<L> as MultilinearPcs<EF, MyChallenger>>::verify(
            pcs,
            commitment,
            proof,
            &mut verifier_challenger,
            protocol,
        )
    }

    #[test]
    fn rejects_with_batch_count_mismatch_when_a_batch_is_missing() {
        // Domain note: protocol.num_openings() must equal proof.evals.len().
        // The adapter checks this before any sumcheck or Merkle work, so the
        // failure mode is structural — the variant carries the two integers
        // verbatim.
        //
        // Fixture state: protocol declares 2 batches; honest proof has 2.
        //
        // Mutation: drop the trailing batch from the proof.
        //
        //     protocol batches:  2
        //     proof.evals:       2  ->  1
        //     -> expected = 2, actual = 1
        let (pcs, commitment, mut proof, protocol) = commit_and_open();
        assert_eq!(proof.evals.len(), 2);
        proof.evals.pop();

        let err = verify(&pcs, &commitment, &proof, protocol).unwrap_err();
        match err {
            VerifierError::OpeningBatchCountMismatch { expected, actual } => {
                assert_eq!(expected, 2);
                assert_eq!(actual, 1);
            }
            other => panic!("expected OpeningBatchCountMismatch, got {other:?}"),
        }
    }

    #[test]
    fn rejects_with_batch_size_mismatch_when_an_eval_is_dropped() {
        // Domain note: every batch i must carry exactly polys[i].len()
        // evaluations. The check runs per batch in protocol order, so the
        // first offender wins.
        //
        // Fixture state: batch 0 opens columns [0, 1] -> 2 evaluations expected.
        //
        // Mutation: drop one evaluation from batch 0.
        //
        //     protocol batch 0 columns:  [0, 1]   (len 2)
        //     proof.evals[0]:            [v0, v1] -> [v0]
        //     -> table_idx = 0, expected = 2, actual = 1
        let (pcs, commitment, mut proof, protocol) = commit_and_open();
        assert_eq!(proof.evals[0].len(), 2);
        proof.evals[0].pop();

        let err = verify(&pcs, &commitment, &proof, protocol).unwrap_err();
        match err {
            VerifierError::OpeningBatchSizeMismatch {
                table_idx,
                expected,
                actual,
            } => {
                assert_eq!(table_idx, 0);
                assert_eq!(expected, 2);
                assert_eq!(actual, 1);
            }
            other => panic!("expected OpeningBatchSizeMismatch, got {other:?}"),
        }
    }

    #[test]
    fn rejects_with_round_count_mismatch_when_a_round_is_dropped() {
        // Invariant: round count is fixed by the protocol config.
        //
        // Fixture state: N honest rounds → expected = N.
        //
        // Mutation: drop the trailing round.
        //
        //     proof.whir.rounds:  [r_0, r_1, ..., r_{N-1}]  ->  [r_0, ..., r_{N-2}]
        //     expected:           N
        //     actual:             N - 1
        let (pcs, commitment, mut proof, protocol) = commit_and_open();
        assert!(
            !proof.whir.rounds.is_empty(),
            "fixture should produce at least one WHIR round"
        );
        let expected = proof.whir.rounds.len();
        proof.whir.rounds.pop();

        let err = verify(&pcs, &commitment, &proof, protocol).unwrap_err();
        match err {
            VerifierError::RoundCountMismatch {
                expected: e,
                actual: a,
            } => {
                assert_eq!(e, expected);
                assert_eq!(a, expected - 1);
            }
            other => panic!("expected RoundCountMismatch, got {other:?}"),
        }
    }

    #[test]
    fn rejects_with_missing_round_commitment_when_a_root_is_cleared() {
        // Invariant: every round must expose a Merkle root.
        //
        // Fixture state: round 0 carries Some(root).
        //
        // Mutation: clear the slot.
        //
        //     proof.whir.rounds[0].commitment:  Some(root)  ->  None
        //     -> error identifies round = 0
        let (pcs, commitment, mut proof, protocol) = commit_and_open();
        assert!(
            !proof.whir.rounds.is_empty(),
            "fixture should produce at least one WHIR round"
        );
        proof.whir.rounds[0].commitment = None;

        let err = verify(&pcs, &commitment, &proof, protocol).unwrap_err();
        match err {
            VerifierError::MissingRoundCommitment { round } => {
                assert_eq!(round, 0);
            }
            other => panic!("expected MissingRoundCommitment, got {other:?}"),
        }
    }

    #[test]
    fn rejects_with_missing_final_poly_when_cleared() {
        // Invariant: the tail polynomial is required for the final identity check.
        //
        // Fixture state: final_poly = Some(tail).
        //
        // Mutation: clear the slot.
        //
        //     proof.whir.final_poly:  Some(tail)  ->  None
        let (pcs, commitment, mut proof, protocol) = commit_and_open();
        proof.whir.final_poly = None;

        let err = verify(&pcs, &commitment, &proof, protocol).unwrap_err();
        assert!(
            matches!(err, VerifierError::MissingFinalPoly),
            "expected MissingFinalPoly, got {err:?}"
        );
    }
}

mod keccak_tests {
    //! Same lifecycle test using Keccak-based Merkle trees over a different field.

    use alloc::vec;

    use p3_challenger::{HashChallenger, SerializingChallenger32};
    use p3_commit::MultilinearPcs;
    use p3_dft::Radix2DFTSmallBatch;
    use p3_field::extension::BinomialExtensionField;
    use p3_keccak::{Keccak256Hash, KeccakF};
    use p3_koala_bear::KoalaBear;
    use p3_merkle_tree::MerkleTreeMmcs;
    use p3_multilinear_util::poly::Poly;
    use p3_symmetric::{CompressionFunctionFromHasher, PaddingFreeSponge, SerializingHasher};
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use crate::fiat_shamir::domain_separator::DomainSeparator;
    use crate::parameters::{FoldingFactor, ProtocolParameters, SecurityAssumption, WhirConfig};
    use crate::pcs::prover::WhirProver;
    use crate::sumcheck::layout::{Layout, PrefixProver, SuffixProver, Table};
    use crate::sumcheck::{OpeningProtocol, TableShape, TableSpec};

    type F = KoalaBear;
    type EF = BinomialExtensionField<F, 4>;

    type U64Hash = PaddingFreeSponge<KeccakF, 25, 17, 4>;
    type KeccakFieldHash = SerializingHasher<U64Hash>;
    type KeccakCompress = CompressionFunctionFromHasher<U64Hash, 2, 4>;

    type KeccakChallenger = SerializingChallenger32<F, HashChallenger<u8, Keccak256Hash, 32>>;
    type MyMmcs = MerkleTreeMmcs<F, u64, KeccakFieldHash, KeccakCompress, 2, 4>;
    type MyDft = Radix2DFTSmallBatch<F>;
    type TestWhirPcs<L> = WhirProver<EF, F, MyDft, MyMmcs, KeccakChallenger, L>;

    fn challenger() -> KeccakChallenger {
        KeccakChallenger::new(HashChallenger::<u8, Keccak256Hash, 32>::new(
            vec![],
            Keccak256Hash {},
        ))
    }

    /// Runs the full commit + open + verify lifecycle with Keccak Merkle trees.
    fn run_keccak_end_to_end<L: Layout<F, EF>>() {
        // Fixture: a single-table polynomial of arity 16 folded 4 vars at a time.
        const NUM_VARIABLES: usize = 16;
        const FOLDING: usize = 4;

        // Build one random table, stack it through the chosen layout mode.
        let mut rng = SmallRng::seed_from_u64(1);
        let table = Table::new(vec![Poly::<F>::rand(&mut rng, NUM_VARIABLES)]);
        let witness = L::new_witness(vec![table], FOLDING);
        // Public protocol: open the single column at one point.
        let protocol = OpeningProtocol::new(vec![TableSpec::new(
            TableShape::new(NUM_VARIABLES, 1),
            vec![vec![0]],
        )]);
        assert_eq!(witness.table_shapes(), protocol.table_shapes());

        // Wire Keccak-f as both the leaf-hash sponge and the 2-to-1 compressor.
        let u64_hash = U64Hash::new(KeccakF {});
        let merkle_hash = KeccakFieldHash::new(u64_hash);
        let merkle_compress = KeccakCompress::new(u64_hash);
        let mmcs = MyMmcs::new(merkle_hash, merkle_compress, 0);

        // Security level 32 keeps the test fast; not a production setting.
        let params = ProtocolParameters {
            security_level: 32,
            pow_bits: 0,
            rs_domain_initial_reduction_factor: 1,
            folding_factor: FoldingFactor::Constant(FOLDING),
            soundness_type: SecurityAssumption::CapacityBound,
            starting_log_inv_rate: 1,
        };
        let pcs = TestWhirPcs::<L>::new(
            WhirConfig::new(witness.num_variables(), params),
            MyDft::default(),
            mmcs,
        );

        // Prover side: seed the transcript with the protocol description, commit, open.
        let (commitment, proof) = {
            let mut prover_challenger = challenger();
            let mut domain_separator = DomainSeparator::new(vec![]);
            pcs.add_domain_separator::<4>(&mut domain_separator);
            domain_separator.observe_domain_separator(&mut prover_challenger);

            let (commitment, prover_data) = <TestWhirPcs<L> as MultilinearPcs<
                EF,
                KeccakChallenger,
            >>::commit(
                &pcs, witness, &mut prover_challenger
            );
            let proof = <TestWhirPcs<L> as MultilinearPcs<EF, KeccakChallenger>>::open(
                &pcs,
                prover_data,
                protocol.clone(),
                &mut prover_challenger,
            );
            (commitment, proof)
        };

        // Verifier side: replay the same transcript prefix from a fresh challenger.
        let mut verifier_challenger = challenger();
        let mut domain_separator = DomainSeparator::new(vec![]);
        pcs.add_domain_separator::<4>(&mut domain_separator);
        domain_separator.observe_domain_separator(&mut verifier_challenger);

        // Final assertion: the honest proof must verify under both layout modes.
        <TestWhirPcs<L> as MultilinearPcs<EF, KeccakChallenger>>::verify(
            &pcs,
            &commitment,
            &proof,
            &mut verifier_challenger,
            protocol,
        )
        .expect("keccak verification failed");
    }

    #[test]
    fn test_whir_keccak_end_to_end_suffix() {
        // Suffix mode binds the SVO suffix variables first.
        run_keccak_end_to_end::<SuffixProver<F, EF>>();
    }

    #[test]
    fn test_whir_keccak_end_to_end_prefix() {
        // Prefix mode binds the SVO prefix variables first; covers the other layout path.
        run_keccak_end_to_end::<PrefixProver<F, EF>>();
    }
}
