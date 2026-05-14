//! End-to-end tests exercising the WHIR PCS through the multilinear trait.

use alloc::vec;
use alloc::vec::Vec;

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
use crate::sumcheck::test_util::{random_table_specs, table_specs_to_tables};
use crate::sumcheck::{OpeningProtocol, TableShape, TableSpec};

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

fn default_round_log_inv_rates(num_variables: usize, folding_factor: &FoldingFactor) -> Vec<usize> {
    let (num_rounds, _) = folding_factor.compute_number_of_rounds(num_variables);
    let mut rates = Vec::with_capacity(num_rounds);
    let mut rate = 1;
    for round in 0..num_rounds {
        rate += folding_factor.at_round(round) - 1;
        rates.push(rate);
    }
    rates
}

#[allow(clippy::too_many_arguments)]
fn run_whir_pcs<L: Layout<F, EF>>(
    specs: &[TableSpec],
    folding_factor: FoldingFactor,
    soundness_type: SecurityAssumption,
    pow_bits: usize,
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
    );
}

#[allow(clippy::too_many_arguments)]
fn run_whir_pcs_lifecycle_with_witness<L: Layout<F, EF>>(
    witness: Witness<F>,
    protocol: OpeningProtocol,
    folding_factor: FoldingFactor,
    soundness_type: SecurityAssumption,
    pow_bits: usize,
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
        round_log_inv_rates: default_round_log_inv_rates(num_variables, &folding_factor),
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

/// Smoke matrix covering each WHIR parameter axis at least once.
///
/// The full randomized sweep lives in [`test_whir_end_to_end_exhaustive`] and
/// runs from the Heavy CI workflow.
#[test]
fn test_whir_end_to_end() {
    let table_spec_sets = [
        vec![
            TableSpec::new(
                TableShape::new(12, 3),
                vec![vec![0, 1, 2], vec![0, 2], vec![1]],
            ),
            TableSpec::new(TableShape::new(10, 2), vec![vec![0, 1], vec![1]]),
        ],
        vec![TableSpec::new(
            TableShape::new(14, 4),
            vec![vec![0, 1, 2, 3], vec![0, 3]],
        )],
    ];

    let smoke_cases = [
        (
            FoldingFactor::Constant(1),
            SecurityAssumption::JohnsonBound,
            0,
        ),
        (
            FoldingFactor::Constant(2),
            SecurityAssumption::CapacityBound,
            5,
        ),
        (
            FoldingFactor::Constant(3),
            SecurityAssumption::UniqueDecoding,
            10,
        ),
        (
            FoldingFactor::Constant(4),
            SecurityAssumption::JohnsonBound,
            5,
        ),
        (
            FoldingFactor::ConstantFromSecondRound(2, 1),
            SecurityAssumption::CapacityBound,
            10,
        ),
        (
            FoldingFactor::ConstantFromSecondRound(3, 1),
            SecurityAssumption::UniqueDecoding,
            0,
        ),
        (
            FoldingFactor::ConstantFromSecondRound(3, 2),
            SecurityAssumption::JohnsonBound,
            10,
        ),
        (
            FoldingFactor::ConstantFromSecondRound(5, 2),
            SecurityAssumption::CapacityBound,
            5,
        ),
        (
            FoldingFactor::Constant(2),
            SecurityAssumption::UniqueDecoding,
            0,
        ),
        (
            FoldingFactor::ConstantFromSecondRound(5, 2),
            SecurityAssumption::JohnsonBound,
            10,
        ),
    ];

    for (i, (folding_factor, soundness_type, pow_bits)) in smoke_cases.into_iter().enumerate() {
        let specs = &table_spec_sets[i % table_spec_sets.len()];
        run_whir_pcs::<PrefixProver<F, EF>>(
            specs,
            folding_factor.clone(),
            soundness_type,
            pow_bits,
        );
        run_whir_pcs::<SuffixProver<F, EF>>(specs, folding_factor, soundness_type, pow_bits);
    }
}

#[test]
#[ignore = "exhaustive WHIR configuration sweep; run from heavy CI"]
fn test_whir_end_to_end_exhaustive() {
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
    let mut rng = SmallRng::seed_from_u64(7);

    for folding_factor in folding_factors {
        for soundness_type in soundness_type {
            for pow_bits in pow_bits {
                for _ in 0..N {
                    let specs = random_table_specs(&mut rng, folding_factor.at_round(0));
                    run_whir_pcs::<PrefixProver<F, EF>>(
                        &specs,
                        folding_factor.clone(),
                        soundness_type,
                        pow_bits,
                    );
                    run_whir_pcs::<SuffixProver<F, EF>>(
                        &specs,
                        folding_factor.clone(),
                        soundness_type,
                        pow_bits,
                    );
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
    use crate::parameters::{
        FoldingFactor, ProtocolParameters, SecurityAssumption, WhirConfig, WhirZkConfig,
    };
    use crate::pcs::proof::{PcsProof, WhirInitialZkProof, WhirRoundZkProof};
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
            round_log_inv_rates: vec![4],
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
    fn plain_verifier_rejects_zk_config_before_plain_round_checks() {
        let (mut pcs, commitment, mut proof, protocol) = commit_and_open();
        proof.whir.rounds.pop();
        pcs.config = pcs
            .config
            .clone()
            .with_zk_config(WhirZkConfig::prefix_only(4, 2, 1));

        let err = verify(&pcs, &commitment, &proof, protocol).unwrap_err();
        assert!(
            matches!(err, VerifierError::ZkVerifierRequiresPrefixPath),
            "expected ZkVerifierRequiresPrefixPath, got {err:?}"
        );
    }

    #[test]
    fn plain_verifier_rejects_unexpected_zk_payload() {
        let (pcs, commitment, mut proof, protocol) = commit_and_open();
        assert!(
            !proof.whir.rounds.is_empty(),
            "fixture should produce at least one WHIR round"
        );
        proof.whir.rounds[0].zk = Some(WhirRoundZkProof {
            mask_commitment: commitment.clone(),
            private_ood_answers: vec![],
            source_queries: vec![],
            mask_queries: vec![],
            zk_sumcheck: Default::default(),
            zk_sumcheck_mask_commitments: vec![],
        });

        let err = verify(&pcs, &commitment, &proof, protocol).unwrap_err();
        match err {
            VerifierError::UnexpectedZkPayloadInPlainProof { round } => {
                assert_eq!(round, 0);
            }
            other => panic!("expected UnexpectedZkPayloadInPlainProof, got {other:?}"),
        }
    }

    #[test]
    fn plain_verifier_rejects_unexpected_initial_zk_payload() {
        let (pcs, commitment, mut proof, protocol) = commit_and_open();
        proof.whir.initial_zk = Some(WhirInitialZkProof {
            zk_sumcheck: Default::default(),
            zk_sumcheck_mask_commitments: vec![commitment.clone()],
        });

        let err = verify(&pcs, &commitment, &proof, protocol).unwrap_err();
        assert!(
            matches!(err, VerifierError::UnexpectedInitialZkPayloadInPlainProof),
            "expected UnexpectedInitialZkPayloadInPlainProof, got {err:?}"
        );
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

    #[test]
    fn rejects_with_final_poly_length_mismatch_when_tail_has_extra_evals() {
        // Invariant: the final polynomial must have exactly the verifier-expected
        // number of evaluations before it is absorbed into the transcript.
        //
        // Mutation: duplicate the honest tail, preserving Poly's power-of-two
        // shape but changing the WHIR-level final length.
        let (pcs, commitment, mut proof, protocol) = commit_and_open();
        let final_poly = proof
            .whir
            .final_poly
            .as_ref()
            .expect("honest fixture should contain final_poly");
        let expected = final_poly.num_evals();
        let mut evals = final_poly.as_slice().to_vec();
        let duplicate = evals.clone();
        evals.extend_from_slice(&duplicate);
        proof.whir.final_poly = Some(Poly::new(evals));

        let err = verify(&pcs, &commitment, &proof, protocol).unwrap_err();
        match err {
            VerifierError::FinalPolyLengthMismatch {
                expected: e,
                actual,
            } => {
                assert_eq!(e, expected);
                assert_eq!(actual, expected * 2);
            }
            other => panic!("expected FinalPolyLengthMismatch, got {other:?}"),
        }
    }

    #[test]
    fn rejects_with_round_ood_answer_count_mismatch_when_answer_is_dropped() {
        // Invariant: each round carries exactly the verifier-expected OOD answers.
        //
        // Fixture state: round 0 has N OOD answers.
        //
        // Mutation: drop one answer.
        //
        //     proof.whir.rounds[0].ood_answers:  N  ->  N - 1
        let (pcs, commitment, mut proof, protocol) = commit_and_open();
        let expected = proof.whir.rounds[0].ood_answers.len();
        assert!(
            expected > 0,
            "fixture should produce at least one round-0 OOD answer"
        );
        proof.whir.rounds[0].ood_answers.pop();

        let err = verify(&pcs, &commitment, &proof, protocol).unwrap_err();
        match err {
            VerifierError::RoundOodAnswerCountMismatch {
                round,
                expected: e,
                actual: a,
            } => {
                assert_eq!(round, 0);
                assert_eq!(e, expected);
                assert_eq!(a, expected - 1);
            }
            other => panic!("expected RoundOodAnswerCountMismatch, got {other:?}"),
        }
    }

    #[test]
    fn rejects_with_stir_query_count_mismatch_when_intermediate_query_is_dropped() {
        // Invariant: queries.len() == verifier-sampled indices for the round.
        //
        // Mutation: drop the trailing query from round 0.
        //
        //     proof.whir.rounds[0].queries:  n  ->  n - 1
        //     -> round_index = 0, expected = n, actual = n - 1
        let (pcs, commitment, mut proof, protocol) = commit_and_open();
        let expected = proof.whir.rounds[0].queries.len();
        assert!(
            expected > 0,
            "fixture should produce at least one STIR query"
        );
        proof.whir.rounds[0].queries.pop();

        let err = verify(&pcs, &commitment, &proof, protocol).unwrap_err();
        match err {
            VerifierError::StirQueryCountMismatch {
                round_index,
                expected: e,
                actual: a,
            } => {
                assert_eq!(round_index, 0);
                assert_eq!(e, expected);
                assert_eq!(a, expected - 1);
            }
            other => panic!("expected StirQueryCountMismatch, got {other:?}"),
        }
    }

    #[test]
    fn rejects_with_stir_query_count_mismatch_when_final_query_is_dropped() {
        // Invariant: final_queries.len() == verifier-sampled indices for the final round.
        //
        // Mutation: drop the trailing query from final_queries.
        //
        //     proof.whir.final_queries:  n  ->  n - 1
        //     -> round_index = n_rounds, expected = n, actual = n - 1
        let (pcs, commitment, mut proof, protocol) = commit_and_open();
        let n_rounds = pcs.n_rounds();
        let expected = proof.whir.final_queries.len();
        assert!(
            expected > 0,
            "fixture should produce at least one final query"
        );
        proof.whir.final_queries.pop();

        let err = verify(&pcs, &commitment, &proof, protocol).unwrap_err();
        match err {
            VerifierError::StirQueryCountMismatch {
                round_index,
                expected: e,
                actual: a,
            } => {
                assert_eq!(round_index, n_rounds);
                assert_eq!(e, expected);
                assert_eq!(a, expected - 1);
            }
            other => panic!("expected StirQueryCountMismatch, got {other:?}"),
        }
    }
}

mod zk_prefix_api_tests {
    //! Prefix-only ZK opening entrypoint tests.

    use alloc::vec;
    use alloc::vec::Vec;

    use p3_commit::MultilinearPcs;
    use p3_dft::Radix2Dit;
    use p3_field::{Field, PrimeCharacteristicRing, dot_product};
    use p3_multilinear_util::poly::Poly;
    use p3_zk_codes::ReedSolomonZkEncoding;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use super::{
        EF, F, MyChallenger, MyCompress, MyDft, MyHash, MyMmcs, Perm, TestWhirPcs, challenger,
    };
    use crate::fiat_shamir::domain_separator::DomainSeparator;
    use crate::parameters::{
        FoldingFactor, ProtocolParameters, SecurityAssumption, WhirConfig, WhirZkConfig,
    };
    use crate::pcs::adapter::{
        ZkCodeSwitchProverSource, ZkCodeSwitchVerifierSource, ZkEncodedCodeSwitchVerifierSource,
        evaluate_zk_mask_residual,
    };
    use crate::pcs::proof::QueryOpening;
    use crate::sumcheck::layout::{Layout, PrefixProver, Table};
    use crate::sumcheck::{OpeningProtocol, TableShape, TableSpec};

    type L = PrefixProver<F, EF>;

    const NUM_VARIABLES: usize = 16;
    const FOLDING: usize = 4;
    const ZK_MESSAGE_LEN: usize = 4;

    #[test]
    fn begin_zk_prefix_open_records_initial_handoff() {
        let (pcs, witness, protocol, required_query_bound, expected_mask_domain) = setup();

        let mut prover_challenger = challenger();
        let mut domain_separator = DomainSeparator::new(vec![]);
        pcs.add_domain_separator::<8>(&mut domain_separator);
        domain_separator.observe_domain_separator(&mut prover_challenger);

        let (_commitment, prover_data) =
            <TestWhirPcs<L> as MultilinearPcs<EF, MyChallenger>>::commit(
                &pcs,
                witness,
                &mut prover_challenger,
            );
        let encoding = ReedSolomonZkEncoding::new(
            required_query_bound,
            ZK_MESSAGE_LEN,
            expected_mask_domain,
            MyDft::default(),
        );
        let mut zk_rng = SmallRng::seed_from_u64(7);
        let state = pcs.begin_zk_prefix_open(
            prover_data,
            &protocol,
            &mut prover_challenger,
            encoding,
            &mut zk_rng,
        );

        let initial_zk = state
            .proof
            .whir
            .initial_zk
            .as_ref()
            .expect("ZK prefix entrypoint must record the initial ZK transcript");
        assert_eq!(state.proof.evals.len(), protocol.num_openings());
        assert_eq!(initial_zk.zk_sumcheck.ell_zk, ZK_MESSAGE_LEN);
        assert_eq!(initial_zk.zk_sumcheck.round_coefficients.len(), FOLDING);
        assert_eq!(initial_zk.zk_sumcheck_mask_commitments.len(), FOLDING);
        assert_eq!(state.initial_handoff.randomness.num_variables(), FOLDING);
        assert_eq!(state.initial_handoff.mask_oracles.len(), FOLDING);
    }

    #[test]
    fn round_zk_prefix_populates_first_round_payload() {
        let (pcs, witness, protocol, required_query_bound, expected_mask_domain) = setup();

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
        let initial_mask_encoding = ReedSolomonZkEncoding::new(
            required_query_bound,
            ZK_MESSAGE_LEN,
            expected_mask_domain,
            MyDft::default(),
        );
        let mut zk_rng = SmallRng::seed_from_u64(7);
        let state = pcs.begin_zk_prefix_open(
            prover_data,
            &protocol,
            &mut prover_challenger,
            initial_mask_encoding,
            &mut zk_rng,
        );
        let source_message = state
            .initial_handoff
            .residual_prover
            .evals()
            .as_slice()
            .to_vec();
        let inherited_claim = state.initial_handoff.residual_prover.claimed_sum();
        let source_scale = state.initial_handoff.eps;
        let source_claim = inherited_claim / source_scale;
        let (pivot, &pivot_value) = source_message
            .iter()
            .enumerate()
            .find(|(_, value)| !value.is_zero())
            .expect("test source message should contain a nonzero entry");
        let mut source_covector = EF::zero_vec(source_message.len());
        source_covector[pivot] = source_claim / pivot_value;
        let target_num_variables =
            pcs.config.num_variables - pcs.config.folding_factor.total_number(0);
        let target_domain_size = pcs.config.inv_rate(0) * (1usize << target_num_variables);
        let target_folding = pcs.config.folding_factor.at_round(1);

        let round_zk = pcs.config.round_parameters[0].zk.as_ref().unwrap();
        let round_mask_encoding = ReedSolomonZkEncoding::new(
            round_zk.mask_query_budget,
            round_zk.mask_message_len,
            round_zk.mask_domain_size,
            MyDft::default(),
        );

        let round_state = pcs.round_zk_prefix(
            state,
            0,
            &source_covector,
            &round_mask_encoding,
            &mut prover_challenger,
            &mut zk_rng,
        );
        let round = &round_state.proof.whir.rounds[0];
        let round_zk_proof = round
            .zk
            .as_ref()
            .expect("round_zk_prefix must populate the first ZK round payload");

        assert!(round.commitment.is_some());
        assert_eq!(
            round_zk_proof.private_ood_answers.len(),
            round_zk.ood_samples
        );
        assert_eq!(
            round_zk_proof.source_queries.len(),
            round_zk.mask_query_budget
        );
        assert!(
            round_zk_proof
                .source_queries
                .iter()
                .all(|query| matches!(query, QueryOpening::Extension { .. })),
            "code-switch source openings must be tied to the committed target oracle",
        );
        assert_eq!(
            round_zk_proof.mask_queries.len(),
            round_zk.mask_query_budget
        );
        assert_eq!(
            round_zk_proof.zk_sumcheck.round_coefficients.len(),
            pcs.config.folding_factor.at_round(1),
        );
        assert_eq!(
            round_zk_proof.zk_sumcheck_mask_commitments.len(),
            pcs.config.folding_factor.at_round(1),
        );
        assert_eq!(
            round_state.handoff.randomness.num_variables(),
            pcs.config.folding_factor.at_round(1),
        );
        let next_source = round_state
            .next_source
            .as_ref()
            .expect("multi-round fixture should carry a source for the next ZK round");
        assert_eq!(
            next_source.message.as_slice(),
            round_state.handoff.residual_prover.evals().as_slice(),
        );
        assert_eq!(
            next_source.covector.as_slice(),
            round_state.handoff.residual_prover.weights().as_slice(),
        );
        assert_eq!(
            next_source.inherited_claim,
            round_state.handoff.residual_prover.claimed_sum(),
        );
        assert_eq!(
            next_source.inherited_claim,
            dot_product::<EF, _, _>(
                next_source.message.iter().copied(),
                next_source.covector.iter().copied(),
            ),
        );
        assert_eq!(next_source.residual_sumcheck_scale, EF::ONE);
        assert_eq!(next_source.randomness_len, 0);

        let mut verifier_challenger = challenger();
        domain_separator.observe_domain_separator(&mut verifier_challenger);
        let verifier_source = ZkCodeSwitchVerifierSource {
            commitment,
            message_len: source_message.len(),
            covector: source_covector,
            residual_sumcheck_scale: source_scale,
            randomness_len: 0,
            domain_size: target_domain_size,
            folding_factor: target_folding,
        };
        let verifier_handoff = pcs
            .verify_round_zk_prefix(
                &round_state.proof,
                &protocol,
                &verifier_source,
                &mut verifier_challenger,
            )
            .expect("verifier should replay the first ZK round");
        assert_eq!(
            verifier_handoff.randomness.num_variables(),
            pcs.config.folding_factor.at_round(1),
        );
        let nested_gammas = round_state
            .handoff
            .randomness
            .iter()
            .copied()
            .collect::<Vec<_>>();
        let nested_mask_residual =
            evaluate_zk_mask_residual::<F, EF>(&round_state.handoff.mask_messages, &nested_gammas);
        assert_eq!(
            verifier_handoff.claimed_residual,
            round_state.handoff.residual_prover.claimed_sum() + nested_mask_residual,
        );

        let round1_zk = pcs.config.round_parameters[1].zk.as_ref().unwrap();
        let source_domain =
            (next_source.message.len() + round1_zk.mask_query_budget).next_power_of_two();
        let encoded_source = ZkCodeSwitchProverSource {
            message: next_source.message.clone(),
            covector: next_source.covector.clone(),
            inherited_claim: next_source.inherited_claim,
            residual_sumcheck_scale: next_source.residual_sumcheck_scale,
            randomness_len: 0,
            domain_size: source_domain,
            folding_factor: 0,
        };
        let source_encoding = ReedSolomonZkEncoding::<EF, Radix2Dit<EF>>::new(
            round1_zk.mask_query_budget,
            encoded_source.message.len(),
            source_domain,
            Radix2Dit::default(),
        );
        let code_switch_mask_encoding = ReedSolomonZkEncoding::<EF, Radix2Dit<EF>>::new(
            round1_zk.mask_query_budget,
            round1_zk.mask_message_len,
            round1_zk.mask_domain_size,
            Radix2Dit::default(),
        );
        let sumcheck_mask_encoding = ReedSolomonZkEncoding::new(
            round1_zk.mask_query_budget,
            round1_zk.mask_message_len,
            round1_zk.mask_domain_size,
            MyDft::default(),
        );
        let round1_state = pcs.round_zk_prefix_from_encoded_source(
            round_state.proof.clone(),
            &round_state.handoff,
            1,
            &encoded_source,
            &source_encoding,
            &code_switch_mask_encoding,
            &sumcheck_mask_encoding,
            &mut prover_challenger,
            &mut zk_rng,
        );
        let round1 = &round1_state.proof.whir.rounds[1];
        let round1_zk_proof = round1
            .zk
            .as_ref()
            .expect("encoded source consumer must populate the second ZK round");
        assert!(round1.commitment.is_some());
        assert_eq!(
            round1.commitment.as_ref(),
            Some(&round1_state.source_commitment)
        );
        assert!(
            round1_zk_proof.source_queries.iter().all(
                |query| matches!(query, QueryOpening::Extension { values, .. } if values.len() == 1)
            ),
            "encoded source openings must be single extension-field positions",
        );
        assert!(
            round1_zk_proof.mask_queries.iter().all(
                |query| matches!(query, QueryOpening::Extension { values, .. } if values.len() == 1)
            ),
            "encoded source rounds carry EF code-switch mask openings",
        );
        assert_eq!(
            round1_state.handoff.randomness.num_variables(),
            pcs.config.folding_factor.at_round(2),
        );
        let round1_verifier_source = ZkEncodedCodeSwitchVerifierSource {
            message_len: encoded_source.message.len(),
            covector: encoded_source.covector,
            residual_sumcheck_scale: encoded_source.residual_sumcheck_scale,
            domain_size: source_domain,
            randomness_len: round1_zk.mask_query_budget,
        };
        let round1_verifier_handoff = pcs
            .verify_round_zk_prefix_from_encoded_source(
                &round1_state.proof,
                &verifier_handoff,
                1,
                &round1_verifier_source,
                &source_encoding,
                &mut verifier_challenger,
            )
            .expect("verifier should replay the encoded-source ZK round");
        assert_eq!(
            round1_verifier_handoff.randomness.num_variables(),
            pcs.config.folding_factor.at_round(2),
        );
    }

    #[test]
    #[should_panic(expected = "source randomness needs a source-oracle randomness handoff")]
    fn round_zk_prefix_rejects_nonzero_source_randomness_for_first_target_oracle() {
        let (pcs, witness, protocol, required_query_bound, expected_mask_domain) = setup();

        let mut prover_challenger = challenger();
        let mut domain_separator = DomainSeparator::new(vec![]);
        pcs.add_domain_separator::<8>(&mut domain_separator);
        domain_separator.observe_domain_separator(&mut prover_challenger);

        let (_commitment, prover_data) =
            <TestWhirPcs<L> as MultilinearPcs<EF, MyChallenger>>::commit(
                &pcs,
                witness,
                &mut prover_challenger,
            );
        let initial_mask_encoding = ReedSolomonZkEncoding::new(
            required_query_bound,
            ZK_MESSAGE_LEN,
            expected_mask_domain,
            MyDft::default(),
        );
        let mut zk_rng = SmallRng::seed_from_u64(7);
        let state = pcs.begin_zk_prefix_open(
            prover_data,
            &protocol,
            &mut prover_challenger,
            initial_mask_encoding,
            &mut zk_rng,
        );
        let source_message = state
            .initial_handoff
            .residual_prover
            .evals()
            .as_slice()
            .to_vec();
        let inherited_claim = state.initial_handoff.residual_prover.claimed_sum();
        let source_scale = state.initial_handoff.eps;
        let source_claim = inherited_claim / source_scale;
        let (pivot, &pivot_value) = source_message
            .iter()
            .enumerate()
            .find(|(_, value)| !value.is_zero())
            .expect("test source message should contain a nonzero entry");
        let mut source_covector = EF::zero_vec(source_message.len());
        source_covector[pivot] = source_claim / pivot_value;
        let target_num_variables =
            pcs.config.num_variables - pcs.config.folding_factor.total_number(0);
        let target_domain_size = pcs.config.inv_rate(0) * (1usize << target_num_variables);
        let target_folding = pcs.config.folding_factor.at_round(1);

        let source = ZkCodeSwitchProverSource {
            message: source_message,
            covector: source_covector,
            inherited_claim,
            residual_sumcheck_scale: source_scale,
            randomness_len: 1,
            domain_size: target_domain_size,
            folding_factor: target_folding,
        };
        let round_zk = pcs.config.round_parameters[0].zk.as_ref().unwrap();
        let round_mask_encoding = ReedSolomonZkEncoding::new(
            round_zk.mask_query_budget,
            round_zk.mask_message_len,
            round_zk.mask_domain_size,
            MyDft::default(),
        );

        let _ = pcs.round0_zk_prefix_from_folded_source(
            state.proof,
            &state.initial_handoff,
            &source,
            &round_mask_encoding,
            &mut prover_challenger,
            &mut zk_rng,
        );
    }

    #[test]
    #[should_panic(expected = "ZK encoding codeword length must match the derived mask domain")]
    fn begin_zk_prefix_open_rejects_encoding_domain_mismatch() {
        let (pcs, witness, protocol, required_query_bound, expected_mask_domain) = setup();
        let too_small_domain = (ZK_MESSAGE_LEN + required_query_bound).next_power_of_two();
        assert!(
            too_small_domain < expected_mask_domain,
            "fixture should reproduce the old undersized encoding domain",
        );

        let mut prover_challenger = challenger();
        let mut domain_separator = DomainSeparator::new(vec![]);
        pcs.add_domain_separator::<8>(&mut domain_separator);
        domain_separator.observe_domain_separator(&mut prover_challenger);

        let (_commitment, prover_data) =
            <TestWhirPcs<L> as MultilinearPcs<EF, MyChallenger>>::commit(
                &pcs,
                witness,
                &mut prover_challenger,
            );
        let encoding = ReedSolomonZkEncoding::new(
            required_query_bound,
            ZK_MESSAGE_LEN,
            too_small_domain,
            MyDft::default(),
        );
        let mut zk_rng = SmallRng::seed_from_u64(7);
        let _ = pcs.begin_zk_prefix_open(
            prover_data,
            &protocol,
            &mut prover_challenger,
            encoding,
            &mut zk_rng,
        );
    }

    #[test]
    #[should_panic(expected = "ZK encoding codeword length must match the derived mask domain")]
    fn begin_zk_prefix_open_rejects_later_round_domain_for_initial_handoff() {
        let (mut pcs, witness, protocol, required_query_bound, expected_mask_domain) = setup();
        assert!(
            pcs.config.round_parameters.len() > 1,
            "fixture should contain a later round to distinguish first-consumer and max domains",
        );
        let later_domain = expected_mask_domain * 2;
        pcs.config.round_parameters[1]
            .zk
            .as_mut()
            .expect("ZK config should be derived for the later round")
            .mask_domain_size = later_domain;

        let mut prover_challenger = challenger();
        let mut domain_separator = DomainSeparator::new(vec![]);
        pcs.add_domain_separator::<8>(&mut domain_separator);
        domain_separator.observe_domain_separator(&mut prover_challenger);

        let (_commitment, prover_data) =
            <TestWhirPcs<L> as MultilinearPcs<EF, MyChallenger>>::commit(
                &pcs,
                witness,
                &mut prover_challenger,
            );
        let encoding = ReedSolomonZkEncoding::new(
            required_query_bound,
            ZK_MESSAGE_LEN,
            later_domain,
            MyDft::default(),
        );
        let mut zk_rng = SmallRng::seed_from_u64(7);
        let _ = pcs.begin_zk_prefix_open(
            prover_data,
            &protocol,
            &mut prover_challenger,
            encoding,
            &mut zk_rng,
        );
    }

    fn setup() -> (
        TestWhirPcs<L>,
        crate::sumcheck::layout::Witness<F>,
        OpeningProtocol,
        usize,
        usize,
    ) {
        let mut rng = SmallRng::seed_from_u64(1);
        let table = Table::new(vec![
            Poly::<F>::rand(&mut rng, NUM_VARIABLES),
            Poly::<F>::rand(&mut rng, NUM_VARIABLES),
        ]);
        let witness = L::new_witness(vec![table], FOLDING);
        let protocol = OpeningProtocol::new(vec![TableSpec::new(
            TableShape::new(NUM_VARIABLES, 2),
            vec![vec![0, 1], vec![0]],
        )]);

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
        let config = WhirConfig::new(witness.num_variables(), params)
            .with_zk_config(WhirZkConfig::prefix_only(ZK_MESSAGE_LEN, 2, 1));
        let required_query_bound = config
            .round_parameters
            .first()
            .and_then(|round| round.zk.as_ref())
            .map(|zk| zk.mask_query_budget)
            .expect("ZK config should derive a first-round mask query budget");
        let expected_mask_domain = config
            .round_parameters
            .first()
            .and_then(|round| round.zk.as_ref())
            .map(|zk| zk.mask_domain_size)
            .expect("ZK config should derive a first-round mask domain");
        let pcs = TestWhirPcs::<L>::new(config, MyDft::default(), mmcs);

        (
            pcs,
            witness,
            protocol,
            required_query_bound,
            expected_mask_domain,
        )
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
            round_log_inv_rates: vec![4, 7],
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
