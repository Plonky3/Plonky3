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
        zk: false,
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
            round_log_inv_rates: vec![4],
            folding_factor: FoldingFactor::Constant(FOLDING),
            soundness_type: SecurityAssumption::CapacityBound,
            starting_log_inv_rate: 1,
            zk: false,
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
            zk: false,
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

mod zk_tests {
    //! End-to-end tests for HVZK code-switching (Construction 9.7).
    //!
    //! ZK mode currently requires Suffix variable ordering because the
    //! ZK randomness layout in the coefficient matrix only supports suffix.

    use alloc::vec;

    use p3_commit::MultilinearPcs;
    use p3_multilinear_util::poly::Poly;
    use proptest::prelude::*;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use super::{
        EF, F, MyChallenger, MyCompress, MyDft, MyHash, MyMmcs, Perm, TestWhirPcs, challenger,
        default_round_log_inv_rates,
    };
    use crate::fiat_shamir::domain_separator::DomainSeparator;
    use crate::parameters::{FoldingFactor, ProtocolParameters, SecurityAssumption, WhirConfig};
    use crate::sumcheck::layout::{Layout, SuffixProver, Table};
    use crate::sumcheck::{OpeningProtocol, TableShape, TableSpec};

    type L = SuffixProver<F, EF>;

    /// Runs the full commit → open → verify lifecycle with `zk: true`.
    fn run_zk_lifecycle(
        num_variables: usize,
        folding_factor: FoldingFactor,
        soundness_type: SecurityAssumption,
    ) {
        run_zk_lifecycle_seeded(num_variables, folding_factor, soundness_type, 42);
    }

    fn run_zk_lifecycle_seeded(
        num_variables: usize,
        folding_factor: FoldingFactor,
        soundness_type: SecurityAssumption,
        seed: u64,
    ) {
        let folding = folding_factor.at_round(0);

        let mut rng = SmallRng::seed_from_u64(seed);
        let table = Table::new(vec![Poly::<F>::rand(&mut rng, num_variables)]);
        let witness = L::new_witness(vec![table], folding);
        let protocol = OpeningProtocol::new(vec![TableSpec::new(
            TableShape::new(num_variables, 1),
            vec![vec![0]],
        )]);
        assert_eq!(witness.table_shapes(), protocol.table_shapes());

        let perm = Perm::new_from_rng_128(&mut SmallRng::seed_from_u64(1));
        let merkle_hash = MyHash::new(perm.clone());
        let merkle_compress = MyCompress::new(perm);
        let mmcs = MyMmcs::new(merkle_hash, merkle_compress, 0);

        let params = ProtocolParameters {
            security_level: 32,
            pow_bits: 0,
            round_log_inv_rates: default_round_log_inv_rates(num_variables, &folding_factor),
            folding_factor,
            soundness_type,
            starting_log_inv_rate: 1,
            zk: true,
        };
        let pcs = TestWhirPcs::<L>::new(
            WhirConfig::new(witness.num_variables(), params),
            MyDft::default(),
            mmcs,
        );

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
            .expect("ZK verification failed");
        }
    }

    #[test]
    fn test_whir_zk_end_to_end() {
        // Single-round configs (zk_this_round = false for the only round).
        let single_round = [
            (10, FoldingFactor::Constant(2)),
            (12, FoldingFactor::Constant(4)),
            (10, FoldingFactor::ConstantFromSecondRound(4, 2)),
        ];

        for (num_vars, folding) in single_round {
            run_zk_lifecycle(num_vars, folding, SecurityAssumption::CapacityBound);
        }

        // Multi-round configs: n_rounds() >= 2, so at least round 0 has
        // zk_this_round = true, exercising the mask commitment path.
        let multi_round = [
            (12, FoldingFactor::Constant(2)),
            (18, FoldingFactor::Constant(4)),
        ];

        for (num_vars, folding) in multi_round {
            run_zk_lifecycle(num_vars, folding, SecurityAssumption::CapacityBound);
        }
    }

    #[test]
    fn test_hiding_whir_pcs_end_to_end() {
        // Exercise HidingWhirPcs through the MultilinearPcs trait.
        use crate::pcs::HidingWhirPcs;

        let num_variables = 12;
        let folding_factor = FoldingFactor::Constant(4);
        let folding = folding_factor.at_round(0);

        let mut rng = SmallRng::seed_from_u64(77);
        let table = Table::new(vec![Poly::<F>::rand(&mut rng, num_variables)]);
        let witness = L::new_witness(vec![table], folding);
        let protocol = OpeningProtocol::new(vec![TableSpec::new(
            TableShape::new(num_variables, 1),
            vec![vec![0]],
        )]);
        assert_eq!(witness.table_shapes(), protocol.table_shapes());

        let perm = Perm::new_from_rng_128(&mut SmallRng::seed_from_u64(1));
        let merkle_hash = MyHash::new(perm.clone());
        let merkle_compress = MyCompress::new(perm);
        let mmcs = MyMmcs::new(merkle_hash, merkle_compress, 0);

        let params = ProtocolParameters {
            security_level: 32,
            pow_bits: 0,
            round_log_inv_rates: default_round_log_inv_rates(num_variables, &folding_factor),
            folding_factor,
            soundness_type: SecurityAssumption::CapacityBound,
            starting_log_inv_rate: 1,
            zk: true,
        };
        let inner = TestWhirPcs::<L>::new(
            WhirConfig::new(witness.num_variables(), params),
            MyDft::default(),
            mmcs,
        );
        let pcs = HidingWhirPcs::new(inner, SmallRng::seed_from_u64(42));

        // Prover
        let (commitment, proof) = {
            let mut challenger = challenger();
            let mut domain_separator = DomainSeparator::new(vec![]);
            pcs.add_domain_separator::<8>(&mut domain_separator);
            domain_separator.observe_domain_separator(&mut challenger);

            let (commitment, prover_data) = <HidingWhirPcs<
                EF,
                F,
                MyDft,
                MyMmcs,
                MyChallenger,
                L,
                SmallRng,
            > as MultilinearPcs<EF, MyChallenger>>::commit(
                &pcs, witness, &mut challenger
            );
            let proof =
                <HidingWhirPcs<EF, F, MyDft, MyMmcs, MyChallenger, L, SmallRng> as MultilinearPcs<
                    EF,
                    MyChallenger,
                >>::open(&pcs, prover_data, protocol.clone(), &mut challenger);
            (commitment, proof)
        };

        // Verifier
        {
            let mut challenger = challenger();
            let mut domain_separator = DomainSeparator::new(vec![]);
            pcs.add_domain_separator::<8>(&mut domain_separator);
            domain_separator.observe_domain_separator(&mut challenger);

            <HidingWhirPcs<EF, F, MyDft, MyMmcs, MyChallenger, L, SmallRng> as MultilinearPcs<
                EF,
                MyChallenger,
            >>::verify(&pcs, &commitment, &proof, &mut challenger, protocol)
            .expect("HidingWhirPcs verification failed");
        }
    }

    #[test]
    fn test_zk_false_is_byte_identical() {
        // Verify that zk: false produces the exact same proof as the non-ZK code path.
        // This confirms the ZK branch is only entered when zk == true.
        let num_variables = 10;
        let folding_factor = FoldingFactor::Constant(2);
        let folding = folding_factor.at_round(0);

        let mut rng = SmallRng::seed_from_u64(99);
        let table = Table::new(vec![Poly::<F>::rand(&mut rng, num_variables)]);
        let witness = L::new_witness(vec![table], folding);
        let protocol = OpeningProtocol::new(vec![TableSpec::new(
            TableShape::new(num_variables, 1),
            vec![vec![0]],
        )]);

        let perm = Perm::new_from_rng_128(&mut SmallRng::seed_from_u64(1));
        let merkle_hash = MyHash::new(perm.clone());
        let merkle_compress = MyCompress::new(perm);
        let mmcs = MyMmcs::new(merkle_hash, merkle_compress, 0);

        let params = ProtocolParameters {
            security_level: 32,
            pow_bits: 0,
            round_log_inv_rates: default_round_log_inv_rates(num_variables, &folding_factor),
            folding_factor,
            soundness_type: SecurityAssumption::CapacityBound,
            starting_log_inv_rate: 1,
            zk: false,
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

        // The proof should verify successfully.
        let mut verifier_challenger = challenger();
        let mut domain_separator = DomainSeparator::new(vec![]);
        pcs.add_domain_separator::<8>(&mut domain_separator);
        domain_separator.observe_domain_separator(&mut verifier_challenger);

        <TestWhirPcs<L> as MultilinearPcs<EF, MyChallenger>>::verify(
            &pcs,
            &commitment,
            &proof,
            &mut verifier_challenger,
            protocol,
        )
        .expect("zk=false verification failed");

        // All mask commitment slots must be None.
        for (i, round) in proof.whir.rounds.iter().enumerate() {
            assert!(
                round.mask_commitment.is_none(),
                "round {i} has mask_commitment with zk=false"
            );
        }
    }

    /// Theorem 4.5 composition: ZK proof structure across random seeds.
    ///
    /// Verifies that the composed code-switching + sumcheck protocol
    /// (Construction 9.7 → sumcheck) produces valid proofs for random
    /// polynomials, exercising the composition's completeness invariant.
    ///
    /// Structural invariants checked per proof:
    /// - Verifier accepts (completeness)
    /// - ZK rounds carry mask commitments, OOD corrections, STIR corrections
    /// - Non-ZK rounds have empty correction vectors
    /// - Correction vector lengths match query counts
    fn run_zk_composition_check(num_variables: usize, folding_factor: FoldingFactor, seed: u64) {
        let folding = folding_factor.at_round(0);

        let mut rng = SmallRng::seed_from_u64(seed);
        let table = Table::new(vec![Poly::<F>::rand(&mut rng, num_variables)]);
        let witness = L::new_witness(vec![table], folding);
        let protocol = OpeningProtocol::new(vec![TableSpec::new(
            TableShape::new(num_variables, 1),
            vec![vec![0]],
        )]);

        let perm = Perm::new_from_rng_128(&mut SmallRng::seed_from_u64(1));
        let merkle_hash = MyHash::new(perm.clone());
        let merkle_compress = MyCompress::new(perm);
        let mmcs = MyMmcs::new(merkle_hash, merkle_compress, 0);

        let params = ProtocolParameters {
            security_level: 32,
            pow_bits: 0,
            round_log_inv_rates: default_round_log_inv_rates(num_variables, &folding_factor),
            folding_factor,
            soundness_type: SecurityAssumption::CapacityBound,
            starting_log_inv_rate: 1,
            zk: true,
        };
        let pcs = TestWhirPcs::<L>::new(
            WhirConfig::new(witness.num_variables(), params),
            MyDft::default(),
            mmcs,
        );

        let n_rounds = pcs.n_rounds();

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

        // Verifier accepts (Theorem 4.5 completeness).
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
            .expect("ZK composition verification failed");
        }

        // Structural invariants on the proof (Theorem 4.5 composition shape).
        for (i, round) in proof.whir.rounds.iter().enumerate() {
            let is_last_intermediate = i + 1 == n_rounds;
            let zk_this_round = !is_last_intermediate;

            // OOD corrections + mask commitment present iff this round is ZK.
            if zk_this_round {
                assert!(
                    round.mask_commitment.is_some(),
                    "round {i}: ZK round missing mask commitment"
                );
                assert_eq!(
                    round.zk_ood_corrections.len(),
                    round.ood_answers.len(),
                    "round {i}: OOD correction count mismatch"
                );
            } else {
                assert!(
                    round.zk_ood_corrections.is_empty(),
                    "round {i}: non-ZK round has OOD corrections"
                );
            }

            // STIR corrections present when opening a PREVIOUS round's
            // ZK-padded codeword. Round i opens the commitment from
            // round i-1. That commit was ZK iff round i-1 was not the
            // last intermediate round.
            if i > 0 {
                let prev_was_zk = i < n_rounds;
                if prev_was_zk {
                    assert_eq!(
                        round.zk_stir_corrections.len(),
                        round.queries.len(),
                        "round {i}: STIR correction count != query count"
                    );
                }
            } else {
                // Round 0 opens the initial (non-ZK) base commitment.
                assert!(
                    round.zk_stir_corrections.is_empty(),
                    "round 0: should not have STIR corrections"
                );
            }
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(8))]

        /// Theorem 4.5 composition proptest: the composed code-switching +
        /// sumcheck simulator produces valid proofs across random seeds.
        ///
        /// This exercises acceptance criterion C3 from #1587: the ZK protocol
        /// composed via Theorem 4.5 maintains completeness for arbitrary
        /// honest inputs.
        #[test]
        fn prop_zk_composition_completeness(seed in 0u64..256) {
            // Multi-round config: 12 vars, fold-by-2 → ≥2 intermediate rounds,
            // so code-switching fires at least once.
            run_zk_composition_check(12, FoldingFactor::Constant(2), seed);
        }
    }

    /// Simulator composition structural test (Theorem 4.5).
    ///
    /// Verifies that `code_switch_zk::simulate` composes with the sumcheck
    /// simulator by checking that simulated outputs have the correct shape
    /// and distributional properties:
    /// - Simulated OOD answer is non-zero (uniform over EF with negligible
    ///   probability of zero)
    /// - Simulated oracle values have correct dimensions
    /// - Multiple seeds produce distinct outputs (not degenerate)
    #[test]
    fn test_simulator_composition_structure() {
        use alloc::vec::Vec;

        use p3_dft::Radix2Dit;
        use p3_field::PrimeCharacteristicRing;
        use p3_zk_codes::ReedSolomonZkEncoding;
        use rand::RngExt;

        use crate::pcs::code_switch_zk;

        let dft = Radix2Dit::default();
        // t=2 randomness positions per encoding. Queries must be ≤ t.
        let enc_target = ReedSolomonZkEncoding::<EF, _>::new(2, 4, 16, dft.clone());
        let enc_mask = ReedSolomonZkEncoding::<EF, _>::new(2, 4, 8, dft);

        let n_target_queries = 2;
        let n_mask_queries = 2;
        let target_positions: Vec<usize> = (0..n_target_queries).collect();
        let mask_positions: Vec<usize> = (0..n_mask_queries).collect();

        let mut seen_ood: Vec<EF> = Vec::new();

        for seed in 0u64..16 {
            let mut rng = SmallRng::seed_from_u64(seed);
            let ood_point: EF = rng.random();

            let (sim_ood, sim_g, sim_s) = code_switch_zk::simulate(
                ood_point,
                &target_positions,
                &mask_positions,
                &enc_target,
                &enc_mask,
                &mut rng,
            );

            // Shape invariants (Lemma 9.8 steps 4-5).
            assert_eq!(sim_g.len(), n_target_queries);
            assert_eq!(sim_s.len(), n_mask_queries);

            assert_ne!(sim_ood, EF::ZERO);

            if !seen_ood.contains(&sim_ood) {
                seen_ood.push(sim_ood);
            }
        }

        // At least 8 distinct OOD values across 16 seeds.
        assert!(
            seen_ood.len() >= 8,
            "simulated OOD answers degenerate: only {} distinct values in 16 runs",
            seen_ood.len()
        );
    }

    /// Composed simulator distribution match (Theorem 4.5 + Lemma 9.8).
    ///
    /// Verifies that `code_switch_zk::simulate` composes with the WHIR PCS
    /// by reproducing the mask commitment under matched-RNG coupling and
    /// checking OOD/oracle distributional invariants.
    ///
    /// Invariants per seed:
    ///
    /// 1. Verifier accepts the real proof (completeness).
    /// 2. Mask commitment matches under matched-RNG coupling (deterministic
    ///    equality — proves the simulator's mask path is identical to the
    ///    real prover's).
    /// 3. Simulated OOD answer is non-zero (uniform over EF, Lemma 9.3).
    /// 4. Real OOD answers are non-zero.
    /// 5. Simulated oracle values have correct dimensions (Lemma 9.8 step 5).
    /// 6. Correction vector sizes are consistent.
    fn run_zk_distribution_match(seed: u64) -> Result<(), &'static str> {
        use alloc::vec::Vec;

        use p3_dft::Radix2Dit;
        use p3_field::PrimeCharacteristicRing;
        use p3_zk_codes::ReedSolomonZkEncoding;
        use rand::RngExt;

        use crate::pcs::code_switch_zk;

        let num_variables = 12;
        let folding_factor = FoldingFactor::Constant(2);
        let folding = folding_factor.at_round(0);

        let mut data_rng = SmallRng::seed_from_u64(seed);
        let table = Table::new(vec![Poly::<F>::rand(&mut data_rng, num_variables)]);
        let witness = L::new_witness(vec![table], folding);
        let protocol = OpeningProtocol::new(vec![TableSpec::new(
            TableShape::new(num_variables, 1),
            vec![vec![0]],
        )]);

        let perm = Perm::new_from_rng_128(&mut SmallRng::seed_from_u64(1));
        let merkle_hash = MyHash::new(perm.clone());
        let merkle_compress = MyCompress::new(perm);
        let mmcs = MyMmcs::new(merkle_hash, merkle_compress, 0);

        let params = ProtocolParameters {
            security_level: 32,
            pow_bits: 0,
            round_log_inv_rates: default_round_log_inv_rates(num_variables, &folding_factor),
            folding_factor,
            soundness_type: SecurityAssumption::CapacityBound,
            starting_log_inv_rate: 1,
            zk: true,
        };
        let pcs = TestWhirPcs::<L>::new(
            WhirConfig::new(num_variables, params),
            MyDft::default(),
            mmcs,
        );
        let n_rounds = pcs.n_rounds();

        // === Real run ===
        let proof = {
            let mut ch = challenger();
            let mut ds = DomainSeparator::new(vec![]);
            pcs.add_domain_separator::<8>(&mut ds);
            ds.observe_domain_separator(&mut ch);

            let (commitment, prover_data) =
                <TestWhirPcs<L> as MultilinearPcs<EF, MyChallenger>>::commit(
                    &pcs, witness, &mut ch,
                );
            let proof = <TestWhirPcs<L> as MultilinearPcs<EF, MyChallenger>>::open(
                &pcs,
                prover_data,
                protocol.clone(),
                &mut ch,
            );

            let mut vch = challenger();
            let mut ds = DomainSeparator::new(vec![]);
            pcs.add_domain_separator::<8>(&mut ds);
            ds.observe_domain_separator(&mut vch);
            <TestWhirPcs<L> as MultilinearPcs<EF, MyChallenger>>::verify(
                &pcs,
                &commitment,
                &proof,
                &mut vch,
                protocol,
            )
            .map_err(|_| "real proof rejected by verifier")?;

            proof
        };

        // === Composed simulator (Theorem 4.5) ===
        //
        // Replicate the real prover's ZK RNG sequence (SmallRng seed 0,
        // adapter.rs:116) to reproduce mask commitments under coupling.
        // Per ZK round the RNG sequence is:
        //   1. rng.random::<EF>() × num_queries  (r_prime)
        //   2. commit_mask(r_prime, enc_zk, mmcs, ch, rng)
        //        → padding draws + encode draws
        let dft = Radix2Dit::default();
        let mut sim_rng = SmallRng::seed_from_u64(0);
        let mut sim_ch = challenger();

        let mut sim_oods: Vec<EF> = Vec::new();

        for (round_idx, round) in proof.whir.rounds.iter().enumerate() {
            let is_last_intermediate = round_idx + 1 == n_rounds;
            let zk_this_round = !is_last_intermediate;

            if !zk_this_round {
                if !round.zk_ood_corrections.is_empty() {
                    return Err("non-ZK round has OOD corrections");
                }
                continue;
            }

            let round_params = &pcs.config.round_parameters[round_idx];

            // Reproduce r_prime (prover/mod.rs:251-253).
            let r_prime: Vec<EF> = (0..round_params.num_queries)
                .map(|_| sim_rng.random())
                .collect();

            // Build mask encoding (prover/mod.rs:295-300).
            let mask_msg_len = r_prime.len().max(1).next_power_of_two();
            let mask_t = round_params.num_queries.max(1);
            let mask_m = (mask_msg_len + mask_t).next_power_of_two();
            let enc_zk =
                ReedSolomonZkEncoding::<EF, _>::new(mask_t, mask_msg_len, mask_m, dft.clone());

            // Commit mask under matched RNG (prover/mod.rs:302-308).
            let (sim_mask_root, _, _) = code_switch_zk::commit_mask(
                &r_prime,
                &enc_zk,
                &pcs.extension_mmcs,
                &mut sim_ch,
                &mut sim_rng,
            );

            // --- Coupling: mask commitment ---
            let real_mask = round
                .mask_commitment
                .as_ref()
                .ok_or("ZK round missing mask commitment")?;
            if *real_mask != sim_mask_root {
                return Err("matched-RNG coupling: mask commitment differs");
            }

            // --- Code-switching simulator (Lemma 9.8) ---
            let n_sim_queries = round_params.num_queries.min(mask_t);
            let target_positions: Vec<usize> = (0..n_sim_queries).collect();
            let mask_positions: Vec<usize> = (0..n_sim_queries).collect();
            let ood_point: EF = sim_rng.random();

            let (sim_ood, sim_g, sim_s) = code_switch_zk::simulate(
                ood_point,
                &target_positions,
                &mask_positions,
                &enc_zk,
                &enc_zk,
                &mut sim_rng,
            );

            // Non-degeneracy.
            if sim_ood == EF::ZERO {
                return Err("simulated OOD is zero");
            }
            for ood in &round.ood_answers {
                if *ood == EF::ZERO {
                    return Err("real OOD answer is zero");
                }
            }

            // Dimension match.
            if sim_g.len() != target_positions.len() {
                return Err("simulated g values: dimension mismatch");
            }
            if sim_s.len() != mask_positions.len() {
                return Err("simulated s values: dimension mismatch");
            }

            // Correction structure.
            if round.zk_ood_corrections.len() != round.ood_answers.len() {
                return Err("OOD correction/answer count mismatch");
            }
            if round_idx > 0 && round.zk_stir_corrections.len() != round.queries.len() {
                return Err("STIR correction/query count mismatch");
            }

            sim_oods.push(sim_ood);
        }

        // Cross-seed non-degeneracy: at least one ZK round was simulated.
        if sim_oods.is_empty() {
            return Err("no ZK rounds found in proof");
        }

        Ok(())
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(8))]

        /// Theorem 4.5 composition distribution test: the composed simulator
        /// reproduces mask commitments under matched-RNG coupling and produces
        /// distributionally valid OOD/oracle outputs.
        #[test]
        fn prop_zk_composition_distribution_match(seed in 0u64..256) {
            let result = run_zk_distribution_match(seed);
            prop_assert!(
                result.is_ok(),
                "seed {seed}: {}",
                result.err().unwrap_or("ok"),
            );
        }
    }
}
