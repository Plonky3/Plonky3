use alloc::vec;

use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::DuplexChallenger;
use p3_dft::Radix2DFTSmallBatch;
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, PrimeCharacteristicRing};
use p3_matrix::dense::DenseMatrix;
use p3_merkle_tree::{MerkleTree, MerkleTreeMmcs};
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use rand::SeedableRng;
use rand::rngs::SmallRng;

use crate::constraints::statement::initial::InitialStatement;
use crate::fiat_shamir::domain_separator::DomainSeparator;
use crate::parameters::{
    FoldingFactor, ProtocolParameters, SecurityAssumption, SumcheckStrategy, WhirConfig,
};
use crate::pcs::committer::writer::CommitmentWriter;
use crate::pcs::committer::{ProverData, ProverDataExt};
use crate::pcs::proof::WhirProof;
use crate::pcs::prover::round_state::RoundState;

type F = BabyBear;
type EF = BinomialExtensionField<F, 4>;

type Perm = Poseidon2BabyBear<16>;

type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
type MyChallenger = DuplexChallenger<F, Perm, 16, 8>;

const DIGEST_ELEMS: usize = 8;

type PackedF = <F as Field>::Packing;
type MyMmcs = MerkleTreeMmcs<PackedF, PackedF, MyHash, MyCompress, 2, DIGEST_ELEMS>;

fn make_test_config(
    num_variables: usize,
    folding_factor: usize,
    pow_bits: usize,
) -> WhirConfig<EF, F, MyMmcs, MyChallenger> {
    let mut rng = SmallRng::seed_from_u64(1);
    let perm = Perm::new_from_rng_128(&mut rng);

    let merkle_hash = MyHash::new(perm.clone());
    let merkle_compress = MyCompress::new(perm);
    let mmcs = MyMmcs::new(merkle_hash, merkle_compress, 0);

    let protocol_params = ProtocolParameters {
        security_level: 80,
        pow_bits,
        rs_domain_initial_reduction_factor: 1,
        folding_factor: FoldingFactor::Constant(folding_factor),
        mmcs,
        soundness_type: SecurityAssumption::CapacityBound,
        starting_log_inv_rate: 1,
    };

    WhirConfig::new(num_variables, protocol_params)
}

#[allow(clippy::type_complexity)]
fn setup_domain_and_commitment(
    params: &WhirConfig<EF, F, MyMmcs, MyChallenger>,
    poly: Poly<F>,
) -> (
    WhirProof<F, EF, MyMmcs>,
    MyChallenger,
    MerkleTree<F, F, DenseMatrix<F>, 2, DIGEST_ELEMS>,
) {
    let protocol_params = ProtocolParameters {
        security_level: params.security_level,
        pow_bits: params.starting_folding_pow_bits,
        folding_factor: params.folding_factor,
        mmcs: params.mmcs.clone(),
        soundness_type: params.soundness_type,
        starting_log_inv_rate: params.starting_log_inv_rate,
        rs_domain_initial_reduction_factor: 1,
    };

    let whir_proof = WhirProof::from_protocol_parameters(&protocol_params, poly.num_variables());

    let mut domsep = DomainSeparator::new(vec![]);
    domsep.commit_statement::<_, _, DIGEST_ELEMS>(params);
    domsep.add_whir_proof::<_, _, DIGEST_ELEMS>(params);

    let mut rng = SmallRng::seed_from_u64(1);
    let mut prover_challenger = MyChallenger::new(Perm::new_from_rng_128(&mut rng));
    domsep.observe_domain_separator(&mut prover_challenger);

    let committer = CommitmentWriter::new(params);
    let mut initial_statement = InitialStatement::new(poly, 0, SumcheckStrategy::default());
    let mut proof =
        WhirProof::from_protocol_parameters(&protocol_params, initial_statement.num_variables());

    let prover_data = committer
        .commit(
            &Radix2DFTSmallBatch::<F>::default(),
            &mut proof,
            &mut prover_challenger,
            &mut initial_statement,
        )
        .unwrap();

    (whir_proof, prover_challenger, prover_data)
}

#[test]
fn test_no_initial_statement_no_sumcheck() {
    let num_variables = 2;
    let config = make_test_config(num_variables, 2, 0);
    let folding0 = config.folding_factor.at_round(0);

    let poly = Poly::new(vec![F::from_u64(3); 1 << num_variables]);

    let (mut proof, mut challenger, prover_data) =
        setup_domain_and_commitment(&config, poly.clone());

    let statement = InitialStatement::new(poly, folding0, SumcheckStrategy::default());

    let state = RoundState::<
        _,
        _,
        ProverData<F, 2, DIGEST_ELEMS>,
        ProverDataExt<F, EF, 2, DIGEST_ELEMS>,
    >::initialize_first_round_state(
        &mut proof.initial_sumcheck,
        &mut challenger,
        &statement,
        prover_data,
        config.folding_factor.at_round(0),
        0,
    )
    .unwrap();

    // Folding factor 2 => 2 sampled randomness values.
    assert_eq!(state.folding_randomness.num_variables(), 2);
    // First round: no extension field Merkle data yet.
    assert!(state.merkle_prover_data.is_none());
}

#[test]
fn test_initial_statement_with_folding_factor_3() {
    let num_variables = 3;
    let config = make_test_config(num_variables, 3, 0);
    let folding0 = config.folding_factor.at_round(0);

    // f(x0, x1, x2) = 1 + 2*x2 + 3*x1 + 4*x1*x2 + 5*x0 + 6*x0*x2 + 7*x0*x1 + 8*x0*x1*x2
    let e1 = F::from_u64(1);
    let e2 = F::from_u64(2);
    let e3 = F::from_u64(3);
    let e4 = F::from_u64(4);
    let e5 = F::from_u64(5);
    let e6 = F::from_u64(6);
    let e7 = F::from_u64(7);
    let e8 = F::from_u64(8);

    let poly = Poly::new(vec![
        e1,
        e1 + e2,
        e1 + e3,
        e1 + e2 + e3 + e4,
        e1 + e5,
        e1 + e2 + e5 + e6,
        e1 + e3 + e5 + e7,
        e1 + e2 + e3 + e4 + e5 + e6 + e7 + e8,
    ]);

    let f = |x0: EF, x1: EF, x2: EF| {
        x2 * e2
            + x1 * e3
            + x1 * x2 * e4
            + x0 * e5
            + x0 * x2 * e6
            + x0 * x1 * e7
            + x0 * x1 * x2 * e8
            + e1
    };

    let (mut proof, mut challenger_rf, prover_data) =
        setup_domain_and_commitment(&config, poly.clone());

    let mut statement = InitialStatement::new(poly, folding0, SumcheckStrategy::default());
    let _ = statement.evaluate(&Point::new(vec![EF::ONE, EF::ONE, EF::ONE]));

    let state = RoundState::<
        _,
        _,
        ProverData<F, 2, DIGEST_ELEMS>,
        ProverDataExt<F, EF, 2, DIGEST_ELEMS>,
    >::initialize_first_round_state(
        &mut proof.initial_sumcheck,
        &mut challenger_rf,
        &statement,
        prover_data,
        config.folding_factor.at_round(0),
        0,
    )
    .unwrap();

    let sumcheck = &state.sumcheck_prover;
    let sumcheck_randomness = state.folding_randomness.clone();

    // All 3 variables folded in one round => 0 variables remain.
    assert_eq!(sumcheck.num_variables(), 0);

    // The single remaining evaluation must match f at the folding point.
    let eval_at_point = sumcheck.evals().as_slice()[0];
    let expected = f(
        sumcheck_randomness[0],
        sumcheck_randomness[1],
        sumcheck_randomness[2],
    );
    assert_eq!(eval_at_point, expected);

    assert!(state.merkle_prover_data.is_none());
}

#[test]
fn test_zero_poly_multiple_constraints() {
    let num_variables = 3;
    let config = make_test_config(num_variables, 1, 0);
    let folding0 = config.folding_factor.at_round(0);

    let poly = Poly::new(vec![F::ZERO; 1 << num_variables]);

    let (mut proof, mut challenger_rf, prover_data) =
        setup_domain_and_commitment(&config, poly.clone());

    let mut statement = InitialStatement::new(poly, folding0, SumcheckStrategy::default());

    // f(x) = 0 for all x in {0,1}^3.
    for i in 0..1 << num_variables {
        let point = (0..num_variables)
            .map(|b| EF::from_u64(((i >> b) & 1) as u64))
            .collect();
        let eval = statement.evaluate(&Point::new(point));
        assert_eq!(eval, EF::ZERO);
    }

    let state = RoundState::<
        _,
        _,
        ProverData<F, 2, DIGEST_ELEMS>,
        ProverDataExt<F, EF, 2, DIGEST_ELEMS>,
    >::initialize_first_round_state(
        &mut proof.initial_sumcheck,
        &mut challenger_rf,
        &statement,
        prover_data,
        folding0,
        0,
    )
    .unwrap();

    let sumcheck = &state.sumcheck_prover;

    for f in &sumcheck.evals() {
        assert_eq!(*f, EF::ZERO);
    }
    assert_eq!(state.folding_randomness.num_variables(), 1);
    assert!(state.merkle_prover_data.is_none());
}

#[test]
fn test_initialize_round_state_with_initial_statement() {
    let num_variables = 3;
    let pow_bits = 4;
    let config = make_test_config(num_variables, 1, pow_bits);
    let folding0 = config.folding_factor.at_round(0);

    let e1 = F::from_u64(1);
    let e2 = F::from_u64(2);
    let e3 = F::from_u64(3);
    let e4 = F::from_u64(4);
    let e5 = F::from_u64(5);
    let e6 = F::from_u64(6);
    let e7 = F::from_u64(7);
    let e8 = F::from_u64(8);

    let poly = Poly::new(vec![
        e1,
        e1 + e2,
        e1 + e3,
        e1 + e2 + e3 + e4,
        e1 + e5,
        e1 + e2 + e5 + e6,
        e1 + e3 + e5 + e7,
        e1 + e2 + e3 + e4 + e5 + e6 + e7 + e8,
    ]);

    let f = |x0: EF, x1: EF, x2: EF| {
        x2 * e2
            + x1 * e3
            + x1 * x2 * e4
            + x0 * e5
            + x0 * x2 * e6
            + x0 * x1 * e7
            + x0 * x1 * x2 * e8
            + e1
    };

    let (mut proof, mut challenger_rf, prover_data) =
        setup_domain_and_commitment(&config, poly.clone());

    let mut statement = InitialStatement::new(poly, folding0, SumcheckStrategy::default());
    let _ = statement.evaluate(&Point::new(vec![EF::ONE, EF::ZERO, EF::ONE]));

    let state = RoundState::<
        _,
        _,
        ProverData<F, 2, DIGEST_ELEMS>,
        ProverDataExt<F, EF, 2, DIGEST_ELEMS>,
    >::initialize_first_round_state(
        &mut proof.initial_sumcheck,
        &mut challenger_rf,
        &statement,
        prover_data,
        folding0,
        pow_bits,
    )
    .expect("RoundState initialization failed");

    let sumcheck = &state.sumcheck_prover;
    let sumcheck_randomness = &state.folding_randomness;

    // Verify polynomial evaluation at an arbitrary point.
    let evals_f = &sumcheck.evals();
    assert_eq!(
        evals_f.eval_ext::<F>(&Point::new(vec![EF::from_u64(32636), EF::from_u64(9876)])),
        f(
            sumcheck_randomness[0],
            EF::from_u64(32636),
            EF::from_u64(9876),
        )
    );

    assert!(state.merkle_prover_data.is_none());
    assert_eq!(
        state.folding_randomness,
        Point::new(vec![sumcheck_randomness[0]])
    );
}
