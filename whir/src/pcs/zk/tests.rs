//! End-to-end and zero-knowledge tests for the HVZK-WHIR pipeline.

use alloc::vec;
use alloc::vec::Vec;

use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::DuplexChallenger;
use p3_commit::MultilinearPcs;
use p3_dft::Radix2DFTSmallBatch;
use p3_field::extension::BinomialExtensionField;
use p3_field::{BasedVectorSpace, Field, PrimeCharacteristicRing};
use p3_merkle_tree::MerkleTreeMmcs;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

use super::adapter::HidingWhirPcs;
use super::base_case::BaseCaseZkError;
use super::config::{ZkParameters, ZkWhirConfig};
use super::verifier::ZkVerifierError;
use crate::fiat_shamir::domain_separator::DomainSeparator;
use crate::parameters::{FoldingFactor, ProtocolParameters, SecurityAssumption};

type F = BabyBear;
type EF = BinomialExtensionField<F, 4>;
type Perm = Poseidon2BabyBear<16>;
type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
type MyChallenger = DuplexChallenger<F, Perm, 16, 8>;
type PackedF = <F as Field>::Packing;
type MyMmcs = MerkleTreeMmcs<PackedF, PackedF, MyHash, MyCompress, 2, 8>;
type MyDft = Radix2DFTSmallBatch<F>;
type TestZkPcs = HidingWhirPcs<EF, F, MyDft, MyMmcs, MyChallenger, SmallRng>;

fn challenger() -> MyChallenger {
    let mut rng = SmallRng::seed_from_u64(1);
    let perm = Perm::new_from_rng_128(&mut rng);
    MyChallenger::new(perm)
}

/// Fresh challenger seeded with the protocol's domain separator.
fn separated_challenger(pcs: &TestZkPcs) -> MyChallenger {
    let mut challenger = challenger();
    let mut separator = DomainSeparator::new(vec![]);
    pcs.add_domain_separator::<8>(&mut separator);
    separator.observe_domain_separator(&mut challenger);
    challenger
}

fn mmcs() -> MyMmcs {
    let perm = Perm::new_from_rng_128(&mut SmallRng::seed_from_u64(1));
    MyMmcs::new(MyHash::new(perm.clone()), MyCompress::new(perm), 0)
}

fn protocol_params(
    folding_factor: FoldingFactor,
    soundness_type: SecurityAssumption,
    pow_bits: usize,
) -> ProtocolParameters {
    ProtocolParameters {
        security_level: 32,
        pow_bits,
        round_log_inv_rates: vec![],
        folding_factor,
        soundness_type,
        starting_log_inv_rate: 2,
    }
}

fn make_pcs(
    num_variables: usize,
    folding_factor: FoldingFactor,
    soundness_type: SecurityAssumption,
    pow_bits: usize,
    seed: u64,
) -> TestZkPcs {
    let config = ZkWhirConfig::new(
        num_variables,
        protocol_params(folding_factor, soundness_type, pow_bits),
        ZkParameters {
            ell_zk: 4,
            mask_queries: 8,
            mask_log_inv_rate: 1,
        },
    )
    .unwrap();
    HidingWhirPcs::new(
        config,
        MyDft::default(),
        mmcs(),
        SmallRng::seed_from_u64(seed),
    )
}

/// Full commit / open / verify lifecycle for one parameter shape.
fn run_zk_whir(
    num_variables: usize,
    num_points: usize,
    folding_factor: FoldingFactor,
    soundness_type: SecurityAssumption,
    pow_bits: usize,
    seed: u64,
) {
    let pcs = make_pcs(
        num_variables,
        folding_factor,
        soundness_type,
        pow_bits,
        seed,
    );

    let mut rng = SmallRng::seed_from_u64(seed.wrapping_add(1));
    let witness = Poly::<F>::rand(&mut rng, num_variables);
    let points: Vec<Point<EF>> = (0..num_points)
        .map(|_| Point::rand(&mut rng, num_variables))
        .collect();

    // Prover.
    let mut prover_challenger = separated_challenger(&pcs);
    let (commitment, prover_data) = pcs.commit(witness.clone(), &mut prover_challenger);
    let proof = pcs.open(prover_data, points.clone(), &mut prover_challenger);

    // The claimed evaluations are the honest ones.
    for (point, &eval) in points.iter().zip(&proof.evals) {
        assert_eq!(witness.eval_base(point), eval);
    }

    // Verifier.
    let mut verifier_challenger = separated_challenger(&pcs);
    pcs.verify(&commitment, &proof, &mut verifier_challenger, points)
        .expect("honest HVZK-WHIR proof should verify");
}

#[test]
fn zk_whir_end_to_end_single_round() {
    // One code-switching round: 12 vars, constant folding 4
    // (initial fold + 1 switch + final batch).
    run_zk_whir(
        12,
        2,
        FoldingFactor::Constant(4),
        SecurityAssumption::CapacityBound,
        0,
        1,
    );
}

#[test]
fn zk_whir_end_to_end_no_rounds() {
    // Degenerate pipeline: a single fold batch straight into the base case.
    run_zk_whir(
        6,
        1,
        FoldingFactor::Constant(3),
        SecurityAssumption::CapacityBound,
        0,
        2,
    );
}

#[test]
fn zk_whir_end_to_end_multi_round() {
    // Two code-switching rounds with mixed folding factors and grinding.
    run_zk_whir(
        14,
        3,
        FoldingFactor::ConstantFromSecondRound(5, 3),
        SecurityAssumption::CapacityBound,
        5,
        3,
    );
}

#[test]
fn zk_whir_end_to_end_unique_decoding() {
    // Unique decoding has zero OOD samples: empty pads and OOD layers.
    run_zk_whir(
        12,
        2,
        FoldingFactor::Constant(4),
        SecurityAssumption::UniqueDecoding,
        0,
        4,
    );
}

#[test]
fn zk_whir_end_to_end_johnson_bound() {
    run_zk_whir(
        12,
        1,
        FoldingFactor::Constant(3),
        SecurityAssumption::JohnsonBound,
        5,
        5,
    );
}

#[test]
fn zk_whir_rejects_wrong_eval() {
    // A tampered claimed evaluation must be rejected.
    let pcs = make_pcs(
        12,
        FoldingFactor::Constant(4),
        SecurityAssumption::CapacityBound,
        0,
        6,
    );
    let mut rng = SmallRng::seed_from_u64(7);
    let witness = Poly::<F>::rand(&mut rng, 12);
    let points = vec![Point::<EF>::rand(&mut rng, 12)];

    let mut prover_challenger = separated_challenger(&pcs);
    let (commitment, prover_data) = pcs.commit(witness, &mut prover_challenger);
    let mut proof = pcs.open(prover_data, points.clone(), &mut prover_challenger);

    proof.evals[0] += EF::ONE;

    let mut verifier_challenger = separated_challenger(&pcs);
    let err = pcs
        .verify(&commitment, &proof, &mut verifier_challenger, points)
        .unwrap_err();
    // Diverged transcript: the verifier samples different STIR positions,
    // so the first round-0 opening fails to authenticate.
    assert_eq!(
        err,
        ZkVerifierError::MerkleVerificationFailed {
            round: 0,
            position: 106,
        },
    );
}

#[test]
fn zk_whir_rejects_wrong_arity_opening_point() {
    // Invariant: a malformed public statement errors, never panics.
    //
    //     committed arity      = 12
    //     verification point   = 3 variables
    //     -> arity mismatch error before any folding arithmetic
    let pcs = make_pcs(
        12,
        FoldingFactor::Constant(4),
        SecurityAssumption::CapacityBound,
        0,
        6,
    );
    let mut rng = SmallRng::seed_from_u64(7);
    let witness = Poly::<F>::rand(&mut rng, 12);
    let points = vec![Point::<EF>::rand(&mut rng, 12)];

    // Honest commit and open against the full-arity point.
    let mut prover_challenger = separated_challenger(&pcs);
    let (commitment, prover_data) = pcs.commit(witness, &mut prover_challenger);
    let proof = pcs.open(prover_data, points, &mut prover_challenger);

    // Verify against a short point instead.
    let short_points = vec![Point::<EF>::rand(&mut rng, 3)];
    let mut verifier_challenger = separated_challenger(&pcs);
    let err = pcs
        .verify(&commitment, &proof, &mut verifier_challenger, short_points)
        .unwrap_err();
    assert_eq!(
        err,
        ZkVerifierError::ClaimArityMismatch {
            claim: 0,
            expected: 12,
            actual: 3,
        },
    );
}

#[test]
fn zk_whir_rejects_tampered_ood_answer() {
    // A shifted private OOD answer desynchronizes the carried claim.
    let pcs = make_pcs(
        12,
        FoldingFactor::Constant(4),
        SecurityAssumption::CapacityBound,
        0,
        8,
    );
    let mut rng = SmallRng::seed_from_u64(9);
    let witness = Poly::<F>::rand(&mut rng, 12);
    let points = vec![Point::<EF>::rand(&mut rng, 12)];

    let mut prover_challenger = separated_challenger(&pcs);
    let (commitment, prover_data) = pcs.commit(witness, &mut prover_challenger);
    let mut proof = pcs.open(prover_data, points.clone(), &mut prover_challenger);

    assert!(
        !proof.rounds.is_empty() && !proof.rounds[0].ood_answers.is_empty(),
        "fixture should produce at least one private OOD answer",
    );
    proof.rounds[0].ood_answers[0] += EF::ONE;

    let mut verifier_challenger = separated_challenger(&pcs);
    let err = pcs
        .verify(&commitment, &proof, &mut verifier_challenger, points)
        .unwrap_err();
    // Diverged transcript: the verifier samples different STIR positions,
    // so the first round-0 opening fails to authenticate.
    assert_eq!(
        err,
        ZkVerifierError::MerkleVerificationFailed {
            round: 0,
            position: 18,
        },
    );
}

#[test]
fn zk_whir_rejects_tampered_base_case_reveal() {
    // A shifted blinded message breaks the joint target identity.
    let pcs = make_pcs(
        12,
        FoldingFactor::Constant(4),
        SecurityAssumption::CapacityBound,
        0,
        10,
    );
    let mut rng = SmallRng::seed_from_u64(11);
    let witness = Poly::<F>::rand(&mut rng, 12);
    let points = vec![Point::<EF>::rand(&mut rng, 12)];

    let mut prover_challenger = separated_challenger(&pcs);
    let (commitment, prover_data) = pcs.commit(witness, &mut prover_challenger);
    let mut proof = pcs.open(prover_data, points.clone(), &mut prover_challenger);

    proof.base_case.blinded_message[0] += EF::ONE;

    let mut verifier_challenger = separated_challenger(&pcs);
    let err = pcs
        .verify(&commitment, &proof, &mut verifier_challenger, points)
        .unwrap_err();
    assert_eq!(
        err,
        ZkVerifierError::BaseCase(BaseCaseZkError::TargetCheckFailed),
    );
}

/// Conditions a uniformly random witness on the public claims.
///
/// - Claims at base-field points have base-field values.
/// - Corrections on a few hypercube coefficients therefore suffice.
/// - With pivot slots `b_j`, the corrections solve the linear system
///
/// ```text
///     sum_j delta_j * eq(z_i, b_j) = v_i - g(z_i)
/// ```
///
/// - The system is solved jointly via Gaussian elimination over the
///   extension field.
/// - Base-field inputs keep the solution in the base field.
fn condition_witness(
    rng: &mut SmallRng,
    num_variables: usize,
    claims: &[(Point<EF>, EF)],
) -> Poly<F> {
    let mut garbage = Poly::<F>::rand(rng, num_variables);
    let n = claims.len();
    if n == 0 {
        return garbage;
    }

    // System matrix A[i][j] = eq(z_i, pivot_j) and right-hand side.
    let pivots: Vec<Point<EF>> = (0..n)
        .map(|j| Point::<EF>::hypercube(j, num_variables))
        .collect();
    let mut matrix: Vec<Vec<EF>> = claims
        .iter()
        .map(|(point, _)| pivots.iter().map(|pivot| pivot.eq_poly(point)).collect())
        .collect();
    let mut rhs: Vec<EF> = claims
        .iter()
        .map(|(point, value)| *value - garbage.eval_base(point))
        .collect();

    // Gaussian elimination without pivot search: random base points make the
    // leading minors nonzero with overwhelming probability.
    for col in 0..n {
        let inv = matrix[col][col].inverse();
        for entry in matrix[col].iter_mut() {
            *entry *= inv;
        }
        rhs[col] *= inv;
        for row in 0..n {
            if row != col {
                let factor = matrix[row][col];
                let pivot_row = matrix[col].clone();
                for (entry, &pivot) in matrix[row].iter_mut().zip(&pivot_row) {
                    *entry -= factor * pivot;
                }
                let sub = factor * rhs[col];
                rhs[row] -= sub;
            }
        }
    }

    for (j, delta) in rhs.iter().enumerate() {
        // Base-field points with base-field values keep corrections in F.
        let coefficients = <EF as BasedVectorSpace<F>>::as_basis_coefficients_slice(delta);
        assert!(coefficients[1..].iter().all(|c| *c == F::ZERO));
        garbage.as_mut_slice()[j] += coefficients[0];
    }
    garbage
}

#[test]
fn zk_whir_simulator_produces_accepting_transcript() {
    // Invariant (Lemma 7.3 composed via Theorem 4.5): a simulator that
    // never sees the witness produces a full accepting transcript.
    //
    //     simulator = honest pipeline on a random witness
    //                 conditioned only on the public claims
    //
    // Distributional indistinguishability of the components is pinned
    // elsewhere:
    //
    //     masked sumcheck wires  ->  p3-sumcheck simulator tests
    //     private OOD answers    ->  code_switch programmability tests
    //     one-time-pad reveals   ->  base case tests
    let num_variables = 12;
    let mut rng = SmallRng::seed_from_u64(21);

    // Real witness and its public claims at base-field points.
    let witness = Poly::<F>::rand(&mut rng, num_variables);
    let points: Vec<Point<EF>> = (0..2)
        .map(|_| {
            Point::new(
                (0..num_variables)
                    .map(|_| EF::from(rng.random::<F>()))
                    .collect(),
            )
        })
        .collect();
    let claims: Vec<(Point<EF>, EF)> = points
        .iter()
        .map(|point| (point.clone(), witness.eval_base(point)))
        .collect();

    // Simulator side: garbage witness agreeing exactly on the claims.
    let mut simulator_rng = SmallRng::seed_from_u64(22);
    let simulated_witness = condition_witness(&mut simulator_rng, num_variables, &claims);
    for (point, value) in &claims {
        assert_eq!(simulated_witness.eval_base(point), *value);
    }
    assert_ne!(
        simulated_witness.as_slice(),
        witness.as_slice(),
        "the simulated witness must differ from the real one off the claims",
    );

    // The simulated transcript verifies against the simulated commitment
    // with the real public claims.
    let pcs = make_pcs(
        num_variables,
        FoldingFactor::Constant(4),
        SecurityAssumption::CapacityBound,
        0,
        23,
    );
    let mut prover_challenger = separated_challenger(&pcs);
    let (commitment, prover_data) = pcs.commit(simulated_witness, &mut prover_challenger);
    let proof = pcs.open(prover_data, points.clone(), &mut prover_challenger);
    assert_eq!(
        proof.evals,
        claims.iter().map(|(_, value)| *value).collect::<Vec<_>>(),
        "the simulated transcript must expose exactly the public claims",
    );

    let mut verifier_challenger = separated_challenger(&pcs);
    pcs.verify(&commitment, &proof, &mut verifier_challenger, points)
        .expect("simulated HVZK transcript must verify");
}

#[test]
fn zk_whir_masks_are_witness_independent() {
    // Invariant: the hiding material is drawn from the prover RNG
    // independently of the witness.
    //
    //     matched RNG streams + different witnesses
    //         ->  RNG-only draws coincide
    //         ->  both runs verify
    //         ->  blinded reveals differ even though the claims pin
    //             the evaluations
    let num_variables = 12;
    let mut rng = SmallRng::seed_from_u64(31);
    let witness_a = Poly::<F>::rand(&mut rng, num_variables);
    let witness_b = Poly::<F>::rand(&mut rng, num_variables);
    let point = Point::new(
        (0..num_variables)
            .map(|_| EF::from(rng.random::<F>()))
            .collect::<Vec<_>>(),
    );

    let run = |witness: Poly<F>| {
        let pcs = make_pcs(
            num_variables,
            FoldingFactor::Constant(4),
            SecurityAssumption::CapacityBound,
            0,
            41,
        );
        let mut prover_challenger = separated_challenger(&pcs);
        let (commitment, prover_data) = pcs.commit(witness, &mut prover_challenger);
        let proof = pcs.open(prover_data, vec![point.clone()], &mut prover_challenger);
        let mut verifier_challenger = separated_challenger(&pcs);
        pcs.verify(
            &commitment,
            &proof,
            &mut verifier_challenger,
            vec![point.clone()],
        )
        .expect("honest proof verifies");
        proof
    };

    let proof_a = run(witness_a);
    let proof_b = run(witness_b);
    assert_ne!(
        proof_a.base_case.blinded_message, proof_b.base_case.blinded_message,
        "different witnesses produce different (uniformly padded) reveals",
    );
}
