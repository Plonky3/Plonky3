use alloc::vec;
use alloc::vec::Vec;

use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::{DuplexChallenger, FieldChallenger, GrindingChallenger};
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, HornerIter, PackedValue, PrimeCharacteristicRing, TwoAdicField};
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use p3_util::log2_strict_usize;
use proptest::prelude::*;
use rand::SeedableRng;
use rand::rngs::SmallRng;

use crate::constraints::statement::{EqStatement, SelectStatement};
use crate::constraints::{Constraint, Statements};
use crate::layout::{Layout, PrefixProver, SuffixProver, TableShape, Verifier};
use crate::strategy::VariableOrder;
use crate::test_util::{stacked_num_variables, table_point_schedule, table_specs_to_tables};
use crate::{
    OpeningBatch, OpeningEvals, OpeningProtocol, SumcheckData, SumcheckError, TableSpec,
    verify_final_sumcheck_rounds,
};

// Base field: BabyBear (a 31-bit prime field suitable for fast arithmetic).
pub(crate) type F = BabyBear;
// Extension field: degree-4 binomial extension of BabyBear, used for challenge sampling.
pub(crate) type EF = BinomialExtensionField<F, 4>;
// Poseidon2 permutation over BabyBear with width 16, used as the core hash primitive.
pub(crate) type Perm = Poseidon2BabyBear<16>;

// Fiat-Shamir challenger: duplex sponge over BabyBear with width 16 and capacity 8.
pub(crate) type MyChallenger = DuplexChallenger<F, Perm, 16, 8>;

/// Creates a fresh `DuplexChallenger` using a fixed RNG seed.
pub(crate) fn challenger() -> MyChallenger {
    // Initialize a small random number generator with a fixed seed.
    let mut rng = SmallRng::seed_from_u64(1);

    // Construct a Poseidon2 permutation instance using 128 bits of security from the RNG
    let perm = Perm::new_from_rng_128(&mut rng);

    // Create a new duplex challenger over the field `F` with this permutation
    MyChallenger::new(perm)
}

// Simulates the prover side of STIR constraint derivation for an intermediate round.
//
// In the WHIR protocol, between sumcheck folding rounds, the verifier asks for evaluations
// of the current polynomial at randomly chosen points. These evaluations become constraints
// that the next sumcheck round must satisfy. This function:
//   1. Samples `num_eqs` random multilinear points, evaluates the polynomial, and records
//      them as "equality" constraints (poly(point) == eval).
//   2. Samples `num_sels` random indices in the multiplicative subgroup, evaluates the
//      polynomial as a univariate (Horner's method), and records them as "select" constraints.
//   3. Draws a random combining coefficient alpha and bundles everything into a Constraint.
//
// The evaluations are pushed into `constraint_evals` so the verifier can read them later
// (simulating what would normally be transmitted in the proof).
pub(crate) fn make_constraint_ext<Challenger>(
    challenger: &mut Challenger,
    constraint_evals: &mut Vec<EF>,
    num_variables: usize,
    num_eqs: usize,
    num_sels: usize,
    poly: &Poly<EF>,
) -> Constraint<F, EF>
where
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    // omega is the 2^num_variables-th root of unity; powers of omega form the evaluation domain.
    let omega = F::two_adic_generator(num_variables);

    // Initialize empty eq and select statements for this round's variable count.
    let mut eq_statement = EqStatement::initialize(num_variables);
    let mut sel_statement = SelectStatement::initialize(num_variables);

    // - Sample `num_eqs` univariate challenge points.
    // - Evaluate the sumcheck polynomial on them.
    // - Collect (point, eval) pairs for use in the statement and constraint aggregation.
    (0..num_eqs).for_each(|_| {
        // Sample a univariate field element from the prover's challenger.
        let point: EF = challenger.sample_algebra_element();

        // Expand it into a `num_variables`-dimensional multilinear point.
        let point = Point::expand_from_univariate(point, num_variables);

        // Evaluate the current sumcheck polynomial at the sampled point.
        let eval = poly.eval_ext::<F>(&point);

        // Store evaluation for verifier to read later.
        constraint_evals.push(eval);

        // Add the evaluation result to the transcript for Fiat-Shamir soundness.
        challenger.observe_algebra_element(eval);

        // Add the evaluation constraint: poly(point) == eval.
        eq_statement.add_evaluated_constraint(point, eval);
    });

    // Build "select" constraints: sample indices in the multiplicative subgroup H = {omega^i},
    // evaluate the polynomial as a univariate at omega^index using Horner's method,
    // and record (omega^index, eval) pairs.
    (0..num_sels).for_each(|_| {
        // Sample a random index in [0, 2^num_variables) from the challenger.
        let index: usize = challenger.sample_bits(num_variables);

        // Compute the corresponding evaluation point in the multiplicative subgroup.
        let var = omega.exp_u64(index as u64);

        // Evaluate the polynomial as a univariate at `var` using Horner's method.
        // This treats the evaluation vector as coefficients of a univariate polynomial
        // and computes f(var) = c_0 + c_1*var + c_2*var^2 + ... by folding from the right.
        let eval: EF = poly.iter().copied().horner(var);

        // Store evaluation for verifier to read later.
        constraint_evals.push(eval);

        // Add the evaluation result to the transcript for Fiat-Shamir soundness.
        challenger.observe_algebra_element(eval);

        // Add the evaluation constraint: poly(point) == eval.
        sel_statement.add_constraint(var, eval);
    });

    // Sample a random combining coefficient alpha from the Fiat-Shamir transcript.
    // This alpha is used to take a random linear combination of all eq and sel constraints
    // into a single aggregated constraint for the next sumcheck round.
    let alpha: EF = challenger.sample_algebra_element();

    Constraint::new(
        alpha,
        num_variables,
        vec![
            Statements::Eq(eq_statement),
            Statements::Select(sel_statement),
        ],
    )
}

// Verifier-side counterpart of `make_constraint_ext`. Reconstructs the same Constraint
// that the prover built, but without access to the polynomial -- only from the
// evaluations that the prover committed to (passed in via `constraint_evals`).
//
// The key invariant is that the verifier's challenger must stay perfectly synchronized
// with the prover's challenger: both sample the same random points and observe the same
// evaluations, so their Fiat-Shamir transcripts remain identical.
pub(crate) fn read_constraint<Challenger>(
    challenger: &mut Challenger,
    constraint_evals: &[EF],
    num_variables: usize,
    num_eqs: usize,
    num_sels: usize,
) -> Constraint<F, EF>
where
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    // Initialize an empty eq statement for this round.
    let mut eq_statement = EqStatement::initialize(num_variables);

    // Reconstruct each eq constraint: sample the same random point as the prover,
    // then read the evaluation from the proof data (instead of computing it).
    for &eval in constraint_evals.iter().take(num_eqs) {
        // Sample a univariate challenge and expand to a multilinear point.
        let point =
            Point::expand_from_univariate(challenger.sample_algebra_element(), num_variables);

        // Observe the evaluation to keep the challenger synchronized (must match prover)
        challenger.observe_algebra_element(eval);

        // Add the constraint: poly(point) == eval.
        eq_statement.add_evaluated_constraint(point, eval);
    }

    // Initialize an empty select statement for this round.
    let mut sel_statement = SelectStatement::<F, EF>::initialize(num_variables);

    // Same domain generator as the prover used.
    let omega = F::two_adic_generator(num_variables);

    // Reconstruct each select constraint: sample the same random index as the prover,
    // compute the same evaluation point omega^index, then read the evaluation from the proof.
    for i in 0..num_sels {
        // Sample the same random index as the prover did.
        let index: usize = challenger.sample_bits(num_variables);
        let var = omega.exp_u64(index as u64);

        // Read the committed evaluation corresponding to this point from constraint_evals.
        // Sel evaluations are stored after eq evaluations.
        let eval = constraint_evals[num_eqs + i];

        // Observe the evaluation to keep the challenger synchronized (must match prover)
        challenger.observe_algebra_element(eval);

        // Add the constraint: poly(point) == eval.
        sel_statement.add_constraint(var, eval);
    }

    // Sample the same combining coefficient alpha as the prover and assemble the constraint.
    Constraint::new(
        challenger.sample_algebra_element(),
        num_variables,
        vec![
            Statements::Eq(eq_statement),
            Statements::Select(sel_statement),
        ],
    )
}

pub(crate) fn table_specs_strategy() -> impl Strategy<Value = Vec<TableSpec>> {
    (1usize..=5, 1usize..=5).prop_flat_map(|(num_points, num_tables)| {
        proptest::collection::vec(
            (1usize..=12, 1usize..=5).prop_flat_map(move |(num_variables, width)| {
                proptest::collection::vec(poly_subset_strategy(width), num_points - 1).prop_map(
                    move |extra_points| {
                        let extra_points = extra_points
                            .into_iter()
                            .map(|current| OpeningBatch::new(current, Vec::new()))
                            .collect();
                        TableSpec::new(
                            TableShape::new(num_variables, width),
                            table_point_schedule(width, extra_points),
                        )
                    },
                )
            }),
            num_tables,
        )
    })
}

fn poly_subset_strategy(width: usize) -> impl Strategy<Value = Vec<usize>> {
    proptest::collection::vec(any::<bool>(), width).prop_map(|bits| {
        let polys = bits
            .into_iter()
            .enumerate()
            .filter_map(|(poly_idx, selected)| selected.then_some(poly_idx))
            .collect::<Vec<_>>();
        if polys.is_empty() { vec![0] } else { polys }
    })
}

#[allow(clippy::too_many_lines)]
fn run_multi_table_sumcheck_test<L>(specs: &[TableSpec])
where
    L: Layout<F, EF>,
{
    const FOLDING: usize = 4;
    const NUM_EQS: usize = 5;
    const NUM_SELS: usize = 10;

    // Round schedule for a constant folding factor `FOLDING`.
    const MAX_NUM_VARIABLES_TO_SEND_COEFFS: usize = 6;
    let num_variables = stacked_num_variables(specs, FOLDING);
    assert!(
        FOLDING != 0 && FOLDING <= num_variables,
        "invalid folding factor"
    );
    let (num_rounds, final_rounds) = if num_variables <= MAX_NUM_VARIABLES_TO_SEND_COEFFS {
        (0, num_variables - FOLDING)
    } else {
        let rounds = (num_variables - MAX_NUM_VARIABLES_TO_SEND_COEFFS).div_ceil(FOLDING);
        (rounds - 1, num_variables - rounds * FOLDING)
    };

    let protocol = OpeningProtocol::new(specs.to_vec()).pad_to_min_num_variables(FOLDING);
    let witness = L::new_witness(table_specs_to_tables(specs), FOLDING);

    let challenger = challenger();
    let mut prover_challenger = challenger.clone();
    let mut proof = vec![SumcheckData::<F, EF>::default(); num_rounds + 2];
    let mut all_constraint_evals: Vec<Vec<EF>> = Vec::new();

    // Snapshot the stacked polynomial before the witness is consumed.
    let stacked_poly = witness.poly().clone();
    let mut layout = L::from_witness(witness);
    let strategy = L::strategy();

    let opening_evals: Vec<OpeningEvals<EF>> = protocol
        .iter_openings()
        .map(|(table_idx, batch)| layout.eval(table_idx, batch, &mut prover_challenger))
        .collect();

    let (mut sumcheck, mut prover_randomness) =
        layout.into_sumcheck(proof.first_mut().unwrap(), 0, &mut prover_challenger);
    let mut num_variables_inter = num_variables - FOLDING;

    for sumcheck_data in proof.iter_mut().take(num_rounds + 1).skip(1) {
        let mut constraint_evals = Vec::new();
        let constraint = make_constraint_ext(
            &mut prover_challenger,
            &mut constraint_evals,
            num_variables_inter,
            NUM_EQS,
            NUM_SELS,
            &sumcheck.evals(),
        );
        all_constraint_evals.push(constraint_evals);

        let folding = FOLDING;
        prover_randomness.extend(&sumcheck.compute_sumcheck_polynomials(
            sumcheck_data,
            &mut prover_challenger,
            folding,
            0,
            Some(constraint),
        ));
        num_variables_inter -= folding;
        assert_eq!(sumcheck.num_variables(), num_variables_inter);
    }

    assert_eq!(num_variables_inter, final_rounds);
    prover_randomness.extend(&sumcheck.compute_sumcheck_polynomials(
        proof.last_mut().unwrap(),
        &mut prover_challenger,
        final_rounds,
        0,
        None,
    ));

    let final_folded_value = sumcheck.evals().as_constant().unwrap();
    assert_eq!(sumcheck.num_variables(), 0);
    let expected_final_eval = match strategy.variable_order {
        VariableOrder::Prefix => stacked_poly.eval_base(&prover_randomness),
        VariableOrder::Suffix => stacked_poly.eval_base(&prover_randomness.reversed()),
    };
    assert_eq!(expected_final_eval, final_folded_value);
    prover_challenger.observe_algebra_element(final_folded_value);

    let mut verifier_challenger = challenger;
    let mut sum = EF::ZERO;
    let mut verifier_randomness = Point::new(vec![]);
    let mut constraints = vec![];
    let mut num_variables_inter = num_variables;

    {
        let mut layout_verifier = Verifier::<F, EF>::new(&protocol.table_shapes(), strategy);
        for ((table_idx, batch), evals) in protocol.iter_openings().zip(&opening_evals) {
            layout_verifier
                .add_claim(table_idx, batch, evals, &mut verifier_challenger)
                .unwrap();
        }
        let alpha = verifier_challenger.sample_algebra_element();
        let constraint = layout_verifier.constraint(alpha);
        constraint.combine_evals(&mut sum);
        assert_eq!(sum, layout_verifier.sum(alpha));
        constraints.push(constraint);

        verifier_randomness.extend(
            &proof[0]
                .verify_rounds(&mut verifier_challenger, &mut sum, FOLDING, 0)
                .unwrap(),
        );
        num_variables_inter -= FOLDING;
    }

    for round in 1..=num_rounds {
        let constraint = read_constraint(
            &mut verifier_challenger,
            &all_constraint_evals[round - 1],
            num_variables_inter,
            NUM_EQS,
            NUM_SELS,
        );
        constraint.combine_evals(&mut sum);
        constraints.push(constraint);

        verifier_randomness.extend(
            &proof[round]
                .verify_rounds(&mut verifier_challenger, &mut sum, FOLDING, 0)
                .unwrap(),
        );
        num_variables_inter -= FOLDING;
    }

    verifier_randomness.extend(
        &proof
            .last()
            .unwrap()
            .verify_rounds(&mut verifier_challenger, &mut sum, num_variables_inter, 0)
            .unwrap(),
    );

    assert_eq!(prover_randomness, verifier_randomness);
    let weights = strategy
        .variable_order
        .eval_constraints_poly(&constraints, &verifier_randomness);
    assert_eq!(sum, final_folded_value * weights);
}

#[test]
fn test_single_sumcheck() {
    let specs = [TableSpec::new(
        TableShape::new(20, 1),
        vec![OpeningBatch::new(vec![0], Vec::new())],
    )];

    run_multi_table_sumcheck_test::<PrefixProver<F, EF>>(&specs);
    run_multi_table_sumcheck_test::<SuffixProver<F, EF>>(&specs);
}

#[test]
fn test_single_sumcheck_mixed_current_next_large() {
    // The same column is opened both directly and through the successor view.
    // This exercises the suffix prover's residual-reuse path across the two sides.
    //
    // Twenty variables keep the residual weight tables far above the parallel threshold.
    // The successor-weight accumulators then run their parallel branches, not the serial fallback.
    let specs = [TableSpec::new(
        TableShape::new(20, 1),
        vec![OpeningBatch::new(vec![0], vec![0])],
    )];

    run_multi_table_sumcheck_test::<PrefixProver<F, EF>>(&specs);
    run_multi_table_sumcheck_test::<SuffixProver<F, EF>>(&specs);
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 16, ..ProptestConfig::default() })]

    #[test]
    fn multi_table_layout_sumcheck_prop(specs in table_specs_strategy()) {
        const FOLDING: usize = 4;

        let packing_log = log2_strict_usize(<F as Field>::Packing::WIDTH);
        prop_assume!(stacked_num_variables(&specs, FOLDING) >= FOLDING + packing_log);

        run_multi_table_sumcheck_test::<PrefixProver<F, EF>>(&specs);
        run_multi_table_sumcheck_test::<SuffixProver<F, EF>>(&specs);
    }
}

#[test]
fn test_zero_rounds_returns_empty_point() {
    // Invariant: 0 rounds short-circuits — proof data is never inspected.

    // Case 1: no data.
    let mut chal = challenger();
    let mut sum = EF::ZERO;
    let point = verify_final_sumcheck_rounds::<F, EF, _>(None, &mut chal, &mut sum, 0, 0)
        .expect("0 rounds + None must succeed");
    assert!(point.as_slice().is_empty());

    // Case 2: data supplied but ignored.
    let data = SumcheckData::<F, EF> {
        polynomial_evaluations: vec![[EF::ONE, EF::ONE]],
        pow_witnesses: vec![],
    };
    let mut chal = challenger();
    let mut sum = EF::ZERO;
    let point = verify_final_sumcheck_rounds(Some(&data), &mut chal, &mut sum, 0, 0)
        .expect("0 rounds + Some must succeed");
    assert!(point.as_slice().is_empty());
}

#[test]
fn test_missing_sumcheck_data() {
    // Invariant: rounds > 0 with no data must error out.
    let mut chal = challenger();
    let mut sum = EF::ZERO;
    let rounds = 3;

    let err = verify_final_sumcheck_rounds::<F, EF, _>(None, &mut chal, &mut sum, rounds, 0)
        .expect_err("None + rounds > 0 must error");

    match err {
        // Inner field must echo the requested round count.
        SumcheckError::MissingSumcheckData { expected_rounds } => {
            assert_eq!(expected_rounds, rounds);
        }
        other => panic!("expected MissingSumcheckData, got: {other}"),
    }
}

#[test]
fn test_round_count_mismatch() {
    // Invariant: evaluation count must equal requested rounds.
    //
    //     evaluations: 2     requested: 5     -> expected=5, actual=2
    let mut chal = challenger();
    let mut sum = EF::ZERO;
    let expected_rounds = 5;
    let actual_rounds = 2;
    let data = SumcheckData::<F, EF> {
        // Values are unread; the length check fires first.
        polynomial_evaluations: vec![[EF::ZERO, EF::ZERO]; actual_rounds],
        pow_witnesses: vec![],
    };

    let err = verify_final_sumcheck_rounds(Some(&data), &mut chal, &mut sum, expected_rounds, 0)
        .expect_err("length mismatch must error");

    match err {
        SumcheckError::RoundCountMismatch { expected, actual } => {
            assert_eq!(expected, expected_rounds);
            assert_eq!(actual, actual_rounds);
        }
        other => panic!("expected RoundCountMismatch, got: {other}"),
    }
}

#[test]
fn test_verify_rounds_rejects_wrong_round_count() {
    // Invariant: verify_rounds binds the round count itself, before observing or folding.
    //
    //     evaluations: 3     expected: 4     -> expected=4, actual=3
    let mut chal = challenger();
    let mut sum = EF::ZERO;
    let expected_rounds = 4;
    let actual_rounds = 3;
    let data = SumcheckData::<F, EF> {
        // Values are unread; the count check fires first.
        polynomial_evaluations: vec![[EF::ZERO, EF::ZERO]; actual_rounds],
        pow_witnesses: vec![],
    };

    let err = data
        .verify_rounds(&mut chal, &mut sum, expected_rounds, 0)
        .expect_err("wrong round count must error");

    match err {
        SumcheckError::RoundCountMismatch { expected, actual } => {
            assert_eq!(expected, expected_rounds);
            assert_eq!(actual, actual_rounds);
        }
        other => panic!("expected RoundCountMismatch, got: {other}"),
    }
}

#[test]
fn test_pow_witness_count_mismatch() {
    // Invariant: when PoW is enabled, witness count must match round count.
    let mut chal = challenger();
    let mut sum = EF::ONE;
    let expected = 2;
    let actual = 1;
    let data = SumcheckData::<F, EF> {
        polynomial_evaluations: vec![[EF::ZERO, EF::ZERO]; expected],
        pow_witnesses: vec![F::ZERO; actual],
    };

    let err = data
        .verify_rounds(&mut chal, &mut sum, expected, 20)
        .expect_err("witness-count mismatch must error before indexing");

    match err {
        SumcheckError::PowWitnessCountMismatch {
            expected: got_expected,
            actual: got_actual,
        } => {
            assert_eq!(got_expected, expected);
            assert_eq!(got_actual, actual);
        }
        other => panic!("expected PowWitnessCountMismatch, got: {other}"),
    }
}

#[test]
fn test_invalid_pow_witness() {
    // Invariant: a tampered PoW witness must fail the grinding check.
    let mut chal = challenger();
    let mut sum = EF::ONE;
    let pow_bits = 20;
    let data = SumcheckData::<F, EF> {
        // Polynomial coefficients are arbitrary; the PoW check fires first.
        polynomial_evaluations: vec![[EF::ZERO, EF::ZERO]],
        // Mutation: zero replaces the honest ground witness.
        pow_witnesses: vec![F::ZERO],
    };

    let err = data
        .verify_rounds(&mut chal, &mut sum, data.num_rounds(), pow_bits)
        .expect_err("zeroed witness must fail");

    assert!(matches!(err, SumcheckError::InvalidPowWitness));
}
