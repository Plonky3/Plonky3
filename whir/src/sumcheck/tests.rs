use alloc::vec;
use alloc::vec::Vec;

use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::{DuplexChallenger, FieldChallenger, GrindingChallenger};
use p3_field::extension::BinomialExtensionField;
use p3_field::{PrimeCharacteristicRing, TwoAdicField};
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

use crate::constraints::Constraint;
use crate::constraints::evaluator::ConstraintPolyEvaluator;
use crate::constraints::statement::initial::InitialStatement;
use crate::constraints::statement::{EqStatement, SelectStatement};
use crate::fiat_shamir::domain_separator::DomainSeparator;
use crate::parameters::{FoldingFactor, SumcheckStrategy};
use crate::sumcheck::prover::SumcheckProver;
use crate::sumcheck::{SumcheckData, verify_final_sumcheck_rounds};

// Base field: BabyBear (a 31-bit prime field suitable for fast arithmetic).
type F = BabyBear;
// Extension field: degree-4 binomial extension of BabyBear, used for challenge sampling.
type EF = BinomialExtensionField<F, 4>;
// Poseidon2 permutation over BabyBear with width 16, used as the core hash primitive.
type Perm = Poseidon2BabyBear<16>;

// Fiat-Shamir challenger: duplex sponge over BabyBear with width 16 and capacity 8.
type MyChallenger = DuplexChallenger<F, Perm, 16, 8>;

/// Creates a fresh `DomainSeparator` and `DuplexChallenger` using a fixed RNG seed.
fn domainsep_and_challenger() -> (DomainSeparator<EF, F>, MyChallenger) {
    // Initialize a small random number generator with a fixed seed.
    let mut rng = SmallRng::seed_from_u64(1);

    // Construct a Poseidon2 permutation instance using 128 bits of security from the RNG
    let perm = Perm::new_from_rng_128(&mut rng);

    // Create a new duplex challenger over the field `F` with this permutation
    let challenger = MyChallenger::new(perm);

    // Return a fresh (empty) domain separator and the challenger
    (DomainSeparator::new(vec![]), challenger)
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
fn make_constraint_ext<Challenger>(
    challenger: &mut Challenger,
    constraint_evals: &mut Vec<EF>,
    num_vars: usize,
    num_eqs: usize,
    num_sels: usize,
    poly: &Poly<EF>,
) -> Constraint<F, EF>
where
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    // omega is the 2^num_vars-th root of unity; powers of omega form the evaluation domain.
    let omega = F::two_adic_generator(num_vars);

    // Initialize empty eq and select statements for this round's variable count.
    let mut eq_statement = EqStatement::initialize(num_vars);
    let mut sel_statement = SelectStatement::initialize(num_vars);

    // - Sample `num_eqs` univariate challenge points.
    // - Evaluate the sumcheck polynomial on them.
    // - Collect (point, eval) pairs for use in the statement and constraint aggregation.
    (0..num_eqs).for_each(|_| {
        // Sample a univariate field element from the prover's challenger.
        let point: EF = challenger.sample_algebra_element();

        // Expand it into a `num_vars`-dimensional multilinear point.
        let point = Point::expand_from_univariate(point, num_vars);

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
        // Sample a random index in [0, 2^num_vars) from the challenger.
        let index: usize = challenger.sample_bits(num_vars);

        // Compute the corresponding evaluation point in the multiplicative subgroup.
        let var = omega.exp_u64(index as u64);

        // Evaluate the polynomial as a univariate at `var` using Horner's method.
        // This treats the evaluation vector as coefficients of a univariate polynomial
        // and computes f(var) = c_0 + c_1*var + c_2*var^2 + ... by folding from the right.
        let eval = poly
            .iter()
            .rfold(EF::ZERO, |result, &coeff| result * var + coeff);

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

    Constraint::new(alpha, eq_statement, sel_statement)
}

// Verifier-side counterpart of `make_constraint_ext`. Reconstructs the same Constraint
// that the prover built, but without access to the polynomial -- only from the
// evaluations that the prover committed to (passed in via `constraint_evals`).
//
// The key invariant is that the verifier's challenger must stay perfectly synchronized
// with the prover's challenger: both sample the same random points and observe the same
// evaluations, so their Fiat-Shamir transcripts remain identical.
fn read_constraint<Challenger>(
    challenger: &mut Challenger,
    constraint_evals: &[EF],
    num_vars: usize,
    num_eqs: usize,
    num_sels: usize,
) -> Constraint<F, EF>
where
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    // Initialize an empty eq statement for this round.
    let mut eq_statement = EqStatement::initialize(num_vars);

    // Reconstruct each eq constraint: sample the same random point as the prover,
    // then read the evaluation from the proof data (instead of computing it).
    for &eval in constraint_evals.iter().take(num_eqs) {
        // Sample a univariate challenge and expand to a multilinear point.
        let point = Point::expand_from_univariate(challenger.sample_algebra_element(), num_vars);

        // Observe the evaluation to keep the challenger synchronized (must match prover)
        challenger.observe_algebra_element(eval);

        // Add the constraint: poly(point) == eval.
        eq_statement.add_evaluated_constraint(point, eval);
    }

    // Initialize an empty select statement for this round.
    let mut sel_statement = SelectStatement::<F, EF>::initialize(num_vars);

    // Same domain generator as the prover used.
    let omega = F::two_adic_generator(num_vars);

    // Reconstruct each select constraint: sample the same random index as the prover,
    // compute the same evaluation point omega^index, then read the evaluation from the proof.
    for i in 0..num_sels {
        // Sample the same random index as the prover did.
        let index: usize = challenger.sample_bits(num_vars);
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
        eq_statement,
        sel_statement,
    )
}

/// Runs an end-to-end prover-verifier test for the `SumcheckSingle` protocol with nested folding.
/// This test:
/// - Initializes a random multilinear polynomial over `F`.
/// - Runs the prover through several rounds of sumcheck folding.
/// - Verifies the transcript using the verifier.
/// - Checks that all reconstructed constraints match the original ones.
/// - Verifies that the final sum satisfies:
///   \begin{equation}
///   \text{sum} = f(r) \cdot \text{eq}(z, r)
///   \end{equation}
///
/// # Panics
/// Panics if:
/// - Any intermediate or final round does not produce expected sizes.
/// - Any constraint mismatches.
/// - The verifier-side evaluation differs from the expected one.
///
/// # Arguments
/// - `num_vars`: Number of variables of the initial polynomial.
/// - `folding_factors`: List of how many variables to fold per round.
/// - `num_eqs`: Number of equality statements to apply at each stage.
/// - `num_sels`: Number of select statements to apply at each stage.
#[allow(clippy::too_many_lines)]
fn run_sumcheck_test(
    num_vars: usize,
    folding_factor: FoldingFactor,
    num_eqs: &[usize],
    num_sels: &[usize],
    strategy: SumcheckStrategy,
) -> Point<EF> {
    // Compute how many intermediate folding rounds there are, plus the size of the final round.
    // For example, with num_vars=6 and folding_factor=2: num_rounds=2, final_rounds=2.
    let (num_rounds, final_rounds) = folding_factor.compute_number_of_rounds(num_vars);

    // We need num_rounds+1 eq constraint counts (one for the initial round, one per intermediate),
    // and num_rounds sel constraint counts (one per intermediate round; the initial has no sels).
    assert_eq!(num_eqs.len(), num_rounds + 1);
    assert_eq!(num_sels.len(), num_rounds);
    folding_factor.check_validity(num_vars).unwrap();

    // Initialize a random multilinear polynomial with 2^num_vars evaluations.
    let mut rng = SmallRng::seed_from_u64(1);
    let poly = Poly::new((0..1 << num_vars).map(|_| rng.random()).collect());

    // ==================== PROVER PHASE ====================
    // The prover has access to the full polynomial and produces a proof transcript.
    // The challenger is cloned so that prover and verifier start from the same state
    // but evolve independently (they must stay synchronized via Fiat-Shamir).
    let (domsep, challenger) = domainsep_and_challenger();
    let mut prover_challenger = challenger.clone();

    // Crate empty proof for each round
    let mut proof = vec![SumcheckData::<F, EF>::default(); num_rounds + 2];

    // Absorb the domain separator into the prover's transcript.
    domsep.observe_domain_separator(&mut prover_challenger);

    // Store constraint evaluations for each round (prover writes, verifier reads)
    // TODO: read from proof in verifier side
    let mut all_constraint_evals: Vec<Vec<EF>> = Vec::new();

    // Build the initial statement from the polynomial. The initial statement wraps
    // the raw evaluations and supports constraint evaluation before the first fold.
    let folding0 = folding_factor.at_round(0);
    let mut initial_statement = InitialStatement::new(poly.clone(), folding0, strategy);

    // Sample eq constraints for the initial round: for each constraint, draw a random
    // univariate challenge, expand it to a multilinear point, evaluate the polynomial
    // there, and commit the evaluation to the transcript.
    let constaint_evals = (0..num_eqs[0])
        .map(|_| {
            let point: EF = prover_challenger.sample_algebra_element();
            let point = Point::expand_from_univariate(point, num_vars);
            let eval = initial_statement.evaluate(&point);
            prover_challenger.observe_algebra_element(eval);
            eval
        })
        .collect();
    all_constraint_evals.push(constaint_evals);

    // ROUND 0: Initialize the sumcheck prover from the base evaluations.
    // This performs the first folding round: it computes the sumcheck polynomials
    // h(X) = c0 + c1*X + c2*X^2 for each variable being folded, writes them into
    // the proof, and returns the partially folded state along with the verifier's
    // random challenges (prover_randomness) accumulated so far.

    let (mut sumcheck, mut prover_randomness) = SumcheckProver::from_base_evals(
        proof.first_mut().unwrap(),
        &mut prover_challenger,
        folding0,
        0,
        &initial_statement,
    );

    // Track how many variables remain to fold
    let mut num_vars_inter = num_vars - folding0;

    // INTERMEDIATE ROUNDS (rounds 1..num_rounds):
    // Each round simulates the STIR interaction: the verifier asks for evaluations at
    // random points (eq + select constraints), the prover responds, and then a new
    // sumcheck folding step reduces the number of variables by `folding`.
    for (round, (&num_eq_points, &num_sel_points)) in
        num_eqs.iter().skip(1).zip(num_sels.iter()).enumerate()
    {
        // Adjust index since enumerate starts at 0 but we skip the initial round.
        let round = round + 1;
        let folding = folding_factor.at_round(round);

        // Build STIR constraints for this round: sample random evaluation points,
        // compute polynomial evaluations, and assemble them into a Constraint.
        let mut constraint_evals: Vec<EF> = Vec::new();
        let constraint = make_constraint_ext(
            &mut prover_challenger,
            &mut constraint_evals,
            num_vars_inter,
            num_eq_points,
            num_sel_points,
            &sumcheck.evals(),
        );
        all_constraint_evals.push(constraint_evals);

        // Compute the sumcheck polynomials for this folding round.
        // Each step produces a univariate polynomial h(X) whose values at 0 and 1
        // sum to the claimed round sum. The verifier's random challenge r is drawn
        // from the transcript and used to fold the polynomial: the new claimed sum
        // becomes h(r). The constraint is folded into the sumcheck via `Some(constraint)`.
        prover_randomness.extend(&sumcheck.compute_sumcheck_polynomials(
            &mut proof[round],
            &mut prover_challenger,
            folding,
            0,
            Some(constraint),
        ));

        // After folding, the polynomial has `folding` fewer variables.
        num_vars_inter -= folding;

        // Sanity check: the sumcheck state should track the correct variable count.
        assert_eq!(sumcheck.num_variables(), num_vars_inter);
    }

    // After all intermediate rounds, the remaining variables should equal final_rounds.
    assert_eq!(num_vars_inter, final_rounds);

    // FINAL ROUND: fold the remaining `final_rounds` variables down to a constant (0 variables).
    // No constraint is passed (None) because there are no more STIR queries after this point.

    prover_randomness.extend(&sumcheck.compute_sumcheck_polynomials(
        proof.last_mut().unwrap(),
        &mut prover_challenger,
        final_rounds,
        0,
        None,
    ));

    // After folding all variables, the polynomial is a constant: f(r) for the full
    // random point r = (r_0, r_1, ..., r_{n-1}) accumulated across all rounds.
    let final_folded_value = sumcheck.evals().as_constant().unwrap();

    // All variables have been folded away.
    assert_eq!(sumcheck.num_variables(), 0);

    // Core correctness check: the final folded value must equal the polynomial
    // evaluated at the full random point r that was built from all verifier challenges.
    assert_eq!(poly.eval_base(&prover_randomness), final_folded_value);

    // Commit the final evaluation to the Fiat-Shamir transcript.
    prover_challenger.observe_algebra_element(final_folded_value);

    // ==================== VERIFIER PHASE ====================
    // The verifier processes the proof transcript, reconstructing each round’s constraints
    // from the committed evaluations, verifying the sumcheck polynomials h(X), and
    // accumulating the random evaluation point r. At the end it checks:
    //   sum == f(r) * weights(z, r)
    // where weights(z, r) encodes the aggregated constraint polynomial.
    let mut verifier_challenger = challenger;

    // Running accumulated sum: starts at 0, gets updated each round by combine_evals
    // (which adds the weighted constraint evaluations) and verify_sumcheck_rounds
    // (which checks h(0)+h(1)==sum and updates sum := h(r) for the round’s challenge r).
    let mut sum = EF::ZERO;

    // The verifier builds up the same random point r as the prover, one chunk per round.
    let mut verifier_randomness = Point::new(vec![]);

    // Collect all constraints for the final weights check.
    let mut constraints = vec![];

    // Track remaining variables, mirroring the prover’s count.
    let mut num_vars_inter = num_vars;

    // Absorb the same domain separator as the prover to synchronize transcripts.
    domsep.observe_domain_separator(&mut verifier_challenger);

    // println!("{:#?}", proof);

    // VERIFY INITIAL ROUND (round 0):
    // The initial round only has eq constraints (no select constraints, hence 0 for num_sels).
    {
        // Reconstruct the initial round's constraint from the proof evaluations.
        let constraint = read_constraint(
            &mut verifier_challenger,
            &all_constraint_evals[0],
            num_vars_inter,
            num_eqs[0],
            0,
        );
        // Accumulate the weighted sum of constraint values
        constraint.combine_evals(&mut sum);
        // Save constraints for later equality check
        constraints.push(constraint);

        // Verify the initial sumcheck polynomials: for each folded variable, check that
        // h(0) + h(1) == claimed_sum, then update sum := h(r) with the challenge r.
        // The returned challenges are appended to the verifier's random point.
        verifier_randomness.extend(
            &proof[0]
                .verify_rounds(&mut verifier_challenger, &mut sum, 0)
                .unwrap(),
        );

        num_vars_inter -= folding_factor.at_round(0);
    }

    // VERIFY INTERMEDIATE ROUNDS (rounds 1..num_rounds):
    // For each round, reconstruct constraints from the proof, add their weighted sum,
    // then verify the sumcheck polynomials and collect the random challenges.
    for (round, (&num_eq_points, &num_sel_points)) in
        num_eqs.iter().skip(1).zip(num_sels.iter()).enumerate()
    {
        let round = round + 1;
        // Reconstruct round constraint from transcript
        let constraint = read_constraint(
            &mut verifier_challenger,
            &all_constraint_evals[round],
            num_vars_inter,
            num_eq_points,
            num_sel_points,
        );
        // Accumulate the weighted sum of constraint values
        constraint.combine_evals(&mut sum);
        // Save constraints for later equality check
        constraints.push(constraint);

        // Extend r with verifier's folding challenges
        // Note: proof.rounds[round - 1] because rounds are 0-indexed but we start at round 1
        let folding = folding_factor.at_round(round);
        verifier_randomness.extend(
            &proof[round]
                .verify_rounds(&mut verifier_challenger, &mut sum, 0)
                .unwrap(),
        );

        num_vars_inter -= folding;
    }

    // VERIFY FINAL ROUND: fold the last remaining variables and finalize the sum.
    verifier_randomness.extend(
        &verify_final_sumcheck_rounds(
            Some(proof.last().unwrap()),
            &mut verifier_challenger,
            &mut sum,
            final_rounds,
            0,
        )
        .unwrap(),
    );

    // ==================== FINAL CHECKS ====================

    // The prover and verifier must have derived the exact same random point r.
    // If they differ, the Fiat-Shamir transcripts diverged (a bug in the protocol).
    assert_eq!(prover_randomness, verifier_randomness);

    // Evaluate the aggregated constraint weight polynomial at the full random point.
    // This polynomial encodes the linear combination of all eq and select constraints
    // across all rounds. The reversed point accounts for the bit-reversal convention
    // used by the constraint evaluator.
    let evaluator = ConstraintPolyEvaluator::new(folding_factor);
    let weights = evaluator.eval_constraints_poly(&constraints, &verifier_randomness.reversed());

    // The fundamental sumcheck identity: the accumulated sum from all rounds must equal
    // the final folded polynomial value f(r) multiplied by the constraint weights.
    // This confirms that the sumcheck protocol correctly reduced the multilinear claim
    // down to a single-point evaluation.
    assert_eq!(sum, final_folded_value * weights);

    verifier_randomness
}

#[test]
fn test_sumcheck_prover() {
    // Brute-force test over all valid (num_vars, folding_factor) combinations.
    // For each configuration, 100 random constraint setups are tested to cover
    // a wide variety of eq/sel constraint counts (0, 1, or 2 each).
    //
    // Two sumcheck strategies are tested -- Classic and SVO (Shifted Virtual Oracle) --
    // and the test asserts they produce identical verifier randomness, confirming
    // that both strategies implement the same protocol semantics.
    let mut rng = SmallRng::seed_from_u64(0);

    // Iterate over polynomial sizes from 0 to 10 variables (1 to 1024 evaluations).
    for num_vars in 0..=10 {
        // Try every valid folding factor: fold 1..num_vars variables per round.
        for folding_factor in 1..=num_vars {
            // Run 100 random trials per configuration for statistical coverage.
            for _ in 0..1 {
                let folding_factor = FoldingFactor::Constant(folding_factor);

                // Compute total rounds (+1 because num_eqs includes the initial round).
                let num_rounds = folding_factor.compute_number_of_rounds(num_vars).0 + 1;

                // Randomly choose 0, 1, or 2 eq constraints per round.
                let num_eq_points = (0..num_rounds)
                    .map(|_| rng.random_range(0..=2))
                    .collect::<Vec<_>>();

                // Randomly choose 0, 1, or 2 select constraints per intermediate round.
                // (num_rounds - 1 because the initial round has no select constraints.)
                let num_sel_points = (0..num_rounds - 1)
                    .map(|_| rng.random_range(0..=2))
                    .collect::<Vec<_>>();

                // Run the full prover-verifier test with the Classic sumcheck strategy.
                let randomness_classic = run_sumcheck_test(
                    num_vars,
                    folding_factor,
                    &num_eq_points,
                    &num_sel_points,
                    SumcheckStrategy::Classic,
                );

                // Run the same test with the SVO (Shifted Virtual Oracle) strategy.
                let randomness_svo = run_sumcheck_test(
                    num_vars,
                    folding_factor,
                    &num_eq_points,
                    &num_sel_points,
                    SumcheckStrategy::Svo,
                );

                // Both strategies must produce the exact same random evaluation point.
                // This confirms they are functionally equivalent implementations.
                assert_eq!(randomness_classic, randomness_svo);
            }
        }
    }
}
