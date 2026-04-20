use alloc::vec;
use alloc::vec::Vec;

use p3_challenger::FieldChallenger;
use p3_field::PrimeCharacteristicRing;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use rand::SeedableRng;
use rand::rngs::SmallRng;

use super::ProverLayout;
use crate::sumcheck::SumcheckData;
use crate::sumcheck::layout::{
    LayoutStrategy, PrefixLayout, SuffixLayout, Table, TableShape, VerifierLayout, Witness,
};
use crate::sumcheck::strategy::VariableOrder;
use crate::sumcheck::tests::*;

fn run_test<L: LayoutStrategy>(calls: &[(usize, &[usize])]) {
    let folding = 4;
    let pow_bits = 10;

    let mut rng = SmallRng::seed_from_u64(1);

    // Table a:
    let a0 = Poly::<F>::rand(&mut rng, 10);
    let a1 = Poly::<F>::rand(&mut rng, 10);

    // Table b:
    let b0 = Poly::<F>::rand(&mut rng, 9);
    let b1 = Poly::<F>::rand(&mut rng, 9);

    // Get fresh challenger instance for the prover.
    let mut prover_challenger = challenger();

    // Witness to be committed
    let witness = Witness::new(
        vec![Table::new(vec![b0, b1]), Table::new(vec![a0, a1])],
        folding,
    );
    let stacked_num_vars = witness.num_vars();

    // Pretend that prover committed to witness.
    let mut layout = witness.as_committed::<EF, L>();
    let stacked_poly = layout.poly().clone();

    let mut opening_claims = Vec::<(usize, Vec<usize>, Vec<EF>)>::new();
    let mut apply_shared_point = |layout: &mut ProverLayout<F, EF, L>,
                                  challenger: &mut MyChallenger,
                                  table_idx: usize,
                                  polys: &[usize]| {
        let point = Point::expand_from_univariate(
            challenger.sample_algebra_element(),
            layout.num_vars_table(table_idx),
        );
        let evals = layout.eval(&point, table_idx, polys.to_vec());
        challenger.observe_algebra_slice(&evals);
        opening_claims.push((table_idx, polys.to_vec(), evals));
    };

    // Replay the caller-provided opening schedule.
    for &(table_idx, polys) in calls {
        apply_shared_point(&mut layout, &mut prover_challenger, table_idx, polys);
    }

    let virtual_eval = layout.add_virtual_eval(&mut prover_challenger);

    let mut proof0 = SumcheckData::<F, EF>::default();
    let (mut prover, mut prover_randomness) =
        layout.new_prover(&mut proof0, 0, &mut prover_challenger);

    assert_eq!(proof0.num_rounds(), folding);
    // The sumcheck state should track the correct variable count.
    assert_eq!(prover.num_vars(), stacked_num_vars - folding);

    // Build STIR constraints for this round: sample random evaluation points,
    // compute polynomial evaluations, and assemble them into a Constraint.
    let mut intermediate_evals: Vec<EF> = Vec::new();
    let constraint = make_constraint_ext(
        &mut prover_challenger,
        &mut intermediate_evals,
        prover.num_vars(),
        10,
        100,
        &prover.poly(),
    );

    // Compute the sumcheck polynomials for this folding round.
    // Each step produces a univariate polynomial h(X) whose values at 0 and 1
    // sum to the claimed round sum. The verifier's random challenge r is drawn
    // from the transcript and used to fold the polynomial: the new claimed sum
    // becomes h(r). The constraint is folded into the sumcheck via `Some(constraint)`.
    let mut proof1 = SumcheckData::<F, EF>::default();
    prover_randomness.extend(&prover.compute_sumcheck_polynomials(
        &mut proof1,
        &mut prover_challenger,
        folding,
        pow_bits,
        Some(constraint),
    ));
    let remaining_vars = stacked_num_vars - folding - folding;
    assert_eq!(proof1.num_rounds(), folding);
    // The sumcheck state should track the correct variable count.
    assert_eq!(prover.num_vars(), remaining_vars);

    // FINAL ROUND: fold the remaining `final_rounds` variables down to a constant (0 variables).
    // No constraint is passed (None) because there are no more STIR queries after this point.
    let mut proof2 = SumcheckData::<F, EF>::default();
    prover_randomness.extend(&prover.compute_sumcheck_polynomials(
        &mut proof2,
        &mut prover_challenger,
        remaining_vars,
        0,
        None,
    ));
    // All variables have been folded away.
    assert_eq!(proof2.num_rounds(), remaining_vars);
    assert_eq!(prover.num_vars(), 0);

    // After folding all variables, the polynomial is a constant: f(r) for the full
    // random point r = (r_0, r_1, ..., r_{n-1}) accumulated across all rounds.
    let final_folded_value = prover.poly().as_constant().unwrap();

    // Core correctness check: the final folded value must equal the polynomial
    // evaluated at the full random point r that was built from all verifier challenges.
    match L::var_order() {
        // For prefix layout, the random point is built in the same order as the rounds.
        VariableOrder::Prefix => {
            assert_eq!(
                stacked_poly.eval_base(&prover_randomness),
                final_folded_value
            );
        }
        VariableOrder::Suffix => {
            // For suffix layout, the random point is built in reverse order of the rounds.
            assert_eq!(
                stacked_poly.eval_base(&prover_randomness.reversed()),
                final_folded_value
            );
        }
    }

    // Get fresh challenger instance for the prover.
    let mut verifier_challenger = challenger();

    let tables = vec![TableShape::new(9, 2), TableShape::new(10, 2)];
    let mut layout: VerifierLayout<F, EF> = VerifierLayout::new(&tables);
    for (table_idx, polys, evals) in opening_claims {
        let point = Point::expand_from_univariate(
            verifier_challenger.sample_algebra_element(),
            layout.num_vars_table(table_idx),
        );
        verifier_challenger.observe_algebra_slice(&evals);
        layout.add_claim(table_idx, point, &polys, &evals);
    }

    let virtual_point = Point::expand_from_univariate(
        verifier_challenger.sample_algebra_element(),
        stacked_num_vars,
    );
    verifier_challenger.observe_algebra_element(virtual_eval);
    layout.add_virtual_eval(virtual_point, virtual_eval);

    let alpha = verifier_challenger.sample_algebra_element();
    let initial_constraint = layout.constraint(alpha);
    let mut sum = EF::ZERO;
    initial_constraint.combine_evals(&mut sum);
    assert_eq!(sum, layout.sum(alpha));

    let mut constraints = vec![initial_constraint];
    let mut verifier_challenge = Point::new(vec![]);
    verifier_challenge.extend(
        &proof0
            .verify_rounds(&mut verifier_challenger, &mut sum, 0)
            .unwrap(),
    );

    let intermediate_constraint = read_constraint(
        &mut verifier_challenger,
        &intermediate_evals,
        stacked_num_vars - folding,
        10,
        100,
    );
    intermediate_constraint.combine_evals(&mut sum);
    constraints.push(intermediate_constraint);
    verifier_challenge.extend(
        &proof1
            .verify_rounds(&mut verifier_challenger, &mut sum, pow_bits)
            .unwrap(),
    );

    verifier_challenge.extend(
        &proof2
            .verify_rounds(&mut verifier_challenger, &mut sum, 0)
            .unwrap(),
    );

    assert_eq!(prover_randomness, verifier_challenge);
    let weights = L::eval_constraints_poly(&constraints, &verifier_challenge);
    assert_eq!(sum, final_folded_value * weights);
}

// Baseline opening schedule: polys within each claim are in ascending order.
const ASCENDING_POLYS: &[(usize, &[usize])] = &[
    (0, &[0, 1]),
    (0, &[0]),
    (1, &[0, 1]),
    (1, &[1]),
];

// Regression schedule: first call on table 0 uses non-ascending polys.
// Triggers alpha/partial-eval misalignment in the buggy SuffixLayout path.
const NON_ASCENDING_POLYS: &[(usize, &[usize])] = &[
    (0, &[1, 0]),
    (0, &[0]),
    (1, &[0, 1]),
    (1, &[1]),
];

#[test]
fn test_prefix_layout() {
    run_test::<PrefixLayout>(ASCENDING_POLYS);
}

#[test]
fn test_suffix_layout() {
    run_test::<SuffixLayout>(ASCENDING_POLYS);
}

#[test]
fn test_prefix_layout_non_ascending_polys() {
    // Invariant: opening order within a claim must not affect correctness.
    run_test::<PrefixLayout>(NON_ASCENDING_POLYS);
}

#[test]
fn test_suffix_layout_non_ascending_polys() {
    // Regression: SuffixLayout previously pushed alphas in visit order while
    // partial evals iterated in natural order, misaligning the batching
    // coefficients whenever polys within a claim were not sorted ascending.
    run_test::<SuffixLayout>(NON_ASCENDING_POLYS);
}
