use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_multilinear_util::point::Point;

use crate::constraints::Constraint;
use crate::parameters::FoldingFactor;

/// Evaluate a single round's constraint.
fn eval_round<F: Field, EF: ExtensionField<F> + TwoAdicField>(
    constraint: &Constraint<F, EF>,
    point: &Point<EF>,
) -> EF {
    let point = point
        .get_subpoint_over_range(0..constraint.num_variables())
        .reversed();
    // Evaluate eq and sel constraints at the computed point
    let eq_contribution = constraint
        .iter_eqs()
        .map(|(pt, coeff)| pt.eq_poly(&point) * coeff)
        .sum::<EF>();

    let sel_contribution = constraint
        .iter_sels()
        .map(|(&var, coeff)| {
            let expanded = Point::expand_from_univariate(var, constraint.num_variables());
            coeff * expanded.select_poly(&point)
        })
        .sum::<EF>();

    eq_contribution + sel_contribution
}

/// Lightweight evaluator for the combined constraint polynomial W(r).
#[derive(Clone, Debug)]
pub struct ConstraintPolyEvaluator {
    /// The folding factor.
    pub folding_factor: FoldingFactor,
}

impl ConstraintPolyEvaluator {
    /// Creates a new `ConstraintPolyEvaluator` with the given parameters.
    #[must_use]
    pub const fn new(folding_factor: FoldingFactor) -> Self {
        Self { folding_factor }
    }

    /// Evaluate the combined constraint polynomial W(r).
    ///
    /// ## Key Insight
    /// Constraint i needs evaluation point matching its polynomial's remaining variables.
    /// This means using challenges from prover round i onwards + final sumcheck.
    #[must_use]
    pub fn eval_constraints_poly<F: Field, EF: ExtensionField<F> + TwoAdicField>(
        &self,
        constraints: &[Constraint<F, EF>],
        point: &Point<EF>,
    ) -> EF {
        constraints
            .iter()
            .map(|constraint| eval_round(constraint, point))
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;
    use p3_multilinear_util::poly::Poly;
    use proptest::prelude::*;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;
    use crate::constraints::statement::{EqStatement, SelectStatement};
    use crate::parameters::FoldingFactor;

    type F = BabyBear;
    type EF = BinomialExtensionField<BabyBear, 4>;

    #[test]
    fn test_eval_constraints_poly() {
        // -- Test Configuration --
        // We use 20 variables to ensure a non-trivial number of folding rounds.
        let num_vars = 20;
        // A constant folding factor of 5 is used.
        let folding_factor = FoldingFactor::Constant(5);
        // This configuration implies a 3-round folding schedule before the final polynomial:
        // Round 0: 20 vars -> 15 vars
        // Round 1: 15 vars -> 10 vars
        // Round 2: 10 vars ->  5 vars (final polynomial)

        // We will add a varying number of constraints in each round.
        let num_eq_constraints_per_round = &[2usize, 3, 1];
        let num_sel_constraints_per_round = &[31usize, 41, 51];

        // Initialize a deterministic random number generator for reproducibility.
        let mut rng = SmallRng::seed_from_u64(0);

        // -- Random Constraints and Challenges --
        // This block generates the inputs that the verifier would receive in a real proof.
        let mut num_vars_at_round = num_vars;
        let mut constraints = vec![];

        // Generate eq and select constraints and challenges for each of the 3 rounds.
        for (round_idx, (&num_eq, &num_sel)) in num_eq_constraints_per_round
            .iter()
            .zip(num_sel_constraints_per_round.iter())
            .enumerate()
        {
            // Generate a random combination challenge for this round.
            let gamma = rng.random();
            // Create eq statement for the current domain size (20, then 15, then 10).
            let mut eq_statement = EqStatement::initialize(num_vars_at_round);
            (0..num_eq).for_each(|_| {
                eq_statement.add_evaluated_constraint(
                    Point::rand(&mut rng, num_vars_at_round),
                    rng.random(),
                );
            });

            // Create select statement for the current domain size (20, then 15, then 10).
            let mut sel_statement = SelectStatement::<F, EF>::initialize(num_vars_at_round);
            (0..num_sel).for_each(|_| sel_statement.add_constraint(rng.random(), rng.random()));
            constraints.push(Constraint::new(gamma, eq_statement, sel_statement));

            // Shrink the number of variables for the next round.
            num_vars_at_round -= folding_factor.at_round(round_idx);
        }

        // Generate the final, full 20-dimensional challenge point `r`.
        let final_point = Point::rand(&mut rng, num_vars);

        // Calculate W(r) using the function under test
        let evaluator = ConstraintPolyEvaluator::new(folding_factor);
        let result_from_eval_poly = evaluator.eval_constraints_poly(&constraints, &final_point);

        // Calculate W(r) by materializing and evaluating round-by-round
        // This simpler, more direct method serves as our ground truth.
        // Loop through each round to calculate its contribution to the final evaluation.
        let expected_result = constraints
            .iter()
            .map(|constraint| {
                let num_vars = constraint.num_variables();
                let mut combined = Poly::zero(num_vars);
                let mut eval = EF::ZERO;
                constraint.combine(&mut combined, &mut eval);
                let point = final_point.get_subpoint_over_range(0..num_vars).reversed();
                combined.eval_ext::<F>(&point)
            })
            .sum::<EF>();

        // The result from the recursive function must match the materialized ground truth.
        assert_eq!(result_from_eval_poly, expected_result);
    }

    proptest! {
        #[test]
        fn prop_eval_constraints_poly(
            (num_vars, folding_factor_val) in (10..=20usize)
                .prop_flat_map(|n| (
                    Just(n),
                    2..=(n / 2)
                ))
        ) {
            // `Tracks the number of variables remaining before each round.
            let mut num_vars_current = num_vars;
            // The folding factor is constant for all rounds.
            let folding_factor = FoldingFactor::Constant(folding_factor_val);
            // Will store the number of variables folded in each specific round.
            let mut folding_factors_vec = vec![];
            // We simulate the folding process to build the schedule.
            //
            // The protocol folds variables until 0 remain.
            while num_vars_current > 0 {
                // In each round, we fold `folding_factor_val` variables.
                //
                // If this would leave fewer than 0 variables, we fold just enough to reach 0.
                let num_to_fold = core::cmp::min(folding_factor_val, num_vars_current);
                // This check avoids an infinite loop if `num_vars_current` gets stuck.
                if num_to_fold == 0 { break; }
                // Record the number of variables folded in this round.
                folding_factors_vec.push(num_to_fold);
                // Decrease the variable count for the next round.
                num_vars_current -= num_to_fold;
            }
            // The total number of folding rounds.
            let num_rounds = folding_factors_vec.len();

            // Use a seeded RNG for a reproducible test run.
            let mut rng = SmallRng::seed_from_u64(0);
            // For each round, generate a random number of constraints (from 0 to 8).
            let num_eq_constraints_per_round: Vec<usize> = (0..num_rounds)
                .map(|_| rng.random_range(0..=2))
                .collect();
            let num_sel_constraints_per_round: Vec<usize> = (0..num_rounds)
                .map(|_| rng.random_range(0..=2))
                .collect();

            // -- Random Constraints and Challenges --
            // This block generates the inputs that the verifier would receive in a real proof.
            let mut num_vars_current = num_vars;
            let mut constraints = vec![];

            // Generate eq and select constraints and alpha challenges for each of the 3 rounds.
            for (round_idx, (&num_eq, &num_sel)) in num_eq_constraints_per_round
                .iter()
                .zip(num_sel_constraints_per_round.iter())
                .enumerate()
            {
                // Generate a random combination scalar (alpha) for this round.
                let gamma = rng.random();
                // Create eq statement for the current domain size (20, then 15, then 10).
                let mut eq_statement = EqStatement::initialize(num_vars_current);
                (0..num_eq).for_each(|_| {
                    eq_statement.add_evaluated_constraint(
                        Point::rand(&mut rng, num_vars_current),
                        rng.random(),
                    );
                });

                // Create select statement for the current domain size (20, then 15, then 10).
                let mut sel_statement = SelectStatement::<F, EF>::initialize(num_vars_current);
                (0..num_sel).for_each(|_| sel_statement.add_constraint(rng.random(), rng.random()));
                constraints.push(Constraint::new(gamma, eq_statement, sel_statement));

                // Shrink the number of variables for the next round.
                num_vars_current -= folding_factors_vec[round_idx];
            }

            // Generate the final, full n-dimensional challenge point `r`.
            let final_point = Point::rand(&mut rng, num_vars);


            // Calculate W(r) using the function under test
            //
            // This is the recursive method we want to validate.
            let evaluator = ConstraintPolyEvaluator::new(folding_factor);
            let result_from_eval_poly =
                evaluator.eval_constraints_poly(&constraints, &final_point);

            // Calculate W(r) by materializing and evaluating round-by-round
            //
            // This simpler, more direct method serves as our ground truth.
            let mut num_vars_at_round = num_vars;
            // Loop through each round to calculate its contribution to the final evaluation.
            let expected_result = constraints
                .iter()
                .enumerate()
                .map(|(round_idx, constraint)| {
                    let point = final_point.get_subpoint_over_range(0..num_vars_at_round).reversed();
                    let mut combined = Poly::zero(constraint.num_variables());
                    let mut eval = EF::ZERO;
                    constraint.combine(&mut combined, &mut eval);
                    num_vars_at_round -= folding_factors_vec[round_idx];
                    combined.eval_ext::<F>(&point)
                })
                .sum::<EF>();

            // The result from the recursive function must match the materialized ground truth.
            prop_assert_eq!(result_from_eval_poly, expected_result);
        }
    }
}
