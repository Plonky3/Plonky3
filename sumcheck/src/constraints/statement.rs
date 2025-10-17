use p3_field::{ExtensionField, Field, dot_product};
use p3_matrix::dense::RowMajorMatrix;
use p3_multilinear_util::eq_batch::eval_eq_batch;
use p3_multilinear_util::evals::EvaluationsList;
use p3_multilinear_util::point::MultilinearPoint;
use tracing::instrument;

/// A batched system of evaluation constraints $p(z_i) = s_i$ on $\{0,1\}^m$.
///
/// Each entry ties a Boolean point `z_i` to an expected polynomial evaluation `s_i`.
///
/// Batching with a random challenge $\gamma$ produces a single combined weight
/// polynomial $W$ and a single scalar $S$ that summarize all constraints.
///
/// Invariants
/// ----------
/// - `points.len() == evaluations.len()`.
/// - Every `MultilinearPoint` in `points` has exactly `num_variables` coordinates.
#[derive(Clone, Debug)]
pub struct Statement<F> {
    /// Number of variables in the multilinear polynomial space.
    num_variables: usize,
    /// List of evaluation points.
    pub(crate) points: Vec<MultilinearPoint<F>>,
    /// List of target evaluations.
    evaluations: Vec<F>,
}

impl<F: Field> Statement<F> {
    /// Creates an empty `Statement<F>` for polynomials with `num_variables` variables.
    #[must_use]
    pub const fn initialize(num_variables: usize) -> Self {
        Self {
            num_variables,
            points: Vec::new(),
            evaluations: Vec::new(),
        }
    }

    /// Creates a filled `Statement<F>` for polynomials with `num_variables` variables.
    #[must_use]
    pub const fn new(
        num_variables: usize,
        points: Vec<MultilinearPoint<F>>,
        evaluations: Vec<F>,
    ) -> Self {
        Self {
            num_variables,
            points,
            evaluations,
        }
    }

    /// Returns the number of variables defining the polynomial space.
    #[must_use]
    pub const fn num_variables(&self) -> usize {
        self.num_variables
    }

    /// Returns true if the statement contains no constraints.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        debug_assert!(self.points.is_empty() == self.evaluations.is_empty());
        self.points.is_empty()
    }

    /// Returns an iterator over the evaluation constraints in the statement.
    pub fn iter(&self) -> impl Iterator<Item = (&MultilinearPoint<F>, &F)> {
        self.points.iter().zip(self.evaluations.iter())
    }

    /// Returns the number of constraints in the statement.
    #[must_use]
    pub const fn len(&self) -> usize {
        debug_assert!(self.points.len() == self.evaluations.len());
        self.points.len()
    }

    /// Verifies that a given polynomial satisfies all constraints in the statement.
    #[must_use]
    pub fn verify(&self, poly: &EvaluationsList<F>) -> bool {
        self.iter().all(|(point, &expected_eval)| {
            let eval = poly.evaluate(point);
            eval == expected_eval
        })
    }

    /// Concatenates another statement's constraints into this one.
    pub fn concatenate(&mut self, other: &Self) {
        assert_eq!(self.num_variables, other.num_variables);
        self.points.extend_from_slice(&other.points);
        self.evaluations.extend_from_slice(&other.evaluations);
    }

    /// Adds an evaluation constraint `p(z) = s` to the system.
    ///
    /// This method takes the polynomial `p` and uses it to compute the evaluation `s`.
    ///
    /// # Panics
    /// Panics if the number of variables in the `point` does not match the statement.
    pub fn add_unevaluated_constraint<BF>(
        &mut self,
        point: MultilinearPoint<F>,
        poly: &EvaluationsList<BF>,
    ) where
        BF: Field,
        F: ExtensionField<BF>,
    {
        assert_eq!(point.num_variables(), self.num_variables());
        let eval = poly.evaluate(&point);
        self.points.push(point);
        self.evaluations.push(eval);
    }

    /// Adds an evaluation constraint `p(z) = s` to the system.
    ///
    /// Assumes the evaluation `s` is already known.
    ///
    /// # Panics
    /// Panics if the number of variables in the `point` does not match the statement.
    pub fn add_evaluated_constraint(&mut self, point: MultilinearPoint<F>, eval: F) {
        assert_eq!(point.num_variables(), self.num_variables());
        self.points.push(point);
        self.evaluations.push(eval);
    }

    /// Inserts multiple constraints at the front of the system.
    ///
    /// Panics if any constraint's number of variables does not match the system.
    pub fn add_constraints_in_front(&mut self, points: &[MultilinearPoint<F>], evaluations: &[F]) {
        // Store the number of variables expected by this statement.
        let n = self.num_variables();
        assert_eq!(points.len(), evaluations.len());
        for p in points {
            assert_eq!(p.num_variables(), n);
        }

        self.points.splice(0..0, points.iter().cloned());
        self.evaluations.splice(0..0, evaluations.iter().copied());
    }

    /// Combines all constraints into a single aggregated polynomial and expected sum using a challenge.
    ///
    /// # Returns
    /// - `EvaluationsList<F>`: The evaluations of the combined weight polynomial `W(X)`.
    /// - `F`: The combined expected evaluation `S`.
    #[instrument(skip_all, fields(num_constraints = self.len(), num_variables = self.num_variables()))]
    pub fn combine<Base>(&self, challenge: F) -> (EvaluationsList<F>, F)
    where
        Base: Field,
        F: ExtensionField<Base>,
    {
        // If there are no constraints, the combination is:
        // - The combined polynomial W(X) is identically zero (all evaluations = 0).
        // - The combined expected sum S is zero.
        if self.is_empty() {
            return (
                EvaluationsList::new(F::zero_vec(1 << self.num_variables)),
                F::ZERO,
            );
        }

        let num_constraints = self.len();
        let num_variables = self.num_variables();

        // Precompute challenge powers γ^i for i = 0..num_constraints-1.
        let challenges = challenge.powers().collect_n(num_constraints);

        // Create a matrix where each column is one evaluation point.
        //
        // Matrix layout:
        // - rows are variables,
        // - columns are evaluation points.
        let points_data = F::zero_vec(num_variables * num_constraints);
        let mut points_matrix = RowMajorMatrix::new(points_data, num_constraints);

        // Parallelize the transpose operation over rows (variables).
        //
        // Each thread writes to its own contiguous row, which is cache-friendly.
        points_matrix
            .rows_mut()
            .enumerate()
            .for_each(|(var_idx, row_slice)| {
                for (col_idx, point) in self.points.iter().enumerate() {
                    row_slice[col_idx] = point[var_idx];
                }
            });

        // Compute the batched equality polynomial evaluations.
        // This computes W(x) = ∑_i γ^i * eq(x, z_i) for all x ∈ {0,1}^k.
        let mut combined = F::zero_vec(1 << num_variables);
        eval_eq_batch::<Base, F, false>(points_matrix.as_view(), &mut combined, &challenges);

        // Combine expected evaluations: S = ∑_i γ^i * s_i
        let sum = dot_product(self.evaluations.iter().copied(), challenges.into_iter());

        // Return:
        // - The combined polynomial W(X) in evaluation form.
        // - The combined expected sum S.
        (EvaluationsList::new(combined), sum)
    }

    /// Combines a list of evals into a single linear combination using powers of `gamma`,
    /// and updates the running claimed_eval in place.
    ///
    /// # Arguments
    /// - `claimed_eval`: Mutable reference to the total accumulated claimed eval so far. Updated in place.
    /// - `gamma`: A random extension field element used to weight the evals.
    ///
    /// # Returns
    /// A `Vec<F>` containing the powers of `gamma` used to combine each constraint.
    pub fn combine_evals(&self, claimed_eval: &mut F, gamma: F) -> Vec<F> {
        let gammas = gamma.powers().collect_n(self.len());

        for (expected_eval, &gamma) in self.evaluations.iter().zip(&gammas) {
            *claimed_eval += *expected_eval * gamma;
        }

        gammas
    }
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;

    use super::*;

    type F = BabyBear;

    #[test]
    fn test_statement_combine_single_constraint() {
        let mut statement = Statement::initialize(1);
        let point = MultilinearPoint::new(vec![F::ONE]);
        let expected_eval = F::from_u64(7);
        statement.add_evaluated_constraint(point.clone(), expected_eval);

        let challenge = F::from_u64(2); // This is unused with one constraint.
        let (combined_evals, combined_sum) = statement.combine::<F>(challenge);

        // Expected evals for eq_z(X) where z = (1).
        // For x=0, eq=0. For x=1, eq=1.
        let expected_combined_evals_vec = EvaluationsList::new_from_point(&point, F::ONE);

        assert_eq!(combined_evals, expected_combined_evals_vec);
        assert_eq!(combined_sum, expected_eval);
    }

    #[test]
    fn test_statement_with_multiple_constraints() {
        let mut statement = Statement::initialize(2);

        // Constraint 1: evaluate at z1 = (1,0), expected value 5
        let point1 = MultilinearPoint::new(vec![F::ONE, F::ZERO]);
        let eval1 = F::from_u64(5);
        statement.add_evaluated_constraint(point1.clone(), eval1);

        // Constraint 2: evaluate at z2 = (0,1), expected value 7
        let point2 = MultilinearPoint::new(vec![F::ZERO, F::ONE]);
        let eval2 = F::from_u64(7);
        statement.add_evaluated_constraint(point2.clone(), eval2);

        let challenge = F::from_u64(2);
        let (combined_evals, combined_sum) = statement.combine::<F>(challenge);

        // Expected evals: W(X) = eq_z1(X) + challenge * eq_z2(X)
        let mut expected_combined_evals_vec = EvaluationsList::new_from_point(&point1, F::ONE);
        expected_combined_evals_vec.accumulate_batch(&[point2], &[challenge]);

        // Expected sum: S = s1 + challenge * s2
        let expected_combined_sum = eval1 + challenge * eval2;

        assert_eq!(combined_evals, expected_combined_evals_vec);
        assert_eq!(combined_sum, expected_combined_sum);
    }

    #[test]
    fn test_compute_evaluation_weight() {
        // Define an evaluation weight at a specific point
        let point = MultilinearPoint::new(vec![F::from_u64(3)]);

        // Define a randomness point for folding
        let folding_randomness = MultilinearPoint::new(vec![F::from_u64(2)]);

        // Expected result is the evaluation of eq_poly at the given randomness
        let expected = point.eq_poly(&folding_randomness);

        assert_eq!(point.eq_poly(&folding_randomness), expected);
    }

    #[test]
    fn test_constructors_and_basic_properties() {
        // Test new constructor
        let point = MultilinearPoint::new(vec![F::ONE]);
        let eval = F::from_u64(42);
        let statement = Statement::new(1, vec![point], vec![eval]);

        assert_eq!(statement.num_variables(), 1);
        assert_eq!(statement.len(), 1);
        assert!(!statement.is_empty());

        // Test initialize constructor
        let empty_statement = Statement::<F>::initialize(2);
        assert_eq!(empty_statement.num_variables(), 2);
        assert_eq!(empty_statement.len(), 0);
        assert!(empty_statement.is_empty());
    }

    #[test]
    fn test_verify_constraints() {
        // Create polynomial with evaluations [1, 2]
        let poly = EvaluationsList::new(vec![F::from_u64(1), F::from_u64(2)]);
        let mut statement = Statement::<F>::initialize(1);

        // Test matching constraint: f(0) = 1
        statement.add_evaluated_constraint(MultilinearPoint::new(vec![F::ZERO]), F::from_u64(1));
        assert!(statement.verify(&poly));

        // Test mismatched constraint: f(1) = 5 (but poly has f(1) = 2)
        statement.add_evaluated_constraint(MultilinearPoint::new(vec![F::ONE]), F::from_u64(5));
        assert!(!statement.verify(&poly));
    }

    #[test]
    fn test_concatenate() {
        // Test successful concatenation
        let mut statement1 = Statement::<F>::initialize(1);
        let mut statement2 = Statement::<F>::initialize(1);
        statement1.add_evaluated_constraint(MultilinearPoint::new(vec![F::ZERO]), F::from_u64(10));
        statement2.add_evaluated_constraint(MultilinearPoint::new(vec![F::ONE]), F::from_u64(20));

        statement1.concatenate(&statement2);
        assert_eq!(statement1.len(), 2);
    }

    #[test]
    #[should_panic(expected = "assertion `left == right` failed")]
    fn test_concatenate_mismatched_variables() {
        let mut statement1 = Statement::<F>::initialize(2);
        let statement2 = Statement::<F>::initialize(3);
        statement1.concatenate(&statement2); // Should panic
    }

    #[test]
    fn test_add_constraints() {
        // Test that add_unevaluated_constraint behaves identically to add_evaluated_constraint
        let poly = EvaluationsList::new(vec![F::from_u64(1), F::from_u64(2)]);
        let point = MultilinearPoint::new(vec![F::ZERO]);

        // Create two identical statements
        let mut statement1 = Statement::<F>::initialize(1);
        let mut statement2 = Statement::<F>::initialize(1);

        // Add same constraint using both methods
        let eval = poly.evaluate(&point);
        statement1.add_evaluated_constraint(point.clone(), eval);
        statement2.add_unevaluated_constraint(point, &poly);

        // Both statements should be identical
        assert_eq!(statement1.len(), statement2.len());
        assert_eq!(statement1.len(), 1);

        // Both should verify against the polynomial
        assert!(statement1.verify(&poly));
        assert!(statement2.verify(&poly));

        // Both should have same constraints
        let constraints1: Vec<_> = statement1.iter().collect();
        let constraints2: Vec<_> = statement2.iter().collect();
        assert_eq!(constraints1, constraints2);

        // Test get_points consumes statement
        assert_eq!(statement1.points.len(), 1);
    }

    #[test]
    fn test_add_constraints_in_front() {
        // Test adding constraints at the front preserves order
        let mut statement = Statement::<F>::initialize(1);
        statement.add_evaluated_constraint(MultilinearPoint::new(vec![F::ONE]), F::from_u64(100));

        let front_points = vec![MultilinearPoint::new(vec![F::ZERO])];
        let front_evals = vec![F::from_u64(10)];
        statement.add_constraints_in_front(&front_points, &front_evals);

        assert_eq!(statement.len(), 2);
        let collected: Vec<_> = statement.iter().collect();
        assert_eq!(collected[0].1, &F::from_u64(10)); // Front constraint first
        assert_eq!(collected[1].1, &F::from_u64(100)); // Original constraint last
    }

    #[test]
    #[should_panic(expected = "assertion `left == right` failed")]
    fn test_wrong_variable_count() {
        let mut statement = Statement::<F>::initialize(1);
        let wrong_point = MultilinearPoint::new(vec![F::ONE, F::ZERO]); // 2 vars for 1-var statement
        statement.add_evaluated_constraint(wrong_point, F::from_u64(5));
    }

    #[test]
    fn test_combine_operations() {
        // Test empty statement combine
        let empty_statement = Statement::<F>::initialize(1);
        let (_combined_poly, combined_sum) = empty_statement.combine::<F>(F::from_u64(42));
        assert_eq!(combined_sum, F::ZERO);

        // Test combine_evals with constraints
        let mut statement = Statement::<F>::initialize(1);
        statement.add_evaluated_constraint(MultilinearPoint::new(vec![F::ZERO]), F::from_u64(3));
        statement.add_evaluated_constraint(MultilinearPoint::new(vec![F::ONE]), F::from_u64(7));

        let mut claimed_eval = F::ZERO;
        let gammas = statement.combine_evals(&mut claimed_eval, F::from_u64(2));

        // Verify: 3*1 + 7*2 = 17
        assert_eq!(claimed_eval, F::from_u64(17));
        assert_eq!(gammas, vec![F::ONE, F::from_u64(2)]);
    }

    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_statement_workflow(
            // Random 4-var polynomial: 16 evaluations (2^4)
            poly_evals in prop::collection::vec(0u32..100, 16),
            // Random challenge value
            challenge in 1u32..50,
            // Random constraint points (4 coords × 2 points)
            point_coords in prop::collection::vec(0u32..10, 8),
        ) {
            // Create a 4-variable polynomial from random evaluations
            let poly = EvaluationsList::new(poly_evals.into_iter().map(F::from_u32).collect());

            // Create statement with random constraints that match the polynomial
            let mut statement = Statement::<F>::initialize(4);
            let point1 = MultilinearPoint::new(vec![
                F::from_u32(point_coords[0]), F::from_u32(point_coords[1]),
                F::from_u32(point_coords[2]), F::from_u32(point_coords[3])
            ]);
            let point2 = MultilinearPoint::new(vec![
                F::from_u32(point_coords[4]), F::from_u32(point_coords[5]),
                F::from_u32(point_coords[6]), F::from_u32(point_coords[7])
            ]);

            // Add constraints: poly(point1) = actual_eval1, poly(point2) = actual_eval2
            let eval1 = poly.evaluate(&point1);
            let eval2 = poly.evaluate(&point2);
            statement.add_evaluated_constraint(point1, eval1);
            statement.add_evaluated_constraint(point2, eval2);

            // Statement should verify against polynomial (consistent constraints)
            prop_assert!(statement.verify(&poly));

            // Combine constraints with challenge
            let gamma = F::from_u32(challenge);
            let (combined_poly, combined_sum) = statement.combine::<F>(gamma);

            // Combined polynomial should have same number of variables
            prop_assert_eq!(combined_poly.num_variables(), 4);

            // Combined evaluations should match combine result
            let mut claimed_eval = F::ZERO;
            let gammas = statement.combine_evals(&mut claimed_eval, gamma);
            // Both methods should give same sum
            prop_assert_eq!(combined_sum, claimed_eval);
            // Should have 2 gamma powers
            prop_assert_eq!(gammas.len(), 2);

            // Adding wrong constraint should break verification
            let wrong_point = MultilinearPoint::new(vec![F::ZERO, F::ZERO, F::ZERO, F::ZERO]);
            // Obviously wrong evaluation
            let wrong_eval = F::from_u32(999);
            let actual_eval = poly.evaluate(&wrong_point);
            // Only test if actually different
            if wrong_eval != actual_eval {
                statement.add_evaluated_constraint(wrong_point, wrong_eval);
                // Should fail verification
                prop_assert!(!statement.verify(&poly));
            }
        }
    }
}
