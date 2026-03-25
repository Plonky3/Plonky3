use p3_field::{ExtensionField, Field, PackedValue};
use p3_multilinear_util::poly::Poly;
use p3_multilinear_util::point::Point;
use p3_util::log2_strict_usize;

use crate::constraints::statement::{EqStatement, SelectStatement};

/// Constraint evaluation utilities.
pub mod evaluator;

/// Statement types for polynomial evaluation constraints.
pub mod statement;

/// A combined constraint system with equality and selection statements.
///
/// This struct represents a unified constraint system that combines:
/// - **Equality constraints**: Polynomial evaluations at specific points
/// - **Select constraints**: Selection-based polynomial evaluations
///
/// Both constraint types are batched using powers of a random challenge `γ`.
///
/// # Mathematical Structure
///
/// Given `n_eq` equality constraints and `n_sel` select constraints, the combined
/// constraint polynomial is:
///
/// ```text
/// W(X) = Σ_{i=0}^{n_eq-1} γ^i · eq(X, z_eq_i) + Σ_{j=0}^{n_sel-1} γ^{n_eq+j} · select(pow(z_sel_j), X)
/// ```
///
/// The combined expected evaluation is:
///
/// ```text
/// S = Σ_{i=0}^{n_eq-1} γ^i · s_eq_i + Σ_{j=0}^{n_sel-1} γ^{n_eq+j} · s_sel_j
/// ```
#[derive(Clone, Debug)]
pub struct Constraint<F: Field, EF: ExtensionField<F>> {
    /// Equality-based evaluation constraints of the form `p(z_i) = s_i`.
    ///
    /// Each constraint specifies a point `z_i` and expected evaluation `s_i`.
    pub eq_statement: EqStatement<EF>,

    /// Selection-based evaluation constraints of the form `p(z_j) = s_j`.
    ///
    /// Each constraint specifies a univariate value `z_j` that is expanded
    /// via the power map to create a multilinear evaluation point.
    pub sel_statement: SelectStatement<F, EF>,

    /// Random challenge `γ` used for batching constraints.
    ///
    /// Powers of this challenge weight different constraints:
    /// - Equality constraints use `γ^0, γ^1, ..., γ^{n_eq-1}`
    /// - Select constraints use `γ^{n_eq}, γ^{n_eq+1}, ..., γ^{n_eq+n_sel-1}`
    pub challenge: EF,
}

impl<F: Field, EF: ExtensionField<F>> Constraint<F, EF> {
    /// Creates a new constraint combining equality and select statements.
    ///
    /// This constructor initializes a unified constraint system that batches both
    /// equality-based and selection-based polynomial evaluation constraints using
    /// powers of the provided challenge.
    ///
    /// # Parameters
    ///
    /// - `challenge`: Random challenge `γ` for batching constraints
    /// - `eq_statement`: Equality constraints `p(z_i) = s_i`
    /// - `sel_statement`: Selection constraints via power map expansion
    ///
    /// # Panics
    ///
    /// Panics if the number of variables differs between statements.
    ///
    /// # Invariant
    ///
    /// Both statements must operate over the same number of variables to ensure
    /// the combined weight polynomial is well-defined.
    #[must_use]
    pub const fn new(
        challenge: EF,
        eq_statement: EqStatement<EF>,
        sel_statement: SelectStatement<F, EF>,
    ) -> Self {
        // Verify that both statements have the same number of variables.
        //
        // This ensures the combined polynomial has a consistent domain.
        assert!(eq_statement.num_variables() == sel_statement.num_variables());

        // Construct the combined constraint with both statement types.
        Self {
            eq_statement,
            sel_statement,
            challenge,
        }
    }

    /// Creates a constraint with only equality statements.
    ///
    /// This is a convenience constructor for the common case where only equality
    /// constraints are needed. An empty select statement is automatically created
    /// with the same number of variables.
    ///
    /// # Parameters
    ///
    /// - `challenge`: Random challenge `γ` for batching eq constraints
    /// - `eq_statement`: Equality constraints `p(z_i) = s_i`
    ///
    /// # Returns
    ///
    /// A `Constraint` with the given equality statement and an empty select statement.
    #[must_use]
    pub const fn new_eq_only(challenge: EF, eq_statement: EqStatement<EF>) -> Self {
        // Extract the number of variables from the equality statement.
        let num_variables = eq_statement.num_variables();

        // Create a constraint with an empty select statement that has
        // the same number of variables as the equality statement.
        Self::new(
            challenge,
            eq_statement,
            SelectStatement::initialize(num_variables),
        )
    }

    /// Returns the number of variables in the constraint polynomial.
    ///
    /// This value determines the dimension of the Boolean hypercube `{0,1}^k`
    /// over which the constraint polynomial is evaluated.
    ///
    /// # Returns
    ///
    /// The number of variables `k` shared by both statement types.
    #[must_use]
    pub const fn num_variables(&self) -> usize {
        // The number of variables is determined by the equality statement.
        //
        // By construction, the select statement has the same value.
        self.eq_statement.num_variables()
    }

    /// Combines expected evaluations using challenge powers.
    ///
    /// This accumulates the weighted sum of all expected constraint evaluations:
    /// ```text
    /// eval += Σ_{i=0}^{n_eq-1} γ^i · s_eq_i + Σ_{j=0}^{n_sel-1} γ^{n_eq+j} · s_sel_j
    /// ```
    ///
    /// # Parameters
    ///
    /// - `eval`: Mutable accumulator for the combined expected evaluation
    ///
    /// # Implementation Notes
    ///
    /// The equality statement uses challenge powers `γ^0, γ^1, ...`
    /// The select statement continues with powers `γ^{n_eq}, γ^{n_eq+1}, ...`
    /// to ensure distinct weights for all constraints.
    pub fn combine_evals(&self, eval: &mut EF) {
        // Accumulate equality constraint evaluations weighted by γ^i.
        //
        // This adds: Σ_{i=0}^{n_eq-1} γ^i · s_eq_i
        self.eq_statement.combine_evals(eval, self.challenge);

        // Accumulate select constraint evaluations weighted by γ^{n_eq+j}.
        // The shift ensures distinct challenge powers for each constraint.
        //
        // This adds: Σ_{j=0}^{n_sel-1} γ^{n_eq+j} · s_sel_j
        self.sel_statement
            .combine_evals(eval, self.challenge, self.eq_statement.len());
    }

    /// Combines constraint polynomials into weight polynomial and expected evaluation.
    ///
    /// This method accumulates both:
    /// 1. The weight polynomial `W(X)` evaluated at all hypercube points
    /// 2. The expected evaluation `S` as a scalar
    ///
    /// Both are added to the provided accumulators, allowing for incremental
    /// combination across multiple constraints.
    ///
    /// # Parameters
    ///
    /// - `combined`: Accumulator for weight polynomial evaluations `W(b)` at all `b ∈ {0,1}^k`
    /// - `eval`: Accumulator for the combined expected evaluation `S`
    ///
    /// # Mathematical Details
    ///
    /// Updates `combined[b]` for each `b ∈ {0,1}^k`:
    /// ```text
    /// combined[b] += Σ_i γ^i · eq(b, z_eq_i) + Σ_j γ^{n_eq+j} · select(pow(z_sel_j), b)
    /// ```
    ///
    /// Updates `eval`:
    /// ```text
    /// eval += Σ_i γ^i · s_eq_i + Σ_j γ^{n_eq+j} · s_sel_j
    /// ```
    pub fn combine(&self, combined: &mut Poly<EF>, eval: &mut EF) {
        // Combine equality constraints with accumulation enabled (INITIALIZED=true).
        // This adds the equality portion of W(X) to the existing values in `combined`.
        self.eq_statement
            .combine_hypercube::<F, true>(combined, eval, self.challenge);

        // Combine select constraints, continuing from where equality left off.
        // The shift parameter ensures select constraints use distinct challenge powers.
        self.sel_statement
            .combine(combined, eval, self.challenge, self.eq_statement.len());
    }

    /// Combines constraint polynomials into weight polynomial and expected evaluation.
    ///
    /// This method accumulates both:
    /// 1. The weight polynomial `W(X)` evaluated at all hypercube points
    /// 2. The expected evaluation `S` as a scalar
    ///
    /// Both are added to the provided accumulators, allowing for incremental
    /// combination across multiple constraints.
    ///
    /// # Parameters
    ///
    /// - `combined`: Accumulator for packed weight polynomial evaluations `W(b)` at all `b ∈ {0,1}^k`
    /// - `eval`: Accumulator for the combined expected evaluation `S`
    ///
    /// # Mathematical Details
    ///
    /// Updates `combined[b]` for each `b ∈ {0,1}^k`:
    /// ```text
    /// combined[b] += Σ_i γ^i · eq(b, z_eq_i) + Σ_j γ^{n_eq+j} · select(pow(z_sel_j), b)
    /// ```
    ///
    /// Updates `eval`:
    /// ```text
    /// eval += Σ_i γ^i · s_eq_i + Σ_j γ^{n_eq+j} · s_sel_j
    /// ```
    pub fn combine_packed(&self, combined: &mut Poly<EF::ExtensionPacking>, eval: &mut EF) {
        // Combine equality constraints with accumulation enabled (INITIALIZED=true).
        // This adds the equality portion of W(X) to the existing values in `combined`.
        self.eq_statement
            .combine_hypercube_packed::<F, true>(combined, eval, self.challenge);

        // Combine select constraints, continuing from where equality left off.
        // The shift parameter ensures select constraints use distinct challenge powers.
        self.sel_statement
            .combine_packed(combined, eval, self.challenge, self.eq_statement.len());
    }

    /// Creates a new combined weight polynomial and expected evaluation.
    ///
    /// This is similar to [`combine`](Self::combine) but creates fresh accumulators
    /// instead of adding to existing ones.
    ///
    /// # Returns
    ///
    /// A tuple `(W, S)` where:
    /// - `W`: Weight polynomial evaluations at all points in `{0,1}^k`
    /// - `S`: Combined expected evaluation scalar
    ///
    /// # Usage
    ///
    /// Use this method when starting a new constraint combination.
    /// Use [`combine`](Self::combine) when accumulating multiple constraints.
    pub fn combine_new(&self) -> (Poly<EF>, EF) {
        // Initialize fresh accumulators for the weight polynomial and expected evaluation.
        // The weight polynomial needs 2^k entries for the full Boolean hypercube.
        let mut combined = Poly::zero(self.num_variables());
        let mut eval = EF::ZERO;

        // Combine equality constraints without accumulation (INITIALIZED=false).
        // This directly writes the equality portion of W(X) to `combined`.
        self.eq_statement
            .combine_hypercube::<F, false>(&mut combined, &mut eval, self.challenge);

        // Add select constraints to the weight polynomial and expected evaluation.
        // The shift ensures select constraints use distinct challenge powers.
        self.sel_statement.combine(
            &mut combined,
            &mut eval,
            self.challenge,
            self.eq_statement.len(),
        );

        // Return the completed weight polynomial and expected evaluation.
        (combined, eval)
    }

    /// Creates a new combined weight polynomial in packed form and expected evaluation.
    ///
    /// This is similar to [`combine_packed`](Self::combine_packed) but creates fresh accumulators
    /// instead of adding to existing ones.
    ///
    /// # Returns
    ///
    /// A tuple `(W, S)` where:
    /// - `W`: Weight polynomial evaluations at all points in `{0,1}^k`
    /// - `S`: Combined expected evaluation scalar
    ///
    /// # Usage
    ///
    /// Use this method when starting a new constraint combination.
    /// Use [`combine_packed`](Self::combine_packed) when accumulating multiple constraints.
    pub fn combine_new_packed(&self) -> (Poly<EF::ExtensionPacking>, EF) {
        let k_pack = log2_strict_usize(F::Packing::WIDTH);
        let k = self.num_variables();

        // Initialize fresh accumulators for the weight polynomial and expected evaluation.
        // The weight polynomial needs 2^(k-k_pack) packed entries for the full Boolean hypercube.
        let mut combined = Poly::zero(k - k_pack);
        let mut eval = EF::ZERO;

        // Combine equality constraints without accumulation (INITIALIZED=false).
        // This directly writes the equality portion of W(X) to `combined`.
        self.eq_statement.combine_hypercube_packed::<F, false>(
            &mut combined,
            &mut eval,
            self.challenge,
        );

        // Add select constraints to the weight polynomial and expected evaluation.
        // The shift ensures select constraints use distinct challenge powers.
        self.sel_statement.combine_packed(
            &mut combined,
            &mut eval,
            self.challenge,
            self.eq_statement.len(),
        );

        // Return the completed weight polynomial and expected evaluation.
        (combined, eval)
    }

    /// Iterates over equality constraints with their challenge weights.
    ///
    /// This produces pairs `(z_i, γ^i)` for each equality constraint where:
    /// - `z_i` is the evaluation point
    /// - `γ^i` is the challenge power for this constraint
    ///
    /// # Returns
    ///
    /// An iterator over `(&Point<EF>, EF)` pairs.
    pub fn iter_eqs(&self) -> impl Iterator<Item = (&Point<EF>, EF)> {
        // Pair each equality point with its corresponding challenge power.
        // Points are weighted by γ^0, γ^1, γ^2, ...
        self.eq_statement.points.iter().zip(self.challenge.powers())
    }

    /// Iterates over select constraints with their challenge weights.
    ///
    /// This produces pairs `(z_j, γ^{n_eq+j})` for each select constraint where:
    /// - `z_j` is the univariate evaluation point (before power map expansion)
    /// - `γ^{n_eq+j}` is the challenge power for this constraint
    ///
    /// # Returns
    ///
    /// An iterator over `(&F, EF)` pairs.
    ///
    /// # Implementation Notes
    ///
    /// Challenge powers are skipped by `n_eq` to ensure select constraints
    /// use distinct powers from equality constraints.
    pub fn iter_sels(&self) -> impl Iterator<Item = (&F, EF)> {
        // Pair each select variable with its corresponding challenge power.
        // Powers start at γ^{n_eq} to avoid overlap with equality constraints.
        self.sel_statement
            .vars
            .iter()
            .zip(self.challenge.powers().skip(self.eq_statement.len()))
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;

    use super::*;

    /// Type alias for the base field used in tests
    type F = BabyBear;

    /// Type alias for the extension field used in tests
    type EF = BinomialExtensionField<F, 4>;

    #[test]
    fn test_constraint_new() {
        // Declare test parameters explicitly

        // Number of variables for the constraint system
        let num_variables = 3;

        // Random challenge for batching constraints
        let challenge = EF::from_u64(42);

        // Create an equality statement with 2 constraints
        let eq_point_0 = Point::new(vec![EF::from_u64(1), EF::from_u64(2), EF::from_u64(3)]);
        let eq_eval_0 = EF::from_u64(10);
        let eq_point_1 = Point::new(vec![EF::from_u64(4), EF::from_u64(5), EF::from_u64(6)]);
        let eq_eval_1 = EF::from_u64(20);
        let eq_statement =
            EqStatement::new_hypercube(vec![eq_point_0, eq_point_1], vec![eq_eval_0, eq_eval_1]);

        // Create a select statement with 1 constraint
        let sel_var = F::from_u64(7);
        let sel_eval = EF::from_u64(30);
        let sel_statement = SelectStatement::new(num_variables, vec![sel_var], vec![sel_eval]);

        // Construct the combined constraint
        let constraint: Constraint<F, EF> = Constraint::new(challenge, eq_statement, sel_statement);

        // Verify that the constraint was constructed with correct fields
        assert_eq!(constraint.challenge, challenge);
        assert_eq!(constraint.eq_statement.len(), 2);
        assert_eq!(constraint.sel_statement.len(), 1);
        assert_eq!(constraint.num_variables(), num_variables);
    }

    #[test]
    #[should_panic(expected = "assertion failed")]
    fn test_constraint_new_mismatched_variables() {
        // Create statements with different numbers of variables

        // Equality statement with 3 variables
        let eq_point = Point::new(vec![EF::from_u64(1), EF::from_u64(2), EF::from_u64(3)]);
        let eq_eval = EF::from_u64(10);
        let eq_statement = EqStatement::new_hypercube(vec![eq_point], vec![eq_eval]);

        // Select statement with 2 variables (different!)
        let num_variables_sel = 2;
        let sel_var = F::from_u64(7);
        let sel_eval = EF::from_u64(30);
        let sel_statement = SelectStatement::new(num_variables_sel, vec![sel_var], vec![sel_eval]);

        // Random challenge
        let challenge = EF::from_u64(42);

        // This should panic due to mismatched variable counts
        let _constraint = Constraint::new(challenge, eq_statement, sel_statement);
    }

    #[test]
    fn test_constraint_new_eq_only() {
        // Declare test parameters explicitly

        // Number of variables
        let num_variables = 2;

        // Random challenge
        let challenge = EF::from_u64(99);

        // Create an equality statement with 3 constraints
        let eq_point_0 = Point::new(vec![EF::from_u64(1), EF::from_u64(2)]);
        let eq_eval_0 = EF::from_u64(10);
        let eq_point_1 = Point::new(vec![EF::from_u64(3), EF::from_u64(4)]);
        let eq_eval_1 = EF::from_u64(20);
        let eq_point_2 = Point::new(vec![EF::from_u64(5), EF::from_u64(6)]);
        let eq_eval_2 = EF::from_u64(30);
        let eq_statement = EqStatement::new_hypercube(
            vec![eq_point_0, eq_point_1, eq_point_2],
            vec![eq_eval_0, eq_eval_1, eq_eval_2],
        );

        // Create constraint with only equality constraints
        let constraint: Constraint<F, EF> = Constraint::new_eq_only(challenge, eq_statement);

        // Verify that the select statement is empty
        assert_eq!(constraint.sel_statement.len(), 0);
        assert!(constraint.sel_statement.is_empty());

        // Verify that the equality statement is present
        assert_eq!(constraint.eq_statement.len(), 3);

        // Verify that both statements have the same number of variables
        assert_eq!(constraint.num_variables(), num_variables);
        assert_eq!(constraint.sel_statement.num_variables(), num_variables);
    }

    #[test]
    fn test_constraint_num_variables() {
        // Declare test parameters explicitly

        // Number of variables (determines hypercube dimension)
        let num_variables = 4;

        // Random challenge
        let challenge = EF::from_u64(42);

        // Create empty statements with the specified number of variables
        let eq_statement = EqStatement::initialize(num_variables);
        let sel_statement = SelectStatement::initialize(num_variables);

        // Create constraint
        let constraint: Constraint<F, EF> = Constraint::new(challenge, eq_statement, sel_statement);

        // Verify that num_variables returns the correct value
        assert_eq!(constraint.num_variables(), num_variables);
    }

    #[test]
    fn test_constraint_combine_evals() {
        // Declare test parameters explicitly

        // Number of variables
        let num_variables = 2;

        // Random challenge for batching
        // We'll use γ = 2 for easy manual calculation
        let gamma = EF::from_u64(2);

        // Create equality statement with 2 constraints
        // Constraint 0: p(z_0) = 5, weighted by γ^0 = 1
        let eq_point_0 = Point::new(vec![EF::from_u64(1), EF::from_u64(1)]);
        let eq_eval_0 = EF::from_u64(5);

        // Constraint 1: p(z_1) = 7, weighted by γ^1 = 2
        let eq_point_1 = Point::new(vec![EF::from_u64(0), EF::from_u64(1)]);
        let eq_eval_1 = EF::from_u64(7);

        let eq_statement =
            EqStatement::new_hypercube(vec![eq_point_0, eq_point_1], vec![eq_eval_0, eq_eval_1]);

        // Create select statement with 1 constraint
        // Constraint 2: p(z_2) = 11, weighted by γ^2 = 4
        let sel_var = F::from_u64(3);
        let sel_eval = EF::from_u64(11);
        let sel_statement = SelectStatement::new(num_variables, vec![sel_var], vec![sel_eval]);

        // Create constraint
        let constraint: Constraint<F, EF> = Constraint::new(gamma, eq_statement, sel_statement);

        // Initialize accumulator
        let mut eval = EF::ZERO;

        // Combine evaluations
        constraint.combine_evals(&mut eval);

        // Expected result: 1*5 + 2*7 + 4*11 = 5 + 14 + 44 = 63
        let expected_eval = EF::from_u64(5)
            + EF::from_u64(2) * EF::from_u64(7)
            + EF::from_u64(4) * EF::from_u64(11);

        assert_eq!(eval, expected_eval);
        assert_eq!(eval, EF::from_u64(63));
    }

    #[test]
    fn test_constraint_combine_evals_accumulation() {
        // Test that combine_evals adds to existing values rather than overwriting

        // Random challenge
        let challenge = EF::from_u64(3);

        // Create a simple equality-only constraint
        let eq_point = Point::new(vec![EF::from_u64(1), EF::from_u64(1)]);
        let eq_eval = EF::from_u64(10);
        let eq_statement = EqStatement::new_hypercube(vec![eq_point], vec![eq_eval]);
        let constraint: Constraint<F, EF> = Constraint::new_eq_only(challenge, eq_statement);

        // Start with a non-zero accumulator
        let initial_value = EF::from_u64(100);
        let mut eval = initial_value;

        // Combine evaluations (should add to existing value)
        constraint.combine_evals(&mut eval);

        // Verify that the result is initial_value + (γ^0 * 10) = 100 + 10 = 110
        assert_eq!(eval, EF::from_u64(110));
    }

    #[test]
    fn test_constraint_combine_new() {
        // Declare test parameters explicitly

        // Number of variables (2 variables → 2^2 = 4 hypercube points)
        let num_variables = 2;

        // Random challenge
        let challenge = EF::from_u64(5);

        // Create a simple equality statement with 1 constraint
        let eq_point = Point::new(vec![EF::ONE, EF::ZERO]);
        let eq_eval = EF::from_u64(42);
        let eq_statement = EqStatement::new_hypercube(vec![eq_point], vec![eq_eval]);

        // Create constraint (eq-only for simplicity)
        let constraint: Constraint<F, EF> = Constraint::new_eq_only(challenge, eq_statement);

        // Combine into fresh accumulators
        let (combined, eval) = constraint.combine_new();

        // Verify that the combined weight polynomial has the correct size
        // Should have 2^num_variables = 4 entries
        assert_eq!(combined.num_evals(), 1 << num_variables);
        assert_eq!(combined.num_evals(), 4);

        // Verify that the expected evaluation equals γ^0 * 42 = 42
        assert_eq!(eval, EF::from_u64(42));

        // Verify that at least some entries in the weight polynomial are non-zero
        let non_zero_count = combined.iter().filter(|&&x| x != EF::ZERO).count();
        assert!(non_zero_count > 0);
    }

    #[test]
    fn test_constraint_combine_vs_combine_new() {
        // Verify that combine and combine_new produce the same results

        // Number of variables
        let num_variables = 2;

        // Random challenge
        let challenge = EF::from_u64(7);

        // Create a test constraint
        let eq_point = Point::new(vec![EF::ZERO, EF::ONE]);
        let eq_eval = EF::from_u64(15);
        let eq_statement = EqStatement::new_hypercube(vec![eq_point], vec![eq_eval]);
        let constraint: Constraint<F, EF> = Constraint::new_eq_only(challenge, eq_statement);

        // Method 1: Use combine_new
        let (combined_new, eval_new) = constraint.combine_new();

        // Method 2: Use combine with fresh accumulators
        let mut combined_manual = Poly::zero(num_variables);
        let mut eval_manual = EF::ZERO;
        constraint.combine(&mut combined_manual, &mut eval_manual);

        // Verify that both methods produce identical results
        assert_eq!(combined_new.num_evals(), combined_manual.num_evals());
        for (new_val, manual_val) in combined_new
            .as_slice()
            .iter()
            .zip(combined_manual.as_slice().iter())
        {
            assert_eq!(new_val, manual_val);
        }
        assert_eq!(eval_new, eval_manual);
    }

    #[test]
    fn test_constraint_iter_eqs() {
        // Test iteration over equality constraints with challenge weights

        // Number of variables
        let num_variables = 2;

        // Random challenge (γ = 3 for easy verification)
        let gamma = EF::from_u64(3);

        // Create equality statement with 3 constraints
        let eq_point_0 = Point::new(vec![EF::from_u64(1), EF::from_u64(2)]);
        let eq_eval_0 = EF::from_u64(10);

        let eq_point_1 = Point::new(vec![EF::from_u64(3), EF::from_u64(4)]);
        let eq_eval_1 = EF::from_u64(20);

        let eq_point_2 = Point::new(vec![EF::from_u64(5), EF::from_u64(6)]);
        let eq_eval_2 = EF::from_u64(30);

        let eq_statement = EqStatement::new_hypercube(
            vec![eq_point_0, eq_point_1, eq_point_2],
            vec![eq_eval_0, eq_eval_1, eq_eval_2],
        );

        // Create constraint
        let constraint: Constraint<F, EF> = Constraint::new_eq_only(gamma, eq_statement);

        // Collect iterator results
        let results: Vec<_> = constraint.iter_eqs().collect();

        // Verify that we have 3 pairs
        assert_eq!(results.len(), 3);

        // Verify challenge weights: γ^0 = 1, γ^1 = 3, γ^2 = 9
        let expected_weights = [EF::from_u64(1), EF::from_u64(3), EF::from_u64(9)];

        for (i, (point, coeff)) in results.iter().enumerate() {
            // Verify challenge weight
            assert_eq!(*coeff, expected_weights[i]);

            // Verify point reference matches original
            assert_eq!(point.num_vars(), num_variables);
        }
    }

    #[test]
    fn test_constraint_iter_sels() {
        // Test iteration over select constraints with challenge weights

        // Number of variables
        let num_variables = 2;

        // Random challenge (γ = 2 for easy verification)
        let gamma = EF::from_u64(2);

        // Create equality statement with 2 constraints
        // This will use challenge powers γ^0 and γ^1
        let eq_point_0 = Point::new(vec![EF::from_u64(1), EF::from_u64(2)]);
        let eq_eval_0 = EF::from_u64(10);
        let eq_point_1 = Point::new(vec![EF::from_u64(3), EF::from_u64(4)]);
        let eq_eval_1 = EF::from_u64(20);
        let eq_statement =
            EqStatement::new_hypercube(vec![eq_point_0, eq_point_1], vec![eq_eval_0, eq_eval_1]);

        // Create select statement with 2 constraints
        // These should use challenge powers γ^2 and γ^3
        let sel_var_0 = F::from_u64(5);
        let sel_eval_0 = EF::from_u64(30);
        let sel_var_1 = F::from_u64(6);
        let sel_eval_1 = EF::from_u64(40);
        let sel_statement = SelectStatement::new(
            num_variables,
            vec![sel_var_0, sel_var_1],
            vec![sel_eval_0, sel_eval_1],
        );

        // Create constraint
        let constraint: Constraint<F, EF> = Constraint::new(gamma, eq_statement, sel_statement);

        // Collect iterator results
        let results: Vec<_> = constraint.iter_sels().collect();

        // Verify that we have 2 pairs
        assert_eq!(results.len(), 2);

        // Verify challenge weights: γ^2 = 4, γ^3 = 8
        // (skipping the first 2 powers used by equality constraints)
        let expected_weights = [EF::from_u64(4), EF::from_u64(8)];
        let expected_vars = [sel_var_0, sel_var_1];

        for (i, (var, coeff)) in results.iter().enumerate() {
            // Verify challenge weight
            assert_eq!(*coeff, expected_weights[i]);

            // Verify variable reference matches original
            assert_eq!(**var, expected_vars[i]);
        }
    }

    #[test]
    fn test_constraint_iter_sels_empty() {
        // Test that iter_sels works correctly when there are no select constraints

        // Random challenge
        let challenge = EF::from_u64(7);

        // Create equality-only constraint
        let eq_point = Point::new(vec![EF::from_u64(1), EF::from_u64(2)]);
        let eq_eval = EF::from_u64(10);
        let eq_statement = EqStatement::new_hypercube(vec![eq_point], vec![eq_eval]);
        let constraint: Constraint<F, EF> = Constraint::new_eq_only(challenge, eq_statement);

        // Verify that the iterator is empty
        assert_eq!(constraint.iter_sels().count(), 0);
    }
}
