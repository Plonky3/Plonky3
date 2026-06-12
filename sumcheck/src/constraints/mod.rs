use alloc::vec::Vec;

use p3_field::{ExtensionField, Field, PackedValue};
use p3_multilinear_util::poly::Poly;
use p3_util::log2_strict_usize;

use crate::constraints::statement::{EqStatement, NextStatement, SelectStatement};

/// Statement types for polynomial evaluation constraints.
pub mod statement;

/// One explicitly ordered group of evaluation constraints.
#[derive(Clone, Debug)]
pub enum Statements<F: Field, EF: ExtensionField<F>> {
    /// Ordinary multilinear evaluations at concrete points.
    Eq(EqStatement<EF>),
    /// Slot-local repeat-last Next evaluations.
    Next(NextStatement<EF>),
    /// Selection-based evaluations.
    Select(SelectStatement<F, EF>),
}

impl<F: Field, EF: ExtensionField<F>> Statements<F, EF> {
    #[must_use]
    pub const fn num_variables(&self) -> usize {
        match self {
            Self::Eq(statement) => statement.num_variables(),
            Self::Next(statement) => statement.num_variables(),
            Self::Select(statement) => statement.num_variables(),
        }
    }

    #[must_use]
    pub const fn len(&self) -> usize {
        match self {
            Self::Eq(statement) => statement.len(),
            Self::Next(statement) => statement.len(),
            Self::Select(statement) => statement.len(),
        }
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        match self {
            Self::Eq(statement) => statement.is_empty(),
            Self::Next(statement) => statement.is_empty(),
            Self::Select(statement) => statement.is_empty(),
        }
    }

    fn combine_evals(&self, eval: &mut EF, challenge: EF, shift: usize) {
        match self {
            Self::Eq(statement) => statement.combine_evals(eval, challenge, shift),
            Self::Next(statement) => statement.combine_evals(eval, challenge, shift),
            Self::Select(statement) => statement.combine_evals(eval, challenge, shift),
        }
    }

    fn combine(
        &self,
        combined: &mut Poly<EF>,
        eval: &mut EF,
        challenge: EF,
        shift: usize,
        initialized: bool,
    ) -> bool {
        if self.is_empty() {
            return initialized;
        }

        match self {
            Self::Eq(statement) => {
                if initialized {
                    statement.combine_hypercube::<F, true>(combined, eval, challenge, shift);
                } else {
                    statement.combine_hypercube::<F, false>(combined, eval, challenge, shift);
                }
            }
            Self::Next(statement) => {
                statement.combine::<F>(combined, eval, challenge, shift);
            }
            Self::Select(statement) => {
                statement.combine(combined, eval, challenge, shift);
            }
        }

        true
    }

    fn combine_packed(
        &self,
        combined: &mut Poly<EF::ExtensionPacking>,
        eval: &mut EF,
        challenge: EF,
        shift: usize,
        initialized: bool,
    ) -> bool {
        if self.is_empty() {
            return initialized;
        }

        match self {
            Self::Eq(statement) => {
                if initialized {
                    statement.combine_hypercube_packed::<F, true>(combined, eval, challenge, shift);
                } else {
                    statement
                        .combine_hypercube_packed::<F, false>(combined, eval, challenge, shift);
                }
            }
            Self::Next(statement) => {
                statement.combine_packed::<F>(combined, eval, challenge, shift);
            }
            Self::Select(statement) => {
                statement.combine_packed(combined, eval, challenge, shift);
            }
        }

        true
    }
}

/// A combined ordered constraint system.
///
/// This struct represents a unified constraint system that combines:
/// - **Equality constraints**: Polynomial evaluations at specific points
/// - **Next constraints**: Repeat-last successor evaluations
/// - **Select constraints**: Selection-based polynomial evaluations
///
/// All statement groups are batched using powers of a random challenge `γ`.
/// Challenge powers advance by the number of constraints in each group, so the
/// order of `statements` is protocol-visible.
///
/// # Mathematical Structure
///
/// Given ordered statements `S_i`, the combined constraint polynomial is:
///
/// ```text
/// W(X) = Σ_i γ^i · weight_i(X)
/// ```
///
/// The combined expected evaluation is:
///
/// ```text
/// S = Σ_i γ^i · eval_i
/// ```
#[derive(Clone, Debug)]
pub struct Constraint<F: Field, EF: ExtensionField<F>> {
    /// Number of variables shared by every statement group.
    num_variables: usize,

    /// Statement groups in the exact alpha-power order used by batching.
    statements: Vec<Statements<F, EF>>,

    /// Random challenge `γ` used for batching constraints.
    ///
    /// Powers of this challenge weight each ordered constraint.
    challenge: EF,
}

impl<F: Field, EF: ExtensionField<F>> Constraint<F, EF> {
    /// Creates a new constraint from explicitly ordered statement groups.
    ///
    /// # Parameters
    ///
    /// - `challenge`: Random challenge `γ` for batching constraints
    /// - `num_variables`: Shared multilinear arity of every statement group
    /// - `statements`: Ordered statement groups in protocol-visible alpha order
    ///
    /// # Panics
    ///
    /// Panics if any statement group has a different number of variables.
    #[must_use]
    pub fn new(challenge: EF, num_variables: usize, statements: Vec<Statements<F, EF>>) -> Self {
        assert!(
            statements
                .iter()
                .all(|statement| statement.num_variables() == num_variables)
        );

        Self {
            num_variables,
            statements,
            challenge,
        }
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
        self.num_variables
    }

    /// Returns the ordered statement groups.
    #[must_use]
    pub fn statements(&self) -> &[Statements<F, EF>] {
        &self.statements
    }

    /// Returns the batching challenge `γ`.
    #[must_use]
    pub const fn challenge(&self) -> EF {
        self.challenge
    }

    /// Returns powers of the batching challenge starting at `γ^shift`.
    pub fn challenge_powers(&self, shift: usize) -> impl Iterator<Item = EF> {
        self.challenge
            .shifted_powers(self.challenge.exp_u64(shift as u64))
    }

    /// Combines expected evaluations using challenge powers.
    ///
    /// This accumulates the weighted sum of all expected constraint evaluations:
    /// ```text
    /// eval += Σ_i γ^i · s_i
    /// ```
    ///
    /// # Parameters
    ///
    /// - `eval`: Mutable accumulator for the combined expected evaluation
    ///
    /// # Implementation Notes
    ///
    /// Each statement group starts at the challenge power following the
    /// previous group's final constraint.
    pub fn combine_evals(&self, eval: &mut EF) {
        let mut shift = 0;
        for statement in &self.statements {
            statement.combine_evals(eval, self.challenge, shift);
            shift += statement.len();
        }
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
    /// combined[b] += Σ_i γ^i · weight_i(b)
    /// ```
    ///
    /// Updates `eval`:
    /// ```text
    /// eval += Σ_i γ^i · s_i
    /// ```
    pub fn combine(&self, combined: &mut Poly<EF>, eval: &mut EF) {
        let mut shift = 0;
        let mut initialized = true;
        for statement in &self.statements {
            initialized = statement.combine(combined, eval, self.challenge, shift, initialized);
            shift += statement.len();
        }
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
    /// combined[b] += Σ_i γ^i · weight_i(b)
    /// ```
    ///
    /// Updates `eval`:
    /// ```text
    /// eval += Σ_i γ^i · s_i
    /// ```
    pub fn combine_packed(&self, combined: &mut Poly<EF::ExtensionPacking>, eval: &mut EF) {
        let mut shift = 0;
        let mut initialized = true;
        for statement in &self.statements {
            initialized =
                statement.combine_packed(combined, eval, self.challenge, shift, initialized);
            shift += statement.len();
        }
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

        let mut shift = 0;
        let mut initialized = false;
        for statement in &self.statements {
            initialized =
                statement.combine(&mut combined, &mut eval, self.challenge, shift, initialized);
            shift += statement.len();
        }

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

        let mut shift = 0;
        let mut initialized = false;
        for statement in &self.statements {
            initialized = statement.combine_packed(
                &mut combined,
                &mut eval,
                self.challenge,
                shift,
                initialized,
            );
            shift += statement.len();
        }

        // Return the completed weight polynomial and expected evaluation.
        (combined, eval)
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;
    use p3_multilinear_util::point::Point;

    use super::*;
    use crate::strategy::VariableOrder;

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
        let constraint: Constraint<F, EF> = Constraint::new(
            challenge,
            num_variables,
            vec![
                Statements::Eq(eq_statement),
                Statements::Select(sel_statement),
            ],
        );

        // Verify that the constraint was constructed with correct fields
        assert_eq!(constraint.challenge, challenge);
        assert_eq!(constraint.statements.len(), 2);
        assert!(matches!(
            &constraint.statements[0],
            Statements::Eq(statement) if statement.len() == 2
        ));
        assert!(matches!(
            &constraint.statements[1],
            Statements::Select(statement) if statement.len() == 1
        ));
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
        let num_variables = eq_statement.num_variables();

        // This should panic due to mismatched variable counts
        let _constraint = Constraint::new(
            challenge,
            num_variables,
            vec![
                Statements::Eq(eq_statement),
                Statements::Select(sel_statement),
            ],
        );
    }

    #[test]
    fn test_constraint_new_with_single_eq_statement() {
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
        let constraint: Constraint<F, EF> =
            Constraint::new(challenge, num_variables, vec![Statements::Eq(eq_statement)]);

        // Verify that only the equality statement is present
        assert_eq!(constraint.statements.len(), 1);
        assert!(matches!(
            &constraint.statements[0],
            Statements::Eq(statement) if statement.len() == 3
        ));

        // Verify that both statements have the same number of variables
        assert_eq!(constraint.num_variables(), num_variables);
        assert_eq!(constraint.statements[0].num_variables(), num_variables);
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
        let constraint: Constraint<F, EF> = Constraint::new(
            challenge,
            num_variables,
            vec![
                Statements::Eq(eq_statement),
                Statements::Select(sel_statement),
            ],
        );

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
        let constraint: Constraint<F, EF> = Constraint::new(
            gamma,
            num_variables,
            vec![
                Statements::Eq(eq_statement),
                Statements::Select(sel_statement),
            ],
        );

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
        let constraint: Constraint<F, EF> =
            Constraint::new(challenge, 2, vec![Statements::Eq(eq_statement)]);

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
        let constraint: Constraint<F, EF> =
            Constraint::new(challenge, num_variables, vec![Statements::Eq(eq_statement)]);

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
        let constraint: Constraint<F, EF> =
            Constraint::new(challenge, num_variables, vec![Statements::Eq(eq_statement)]);

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
    fn test_constraint_statements_preserve_explicit_order() {
        let num_variables = 2;
        let gamma = EF::from_u64(2);

        let eq_statement = EqStatement::new_hypercube(
            vec![
                Point::new(vec![EF::from_u64(1), EF::from_u64(2)]),
                Point::new(vec![EF::from_u64(3), EF::from_u64(4)]),
            ],
            vec![EF::from_u64(10), EF::from_u64(20)],
        );
        let sel_statement = SelectStatement::new(
            num_variables,
            vec![F::from_u64(5), F::from_u64(6)],
            vec![EF::from_u64(30), EF::from_u64(40)],
        );

        let constraint: Constraint<F, EF> = Constraint::new(
            gamma,
            num_variables,
            vec![
                Statements::Eq(eq_statement),
                Statements::Select(sel_statement),
            ],
        );

        assert_eq!(constraint.statements().len(), 2);
        assert!(matches!(
            &constraint.statements()[0],
            Statements::Eq(statement) if statement.len() == 2
        ));
        assert!(matches!(
            &constraint.statements()[1],
            Statements::Select(statement) if statement.len() == 2
        ));
    }

    #[test]
    fn test_constraint_combine_shifts_later_eq_groups() {
        let num_variables = 4;
        let gamma = EF::from_u64(3);

        let eq_point_0 = Point::new(vec![
            EF::from_u64(1),
            EF::from_u64(2),
            EF::from_u64(3),
            EF::from_u64(4),
        ]);
        let eq_eval_0 = EF::from_u64(5);
        let eq_0 = EqStatement::new_hypercube(vec![eq_point_0.clone()], vec![eq_eval_0]);

        let next_point = Point::new(vec![
            EF::from_u64(6),
            EF::from_u64(7),
            EF::from_u64(8),
            EF::from_u64(9),
        ]);
        let next_eval = EF::from_u64(10);
        let mut next = NextStatement::initialize(num_variables);
        next.add_evaluated_constraint(
            Point::new(Vec::new()),
            next_point.clone(),
            next_eval,
            VariableOrder::Prefix,
        );

        let eq_point_1 = Point::new(vec![
            EF::from_u64(11),
            EF::from_u64(12),
            EF::from_u64(13),
            EF::from_u64(14),
        ]);
        let eq_eval_1 = EF::from_u64(15);
        let eq_1 = EqStatement::new_hypercube(vec![eq_point_1.clone()], vec![eq_eval_1]);

        let constraint: Constraint<F, EF> = Constraint::new(
            gamma,
            num_variables,
            vec![
                Statements::Eq(eq_0),
                Statements::Next(next),
                Statements::Eq(eq_1),
            ],
        );

        let (combined, eval) = constraint.combine_new();

        let mut expected = Poly::new_from_point(eq_point_0.as_slice(), EF::ONE);
        expected
            .as_mut_slice()
            .iter_mut()
            .zip(Poly::new_next_from_point(next_point.as_slice()).iter())
            .for_each(|(out, &weight)| *out += gamma * weight);
        expected
            .as_mut_slice()
            .iter_mut()
            .zip(Poly::new_from_point(eq_point_1.as_slice(), gamma.square()).iter())
            .for_each(|(out, &weight)| *out += weight);

        assert_eq!(combined.as_slice(), expected.as_slice());
        assert_eq!(
            eval,
            eq_eval_0 + gamma * next_eval + gamma.square() * eq_eval_1
        );
    }
}
