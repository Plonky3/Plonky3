use alloc::vec::Vec;

use p3_field::{ExtensionField, Field, PackedValue};
use p3_multilinear_util::poly::Poly;
use p3_util::log2_strict_usize;

use crate::constraints::statement::{EqStatement, NextStatement, SelectStatement};

/// Statement types for polynomial evaluation constraints.
pub mod statement;

/// One explicitly ordered group of evaluation constraints.
///
/// # Overview
///
/// The combiner treats each group as one unit and advances the challenge power by the group's constraint count.
/// The wrapped kind decides how the group's weight polynomial is built.
#[derive(Clone, Debug)]
pub enum Statements<F: Field, EF: ExtensionField<F>> {
    /// Plain multilinear evaluations at concrete points.
    Eq(EqStatement<EF>),
    /// Slot-local repeat-last successor evaluations.
    Next(NextStatement<EF>),
    /// Selection-based evaluations through the power-map expansion.
    Select(SelectStatement<F, EF>),
}

impl<F: Field, EF: ExtensionField<F>> Statements<F, EF> {
    /// Returns the multilinear arity of the wrapped group.
    #[must_use]
    pub const fn num_variables(&self) -> usize {
        // Forward to the wrapped statement regardless of kind.
        match self {
            Self::Eq(statement) => statement.num_variables(),
            Self::Next(statement) => statement.num_variables(),
            Self::Select(statement) => statement.num_variables(),
        }
    }

    /// Returns the number of constraints in the wrapped group.
    ///
    /// # Why this matters
    ///
    /// The combiner advances the challenge power by this count, so it sets how many powers the group consumes.
    #[must_use]
    pub const fn len(&self) -> usize {
        // Forward to the wrapped statement regardless of kind.
        match self {
            Self::Eq(statement) => statement.len(),
            Self::Next(statement) => statement.len(),
            Self::Select(statement) => statement.len(),
        }
    }

    /// Returns true when the wrapped group holds no constraints.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        // Forward to the wrapped statement regardless of kind.
        match self {
            Self::Eq(statement) => statement.is_empty(),
            Self::Next(statement) => statement.is_empty(),
            Self::Select(statement) => statement.is_empty(),
        }
    }

    /// Accumulates the wrapped group's challenge-weighted expected sum.
    ///
    /// # Arguments
    ///
    /// - `eval`: scalar accumulator updated in place.
    /// - `challenge`: the batching challenge whose powers weight each constraint.
    /// - `shift`: offset of the first challenge power for this group.
    fn combine_evals(&self, eval: &mut EF, challenge: EF, shift: usize) {
        // Forward to the wrapped statement; each kind weights its own claimed values.
        match self {
            Self::Eq(statement) => statement.combine_evals(eval, challenge, shift),
            Self::Next(statement) => statement.combine_evals(eval, challenge, shift),
            Self::Select(statement) => statement.combine_evals(eval, challenge, shift),
        }
    }

    /// Folds the wrapped group into a dense weight polynomial and an expected sum.
    ///
    /// # Arguments
    ///
    /// - `combined`: dense weight polynomial accumulator.
    /// - `eval`: scalar accumulator for the expected sum.
    /// - `challenge`: the batching challenge whose powers weight each constraint.
    /// - `shift`: offset of the first challenge power for this group.
    /// - `initialized`: true once an earlier group has written the accumulator.
    ///
    /// # Returns
    ///
    /// Whether the accumulator now holds data, which is true unless the group was empty and stayed so.
    ///
    /// # Why the flag
    ///
    /// The first nonempty group may overwrite the accumulator for speed, while later groups must add onto it.
    /// The flag carries that decision forward across groups.
    fn combine(
        &self,
        combined: &mut Poly<EF>,
        eval: &mut EF,
        challenge: EF,
        shift: usize,
        initialized: bool,
    ) -> bool {
        // An empty group writes nothing, so the prior accumulator state is unchanged.
        if self.is_empty() {
            return initialized;
        }

        // Dispatch on kind; the equality path can overwrite when nothing has been written yet.
        match self {
            Self::Eq(statement) => {
                // Add onto existing data, or overwrite when this is the first writer.
                if initialized {
                    statement.combine_hypercube::<F, true>(combined, eval, challenge, shift);
                } else {
                    statement.combine_hypercube::<F, false>(combined, eval, challenge, shift);
                }
            }
            Self::Next(statement) => {
                // Successor weights always add onto the accumulator.
                statement.combine(combined, eval, challenge, shift);
            }
            Self::Select(statement) => {
                // Selection weights always add onto the accumulator.
                statement.combine(combined, eval, challenge, shift);
            }
        }

        // A nonempty group has now written, so the accumulator is initialized for later groups.
        true
    }

    /// SIMD-packed variant that folds the wrapped group into a packed weight polynomial and an expected sum.
    ///
    /// # Arguments
    ///
    /// - `combined`: packed weight polynomial accumulator.
    /// - `eval`: scalar accumulator for the expected sum.
    /// - `challenge`: the batching challenge whose powers weight each constraint.
    /// - `shift`: offset of the first challenge power for this group.
    /// - `initialized`: true once an earlier group has written the accumulator.
    ///
    /// # Returns
    ///
    /// Whether the accumulator now holds data, which is true unless the group was empty and stayed so.
    fn combine_packed(
        &self,
        combined: &mut Poly<EF::ExtensionPacking>,
        eval: &mut EF,
        challenge: EF,
        shift: usize,
        initialized: bool,
    ) -> bool {
        // An empty group writes nothing, so the prior accumulator state is unchanged.
        if self.is_empty() {
            return initialized;
        }

        // Dispatch on kind; the equality path can overwrite when nothing has been written yet.
        match self {
            Self::Eq(statement) => {
                // Add onto existing data, or overwrite when this is the first writer.
                if initialized {
                    statement.combine_hypercube_packed::<F, true>(combined, eval, challenge, shift);
                } else {
                    statement
                        .combine_hypercube_packed::<F, false>(combined, eval, challenge, shift);
                }
            }
            Self::Next(statement) => {
                // Successor weights always add onto the accumulator.
                statement.combine_packed::<F>(combined, eval, challenge, shift);
            }
            Self::Select(statement) => {
                // Selection weights always add onto the accumulator.
                statement.combine_packed(combined, eval, challenge, shift);
            }
        }

        // A nonempty group has now written, so the accumulator is initialized for later groups.
        true
    }
}

/// An ordered batch of evaluation constraint groups folded under one challenge.
///
/// # Overview
///
/// Every group shares the same variable space and contributes a weight polynomial and an expected value.
/// The groups are combined with powers of a single challenge, advancing by each group's constraint count.
///
/// # Why order matters
///
/// The challenge power index advances by the size of each group, so the group order fixes which power each constraint receives.
/// The prover and the verifier must walk the groups in the same order to agree.
///
/// # Algorithm
///
/// For groups indexed by `i`, the combined weight polynomial and expected value are sums weighted by challenge powers.
///
/// ```text
/// W(X) = sum_i gamma^i * weight_i(X)
/// S    = sum_i gamma^i * eval_i
/// ```
#[derive(Clone, Debug)]
pub struct Constraint<F: Field, EF: ExtensionField<F>> {
    /// Number of variables shared by every statement group.
    num_variables: usize,

    /// Statement groups in the exact challenge-power order used by batching.
    statements: Vec<Statements<F, EF>>,

    /// Batching challenge whose powers weight each constraint.
    challenge: EF,
}

impl<F: Field, EF: ExtensionField<F>> Constraint<F, EF> {
    /// Builds a batch from explicitly ordered statement groups.
    ///
    /// # Arguments
    ///
    /// - `challenge`: the batching challenge whose powers weight the groups.
    /// - `num_variables`: shared multilinear arity that every group must match.
    /// - `statements`: groups in the protocol-visible order used by batching.
    ///
    /// # Fiat-Shamir
    ///
    /// - The caller owns the binding of `challenge`; this type does not absorb anything.
    /// - Every statement's points and evaluations must be observed before `challenge` is sampled.
    /// - A `challenge` drawn before that binding lets a prover steer it and forge the batch.
    ///
    /// # Panics
    ///
    /// Panics if any group has a different variable count than the shared arity.
    #[must_use]
    pub fn new(challenge: EF, num_variables: usize, statements: Vec<Statements<F, EF>>) -> Self {
        // Every group must live in the same variable space, or batching is ill-defined.
        assert!(
            statements
                .iter()
                .all(|statement| statement.num_variables() == num_variables)
        );

        // Store the groups in caller order; that order fixes the challenge powers.
        Self {
            num_variables,
            statements,
            challenge,
        }
    }

    /// Returns the shared number of variables.
    ///
    /// # Returns
    ///
    /// The dimension of the Boolean hypercube over which every group is evaluated.
    #[must_use]
    pub const fn num_variables(&self) -> usize {
        self.num_variables
    }

    /// Returns the statement groups in batching order.
    #[must_use]
    pub fn statements(&self) -> &[Statements<F, EF>] {
        &self.statements
    }

    /// Returns the challenge powers starting at the given offset.
    ///
    /// # Arguments
    ///
    /// - `shift`: exponent of the first power yielded.
    pub fn challenge_powers(&self, shift: usize) -> impl Iterator<Item = EF> {
        // Seed the power sequence at the challenge raised to the requested offset.
        self.challenge
            .shifted_powers(self.challenge.exp_u64(shift as u64))
    }

    /// Accumulates the challenge-weighted expected value across all groups.
    ///
    /// # Arguments
    ///
    /// - `eval`: scalar accumulator updated in place.
    ///
    /// # Algorithm
    ///
    /// Each group starts at the power just after the previous group's last constraint.
    pub fn combine_evals(&self, eval: &mut EF) {
        // Running exponent of the next challenge power to assign.
        let mut shift = 0;
        for statement in &self.statements {
            // Fold this group's expected values starting at the current exponent.
            statement.combine_evals(eval, self.challenge, shift);
            // Advance past this group's constraints so the next group's powers stay disjoint.
            shift += statement.len();
        }
    }

    /// Adds the batched weight polynomial and expected value onto existing accumulators.
    ///
    /// # Overview
    ///
    /// Every hypercube entry gains the challenge-weighted sum of each group's weight at that entry.
    /// The scalar accumulator gains the matching challenge-weighted sum of expected values.
    ///
    /// # Arguments
    ///
    /// - `combined`: weight polynomial accumulator over the hypercube.
    /// - `eval`: scalar accumulator for the expected value.
    pub fn combine(&self, combined: &mut Poly<EF>, eval: &mut EF) {
        // Running exponent of the next challenge power to assign.
        let mut shift = 0;
        // Treat the incoming accumulator as already holding data, so every group adds onto it.
        let mut initialized = true;
        for statement in &self.statements {
            // Fold this group; the returned flag stays true once anything has been written.
            initialized = statement.combine(combined, eval, self.challenge, shift, initialized);
            // Advance past this group's constraints to keep the next group's powers disjoint.
            shift += statement.len();
        }
    }

    /// SIMD-packed variant that adds the batched weight polynomial and expected value onto existing accumulators.
    ///
    /// # Arguments
    ///
    /// - `combined`: packed weight polynomial accumulator over the hypercube.
    /// - `eval`: scalar accumulator for the expected value.
    pub fn combine_packed(&self, combined: &mut Poly<EF::ExtensionPacking>, eval: &mut EF) {
        // Running exponent of the next challenge power to assign.
        let mut shift = 0;
        // Treat the incoming accumulator as already holding data, so every group adds onto it.
        let mut initialized = true;
        for statement in &self.statements {
            // Fold this group; the returned flag stays true once anything has been written.
            initialized =
                statement.combine_packed(combined, eval, self.challenge, shift, initialized);
            // Advance past this group's constraints to keep the next group's powers disjoint.
            shift += statement.len();
        }
    }

    /// Builds the batched weight polynomial and expected value in fresh accumulators.
    ///
    /// # Returns
    ///
    /// A pair holding the weight polynomial over the hypercube and the expected value scalar.
    pub fn combine_new(&self) -> (Poly<EF>, EF) {
        // Fresh weight accumulator: one entry per hypercube point, all zero.
        let mut combined = Poly::zero(self.num_variables());
        // Fresh scalar accumulator for the expected value.
        let mut eval = EF::ZERO;

        // Running exponent of the next challenge power to assign.
        let mut shift = 0;
        // The accumulator starts empty, so the first nonempty group may overwrite instead of add.
        let mut initialized = false;
        for statement in &self.statements {
            // Fold this group; the returned flag flips true once anything has been written.
            initialized =
                statement.combine(&mut combined, &mut eval, self.challenge, shift, initialized);
            // Advance past this group's constraints to keep the next group's powers disjoint.
            shift += statement.len();
        }

        (combined, eval)
    }

    /// SIMD-packed variant that builds the batched weight polynomial and expected value in fresh accumulators.
    ///
    /// # Returns
    ///
    /// A pair holding the packed weight polynomial over the hypercube and the expected value scalar.
    pub fn combine_new_packed(&self) -> (Poly<EF::ExtensionPacking>, EF) {
        // Number of variables collapsed into each packed lane.
        let k_pack = log2_strict_usize(F::Packing::WIDTH);
        let k = self.num_variables();

        // Fresh packed weight accumulator: the lane count absorbs the low variables.
        let mut combined = Poly::zero(k - k_pack);
        // Fresh scalar accumulator for the expected value.
        let mut eval = EF::ZERO;

        // Running exponent of the next challenge power to assign.
        let mut shift = 0;
        // The accumulator starts empty, so the first nonempty group may overwrite instead of add.
        let mut initialized = false;
        for statement in &self.statements {
            // Fold this group; the returned flag flips true once anything has been written.
            initialized = statement.combine_packed(
                &mut combined,
                &mut eval,
                self.challenge,
                shift,
                initialized,
            );
            // Advance past this group's constraints to keep the next group's powers disjoint.
            shift += statement.len();
        }

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
        // Invariant: groups are stored in caller order, since that order fixes the challenge powers.
        let num_variables = 2;
        let gamma = EF::from_u64(2);

        // Fixture state: one equality group of 2 constraints.
        let eq_statement = EqStatement::new_hypercube(
            vec![
                Point::new(vec![EF::from_u64(1), EF::from_u64(2)]),
                Point::new(vec![EF::from_u64(3), EF::from_u64(4)]),
            ],
            vec![EF::from_u64(10), EF::from_u64(20)],
        );
        // Fixture state: one selection group of 2 constraints.
        let sel_statement = SelectStatement::new(
            num_variables,
            vec![F::from_u64(5), F::from_u64(6)],
            vec![EF::from_u64(30), EF::from_u64(40)],
        );

        // Build the batch with equality first, then selection.
        //
        //     index 0: equality (2 constraints)
        //     index 1: selection (2 constraints)
        let constraint: Constraint<F, EF> = Constraint::new(
            gamma,
            num_variables,
            vec![
                Statements::Eq(eq_statement),
                Statements::Select(sel_statement),
            ],
        );

        // Both groups survive in the exact order supplied.
        assert_eq!(constraint.statements().len(), 2);
        // Slot 0 is the equality group with its 2 constraints.
        assert!(matches!(
            &constraint.statements()[0],
            Statements::Eq(statement) if statement.len() == 2
        ));
        // Slot 1 is the selection group with its 2 constraints.
        assert!(matches!(
            &constraint.statements()[1],
            Statements::Select(statement) if statement.len() == 2
        ));
    }

    #[test]
    fn test_constraint_combine_shifts_later_eq_groups() {
        // Invariant: each group's challenge power continues where the previous group's powers ended.
        //
        // Fixture state: three single-constraint groups, so powers run 0, 1, 2.
        //
        //     group 0: equality        -> weight gamma^0 = 1
        //     group 1: successor       -> weight gamma^1
        //     group 2: equality        -> weight gamma^2
        let num_variables = 4;
        let gamma = EF::from_u64(3);

        // Group 0: a single equality constraint over 4 variables.
        let eq_point_0 = Point::new(vec![
            EF::from_u64(1),
            EF::from_u64(2),
            EF::from_u64(3),
            EF::from_u64(4),
        ]);
        let eq_eval_0 = EF::from_u64(5);
        let eq_0 = EqStatement::new_hypercube(vec![eq_point_0.clone()], vec![eq_eval_0]);

        // Group 1: a single full-space successor constraint (empty selector).
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

        // Group 2: a single equality constraint, which must land on gamma^2.
        let eq_point_1 = Point::new(vec![
            EF::from_u64(11),
            EF::from_u64(12),
            EF::from_u64(13),
            EF::from_u64(14),
        ]);
        let eq_eval_1 = EF::from_u64(15);
        let eq_1 = EqStatement::new_hypercube(vec![eq_point_1.clone()], vec![eq_eval_1]);

        // Batch the three groups in this exact order.
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

        // Rebuild the expected weight polynomial by hand, applying the same powers.
        // First group at gamma^0: plain equality weight.
        let mut expected = Poly::new_from_point(eq_point_0.as_slice(), EF::ONE);
        // Second group at gamma^1: successor weight scaled by gamma.
        expected
            .as_mut_slice()
            .iter_mut()
            .zip(Poly::new_next_from_point(next_point.as_slice()).iter())
            .for_each(|(out, &weight)| *out += gamma * weight);
        // Third group at gamma^2: equality weight scaled by gamma squared.
        expected
            .as_mut_slice()
            .iter_mut()
            .zip(Poly::new_from_point(eq_point_1.as_slice(), gamma.square()).iter())
            .for_each(|(out, &weight)| *out += weight);

        // The combined weight polynomial matches the hand-built one entry for entry.
        assert_eq!(combined.as_slice(), expected.as_slice());
        // The scalar side mirrors the same per-group powers on the claimed values.
        assert_eq!(
            eval,
            eq_eval_0 + gamma * next_eval + gamma.square() * eq_eval_1
        );
    }
}
