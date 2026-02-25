use alloc::vec;
use alloc::vec::Vec;
use core::ops::{Add, Mul, Sub};

use p3_field::{Algebra, ExtensionField, Field, PrimeCharacteristicRing};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;

use crate::lookup::{Kind, Lookup, LookupData, LookupEvaluator, LookupInput};

/// The underlying structure of an AIR.
pub trait BaseAir<F>: Sync {
    /// The number of columns (a.k.a. registers) in this AIR.
    fn width(&self) -> usize;
    /// Return an optional preprocessed trace matrix to be included in the prover's trace.
    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        None
    }

    /// Whether this AIR's constraints use the "next" row of preprocessed columns.
    ///
    /// By default this returns `true`, which will require opening preprocessed columns
    /// at both `zeta` and `zeta_next`.
    ///
    /// AIRs that only ever read the current preprocessed row (and never access an
    /// offset-1 preprocessed entry) can override this to return `false` to allow
    /// the prover and verifier to open only at `zeta`.
    fn preprocessed_uses_next_row(&self) -> bool {
        true
    }

    /// Whether this AIR's constraints access the next row of the main trace.
    ///
    /// By default this returns `true`, which will require opening main columns
    /// at both `zeta` and `zeta_next`.
    ///
    /// AIRs that only ever read the current main row (and never access an
    /// offset-1 main entry) can override this to return `false` to allow
    /// the prover and verifier to open only at `zeta`.
    ///
    /// # When to override
    ///
    /// - **Return `false`**: single-row AIRs where all constraints are
    ///   evaluated within one row.
    /// - **Keep `true`** (default): AIRs with transition constraints
    ///   that reference `main.row_slice(1)`.
    ///
    /// # Correctness
    ///
    /// Must be consistent with [`Air::eval`]. Returning `false` when the AIR
    /// actually reads the next row will cause verification failures or, in
    /// the worst case, a soundness gap.
    fn main_uses_next_row(&self) -> bool {
        true
    }

    /// Optional hint for the number of constraints in this AIR.
    ///
    /// Normally the prover runs a full symbolic evaluation just to count
    /// constraints. Overriding this method lets the prover skip that pass.
    ///
    /// The count must cover every constraint asserted during evaluation,
    /// including both transition and boundary constraints. It must **not**
    /// include lookup or permutation constraints, which are counted
    /// separately.
    ///
    /// # Correctness
    ///
    /// The returned value **must** exactly match the actual number of
    /// constraints. A wrong count will cause the prover to panic or
    /// produce an invalid proof.
    ///
    /// Returns `None` by default, which falls back to symbolic evaluation.
    fn num_constraints(&self) -> Option<usize> {
        None
    }

    /// Optional hint for the maximum constraint degree in this AIR.
    ///
    /// The constraint degree is the factor by which trace length N
    /// scales the constraint polynomial degree.
    ///
    /// For example, a constraint `x * y * z` where x, y, z are trace
    /// variables has degree multiple 3.
    ///
    /// Normally the prover runs a full symbolic evaluation to compute this.
    /// Overriding this method lets both the prover and verifier skip that
    /// pass when only the degree (not the full constraint list) is needed.
    ///
    /// The value must be an upper bound on the degree multiple of every
    /// constraint (base and extension). It does not need to be tight, but
    /// overestimating wastes prover work (larger quotient domain).
    ///
    /// # Correctness
    ///
    /// The returned value **must** be >= the actual max constraint degree.
    /// A value that is too small will cause the prover to produce an
    /// invalid proof.
    ///
    /// Returns `None` by default, which falls back to symbolic evaluation.
    fn max_constraint_degree(&self) -> Option<usize> {
        None
    }
}

/// An extension of `BaseAir` that includes support for public values.
pub trait BaseAirWithPublicValues<F>: BaseAir<F> {
    /// Return the number of expected public values.
    fn num_public_values(&self) -> usize {
        0
    }
}

/// An algebraic intermediate representation (AIR) definition.
///
/// Contains an evaluation function for computing the constraints of the AIR.
/// This function can be applied to an evaluation trace in which case each
/// constraint will compute a particular value or it can be applied symbolically
/// with each constraint computing a symbolic expression.
pub trait Air<AB: AirBuilder>: BaseAir<AB::F> {
    /// Update the number of auxiliary columns to account for a new lookup column,
    /// and return its index (or indices).
    ///
    /// Default implementation returns an empty vector, indicating no lookup columns.
    /// Override this method for AIRs that use lookups.
    fn add_lookup_columns(&mut self) -> Vec<usize> {
        vec![]
    }

    /// Register all lookups for the current AIR and return them.
    ///
    /// Default implementation returns an empty vector, indicating no lookups.
    /// Override this method for AIRs that use lookups.
    fn get_lookups(&mut self) -> Vec<Lookup<AB::F>>
    where
        AB: PermutationAirBuilder + AirBuilderWithPublicValues,
    {
        vec![]
    }

    /// Register a lookup to be used in this AIR.
    /// This method can be used before proving or verifying, as the resulting
    /// data is shared between the prover and the verifier.
    fn register_lookup(&mut self, kind: Kind, lookup_inputs: &[LookupInput<AB::F>]) -> Lookup<AB::F>
    where
        AB: PermutationAirBuilder + AirBuilderWithPublicValues,
    {
        let (element_exprs, multiplicities_exprs) = lookup_inputs
            .iter()
            .map(|(elems, mult, dir)| {
                let multiplicity = dir.multiplicity(mult.clone());
                (elems.clone(), multiplicity)
            })
            .unzip();

        Lookup {
            kind,
            element_exprs,
            multiplicities_exprs,
            columns: self.add_lookup_columns(),
        }
    }

    /// Evaluate all AIR constraints using the provided builder.
    ///
    /// The builder provides both the trace on which the constraints
    /// are evaluated on as well as the method of accumulating the
    /// constraint evaluations.
    ///
    /// **Note**: Users do not need to specify lookup constraints evaluation in this method,
    /// but instead only specify the AIR constraints and rely on `eval_with_lookups` to evaluate
    /// both AIR and lookup constraints.
    ///
    /// # Arguments
    /// - `builder`: Mutable reference to an `AirBuilder` for defining constraints.
    fn eval(&self, builder: &mut AB);

    /// Evaluate all AIR and lookup constraints using the provided builder.
    ///
    /// The default implementation calls `eval` and then evaluates lookups if any are provided,
    /// using the provided lookup evaluator.
    /// Users typically don't need to override this method unless they need a custom behavior.
    ///
    /// # Arguments
    /// - `builder`: Mutable reference to an `AirBuilder` for defining constraints.
    /// - `lookups`: References to the lookups to be evaluated.
    /// - `lookup_data`: References to the lookup data to be used for evaluation.
    /// - `lookup_evaluator`: Reference to the lookup evaluator to be used for evaluation.
    fn eval_with_lookups<LE: LookupEvaluator>(
        &self,
        builder: &mut AB,
        lookups: &[Lookup<AB::F>],
        lookup_data: &[LookupData<AB::ExprEF>],
        lookup_evaluator: &LE,
    ) where
        AB: PermutationAirBuilder + AirBuilderWithPublicValues,
    {
        self.eval(builder);

        if !lookups.is_empty() {
            lookup_evaluator.eval_lookups(builder, lookups, lookup_data);
        }
    }
}

/// A builder which contains both a trace on which AIR constraints can be evaluated as well as a method of accumulating the AIR constraint evaluations.
///
/// Supports both symbolic cases where the constraints are treated as polynomials and collected into a vector
/// as well cases where the constraints are evaluated on an evaluation trace and combined using randomness.
pub trait AirBuilder: Sized {
    /// Underlying field type.
    ///
    /// This should usually implement `Field` but there are a few edge cases (mostly involving `PackedFields`) where
    /// it may only implement `PrimeCharacteristicRing`.
    type F: PrimeCharacteristicRing + Sync;

    /// Serves as the output type for an AIR constraint evaluation.
    type Expr: Algebra<Self::F> + Algebra<Self::Var>;

    /// The type of the variable appearing in the trace matrix.
    ///
    /// Serves as the input type for an AIR constraint evaluation.
    type Var: Into<Self::Expr>
        + Clone
        + Send
        + Sync
        + Add<Self::F, Output = Self::Expr>
        + Add<Self::Var, Output = Self::Expr>
        + Add<Self::Expr, Output = Self::Expr>
        + Sub<Self::F, Output = Self::Expr>
        + Sub<Self::Var, Output = Self::Expr>
        + Sub<Self::Expr, Output = Self::Expr>
        + Mul<Self::F, Output = Self::Expr>
        + Mul<Self::Var, Output = Self::Expr>
        + Mul<Self::Expr, Output = Self::Expr>;

    /// Matrix type holding variables.
    type M: Matrix<Self::Var>;

    /// Return the matrix representing the main (primary) trace registers.
    fn main(&self) -> Self::M;

    /// Return an optional matrix of preprocessed registers.
    /// The default implementation returns `None`.
    /// Override this for builders that provide preprocessed columns.
    fn preprocessed(&self) -> Option<Self::M> {
        None
    }

    /// Expression evaluating to 1 on the first row, 0 elsewhere.
    fn is_first_row(&self) -> Self::Expr;

    /// Expression evaluating to 1 on the last row, 0 elsewhere.
    fn is_last_row(&self) -> Self::Expr;

    /// Expression evaluating to 1 on all transition rows (not last row), 0 on last row.
    fn is_transition(&self) -> Self::Expr {
        self.is_transition_window(2)
    }

    /// Expression evaluating to 1 on rows except the last `size - 1` rows, 0 otherwise.
    fn is_transition_window(&self, size: usize) -> Self::Expr;

    /// Returns a sub-builder whose constraints are enforced only when `condition` is nonzero.
    fn when<I: Into<Self::Expr>>(&mut self, condition: I) -> FilteredAirBuilder<'_, Self> {
        FilteredAirBuilder {
            inner: self,
            condition: condition.into(),
        }
    }

    /// Returns a sub-builder whose constraints are enforced only when `x != y`.
    fn when_ne<I1: Into<Self::Expr>, I2: Into<Self::Expr>>(
        &mut self,
        x: I1,
        y: I2,
    ) -> FilteredAirBuilder<'_, Self> {
        self.when(x.into() - y.into())
    }

    /// Returns a sub-builder whose constraints are enforced only on the first row.
    fn when_first_row(&mut self) -> FilteredAirBuilder<'_, Self> {
        self.when(self.is_first_row())
    }

    /// Returns a sub-builder whose constraints are enforced only on the last row.
    fn when_last_row(&mut self) -> FilteredAirBuilder<'_, Self> {
        self.when(self.is_last_row())
    }

    /// Returns a sub-builder whose constraints are enforced on all rows except the last.
    fn when_transition(&mut self) -> FilteredAirBuilder<'_, Self> {
        self.when(self.is_transition())
    }

    /// Returns a sub-builder whose constraints are enforced on all rows except the last `size - 1`.
    fn when_transition_window(&mut self, size: usize) -> FilteredAirBuilder<'_, Self> {
        self.when(self.is_transition_window(size))
    }

    /// Assert that the given element is zero.
    ///
    /// Where possible, batching multiple assert_zero calls
    /// into a single assert_zeros call will improve performance.
    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I);

    /// Assert that every element of a given array is 0.
    ///
    /// This should be preferred over calling `assert_zero` multiple times.
    fn assert_zeros<const N: usize, I: Into<Self::Expr>>(&mut self, array: [I; N]) {
        for elem in array {
            self.assert_zero(elem);
        }
    }

    /// Assert that a given array consists of only boolean values.
    fn assert_bools<const N: usize, I: Into<Self::Expr>>(&mut self, array: [I; N]) {
        let zero_array = array.map(|x| x.into().bool_check());
        self.assert_zeros(zero_array);
    }

    /// Assert that `x` element is equal to `1`.
    fn assert_one<I: Into<Self::Expr>>(&mut self, x: I) {
        self.assert_zero(x.into() - Self::Expr::ONE);
    }

    /// Assert that the given elements are equal.
    fn assert_eq<I1: Into<Self::Expr>, I2: Into<Self::Expr>>(&mut self, x: I1, y: I2) {
        self.assert_zero(x.into() - y.into());
    }

    /// Assert that `x` is a boolean, i.e. either `0` or `1`.
    ///
    /// Where possible, batching multiple assert_bool calls
    /// into a single assert_bools call will improve performance.
    fn assert_bool<I: Into<Self::Expr>>(&mut self, x: I) {
        self.assert_zero(x.into().bool_check());
    }
}

/// Extension trait for `AirBuilder` providing access to public values.
pub trait AirBuilderWithPublicValues: AirBuilder {
    /// Type representing a public variable.
    type PublicVar: Into<Self::Expr> + Copy;

    /// Return the list of public variables.
    fn public_values(&self) -> &[Self::PublicVar];
}

/// Extension trait for builders that carry additional runtime context.
///
/// Some AIRs need access to data that is only available at proving time,
/// such as bus randomness, challenge values, or witness hints. This trait
/// lets the builder carry that data so the AIR can read it during
/// constraint evaluation.
///
/// Existing AIRs that do not need extra context are unaffected. Only AIRs
/// that explicitly bound on this trait will use it.
pub trait AirBuilderWithContext: AirBuilder {
    /// The type of additional runtime context available during evaluation.
    type EvalContext;

    /// Returns a reference to the runtime evaluation context.
    fn eval_context(&self) -> &Self::EvalContext;
}

/// Extension of `AirBuilder` for working over extension fields.
pub trait ExtensionBuilder: AirBuilder<F: Field> {
    /// Extension field type.
    type EF: ExtensionField<Self::F>;

    /// Expression type over extension field elements.
    type ExprEF: From<Self::Expr> + Algebra<Self::EF>;

    /// Variable type over extension field elements.
    type VarEF: Into<Self::ExprEF> + Copy + Send + Sync;

    /// Assert that an extension field expression is zero.
    fn assert_zero_ext<I>(&mut self, x: I)
    where
        I: Into<Self::ExprEF>;

    /// Assert that two extension field expressions are equal.
    fn assert_eq_ext<I1, I2>(&mut self, x: I1, y: I2)
    where
        I1: Into<Self::ExprEF>,
        I2: Into<Self::ExprEF>,
    {
        self.assert_zero_ext(x.into() - y.into());
    }

    /// Assert that an extension field expression is equal to one.
    fn assert_one_ext<I>(&mut self, x: I)
    where
        I: Into<Self::ExprEF>,
    {
        self.assert_eq_ext(x, Self::ExprEF::ONE);
    }
}

/// Trait for builders supporting permutation arguments (e.g., for lookup constraints).
pub trait PermutationAirBuilder: ExtensionBuilder {
    /// Matrix type over extension field variables representing a permutation.
    type MP: Matrix<Self::VarEF>;

    /// Randomness variable type used in permutation commitments.
    type RandomVar: Into<Self::ExprEF> + Copy;

    /// Return the matrix representing permutation registers.
    fn permutation(&self) -> Self::MP;

    /// Return the list of randomness values for permutation argument.
    fn permutation_randomness(&self) -> &[Self::RandomVar];
}

/// A wrapper around an [`AirBuilder`] that enforces constraints only when a specified condition is met.
///
/// This struct allows selectively applying constraints to certain rows or under certain conditions in the AIR,
/// without modifying the underlying logic. All constraints asserted through this filtered builder will be
/// multiplied by the given `condition`, effectively disabling them when `condition` evaluates to zero.
#[derive(Debug)]
pub struct FilteredAirBuilder<'a, AB: AirBuilder> {
    /// Reference to the underlying inner [`AirBuilder`] where constraints are ultimately recorded.
    pub inner: &'a mut AB,

    /// Condition expression that controls when the constraints are enforced.
    ///
    /// If `condition` evaluates to zero, constraints asserted through this builder have no effect.
    condition: AB::Expr,
}

impl<AB: AirBuilder> FilteredAirBuilder<'_, AB> {
    pub fn condition(&self) -> AB::Expr {
        self.condition.clone()
    }
}

impl<AB: AirBuilder> AirBuilder for FilteredAirBuilder<'_, AB> {
    type F = AB::F;
    type Expr = AB::Expr;
    type Var = AB::Var;
    type M = AB::M;

    fn main(&self) -> Self::M {
        self.inner.main()
    }

    fn preprocessed(&self) -> Option<Self::M> {
        self.inner.preprocessed()
    }

    fn is_first_row(&self) -> Self::Expr {
        self.inner.is_first_row()
    }

    fn is_last_row(&self) -> Self::Expr {
        self.inner.is_last_row()
    }

    fn is_transition_window(&self, size: usize) -> Self::Expr {
        self.inner.is_transition_window(size)
    }

    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        self.inner.assert_zero(self.condition() * x.into());
    }
}

impl<AB: ExtensionBuilder> ExtensionBuilder for FilteredAirBuilder<'_, AB> {
    type EF = AB::EF;
    type ExprEF = AB::ExprEF;
    type VarEF = AB::VarEF;

    fn assert_zero_ext<I>(&mut self, x: I)
    where
        I: Into<Self::ExprEF>,
    {
        let ext_x = x.into();
        let condition: Self::ExprEF = self.condition().into();

        self.inner.assert_zero_ext(ext_x * condition);
    }
}

impl<AB: PermutationAirBuilder> PermutationAirBuilder for FilteredAirBuilder<'_, AB> {
    type MP = AB::MP;

    type RandomVar = AB::RandomVar;

    fn permutation(&self) -> Self::MP {
        self.inner.permutation()
    }

    fn permutation_randomness(&self) -> &[Self::RandomVar] {
        self.inner.permutation_randomness()
    }
}

impl<AB: AirBuilderWithContext> AirBuilderWithContext for FilteredAirBuilder<'_, AB> {
    type EvalContext = AB::EvalContext;

    fn eval_context(&self) -> &Self::EvalContext {
        self.inner.eval_context()
    }
}
