use core::ops::{Add, Mul, Sub};

use p3_field::{Algebra, Dup, ExtensionField, Field, PrimeCharacteristicRing};

use crate::filtered::FilteredAirBuilder;
use crate::window::WindowAccess;

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
        + Copy
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

    /// Two-row window over the preprocessed trace columns.
    type PreprocessedWindow: WindowAccess<Self::Var> + Clone;

    /// Two-row window over the main trace columns.
    type MainWindow: WindowAccess<Self::Var> + Clone;

    /// Variable type for public values.
    type PublicVar: Into<Self::Expr> + Copy;

    /// Variable type for periodic column values at the current row.
    ///
    /// - Periodic columns are commitment-free witness data.
    /// - Their values repeat with a fixed period.
    /// - Builders without periodic data still pick a concrete type, so
    ///   the trait can be implemented uniformly.
    type PeriodicVar: Into<Self::Expr> + Copy;

    /// Return the current and next row slices of the main (primary) trace.
    fn main(&self) -> Self::MainWindow;

    /// Return the preprocessed registers as a two-row window.
    ///
    /// When no preprocessed columns exist, this returns a zero-width window.
    fn preprocessed(&self) -> &Self::PreprocessedWindow;

    /// Expression evaluating to a non-zero value only on the first row.
    fn is_first_row(&self) -> Self::Expr;

    /// Expression evaluating to a non-zero value only on the last row.
    fn is_last_row(&self) -> Self::Expr;

    /// Expression evaluating to zero only on the last row.
    fn is_transition(&self) -> Self::Expr {
        self.is_transition_window(2)
    }

    /// Expression evaluating to zero only on the last `size - 1` rows.
    ///
    /// # Panics
    ///
    /// Implementations should panic if `size > 2`, since only two-row
    /// windows are currently supported.
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

    /// Like [`when_transition`](Self::when_transition), but requires a window of `size` rows.
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

    /// Assert that two sequences of same length are equal element-wise.
    fn assert_eq_arrays<I1, I2, const N: usize>(&mut self, lhs: [I1; N], rhs: [I2; N])
    where
        I1: Dup + Into<Self::Expr>,
        I2: Dup + Into<Self::Expr>,
    {
        debug_assert_eq!(
            lhs.len(),
            rhs.len(),
            "assert_eq_arrays: length mismatch ({} vs {})",
            lhs.len(),
            rhs.len()
        );

        let diff: [Self::Expr; N] =
            core::array::from_fn(|i| lhs[i].dup().into() - rhs[i].dup().into());

        self.assert_zeros(diff);
    }

    /// Public input values available during constraint evaluation.
    ///
    /// Returns an empty slice by default.
    fn public_values(&self) -> &[Self::PublicVar] {
        &[]
    }

    /// Values of every periodic column at the current row.
    ///
    /// - One entry per periodic column declared by the AIR.
    /// - Ordering matches the AIR's declared column order.
    /// - Default is empty for builders without periodic data.
    fn periodic_values(&self) -> &[Self::PeriodicVar] {
        &[]
    }

    /// Assert that `x` is a boolean, i.e. either `0` or `1`.
    ///
    /// Where possible, batching multiple assert_bool calls
    /// into a single assert_bools call will improve performance.
    fn assert_bool<I: Into<Self::Expr>>(&mut self, x: I) {
        self.assert_zero(x.into().bool_check());
    }
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
    type ExprEF: Algebra<Self::Expr> + Algebra<Self::EF>;

    /// Variable type over extension field elements.
    type VarEF: Into<Self::ExprEF> + Copy + Send + Sync;

    /// Assert that an extension field expression is zero.
    fn assert_zero_ext<I>(&mut self, x: I)
    where
        I: Into<Self::ExprEF>;

    /// Assert that every element of a given array is zero in the extension field.
    ///
    /// Where possible, batching multiple `assert_zero_ext` calls into a single
    /// `assert_zeros_ext` call will improve performance.
    fn assert_zeros_ext<const N: usize, I>(&mut self, array: [I; N])
    where
        I: Into<Self::ExprEF>,
    {
        for x in array {
            self.assert_zero_ext(x);
        }
    }

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
    /// Two-row window over the permutation trace columns.
    type MP: WindowAccess<Self::VarEF>;

    /// Randomness variable type used in permutation commitments.
    type RandomVar: Into<Self::ExprEF> + Copy;

    /// Value type for expected cumulated values used in global lookup arguments.
    type PermutationVar: Into<Self::ExprEF> + Clone;

    /// Return the current and next row slices of the permutation trace.
    fn permutation(&self) -> Self::MP;

    /// Return the list of randomness values for permutation argument.
    fn permutation_randomness(&self) -> &[Self::RandomVar];

    /// Return the expected cumulated values for global lookup arguments.
    fn permutation_values(&self) -> &[Self::PermutationVar];
}
