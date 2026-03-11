use alloc::vec::Vec;
use core::ops::{Add, Mul, Sub};

use p3_field::{Algebra, ExtensionField, Field, PrimeCharacteristicRing};
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixView};

/// Read access to a pair of trace rows (typically current and next).
///
/// Implementors expose two flat slices that constraint evaluators use
/// to express algebraic relations between rows.
pub trait WindowAccess<T> {
    /// Full slice of the current row.
    fn current_slice(&self) -> &[T];

    /// Full slice of the next row.
    fn next_slice(&self) -> &[T];

    /// Single element from the current row by index.
    ///
    /// Returns `None` if `i` is out of bounds.
    #[inline]
    fn current(&self, i: usize) -> Option<T>
    where
        T: Clone,
    {
        self.current_slice().get(i).cloned()
    }

    /// Single element from the next row by index.
    ///
    /// Returns `None` if `i` is out of bounds.
    #[inline]
    fn next(&self, i: usize) -> Option<T>
    where
        T: Clone,
    {
        self.next_slice().get(i).cloned()
    }
}

/// A lightweight two-row window into a trace matrix.
///
/// Stores two `&[T]` slices — one for the current row and one for
/// the next — without carrying any matrix metadata.  This is cheaper
/// than a full `ViewPair` and is the concrete type used by most
/// [`AirBuilder`] implementations for `type MainWindow` / `type PreprocessedWindow`.
#[derive(Debug, Clone, Copy)]
pub struct RowWindow<'a, T> {
    /// The current row.
    current: &'a [T],
    /// The next row.
    next: &'a [T],
}

impl<'a, T> RowWindow<'a, T> {
    /// Create a window from a [`RowMajorMatrixView`] that has exactly
    /// two rows. The first row becomes `current`, the second `next`.
    ///
    /// # Panics
    ///
    /// Panics if the view does not contain exactly `2 * width` elements.
    #[inline]
    pub fn from_view(view: &RowMajorMatrixView<'a, T>) -> Self {
        let width = view.width;
        assert_eq!(
            view.values.len(),
            2 * width,
            "RowWindow::from_view: expected 2 rows (2*{width} elements), got {}",
            view.values.len()
        );
        let (current, next) = view.values.split_at(width);
        Self { current, next }
    }

    /// Create a window from two separate row slices.
    ///
    /// The caller is responsible for providing slices that represent
    /// the intended (current, next) pair.
    ///
    /// # Panics
    ///
    /// Panics (in debug builds) if the slices have different lengths.
    #[inline]
    pub fn from_two_rows(current: &'a [T], next: &'a [T]) -> Self {
        debug_assert_eq!(
            current.len(),
            next.len(),
            "RowWindow::from_two_rows: row lengths differ ({} vs {})",
            current.len(),
            next.len()
        );
        Self { current, next }
    }
}

impl<T> WindowAccess<T> for RowWindow<'_, T> {
    #[inline]
    fn current_slice(&self) -> &[T] {
        self.current
    }

    #[inline]
    fn next_slice(&self) -> &[T] {
        self.next
    }
}

/// The underlying structure of an AIR.
pub trait BaseAir<F>: Sync {
    /// The number of columns (a.k.a. registers) in this AIR.
    fn width(&self) -> usize;
    /// Return an optional preprocessed trace matrix to be included in the prover's trace.
    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        None
    }

    /// Which main trace columns have their next row accessed by this AIR's
    /// constraints.
    ///
    /// By default this returns every column index, which will require
    /// opening all main columns at both `zeta` and `zeta_next`.
    ///
    /// AIRs that only ever read the current main row (and never access an
    /// offset-1 main entry) can override this to return an empty vector to
    /// allow the prover and verifier to open only at `zeta`.
    ///
    /// # When to override
    ///
    /// - **Return empty**: single-row AIRs where all constraints are
    ///   evaluated within one row.
    /// - **Keep default** (all columns): AIRs with transition constraints
    ///   that reference `main.next_slice()`.
    /// - **Return a subset**: AIRs where only a few columns need next-row
    ///   access, enabling future per-column opening optimizations.
    ///
    /// # Correctness
    ///
    /// Must be consistent with [`Air::eval`]. Omitting a column index when
    /// the AIR actually reads its next row will cause verification failures
    /// or, in the worst case, a soundness gap.
    fn main_next_row_columns(&self) -> Vec<usize> {
        (0..self.width()).collect()
    }

    /// Which preprocessed trace columns have their next row accessed by this
    /// AIR's constraints.
    ///
    /// By default this returns every preprocessed column index, which will
    /// require opening preprocessed columns at both `zeta` and `zeta_next`.
    ///
    /// AIRs that only ever read the current preprocessed row (and never
    /// access an offset-1 preprocessed entry) can override this to return an
    /// empty vector to allow the prover and verifier to open only at `zeta`.
    fn preprocessed_next_row_columns(&self) -> Vec<usize> {
        self.preprocessed_trace()
            .map(|t| (0..t.width).collect())
            .unwrap_or_default()
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
    /// Evaluate all AIR constraints using the provided builder.
    ///
    /// The builder provides both the trace on which the constraints
    /// are evaluated on as well as the method of accumulating the
    /// constraint evaluations.
    ///
    /// # Arguments
    /// - `builder`: Mutable reference to an `AirBuilder` for defining constraints.
    fn eval(&self, builder: &mut AB);
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

    /// Assert that the given element is zero, with a human-readable label.
    ///
    /// # Default behavior
    ///
    /// Discards the label and delegates to the unlabeled variant.
    /// This means **zero overhead** in production builders.
    ///
    /// # Debug builder override
    ///
    /// Only the debug builder overrides this to capture labels.
    /// When a constraint fails, the label appears in the diagnostic
    /// report, making failures immediately identifiable.
    fn assert_zero_named<I: Into<Self::Expr>>(&mut self, x: I, _label: &'static str) {
        self.assert_zero(x);
    }

    /// Assert that every element of a given array is 0.
    ///
    /// This should be preferred over calling `assert_zero` multiple times.
    fn assert_zeros<const N: usize, I: Into<Self::Expr>>(&mut self, array: [I; N]) {
        for elem in array {
            self.assert_zero(elem);
        }
    }

    /// Labeled variant of [`assert_zeros`](Self::assert_zeros).
    fn assert_zeros_named<const N: usize, I: Into<Self::Expr>>(
        &mut self,
        array: [I; N],
        label: &'static str,
    ) {
        for elem in array {
            self.assert_zero_named(elem, label);
        }
    }

    /// Assert that a given array consists of only boolean values.
    fn assert_bools<const N: usize, I: Into<Self::Expr>>(&mut self, array: [I; N]) {
        let zero_array = array.map(|x| x.into().bool_check());
        self.assert_zeros(zero_array);
    }

    /// Labeled variant of [`assert_bools`](Self::assert_bools).
    fn assert_bools_named<const N: usize, I: Into<Self::Expr>>(
        &mut self,
        array: [I; N],
        label: &'static str,
    ) {
        let zero_array = array.map(|x| x.into().bool_check());
        self.assert_zeros_named(zero_array, label);
    }

    /// Assert that `x` element is equal to `1`.
    fn assert_one<I: Into<Self::Expr>>(&mut self, x: I) {
        self.assert_zero(x.into() - Self::Expr::ONE);
    }

    /// Labeled variant of [`assert_one`](Self::assert_one).
    fn assert_one_named<I: Into<Self::Expr>>(&mut self, x: I, label: &'static str) {
        self.assert_zero_named(x.into() - Self::Expr::ONE, label);
    }

    /// Assert that the given elements are equal.
    fn assert_eq<I1: Into<Self::Expr>, I2: Into<Self::Expr>>(&mut self, x: I1, y: I2) {
        self.assert_zero(x.into() - y.into());
    }

    /// Labeled variant of [`assert_eq`](Self::assert_eq).
    fn assert_eq_named<I1: Into<Self::Expr>, I2: Into<Self::Expr>>(
        &mut self,
        x: I1,
        y: I2,
        label: &'static str,
    ) {
        self.assert_zero_named(x.into() - y.into(), label);
    }

    /// Public input values available during constraint evaluation.
    ///
    /// Returns an empty slice by default.
    fn public_values(&self) -> &[Self::PublicVar] {
        &[]
    }

    /// Assert that `x` is a boolean, i.e. either `0` or `1`.
    ///
    /// Where possible, batching multiple assert_bool calls
    /// into a single assert_bools call will improve performance.
    fn assert_bool<I: Into<Self::Expr>>(&mut self, x: I) {
        self.assert_zero(x.into().bool_check());
    }

    /// Labeled variant of [`assert_bool`](Self::assert_bool).
    fn assert_bool_named<I: Into<Self::Expr>>(&mut self, x: I, label: &'static str) {
        self.assert_zero_named(x.into().bool_check(), label);
    }
}

/// Extension of [`AirBuilder`] for builders that supply periodic column values.
pub trait PeriodicAirBuilder: AirBuilder {
    /// Variable type for periodic column values.
    type PeriodicVar: Into<Self::Expr> + Copy;

    /// Periodic column values at the current row.
    fn periodic_values(&self) -> &[Self::PeriodicVar];
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

    /// Labeled variant of [`assert_zero_ext`](Self::assert_zero_ext).
    ///
    /// Discards the label by default. Only the debug builder overrides
    /// this to capture labels for diagnostic output.
    fn assert_zero_ext_named<I>(&mut self, x: I, _label: &'static str)
    where
        I: Into<Self::ExprEF>,
    {
        self.assert_zero_ext(x);
    }

    /// Assert that two extension field expressions are equal.
    fn assert_eq_ext<I1, I2>(&mut self, x: I1, y: I2)
    where
        I1: Into<Self::ExprEF>,
        I2: Into<Self::ExprEF>,
    {
        self.assert_zero_ext(x.into() - y.into());
    }

    /// Labeled variant of [`assert_eq_ext`](Self::assert_eq_ext).
    fn assert_eq_ext_named<I1, I2>(&mut self, x: I1, y: I2, label: &'static str)
    where
        I1: Into<Self::ExprEF>,
        I2: Into<Self::ExprEF>,
    {
        self.assert_zero_ext_named(x.into() - y.into(), label);
    }

    /// Assert that an extension field expression is equal to one.
    fn assert_one_ext<I>(&mut self, x: I)
    where
        I: Into<Self::ExprEF>,
    {
        self.assert_eq_ext(x, Self::ExprEF::ONE);
    }

    /// Labeled variant of [`assert_one_ext`](Self::assert_one_ext).
    fn assert_one_ext_named<I>(&mut self, x: I, label: &'static str)
    where
        I: Into<Self::ExprEF>,
    {
        self.assert_eq_ext_named(x, Self::ExprEF::ONE, label);
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
    type PreprocessedWindow = AB::PreprocessedWindow;
    type MainWindow = AB::MainWindow;
    type PublicVar = AB::PublicVar;

    fn main(&self) -> Self::MainWindow {
        self.inner.main()
    }

    fn preprocessed(&self) -> &Self::PreprocessedWindow {
        self.inner.preprocessed()
    }

    fn is_first_row(&self) -> Self::Expr {
        self.inner.is_first_row()
    }

    fn is_last_row(&self) -> Self::Expr {
        self.inner.is_last_row()
    }

    fn is_transition(&self) -> Self::Expr {
        self.inner.is_transition()
    }

    fn is_transition_window(&self, size: usize) -> Self::Expr {
        self.inner.is_transition_window(size)
    }

    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        self.inner.assert_zero(self.condition() * x.into());
    }

    /// Forward the labeled assertion, multiplied by the condition.
    ///
    /// The label passes through unchanged so that diagnostic output
    /// correctly identifies the constraint, even when conditionally applied.
    fn assert_zero_named<I: Into<Self::Expr>>(&mut self, x: I, label: &'static str) {
        self.inner
            .assert_zero_named(self.condition() * x.into(), label);
    }

    fn public_values(&self) -> &[Self::PublicVar] {
        self.inner.public_values()
    }
}

impl<AB: PeriodicAirBuilder> PeriodicAirBuilder for FilteredAirBuilder<'_, AB> {
    type PeriodicVar = AB::PeriodicVar;

    fn periodic_values(&self) -> &[Self::PeriodicVar] {
        self.inner.periodic_values()
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
        let ext_x: Self::ExprEF = x.into();
        let condition: AB::Expr = self.condition();

        self.inner.assert_zero_ext(ext_x * condition);
    }

    fn assert_zero_ext_named<I>(&mut self, x: I, label: &'static str)
    where
        I: Into<Self::ExprEF>,
    {
        let ext_x: Self::ExprEF = x.into();
        let condition: AB::Expr = self.condition();

        self.inner.assert_zero_ext_named(ext_x * condition, label);
    }
}

impl<AB: PermutationAirBuilder> PermutationAirBuilder for FilteredAirBuilder<'_, AB> {
    type MP = AB::MP;

    type RandomVar = AB::RandomVar;

    type PermutationVar = AB::PermutationVar;

    fn permutation(&self) -> Self::MP {
        self.inner.permutation()
    }

    fn permutation_randomness(&self) -> &[Self::RandomVar] {
        self.inner.permutation_randomness()
    }

    fn permutation_values(&self) -> &[Self::PermutationVar] {
        self.inner.permutation_values()
    }
}

impl<AB: AirBuilderWithContext> AirBuilderWithContext for FilteredAirBuilder<'_, AB> {
    type EvalContext = AB::EvalContext;

    fn eval_context(&self) -> &Self::EvalContext {
        self.inner.eval_context()
    }
}
