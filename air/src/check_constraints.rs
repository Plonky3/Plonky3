use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt;

use p3_field::{ExtensionField, Field};
use p3_matrix::Matrix;
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixView};
use p3_matrix::stack::ViewPair;

use crate::boundary::{self, BoundaryPublic};
use crate::{
    Air, AirBuilder, AirBuilderWithContext, BaseAir, ExtensionBuilder, Name, NamedAirBuilder,
    NamedExtensionBuilder, PermutationAirBuilder, RowWindow,
};

/// A single constraint violation captured during debug evaluation.
///
/// During constraint checking the builder evaluates every AIR constraint
/// on every trace row.
///
/// When a constraint evaluates to a non-zero value, the violation is
/// recorded here **instead of panicking**.
/// This allows the caller to inspect all failures at once and produce
/// comprehensive diagnostic output.
///
/// # Label support
///
/// Constraints asserted with the labeled variant carry a human-readable
/// string identifying them by purpose (e.g. `"range_check"`).
///
/// Labels are zero-cost in production builders because:
/// - The default trait implementation discards them
/// - Only the debug builder overrides it to capture them
/// - `&'static str` avoids any heap allocation
#[derive(Debug, Clone)]
pub struct ConstraintFailure {
    /// Zero-based index of the trace row where the violation occurred.
    pub row: usize,

    /// Zero-based position of the constraint within one evaluation pass.
    ///
    /// The counter increments after every assertion, **pass or fail**.
    /// This gives each constraint a stable index regardless of which
    /// other constraints fail.
    pub constraint: usize,

    /// Human-readable label for this constraint.
    ///
    /// - Set when the constraint was asserted via [`NamedAirBuilder`].
    /// - `None` when the standard unlabeled assertion was used.
    pub label: Option<String>,
}

impl fmt::Display for ConstraintFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "#{}", self.constraint)?;
        if let Some(label) = &self.label {
            write!(f, " {label:?}")?;
        }
        Ok(())
    }
}

/// Summary of all constraint violations across a full trace evaluation.
///
/// Collects **every** violation in a single pass, giving the developer
/// a complete picture of what went wrong.
///
/// # Why
///
/// Wide AIRs can have 50+ constraints per row.
/// Without a full report, debugging becomes a tedious loop:
///
/// 1. Run the checker
/// 2. Fix the one reported failure
/// 3. Re-run, discover the next failure
/// 4. Repeat
///
/// With a full report, **all** failures are visible at once.
///
/// # Memory safety
///
/// Use the `max_failures` parameter in [`check_all_constraints`] to cap
/// the number of recorded violations.
/// Without a cap, a fully-broken trace could allocate unbounded memory.
#[derive(Debug, Clone)]
pub struct ConstraintReport {
    /// Every constraint violation found during the evaluation pass.
    ///
    /// Row-ordered for the AIR's own constraints.
    /// Listed public boundary cells are compared after the row scan and appended last,
    /// so they are not in row order with respect to the rest.
    pub failures: Vec<ConstraintFailure>,

    /// Total number of rows in the evaluated trace.
    pub total_rows: usize,

    /// Number of constraints evaluated per row.
    ///
    /// Captured from the first row's evaluation.
    /// If the AIR dynamically varies its constraint count per row,
    /// this reflects only the first.
    ///
    /// A listed public boundary cell is not one of these.
    /// Its failure carries an index at or past this count, matching where the backend
    /// injects the pin.
    pub total_constraints_per_row: usize,
}

impl ConstraintReport {
    /// Returns `true` when no constraint violations were found.
    pub const fn is_ok(&self) -> bool {
        self.failures.is_empty()
    }
}

/// Debug-mode constraint builder that evaluates an AIR over concrete field
/// values and collects every violation rather than panicking immediately.
///
/// This is the single, canonical builder used by both uni-stark and
/// batch-stark for debug constraint checking.
///
/// The second type parameter defaults to the base field, so callers that
/// do not need extension-field support can simply write
/// `DebugConstraintBuilder<'a, F>`. When permutation or lookup arguments
/// are involved, the full form `DebugConstraintBuilder<'a, F, EF>` is used.
#[derive(Debug)]
pub struct DebugConstraintBuilder<'a, F: Field, EF: ExtensionField<F> = F> {
    /// Zero-based index of the trace row currently under evaluation.
    row_index: usize,

    /// Counter incremented after every constraint assertion, giving each
    /// constraint a stable index within a single evaluation pass.
    constraint_index: usize,

    /// All constraint violations recorded so far for the current row.
    failures: Vec<ConstraintFailure>,

    /// Vertical pair giving access to the current and next witness rows.
    main: ViewPair<'a, F>,

    /// Window over the current and next preprocessed rows.
    /// When the AIR has no preprocessed trace this is a zero-width window.
    preprocessed: RowWindow<'a, F>,

    /// Slice of public values made available to the AIR during evaluation.
    public_values: &'a [F],

    /// Selector that equals one on the first row and zero elsewhere.
    is_first_row: F,

    /// Selector that equals one on the last row and zero elsewhere.
    is_last_row: F,

    /// Selector that equals one on every row except the last.
    is_transition: F,

    /// Vertical pair for the current and next permutation rows, present
    /// only when the AIR uses a permutation argument.
    permutation: Option<ViewPair<'a, EF>>,

    /// Challenge values for the permutation argument, empty when unused.
    permutation_challenges: &'a [EF],

    /// Expected cumulated values for global lookup arguments.
    permutation_values: &'a [EF],

    /// Values of each periodic column at [`Self::row_index`], in column order.
    ///
    /// Empty when the AIR has no periodic columns.
    periodic_row: &'a [F],
}

impl<'a, F: Field> DebugConstraintBuilder<'a, F> {
    /// Build a constraint checker for AIRs that do not use permutations.
    ///
    /// Permutation-related fields are set to `None` / empty so that the
    /// builder can still satisfy trait bounds that require extension-field
    /// support, but calling permutation accessors will panic.
    ///
    /// `periodic_row` must hold the slice returned by `BaseAir::periodic_values` for
    /// the current row (or be empty when the AIR has no periodic columns).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        row_index: usize,
        main: ViewPair<'a, F>,
        preprocessed: ViewPair<'a, F>,
        public_values: &'a [F],
        is_first_row: F,
        is_last_row: F,
        is_transition: F,
        periodic_row: &'a [F],
    ) -> Self {
        Self {
            row_index,
            constraint_index: 0,
            failures: Vec::new(),
            main,
            preprocessed: RowWindow::from_two_rows(
                preprocessed.top.values,
                preprocessed.bottom.values,
            ),
            public_values,
            is_first_row,
            is_last_row,
            is_transition,
            permutation: None,
            permutation_challenges: &[],
            permutation_values: &[],
            periodic_row,
        }
    }
}

impl<'a, F: Field, EF: ExtensionField<F>> DebugConstraintBuilder<'a, F, EF> {
    /// Build a constraint checker that also carries permutation data.
    ///
    /// Use this when the AIR declares lookup or permutation arguments
    /// that require access to the permutation trace and challenges.
    ///
    /// `periodic_row` must hold the slice returned by `BaseAir::periodic_values` for
    /// the current row (or be empty when the AIR has no periodic columns).
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_permutation(
        row_index: usize,
        main: ViewPair<'a, F>,
        preprocessed: ViewPair<'a, F>,
        public_values: &'a [F],
        is_first_row: F,
        is_last_row: F,
        is_transition: F,
        permutation: ViewPair<'a, EF>,
        permutation_challenges: &'a [EF],
        permutation_values: &'a [EF],
        periodic_row: &'a [F],
    ) -> Self {
        Self {
            row_index,
            constraint_index: 0,
            failures: Vec::new(),
            main,
            preprocessed: RowWindow::from_two_rows(
                preprocessed.top.values,
                preprocessed.bottom.values,
            ),
            public_values,
            is_first_row,
            is_last_row,
            is_transition,
            permutation: Some(permutation),
            permutation_challenges,
            permutation_values,
            periodic_row,
        }
    }

    /// Whether at least one constraint violation has been recorded.
    pub const fn has_failures(&self) -> bool {
        !self.failures.is_empty()
    }

    /// Borrow the list of recorded constraint violations.
    pub fn failures(&self) -> &[ConstraintFailure] {
        &self.failures
    }

    /// Render the recorded violations as a bracketed, comma-separated list.
    ///
    /// Output form: `[#0, #1 "col_1_must_be_zero", #4]`. Every entry starts
    /// with `#`, so mixed labeled and unlabeled entries stay unambiguous.
    pub fn formatted_failures(&self) -> String {
        let entries: Vec<String> = self.failures.iter().map(ToString::to_string).collect();
        format!("[{}]", entries.join(", "))
    }

    /// Consume the builder and return all recorded constraint violations.
    pub fn into_failures(self) -> Vec<ConstraintFailure> {
        self.failures
    }
}

impl<'a, F, EF> AirBuilder for DebugConstraintBuilder<'a, F, EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    type F = F;
    type Expr = F;
    type Var = F;
    type PreprocessedWindow = RowWindow<'a, F>;
    type MainWindow = RowWindow<'a, F>;
    type PublicVar = F;
    type PeriodicVar = F;

    fn main(&self) -> Self::MainWindow {
        RowWindow::from_two_rows(self.main.top.values, self.main.bottom.values)
    }

    fn preprocessed(&self) -> &Self::PreprocessedWindow {
        &self.preprocessed
    }

    fn is_first_row(&self) -> Self::Expr {
        self.is_first_row
    }

    fn is_last_row(&self) -> Self::Expr {
        self.is_last_row
    }

    fn is_transition(&self) -> Self::Expr {
        self.is_transition
    }

    /// Delegates to the named variant with an empty label.
    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        self.assert_zero_named(x, "");
    }

    fn public_values(&self) -> &[Self::PublicVar] {
        self.public_values
    }

    fn periodic_values(&self) -> &[Self::PeriodicVar] {
        self.periodic_row
    }
}

impl<F: Field, EF: ExtensionField<F>> AirBuilderWithContext for DebugConstraintBuilder<'_, F, EF> {
    /// No extra context is needed during debug evaluation.
    type EvalContext = ();

    fn eval_context(&self) -> &Self::EvalContext {
        &()
    }
}

impl<F: Field, EF: ExtensionField<F>> ExtensionBuilder for DebugConstraintBuilder<'_, F, EF> {
    type EF = EF;
    type ExprEF = EF;
    type VarEF = EF;

    /// Delegates to the named variant with an empty label.
    fn assert_zero_ext<I>(&mut self, x: I)
    where
        I: Into<Self::ExprEF>,
    {
        self.assert_zero_ext_named(x, "");
    }
}

impl<'a, F: Field, EF: ExtensionField<F>> PermutationAirBuilder
    for DebugConstraintBuilder<'a, F, EF>
{
    type MP = RowWindow<'a, EF>;
    type RandomVar = EF;
    type PermutationVar = EF;

    /// # Panics
    ///
    /// Panics when the builder was created without permutation data.
    fn permutation(&self) -> Self::MP {
        let p = self.permutation
            .expect("permutation() called on a builder created without permutation data; use new_with_permutation()");
        RowWindow::from_two_rows(p.top.values, p.bottom.values)
    }

    fn permutation_randomness(&self) -> &[Self::RandomVar] {
        self.permutation_challenges
    }

    fn permutation_values(&self) -> &[Self::PermutationVar] {
        self.permutation_values
    }
}

impl<F: Field, EF: ExtensionField<F>> NamedAirBuilder for DebugConstraintBuilder<'_, F, EF> {
    /// Primary constraint implementation for the debug builder.
    ///
    /// Evaluates the name and captures the label on failure.
    /// The unlabeled path delegates here with an empty string.
    fn assert_zero_named<I, N>(&mut self, x: I, name: N)
    where
        I: Into<Self::Expr>,
        N: Name,
    {
        if x.into() != F::ZERO {
            let label = name.evaluate().to_string();
            self.failures.push(ConstraintFailure {
                row: self.row_index,
                constraint: self.constraint_index,
                label: if label.is_empty() { None } else { Some(label) },
            });
        }
        self.constraint_index += 1;
    }
}

impl<F: Field, EF: ExtensionField<F>> NamedExtensionBuilder for DebugConstraintBuilder<'_, F, EF> {
    /// Primary extension-field constraint implementation for the debug
    /// builder.
    ///
    /// Same pattern: unlabeled path delegates here with an empty string.
    fn assert_zero_ext_named<I, N>(&mut self, x: I, name: N)
    where
        I: Into<Self::ExprEF>,
        N: Name,
    {
        if x.into() != EF::ZERO {
            let label = name.evaluate().to_string();
            self.failures.push(ConstraintFailure {
                row: self.row_index,
                constraint: self.constraint_index,
                label: if label.is_empty() { None } else { Some(label) },
            });
        }
        self.constraint_index += 1;
    }
}

/// Find every cell an AIR lists as a public boundary input that does not hold its value.
///
/// A listed cell carries no AIR constraint of its own:
///
/// ```text
///     proving backend : binds the cell to the public value
///     this checker    : compares the two directly
/// ```
///
/// # Returns
///
/// One entry per mismatch, as `(index in declaration order, trace row)`.
///
/// # Panics
///
/// Panics when the declaration names a column or a public value that does not exist.
/// That is a bug in the AIR rather than a failure of the witness.
fn boundary_io_mismatches<F, A>(
    air: &A,
    main: &RowMajorMatrix<F>,
    public_values: &[F],
) -> Vec<(usize, usize)>
where
    F: Field,
    A: BaseAir<F>,
{
    let cells = air.public_boundary_io();

    // Check the declaration against what is actually on hand, not against what the AIR declares.
    boundary::validate(cells, main.width(), public_values.len())
        .unwrap_or_else(|error| panic!("{error}"));

    // Both trace ends are addressed relative to the row count.
    let height = main.height();

    cells
        .iter()
        .enumerate()
        .filter_map(|(index, cell)| {
            let row = cell.row(height);
            let value = main
                .get(row, cell.column)
                .expect("row and column checked in range");
            (value != public_values[cell.public_value]).then_some((index, row))
        })
        .collect()
}

/// Human-readable name of one listed public boundary cell.
fn boundary_io_label(cell: &BoundaryPublic) -> String {
    format!(
        "boundary-IO cell in column {} does not hold public value {}",
        cell.column, cell.public_value
    )
}

/// Reject a periodic column whose shape the trace cannot carry.
///
/// - A column of period `p` holds the evaluations of one polynomial over a subgroup of order `p`.
/// - Such a subgroup exists only when `p` is a power of two.
/// - It embeds in the trace domain only when `p` divides the number of rows.
///
/// Row lookups wrap with `row mod p`, so an ill-shaped column still yields a value on every row.
/// Screening the shape here keeps the diagnosis in the layer that owns the AIR.
///
/// # Panics
///
/// - A period that is not a power of two has no subgroup to interpolate over.
/// - A period that does not divide the row count cannot tile the trace.
fn assert_periodic_column_shapes<F: Field, A: BaseAir<F>>(air: &A, height: usize) {
    for (index, col) in air.periodic_columns().iter().enumerate() {
        // The period is how many values the column lists before repeating.
        let period = col.len();

        // Powers of two are the orders for which a two-adic subgroup exists.
        // A zero-length column fails here, ahead of the row lookup that would divide by it.
        assert!(
            period.is_power_of_two(),
            "debug constraint check requires periodic column {index} to have a power-of-two length, got {period}"
        );

        // Divisibility lands every repetition on a whole copy of that subgroup.
        //
        //     height = 8, period = 2  ->  [0,1][0,1][0,1][0,1]   tiles
        //     height = 8, period = 16 ->  [0,1,..,7|8,..,15]     truncated
        //
        // The commitment layer asserts the same relation, so both verdicts stay in step.
        assert!(
            height.is_multiple_of(period),
            "debug constraint check requires periodic column {index} (length {period}) to divide the trace height ({height})"
        );
    }
}

/// Evaluate every AIR constraint against a concrete trace and panic on failure.
///
/// The function walks the trace row by row. For each row it:
///
/// 1. Builds a vertical pair of the current and next rows (wrapping around
///    at the end).
/// 2. Sets the first-row, last-row and transition selectors.
/// 3. Evaluates the AIR, collecting all violated constraints
///    (index with optional label).
/// 4. Stops at the first row that has at least one violation and panics
///    with a summary of every violated constraint on that row.
///
/// Cells the AIR lists as public inputs are compared before the loop.
/// No AIR constraint asserts them.
///
/// This is the simple variant that does not involve permutation or lookup
/// arguments. Batch-stark provides its own wrapper that additionally
/// supplies permutation data and lookup inputs.
#[allow(unused)] // Suppresses warnings in release mode where this is dead code.
pub fn check_constraints<F, A>(air: &A, main: &RowMajorMatrix<F>, public_values: &[F])
where
    F: Field,
    A: for<'a> Air<DebugConstraintBuilder<'a, F>>,
{
    let height = main.height();
    let preprocessed = air.preprocessed_trace();
    if let Some(prep) = preprocessed.as_ref() {
        assert_eq!(
            prep.height(),
            height,
            "debug constraint check requires preprocessed trace height ({}) to match main trace height ({})",
            prep.height(),
            height
        );
    }
    assert_periodic_column_shapes(air, height);

    // A listed cell has no AIR constraint, so the row loop below would never see it.
    if let Some(&(index, row)) = boundary_io_mismatches(air, main, public_values).first() {
        panic!(
            "row {row}: {}",
            boundary_io_label(&air.public_boundary_io()[index])
        );
    }

    for row_index in 0..height {
        let row_index_next = (row_index + 1) % height;

        // SAFETY: both indices are strictly less than `height`.
        let local = unsafe { main.row_slice_unchecked(row_index) };
        let next = unsafe { main.row_slice_unchecked(row_index_next) };

        // Pair the current and next witness rows into a vertical view.
        let main_pair = ViewPair::new(
            RowMajorMatrixView::new_row(&*local),
            RowMajorMatrixView::new_row(&*next),
        );

        // Pair the preprocessed rows. When the AIR has no preprocessed
        // trace we build a zero-width pair so the builder always has a
        // valid (possibly empty) preprocessed matrix.
        let (prep_local, prep_next) = preprocessed.as_ref().map_or((None, None), |prep| unsafe {
            // SAFETY: same index range as the main trace.
            (
                Some(prep.row_slice_unchecked(row_index)),
                Some(prep.row_slice_unchecked(row_index_next)),
            )
        });
        let preprocessed_pair = match (prep_local.as_ref(), prep_next.as_ref()) {
            (Some(l), Some(n)) => ViewPair::new(
                RowMajorMatrixView::new_row(&**l),
                RowMajorMatrixView::new_row(&**n),
            ),
            _ => ViewPair::new(
                RowMajorMatrixView::new(&[], 0),
                RowMajorMatrixView::new(&[], 0),
            ),
        };

        // Construct the builder with row selectors derived from the position.
        let periodic_row = air.periodic_values(row_index);
        let mut builder = DebugConstraintBuilder::new(
            row_index,
            main_pair,
            preprocessed_pair,
            public_values,
            F::from_bool(row_index == 0),
            F::from_bool(row_index == height - 1),
            F::from_bool(row_index != height - 1),
            &periodic_row,
        );

        // Run every AIR constraint on this row.
        air.eval(&mut builder);

        // Stop at the first failing row and report all violations at once.
        if builder.has_failures() {
            let rendered = builder.formatted_failures();
            panic!(
                "constraints not satisfied on row {row_index}: \
                 failed constraints = {rendered}"
            );
        }
    }
}

/// Evaluate every AIR constraint against a concrete trace and collect
/// **all** violations, returning a [`ConstraintReport`].
///
/// Unlike [`check_constraints`] which panics at the first failing row,
/// this function continues through the **entire** trace.
/// Invaluable for debugging wide AIRs where multiple rows fail
/// independently.
///
/// # Failure cap
///
/// The optional `max_failures` parameter prevents unbounded memory on
/// large traces with many violations.
///
/// The cap is checked **between** rows, so the final count may slightly
/// exceed it (by up to one row's worth of failures).
///
/// # Permutation arguments
///
/// This is the simple variant — no permutation or lookup arguments.
/// Batch-stark provides its own wrapper for those.
///
/// # Public boundary cells
///
/// A cell an AIR lists as a public input is compared after the row scan.
/// Each mismatch is reported at an index past the AIR's own constraints,
/// which is where the proving backend injects its pin.
///
/// These count against `max_failures` like any other violation.
#[allow(unused)] // Suppresses warnings in release mode where this is dead code.
pub fn check_all_constraints<F, A>(
    air: &A,
    main: &RowMajorMatrix<F>,
    public_values: &[F],
    max_failures: Option<usize>,
) -> ConstraintReport
where
    F: Field,
    A: for<'a> Air<DebugConstraintBuilder<'a, F>>,
{
    let height = main.height();
    let preprocessed = air.preprocessed_trace();
    if let Some(prep) = preprocessed.as_ref() {
        assert_eq!(
            prep.height(),
            height,
            "debug constraint check requires preprocessed trace height ({}) to match main trace height ({})",
            prep.height(),
            height
        );
    }
    assert_periodic_column_shapes(air, height);

    // Accumulate violations across all rows.
    let mut all_failures = Vec::new();

    // Capture how many constraints the AIR asserts per row.
    let mut total_constraints_per_row = 0;

    for row_index in 0..height {
        // Early exit when the failure cap is reached.
        if let Some(cap) = max_failures
            && all_failures.len() >= cap
        {
            break;
        }

        // Wrap around to row 0 after the last row.
        let row_index_next = (row_index + 1) % height;

        // SAFETY: both indices are strictly less than `height`.
        let local = unsafe { main.row_slice_unchecked(row_index) };
        let next = unsafe { main.row_slice_unchecked(row_index_next) };

        // Pair the current and next witness rows into a vertical view.
        let main_pair = ViewPair::new(
            RowMajorMatrixView::new_row(&*local),
            RowMajorMatrixView::new_row(&*next),
        );

        // Build the preprocessed pair, falling back to zero-width when
        // the AIR has no preprocessed trace.
        let (prep_local, prep_next) = preprocessed.as_ref().map_or((None, None), |prep| unsafe {
            // SAFETY: same index range as the main trace.
            (
                Some(prep.row_slice_unchecked(row_index)),
                Some(prep.row_slice_unchecked(row_index_next)),
            )
        });
        let preprocessed_pair = match (prep_local.as_ref(), prep_next.as_ref()) {
            (Some(l), Some(n)) => ViewPair::new(
                RowMajorMatrixView::new_row(&**l),
                RowMajorMatrixView::new_row(&**n),
            ),
            _ => ViewPair::new(
                RowMajorMatrixView::new(&[], 0),
                RowMajorMatrixView::new(&[], 0),
            ),
        };

        // Derive the row selectors from the current position.
        let periodic_row = air.periodic_values(row_index);
        let mut builder = DebugConstraintBuilder::new(
            row_index,
            main_pair,
            preprocessed_pair,
            public_values,
            F::from_bool(row_index == 0),
            F::from_bool(row_index == height - 1),
            F::from_bool(row_index != height - 1),
            &periodic_row,
        );

        // Run every AIR constraint on this row.
        air.eval(&mut builder);

        // Record the constraint count from the first row only.
        if row_index == 0 {
            total_constraints_per_row = builder.constraint_index;
        }

        // Collect any violations from this row.
        all_failures.extend(builder.into_failures());
    }

    // A listed cell has no AIR constraint, so it takes an index after the AIR's own.
    // That is where the proving backend injects its pin.
    //
    // The cap bounds these like any other failure, since the row loop may have used it up.
    let cells = air.public_boundary_io();
    let remaining = max_failures.map_or(usize::MAX, |cap| cap.saturating_sub(all_failures.len()));
    all_failures.extend(
        boundary_io_mismatches(air, main, public_values)
            .into_iter()
            .take(remaining)
            .map(|(index, row)| ConstraintFailure {
                row,
                constraint: total_constraints_per_row + index,
                label: Some(boundary_io_label(&cells[index])),
            }),
    );

    ConstraintReport {
        failures: all_failures,
        total_rows: height,
        total_constraints_per_row,
    }
}

#[cfg(test)]
mod tests {
    use alloc::borrow::Cow;
    use alloc::{format, vec};

    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;

    use super::*;
    use crate::{BoundaryEnd, BoundaryPublic, WindowAccess};

    /// Minimal AIR for testing transition and boundary constraints.
    ///
    /// Enforces two rules:
    /// - **Transition**: every column increments by one between consecutive rows.
    /// - **Boundary**: the last row must equal the provided public values.
    #[derive(Debug)]
    struct RowLogicAir<const W: usize>;

    impl<F: Field, const W: usize> BaseAir<F> for RowLogicAir<W> {
        fn width(&self) -> usize {
            W
        }
    }

    impl<F: Field, const W: usize> Air<DebugConstraintBuilder<'_, F>> for RowLogicAir<W> {
        fn eval(&self, builder: &mut DebugConstraintBuilder<'_, F>) {
            let main = builder.main();

            // Transition constraint: next == current + 1 for every column.
            for col in 0..W {
                let current = main.current(col).unwrap();
                let next = main.next(col).unwrap();
                builder.when_transition().assert_eq(next, current + F::ONE);
            }

            // Boundary constraint: last row matches public values.
            let public_values = builder.public_values;
            let mut when_last = builder.when(builder.is_last_row);
            for (i, &pv) in public_values.iter().enumerate().take(W) {
                when_last.assert_eq(main.current(i).unwrap(), pv);
            }
        }
    }

    #[test]
    fn test_incremental_rows_with_last_row_check() {
        let air = RowLogicAir::<2>;
        let values = vec![
            BabyBear::ONE,
            BabyBear::ONE, // Row 0
            BabyBear::new(2),
            BabyBear::new(2), // Row 1
            BabyBear::new(3),
            BabyBear::new(3), // Row 2
            BabyBear::new(4),
            BabyBear::new(4), // Row 3 (last)
        ];
        let main = RowMajorMatrix::new(values, 2);
        check_constraints(&air, &main, &[BabyBear::new(4); 2]);
    }

    #[test]
    #[should_panic]
    fn test_incorrect_increment_logic() {
        let air = RowLogicAir::<2>;
        let values = vec![
            BabyBear::ONE,
            BabyBear::ONE, // Row 0
            BabyBear::new(2),
            BabyBear::new(2), // Row 1
            BabyBear::new(5),
            BabyBear::new(5), // Row 2 (wrong)
            BabyBear::new(6),
            BabyBear::new(6), // Row 3
        ];
        let main = RowMajorMatrix::new(values, 2);
        check_constraints(&air, &main, &[BabyBear::new(6); 2]);
    }

    #[test]
    #[should_panic]
    fn test_wrong_last_row_public_value() {
        let air = RowLogicAir::<2>;
        let values = vec![
            BabyBear::ONE,
            BabyBear::ONE, // Row 0
            BabyBear::new(2),
            BabyBear::new(2), // Row 1
            BabyBear::new(3),
            BabyBear::new(3), // Row 2
            BabyBear::new(4),
            BabyBear::new(4), // Row 3
        ];
        let main = RowMajorMatrix::new(values, 2);
        check_constraints(&air, &main, &[BabyBear::new(4), BabyBear::new(5)]);
    }

    #[test]
    fn test_single_row_wraparound_logic() {
        let air = RowLogicAir::<2>;
        let values = vec![
            BabyBear::new(99),
            BabyBear::new(77), // Row 0
        ];
        let main = RowMajorMatrix::new(values, 2);
        check_constraints(&air, &main, &[BabyBear::new(99), BabyBear::new(77)]);
    }

    /// Helper: build a single-row builder with the given main row values
    /// and evaluate an AIR, returning the builder for inspection.
    fn eval_single_row<const W: usize, A>(
        air: &A,
        row: [BabyBear; W],
    ) -> DebugConstraintBuilder<'static, BabyBear>
    where
        A: for<'a> Air<DebugConstraintBuilder<'a, BabyBear>>,
    {
        // Leak the row so we get a 'static borrow (fine in tests).
        let row: &'static [BabyBear] = Vec::from(row).leak();
        let view = RowMajorMatrixView::new_row(row);
        let main = ViewPair::new(view, view);

        let empty_view = RowMajorMatrixView::new(&[], 0);
        let mut builder = DebugConstraintBuilder::new(
            0,
            main,
            ViewPair::new(empty_view, empty_view),
            &[],
            BabyBear::ONE,  // is_first_row
            BabyBear::ONE,  // is_last_row (single row)
            BabyBear::ZERO, // is_transition
            &[],
        );
        air.eval(&mut builder);
        builder
    }

    /// AIR that asserts every column is zero. With W columns this produces
    /// W independent constraints, useful for testing multi-failure collection.
    #[derive(Debug)]
    struct AllZeroAir<const W: usize>;

    impl<F: Field, const W: usize> BaseAir<F> for AllZeroAir<W> {
        fn width(&self) -> usize {
            W
        }
    }

    impl<F: Field, const W: usize> Air<DebugConstraintBuilder<'_, F>> for AllZeroAir<W> {
        fn eval(&self, builder: &mut DebugConstraintBuilder<'_, F>) {
            let main = builder.main();
            for col in 0..W {
                builder.assert_zero(main.current(col).unwrap());
            }
        }
    }

    #[test]
    fn test_no_failures_when_all_constraints_pass() {
        let builder = eval_single_row(&AllZeroAir::<3>, [BabyBear::ZERO; 3]);
        assert!(!builder.has_failures());
        assert!(builder.failures().is_empty());
    }

    #[test]
    fn test_multiple_failures_collected() {
        // Columns 0 and 2 are non-zero, column 1 is zero.
        let builder = eval_single_row(
            &AllZeroAir::<3>,
            [BabyBear::ONE, BabyBear::ZERO, BabyBear::new(42)],
        );

        assert!(builder.has_failures());

        let failures = builder.failures();
        assert_eq!(failures.len(), 2);

        // Constraint 0 (column 0) failed.
        assert_eq!(failures[0].row, 0);
        assert_eq!(failures[0].constraint, 0);

        // Constraint 2 (column 2) failed; constraint 1 passed but still
        // advanced the counter.
        assert_eq!(failures[1].row, 0);
        assert_eq!(failures[1].constraint, 2);
    }

    #[test]
    fn test_into_failures() {
        let builder = eval_single_row(&AllZeroAir::<2>, [BabyBear::ONE, BabyBear::ONE]);

        let failures = builder.into_failures();
        assert_eq!(failures.len(), 2);
        assert_eq!(failures[0].constraint, 0);
        assert_eq!(failures[1].constraint, 1);
    }

    #[test]
    #[should_panic(expected = "failed constraints = [#0, #2]")]
    fn test_panic_message_lists_all_failed_indices() {
        let air = AllZeroAir::<3>;
        let values = vec![BabyBear::ONE, BabyBear::ZERO, BabyBear::new(7)];
        let main = RowMajorMatrix::new(values, 3);
        check_constraints(&air, &main, &[]);
    }

    #[test]
    #[should_panic(expected = "failed constraints = [#0, #1 \"col_1_must_be_zero\"]")]
    fn test_panic_message_includes_label_when_available() {
        let air = NamedConstraintAir;
        let values = vec![BabyBear::ONE, BabyBear::ONE];
        let main = RowMajorMatrix::new(values, 2);
        check_constraints(&air, &main, &[]);
    }

    #[test]
    fn test_check_all_constraints_no_failures() {
        let air = AllZeroAir::<2>;
        let values = BabyBear::zero_vec(4); // 2 rows × 2 cols, all zero
        let main = RowMajorMatrix::new(values, 2);
        let report = check_all_constraints(&air, &main, &[], None);
        assert!(report.is_ok());
        assert_eq!(report.total_rows, 2);
        assert_eq!(report.total_constraints_per_row, 2);
    }

    #[test]
    fn test_check_all_constraints_multiple_rows_fail() {
        let air = AllZeroAir::<2>;
        // Row 0: [1, 0] → constraint 0 fails
        // Row 1: [0, 1] → constraint 1 fails
        let values = vec![BabyBear::ONE, BabyBear::ZERO, BabyBear::ZERO, BabyBear::ONE];
        let main = RowMajorMatrix::new(values, 2);
        let report = check_all_constraints(&air, &main, &[], None);
        assert!(!report.is_ok());
        assert_eq!(report.failures.len(), 2);

        assert_eq!(report.failures[0].row, 0);
        assert_eq!(report.failures[0].constraint, 0);

        assert_eq!(report.failures[1].row, 1);
        assert_eq!(report.failures[1].constraint, 1);
    }

    #[test]
    fn test_check_all_constraints_max_failures_cap() {
        let air = AllZeroAir::<2>;
        // 4 rows, all non-zero → 8 failures total, but cap at 3.
        let values = vec![BabyBear::ONE; 8];
        let main = RowMajorMatrix::new(values, 2);
        let report = check_all_constraints(&air, &main, &[], Some(3));
        // With cap=3, we stop after collecting at least 3 failures.
        // Row 0 produces 2 failures, row 1 produces 2 more → we get 4
        // because the cap is checked *between* rows.
        assert!(report.failures.len() >= 3);
        assert!(report.failures.len() <= 4);
    }

    /// AIR that uses `assert_zero_named` for labeled constraints.
    #[derive(Debug)]
    struct NamedConstraintAir;

    impl<F: Field> BaseAir<F> for NamedConstraintAir {
        fn width(&self) -> usize {
            2
        }
    }

    impl<F: Field> Air<DebugConstraintBuilder<'_, F>> for NamedConstraintAir {
        fn eval(&self, builder: &mut DebugConstraintBuilder<'_, F>) {
            let main = builder.main();
            // Constraint 0: unlabeled
            builder.assert_zero(main.current(0).unwrap());
            // Constraint 1: labeled via NamedAirBuilder
            builder.assert_zero_named(main.current(1).unwrap(), "col_1_must_be_zero");
        }
    }

    #[test]
    fn test_named_constraint_label_captured() {
        let builder = eval_single_row(
            &NamedConstraintAir,
            [BabyBear::ONE, BabyBear::ONE], // both fail
        );
        let failures = builder.failures();
        assert_eq!(failures.len(), 2);

        // First failure: unlabeled.
        assert_eq!(failures[0].constraint, 0);
        assert!(failures[0].label.is_none());

        // Second failure: labeled.
        assert_eq!(failures[1].constraint, 1);
        assert_eq!(failures[1].label.as_deref(), Some("col_1_must_be_zero"));
    }

    #[test]
    fn test_named_constraint_no_label_when_passing() {
        let builder = eval_single_row(
            &NamedConstraintAir,
            [BabyBear::ZERO, BabyBear::ZERO], // both pass
        );
        assert!(!builder.has_failures());
    }

    #[test]
    fn test_named_constraint_in_full_report() {
        let air = NamedConstraintAir;
        // Two rows, both fail on both columns.
        let values = vec![BabyBear::ONE; 4];
        let main = RowMajorMatrix::new(values, 2);
        let report = check_all_constraints(&air, &main, &[], None);
        assert_eq!(report.failures.len(), 4);

        // Row 0, constraint 1 should have the label.
        let labeled: Vec<_> = report
            .failures
            .iter()
            .filter(|f| f.label.is_some())
            .collect();
        assert_eq!(labeled.len(), 2); // one per row
        assert!(
            labeled
                .iter()
                .all(|f| f.label.as_deref() == Some("col_1_must_be_zero"))
        );
    }

    /// AIR that exercises closure-based names and namespace composition.
    #[derive(Debug)]
    struct NamespacedAir;

    impl<F: Field> BaseAir<F> for NamespacedAir {
        fn width(&self) -> usize {
            3
        }
    }

    impl<F: Field> Air<DebugConstraintBuilder<'_, F>> for NamespacedAir {
        fn eval(&self, builder: &mut DebugConstraintBuilder<'_, F>) {
            use crate::NamespaceExt;

            let main = builder.main();
            let ns = "range_check";

            // Static namespace joined with static name.
            builder.assert_zero_named(main.current(0).unwrap(), ns.join("limb_0"));

            // Namespace joined with a closure name.
            let i = 1;
            builder.assert_zero_named(main.current(1).unwrap(), ns.name(|| format!("limb_{i}")));

            // Plain closure name (not a namespace).
            builder.assert_zero_named(main.current(2).unwrap(), || format!("col_{}", 2));
        }
    }

    #[test]
    fn test_namespace_join_labels() {
        let builder = eval_single_row(
            &NamespacedAir,
            [BabyBear::ONE, BabyBear::ONE, BabyBear::ONE],
        );
        let failures = builder.into_failures();
        assert_eq!(failures.len(), 3);

        assert_eq!(failures[0].label.as_deref(), Some("range_check::limb_0"));
        assert_eq!(failures[1].label.as_deref(), Some("range_check::limb_1"));
        assert_eq!(failures[2].label.as_deref(), Some("col_2"));
    }

    /// No-constraint AIR with a configurable preprocessed trace.
    ///
    /// Lets a test force a preprocessed shape independent of the main trace.
    #[derive(Debug)]
    struct ShapeProbeAir {
        /// Rows advertised in the preprocessed trace. `0` reports `None`.
        prep_height: usize,
        /// Columns of the advertised preprocessed trace. Ignored when height is `0`.
        prep_width: usize,
    }

    impl<F: Field> BaseAir<F> for ShapeProbeAir {
        fn width(&self) -> usize {
            // Single column; every fixture is `F::zero_vec(height)`.
            1
        }

        fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
            // Height == 0 is the sentinel for "AIR has no preprocessed trace".
            if self.prep_height == 0 {
                return None;
            }

            // Row-major flat buffer: height * width zero elements.
            //
            //     layout (prep_height = 2, prep_width = 3):
            //       row 0: [ 0, 0, 0 ]
            //       row 1: [ 0, 0, 0 ]
            //       flat : [ 0, 0, 0, 0, 0, 0 ]
            let total = self.prep_height * self.prep_width;
            Some(RowMajorMatrix::new(F::zero_vec(total), self.prep_width))
        }
    }

    impl<F: Field> Air<DebugConstraintBuilder<'_, F>> for ShapeProbeAir {
        fn eval(&self, _builder: &mut DebugConstraintBuilder<'_, F>) {
            // Empty: every panic must come from a pre-loop guard, not from eval.
        }
    }

    #[test]
    fn test_preprocessed_height_matches_main_passes() {
        // Invariant: matching heights → guard accepts → empty eval loop runs cleanly.
        //
        // Fixture state:
        //   main         : 4 rows × 1 col
        //   preprocessed : 4 rows × 1 col  (advertised by the AIR)
        //
        //     main rows : [0, 1, 2, 3]
        //     prep rows : [0, 1, 2, 3]
        //                 → 4 == 4 → guard passes
        let air = ShapeProbeAir {
            prep_height: 4,
            prep_width: 1,
        };

        // Zero-valued rows; content is irrelevant because no constraint reads it.
        let main = RowMajorMatrix::new(BabyBear::zero_vec(4), 1);

        // Must return cleanly. A panic here would mean the guard rejected a well-shaped input.
        check_constraints(&air, &main, &[]);
    }

    #[test]
    #[should_panic(expected = "preprocessed trace height")]
    fn test_preprocessed_height_mismatch_panics_in_check_constraints() {
        // Invariant: a taller preprocessed trace must trip the guard before
        // any `unsafe` row-indexing on the oversized matrix runs.
        //
        // Fixture state:
        //   main         : 4 rows × 1 col
        //   preprocessed : 8 rows × 1 col  (AIR advertises an oversized shape)
        //
        //     main rows : [0, 1, 2, 3]
        //     prep rows : [0, 1, 2, 3, 4, 5, 6, 7]
        //                 → 8 != 4 → guard panics on entry
        let air = ShapeProbeAir {
            prep_height: 8,
            prep_width: 1,
        };

        // Main deliberately shorter than the advertised preprocessed trace.
        let main = RowMajorMatrix::new(BabyBear::zero_vec(4), 1);

        // Expected: panic before row 0 is ever dereferenced.
        check_constraints(&air, &main, &[]);
    }

    #[test]
    fn test_preprocessed_height_matches_main_passes_in_check_all_constraints() {
        // Invariant: collect-all carries the same guard; matching heights also succeed.
        //
        // Fixture state:
        //   main         : 4 rows × 1 col
        //   preprocessed : 4 rows × 1 col  → empty report returned
        let air = ShapeProbeAir {
            prep_height: 4,
            prep_width: 1,
        };
        let main = RowMajorMatrix::new(BabyBear::zero_vec(4), 1);

        // No failure cap; we expect no failures anyway.
        let report = check_all_constraints(&air, &main, &[], None);

        // Empty AIR → no failures recorded.
        assert!(report.is_ok());

        // Loop still walked every row.
        assert_eq!(report.total_rows, 4);
    }

    /// The one cell the constraint-free AIR below lists as a public input.
    ///
    /// ```text
    ///     column 0, last row, carrying public value 0
    /// ```
    const BOUNDARY_IO_CELLS: [BoundaryPublic; 1] = [BoundaryPublic::new(0, BoundaryEnd::Last, 0)];

    /// AIR that binds its only public input by position, beside one constraint of its own.
    ///
    /// No constraint touches the cell.
    /// Only the boundary check can catch a mismatch.
    ///
    /// The one transition constraint is what gives the injected pin a nonzero index.
    #[derive(Debug)]
    struct BoundaryIoAir;

    impl<F: Field> BaseAir<F> for BoundaryIoAir {
        fn width(&self) -> usize {
            1
        }

        fn num_public_values(&self) -> usize {
            1
        }

        fn public_boundary_io(&self) -> &[BoundaryPublic] {
            &BOUNDARY_IO_CELLS
        }
    }

    impl<F: Field> Air<DebugConstraintBuilder<'_, F>> for BoundaryIoAir {
        fn eval(&self, builder: &mut DebugConstraintBuilder<'_, F>) {
            // The column counts up by one, which every fixture below satisfies.
            // The listed cell is bound by position, not by this or any other constraint.
            let main = builder.main();
            let local = main.current_slice()[0];
            let next = main.next_slice()[0];
            builder.when_transition().assert_eq(next, local + F::ONE);
        }
    }

    #[test]
    fn test_boundary_io_cell_matching_its_public_value_passes() {
        // Invariant: a listed cell holding its public value is accepted.
        //
        // Fixture state: 4 rows, 1 column, last row 7.
        //
        //     rows          : [4, 5, 6, 7]
        //     public values : [7]
        //                     → 7 == 7 → accepted
        let main = RowMajorMatrix::new([4u32, 5, 6, 7].map(BabyBear::new).to_vec(), 1);

        // Returns cleanly.
        // A panic would mean a well-formed fixture was rejected.
        check_constraints(&BoundaryIoAir, &main, &[BabyBear::new(7)]);
    }

    #[test]
    #[should_panic(expected = "does not hold public value")]
    fn test_boundary_io_cell_disagreeing_with_its_public_value_panics() {
        // Invariant: the check fires even though the AIR asserts nothing at all.
        //
        // Mutation: claim a public value the last row does not carry.
        //
        //     rows          : [4, 5, 6, 7]
        //     public values : [8]
        //                     → 7 != 8 → panic
        let main = RowMajorMatrix::new([4u32, 5, 6, 7].map(BabyBear::new).to_vec(), 1);

        // Expected: panic before the row loop starts.
        check_constraints(&BoundaryIoAir, &main, &[BabyBear::new(8)]);
    }

    #[test]
    fn test_boundary_io_mismatch_is_collected_rather_than_panicking() {
        // Invariant: the collecting variant reports a cell mismatch like any other failure.
        //
        // Mutation: claim a public value the last row does not carry.
        //
        //     rows          : [4, 5, 6, 7]
        //     public values : [8]
        //                     → one failure on row 3
        let main = RowMajorMatrix::new([4u32, 5, 6, 7].map(BabyBear::new).to_vec(), 1);

        let report = check_all_constraints(&BoundaryIoAir, &main, &[BabyBear::new(8)], None);

        // The pin takes the index after the AIR's own constraint, which is where the
        // backend injects it. Reporting it at `index` alone would put it at 0.
        assert_eq!(report.total_constraints_per_row, 1);
        assert_eq!(report.failures.len(), 1);
        assert_eq!(report.failures[0].row, 3);
        assert_eq!(report.failures[0].constraint, 1);
        assert!(
            report.failures[0]
                .label
                .as_deref()
                .is_some_and(|label| label.contains("public value 0"))
        );
    }

    #[test]
    fn test_boundary_io_mismatch_counts_against_the_failure_cap() {
        // Invariant: a cell failure is capped like any other, even though it is found
        //   after the row scan rather than inside it.
        //
        // Fixture state: the same wrong claim, with no room left to record it.
        let main = RowMajorMatrix::new([4u32, 5, 6, 7].map(BabyBear::new).to_vec(), 1);

        let report = check_all_constraints(&BoundaryIoAir, &main, &[BabyBear::new(8)], Some(0));

        assert!(report.failures.is_empty());
    }

    #[test]
    #[should_panic(expected = "preprocessed trace height")]
    fn test_preprocessed_height_mismatch_panics_in_check_all_constraints() {
        // Invariant: same mismatch as the single-pass case → collect-all panics the same way.
        //
        // Fixture state:
        //   main         : 4 rows × 1 col
        //   preprocessed : 8 rows × 1 col  → 8 != 4 → guard panics on entry
        let air = ShapeProbeAir {
            prep_height: 8,
            prep_width: 1,
        };
        let main = RowMajorMatrix::new(BabyBear::zero_vec(4), 1);

        // Expected: panic on entry. The would-be report is unreachable → bound to `_`.
        let _ = check_all_constraints(&air, &main, &[], None);
    }

    /// Single column pinned to a periodic column listing `0, 1, ..., period - 1`.
    #[derive(Debug)]
    struct PeriodicCopyAir {
        period: usize,
    }

    impl<F: Field> BaseAir<F> for PeriodicCopyAir {
        fn width(&self) -> usize {
            1
        }

        fn num_periodic_columns(&self) -> usize {
            1
        }

        fn periodic_columns(&self) -> Cow<'_, [Vec<F>]> {
            Cow::Owned(vec![(0..self.period).map(F::from_usize).collect()])
        }
    }

    impl<F: Field> Air<DebugConstraintBuilder<'_, F>> for PeriodicCopyAir {
        fn eval(&self, builder: &mut DebugConstraintBuilder<'_, F>) {
            let main = builder.main();
            let periodic = builder.periodic_values()[0];
            builder.assert_eq(main.current(0).unwrap(), periodic);
        }
    }

    // Trace holding `row mod period`, which is exactly what the AIR reads off the
    // periodic column, so every rejection below can only come from the shape check.
    fn periodic_copy_trace(period: usize, height: usize) -> RowMajorMatrix<BabyBear> {
        RowMajorMatrix::new_col(
            (0..height)
                .map(|i| BabyBear::from_usize(i % period))
                .collect(),
        )
    }

    #[test]
    fn test_periodic_column_with_valid_shape_passes() {
        // Three shapes that tile an 8-row trace, spanning the accepted range:
        //
        //     period 1 : [0][0][0][0][0][0][0][0]   constant column
        //     period 2 : [0,1][0,1][0,1][0,1]       the common case
        //     period 8 : [0,1,2,3,4,5,6,7]          period equal to the height
        for period in [1, 2, 8] {
            let air = PeriodicCopyAir { period };
            let trace = periodic_copy_trace(period, 8);

            check_constraints(&air, &trace, &[]);

            let report = check_all_constraints(&air, &trace, &[], None);
            assert!(report.failures.is_empty(), "period {period} was rejected");
        }
    }

    #[test]
    #[should_panic(expected = "periodic column 0 to have a power-of-two length, got 3")]
    fn test_periodic_column_length_not_power_of_two_is_rejected() {
        // Period 3 sits inside the 8 rows but has no subgroup of order 3 to interpolate over.
        let air = PeriodicCopyAir { period: 3 };
        check_constraints(&air, &periodic_copy_trace(3, 8), &[]);
    }

    #[test]
    #[should_panic(expected = "periodic column 0 to have a power-of-two length, got 0")]
    fn test_empty_periodic_column_is_rejected() {
        // An empty column would send the row lookup into `row mod 0`.
        // Zero is not a power of two, so the shape check names the column first.
        let air = PeriodicCopyAir { period: 0 };
        let trace = RowMajorMatrix::new_col(BabyBear::zero_vec(8));
        let _ = check_all_constraints(&air, &trace, &[], None);
    }

    #[test]
    #[should_panic(expected = "periodic column 0 (length 16) to divide the trace height (8)")]
    fn test_periodic_column_longer_than_trace_is_rejected() {
        // Only the first 8 of the 16 values are ever read, so the debug run would
        // score a different AIR than the one the prover builds.
        let air = PeriodicCopyAir { period: 16 };
        let _ = check_all_constraints(&air, &periodic_copy_trace(16, 8), &[], None);
    }

    #[test]
    #[should_panic(expected = "periodic column 0 (length 8) to divide the trace height (12)")]
    fn test_periodic_column_not_dividing_trace_is_rejected() {
        // Period 8 is a power of two and fits inside 12 rows, yet 12 = 8 + 4 leaves
        // a partial repetition, which the commitment layer rejects as well.
        let air = PeriodicCopyAir { period: 8 };
        check_constraints(&air, &periodic_copy_trace(8, 12), &[]);
    }
}
