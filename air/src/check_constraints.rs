use alloc::vec::Vec;

use p3_field::{ExtensionField, Field};
use p3_matrix::Matrix;
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixView};
use p3_matrix::stack::{VerticalPair, ViewPair};

use crate::{
    Air, AirBuilder, AirBuilderWithContext, AirBuilderWithPublicValues, ExtensionBuilder,
    PermutationAirBuilder,
};

/// A single constraint violation captured during debug evaluation.
///
/// Instead of panicking on the first failure, the builder records every
/// violation it encounters so the caller can report them all at once.
#[derive(Debug, Clone)]
pub struct ConstraintFailure {
    /// Zero-based index of the trace row where the violation occurred.
    pub row: usize,

    /// Zero-based index of the constraint (in evaluation order) that was violated.
    pub constraint: usize,
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

    /// Vertical pair for the current and next preprocessed rows, if the
    /// AIR declares a preprocessed trace.
    preprocessed: Option<ViewPair<'a, F>>,

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
}

impl<'a, F: Field> DebugConstraintBuilder<'a, F> {
    /// Build a constraint checker for AIRs that do not use permutations.
    ///
    /// Permutation-related fields are set to `None` / empty so that the
    /// builder can still satisfy trait bounds that require extension-field
    /// support, but calling permutation accessors will panic.
    pub const fn new(
        row_index: usize,
        main: ViewPair<'a, F>,
        preprocessed: Option<ViewPair<'a, F>>,
        public_values: &'a [F],
        is_first_row: F,
        is_last_row: F,
        is_transition: F,
    ) -> Self {
        Self {
            row_index,
            constraint_index: 0,
            failures: Vec::new(),
            main,
            preprocessed,
            public_values,
            is_first_row,
            is_last_row,
            is_transition,
            permutation: None,
            permutation_challenges: &[],
        }
    }
}

impl<'a, F: Field, EF: ExtensionField<F>> DebugConstraintBuilder<'a, F, EF> {
    /// Build a constraint checker that also carries permutation data.
    ///
    /// Use this when the AIR declares lookup or permutation arguments
    /// that require access to the permutation trace and challenges.
    #[allow(clippy::too_many_arguments)]
    pub const fn new_with_permutation(
        row_index: usize,
        main: ViewPair<'a, F>,
        preprocessed: Option<ViewPair<'a, F>>,
        public_values: &'a [F],
        is_first_row: F,
        is_last_row: F,
        is_transition: F,
        permutation: ViewPair<'a, EF>,
        permutation_challenges: &'a [EF],
    ) -> Self {
        Self {
            row_index,
            constraint_index: 0,
            failures: Vec::new(),
            main,
            preprocessed,
            public_values,
            is_first_row,
            is_last_row,
            is_transition,
            permutation: Some(permutation),
            permutation_challenges,
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
    type M = ViewPair<'a, F>;

    fn main(&self) -> Self::M {
        self.main
    }

    fn preprocessed(&self) -> Option<Self::M> {
        self.preprocessed
    }

    fn is_first_row(&self) -> Self::Expr {
        self.is_first_row
    }

    fn is_last_row(&self) -> Self::Expr {
        self.is_last_row
    }

    /// # Panics
    ///
    /// Panics when `size` is not `2`, since this builder only supports
    /// two-row transition windows.
    fn is_transition_window(&self, size: usize) -> Self::Expr {
        if size == 2 {
            self.is_transition
        } else {
            panic!("only supports a window size of 2")
        }
    }

    /// Check that the expression evaluates to zero.
    ///
    /// If the value is non-zero a failure is recorded; in either case
    /// the constraint counter advances so that every constraint keeps
    /// a stable index.
    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        if x.into() != F::ZERO {
            self.failures.push(ConstraintFailure {
                row: self.row_index,
                constraint: self.constraint_index,
            });
        }
        self.constraint_index += 1;
    }
}

impl<F: Field, EF: ExtensionField<F>> AirBuilderWithPublicValues
    for DebugConstraintBuilder<'_, F, EF>
{
    type PublicVar = F;

    fn public_values(&self) -> &[Self::F] {
        self.public_values
    }
}

impl<F: Field, EF: ExtensionField<F>> AirBuilderWithContext for DebugConstraintBuilder<'_, F, EF> {
    /// No extra context is needed during debug evaluation.
    type EvalContext = ();

    fn eval_context(&self) -> &Self::EvalContext {
        &()
    }
}

impl<'a, F: Field, EF: ExtensionField<F>> ExtensionBuilder for DebugConstraintBuilder<'a, F, EF> {
    type EF = EF;
    type ExprEF = EF;
    type VarEF = EF;

    /// Same semantics as the base-field version: record a failure when
    /// the extension-field expression is non-zero, then advance the
    /// constraint counter.
    fn assert_zero_ext<I>(&mut self, x: I)
    where
        I: Into<Self::ExprEF>,
    {
        if x.into() != EF::ZERO {
            self.failures.push(ConstraintFailure {
                row: self.row_index,
                constraint: self.constraint_index,
            });
        }
        self.constraint_index += 1;
    }
}

impl<'a, F: Field, EF: ExtensionField<F>> PermutationAirBuilder
    for DebugConstraintBuilder<'a, F, EF>
{
    type MP = VerticalPair<RowMajorMatrixView<'a, EF>, RowMajorMatrixView<'a, EF>>;
    type RandomVar = EF;

    /// # Panics
    ///
    /// Panics when the builder was created without permutation data.
    fn permutation(&self) -> Self::MP {
        self.permutation
            .expect("permutation() called on a builder created without permutation data; use new_with_permutation()")
    }

    fn permutation_randomness(&self) -> &[Self::RandomVar] {
        self.permutation_challenges
    }
}

/// Evaluate every AIR constraint against a concrete trace and panic on failure.
///
/// The function walks the trace row by row. For each row it:
///
/// 1. Builds a vertical pair of the current and next rows (wrapping around
///    at the end).
/// 2. Sets the first-row, last-row and transition selectors.
/// 3. Evaluates the AIR, collecting all violated constraint indices.
/// 4. Stops at the first row that has at least one violation and panics
///    with a summary of every violated constraint on that row.
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

        // Pair the preprocessed rows if the AIR provides a preprocessed trace.
        //
        // The slices must be bound outside the closure so the borrows
        // outlive the `ViewPair` that references them.
        let (prep_local, prep_next) = preprocessed.as_ref().map_or((None, None), |prep| unsafe {
            // SAFETY: same index range as the main trace.
            (
                Some(prep.row_slice_unchecked(row_index)),
                Some(prep.row_slice_unchecked(row_index_next)),
            )
        });
        let preprocessed_pair = prep_local.as_ref().zip(prep_next.as_ref()).map(|(l, n)| {
            ViewPair::new(
                RowMajorMatrixView::new_row(&**l),
                RowMajorMatrixView::new_row(&**n),
            )
        });

        // Construct the builder with row selectors derived from the position.
        let mut builder = DebugConstraintBuilder::new(
            row_index,
            main_pair,
            preprocessed_pair,
            public_values,
            F::from_bool(row_index == 0),
            F::from_bool(row_index == height - 1),
            F::from_bool(row_index != height - 1),
        );

        // Run every AIR constraint on this row.
        air.eval(&mut builder);

        // Stop at the first failing row and report all violations at once.
        if builder.has_failures() {
            let indices: Vec<usize> = builder.failures().iter().map(|f| f.constraint).collect();
            panic!(
                "constraints not satisfied on row {row_index}: \
                 failed constraint indices = {indices:?}"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;

    use super::*;
    use crate::{BaseAir, BaseAirWithPublicValues};

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

    impl<F: Field, const W: usize> BaseAirWithPublicValues<F> for RowLogicAir<W> {}

    impl<F: Field, const W: usize> Air<DebugConstraintBuilder<'_, F>> for RowLogicAir<W> {
        fn eval(&self, builder: &mut DebugConstraintBuilder<'_, F>) {
            let main = builder.main();

            // Transition constraint: next == current + 1 for every column.
            for col in 0..W {
                let current = main.top.get(0, col).unwrap();
                let next = main.bottom.get(0, col).unwrap();
                builder.when_transition().assert_eq(next, current + F::ONE);
            }

            // Boundary constraint: last row matches public values.
            let public_values = builder.public_values;
            let mut when_last = builder.when(builder.is_last_row);
            for (i, &pv) in public_values.iter().enumerate().take(W) {
                when_last.assert_eq(main.top.get(0, i).unwrap(), pv);
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

        let mut builder = DebugConstraintBuilder::new(
            0,
            main,
            None,
            &[],
            BabyBear::ONE,  // is_first_row
            BabyBear::ONE,  // is_last_row (single row)
            BabyBear::ZERO, // is_transition
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
                builder.assert_zero(main.top.get(0, col).unwrap());
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
    #[should_panic(expected = "failed constraint indices = [0, 2]")]
    fn test_panic_message_lists_all_failed_indices() {
        let air = AllZeroAir::<3>;
        let values = vec![BabyBear::ONE, BabyBear::ZERO, BabyBear::new(7)];
        let main = RowMajorMatrix::new(values, 3);
        check_constraints(&air, &main, &[]);
    }
}
