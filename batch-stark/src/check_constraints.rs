use p3_air::{
    Air, AirBuilder, AirBuilderWithPublicValues, ExtensionBuilder, PairBuilder,
    PermutationAirBuilder,
};
use p3_field::{ExtensionField, Field};
use p3_lookup::lookup_traits::{Lookup, LookupData, LookupGadget};
use p3_matrix::Matrix;
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixView};
use p3_matrix::stack::{VerticalPair, ViewPair};
use tracing::instrument;

/// Type alias for the inputs to lookup constraint checking.
/// - The first element is a slice of [`Lookup`] values (generic over a field `F`) representing the symbolic lookups to be performed.
/// - The second element is a slice of [`LookupData`] values (generic over an extension field `EF`) representing the lookup data for global lookups.
/// - The third element is a reference to the [`LookupGadget`] implementation.
#[allow(unused)]
type LookupConstraintsInputs<'a, F, EF, LG> = (&'a [Lookup<F>], &'a [LookupData<EF>], &'a LG);

/// Runs constraint checks using a given [AIR](`p3_air::Air`) implementation and trace matrix.
///
/// Iterates over every row in `main`, providing both the current and next row
/// (with wraparound) to the [AIR](`p3_air::Air`) logic. Also injects public values into the builder
/// for first/last row assertions.
///
/// # Arguments
/// - `air`: The [AIR](`p3_air::Air`) logic to run.
/// - `main`: The [`RowMajorMatrix`] containing rows of witness values.
/// - `permutation`: The permutation [`RowMajorMatrix`] (rows of permutation values).
/// - `permutation_challenges`: The challenges used for the permutation argument.
/// - `public_values`: Public values provided to the builder.
/// - `lookup_constraints_inputs`: Inputs necessary to check lookup constraints:
///     - the symbolic representation of the [`Lookup`] values,
///     - the [`LookupData`] for global lookups,
///     - the [`LookupGadget`] implementation.
#[instrument(name = "check constraints", skip_all)]
#[allow(unused)]
pub(crate) fn check_constraints<'b, F, EF, A, LG>(
    air: &A,
    main: &RowMajorMatrix<F>,
    preprocessed: &Option<RowMajorMatrix<F>>,
    permutation: &RowMajorMatrix<EF>,
    permutation_challenges: &[EF],
    public_values: &[F],
    lookup_constraints_inputs: LookupConstraintsInputs<'b, F, EF, LG>,
) where
    F: Field,
    EF: ExtensionField<F>,
    A: for<'a> Air<DebugConstraintBuilderWithLookups<'a, F, EF>>,
    LG: LookupGadget,
{
    let height = main.height();

    let (lookups, lookup_data, lookup_gadget) = lookup_constraints_inputs;

    (0..height).for_each(|row_index| {
        let row_index_next = (row_index + 1) % height;

        // Safety:
        // - row_index < height so we can use unchecked indexing.
        // - row_index_next < height so we can use unchecked indexing.
        let (local, next, prep_local, prep_next, perm_local, perm_next) = unsafe {
            (
                main.row_slice_unchecked(row_index),
                main.row_slice_unchecked(row_index_next),
                preprocessed
                    .as_ref()
                    .map(|p| p.row_slice_unchecked(row_index)),
                preprocessed
                    .as_ref()
                    .map(|p| p.row_slice_unchecked(row_index_next)),
                permutation.row_slice_unchecked(row_index),
                permutation.row_slice_unchecked(row_index_next),
            )
        };
        let main = VerticalPair::new(
            RowMajorMatrixView::new_row(&*local),
            RowMajorMatrixView::new_row(&*next),
        );

        let preprocessed_rows_data = prep_local.as_ref().zip(prep_next.as_ref());
        let preprocessed = preprocessed_rows_data.map(|(prep_local, prep_next)| {
            VerticalPair::new(
                RowMajorMatrixView::new_row(&**prep_local),
                RowMajorMatrixView::new_row(&**prep_next),
            )
        });

        let permutation = VerticalPair::new(
            RowMajorMatrixView::new_row(&*perm_local),
            RowMajorMatrixView::new_row(&*perm_next),
        );

        let mut builder = DebugConstraintBuilderWithLookups {
            row_index,
            main,
            preprocessed,
            permutation,
            permutation_challenges,
            public_values,
            is_first_row: F::from_bool(row_index == 0),
            is_last_row: F::from_bool(row_index == height - 1),
            is_transition: F::from_bool(row_index != height - 1),
        };

        <A as Air<DebugConstraintBuilderWithLookups<'_, F, EF>>>::eval_with_lookups(
            air,
            &mut builder,
            lookups,
            lookup_data,
            lookup_gadget,
        );
    });
}

/// A builder that runs constraint assertions during testing.
///
/// Used in conjunction with [`check_constraints`] to simulate
/// an execution trace and verify that the [AIR](`p3_air::Air`) logic enforces all constraints.
#[derive(Debug)]
#[allow(unused)]
pub struct DebugConstraintBuilderWithLookups<'a, F: Field, EF: ExtensionField<F>> {
    /// The index of the row currently being evaluated.
    row_index: usize,
    /// A view of the current and next row as a vertical pair.
    main: ViewPair<'a, F>,
    /// A view of the current and next preprocessed row as a vertical pair.
    preprocessed: Option<ViewPair<'a, F>>,
    /// The public values provided for constraint validation (e.g. inputs or outputs).
    public_values: &'a [F],
    /// A flag indicating whether this is the first row.
    is_first_row: F,
    /// A flag indicating whether this is the last row.
    is_last_row: F,
    /// A flag indicating whether this is a transition row (not the last row).
    is_transition: F,
    /// A view of the current and next permutation rows as a vertical pair.
    permutation: ViewPair<'a, EF>,
    /// The challenges used for the permutation argument.
    permutation_challenges: &'a [EF],
}

impl<'a, F, EF> AirBuilder for DebugConstraintBuilderWithLookups<'a, F, EF>
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

    fn is_first_row(&self) -> Self::Expr {
        self.is_first_row
    }

    fn is_last_row(&self) -> Self::Expr {
        self.is_last_row
    }

    /// # Panics
    /// This function panics if `size` is not `2`.
    fn is_transition_window(&self, size: usize) -> Self::Expr {
        if size == 2 {
            self.is_transition
        } else {
            panic!("only supports a window size of 2")
        }
    }

    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        assert_eq!(
            x.into(),
            F::ZERO,
            "constraints had nonzero value on row {}",
            self.row_index
        );
    }

    fn assert_eq<I1: Into<Self::Expr>, I2: Into<Self::Expr>>(&mut self, x: I1, y: I2) {
        let x = x.into();
        let y = y.into();
        assert_eq!(
            x, y,
            "values didn't match on row {}: {} != {}",
            self.row_index, x, y
        );
    }
}

impl<F: Field, EF: ExtensionField<F>> AirBuilderWithPublicValues
    for DebugConstraintBuilderWithLookups<'_, F, EF>
{
    type PublicVar = Self::F;

    fn public_values(&self) -> &[Self::F] {
        self.public_values
    }
}

impl<'a, F: Field, EF: ExtensionField<F>> ExtensionBuilder
    for DebugConstraintBuilderWithLookups<'a, F, EF>
{
    type EF = EF;

    type ExprEF = EF;

    type VarEF = EF;

    fn assert_zero_ext<I>(&mut self, x: I)
    where
        I: Into<Self::ExprEF>,
    {
        assert_eq!(
            x.into(),
            EF::ZERO,
            "constraints had nonzero value on row {}",
            self.row_index
        );
    }
}

impl<'a, F: Field, EF: ExtensionField<F>> PermutationAirBuilder
    for DebugConstraintBuilderWithLookups<'a, F, EF>
{
    type MP = VerticalPair<RowMajorMatrixView<'a, EF>, RowMajorMatrixView<'a, EF>>;

    type RandomVar = EF;

    fn permutation(&self) -> Self::MP {
        self.permutation
    }

    fn permutation_randomness(&self) -> &[Self::RandomVar] {
        self.permutation_challenges
    }
}

impl<'a, F: Field, EF: ExtensionField<F>> PairBuilder
    for DebugConstraintBuilderWithLookups<'a, F, EF>
{
    fn preprocessed(&self) -> Self::M {
        self.preprocessed
            .unwrap_or_else(|| panic!("Missing preprocessed columns"))
    }
}
