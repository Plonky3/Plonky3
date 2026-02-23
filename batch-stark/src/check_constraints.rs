use alloc::vec::Vec;

use p3_air::{Air, DebugConstraintBuilder};
use p3_field::{ExtensionField, Field};
use p3_lookup::lookup_traits::{Lookup, LookupData, LookupGadget};
use p3_matrix::Matrix;
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixView};
use p3_matrix::stack::VerticalPair;
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
/// Collects all constraint failures for the first failing row, then panics with
/// a summary listing every violated constraint index.
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
#[allow(unused)] // Do not remove, or this will trigger warnings in release mode.
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
    A: for<'a> Air<DebugConstraintBuilder<'a, F, EF>>,
    LG: LookupGadget,
{
    let height = main.height();

    let (lookups, lookup_data, lookup_gadget) = lookup_constraints_inputs;

    for row_index in 0..height {
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

        let mut builder = DebugConstraintBuilder::new_with_permutation(
            row_index,
            main,
            preprocessed,
            public_values,
            F::from_bool(row_index == 0),
            F::from_bool(row_index == height - 1),
            F::from_bool(row_index != height - 1),
            permutation,
            permutation_challenges,
        );

        air.eval_with_lookups(&mut builder, lookups, lookup_data, lookup_gadget);

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
