use p3_air::{Air, DebugConstraintBuilder};
use p3_field::{ExtensionField, Field};
use p3_lookup::{Lookup, LookupProtocol};
use p3_matrix::Matrix;
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixView};
use p3_matrix::stack::VerticalPair;
use tracing::instrument;

/// Type alias for the inputs to lookup constraint checking.
/// - The first element is a slice of [`Lookup`] values (generic over a field `F`) representing the symbolic lookups to be performed.
/// - The second element is a reference to the [`LookupGadget`] implementation.
#[allow(unused)]
type LookupConstraintsInputs<'a, F, LG> = (&'a [Lookup<F>], &'a LG);

/// Runs constraint checks using a given [AIR](`p3_air::Air`) implementation and trace matrix.
///
/// Iterates over every row in `main`, providing both the current and next row
/// (with wraparound) to the [AIR](`p3_air::Air`) logic. Also injects public values into the builder
/// for first/last row assertions.
///
/// Collects all constraint failures for the first failing row, then panics with
/// a summary listing every violated constraint (index with optional label).
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
#[allow(clippy::too_many_arguments)]
pub(crate) fn check_constraints<'b, F, EF, A, LG>(
    air: &A,
    main: &RowMajorMatrix<F>,
    preprocessed: &Option<RowMajorMatrix<F>>,
    permutation: &RowMajorMatrix<EF>,
    permutation_challenges: &[EF],
    permutation_values: &[EF],
    public_values: &[F],
    lookup_constraints_inputs: LookupConstraintsInputs<'b, F, LG>,
) where
    F: Field,
    EF: ExtensionField<F>,
    A: for<'a> Air<DebugConstraintBuilder<'a, F, EF>>,
    LG: LookupProtocol,
{
    let height = main.height();
    if let Some(prep) = preprocessed.as_ref() {
        assert_eq!(
            prep.height(),
            height,
            "debug constraint check requires preprocessed trace height ({}) to match main trace height ({})",
            prep.height(),
            height
        );
    }
    assert_eq!(
        permutation.height(),
        height,
        "debug constraint check requires permutation trace height ({}) to match main trace height ({})",
        permutation.height(),
        height
    );

    let (lookups, lookup_gadget) = lookup_constraints_inputs;

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

        let preprocessed = match (prep_local.as_ref(), prep_next.as_ref()) {
            (Some(l), Some(n)) => VerticalPair::new(
                RowMajorMatrixView::new_row(&**l),
                RowMajorMatrixView::new_row(&**n),
            ),
            _ => VerticalPair::new(
                RowMajorMatrixView::new(&[], 0),
                RowMajorMatrixView::new(&[], 0),
            ),
        };

        let permutation = VerticalPair::new(
            RowMajorMatrixView::new_row(&*perm_local),
            RowMajorMatrixView::new_row(&*perm_next),
        );

        let periodic_row = air.periodic_values(row_index);
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
            permutation_values,
            &periodic_row,
        );

        lookup_gadget.eval_air_and_lookups(air, &mut builder, lookups);

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

#[cfg(test)]
mod tests {
    use p3_air::{Air, BaseAir, DebugConstraintBuilder};
    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;
    use p3_lookup::logup::LogUpGadget;

    use super::*;

    /// Quartic extension over the base field; matches the arity used in the rest of the test suite.
    type EF = BinomialExtensionField<BabyBear, 4>;

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

    impl<F, EF> Air<DebugConstraintBuilder<'_, F, EF>> for ShapeProbeAir
    where
        F: Field,
        EF: ExtensionField<F>,
    {
        fn eval(&self, _builder: &mut DebugConstraintBuilder<'_, F, EF>) {
            // Empty: every panic must come from a pre-loop guard, not from eval.
        }
    }

    #[test]
    fn test_matching_heights_pass_through() {
        // Invariant: all three traces share a height → both guards accept → empty loop runs.
        //
        // Fixture state:
        //   main         : 4 rows × 1 col  (base field)
        //   preprocessed : 4 rows × 1 col  (advertised by the AIR)
        //   permutation  : 4 rows × 1 col  (extension field)
        //
        //     row index :  0   1   2   3
        //     main      : [0] [0] [0] [0]
        //     prep      : [0] [0] [0] [0]
        //     perm      : [0] [0] [0] [0]
        //                 → all heights == 4 → guards pass
        let air = ShapeProbeAir {
            prep_height: 4,
            prep_width: 1,
        };

        // Zero-valued traces; content is irrelevant because eval reads nothing.
        let main: RowMajorMatrix<BabyBear> = RowMajorMatrix::new(BabyBear::zero_vec(4), 1);
        let preprocessed = <ShapeProbeAir as BaseAir<BabyBear>>::preprocessed_trace(&air);
        let permutation: RowMajorMatrix<EF> = RowMajorMatrix::new(EF::zero_vec(4), 1);

        // Must return cleanly. A panic here would mean a guard rejected a well-shaped input.
        check_constraints::<BabyBear, EF, _, LogUpGadget>(
            &air,
            &main,
            &preprocessed,
            &permutation,
            &[],
            &[],
            &[],
            (&[], &LogUpGadget),
        );
    }

    #[test]
    #[should_panic(expected = "preprocessed trace height")]
    fn test_preprocessed_height_mismatch_panics() {
        // Invariant: a taller preprocessed trace must trip the first pre-loop guard.
        //
        // Fixture state:
        //   main         : 4 rows × 1 col
        //   preprocessed : 8 rows × 1 col  (AIR advertises an oversized shape)
        //   permutation  : 4 rows × 1 col  (matches main deliberately)
        //
        //     main rows : [0, 1, 2, 3]
        //     prep rows : [0, 1, 2, 3, 4, 5, 6, 7]
        //     perm rows : [0, 1, 2, 3]
        //                 → prep guard trips first: 8 != 4
        //
        // Why permutation matches: a correct permutation isolates the preprocessed guard.
        let air = ShapeProbeAir {
            prep_height: 8,
            prep_width: 1,
        };
        let main: RowMajorMatrix<BabyBear> = RowMajorMatrix::new(BabyBear::zero_vec(4), 1);
        let preprocessed = <ShapeProbeAir as BaseAir<BabyBear>>::preprocessed_trace(&air);
        let permutation: RowMajorMatrix<EF> = RowMajorMatrix::new(EF::zero_vec(4), 1);

        // Expected: panic on entry, before any row is dereferenced.
        check_constraints::<BabyBear, EF, _, LogUpGadget>(
            &air,
            &main,
            &preprocessed,
            &permutation,
            &[],
            &[],
            &[],
            (&[], &LogUpGadget),
        );
    }

    #[test]
    #[should_panic(expected = "permutation trace height")]
    fn test_permutation_height_mismatch_panics() {
        // Invariant: the permutation guard is unconditional → fires regardless of preprocessed.
        //
        // Fixture state:
        //   main         : 4 rows × 1 col
        //   preprocessed : absent   (None — first guard is skipped entirely)
        //   permutation  : 8 rows × 1 col  (deliberately oversized)
        //
        //     main rows : [0, 1, 2, 3]
        //     perm rows : [0, 1, 2, 3, 4, 5, 6, 7]
        //                 → first guard skipped, second trips: 8 != 4
        //
        // Why preprocessed is absent: first guard skipped → panic must come from the second.
        let air = ShapeProbeAir {
            prep_height: 0,
            prep_width: 0,
        };
        let main: RowMajorMatrix<BabyBear> = RowMajorMatrix::new(BabyBear::zero_vec(4), 1);
        let preprocessed: Option<RowMajorMatrix<BabyBear>> = None;
        let permutation: RowMajorMatrix<EF> = RowMajorMatrix::new(EF::zero_vec(8), 1);

        // Expected: panic on entry, before any row is dereferenced.
        check_constraints::<BabyBear, EF, _, LogUpGadget>(
            &air,
            &main,
            &preprocessed,
            &permutation,
            &[],
            &[],
            &[],
            (&[], &LogUpGadget),
        );
    }
}
