use alloc::vec::Vec;

use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues};
use p3_field::Field;
use p3_matrix::Matrix;
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixView};
use p3_matrix::stack::VerticalPair;
use tracing::instrument;

/// Runs constraint checks using a given AIR definition and trace matrix.
///
/// Iterates over every row in `main`, providing both the current and next row
/// (with wraparound) to the AIR logic. Also injects public values into the builder
/// for first/last row assertions.
///
/// # Arguments
/// - `air`: The AIR logic to run
/// - `main`: The trace matrix (rows of witness values)
/// - `public_values`: Public values provided to the builder
#[instrument(name = "check constraints", skip_all)]
pub(crate) fn check_constraints<F, A>(air: &A, main: &RowMajorMatrix<F>, public_values: &Vec<F>)
where
    F: Field,
    A: for<'a> Air<DebugConstraintBuilder<'a, F>>,
{
    let height = main.height();

    (0..height).for_each(|i| {
        let i_next = (i + 1) % height;

        let local = main.row_slice(i);
        let next = main.row_slice(i_next);
        let main = VerticalPair::new(
            RowMajorMatrixView::new_row(&*local),
            RowMajorMatrixView::new_row(&*next),
        );

        let mut builder = DebugConstraintBuilder {
            row_index: i,
            main,
            public_values,
            is_first_row: F::from_bool(i == 0),
            is_last_row: F::from_bool(i == height - 1),
            is_transition: F::from_bool(i != height - 1),
        };

        air.eval(&mut builder);
    });
}

/// A builder that runs constraint assertions during testing.
///
/// Used in conjunction with [`check_constraints`] to simulate
/// an execution trace and verify that the AIR logic enforces all constraints.
#[derive(Debug)]
pub struct DebugConstraintBuilder<'a, F: Field> {
    /// The index of the row currently being evaluated.
    row_index: usize,
    /// A view of the current and next row as a vertical pair.
    main: VerticalPair<RowMajorMatrixView<'a, F>, RowMajorMatrixView<'a, F>>,
    /// The public values provided for constraint validation (e.g. inputs or outputs).
    public_values: &'a [F],
    /// A flag indicating whether this is the first row.
    is_first_row: F,
    /// A flag indicating whether this is the last row.
    is_last_row: F,
    /// A flag indicating whether this is a transition row (not the last row).
    is_transition: F,
}

impl<'a, F> AirBuilder for DebugConstraintBuilder<'a, F>
where
    F: Field,
{
    type F = F;
    type Expr = F;
    type Var = F;
    type M = VerticalPair<RowMajorMatrixView<'a, F>, RowMajorMatrixView<'a, F>>;

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

impl<F: Field> AirBuilderWithPublicValues for DebugConstraintBuilder<'_, F> {
    type PublicVar = Self::F;

    fn public_values(&self) -> &[Self::F] {
        self.public_values
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_air::{BaseAir, BaseAirWithPublicValues};
    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;

    use super::*;

    /// A test AIR that enforces a simple linear transition logic:
    /// - Each cell in the next row must equal the current cell plus 1 (i.e., `next = current + 1`)
    /// - On the last row, the current row must match the provided public values.
    ///
    /// This is useful for validating constraint evaluation, transition logic,
    /// and row condition flags (first/last/transition).
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

            for col in 0..W {
                let a = main.top.get(0, col);
                let b = main.bottom.get(0, col);

                // New logic: enforce row[i+1] = row[i] + 1, only on transitions
                builder.when_transition().assert_eq(b, a + F::ONE);
            }

            // Add public value equality on last row for extra coverage
            let public_values = builder.public_values;
            let mut when_last = builder.when(builder.is_last_row);
            for (i, &pv) in public_values.iter().enumerate().take(W) {
                when_last.assert_eq(main.top.get(0, i), pv);
            }
        }
    }

    #[test]
    fn test_incremental_rows_with_last_row_check() {
        // Each row = previous + 1, with 4 rows total, 2 columns.
        // Last row must match public values [4, 4]
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
        check_constraints(&air, &main, &vec![BabyBear::new(4); 2]);
    }

    #[test]
    #[should_panic]
    fn test_incorrect_increment_logic() {
        // Row 2 does not equal row 1 + 1 → should fail on transition from row 1 to 2.
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
        check_constraints(&air, &main, &vec![BabyBear::new(6); 2]);
    }

    #[test]
    #[should_panic]
    fn test_wrong_last_row_public_value() {
        // The transition logic is fine, but public value check fails at the last row.
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
        // Wrong public value on column 1
        check_constraints(&air, &main, &vec![BabyBear::new(4), BabyBear::new(5)]);
    }

    #[test]
    fn test_single_row_wraparound_logic() {
        // A single-row matrix still performs a wraparound check with itself.
        // row[0] == row[0] + 1 ⇒ fails unless handled properly by transition logic.
        // Here: is_transition == false ⇒ so no assertions are enforced.
        let air = RowLogicAir::<2>;
        let values = vec![
            BabyBear::new(99),
            BabyBear::new(77), // Row 0
        ];
        let main = RowMajorMatrix::new(values, 2);
        check_constraints(&air, &main, &vec![BabyBear::new(99), BabyBear::new(77)]);
    }
}
