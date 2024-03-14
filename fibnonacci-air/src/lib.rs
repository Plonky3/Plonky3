use std::borrow::{Borrow, BorrowMut};

use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, PrimeField64};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::MatrixRowSlices;

const NUM_FIBONACCI_COLS: usize = 3;

pub struct FibonacciRow<F> {
    pub index: F,
    pub left: F,
    pub right: F,
}

impl<F> FibonacciRow<F> {
    fn new(index: F, left: F, right: F) -> FibonacciRow<F> {
        FibonacciRow { index, left, right }
    }
}

impl<F> Borrow<FibonacciRow<F>> for [F] {
    fn borrow(&self) -> &FibonacciRow<F> {
        debug_assert_eq!(self.len(), NUM_FIBONACCI_COLS);
        let (prefix, shorts, suffix) = unsafe { self.align_to::<FibonacciRow<F>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &shorts[0]
    }
}

pub struct FibonacciAir {}

impl<F> BaseAir<F> for FibonacciAir {
    fn width(&self) -> usize {
        NUM_FIBONACCI_COLS
    }
}

impl<AB: AirBuilder> Air<AB> for FibonacciAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let local: &FibonacciRow<AB::Var> = main.row_slice(0).borrow();
        let next: &FibonacciRow<AB::Var> = main.row_slice(1).borrow();

        builder.when_first_row().assert_zero(local.index);

        let mut when_transition = builder.when_transition();

        // assert index increment by 1 for each row. index is needed because we want to prove that
        // "the n-th element in fib is x", but I'm still not sure how to constrain public inputs
        // onto the index and initial a, b values.
        when_transition.assert_one(next.index - local.index);

        // a' <- b
        when_transition.assert_eq(local.right, next.left);

        // b' <- a + b
        when_transition.assert_eq(local.left + local.right, next.right);
    }
}

pub fn generate_trace_rows<F: PrimeField64>(a: u64, b: u64, n: usize) -> RowMajorMatrix<F> {
    let num_rows = n + 1;

    // the length of the matrix row is num_rows * NUM_FIBONACCI_COLS because later we are going to
    // "window" the trace slice by the number of fields defined in FibonacciRow
    let mut trace =
        RowMajorMatrix::new(vec![F::zero(); num_rows * NUM_FIBONACCI_COLS], NUM_FIBONACCI_COLS);

    // This uses rust's `transmute` to "align" the struct `FibonacciRow` onto a window of trace values
    let (prefix, rows, suffix) = unsafe {
        trace.values.align_to_mut::<FibonacciRow<F>>()
    };
    assert!(prefix.is_empty(), "Alignment should match");
    assert!(suffix.is_empty(), "Alignment should match");
    assert_eq!(rows.len(), num_rows);

    rows[0] = FibonacciRow::new(F::zero(), F::from_canonical_u64(a), F::from_canonical_u64(b));

    for i in 1..num_rows {
        rows[i].index = F::from_canonical_usize(i);
        rows[i].left = rows[i - 1].right;
        rows[i].right = rows[i - 1].left + rows[i - 1].right;
    }

    trace
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_matrix::MatrixGet;

    use super::*;

    #[test]
    fn generate() {
        let m = generate_trace_rows::<BabyBear>(1, 1, 3);
        let f = BabyBear::from_canonical_u64;

        // indices
        assert_eq!(m.get(0, 0), f(0));
        assert_eq!(m.get(1, 0), f(1));
        assert_eq!(m.get(2, 0), f(2));

        // left right values
        assert_eq!(m.get(0, 1), f(1));
        assert_eq!(m.get(0, 2), f(1)); // 1 1

        assert_eq!(m.get(1, 1), f(1));
        assert_eq!(m.get(1, 2), f(2)); // 1 2

        assert_eq!(m.get(2, 1), f(2));
        assert_eq!(m.get(2, 2), f(3)); // 2 3
    }
}
