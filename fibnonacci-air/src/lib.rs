use std::borrow::{Borrow, BorrowMut};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, PrimeField64};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::MatrixRowSlices;

const NUM_FIBONACCI_COLS: usize = 2;

pub struct FibonacciCol<F: AbstractField> {
    pub index: F,
    pub val: F,
}

impl<F: AbstractField> FibonacciCol<F> {
    fn new(index: F, val: F) -> FibonacciCol<F> {
        FibonacciCol { index, val }
    }
}

impl<F: AbstractField> Borrow<FibonacciCol<F>> for [F] {
    fn borrow(&self) -> &FibonacciCol<F> {
        debug_assert_eq!(self.len(), NUM_FIBONACCI_COLS);
        let (prefix, shorts, suffix) = unsafe { self.align_to::<FibonacciCol<F>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &shorts[0]
    }
}

impl<F: AbstractField> BorrowMut<FibonacciCol<F>> for [F] {
    fn borrow_mut(&mut self) -> &mut FibonacciCol<F> {
        debug_assert_eq!(self.len(), NUM_FIBONACCI_COLS);
        let (prefix, shorts, suffix) = unsafe { self.align_to_mut::<FibonacciCol<F>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &mut shorts[0]
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

        let [a_i, a_val] = main.row_slice(0) else { todo!() };
        let [b_i, b_val] = main.row_slice(1) else { todo!() };
        let [x_i, x_val] = main.row_slice(2) else { todo!() };

        builder.when_first_row().assert_zero(*a_i);

        builder.when_transition().assert_one(*b_i - *a_i);

        builder.when_transition_window(3).assert_eq(*a_val + *b_val, *x_val);
    }
}

pub fn generate_trace_rows<F: PrimeField64>(a: u64, b: u64, n: usize) -> RowMajorMatrix<F> {
    let num_rows = n + 1;
    let mut trace = RowMajorMatrix::new(vec![F::zero(); num_rows * NUM_FIBONACCI_COLS], NUM_FIBONACCI_COLS);

    let (prefix, rows, suffix) = unsafe {
        trace.values.align_to_mut::<FibonacciCol<F>>()
    };
    assert!(prefix.is_empty(), "Alignment should match");
    assert!(suffix.is_empty(), "Alignment should match");
    assert_eq!(rows.len(), num_rows);

    rows[0] = FibonacciCol::new(F::zero(), F::from_canonical_u64(a));
    rows[1] = FibonacciCol::new(F::one(), F::from_canonical_u64(b));

    let mut _a = a;
    let mut _b = b;
    for i in 2..n {
        let sum = _a + _b;
        rows[i].index = F::from_canonical_usize(i);
        rows[i].val = F::from_canonical_u64(sum);
        _a = _b;
        _b = sum
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
        assert_eq!(m.get(0, 0), BabyBear::from_canonical_u64(0));
        assert_eq!(m.get(1, 0), BabyBear::from_canonical_u64(1));
        assert_eq!(m.get(2, 0), BabyBear::from_canonical_u64(2));

        assert_eq!(m.get(0, 1), BabyBear::from_canonical_u64(1));
        assert_eq!(m.get(1, 1), BabyBear::from_canonical_u64(1));
        assert_eq!(m.get(2, 1), BabyBear::from_canonical_u64(2));
        assert_eq!(m.get(3, 1), BabyBear::from_canonical_u64(3));
    }
}
