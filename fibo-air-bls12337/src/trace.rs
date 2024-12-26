use crate::AIR_WIDTH;
use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;

pub fn generate_fibonacci_trace<F: Field>(num_steps: usize) -> RowMajorMatrix<F> {
    let mut a = F::ZERO;
    let mut b = F::ONE;

    let mut trace = RowMajorMatrix::default(AIR_WIDTH, num_steps);

    for i in 0..num_steps {
        trace.row_mut(i)[0] = a;
        trace.row_mut(i)[1] = b;
        let c = a + b;
        a = b;
        b = c;
    }

    trace
}
