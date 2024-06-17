use p3_field::Field;
use p3_matrix::{bitrev::BitReversableMatrix, Matrix};

pub trait FftAlgorithm<F: Field>: Clone + Default {
    type Coeffs: BitReversableMatrix<F> + 'static;
    type Evals: BitReversableMatrix<F> + 'static;

    fn interpolate(&self, evals: impl Matrix<F>) -> Self::Coeffs;
    fn evaluate(&self, coeffs: Self::Coeffs) -> Self::Evals;
}
