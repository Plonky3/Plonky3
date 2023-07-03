use crate::permutation::{ArrayPermutation, CryptographicPermutation};
use p3_field::PrimeField;

pub trait MDSPermutation<T, const WIDTH: usize>: ArrayPermutation<T, WIDTH> {}

pub struct MDSMatrixNaive<F: PrimeField, const WIDTH: usize> {
    matrix: [[F; WIDTH]; WIDTH],
}

impl<F: PrimeField, const WIDTH: usize> MDSMatrixNaive<F, WIDTH> {
    pub fn new(matrix: [[F; WIDTH]; WIDTH]) -> Self {
        Self { matrix }
    }
}

impl<F: PrimeField, const WIDTH: usize> CryptographicPermutation<[F; WIDTH]>
    for MDSMatrixNaive<F, WIDTH>
{
    fn permute(&self, input: [F; WIDTH]) -> [F; WIDTH] {
        let mut output = [F::ZERO; WIDTH];
        for (i, row) in self.matrix.iter().enumerate() {
            for (j, &x) in row.iter().enumerate() {
                output[i] += input[j] * x;
            }
        }
        output
    }
}

impl<F: PrimeField, const WIDTH: usize> ArrayPermutation<F, WIDTH> for MDSMatrixNaive<F, WIDTH> {}

impl<F: PrimeField, const WIDTH: usize> MDSPermutation<F, WIDTH> for MDSMatrixNaive<F, WIDTH> {}
