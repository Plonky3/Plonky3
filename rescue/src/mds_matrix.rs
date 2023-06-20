use p3_field::{AbstractField, PrimeField};
use p3_mersenne_31::Mersenne31;
use p3_symmetric::permutation::{ArrayPermutation, CryptographicPermutation, MDSPermutation};

struct MDSMatrix<F: PrimeField, const WIDTH: usize> {
    matrix: [[F; WIDTH]; WIDTH],
}

impl<F: PrimeField, const WIDTH: usize> CryptographicPermutation<[F; WIDTH]>
    for MDSMatrix<F, WIDTH>
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

impl<F: PrimeField, const WIDTH: usize> ArrayPermutation<F, WIDTH> for MDSMatrix<F, WIDTH> {}

impl<F: PrimeField, const WIDTH: usize> MDSPermutation<F, WIDTH> for MDSMatrix<F, WIDTH> {}

// Generated with SageMath, using the get_mds_matrix function from the Rescue-Prime paper.
const RESCUE_PRIME_MERSENNE_31_WIDTH_8_MDS_MATRIX: [[Mersenne31; 8]; 8] = [
    [Mersenne31::ZERO; 8],
    [Mersenne31::ZERO; 8],
    [Mersenne31::ZERO; 8],
    [Mersenne31::ZERO; 8],
    [Mersenne31::ZERO; 8],
    [Mersenne31::ZERO; 8],
    [Mersenne31::ZERO; 8],
    [Mersenne31::ZERO; 8],
];
