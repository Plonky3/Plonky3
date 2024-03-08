use alloc::vec::Vec;
use core::array;
use core::ops::{AddAssign, Mul};

use p3_dft::TwoAdicSubgroupDft;
use p3_field::{AbstractField, TwoAdicField};

// NB: These are all MDS for M31, BabyBear and Goldilocks
// const MATRIX_CIRC_MDS_8_2EXP: [u64; 8] = [1, 1, 2, 1, 8, 32, 4, 256];
// const MATRIX_CIRC_MDS_8_SML: [u64; 8] = [4, 1, 2, 9, 10, 5, 1, 1];
// Much smaller: [1, 1, -1, 2, 3, 8, 2, -3] but need to deal with the -ve's

// const MATRIX_CIRC_MDS_12_2EXP: [u64; 12] = [1, 1, 2, 1, 8, 32, 2, 256, 4096, 8, 65536, 1024];
// const MATRIX_CIRC_MDS_12_SML: [u64; 12] = [9, 7, 4, 1, 16, 2, 256, 128, 3, 32, 1, 1];
// const MATRIX_CIRC_MDS_12_SML: [u64; 12] = [1, 1, 2, 1, 8, 9, 10, 7, 5, 9, 4, 10];

// Trying to maximise the # of 1's in the vector.
// Not clear exactly what we should be optimising here but that seems reasonable.
// const MATRIX_CIRC_MDS_16_SML: [u64; 16] =
//   [1, 1, 51, 1, 11, 17, 2, 1, 101, 63, 15, 2, 67, 22, 13, 3];
// 1, 1, 51, 52, 11, 63, 1, 2, 1, 2, 15, 67, 2, 22, 13, 3
// [1, 1, 2, 1, 8, 32, 2, 65, 77, 8, 91, 31, 3, 65, 32, 7];

/// This will throw an error if N = 0 but it's hard to imagine this case coming up.
#[inline(always)]
pub fn dot_product<T, const N: usize>(u: [T; N], v: [T; N]) -> T
where
    T: Copy + AddAssign + Mul<Output = T>,
{
    debug_assert_ne!(N, 0);
    let mut dp = u[0] * v[0];
    for i in 1..N {
        dp += u[i] * v[i];
    }
    dp
}

/// Given the first row `circ_matrix` of an NxN circulant matrix, say
/// C, return the product `C*input`.
///
/// NB: This function is a naive implementation of the n²
/// evaluation. It is a placeholder until we have FFT implementations
/// for all combinations of field and size.
pub fn apply_circulant<AF: AbstractField, const N: usize>(
    circ_matrix: &[u64; N],
    input: [AF; N],
) -> [AF; N] {
    let mut matrix: [AF; N] = circ_matrix.map(AF::from_canonical_u64);

    let mut output = array::from_fn(|_| AF::zero());
    for out_i in output.iter_mut().take(N - 1) {
        *out_i = AF::dot_product(&matrix, &input);
        matrix.rotate_right(1);
    }
    output[N - 1] = AF::dot_product(&matrix, &input);
    output
}

/// Given the first row of a circulant matrix, return the first column
/// of that circulant matrix. For example, v = [0, 1, 2, 3, 4, 5],
/// then output = [0, 5, 4, 3, 2, 1], i.e. the first element is the
/// same and the other elements are reversed.
///
/// This is useful to prepare a circulant matrix for input to an FFT
/// algorithm, which expects the first column of the matrix rather
/// than the first row (as we normally store them).
///
/// NB: The algorithm is inefficient but simple enough that this
/// function can be declared `const`, and that is the intended context
/// for use.
pub const fn first_row_to_first_col<const N: usize, T: Copy>(v: &[T; N]) -> [T; N] {
    // Can do this to get a simple Default value. Might be better ways?
    let mut output = [v[0]; N];
    let mut i = 1;
    loop {
        if i >= N {
            break;
        }
        output[i] = v[N - i];
        i += 1;
    }
    output
}

/// Use the convolution theorem to calculate the product of the given
/// circulant matrix and the given vector. The circulant matrix must
/// be specified by its first *column*, not its first row. If you have
/// the row as an array, you can obtain the column with `first_row_to_first_col()`.
#[inline]
pub fn apply_circulant_fft<F: TwoAdicField, const N: usize, FFT: TwoAdicSubgroupDft<F>>(
    fft: FFT,
    column: [u64; N],
    input: &[F; N],
) -> [F; N] {
    let column = column.map(F::from_canonical_u64).to_vec();
    let matrix = fft.dft(column);
    let input = fft.dft(input.to_vec());

    // point-wise product
    let product = matrix
        .iter()
        .zip(input)
        .map(|(&x, y)| x * y)
        .collect::<Vec<_>>();

    let output = fft.idft(product);
    output.try_into().unwrap()
}

#[cfg(test)]
mod tests {
    use super::first_row_to_first_col;

    #[test]
    fn rotation() {
        let input = [0, 1, 2, 3, 4, 5];
        let output = [0, 5, 4, 3, 2, 1];

        assert_eq!(first_row_to_first_col(&input), output);
    }
}
