use p3_dft::TwoAdicSubgroupDft;
use p3_field::{PrimeCharacteristicRing, TwoAdicField};

/// Given the first row `circ_matrix` of an NxN circulant matrix, say
/// C, return the product `C*input`.
///
/// NB: This is a naive O(N^2) implementation. It serves as a fallback
/// for cases where faster paths (Karatsuba convolution or FFT) do not
/// apply — e.g. non-power-of-two widths, non-two-adic fields, or
/// packed types without a specialised implementation.
pub fn apply_circulant<R: PrimeCharacteristicRing, const N: usize>(
    circ_matrix: &[u64; N],
    input: &[R; N],
) -> [R; N] {
    let matrix = circ_matrix.map(R::from_u64);

    core::array::from_fn(|row| {
        // Build the circulant row: C[row][col] = first_row[(N + col - row) % N].
        let rotated: [R; N] = core::array::from_fn(|col| matrix[(N + col - row) % N].clone());
        R::dot_product(&rotated, input)
    })
}

/// Given the first row of a circulant matrix, return the first column.
///
/// For example if, `v = [0, 1, 2, 3, 4, 5]` then `output = [0, 5, 4, 3, 2, 1]`,
/// i.e. the first element is the same and the other elements are reversed.
///
/// This is useful to prepare a circulant matrix for input to an FFT
/// algorithm, which expects the first column of the matrix rather
/// than the first row (as we normally store them).
///
/// NB: The algorithm is inefficient but simple enough that this
/// function can be declared `const`, and that is the intended context
/// for use.
pub const fn first_row_to_first_col<const N: usize, T: Copy>(v: &[T; N]) -> [T; N] {
    // Start with a copy; the first element is shared between row and column.
    let mut output = *v;
    let mut i = 1;
    while i < N {
        // Reverse the remaining elements: col[i] = row[N - i].
        output[i] = v[N - i];
        i += 1;
    }
    output
}

/// Use the convolution theorem to calculate the product of the given
/// circulant matrix and the given vector.
///
/// The circulant matrix must be specified by its first *column*, not its first row. If you have
/// the row as an array, you can obtain the column with `first_row_to_first_col()`.
#[inline]
pub fn apply_circulant_fft<F: TwoAdicField, const N: usize, FFT: TwoAdicSubgroupDft<F>>(
    fft: &FFT,
    column: [u64; N],
    input: &[F; N],
) -> [F; N] {
    // Transform the circulant column to the frequency domain.
    let column = column.map(F::from_u64).to_vec();
    let matrix = fft.dft(column);

    // Transform the input vector to the frequency domain.
    let input = fft.dft(input.to_vec());

    // Convolution theorem: point-wise multiply in frequency domain.
    let product = matrix.iter().zip(input).map(|(&x, y)| x * y).collect();

    // Transform back to the time domain to get the circulant product.
    let output = fft.idft(product);
    output.try_into().unwrap()
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_dft::NaiveDft;
    use p3_field::PrimeCharacteristicRing;
    use proptest::prelude::*;

    use super::*;

    type F = BabyBear;

    fn arb_f() -> impl Strategy<Value = F> {
        prop::num::u32::ANY.prop_map(F::from_u32)
    }

    #[test]
    fn first_row_to_first_col_even_length() {
        let input = [0, 1, 2, 3, 4, 5];
        assert_eq!(first_row_to_first_col(&input), [0, 5, 4, 3, 2, 1]);
    }

    #[test]
    fn first_row_to_first_col_odd_length() {
        let input = [10, 20, 30, 40, 50];
        assert_eq!(first_row_to_first_col(&input), [10, 50, 40, 30, 20]);
    }

    #[test]
    fn first_row_to_first_col_single_element() {
        assert_eq!(first_row_to_first_col(&[42]), [42]);
    }

    #[test]
    fn first_row_to_first_col_two_elements() {
        assert_eq!(first_row_to_first_col(&[1, 2]), [1, 2]);
    }

    #[test]
    fn apply_circulant_identity() {
        // The identity circulant [1, 0, 0, ...] must return the input unchanged.
        let identity_row: [u64; 4] = [1, 0, 0, 0];
        let input: [F; 4] = [5, 10, 15, 20].map(F::from_u32);
        assert_eq!(apply_circulant(&identity_row, &input), input);
    }

    #[test]
    fn apply_circulant_all_ones() {
        // An all-ones circulant sums every input element into every output slot.
        let ones: [u64; 4] = [1, 1, 1, 1];
        let input: [F; 4] = [1, 2, 3, 4].map(F::from_u32);
        let sum = F::from_u32(10);
        assert_eq!(apply_circulant(&ones, &input), [sum; 4]);
    }

    #[test]
    fn apply_circulant_scalar() {
        // A scalar circulant [k, 0, 0, ...] multiplies each element by k.
        let row: [u64; 4] = [7, 0, 0, 0];
        let input: [F; 4] = [1, 2, 3, 4].map(F::from_u32);
        let expected: [F; 4] = [7, 14, 21, 28].map(F::from_u32);
        assert_eq!(apply_circulant(&row, &input), expected);
    }

    #[test]
    fn apply_circulant_size_1() {
        // A 1x1 circulant is just scalar multiplication.
        let row: [u64; 1] = [5];
        let input: [F; 1] = [F::from_u32(3)];
        assert_eq!(apply_circulant(&row, &input), [F::from_u32(15)]);
    }

    #[test]
    fn apply_circulant_fft_matches_naive_4() {
        // The FFT-based path must agree with the naive O(N^2) path.
        let row: [u64; 4] = [2, 3, 5, 7];
        let col = first_row_to_first_col(&row);
        let input: [F; 4] = [1, 2, 3, 4].map(F::from_u32);

        let naive = apply_circulant(&row, &input);
        let fft_result = apply_circulant_fft(&NaiveDft, col, &input);
        assert_eq!(naive, fft_result);
    }

    #[test]
    fn apply_circulant_fft_identity() {
        // The FFT-based identity circulant must also return the input unchanged.
        let row: [u64; 4] = [1, 0, 0, 0];
        let col = first_row_to_first_col(&row);
        let input: [F; 4] = [5, 10, 15, 20].map(F::from_u32);
        assert_eq!(apply_circulant_fft(&NaiveDft, col, &input), input);
    }

    proptest! {
        #[test]
        fn first_row_to_first_col_involution(v in prop::array::uniform4(0u64..1000)) {
            let col = first_row_to_first_col(&v);
            let back = first_row_to_first_col(&col);
            prop_assert_eq!(back, v);
        }

        #[test]
        fn apply_circulant_fft_matches_naive(
            row in prop::array::uniform4(0u64..1000),
            input in prop::array::uniform4(arb_f()),
        ) {
            let col = first_row_to_first_col(&row);
            let naive = apply_circulant(&row, &input);
            let fft_result = apply_circulant_fft(&NaiveDft, col, &input);
            prop_assert_eq!(naive, fft_result);
        }

        #[test]
        fn apply_circulant_linearity(
            row in prop::array::uniform4(0u64..100),
            a in prop::array::uniform4(arb_f()),
            b in prop::array::uniform4(arb_f()),
        ) {
            let sum_input: [F; 4] = core::array::from_fn(|i| a[i] + b[i]);
            let ca = apply_circulant(&row, &a);
            let cb = apply_circulant(&row, &b);
            let c_sum = apply_circulant(&row, &sum_input);
            for i in 0..4 {
                prop_assert_eq!(c_sum[i], ca[i] + cb[i]);
            }
        }

        #[test]
        fn apply_circulant_zero_matrix(input in prop::array::uniform4(arb_f())) {
            let zeros: [u64; 4] = [0; 4];
            let result = apply_circulant(&zeros, &input);
            prop_assert_eq!(result, [F::ZERO; 4]);
        }
    }
}
