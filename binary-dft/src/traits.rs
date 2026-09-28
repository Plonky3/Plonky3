//! The additive NTT interface shared by the reference and fast transforms.

use p3_binary_field::TowerLevel;
use p3_commit::zero_padded;
use p3_matrix::Matrix;
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixView};
use p3_util::log2_strict_usize;

/// An additive NTT: evaluation in the novel polynomial basis over a linear subspace of the field.
///
/// - Column entry `i` is the coefficient of `X_i = prod_j W_j^(bit j of i)`, a product of subspace polynomials.
/// - Output row `i` is the evaluation at `shift + domain_point(i)`.
/// - Both sides are in natural order.
/// - Heights are powers of two, `2^l` rows spanning the first `l` Cantor basis vectors.
/// - So `l` is at most the bit width of the level.
///
/// There is no clone-and-default supertrait, so a transform may carry state such as a table.
pub trait AdditiveNtt<F: TowerLevel> {
    /// Evaluate each column on the coset `shift + S_l`.
    ///
    /// # Panics
    ///
    /// - Panics if the height is not a power of two.
    /// - Panics if `l` exceeds the bit width of the level, which has only that many basis vectors.
    fn shifted_ntt_batch(&self, mat: RowMajorMatrix<F>, shift: F) -> RowMajorMatrix<F>;

    /// Recover the coefficients from evaluations on the coset `shift + S_l`.
    ///
    /// # Panics
    ///
    /// Panics under the same conditions as the forward transform.
    fn shifted_intt_batch(&self, mat: RowMajorMatrix<F>, shift: F) -> RowMajorMatrix<F>;

    /// Evaluate each column on the subspace `S_l` itself.
    fn ntt_batch(&self, mat: RowMajorMatrix<F>) -> RowMajorMatrix<F> {
        self.shifted_ntt_batch(mat, F::ZERO)
    }

    /// Evaluate a matrix whose rows past the first `2^-log_inv_rate` fraction are all zero.
    ///
    /// - The caller must leave that tail zero.
    /// - A transform may then skip the work the zero tail would do.
    /// - This default evaluates the whole matrix.
    ///
    /// # Panics
    ///
    /// - Panics if the height is not a power of two.
    /// - Panics if the padding exceeds the height.
    fn ntt_batch_padded(&self, mat: RowMajorMatrix<F>, log_inv_rate: usize) -> RowMajorMatrix<F> {
        let log_n = log2_strict_usize(mat.height());
        assert!(log_inv_rate <= log_n, "padding exceeds matrix height");
        self.ntt_batch(mat)
    }

    /// Evaluate a borrowed coefficient matrix zero-padded to `2^log_inv_rate` times its height.
    ///
    /// - The borrowed matrix is the unpadded one, and is left as it is.
    /// - The result is what the padded transform makes of the zero-padded matrix.
    /// - This default copies the matrix into a zeroed one of the padded height first.
    /// - A transform that reads the matrix where it lies skips that copy.
    ///
    /// # Panics
    ///
    /// - Panics if the height is not a power of two.
    /// - Panics if the padded height overflows the address space.
    fn ntt_batch_borrowed(
        &self,
        mat: RowMajorMatrixView<'_, F>,
        log_inv_rate: usize,
    ) -> RowMajorMatrix<F> {
        self.ntt_batch_padded(zero_padded(mat, log_inv_rate), log_inv_rate)
    }

    /// Recover the coefficients from evaluations on the subspace `S_l` itself.
    fn intt_batch(&self, mat: RowMajorMatrix<F>) -> RowMajorMatrix<F> {
        self.shifted_intt_batch(mat, F::ZERO)
    }

    /// Extend evaluations on `S_l` to the larger subspace `S_(l + added_bits)`.
    ///
    /// `S_l` is the index prefix of the larger subspace, so the input rows reappear as the output's prefix.
    ///
    /// # Panics
    ///
    /// - Panics if `l + added_bits` exceeds the bit width of the level.
    /// - Panics under the conditions of the forward transform.
    fn lde_batch(&self, mat: RowMajorMatrix<F>, added_bits: usize) -> RowMajorMatrix<F> {
        self.shifted_lde_batch(mat, added_bits, F::ZERO)
    }

    /// Extend evaluations on `shift + S_l` to the larger coset `shift + S_(l + added_bits)`.
    ///
    /// # Panics
    ///
    /// Panics under the same conditions as the unshifted extension.
    fn shifted_lde_batch(
        &self,
        mat: RowMajorMatrix<F>,
        added_bits: usize,
        shift: F,
    ) -> RowMajorMatrix<F> {
        let log_n = log2_strict_usize(mat.height());
        assert!(log_n <= 1 << F::LOG_BITS, "domain exceeds field dimension");
        if added_bits == 0 {
            return mat;
        }

        // Back to coefficients, over the same coset.
        let coeffs = self.shifted_intt_batch(mat, shift);
        let width = coeffs.width;
        let len = coeffs.values.len();

        // Zero-padding the coefficients keeps the same polynomials on the larger domain.
        let mut values = F::zero_vec(padded_len(len, added_bits));
        values[..len].copy_from_slice(&coeffs.values);
        self.shifted_ntt_batch(RowMajorMatrix::new(values, width), shift)
    }
}

/// The length of a buffer zero-padded by `added_bits` domain dimensions.
///
/// # Panics
///
/// Panics if that length overflows the address space.
pub(crate) fn padded_len(len: usize, added_bits: usize) -> usize {
    // A shift amount below the word size still lets the value itself overflow.
    //
    // Shifting back and recovering the length proves no bit was lost.
    u32::try_from(added_bits)
        .ok()
        .and_then(|bits| len.checked_shl(bits))
        .filter(|&padded| padded >> added_bits == len)
        .expect("codeword length overflows usize")
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_binary_field::{BinaryField8, BinaryField64, TowerLevel};
    use p3_field::PrimeCharacteristicRing;
    use p3_matrix::dense::RowMajorMatrix;

    use super::{AdditiveNtt, padded_len};
    use crate::LchNtt;

    #[test]
    fn a_borrowed_matrix_transforms_as_its_zero_padded_copy() {
        // The default copies the borrowed matrix under a zero tail, then transforms that.
        let mat = RowMajorMatrix::new(
            (0..32u64)
                .map(|i| BinaryField64::from_repr(i.wrapping_mul(0x9e37_79b9_7f4a_7c15)))
                .collect(),
            4,
        );
        let ntt = LchNtt::<BinaryField64>::default();
        for log_inv_rate in 0..=2 {
            let mut padded = mat.clone();
            padded
                .values
                .resize(mat.values.len() << log_inv_rate, BinaryField64::ZERO);
            assert_eq!(
                ntt.ntt_batch_borrowed(mat.as_view(), log_inv_rate),
                ntt.ntt_batch_padded(padded, log_inv_rate),
                "rate={log_inv_rate}"
            );
        }
    }

    #[test]
    fn identity_lde_preserves_input_allocation() {
        let mat = RowMajorMatrix::new(vec![BinaryField8::ONE; 32], 4);
        let ptr = mat.values.as_ptr();
        let result = LchNtt::default().lde_batch(mat, 0);
        assert_eq!(result.values.as_ptr(), ptr);
        assert_eq!(result.values, vec![BinaryField8::ONE; 32]);
    }

    #[test]
    #[should_panic]
    fn identity_lde_rejects_invalid_height() {
        let _ = LchNtt::default().lde_batch(RowMajorMatrix::new(vec![BinaryField8::ONE; 3], 1), 0);
    }

    #[test]
    #[should_panic]
    fn identity_lde_rejects_domain_above_field_dimension() {
        let _ =
            LchNtt::default().lde_batch(RowMajorMatrix::new(vec![BinaryField8::ONE; 512], 1), 0);
    }

    #[test]
    fn the_padded_length_is_the_message_length_shifted() {
        // Fixture state: a rate of 1/8 multiplies the coefficient count by eight.
        assert_eq!(padded_len(48, 3), 384);

        // No added dimension leaves the message length alone.
        assert_eq!(padded_len(48, 0), 48);

        // An empty message stays empty at every rate.
        assert_eq!(padded_len(0, 60), 0);
    }

    #[test]
    #[should_panic = "codeword length overflows usize"]
    fn the_padded_length_refuses_a_shift_past_the_word_size() {
        // A shift of the whole word width has no result `usize` can hold.
        let _ = padded_len(2, usize::BITS as usize);
    }

    #[test]
    #[should_panic = "codeword length overflows usize"]
    fn the_padded_length_refuses_a_value_that_overflows() {
        // The shift amount fits the word, and the shifted value does not.
        //
        //     1 << (BITS - 1)  shifted once more drops its only set bit
        let _ = padded_len(1 << (usize::BITS - 1), 1);
    }
}
