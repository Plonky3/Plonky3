//! The additive NTT interface shared by the reference and fast transforms.

use p3_binary_field::TowerLevel;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;

/// An additive NTT: evaluation of the novel polynomial basis on an `F_2`-linear subspace.
///
/// Columns hold coefficients in the novel polynomial basis `X_i = ∏_j Ŵ_j^{bit_j(i)}`, the
/// products of normalised subspace polynomials; output row `i` is the evaluation at
/// `shift + domain_point(i)`. Both sides are in natural order.
///
/// Heights are powers of two, and `ℓ = log2(height)` is at most the bit width `2^LOG_BITS` of
/// `F`: `S_ℓ` is spanned by the first `ℓ` Cantor basis vectors, of which there are only that many.
///
/// No `Clone + Default` supertrait: an implementation carrying state — a precomputed twiddle
/// table, a configurable grain, a non-Cantor subspace basis — should not be ruled out by the
/// interface. [`AdditiveRsEncoder`](crate::AdditiveRsEncoder) requires those bounds itself.
pub trait AdditiveNtt<F: TowerLevel> {
    /// Evaluates each column on the coset `shift + S_ℓ`.
    ///
    /// # Panics
    /// Panics if the height of `mat` is not a power of two, or if `ℓ` exceeds the bit width of
    /// `F`, since `S_ℓ` then calls for a Cantor basis vector this level does not have.
    fn shifted_ntt_batch(&self, mat: RowMajorMatrix<F>, shift: F) -> RowMajorMatrix<F>;

    /// Inverse of [`shifted_ntt_batch`](Self::shifted_ntt_batch).
    ///
    /// # Panics
    /// Panics under the same conditions as [`shifted_ntt_batch`](Self::shifted_ntt_batch).
    fn shifted_intt_batch(&self, mat: RowMajorMatrix<F>, shift: F) -> RowMajorMatrix<F>;

    /// Evaluates each column on `S_ℓ`.
    fn ntt_batch(&self, mat: RowMajorMatrix<F>) -> RowMajorMatrix<F> {
        self.shifted_ntt_batch(mat, F::ZERO)
    }

    /// Transforms a matrix whose coefficient prefix has been padded with zero rows.
    ///
    /// The caller supplies `2^log_inv_rate` times the original height and must leave
    /// every entry after that original prefix zero. Implementations may ignore that tail.
    /// The default evaluates the full matrix; optimized implementations can skip zero work.
    ///
    /// # Panics
    /// Panics for an invalid transform height or if the padding exceeds that height.
    fn ntt_batch_padded(&self, mat: RowMajorMatrix<F>, log_inv_rate: usize) -> RowMajorMatrix<F> {
        let log_n = p3_util::log2_strict_usize(mat.height());
        assert!(log_inv_rate <= log_n, "padding exceeds matrix height");
        self.ntt_batch(mat)
    }

    /// Inverse of [`ntt_batch`](Self::ntt_batch).
    fn intt_batch(&self, mat: RowMajorMatrix<F>) -> RowMajorMatrix<F> {
        self.shifted_intt_batch(mat, F::ZERO)
    }

    /// Low-degree extension onto `S_{ℓ + added_bits} ⊃ S_ℓ`.
    ///
    /// Because `S_ℓ` is the index prefix of the larger domain, the input rows reappear as the
    /// prefix of the output.
    ///
    /// # Panics
    /// Panics unless `ℓ + added_bits` is at most the bit width of `F`, on top of the conditions
    /// of [`shifted_ntt_batch`](Self::shifted_ntt_batch).
    fn lde_batch(&self, mat: RowMajorMatrix<F>, added_bits: usize) -> RowMajorMatrix<F> {
        self.shifted_lde_batch(mat, added_bits, F::ZERO)
    }

    /// [`lde_batch`](Self::lde_batch) on the coset `shift + S_{ℓ + added_bits}`.
    fn shifted_lde_batch(
        &self,
        mat: RowMajorMatrix<F>,
        added_bits: usize,
        shift: F,
    ) -> RowMajorMatrix<F> {
        let log_n = p3_util::log2_strict_usize(mat.height());
        assert!(log_n <= 1 << F::LOG_BITS, "domain exceeds field dimension");
        if added_bits == 0 {
            return mat;
        }
        let coeffs = self.shifted_intt_batch(mat, shift);
        let width = coeffs.width;
        let len = coeffs.values.len();
        let padded_len = u32::try_from(added_bits)
            .ok()
            .and_then(|bits| len.checked_shl(bits))
            // `checked_shl` only rejects a shift amount that is too wide; it does not detect
            // the value itself overflowing, so recovering `len` from the shifted result is
            // what actually proves no bits were lost.
            .filter(|&padded| padded >> added_bits == len)
            .expect("extended codeword length overflows usize");

        // Zero-padding the coefficients keeps the same polynomial on the larger domain. `F::zero_vec`
        // plus a `copy_from_slice` avoids `Vec::resize`'s reallocate-and-memcpy of the whole prefix.
        let mut values = F::zero_vec(padded_len);
        values[..len].copy_from_slice(&coeffs.values);
        self.shifted_ntt_batch(RowMajorMatrix::new(values, width), shift)
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_binary_field::BinaryField8;
    use p3_field::PrimeCharacteristicRing;
    use p3_matrix::dense::RowMajorMatrix;

    use super::AdditiveNtt;
    use crate::LchNtt;

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
}
