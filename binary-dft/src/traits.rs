//! The additive NTT interface shared by the reference and fast transforms.

use p3_binary_field::TowerLevel;
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
