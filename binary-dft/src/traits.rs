use p3_binary_field::TowerLevel;
use p3_matrix::dense::RowMajorMatrix;

/// An additive NTT: evaluation of the novel polynomial basis on an `F_2`-linear subspace.
///
/// Columns hold coefficients in the novel polynomial basis `X_i = ∏_j W_j^{bit_j(i)}`; output
/// row `i` is the evaluation at `shift + domain_point(i)`. Both sides are in natural order.
///
/// Heights must be powers of two no larger than the bit width of `F`.
pub trait AdditiveNtt<F: TowerLevel>: Clone + Default {
    /// Evaluates each column on the coset `shift + S_ℓ`.
    fn shifted_ntt_batch(&self, mat: RowMajorMatrix<F>, shift: F) -> RowMajorMatrix<F>;

    /// Inverse of [`shifted_ntt_batch`](Self::shifted_ntt_batch).
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
        let mut coeffs = self.shifted_intt_batch(mat, shift);
        // Zero-padding the coefficients keeps the same polynomial on the larger domain.
        coeffs
            .values
            .resize(coeffs.values.len() << added_bits, F::ZERO);
        self.shifted_ntt_batch(coeffs, shift)
    }
}
