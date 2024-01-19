use alloc::vec::Vec;

use p3_field::extension::{Complex, ComplexExtendable};
use p3_field::{ComplexExtension, Field};
use p3_matrix::bitrev::BitReversableMatrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::{Matrix, MatrixRows};
use p3_util::log2_strict_usize;

// In comparison to TwoAdicSubgroupDft, CircleSubgroupFt denotes a group theoretic, non harmonic DFT.
// In particular this means that cfft and icfft are fundamentally different algorithms and cannot be derived from each other.

pub trait CircleSubgroupFt<F: ComplexExtendable>: Clone + Default {
    // Effectively this is either RowMajorMatrix or BitReversedMatrixView<RowMajorMatrix>.
    type Evaluations: BitReversableMatrix<F>;

    /// Compute the Circle Finite Fourier transform (CFFT) `vec`.
    fn cfft(&self, vec: Vec<F>) -> Vec<F> {
        self.cfft_batch(RowMajorMatrix::new_col(vec))
            .to_row_major_matrix()
            .values
    }

    /// Compute the Circle Finite Fourier transform (CFFT) of each column in `mat`.
    /// This takes a vector of evaluations and computes a vector of coefficients.
    fn cfft_batch(&self, mat: RowMajorMatrix<F>) -> RowMajorMatrix<F>;

    /// Compute the "coset CFFT" of `vec`. This is interpolation onto a different twin coset of
    /// the circle group rather than the standard one.
    fn coset_icfft(&self, vec: Vec<F>, shift: Complex<F>) -> Vec<F> {
        self.coset_icfft_batch(RowMajorMatrix::new_col(vec), shift)
            .to_row_major_matrix()
            .values
    }

    /// Compute the "coset CFFT" of each column in `mat`. This is interpolation onto a different twin coset of
    /// the circle group rather than the standard one.
    fn coset_icfft_batch(&self, mat: RowMajorMatrix<F>, shift: Complex<F>) -> RowMajorMatrix<F>;

    /// Compute the inverse CFFT of `vec`.
    fn icfft(&self, vec: Vec<F>) -> Vec<F> {
        self.icfft_batch(RowMajorMatrix::new(vec, 1)).values
    }

    /// Compute the inverse CFFT of each column in `mat`.
    fn icfft_batch(&self, mat: RowMajorMatrix<F>) -> RowMajorMatrix<F> {
        let height = mat.height();
        let log_height = log2_strict_usize(height);
        let gen = F::circle_two_adic_generator(log_height + 1);
        self.coset_icfft_batch(mat, gen) // It will likely be faster to reimplement this as opposed to calling coset_icfft_batch.
    }

    /// Compute the low-degree extension of `vec` onto a different twin coset.
    fn coset_lde(&self, vec: Vec<F>, shift: Complex<F>) -> Vec<F> {
        self.coset_lde_batch(RowMajorMatrix::new(vec, 1), shift)
            .to_row_major_matrix()
            .values
    }

    /// Compute the low-degree extension of each column in `mat` onto a different twin coset.
    /// Interprets columns of `mat` as evaluations over gH = gH^2 u g^{-1}H^2
    /// Computes their extensions to shift H^2 u shift^{-1} H^2.
    fn coset_lde_batch(&self, mat: RowMajorMatrix<F>, shift: Complex<F>) -> RowMajorMatrix<F> {
        debug_assert!(shift.norm().is_one()); // Shift needs to be in S^1.
        let coeffs = self.cfft_batch(mat);
        self.coset_icfft_batch(coeffs, shift)
    }
}
