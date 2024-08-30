use alloc::vec::Vec;

use p3_field::TwoAdicField;
use p3_matrix::bitrev::BitReversableMatrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::util::swap_rows;
use p3_matrix::Matrix;

use crate::util::{coset_shift_cols, divide_by_height};

pub trait TwoAdicSubgroupDft<F: TwoAdicField>: Clone + Default {
    // Effectively this is either RowMajorMatrix or BitReversedMatrixView<RowMajorMatrix>.
    // Always owned.
    type Evaluations: BitReversableMatrix<F> + 'static;

    /// Compute the discrete Fourier transform (DFT) `vec`.
    fn dft(&self, vec: Vec<F>) -> Vec<F> {
        self.dft_batch(RowMajorMatrix::new_col(vec))
            .to_row_major_matrix()
            .values
    }

    /// Compute the discrete Fourier transform (DFT) of each column in `mat`.
    /// This is the only method an implementer needs to define, all other
    /// methods can be derived from this one.
    fn dft_batch(&self, mat: RowMajorMatrix<F>) -> Self::Evaluations;

    /// Compute the "coset DFT" of `vec`. This can be viewed as interpolation onto a coset of a
    /// multiplicative subgroup, rather than the subgroup itself.
    fn coset_dft(&self, vec: Vec<F>, shift: F) -> Vec<F> {
        self.coset_dft_batch(RowMajorMatrix::new_col(vec), shift)
            .to_row_major_matrix()
            .values
    }

    /// Compute the "coset DFT" of each column in `mat`. This can be viewed as interpolation onto a
    /// coset of a multiplicative subgroup, rather than the subgroup itself.
    fn coset_dft_batch(&self, mut mat: RowMajorMatrix<F>, shift: F) -> Self::Evaluations {
        // Observe that
        //     y_i = \sum_j c_j (s g^i)^j
        //         = \sum_j (c_j s^j) (g^i)^j
        // which has the structure of an ordinary DFT, except each coefficient c_j is first replaced
        // by c_j s^j.
        coset_shift_cols(&mut mat, shift);
        self.dft_batch(mat)
    }

    /// Compute the inverse DFT of `vec`.
    fn idft(&self, vec: Vec<F>) -> Vec<F> {
        self.idft_batch(RowMajorMatrix::new(vec, 1)).values
    }

    /// Compute the inverse DFT of each column in `mat`.
    fn idft_batch(&self, mat: RowMajorMatrix<F>) -> RowMajorMatrix<F> {
        let mut dft = self.dft_batch(mat).to_row_major_matrix();
        let h = dft.height();

        divide_by_height(&mut dft);

        for row in 1..h / 2 {
            swap_rows(&mut dft, row, h - row);
        }

        dft
    }

    /// Compute the "coset iDFT" of `vec`. This can be viewed as an inverse operation of
    /// "coset DFT", that interpolates over a coset of a multiplicative subgroup, rather than
    /// subgroup itself.
    fn coset_idft(&self, vec: Vec<F>, shift: F) -> Vec<F> {
        self.coset_idft_batch(RowMajorMatrix::new(vec, 1), shift)
            .values
    }

    /// Compute the "coset iDFT" of each column in `mat`. This can be viewed as an inverse operation
    /// of "coset DFT", that interpolates over a coset of a multiplicative subgroup, rather than the
    /// subgroup itself.
    fn coset_idft_batch(&self, mut mat: RowMajorMatrix<F>, shift: F) -> RowMajorMatrix<F> {
        mat = self.idft_batch(mat);
        coset_shift_cols(&mut mat, shift.inverse());
        mat
    }

    /// Compute the low-degree extension of `vec` onto a larger subgroup.
    fn lde(&self, vec: Vec<F>, added_bits: usize) -> Vec<F> {
        self.lde_batch(RowMajorMatrix::new(vec, 1), added_bits)
            .to_row_major_matrix()
            .values
    }

    /// Compute the low-degree extension of each column in `mat` onto a larger subgroup.
    fn lde_batch(&self, mat: RowMajorMatrix<F>, added_bits: usize) -> Self::Evaluations {
        let mut coeffs = self.idft_batch(mat);
        coeffs
            .values
            .resize(coeffs.values.len() << added_bits, F::zero());
        self.dft_batch(coeffs)
    }

    /// Compute the low-degree extension of each column in `mat` onto a coset of a larger subgroup.
    fn coset_lde(&self, vec: Vec<F>, added_bits: usize, shift: F) -> Vec<F> {
        self.coset_lde_batch(RowMajorMatrix::new(vec, 1), added_bits, shift)
            .to_row_major_matrix()
            .values
    }

    /// Compute the low-degree extension of each column in `mat` onto a coset of a larger subgroup.
    fn coset_lde_batch(
        &self,
        mat: RowMajorMatrix<F>,
        added_bits: usize,
        shift: F,
    ) -> Self::Evaluations {
        let mut coeffs = self.idft_batch(mat);
        // PANICS: possible panic if the new resized length overflows
        coeffs.values.resize(
            coeffs
                .values
                .len()
                .checked_shl(added_bits.try_into().unwrap())
                .unwrap(),
            F::zero(),
        );
        self.coset_dft_batch(coeffs, shift)
    }
}
