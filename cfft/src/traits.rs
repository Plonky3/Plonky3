use alloc::vec::Vec;

use p3_field::{ComplexExtension, Field};
use p3_matrix::bitrev::BitReversableMatrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::MatrixRows;

// TODO, import the right thing here.

pub trait CircleSubgroupFFT<Base: Field, Ext: ComplexExtension<Base>>: Clone + Default {
    // Effectively this is either RowMajorMatrix or BitReversedMatrixView<RowMajorMatrix>.
    type Evaluations: BitReversableMatrix<Base>;

    /// Compute the Circle Finite Fourier transform (CFFT) `vec`.
    fn cfft(&self, vec: Vec<Base>) -> Vec<Base> {
        self.cfft_batch(RowMajorMatrix::new_col(vec))
            .to_row_major_matrix()
            .values
    }

    /// Compute the Circle Finite Fourier transform (CFFT) of each column in `mat`.
    /// This is the only method an implementer needs to define, all other
    /// methods can be derived from this one.
    fn cfft_batch(&self, mat: RowMajorMatrix<Base>) -> RowMajorMatrix<Base>;

    /// Compute the "coset CFFT" of `vec`. This is interpolation onto a different twin coset of
    /// the circle group rather than the standard one.
    fn coset_icfft(&self, vec: Vec<Base>, shift: Ext) -> Vec<Base> {
        self.coset_icfft_batch(RowMajorMatrix::new_col(vec), shift)
            .to_row_major_matrix()
            .values
    }

    /// Compute the "coset CFFT" of each column in `mat`. This is interpolation onto a different twin coset of
    /// the circle group rather than the standard one.
    fn coset_icfft_batch(&self, mat: RowMajorMatrix<Base>, shift: Ext) -> RowMajorMatrix<Base>;

    /// Compute the inverse CFFT of `vec`.
    fn icfft(&self, vec: Vec<Base>) -> Vec<Base> {
        self.icfft_batch(RowMajorMatrix::new(vec, 1)).values
    }

    /// Compute the inverse CFFT of each column in `mat`.
    fn icfft_batch(&self, mat: RowMajorMatrix<Base>) -> RowMajorMatrix<Base>;

    /// Compute the low-degree extension of `vec` onto a larger subgroup.
    fn lde(&self, vec: Vec<Base>, added_bits: usize) -> Vec<Base> {
        self.lde_batch(RowMajorMatrix::new(vec, 1), added_bits)
            .to_row_major_matrix()
            .values
    }

    /// Compute the low-degree extension of each column in `mat` onto a larger subgroup.
    fn lde_batch(&self, mat: RowMajorMatrix<Base>, added_bits: usize) -> RowMajorMatrix<Base> {
        let mut coeffs = self.cfft_batch(mat);
        coeffs
            .values
            .resize(coeffs.values.len() << added_bits, Base::zero());
        self.icfft_batch(coeffs)
    }

    /// Compute the low-degree extension of each column in `mat` onto a coset of a larger subgroup.
    fn coset_lde(&self, vec: Vec<Base>, added_bits: usize, shift: Ext) -> Vec<Base> {
        self.coset_lde_batch(RowMajorMatrix::new(vec, 1), added_bits, shift)
            .to_row_major_matrix()
            .values
    }

    /// Compute the low-degree extension of each column in `mat` onto a coset of a larger subgroup.
    fn coset_lde_batch(
        &self,
        mat: RowMajorMatrix<Base>,
        added_bits: usize,
        shift: Ext,
    ) -> RowMajorMatrix<Base> {
        let mut coeffs = self.icfft_batch(mat);
        // PANICS: possible panic if the new resized length overflows
        coeffs.values.resize(
            coeffs
                .values
                .len()
                .checked_shl(added_bits.try_into().unwrap())
                .unwrap(),
            Base::zero(),
        );
        self.coset_icfft_batch(coeffs, shift)
    }
}
