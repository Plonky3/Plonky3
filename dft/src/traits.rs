use alloc::vec::Vec;

use p3_field::{Algebra, BasedVectorSpace, TwoAdicField};
use p3_matrix::Matrix;
use p3_matrix::bitrev::BitReversibleMatrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::util::swap_rows;

use crate::util::{coset_shift_cols, divide_by_height};

pub trait TwoAdicSubgroupDft<F: TwoAdicField>: Clone + Default {
    // Effectively this is either RowMajorMatrix or BitReversedMatrixView<RowMajorMatrix>.
    // Always owned.
    type Evaluations: BitReversibleMatrix<F> + 'static;

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
            .resize(coeffs.values.len() << added_bits, F::ZERO);
        self.dft_batch(coeffs)
    }

    /// Compute the low-degree extension of of `vec` onto a coset of a larger subgroup.
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
            F::ZERO,
        );
        self.coset_dft_batch(coeffs, shift)
    }

    // As FFT's are linear, we can lift these FFT algorithms to work on any algebra
    // over the field. We don't actually need the `Algebra<F>` trait for this to
    // compile but, if a vector space does not implement `Algebra<F>`, applying
    // an F-FFT to it is meaningless.

    // When `V` is an extension field, this is much faster than using `TwoAdicSubgroupDft<V>`
    // as it avoids extension field multiplications and makes better use of vectorization.

    // If you are using this to compute FFT/IFFT's of a single polynomial (e.g. no batching)
    // you should also ensure to use RecursiveDft instead of Radix2Dit if not using the parallel
    // feature and either RecursiveDft or Radix2DitParallel if you are using that feature.

    /// Compute the discrete Fourier transform (DFT) `vec`.
    fn dft_algebra<V: Algebra<F> + BasedVectorSpace<F> + Clone + Send + Sync>(
        &self,
        vec: Vec<V>,
    ) -> Vec<V> {
        self.dft_algebra_batch(RowMajorMatrix::new_col(vec)).values
    }

    /// Compute the discrete Fourier transform (DFT) of each column in `mat`.
    fn dft_algebra_batch<V: Algebra<F> + BasedVectorSpace<F> + Clone + Send + Sync>(
        &self,
        mat: RowMajorMatrix<V>,
    ) -> RowMajorMatrix<V> {
        let init_width = mat.width();
        let base_mat =
            RowMajorMatrix::new(V::flatten_to_base(mat.values), init_width * V::DIMENSION);
        let base_dft_output = self.dft_batch(base_mat).to_row_major_matrix();
        RowMajorMatrix::new(
            V::reconstitute_from_base(base_dft_output.values),
            init_width,
        )
    }

    /// Compute the "coset DFT" of `vec`. This can be viewed as interpolation onto a coset of a
    /// multiplicative subgroup, rather than the subgroup itself.
    fn coset_dft_algebra<V: Algebra<F> + BasedVectorSpace<F> + Clone + Send + Sync>(
        &self,
        vec: Vec<V>,
        shift: F,
    ) -> Vec<V> {
        self.coset_dft_algebra_batch(RowMajorMatrix::new_col(vec), shift)
            .to_row_major_matrix()
            .values
    }

    /// Compute the "coset DFT" of each column in `mat`. This can be viewed as interpolation onto a
    /// coset of a multiplicative subgroup, rather than the subgroup itself.
    fn coset_dft_algebra_batch<V: Algebra<F> + BasedVectorSpace<F> + Clone + Send + Sync>(
        &self,
        mat: RowMajorMatrix<V>,
        shift: F,
    ) -> RowMajorMatrix<V> {
        let init_width = mat.width();
        let base_mat =
            RowMajorMatrix::new(V::flatten_to_base(mat.values), init_width * V::DIMENSION);
        let base_dft_output = self.coset_dft_batch(base_mat, shift).to_row_major_matrix();
        RowMajorMatrix::new(
            V::reconstitute_from_base(base_dft_output.values),
            init_width,
        )
    }

    /// Compute the inverse DFT of `vec`.
    fn idft_algebra<V: Algebra<F> + BasedVectorSpace<F> + Clone + Send + Sync>(
        &self,
        vec: Vec<V>,
    ) -> Vec<V> {
        self.idft_algebra_batch(RowMajorMatrix::new(vec, 1)).values
    }

    /// Compute the inverse DFT of each column in `mat`.
    fn idft_algebra_batch<V: Algebra<F> + BasedVectorSpace<F> + Clone + Send + Sync>(
        &self,
        mat: RowMajorMatrix<V>,
    ) -> RowMajorMatrix<V> {
        let init_width = mat.width();
        let base_mat =
            RowMajorMatrix::new(V::flatten_to_base(mat.values), init_width * V::DIMENSION);
        let base_dft_output = self.idft_batch(base_mat);
        RowMajorMatrix::new(
            V::reconstitute_from_base(base_dft_output.values),
            init_width,
        )
    }

    /// Compute the "coset iDFT" of `vec`. This can be viewed as an inverse operation of
    /// "coset DFT", that interpolates over a coset of a multiplicative subgroup, rather than
    /// subgroup itself.
    fn coset_idft_algebra<V: Algebra<F> + BasedVectorSpace<F> + Clone + Send + Sync>(
        &self,
        vec: Vec<V>,
        shift: F,
    ) -> Vec<V> {
        self.coset_idft_algebra_batch(RowMajorMatrix::new(vec, 1), shift)
            .values
    }

    /// Compute the "coset iDFT" of each column in `mat`. This can be viewed as an inverse operation
    /// of "coset DFT", that interpolates over a coset of a multiplicative subgroup, rather than the
    /// subgroup itself.
    fn coset_idft_algebra_batch<V: Algebra<F> + BasedVectorSpace<F> + Clone + Send + Sync>(
        &self,
        mat: RowMajorMatrix<V>,
        shift: F,
    ) -> RowMajorMatrix<V> {
        let init_width = mat.width();
        let base_mat =
            RowMajorMatrix::new(V::flatten_to_base(mat.values), init_width * V::DIMENSION);
        let base_dft_output = self.coset_idft_batch(base_mat, shift);
        RowMajorMatrix::new(
            V::reconstitute_from_base(base_dft_output.values),
            init_width,
        )
    }

    /// Compute the low-degree extension of `vec` onto a larger subgroup.
    fn lde_algebra<V: Algebra<F> + BasedVectorSpace<F> + Clone + Send + Sync>(
        &self,
        vec: Vec<V>,
        added_bits: usize,
    ) -> Vec<V> {
        self.lde_algebra_batch(RowMajorMatrix::new(vec, 1), added_bits)
            .to_row_major_matrix()
            .values
    }

    /// Compute the low-degree extension of each column in `mat` onto a larger subgroup.
    fn lde_algebra_batch<V: Algebra<F> + BasedVectorSpace<F> + Clone + Send + Sync>(
        &self,
        mat: RowMajorMatrix<V>,
        added_bits: usize,
    ) -> RowMajorMatrix<V> {
        let init_width = mat.width();
        let base_mat =
            RowMajorMatrix::new(V::flatten_to_base(mat.values), init_width * V::DIMENSION);
        let base_dft_output = self.lde_batch(base_mat, added_bits).to_row_major_matrix();
        RowMajorMatrix::new(
            V::reconstitute_from_base(base_dft_output.values),
            init_width,
        )
    }

    /// Compute the low-degree extension of `vec` onto a coset of a larger subgroup.
    fn coset_lde_algebra<V: Algebra<F> + BasedVectorSpace<F> + Clone + Send + Sync>(
        &self,
        vec: Vec<V>,
        added_bits: usize,
        shift: F,
    ) -> Vec<V> {
        self.coset_lde_algebra_batch(RowMajorMatrix::new(vec, 1), added_bits, shift)
            .to_row_major_matrix()
            .values
    }

    /// Compute the low-degree extension of each column in `mat` onto a coset of a larger subgroup.
    fn coset_lde_algebra_batch<V: Algebra<F> + BasedVectorSpace<F> + Clone + Send + Sync>(
        &self,
        mat: RowMajorMatrix<V>,
        added_bits: usize,
        shift: F,
    ) -> RowMajorMatrix<V> {
        let init_width = mat.width();
        let base_mat =
            RowMajorMatrix::new(V::flatten_to_base(mat.values), init_width * V::DIMENSION);
        let base_dft_output = self
            .coset_lde_batch(base_mat, added_bits, shift)
            .to_row_major_matrix();
        RowMajorMatrix::new(
            V::reconstitute_from_base(base_dft_output.values),
            init_width,
        )
    }
}
