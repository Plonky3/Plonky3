use alloc::vec::Vec;

use p3_field::TwoAdicField;
use p3_matrix::bitrev::BitReversableMatrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::util::swap_rows;
use p3_matrix::Matrix;
use p3_util::log2_strict_usize;

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
            .resize(coeffs.values.len() << added_bits, F::ZERO);
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
            F::ZERO,
        );
        self.coset_dft_batch(coeffs, shift)
    }

    /// Compute the low-degree extension of each column in `mat` onto a coset of a larger subgroup, with randomization.
    fn coset_lde_batch_zk(
        &self,
        mat: RowMajorMatrix<F>,
        added_bits: usize,
        shift: F,
        actual_s: F,
        added_values: &[F],
    ) -> Self::Evaluations {
        let h = mat.height();
        let l_h = log2_strict_usize(h);
        let w = mat.width();

        let mut coeffs = self.idft_batch(mat.clone());
        assert!(added_values.len() == h);
        let orig_coeffs = coeffs.clone();
        // coeffs.values.extend(added_values);

        // The quotient matrix corresponds to the decomposition of the quotient poly on the extended basis.
        // For now, I'm only adding random values to the first polynomial, for simplicity and debugging purposes.
        coeffs.values.extend(F::zero_vec(h * w));
        // This adds v_H * r(X). So on H, the evaluation is not affected by this change.
        for i in 0..added_values.len() {
            coeffs.values[i * w] -= added_values[i];
            coeffs.values[h * w + i * w] = added_values[i] / actual_s.exp_u64(h as u64);
        }

        // Debugging.
        // let interp0 = self.coset_dft_batch(orig_coeffs, shift);
        // let sub1 = RowMajorMatrix::new(coeffs.values[..h * w].to_vec(), w);
        // let sub2 = RowMajorMatrix::new(coeffs.values[h * w..].to_vec(), w);

        // assert!(sub1.height() == sub2.height());
        // let interp1 = self.coset_dft_batch(sub1, shift);

        // let interp2 = self.coset_dft_batch(sub2, shift);
        // for i in 0..h {
        //     for j in 0..w {
        //         assert!(
        //             interp1.get(i, j) + actual_s.exp_u64(h as u64) * interp2.get(i, j)
        //                 == interp0.get(i, j),
        //             "interp1 {}, interp2 {}, mat {}",
        //             interp1.get(i, j),
        //             interp2.get(i, j),
        //             interp0.get(i, j)
        //         );
        //     }
        // }

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
}
