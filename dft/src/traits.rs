use alloc::vec::Vec;

use p3_field::{Canonicalize, PrimeField32, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;

use crate::util::{divide_by_height, swap_rows};

pub trait TwoAdicSubgroupDft<F: TwoAdicField>: Clone + Default {
    /// Compute the discrete Fourier transform (DFT) `vec`.
    fn dft(&self, vec: Vec<F>) -> Vec<F> {
        self.dft_batch(RowMajorMatrix::new(vec, 1)).values
    }

    /// Compute the discrete Fourier transform (DFT) of each column in `mat`.
    fn dft_batch(&self, mat: RowMajorMatrix<F>) -> RowMajorMatrix<F>;

    /// Compute the "coset DFT" of `vec`. This can be viewed as interpolation onto a coset of a
    /// multiplicative subgroup, rather than the subgroup itself.
    fn coset_dft(&self, vec: Vec<F>, shift: F) -> Vec<F> {
        self.coset_dft_batch(RowMajorMatrix::new(vec, 1), shift)
            .values
    }

    /// Compute the "coset DFT" of each column in `mat`. This can be viewed as interpolation onto a
    /// coset of a multiplicative subgroup, rather than the subgroup itself.
    fn coset_dft_batch(&self, mut mat: RowMajorMatrix<F>, shift: F) -> RowMajorMatrix<F> {
        // Observe that
        //     y_i = \sum_j c_j (s g^i)^j
        //         = \sum_j (c_j s^j) (g^i)^j
        // which has the structure of an ordinary DFT, except each coefficient c_j is first replaced
        // by c_j s^j.
        mat.rows_mut()
            .zip(shift.powers())
            .for_each(|(row, weight)| {
                row.iter_mut().for_each(|coeff| {
                    *coeff *= weight;
                })
            });
        self.dft_batch(mat)
    }

    /// Compute the inverse DFT of `vec`.
    fn idft(&self, vec: Vec<F>) -> Vec<F> {
        self.idft_batch(RowMajorMatrix::new(vec, 1)).values
    }

    /// Compute the inverse DFT of each column in `mat`.
    fn idft_batch(&self, mat: RowMajorMatrix<F>) -> RowMajorMatrix<F> {
        let mut dft = self.dft_batch(mat);
        let h = dft.height();

        divide_by_height(&mut dft);

        for row in 1..h / 2 {
            swap_rows(&mut dft, row, h - row);
        }

        dft
    }

    /// Compute the low-degree extension of `vec` onto a larger subgroup.
    fn lde(&self, vec: Vec<F>, added_bits: usize) -> Vec<F> {
        self.lde_batch(RowMajorMatrix::new(vec, 1), added_bits)
            .values
    }

    /// Compute the low-degree extension of each column in `mat` onto a larger subgroup.
    fn lde_batch(&self, mat: RowMajorMatrix<F>, added_bits: usize) -> RowMajorMatrix<F> {
        let mut coeffs = self.idft_batch(mat);
        coeffs
            .values
            .resize(coeffs.values.len() << added_bits, F::zero());
        self.dft_batch(coeffs)
    }

    /// Compute the low-degree extension of each column in `mat` onto a coset of a larger subgroup.
    fn coset_lde(&self, vec: Vec<F>, added_bits: usize, shift: F) -> Vec<F> {
        self.coset_lde_batch(RowMajorMatrix::new(vec, 1), added_bits, shift)
            .values
    }

    /// Compute the low-degree extension of each column in `mat` onto a coset of a larger subgroup.
    fn coset_lde_batch(
        &self,
        mat: RowMajorMatrix<F>,
        added_bits: usize,
        shift: F,
    ) -> RowMajorMatrix<F> {
        let mut coeffs = self.idft_batch(mat);
        coeffs
            .values
            .resize(coeffs.values.len() << added_bits, F::zero());
        self.coset_dft_batch(coeffs, shift)
    }
}

pub trait TwoAdicSubgroupDftNC<F: TwoAdicField + PrimeField32, NCF: Canonicalize<F>>:
    Clone + Default
{
    /// Compute the discrete Fourier transform (DFT) `vec`.
    fn dft(&self, vec: Vec<NCF>) -> Vec<NCF> {
        self.dft_batch(RowMajorMatrix::new(vec, 1)).values
    }

    /// Compute the discrete Fourier transform (DFT) of each column in `mat`.
    fn dft_batch(&self, mat: RowMajorMatrix<NCF>) -> RowMajorMatrix<NCF>;
}
