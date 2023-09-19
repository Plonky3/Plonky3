use alloc::vec::Vec;

use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;

pub trait FourierTransform<Domain: Field>: Clone {
    type Range: Field;

    /// Compute the discrete Fourier transform (DFT) `vec`.
    fn dft(&self, vec: Vec<Domain>) -> Vec<Self::Range> {
        self.dft_batch(RowMajorMatrix::new(vec, 1)).values
    }

    /// Compute the discrete Fourier transform (DFT) of each column in `mat`.
    fn dft_batch(&self, mat: RowMajorMatrix<Domain>) -> RowMajorMatrix<Self::Range>;

    /// Compute the "coset DFT" of `vec`. This can be viewed as interpolation onto a coset of a
    /// multiplicative subgroup, rather than the subgroup itself.
    fn coset_dft(&self, vec: Vec<Domain>, shift: Domain) -> Vec<Self::Range> {
        self.coset_dft_batch(RowMajorMatrix::new(vec, 1), shift)
            .values
    }

    /// Compute the "coset DFT" of each column in `mat`. This can be viewed as interpolation onto a
    /// coset of a multiplicative subgroup, rather than the subgroup itself.
    fn coset_dft_batch(
        &self,
        mut mat: RowMajorMatrix<Domain>,
        shift: Domain,
    ) -> RowMajorMatrix<Self::Range> {
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
    fn idft(&self, vec: Vec<Self::Range>) -> Vec<Domain> {
        self.idft_batch(RowMajorMatrix::new(vec, 1)).values
    }

    /// Compute the inverse DFT of each column in `mat`.
    fn idft_batch(&self, mat: RowMajorMatrix<Self::Range>) -> RowMajorMatrix<Domain>;

    /// Compute the low-degree extension of `vec` onto a larger subgroup.
    fn lde(&self, vec: Vec<Self::Range>, added_bits: usize) -> Vec<Self::Range> {
        self.lde_batch(RowMajorMatrix::new(vec, 1), added_bits)
            .values
    }

    /// Compute the low-degree extension of each column in `mat` onto a larger subgroup.
    fn lde_batch(
        &self,
        mat: RowMajorMatrix<Self::Range>,
        added_bits: usize,
    ) -> RowMajorMatrix<Self::Range> {
        let mut coeffs = self.idft_batch(mat);
        coeffs
            .values
            .resize(coeffs.values.len() << added_bits, Domain::ZERO);
        self.dft_batch(coeffs)
    }

    /// Compute the low-degree extension of each column in `mat` onto a coset of a larger subgroup.
    fn coset_lde(
        &self,
        vec: Vec<Self::Range>,
        added_bits: usize,
        shift: Domain,
    ) -> Vec<Self::Range> {
        self.coset_lde_batch(RowMajorMatrix::new(vec, 1), added_bits, shift)
            .values
    }

    /// Compute the low-degree extension of each column in `mat` onto a coset of a larger subgroup.
    fn coset_lde_batch(
        &self,
        mat: RowMajorMatrix<Self::Range>,
        added_bits: usize,
        shift: Domain,
    ) -> RowMajorMatrix<Self::Range> {
        let mut coeffs = self.idft_batch(mat);
        coeffs
            .values
            .resize(coeffs.values.len() << added_bits, Domain::ZERO);
        self.coset_dft_batch(coeffs, shift)
    }
}
