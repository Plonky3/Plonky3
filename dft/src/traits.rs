use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;

use crate::util::swap_rows;

pub trait TwoAdicSubgroupDft<Val, Dom>
where
    Val: Field,
    Dom: ExtensionField<Val> + TwoAdicField,
{
    /// Compute the discrete Fourier transform (DFT) of each column in `mat`.
    fn dft_batch(&self, mat: RowMajorMatrix<Val>) -> RowMajorMatrix<Dom>;

    /// Compute the "coset DFT" of each column in `mat`. This can be viewed as interpolation onto a
    /// coset of a multiplicative subgroup, rather than the subgroup itself.
    fn coset_dft_batch(&self, mut mat: RowMajorMatrix<Val>, shift: Val) -> RowMajorMatrix<Dom> {
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

    /// Compute the inverse DFT of each column in `mat`.
    fn idft_batch(&self, mat: RowMajorMatrix<Val>) -> RowMajorMatrix<Dom> {
        let mut dft = self.dft_batch(mat);
        let h = dft.height();

        let inv_height = Val::from_canonical_usize(h).inverse();
        dft.values.iter_mut().for_each(|coeff| *coeff *= inv_height);

        for row in 1..h / 2 {
            swap_rows(&mut dft, row, h - row);
        }

        dft
    }
}
