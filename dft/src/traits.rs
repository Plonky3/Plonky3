use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;

use crate::util::swap_rows;

pub trait TwoAdicSubgroupDFT<Val, Dom>
where
    Val: Field,
    Dom: ExtensionField<Val> + TwoAdicField,
{
    /// Compute the DFT of each column in `mat`.
    fn dft_batch(&self, mat: RowMajorMatrix<Val>) -> RowMajorMatrix<Dom>;

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
