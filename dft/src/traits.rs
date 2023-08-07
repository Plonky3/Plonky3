use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;

pub trait TwoAdicSubgroupDFT<Val, Dom>
where
    Val: Field,
    Dom: ExtensionField<Val> + TwoAdicField,
{
    /// Compute the DFT of each column in `mat`.
    fn dft_batch(&self, mat: RowMajorMatrix<Val>) -> RowMajorMatrix<Dom>;
}
