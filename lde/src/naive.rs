use crate::TwoAdicLDE;
use p3_field::field::{AbstractFieldExtension, Field, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;

pub struct NaiveLDE;

impl<F, FE> TwoAdicLDE<F, FE> for NaiveLDE
where
    F: Field,
    FE: AbstractFieldExtension<F> + TwoAdicField,
{
    type Res = FE;

    fn subgroup_lde_batch(
        &self,
        polys: RowMajorMatrix<F>,
        lde_bits: usize,
    ) -> RowMajorMatrix<Self::Res> {
        todo!()
    }

    fn coset_lde_batch(
        &self,
        polys: RowMajorMatrix<F>,
        lde_bits: usize,
        shift: F,
    ) -> RowMajorMatrix<Self::Res> {
        todo!()
    }
}
