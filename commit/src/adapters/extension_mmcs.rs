use alloc::vec::Vec;
use core::marker::PhantomData;
use core::ops::Deref;

use p3_field::{ExtensionField, Field};
use p3_matrix::extension::FlatMatrixView;
use p3_matrix::{Dimensions, Matrix};

use crate::{BatchOpening, BatchOpeningRef, Mmcs};

/// A wrapper to lift an MMCS from a base field `F` to an extension field `EF`.
///
/// `ExtensionMmcs` allows committing to and opening matrices over an extension field by internally
/// using an MMCS defined on the base field. It works by flattening each extension field element
/// into its base field coordinates for commitment, and then reconstructing them on opening.
#[derive(Clone, Debug)]
pub struct ExtensionMmcs<F, EF, InnerMmcs> {
    /// The inner MMCS instance used to handle commitments at the base field level.
    pub(crate) inner: InnerMmcs,

    pub(crate) _phantom: PhantomData<(F, EF)>,
}

impl<F, EF, InnerMmcs> ExtensionMmcs<F, EF, InnerMmcs> {
    pub const fn new(inner: InnerMmcs) -> Self {
        Self {
            inner,
            _phantom: PhantomData,
        }
    }
}

impl<F, EF, InnerMmcs> Mmcs<EF> for ExtensionMmcs<F, EF, InnerMmcs>
where
    F: Field,
    EF: ExtensionField<F>,
    InnerMmcs: Mmcs<F>,
{
    type ProverData<M> = InnerMmcs::ProverData<FlatMatrixView<F, EF, M>>;
    type Commitment = InnerMmcs::Commitment;
    type Proof = InnerMmcs::Proof;
    type MultiProof = InnerMmcs::MultiProof;
    type Error = InnerMmcs::Error;

    fn commit<M: Matrix<EF>>(&self, inputs: Vec<M>) -> (Self::Commitment, Self::ProverData<M>) {
        self.inner
            .commit(inputs.into_iter().map(FlatMatrixView::new).collect())
    }

    fn open_batch<M: Matrix<EF>>(
        &self,
        index: usize,
        prover_data: &Self::ProverData<M>,
    ) -> BatchOpening<EF, Self> {
        let (inner_opened_values, inner_proof) = self.inner.open_batch(index, prover_data).unpack();
        let opened_ext_values = inner_opened_values
            .into_iter()
            .map(EF::reconstitute_from_base)
            .collect();
        BatchOpening::new(opened_ext_values, inner_proof)
    }

    fn get_matrices<'a, M: Matrix<EF>>(&self, prover_data: &'a Self::ProverData<M>) -> Vec<&'a M> {
        self.inner
            .get_matrices(prover_data)
            .into_iter()
            .map(|mat| mat.deref())
            .collect()
    }

    fn verify_batch(
        &self,
        commit: &Self::Commitment,
        dimensions: &[Dimensions],
        index: usize,
        batch_opening: BatchOpeningRef<'_, EF, Self>,
    ) -> Result<(), Self::Error> {
        let opened_base_values: Vec<Vec<F>> = batch_opening
            .opened_values
            .iter()
            .cloned()
            .map(EF::flatten_to_base)
            .collect();
        let base_dimensions = dimensions
            .iter()
            .map(|dim| Dimensions {
                width: dim.width * EF::DIMENSION,
                height: dim.height,
            })
            .collect::<Vec<_>>();
        self.inner.verify_batch(
            commit,
            &base_dimensions,
            index,
            BatchOpeningRef::new(&opened_base_values, batch_opening.opening_proof),
        )
    }

    fn open_multi_batch<M: Matrix<EF>>(
        &self,
        indices: &[usize],
        prover_data: &Self::ProverData<M>,
    ) -> (Vec<Vec<Vec<EF>>>, Self::MultiProof) {
        let (base_values, proof) = self.inner.open_multi_batch(indices, prover_data);
        let ext_values = base_values
            .into_iter()
            .map(|rows| rows.into_iter().map(EF::reconstitute_from_base).collect())
            .collect();
        (ext_values, proof)
    }

    fn verify_multi_batch<R: AsRef<[EF]> + PartialEq>(
        &self,
        commit: &Self::Commitment,
        dimensions: &[Dimensions],
        indices: &[usize],
        opened_values: &[Vec<R>],
        proof: &Self::MultiProof,
    ) -> Result<(), Self::Error> {
        // Each extension row reinterprets as `EF::DIMENSION` base elements.
        // This layer always materializes owned base rows.
        let opened_base_values: Vec<Vec<Vec<F>>> = opened_values
            .iter()
            .map(|rows| {
                rows.iter()
                    .map(|row| EF::flatten_to_base(row.as_ref().to_vec()))
                    .collect()
            })
            .collect();
        let base_dimensions = dimensions
            .iter()
            .map(|dim| Dimensions {
                width: dim.width * EF::DIMENSION,
                height: dim.height,
            })
            .collect::<Vec<_>>();
        self.inner.verify_multi_batch(
            commit,
            &base_dimensions,
            indices,
            &opened_base_values,
            proof,
        )
    }
}
