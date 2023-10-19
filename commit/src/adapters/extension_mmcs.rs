use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_field::{AbstractExtensionField, ExtensionField, Field};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::{Dimensions, Matrix, MatrixRows};

use crate::{DirectMmcs, Mmcs};

#[derive(Clone)]
pub struct ExtensionMmcs<F, EF, InnerMmcs> {
    inner: InnerMmcs,
    _phantom: PhantomData<(F, EF)>,
}

impl<F, EF, InnerMmcs> ExtensionMmcs<F, EF, InnerMmcs> {
    pub fn new(inner: InnerMmcs) -> Self {
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
    type ProverData = InnerMmcs::ProverData;
    type Commitment = InnerMmcs::Commitment;
    type Proof = InnerMmcs::Proof;
    type Error = InnerMmcs::Error;
    type Mat<'a> = ExtensionMatrix<F, EF, InnerMmcs::Mat<'a>> where Self: 'a;

    fn open_batch(
        &self,
        index: usize,
        prover_data: &Self::ProverData,
    ) -> (Vec<Vec<EF>>, Self::Proof) {
        let (opened_base_values, proof) = self.inner.open_batch(index, prover_data);
        let opened_ext_values = opened_base_values
            .into_iter()
            .map(|row| row.chunks(EF::D).map(EF::from_base_slice).collect())
            .collect();
        (opened_ext_values, proof)
    }

    fn get_matrices<'a>(&'a self, prover_data: &'a Self::ProverData) -> Vec<Self::Mat<'a>> {
        self.inner
            .get_matrices(prover_data)
            .into_iter()
            .map(|mat| ExtensionMatrix {
                inner: mat,
                _phantom: PhantomData,
            })
            .collect()
    }

    fn verify_batch(
        &self,
        commit: &Self::Commitment,
        dimensions: &[Dimensions],
        index: usize,
        opened_values: &[Vec<EF>],
        proof: &Self::Proof,
    ) -> Result<(), Self::Error> {
        let opened_base_values: Vec<Vec<F>> = opened_values
            .iter()
            .map(|row| {
                row.iter()
                    .flat_map(|el| el.as_base_slice())
                    .copied()
                    .collect()
            })
            .collect();
        let base_dimensions = dimensions
            .iter()
            .map(|dim| Dimensions {
                width: dim.width * EF::D,
                height: dim.height,
            })
            .collect::<Vec<_>>();
        self.inner
            .verify_batch(commit, &base_dimensions, index, &opened_base_values, proof)
    }
}

impl<F, EF, InnerMmcs> DirectMmcs<EF> for ExtensionMmcs<F, EF, InnerMmcs>
where
    F: Field,
    EF: ExtensionField<F>,
    InnerMmcs: DirectMmcs<F>,
{
    fn commit(&self, inputs: Vec<RowMajorMatrix<EF>>) -> (Self::Commitment, Self::ProverData) {
        self.inner.commit(
            inputs
                .into_iter()
                .map(|mat| mat.flatten_to_base())
                .collect(),
        )
    }
}

pub struct ExtensionMatrix<F, EF, InnerMat> {
    inner: InnerMat,
    _phantom: PhantomData<(F, EF)>,
}

impl<F, EF, InnerMat> Matrix<EF> for ExtensionMatrix<F, EF, InnerMat>
where
    F: Field,
    EF: ExtensionField<F>,
    InnerMat: Matrix<F>,
{
    fn width(&self) -> usize {
        self.inner.width() * <EF as AbstractExtensionField<F>>::D
    }

    fn height(&self) -> usize {
        self.inner.height()
    }
}

impl<F, EF, InnerMat> MatrixRows<EF> for ExtensionMatrix<F, EF, InnerMat>
where
    F: Field,
    EF: ExtensionField<F>,
    InnerMat: MatrixRows<F>,
{
    type Row<'a> = ExtensionRow<F, EF, <<InnerMat as MatrixRows<F>>::Row<'a> as IntoIterator>::IntoIter> where Self: 'a;

    fn row(&self, r: usize) -> Self::Row<'_> {
        ExtensionRow {
            inner: self.inner.row(r).into_iter(),
            _phantom: PhantomData,
        }
    }
}

pub struct ExtensionRow<F, EF, InnerRowIter> {
    inner: InnerRowIter,
    _phantom: PhantomData<(F, EF)>,
}

impl<F, EF, InnerRowIter> Iterator for ExtensionRow<F, EF, InnerRowIter>
where
    F: Field,
    EF: ExtensionField<F>,
    InnerRowIter: Iterator<Item = F>,
{
    type Item = EF;

    fn next(&mut self) -> Option<Self::Item> {
        let bs: Vec<_> = (&mut self.inner).take(EF::D).collect();
        if bs.is_empty() {
            return None;
        }
        if bs.len() == EF::D {
            return Some(EF::from_base_slice(&bs));
        }
        panic!("Row length does not divide EF::D");
    }
}
