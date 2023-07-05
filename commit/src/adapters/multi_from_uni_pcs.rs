use crate::pcs::{MultivariatePCS, UnivariatePCS, PCS};
use alloc::vec::Vec;
use core::marker::PhantomData;
use p3_challenger::Challenger;
use p3_field::{ExtensionField, Field};
use p3_matrix::dense::RowMajorMatrix;

pub struct MultiFromUniPCS<F: Field, U: UnivariatePCS<F>> {
    _uni: U,
    _phantom_f: PhantomData<F>,
}

impl<F: Field, U: UnivariatePCS<F>> PCS<F> for MultiFromUniPCS<F, U> {
    type Commitment = ();
    type ProverData = U::ProverData;
    type Proof = ();
    type Error = ();

    fn commit_batches(
        &self,
        _polynomials: Vec<RowMajorMatrix<F>>,
    ) -> (Self::Commitment, Self::ProverData) {
        todo!()
    }

    fn get_committed_height(&self, _prover_data: &Self::ProverData, _matrix: usize) -> usize {
        todo!()
    }

    fn get_committed_row(
        &self,
        _prover_data: &Self::ProverData,
        _matrix: usize,
        _row: usize,
    ) -> &[F] {
        todo!()
    }
}

impl<F: Field, U: UnivariatePCS<F>> MultivariatePCS<F> for MultiFromUniPCS<F, U> {
    fn open_multi_batches<EF, Chal>(
        &self,
        _prover_data: &[Self::ProverData],
        _points: &[Vec<EF>],
        _challenger: &mut Chal,
    ) -> (Vec<Vec<Vec<EF>>>, Self::Proof)
    where
        EF: ExtensionField<F>,
        Chal: Challenger<F>,
    {
        todo!()
    }

    fn verify_multi_batches<EF, Chal>(
        &self,
        _commits: &[Self::Commitment],
        _points: &[Vec<EF>],
        _values: &[Vec<Vec<EF>>],
        _proof: &Self::Proof,
    ) -> Result<(), Self::Error>
    where
        EF: ExtensionField<F>,
        Chal: Challenger<F>,
    {
        todo!()
    }
}
