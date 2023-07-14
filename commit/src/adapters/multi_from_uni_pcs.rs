use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_challenger::Challenger;
use p3_field::{ExtensionField, Field};
use p3_matrix::MatrixRows;

use crate::pcs::{UnivariatePCS, PCS};
use crate::MultivariatePCS;

pub struct MultiFromUniPCS<F, In, U>
where
    F: Field,
    In: for<'a> MatrixRows<'a, F>,
    U: UnivariatePCS<F, In>,
{
    _uni: U,
    _phantom_f: PhantomData<F>,
    _phantom_in: PhantomData<In>,
}

impl<F, In, U> PCS<F, In> for MultiFromUniPCS<F, In, U>
where
    F: Field,
    In: for<'a> MatrixRows<'a, F>,
    U: UnivariatePCS<F, In>,
{
    type Commitment = ();
    type ProverData = U::ProverData;
    type Proof = ();
    type Error = ();

    fn commit_batches(&self, _polynomials: Vec<In>) -> (Self::Commitment, Self::ProverData) {
        todo!()
    }
}

impl<F, In, U> MultivariatePCS<F, In> for MultiFromUniPCS<F, In, U>
where
    F: Field,
    In: for<'a> MatrixRows<'a, F>,
    U: UnivariatePCS<F, In>,
{
    fn open_multi_batches<EF, Chal>(
        &self,
        _prover_data: &[&Self::ProverData],
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
