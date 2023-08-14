use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_challenger::FieldChallenger;
use p3_field::{ExtensionField, Field};
use p3_matrix::MatrixRows;

use crate::pcs::{UnivariatePCS, PCS};
use crate::MultivariatePCS;

pub struct MultiFromUniPCS<F, In, U, Chal>
where
    F: Field,
    In: MatrixRows<F>,
    U: UnivariatePCS<F, In, Chal>,
    Chal: FieldChallenger<F>,
{
    _uni: U,
    _phantom_f: PhantomData<F>,
    _phantom_in: PhantomData<In>,
    _phantom_chal: PhantomData<Chal>,
}

impl<F, In, U, Chal> PCS<F, In, Chal> for MultiFromUniPCS<F, In, U, Chal>
where
    F: Field,
    In: MatrixRows<F>,
    U: UnivariatePCS<F, In, Chal>,
    Chal: FieldChallenger<F>,
{
    type Commitment = ();
    type ProverData = U::ProverData;
    type Proof = ();
    type Error = ();

    fn commit_batches(&self, _polynomials: Vec<In>) -> (Self::Commitment, Self::ProverData) {
        todo!()
    }
}

impl<F, In, U, Chal> MultivariatePCS<F, In, Chal> for MultiFromUniPCS<F, In, U, Chal>
where
    F: Field,
    In: MatrixRows<F>,
    U: UnivariatePCS<F, In, Chal>,
    Chal: FieldChallenger<F>,
{
    fn open_multi_batches<EF>(
        &self,
        _prover_data: &[&Self::ProverData],
        _points: &[Vec<EF>],
        _challenger: &mut Chal,
    ) -> (Vec<Vec<Vec<EF>>>, Self::Proof)
    where
        EF: ExtensionField<F>,
    {
        todo!()
    }

    fn verify_multi_batches<EF>(
        &self,
        _commits: &[Self::Commitment],
        _points: &[Vec<EF>],
        _values: &[Vec<Vec<EF>>],
        _proof: &Self::Proof,
    ) -> Result<(), Self::Error>
    where
        EF: ExtensionField<F>,
    {
        todo!()
    }
}
