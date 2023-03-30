use crate::{MultivariatePCS, UnivariatePCS, PCS};
use alloc::vec::Vec;
use core::marker::PhantomData;
use p3_field::field::{Field, FieldExtension};
use p3_field::matrix::dense::DenseMatrix;

pub struct MultiFromUniPCS<F: Field, U: UnivariatePCS<F>> {
    _uni: U,
    _phantom_f: PhantomData<F>,
}

impl<F: Field, U: UnivariatePCS<F>> PCS<F> for MultiFromUniPCS<F, U> {
    type Commitment = ();
    type ProverData = U::ProverData;
    type Proof = ();

    fn commit_batches(_polynomials: Vec<DenseMatrix<F>>) -> (Self::Commitment, Self::ProverData) {
        todo!()
    }
}

impl<F: Field, U: UnivariatePCS<F>> MultivariatePCS<F> for MultiFromUniPCS<F, U> {
    fn open_batches<FE: FieldExtension<Base = F>>(
        _points: &[FE],
        _prover_data: &[Self::ProverData],
    ) -> (Vec<Vec<Vec<FE>>>, Self::Proof) {
        todo!()
    }

    fn verify_batches<FE: FieldExtension<Base = F>>(
        _commit: &Self::Commitment,
        _points: &[FE],
        _values: &Vec<Vec<Vec<FE>>>,
        _proof: &Self::Proof,
    ) {
        todo!()
    }
}
