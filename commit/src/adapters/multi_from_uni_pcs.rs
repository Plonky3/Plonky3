use crate::pcs::{MultivariatePCS, UnivariatePCS, PCS};
use alloc::vec::Vec;
use core::marker::PhantomData;
use p3_field::field::Field;
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
        _polynomials: Vec<RowMajorMatrix<F>>,
    ) -> (Self::Commitment, Self::ProverData) {
        todo!()
    }

    fn get_committed_value(_prover_data: &Self::ProverData, _poly: usize, _value: usize) -> F {
        todo!()
    }
}

impl<F: Field, U: UnivariatePCS<F>> MultivariatePCS<F> for MultiFromUniPCS<F, U> {
    fn open_multi_batches<FE: Field<DistinguishedSubfield = F>>(
        _points: &[FE],
        _prover_data: &[Self::ProverData],
    ) -> (Vec<Vec<Vec<FE>>>, Self::Proof) {
        todo!()
    }

    fn verify_multi_batches<FE: Field<DistinguishedSubfield = F>>(
        _commits: &[Self::Commitment],
        _points: &[FE],
        _values: &[Vec<Vec<FE>>],
        _proof: &Self::Proof,
    ) -> Result<(), Self::Error> {
        todo!()
    }
}
