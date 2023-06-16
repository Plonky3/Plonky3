use crate::wrapped_matrix::WrappedMatrix;
use alloc::vec::Vec;
use core::marker::PhantomData;
use p3_challenger::Challenger;
use p3_code::LinearCodeFamily;
use p3_commit::{DirectMMCS, MultivariatePCS, PCS};
use p3_field::{ExtensionField, Field};
use p3_matrix::dense::RowMajorMatrix;

pub struct TensorPCS<F, C, M>
where
    F: Field,
    C: LinearCodeFamily<F, WrappedMatrix<F, RowMajorMatrix<F>>>,
    M: DirectMMCS<F>,
{
    codes: C,
    mmcs: M,
    _phantom_f: PhantomData<F>,
    _phantom_m: PhantomData<M>,
}

impl<F, C, M> PCS<F> for TensorPCS<F, C, M>
where
    F: Field,
    C: LinearCodeFamily<F, WrappedMatrix<F, RowMajorMatrix<F>>>,
    M: DirectMMCS<F, Mat = C::Out>,
{
    type Commitment = M::Commitment;
    type ProverData = M::ProverData;
    type Proof = ();
    type Error = ();

    fn commit_batches(
        &self,
        polynomials: Vec<RowMajorMatrix<F>>,
    ) -> (Self::Commitment, Self::ProverData) {
        let encoded_polynomials = polynomials
            .into_iter()
            .map(|mat| {
                let wraps = 16; // TODO
                let wrapped = WrappedMatrix::new(mat, wraps);
                self.codes.encode_batch(wrapped)
            })
            .collect();
        self.mmcs.commit(encoded_polynomials)
    }

    fn get_committed_value(
        &self,
        _prover_data: &Self::ProverData,
        _poly: usize,
        _value: usize,
    ) -> F {
        todo!()
    }
}

impl<F, C, M> MultivariatePCS<F> for TensorPCS<F, C, M>
where
    F: Field,
    C: LinearCodeFamily<F, WrappedMatrix<F, RowMajorMatrix<F>>>,
    M: DirectMMCS<F, Mat = C::Out>,
{
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
