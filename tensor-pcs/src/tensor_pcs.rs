use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_challenger::Challenger;
use p3_code::LinearCodeFamily;
use p3_commit::{DirectMMCS, MultivariatePCS, PCS};
use p3_field::{ExtensionField, Field};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::MatrixRows;

use crate::reshape::optimal_wraps;
use crate::wrapped_matrix::WrappedMatrix;

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

impl<F, C, M> TensorPCS<F, C, M>
where
    F: Field,
    C: LinearCodeFamily<F, WrappedMatrix<F, RowMajorMatrix<F>>>,
    M: DirectMMCS<F, Mat = C::Out>,
{
    pub fn new(codes: C, mmcs: M) -> Self {
        Self {
            codes,
            mmcs,
            _phantom_f: PhantomData,
            _phantom_m: PhantomData,
        }
    }
}

impl<F, In, C, M> PCS<F, In> for TensorPCS<F, C, M>
where
    F: Field,
    In: for<'a> MatrixRows<'a, F>,
    C: LinearCodeFamily<F, WrappedMatrix<F, RowMajorMatrix<F>>>,
    M: DirectMMCS<F, Mat = C::Out>,
{
    type Commitment = M::Commitment;
    type ProverData = M::ProverData;
    type Proof = Vec<M::Proof>;
    type Error = ();

    fn commit_batches(&self, polynomials: Vec<In>) -> (Self::Commitment, Self::ProverData) {
        let encoded_polynomials = polynomials
            .into_iter()
            .map(|mat| {
                let wraps = optimal_wraps(mat.width(), mat.height());
                let wrapped = WrappedMatrix::new(mat.to_row_major_matrix(), wraps);
                self.codes.encode_batch(wrapped)
            })
            .collect();
        self.mmcs.commit(encoded_polynomials)
    }
}

impl<F, In, C, M> MultivariatePCS<F, In> for TensorPCS<F, C, M>
where
    F: Field,
    In: for<'a> MatrixRows<'a, F>,
    C: LinearCodeFamily<F, WrappedMatrix<F, RowMajorMatrix<F>>>,
    M: DirectMMCS<F, Mat = C::Out>,
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
