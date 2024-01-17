use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_code::LinearCodeFamily;
use p3_commit::{DirectMmcs, Pcs};
use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::MatrixRows;

use crate::reshape::optimal_wraps;
use crate::wrapped_matrix::WrappedMatrix;

pub struct TensorPcs<F, C, M>
where
    F: Field,
    C: LinearCodeFamily<F, WrappedMatrix<F, RowMajorMatrix<F>>>,
    M: DirectMmcs<F>,
{
    codes: C,
    mmcs: M,
    _phantom_f: PhantomData<F>,
    _phantom_m: PhantomData<M>,
}

impl<F, C, M> TensorPcs<F, C, M>
where
    F: Field,
    C: LinearCodeFamily<F, WrappedMatrix<F, RowMajorMatrix<F>>, Out = RowMajorMatrix<F>>,
    M: DirectMmcs<F>,
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

impl<F, In, C, M> Pcs<F, In> for TensorPcs<F, C, M>
where
    F: Field,
    In: MatrixRows<F>,
    C: LinearCodeFamily<F, WrappedMatrix<F, RowMajorMatrix<F>>, Out = RowMajorMatrix<F>>,
    M: DirectMmcs<F>,
{
    type Commitment = M::Commitment;
    type ProverData = M::ProverData;
    type Proof = Vec<M::Proof>;
    type Error = ();

    fn combine(&self, data: &[Self::ProverData]) -> (Self::Commitment, Self::ProverData) {
        self.mmcs.combine(data)
    }

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

// TODO: Impl MultivariatePcs
