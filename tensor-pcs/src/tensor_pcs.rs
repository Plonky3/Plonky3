use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_challenger::FieldChallenger;
use p3_code::LinearCodeFamily;
use p3_commit::{DirectMmcs, MultivariatePcs, Pcs};
use p3_field::{ExtensionField, Field};
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

impl<F, In, C, M, Chal> Pcs<F, In, Chal> for TensorPcs<F, C, M>
where
    F: Field,
    In: MatrixRows<F>,
    C: LinearCodeFamily<F, WrappedMatrix<F, RowMajorMatrix<F>>, Out = RowMajorMatrix<F>>,
    M: DirectMmcs<F>,
    Chal: FieldChallenger<F>,
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

impl<F, In, C, M, Chal> MultivariatePcs<F, In, Chal> for TensorPcs<F, C, M>
where
    F: Field,
    In: MatrixRows<F>,
    C: LinearCodeFamily<F, WrappedMatrix<F, RowMajorMatrix<F>>, Out = RowMajorMatrix<F>>,
    M: DirectMmcs<F>,
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
