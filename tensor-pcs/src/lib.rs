//! A PCS using degree 2 tensor codes, based on BCG20 <https://eprint.iacr.org/2020/1426>.

#![no_std]

extern crate alloc;

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
    C: LinearCodeFamily<F, RowMajorMatrix<F>>,
    M: DirectMMCS<F>,
{
    _codes: C,
    mmcs: M,
    _phantom_f: PhantomData<F>,
    _phantom_m: PhantomData<M>,
}

impl<F, C, M> PCS<F> for TensorPCS<F, C, M>
where
    F: Field,
    C: LinearCodeFamily<F, RowMajorMatrix<F>>,
    M: DirectMMCS<F, Mat = RowMajorMatrix<F>>,
{
    type Commitment = M::Commitment;
    type ProverData = M::ProverData;
    type Proof = ();
    type Error = ();

    fn commit_batches(
        &self,
        mut polynomials: Vec<RowMajorMatrix<F>>,
    ) -> (Self::Commitment, Self::ProverData) {
        for _m in polynomials.iter_mut() {
            // TODO
            // self.codes.write_parity(m);
        }
        self.mmcs.commit(polynomials)
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
    C: LinearCodeFamily<F, RowMajorMatrix<F>>,
    M: DirectMMCS<F, Mat = RowMajorMatrix<F>>,
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
