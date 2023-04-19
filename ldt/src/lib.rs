//! This crate contains a framework for low-degree tests (LDTs).

#![no_std]

extern crate alloc;

use alloc::vec::Vec;
use core::marker::PhantomData;
use p3_commit::mmcs::MMCS;
use p3_commit::pcs::PCS;
use p3_field::field::Field;
use p3_matrix::dense::RowMajorMatrix;

/// A batch low-degree test (LDT).
pub trait LDT<F: Field, M: MMCS<F>> {
    type Proof;
    type Error;

    /// Prove that each column of each matrix in `codewords` is a codeword.
    fn prove(codewords: &[M::ProverData]) -> Self::Proof;

    fn verify(codeword_commits: &[M::Commitment], proof: &Self::Proof) -> Result<(), Self::Error>;
}

pub struct LDTBasedPCS<F, M, L>
where
    F: Field,
    M: MMCS<F>,
    L: LDT<F, M>,
{
    _phantom_f: PhantomData<F>,
    _phantom_m: PhantomData<M>,
    _phantom_l: PhantomData<L>,
}

impl<F, M, L> PCS<F> for LDTBasedPCS<F, M, L>
where
    F: Field,
    M: MMCS<F>,
    L: LDT<F, M>,
{
    type Commitment = Vec<M::Commitment>;
    type ProverData = Vec<M::ProverData>;
    type Proof = L::Proof;
    type Error = L::Error;

    fn commit_batches(
        _polynomials: Vec<RowMajorMatrix<F>>,
    ) -> (Self::Commitment, Self::ProverData) {
        todo!()
    }

    fn get_committed_value(
        _prover_data: &Self::ProverData,
        _batch: usize,
        _poly: usize,
        _value: usize,
    ) -> F {
        // prover_data[batch].get(...)
        todo!()
    }
}
