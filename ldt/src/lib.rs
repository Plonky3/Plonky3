//! This crate contains a framework for low-degree tests (LDTs).

#![no_std]

extern crate alloc;

use alloc::vec::Vec;
use core::marker::PhantomData;
use p3_challenger::Challenger;
use p3_commit::MMCS;
use p3_commit::PCS;
use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;

/// A batch low-degree test (LDT).
pub trait LDT<F: Field, M: MMCS<F>> {
    type Proof;
    type Error;

    /// Prove that each column of each matrix in `codewords` is a codeword.
    fn prove<Chal>(codewords: &[M::ProverData], challenger: &mut Chal) -> Self::Proof
    where
        Chal: Challenger<F>;

    fn verify<Chal>(
        codeword_commits: &[M::Commitment],
        proof: &Self::Proof,
        challenger: &mut Chal,
    ) -> Result<(), Self::Error>
    where
        Chal: Challenger<F>;
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
    type Commitment = M::Commitment;
    type ProverData = M::ProverData;
    type Proof = L::Proof;
    type Error = L::Error;

    fn commit_batches(
        _polynomials: Vec<RowMajorMatrix<F>>,
    ) -> (Self::Commitment, Self::ProverData) {
        // (Streaming?) LDE + Merklize
        todo!()
    }

    fn get_committed_value(_prover_data: &Self::ProverData, _poly: usize, _value: usize) -> F {
        todo!()
    }
}
