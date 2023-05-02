//! An implementation of the FRI low-degree test (LDT).

#![no_std]

extern crate alloc;

use crate::proof::FriProof;
use crate::prover::prove;
use core::marker::PhantomData;
use p3_commit::mmcs::MMCS;
use p3_field::field::Field;
use p3_ldt::{LDTBasedPCS, LDT};

pub mod proof;
pub mod prover;

pub struct FriLDT<F, M>
where
    F: Field,
    M: MMCS<F>,
{
    _phantom_f: PhantomData<F>,
    _phantom_o: PhantomData<M>,
}

impl<F, M> LDT<F, M> for FriLDT<F, M>
where
    F: Field,
    M: MMCS<F>,
{
    type Proof = FriProof<F, M>;
    type Error = ();

    fn prove(codewords: &[M::ProverData]) -> Self::Proof {
        prove(codewords)
    }

    fn verify(_codewords: &[M::Commitment], _proof: &Self::Proof) -> Result<(), Self::Error> {
        todo!()
    }
}

pub type FRIBasedPCS<F, M> = LDTBasedPCS<F, M, FriLDT<F, M>>;
