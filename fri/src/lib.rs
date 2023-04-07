//! An implementation of the FRI low-degree test (LDT).

#![no_std]

extern crate alloc;

use crate::proof::FriProof;
use crate::prover::prove;
use core::marker::PhantomData;
use p3_commit::mmcs::MMCS;
use p3_field::field::FieldExtension;
use p3_ldt::LDT;

pub mod fri_pcs;
pub mod proof;
mod prover;

struct FriLDT<FE, O>
where
    FE: FieldExtension,
    O: MMCS<FE::Base>,
{
    _phantom_fe: PhantomData<FE>,
    _phantom_o: PhantomData<O>,
}

impl<FE, M> LDT<FE::Base, M> for FriLDT<FE, M>
where
    FE: FieldExtension,
    M: MMCS<FE::Base>,
{
    type Proof = FriProof<FE, M>;
    type Error = ();

    fn prove(codewords: &[M::ProverData]) -> Self::Proof {
        prove(codewords)
    }

    fn verify(_codewords: &[M::Commitment], _proof: &Self::Proof) -> Result<(), Self::Error> {
        todo!()
    }
}
