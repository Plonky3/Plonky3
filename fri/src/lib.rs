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

impl<FE, O> LDT<FE::Base, O> for FriLDT<FE, O>
where
    FE: FieldExtension,
    O: MMCS<FE::Base>,
{
    type Proof = FriProof<FE, O>;
    type Error = ();

    fn prove(codewords: &[O::Commitment]) -> Self::Proof {
        prove(codewords)
    }

    fn verify(_codewords: &[O::Commitment], _proof: &Self::Proof) -> Result<(), Self::Error> {
        todo!()
    }
}
