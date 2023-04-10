//! An implementation of the FRI low-degree test (LDT).

#![no_std]

extern crate alloc;

use crate::proof::FriProof;
use crate::prover::prove;
use core::marker::PhantomData;
use p3_commit::mmcs::MMCS;
use p3_field::field::{Field, FieldExtension};
use p3_ldt::LDT;

pub mod fri_pcs;
pub mod proof;
mod prover;

struct FriLDT<F, FE, O>
where
    F: Field,
    FE: FieldExtension<F>,
    O: MMCS<F>,
{
    _phantom_f: PhantomData<F>,
    _phantom_fe: PhantomData<FE>,
    _phantom_o: PhantomData<O>,
}

impl<F, FE, M> LDT<F, M> for FriLDT<F, FE, M>
where
    F: Field,
    FE: FieldExtension<F>,
    M: MMCS<F>,
{
    type Proof = FriProof<F, FE, M>;
    type Error = ();

    fn prove(codewords: &[M::ProverData]) -> Self::Proof {
        prove(codewords)
    }

    fn verify(_codewords: &[M::Commitment], _proof: &Self::Proof) -> Result<(), Self::Error> {
        todo!()
    }
}
