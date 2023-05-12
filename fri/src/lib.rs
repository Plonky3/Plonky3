//! An implementation of the FRI low-degree test (LDT).

#![no_std]

extern crate alloc;

use crate::proof::FriProof;
use crate::prover::prove;
use crate::verifier::verify;
use core::marker::PhantomData;
use p3_challenger::Challenger;
use p3_commit::{DirectMMCS, MMCS};
use p3_field::Field;
use p3_ldt::{LDTBasedPCS, LDT};

mod proof;
mod prover;
mod verifier;

pub use proof::*;

pub struct FriLDT<F, M, MC>
where
    F: Field,
    M: MMCS<F::Base>,
    MC: DirectMMCS<F::Base>,
{
    _phantom_f: PhantomData<F>,
    _phantom_m: PhantomData<M>,
    _phantom_mc: PhantomData<MC>,
}

impl<F, M, MC> LDT<F::Base, M> for FriLDT<F, M, MC>
where
    F: Field,
    M: MMCS<F::Base>,
    MC: DirectMMCS<F::Base>,
{
    type Proof = FriProof<F, M, MC>;
    type Error = ();

    fn prove<Chal>(codewords: &[M::ProverData], challenger: &mut Chal) -> Self::Proof
    where
        Chal: Challenger<F::Base>,
    {
        prove::<F, M, MC, Chal>(codewords, challenger)
    }

    fn verify<Chal>(
        _codeword_commits: &[M::Commitment],
        proof: &Self::Proof,
        challenger: &mut Chal,
    ) -> Result<(), Self::Error>
    where
        Chal: Challenger<F::Base>,
    {
        verify::<F, M, MC, Chal>(proof, challenger)
    }
}

pub type FRIBasedPCS<F, M, MC> = LDTBasedPCS<<F as Field>::Base, M, FriLDT<F, M, MC>>;
