//! An implementation of the FRI low-degree test (LDT).

#![no_std]

extern crate alloc;

use crate::proof::FriProof;
use crate::prover::prove;
use crate::verifier::verify;
use core::marker::PhantomData;
use p3_challenger::Challenger;
use p3_commit::{DirectMMCS, MMCS};
use p3_field::FieldExtension;
use p3_ldt::{LDTBasedPCS, LDT};

mod proof;
mod prover;
mod verifier;

pub use proof::*;

pub struct FriLDT<FE, M, MC>
where
    FE: FieldExtension,
    M: MMCS<FE::Base>,
    MC: DirectMMCS<FE::Base>,
{
    _phantom_fe: PhantomData<FE>,
    _phantom_m: PhantomData<M>,
    _phantom_mc: PhantomData<MC>,
}

impl<FE, M, MC> LDT<FE::Base, M> for FriLDT<FE, M, MC>
where
    FE: FieldExtension,
    M: MMCS<FE::Base>,
    MC: DirectMMCS<FE::Base>,
{
    type Proof = FriProof<FE, M, MC>;
    type Error = ();

    fn prove<Chal>(codewords: &[M::ProverData], challenger: &mut Chal) -> Self::Proof
    where
        Chal: Challenger<FE::Base>,
    {
        prove::<FE, M, MC, Chal>(codewords, challenger)
    }

    fn verify<Chal>(
        _codeword_commits: &[M::Commitment],
        proof: &Self::Proof,
        challenger: &mut Chal,
    ) -> Result<(), Self::Error>
    where
        Chal: Challenger<FE::Base>,
    {
        verify::<FE, M, MC, Chal>(proof, challenger)
    }
}

pub type FRIBasedPCS<FE, M, MC> = LDTBasedPCS<<FE as FieldExtension>::Base, M, FriLDT<FE, M, MC>>;
