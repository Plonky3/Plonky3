//! An implementation of the FRI low-degree test (LDT).

#![no_std]

extern crate alloc;

use p3_challenger::Challenger;
use p3_commit::MMCS;
use p3_ldt::{LDTBasedPCS, LDT};

pub use crate::proof::FriProof;
use crate::prover::prove;
use crate::verifier::verify;

mod config;
mod proof;
mod prover;
mod verifier;

pub use config::*;
pub use proof::*;

pub struct FriLDT<FC: FriConfig> {
    config: FC,
}

impl<FC: FriConfig> LDT<FC::Val, FC::InputMmcs> for FriLDT<FC> {
    type Proof = FriProof<FC>;
    type Error = ();

    fn prove<Chal>(
        &self,
        inputs: &[<FC::InputMmcs as MMCS<FC::Val>>::ProverData],
        challenger: &mut Chal,
    ) -> Self::Proof
    where
        Chal: Challenger<FC::Val>,
    {
        prove::<FC, Chal>(&self.config, inputs, challenger)
    }

    fn verify<Chal>(
        &self,
        _input_commits: &[<FC::InputMmcs as MMCS<FC::Val>>::Commitment],
        proof: &Self::Proof,
        challenger: &mut Chal,
    ) -> Result<(), Self::Error>
    where
        Chal: Challenger<FC::Val>,
    {
        verify::<FC, Chal>(proof, challenger)
    }
}

pub type FRIBasedPCS<FC, LDE> = LDTBasedPCS<
    <FC as FriConfig>::Val,
    <FC as FriConfig>::Challenge,
    LDE,
    <FC as FriConfig>::InputMmcs,
    FriLDT<FC>,
>;
