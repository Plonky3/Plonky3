//! An implementation of the FRI low-degree test (LDT).

#![no_std]

extern crate alloc;

use p3_commit::Mmcs;
use p3_ldt::{Ldt, LdtBasedPcs};

pub use crate::proof::FriProof;
use crate::prover::prove;
use crate::verifier::verify;

mod config;
mod fold_even_odd;
mod proof;
mod prover;
mod verifier;

pub use config::*;
pub use proof::*;

pub struct FriLdt<FC: FriConfig> {
    config: FC,
}

impl<FC: FriConfig> Ldt<FC::Val, FC::InputMmcs, FC::Challenger> for FriLdt<FC> {
    type Proof = FriProof<FC>;
    type Error = ();

    fn prove(
        &self,
        inputs: &[<FC::InputMmcs as Mmcs<FC::Val>>::ProverData],
        challenger: &mut FC::Challenger,
    ) -> Self::Proof {
        prove::<FC>(&self.config, inputs, challenger)
    }

    fn verify(
        &self,
        _input_commits: &[<FC::InputMmcs as Mmcs<FC::Val>>::Commitment],
        proof: &Self::Proof,
        challenger: &mut FC::Challenger,
    ) -> Result<(), Self::Error> {
        verify::<FC>(proof, challenger)
    }
}

pub type FriBasedPcs<FC, Dft> = LdtBasedPcs<
    <FC as FriConfig>::Val,
    <FC as FriConfig>::Challenge,
    Dft,
    <FC as FriConfig>::InputMmcs,
    FriLdt<FC>,
>;
