//! An implementation of the FRI low-degree test (LDT).

#![no_std]

extern crate alloc;

use p3_commit::Mmcs;
use p3_ldt::{Ldt, LdtBasedPcs};

use crate::prover::prove;
use crate::verifier::verify;

mod config;
mod fold_even_odd;
mod proof;
mod prover;
mod verifier;

pub use config::*;
pub use fold_even_odd::*;
pub use proof::*;

pub struct FriLdt<FC: FriConfig> {
    pub config: FC,
}

impl<FC: FriConfig> Ldt<FC::Val, FC::Domain, FC::InputMmcs, FC::Challenger> for FriLdt<FC> {
    type Proof = FriProof<FC>;
    type Error = ();

    fn prove(
        &self,
        mmcs: &[FC::InputMmcs],
        inputs: &[&<FC::InputMmcs as Mmcs<FC::Domain>>::ProverData],
        challenger: &mut FC::Challenger,
    ) -> Self::Proof {
        prove::<FC>(&self.config, mmcs, inputs, challenger)
    }

    fn verify(
        &self,
        _input_commits: &[<FC::InputMmcs as Mmcs<FC::Domain>>::Commitment],
        proof: &Self::Proof,
        challenger: &mut FC::Challenger,
    ) -> Result<(), Self::Error> {
        verify::<FC>(proof, challenger)
    }
}

pub type FriBasedPcs<FC, Mmcs, Dft, Challenger> = LdtBasedPcs<
    <FC as FriConfig>::Val,
    <FC as FriConfig>::Domain,
    <FC as FriConfig>::Challenge,
    Dft,
    Mmcs,
    FriLdt<FC>,
    Challenger,
>;
