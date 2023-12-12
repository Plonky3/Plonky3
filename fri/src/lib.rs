//! An implementation of the FRI low-degree test (LDT).

// #![no_std]

extern crate alloc;

use alloc::vec::Vec;
use p3_commit::Mmcs;
use p3_ldt::{Ldt, LdtBasedPcs};
use p3_matrix::Dimensions;
use verifier::VerificationErrorForFriConfig;

use crate::prover::prove;
use crate::verifier::verify;

mod config;
mod fold_even_odd;
mod matrix_reducer;
mod proof;
mod prover;
mod verifier;

pub use config::*;
pub use fold_even_odd::*;
pub use proof::*;

pub struct FriLdt<FC: FriConfig> {
    pub config: FC,
}

impl<FC: FriConfig> Ldt<FC::Val, FC::InputMmcs, FC::Challenger> for FriLdt<FC> {
    type Proof = FriProof<FC>;
    type Error = VerificationErrorForFriConfig<FC>;

    fn log_blowup(&self) -> usize {
        self.config.log_blowup()
    }

    fn prove(
        &self,
        input_mmcs: &[FC::InputMmcs],
        input_data: &[&<FC::InputMmcs as Mmcs<FC::Val>>::ProverData],
        challenger: &mut FC::Challenger,
    ) -> Self::Proof {
        prove::<FC>(&self.config, input_mmcs, input_data, challenger)
    }

    fn verify(
        &self,
        input_mmcs: &[FC::InputMmcs],
        input_dims: &[Vec<Dimensions>],
        input_commits: &[<FC::InputMmcs as Mmcs<FC::Val>>::Commitment],
        proof: &Self::Proof,
        challenger: &mut FC::Challenger,
    ) -> Result<(), Self::Error> {
        verify::<FC>(
            &self.config,
            input_mmcs,
            input_dims,
            input_commits,
            proof,
            challenger,
        )
    }
}

pub type FriBasedPcs<FC, Mmcs, Dft, Challenger> = LdtBasedPcs<
    <FC as FriConfig>::Val,
    <FC as FriConfig>::Challenge,
    Dft,
    Mmcs,
    FriLdt<FC>,
    Challenger,
>;
