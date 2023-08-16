//! This crate contains a framework for low-degree tests (LDTs).

#![no_std]

mod ldt_based_pcs;
mod quotient;

pub use ldt_based_pcs::*;
use p3_challenger::FieldChallenger;
use p3_commit::Mmcs;
use p3_field::Field;

extern crate alloc;

/// A batch low-degree test (LDT).
pub trait Ldt<F: Field, M: Mmcs<F>, Challenger: FieldChallenger<F>> {
    type Proof;
    type Error;

    /// Prove that each column of each matrix in `codewords` is a codeword.
    fn prove(&self, inputs: &[M::ProverData], challenger: &mut Challenger) -> Self::Proof;

    fn verify(
        &self,
        input_commits: &[M::Commitment],
        proof: &Self::Proof,
        challenger: &mut Challenger,
    ) -> Result<(), Self::Error>;
}
