//! This crate contains a framework for low-degree tests (LDTs).

#![no_std]

mod ldt_based_pcs;
mod quotient;

use alloc::vec::Vec;

pub use ldt_based_pcs::*;
use p3_challenger::FieldChallenger;
use p3_commit::Mmcs;
use p3_field::Field;
use p3_matrix::Dimensions;
pub use quotient::*;

extern crate alloc;

/// A batch low-degree test (LDT).
pub trait Ldt<Val, M, Challenger>
where
    Val: Field,
    M: Mmcs<Val>,
    Challenger: FieldChallenger<Val>,
{
    type Proof;
    type Error;

    fn log_blowup(&self) -> usize;

    fn blowup(&self) -> usize {
        1 << self.log_blowup()
    }

    /// Prove that each column of each matrix in `codewords` is a codeword.
    fn prove(
        &self,
        input_mmcs: &[M],
        input_data: &[&M::ProverData],
        challenger: &mut Challenger,
    ) -> Self::Proof;

    fn verify(
        &self,
        input_mmcs: &[M],
        input_dims: &[Vec<Dimensions>],
        input_commits: &[M::Commitment],
        proof: &Self::Proof,
        challenger: &mut Challenger,
    ) -> Result<(), Self::Error>;
}
