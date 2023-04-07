//! This crate contains a framework for low-degree tests (LDTs).

#![no_std]

use p3_commit::mmcs::MMCS;
use p3_field::field::Field;

/// A batch low-degree test (LDT).
pub trait LDT<F: Field, M: MMCS<F>> {
    type Proof;
    type Error;

    /// Prove that each column of each matrix in `codewords` is a codeword.
    fn prove(codewords: &[M::ProverData]) -> Self::Proof;

    fn verify(codeword_commits: &[M::Commitment], proof: &Self::Proof) -> Result<(), Self::Error>;
}

// TODO: Adapter which builds PCS from LDT.
