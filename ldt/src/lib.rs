//! This crate contains a framework for low-degree tests (LDTs).

#![no_std]

use p3_commit::mmcs::MMCS;
use p3_field::field::Field;

/// A batch low-degree test (LDT).
pub trait LDT<F: Field, O: MMCS<F>> {
    type Proof;
    type Error;

    fn prove(codewords: &[O::Commitment]) -> Self::Proof;

    fn verify(codeword_commits: &[O::Commitment], proof: &Self::Proof) -> Result<(), Self::Error>;
}

// TODO: PCS from LDT.
