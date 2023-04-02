//! This crate contains a framework for low-degree tests (LDTs).

#![no_std]

use p3_commit::vector_commit::VectorCommitmentScheme;
use p3_field::field::Field;

/// A low-degree test (LDT).
pub trait LDT<F: Field, VCS: VectorCommitmentScheme<F>> {
    type Error;

    fn test() -> Result<(), Self::Error>;
}

// TODO: PCS from LDT.
