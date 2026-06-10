#![doc = include_str!("../README.md")]
#![no_std]

extern crate alloc;

#[cfg(debug_assertions)]
mod check_constraints;
pub mod common;
pub mod config;
pub mod error;
pub mod proof;
pub mod prover;
pub mod symbolic;
pub(crate) mod transcript;
pub mod verifier;

// Re-export main types and functions for convenience
pub use common::{CommonData, ProverData, ProverOnlyData};
pub use config::{
    Challenge, Commitment, Domain, PackedChallenge, PackedVal, PcsError, PcsProof,
    StarkGenericConfig, Val,
};
pub use error::BatchVerificationError;
pub use p3_uni_stark::{OpenedValues, VerificationError};
pub use proof::{BatchCommitments, BatchOpenedValues, BatchProof};
pub use prover::{StarkInstance, prove_batch};
pub use transcript::BatchTranscript;
pub use verifier::{VerifierData, verify_batch};
