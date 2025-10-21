//! Multi-STARK proving and verification.
//!
//! This crate provides functionality for proving and verifying multiple STARK instances
//! within a single proof, using a unified commitment scheme and shared transcript.
//!
//! # Overview
//!
//! The main workflow is:
//! 1. Create multiple [`StarkInstance`]s, each with an AIR, trace, and public values
//! 2. Call [`prove_multi`] to generate a [`MultiProof`]
//! 3. Call [`verify_multi`] to verify the proof against the AIRs and public values
//!
//! # Example
//!
//! ```ignore
//! use p3_multi_stark::{prove_multi, verify_multi, StarkInstance};
//!
//! // Create instances for different computations
//! let instances = vec![
//!     StarkInstance { air: &air1, trace: trace1, public_values: pv1 },
//!     StarkInstance { air: &air2, trace: trace2, public_values: pv2 },
//! ];
//!
//! // Generate a unified proof
//! let proof = prove_multi(&config, instances);
//!
//! // Verify the proof
//! verify_multi(&config, &[&air1, &air2], &proof, &[pv1, pv2])?;
//! ```

#![no_std]

extern crate alloc;

pub mod config;
pub mod proof;
pub mod prover;
pub mod verifier;

// Re-export main types and functions for convenience
pub use config::{
    Challenge, Commitment, Domain, PackedChallenge, PackedVal, PcsError, PcsProof,
    StarkGenericConfig, Val, observe_base_as_ext,
};
pub use p3_uni_stark::{OpenedValues, VerificationError};
pub use proof::{MultiCommitments, MultiOpenedValues, MultiProof};
pub use prover::{StarkInstance, prove_multi};
pub use verifier::verify_multi;
