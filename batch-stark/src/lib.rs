//! Batch-STARK proving and verification.
//!
//! ```ignore
//! use p3_batch_stark::{prove_batch_no_lookups, verify_batch_no_lookups, CommonData, StarkInstance};
//!
//! let instances = vec![
//!     StarkInstance { air: &air1, trace: trace1, public_values: pv1, lookups: vec![] },
//!     StarkInstance { air: &air2, trace: trace2, public_values: pv2: lookups: vec![] },
//! ];
//!
//! let common = CommonData::from_instances(&config, &instances);
//! let proof = prove_batch_no_lookups(&config, instances, &common);
//! verify_batch_no_lookups(&config, &[air1, air2], &proof, &[pv1, pv2], &common)?;
//! ```

#![no_std]

extern crate alloc;

#[cfg(debug_assertions)]
mod check_constraints;
pub mod common;
pub mod config;
pub mod proof;
pub mod prover;
pub mod symbolic;
pub mod verifier;

// Re-export main types and functions for convenience
#[cfg(debug_assertions)]
pub use check_constraints::*;
pub use common::{CommonData, get_perm_challenges};
pub use config::{
    Challenge, Commitment, Domain, PackedChallenge, PackedVal, PcsError, PcsProof,
    StarkGenericConfig, Val,
};
pub use p3_uni_stark::{OpenedValues, VerificationError};
pub use proof::{BatchCommitments, BatchOpenedValues, BatchProof};
pub use prover::{StarkInstance, prove_batch};
pub use verifier::verify_batch;
