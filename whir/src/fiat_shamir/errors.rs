//! Fiat-Shamir transcript errors for WHIR protocol challenges.

use thiserror::Error;

/// Granular error types for Fiat-Shamir operations.
///
/// Each variant represents a specific failure mode in the Fiat-Shamir transform.
#[derive(Error, Debug, Clone)]
pub enum FiatShamirError {
    /// Proof-of-work witness fails difficulty requirement.
    #[error("Invalid grinding witness: proof-of-work verification failed")]
    InvalidGrindingWitness,
}
