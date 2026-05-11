//! Error types for recursive verification.

use alloc::string::String;

use p3_circuit::{CircuitBuilderError, CircuitError};
use thiserror::Error;

use crate::generation::GenerationError;

/// Errors that can occur during recursive STARK verification.
#[derive(Debug, Error)]
pub enum VerificationError {
    /// The proof structure is invalid (wrong dimensions, missing data, etc.)
    #[error("Invalid proof shape: {0}")]
    InvalidProofShape(String),

    /// ZK randomization is inconsistent (random commitment exists but no opened values)
    #[error("Missing random opened values for existing random commitment")]
    RandomizationError,

    /// Error from the circuit execution layer
    #[error("Circuit error: {0}")]
    Circuit(#[from] CircuitError),

    /// Error from the circuit builder layer
    #[error("Circuit builder error: {0}")]
    CircuitBuilder(#[from] CircuitBuilderError),

    /// Error from challenge generation
    #[error("Generation error: {0}")]
    Generation(#[from] GenerationError),
}
