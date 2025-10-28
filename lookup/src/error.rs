//! Error Types for Interaction Constraints

use alloc::string::String;
use core::fmt;

use thiserror::Error;

/// Error type for interaction constraint failures.
#[derive(Debug, Clone, Error)]
pub enum LookupError {
    /// Global cumulative sum across all AIRs doesn't sum to zero.
    ///
    /// This means sends and receives don't balance across the system.
    #[error(
        "Global interaction '{name}' failed: cumulative sum is {actual}, expected 0 (across {num_airs} AIRs)"
    )]
    GlobalCumulativeMismatch {
        /// Name of the global interaction
        name: String,
        /// The actual cumulative sum (should be zero)
        actual: String,
        /// Number of AIRs contributing to this interaction
        num_airs: usize,
    },
}

impl LookupError {
    /// Creates a global cumulative mismatch error with rich context.
    pub fn global_mismatch(
        name: impl Into<String>,
        actual: impl fmt::Display,
        num_airs: usize,
    ) -> Self {
        Self::GlobalCumulativeMismatch {
            name: name.into(),
            actual: alloc::format!("{}", actual),
            num_airs,
        }
    }
}

/// Result type alias for interaction operations.
pub type LookupResult<T> = core::result::Result<T, LookupError>;
