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

    /// The number of elements doesn't match the number of multiplicities.
    #[error(
        "Mismatched lengths in interaction '{name}': {element_count} elements but {multiplicity_count} multiplicities"
    )]
    LengthMismatch {
        /// Name or identifier of the interaction
        name: String,
        /// Number of element tuples
        element_count: usize,
        /// Number of multiplicities
        multiplicity_count: usize,
    },

    /// Insufficient auxiliary columns allocated for the interaction.
    #[error(
        "Interaction '{name}' needs {required} auxiliary columns but only {available} were allocated"
    )]
    InsufficientAuxiliaryColumns {
        /// Name of the interaction
        name: String,
        /// Required number of columns
        required: usize,
        /// Actually allocated columns
        available: usize,
    },

    /// Not enough challenges provided for the protocol.
    #[error("Interaction needs {required} challenges but only {available} were provided")]
    InsufficientChallenges {
        /// Required number of challenges
        required: usize,
        /// Number of challenges actually available
        available: usize,
    },

    /// A constraint violation detected during verification.
    #[error("Constraint violation in interaction '{name}' at row {row}: {constraint_type}")]
    ConstraintViolation {
        /// Name of the interaction
        name: String,
        /// Row index where violation occurred
        row: usize,
        /// Description of what constraint failed
        constraint_type: String,
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

    /// Creates a length mismatch error.
    pub fn length_mismatch(
        name: impl Into<String>,
        element_count: usize,
        multiplicity_count: usize,
    ) -> Self {
        Self::LengthMismatch {
            name: name.into(),
            element_count,
            multiplicity_count,
        }
    }

    /// Creates an insufficient columns error.
    pub fn insufficient_columns(
        name: impl Into<String>,
        required: usize,
        available: usize,
    ) -> Self {
        Self::InsufficientAuxiliaryColumns {
            name: name.into(),
            required,
            available,
        }
    }

    /// Creates an insufficient challenges error.
    pub const fn insufficient_challenges(required: usize, available: usize) -> Self {
        Self::InsufficientChallenges {
            required,
            available,
        }
    }

    /// Creates a constraint violation error.
    pub fn constraint_violation(
        name: impl Into<String>,
        row: usize,
        constraint_type: impl Into<String>,
    ) -> Self {
        Self::ConstraintViolation {
            name: name.into(),
            row,
            constraint_type: constraint_type.into(),
        }
    }
}

/// Result type alias for interaction operations.
pub type LookupResult<T> = core::result::Result<T, LookupError>;
