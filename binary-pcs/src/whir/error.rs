//! Why a binary WHIR commitment, schedule or opening was refused.

use p3_sumcheck::ring_switch::bits::{BitRingSwitchError, BitRingSwitchProofError};
use p3_whir::{VerifierError, WhirConfigError};
use thiserror::Error;

use crate::packing::PackError;

/// Why a Boolean commitment or opening was refused.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum BooleanWhirError {
    /// The witness is narrower than the coordinates one element absorbs.
    #[error("a bit witness of {actual} variables cannot absorb {needed} into one element")]
    WitnessTooNarrow {
        /// Coordinates one element absorbs.
        needed: usize,
        /// Variables the witness has.
        actual: usize,
    },

    /// The schedule does not commit the elements the packing holds.
    #[error("the schedule commits {actual} variables, the packing holds {expected}")]
    ConfigArity {
        /// Variables the packing holds.
        expected: usize,
        /// Variables the schedule commits.
        actual: usize,
    },

    /// The witness does not cover the committed hypercube.
    #[error("the packed witness has {actual} variables, expected {expected}")]
    WitnessArity {
        /// Variables the commitment holds.
        expected: usize,
        /// Variables the witness packs to.
        actual: usize,
    },

    /// No opening point was supplied, so there is nothing to prove.
    #[error("an opening needs at least one point")]
    NoPoints,

    /// The claim counts on the two sides of an opening disagree.
    #[error("{expected} points against {values} values and {reductions} reductions")]
    ClaimCount {
        /// Points supplied.
        expected: usize,
        /// Values supplied.
        values: usize,
        /// Reductions the proof carries.
        reductions: usize,
    },

    /// The opening point does not name the committed function's variables.
    #[error("the opening point names {actual} variables, expected {expected}")]
    PointArity {
        /// Variables the committed function has.
        expected: usize,
        /// Variables the point names.
        actual: usize,
    },

    /// An opening asks for no reading at all.
    #[error("opening {index} asks for no reading")]
    NoReading {
        /// Position of the opening.
        index: usize,
    },

    /// An opening steps within more rows than the witness has.
    #[error("opening {index} steps within {row_variables} rows of {num_variables} variables")]
    RowVariables {
        /// Position of the opening.
        index: usize,
        /// Trailing coordinates the successor view steps within.
        row_variables: usize,
        /// Variables the witness has.
        num_variables: usize,
    },

    /// A reading is present where its opening asks for none, or missing where it asks for one.
    #[error("the readings of opening {index} do not match what it asks for")]
    ReadingShape {
        /// Position of the opening.
        index: usize,
    },

    /// The bits do not reinterpret as whole elements of the alphabet.
    #[error(transparent)]
    Packing(PackError),

    /// A reduction could not be set up for its opening.
    #[error(transparent)]
    Reduction(BitRingSwitchError),

    /// A reduction proof was rejected.
    #[error(transparent)]
    ReductionProof(BitRingSwitchProofError),

    /// The proximity argument refused to commit or to open.
    #[error(transparent)]
    Commit(WhirConfigError),

    /// The proximity opening was rejected.
    #[error(transparent)]
    Opening(VerifierError),

    /// A surviving claim was not closed by the value the commitment opened.
    #[error("a surviving claim does not match the opened value")]
    SurvivingClaim,
}

/// Why a schedule or a proof was refused by the ceiling.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum BudgetError {
    /// The verifier would open more positions than the ceiling allows.
    #[error("{actual} opened positions exceed the budget of {budget}")]
    Queries {
        /// Positions the schedule opens.
        actual: usize,
        /// Positions the ceiling allows.
        budget: usize,
    },

    /// The proof is larger than the ceiling allows.
    #[error("{actual} proof bytes exceed the budget of {budget}")]
    Bytes {
        /// Bytes the proof occupies.
        actual: usize,
        /// Bytes the ceiling allows.
        budget: usize,
    },

    /// The schedule demands more grinding than the ceiling allows.
    #[error("{actual} grinding bits exceed the budget of {budget}")]
    Grinding {
        /// Bits the schedule demands.
        actual: usize,
        /// Bits the ceiling allows.
        budget: usize,
    },
}

/// Why a profile could not be turned into a schedule.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ProfileError {
    /// The derivation refused the profile.
    #[error(transparent)]
    Schedule(WhirConfigError),
}
