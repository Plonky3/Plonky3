//! Univariate skip: collapse several zerocheck rounds into one round over a subspace.
//!
//! # Overview
//!
//! A zerocheck over a bit-valued witness widens every surviving cell in its first round.
//!
//! One bit becomes a full extension element, and nothing has shrunk the hypercube yet.
//!
//! That round therefore dominates all the others put together.
//!
//! Reading `k` coordinates as subspace points replaces those `k` rounds with one message.
//!
//! The widening is then paid once instead of `k` times.
//!
//! ```text
//!     plain:  m rounds, the first widening 2^(m-1) cells
//!     skip:   1 subspace round, then m - k ordinary rounds
//! ```
//!
//! # What the caller owes
//!
//! Two things are not discharged here, both stated in full on the round:
//!
//! - The zerocheck point, which the surrounding protocol draws after committing.
//! - The final opening, which is an inner product and not yet a multilinear evaluation.
//!
//! # References
//!
//! - Gruen, *Some Improvements for the PIOP for ZeroCheck*, <https://eprint.iacr.org/2024/108>
//! - Dao, Thaler, *More Optimizations to Sum-Check Proving*, <https://eprint.iacr.org/2024/1210>
//! - Bünz, Rothblum, Wang, *Flock*, Sections 4.2 and 4.3, <https://eprint.iacr.org/2026/1329>

pub mod composition;
pub mod domain;
pub mod lde;
pub mod opening;
pub mod pinned;
pub mod round;
pub mod transcript;
pub mod zerocheck;
pub mod zerocheck_transcript;

pub use composition::{Composition, Conjunction, SquareProduct};
pub use domain::{SkipDomain, SkipDomainError};
#[cfg(any(test, feature = "test-util"))]
pub use lde::extend_reference;
pub use lde::{CHUNK_BITS, CompressedLde, CompressedLdeError};
pub use opening::{OPENING_DEGREE, SkipOpening, SkipOpeningError, SkipOpeningProver};
pub use pinned::{PinnedEqError, PinnedEqWeights, SubfieldFoldTable};
pub use round::{MessageLenMismatch, RowSelector, SkipRound, SkipRoundError};
pub use transcript::{
    SkipOpeningProverTranscript, SkipOpeningShape, SkipOpeningTranscriptError,
    SkipOpeningVerifierTranscript, UnivariateSkipProverTranscript, UnivariateSkipShape,
    UnivariateSkipTranscriptError, UnivariateSkipVerifierTranscript,
};
pub use zerocheck::{BinaryZerocheck, ZerocheckClaim, ZerocheckError, ZerocheckProof};
pub use zerocheck_transcript::{
    ZerocheckProverTranscript, ZerocheckShape, ZerocheckTranscriptError,
    ZerocheckVerifierTranscript,
};
