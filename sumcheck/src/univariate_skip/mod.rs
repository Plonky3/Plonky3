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
//! # What lives here
//!
//! - The subspace the round vanishes on, and the larger one it is transmitted on.
//! - The lookup table that carries bit-valued rows from the first to the second.
//! - Zerocheck challenge coordinates fixed ahead of time, with their equality weights.
//! - The round itself: its message, its verifier check, and the binding it leaves behind.
//!
//! The round hands back an ordinary multilinear claim.
//!
//! The generic-degree sumcheck driver finishes it from there.
//!
//! # References
//!
//! - Gruen, *Some Improvements for the PIOP for ZeroCheck*, <https://eprint.iacr.org/2024/108>
//! - Dao, Thaler, *More Optimizations to Sum-Check Proving*, <https://eprint.iacr.org/2024/1210>
//! - Bünz, Rothblum, Wang, *Flock*, Sections 4.2 and 4.3, <https://eprint.iacr.org/2026/1329>

pub mod domain;
pub mod lde;
pub mod pinned;
pub mod round;
pub mod transcript;

pub use domain::{SkipDomain, SkipDomainError};
pub use lde::{CHUNK_BITS, CompressedLde, CompressedLdeError, extend_reference};
pub use pinned::{PinnedEqError, PinnedEqWeights, SubfieldFoldTable};
pub use round::{RowSelector, SkipRound, SkipRoundError};
pub use transcript::{
    UnivariateSkipProverTranscript, UnivariateSkipShape, UnivariateSkipTranscriptError,
    UnivariateSkipVerifierTranscript,
};
