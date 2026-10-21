//! The lincheck: the zerocheck's claims on `A z`, `B z` and `z`, reduced to one claim on `z` (ePrint 2026/1329, Appendix A).
//!
//! The matrices are block-diagonal, so a claim at `(s_out, s_in)` collapses to `2^k_log` terms of the witness folded at `s_out`.
//!
//! The prover forms the column weights of `A_0 + alpha B_0 + alpha^2 I` by one backward walk, pins the constant wire at `alpha^3`, and runs a product sumcheck over the inner variables past the skip.
//!
//! The verifier closes it by one forward walk, then folds the 64 slices left into one claim at a fresh skip coordinate.
//!
//! Soundness error at most `L (3 + 2 (k_log - 6) + 63 + log_instances) / |E|`, `L` the opening's list size; `log_instances / |E|` is the pin's.

mod fold;
mod prover;
mod sumcheck;
mod transcript;
mod verifier;

#[cfg(test)]
mod tests;

use alloc::vec::Vec;

use p3_challenger::fs::TranscriptError;
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub(crate) use self::fold::partial_fold;
pub use self::prover::prove;
pub use self::verifier::verify;
use crate::point::{QuirkyClaim, QuirkyPoint};
use crate::skip::{K_SKIP, interpolate};
use crate::{Batch, Challenge};

/// Every prover message of one lincheck.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LincheckProof {
    /// Per inner round past the skip, `q(0)` and `q(inf)`.
    pub rounds: Vec<[Challenge; 2]>,
    /// The witness at the sumcheck point, one value per low bit pattern.
    pub slices: Vec<Challenge>,
}

/// Why the lincheck verifier rejects.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum LincheckError {
    /// The zerocheck point does not name the witness variables.
    #[error("the zerocheck point names {actual} coordinates past the skip, expected {expected}")]
    PointArity {
        /// The witness variables less the skipped ones.
        expected: usize,
        /// Supplied.
        actual: usize,
    },
    /// Wrong number of sumcheck rounds.
    #[error("the proof carries {actual} rounds, expected {expected}")]
    RoundCount {
        /// The inner variables past the skip.
        expected: usize,
        /// Carried.
        actual: usize,
    },
    /// Wrong number of slices.
    #[error("the proof carries {actual} slices, expected {expected}")]
    SliceCount {
        /// One per skip point.
        expected: usize,
        /// Carried.
        actual: usize,
    },
    /// The transcript refused a message.
    #[error(transparent)]
    Transcript(#[from] TranscriptError),
    /// The sumcheck's final value is not the circuit's terms at its point.
    #[error("the sumcheck's final claim does not match the circuit")]
    FinalCheck,
}

/// Split a zerocheck point's multilinear coordinates into instance and inner ones.
///
/// # Errors
///
/// The point does not name the batch's variables past the skip.
fn split_point<'p>(
    point: &'p QuirkyPoint,
    batch: &Batch<'_>,
) -> Result<(&'p [Challenge], &'p [Challenge]), LincheckError> {
    let expected = batch.log_n() - K_SKIP;
    if point.rest.len() != expected {
        return Err(LincheckError::PointArity {
            expected,
            actual: point.rest.len(),
        });
    }
    Ok(point.rest.split_at(batch.log_instances()))
}

/// The claim the slices leave: their interpolant at a fresh skip coordinate, at the instance and sumcheck coordinates.
fn slice_claim(
    skip: Challenge,
    outer: &[Challenge],
    inner: &[Challenge],
    slices: &[Challenge],
) -> QuirkyClaim {
    QuirkyClaim {
        point: QuirkyPoint {
            skip,
            rest: [outer, inner].concat(),
        },
        value: interpolate(skip, slices),
    }
}
