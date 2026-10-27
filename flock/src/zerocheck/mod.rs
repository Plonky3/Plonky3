//! The zerocheck of one circuit batch: `a(x) b(x) - z(x) = 0` on every bit index (ePrint 2026/1329, Appendix A).
//!
//! It reduces the batch to claims on the quirky extensions of `a`, `b` and `z` at one point.
//!
//! 1. The verifier draws the eq point past seven fixed coordinates.
//! 2. Round 1 is a univariate skip over the six low bits, sent on the coset `Lambda`.
//! 3. One multilinear round per remaining variable, low variable first, each split as in ePrint 2024/108.
//! 4. The prover sends `a`, `b` and `c` at the point; `c` is fixed by the final claim.
//!
//! The prover never reads `z`: it derives `c = a AND b` from the bits it reads.
//!
//! A dishonest `z` changes nothing it sends, and the claim on `c` then misses `z`'s extension.
//!
//! Soundness error at most `L ((n - 7) + 127 + 2 n) / |E|`, `n` the variables past the skip and `L` the opening's list size.

mod fixed;
mod fold;
mod multilinear;
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vbmi",
    target_feature = "gfni",
    target_feature = "vpclmulqdq"
))]
mod planar;
mod prover;
mod round1;
mod transcript;
mod verifier;

#[cfg(test)]
mod tests;

use alloc::vec::Vec;

use p3_challenger::fs::TranscriptError;
use p3_multilinear_util::point::Point;
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub use self::prover::prove;
pub use self::verifier::verify;
use crate::point::QuirkyPoint;
use crate::skip::K_SKIP;
use crate::{Challenge, Val};

/// The fewest witness variables a zerocheck takes: the skip, then the fixed coordinates.
pub const MIN_LOG_N: usize = K_SKIP + fixed::N_FIXED;

pub(crate) use self::fixed::N_FIXED;

/// Claims on the quirky extensions of `a = A z`, `b = B z` and `c = z`, all at one point.
#[must_use = "the claims prove nothing until the lincheck pins them to the witness"]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ZerocheckClaims {
    /// The point.
    pub point: QuirkyPoint,
    /// `a` at the point.
    pub a: Challenge,
    /// `b` at the point.
    pub b: Challenge,
    /// `c` at the point.
    pub c: Challenge,
}

/// One multilinear round of the proof.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ZerocheckRound {
    /// The linear and quadratic coefficients of the round polynomial; the claim fixes the constant.
    pub coeffs: [Challenge; 2],
    /// The grinding witness before the round's challenge, when grinding is on.
    pub pow: Option<Val>,
}

/// Everything the zerocheck sends.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ZerocheckProof {
    /// The round-1 polynomial on `Lambda`.
    pub round1: Vec<Challenge>,
    /// The grinding witness before the skip challenge, when grinding is on.
    pub round1_pow: Option<Val>,
    /// One entry per variable past the skip.
    pub rounds: Vec<ZerocheckRound>,
    /// `a`, `b` and `c` at the point.
    pub evals: [Challenge; 3],
}

/// Why the zerocheck verifier refuses.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum ZerocheckError {
    /// Round 1 does not have one value per coset point.
    #[error("round 1 carries {actual} values, expected {expected}")]
    Round1Len {
        /// One per point of `Lambda`.
        expected: usize,
        /// Carried.
        actual: usize,
    },
    /// The proof does not have one round per variable past the skip.
    #[error("the proof carries {actual} rounds, expected {expected}")]
    RoundCount {
        /// The variables past the skip.
        expected: usize,
        /// Carried.
        actual: usize,
    },
    /// The transcript refused a message or a grinding witness.
    #[error(transparent)]
    Transcript(#[from] TranscriptError),
    /// The evaluations miss the sumcheck's final claim.
    #[error("the evaluations do not reproduce the final claim")]
    TerminalMismatch,
}

/// Where a batch's witness is known zero: past the useful bits of every instance.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Padding {
    /// Bits per instance, as a power of two.
    pub(crate) k_log: usize,
    /// Bits at the start of every instance that may be nonzero.
    pub(crate) useful_bits: usize,
}

impl Padding {
    /// A cube of `2^m` bits, every bit useful.
    #[cfg(test)]
    pub(crate) const fn dense(m: usize) -> Self {
        Self {
            k_log: m,
            useful_bits: 1 << m,
        }
    }
}

/// `eq(r, x)` as a low table times a high table, coordinate `k` binding bit `k` of `x`.
#[derive(Clone, Debug)]
pub(crate) struct EqHalves {
    /// The table over the low variables.
    pub(crate) low: Vec<Challenge>,
    /// The table over the remaining variables.
    pub(crate) high: Vec<Challenge>,
}

impl EqHalves {
    /// The split after the first `low_vars` coordinates.
    pub(crate) fn new(r: &[Challenge], low_vars: usize) -> Self {
        let (low, high) = r.split_at(low_vars);
        Self {
            low: Point::new(low.to_vec()).equality_weights_lsb(),
            high: Point::new(high.to_vec()).equality_weights_lsb(),
        }
    }
}
