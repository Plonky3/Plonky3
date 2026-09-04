//! Prover-side and verifier-side transcript drivers.

mod prover;
mod verifier;

pub use prover::ProverState;
pub use verifier::VerifierState;

use crate::fs::codecs::{Codec, MIN_CHALLENGE_SECURITY_BITS};

/// Refuse to sample challenge material through a codec below the security budget.
///
/// The check is a compile-time constant evaluation.
/// A protocol wired to a biased codec fails to build instead of shipping weak challenges.
#[inline(always)]
pub(crate) const fn assert_challenge_security<C, T, Cdc: Codec<C, T>>() {
    const {
        assert!(
            Cdc::SECURITY_BITS >= MIN_CHALLENGE_SECURITY_BITS,
            "codec is too biased to sample challenge material",
        );
    }
}
