//! Grinding (proof-of-work) bits — additive contribution to security.
//!
//! A grind sited immediately before a Fiat–Shamir challenge forces a
//! malicious prover to redo `2^pow_bits` work per resampling attempt, so it
//! adds `pow_bits` to the round-by-round error of the round that challenge
//! opens (ethSTARK [2021/582](https://eprint.iacr.org/2021/582) §5, and
//! [2024/1553](https://eprint.iacr.org/2024/1553) §2 for the round-by-round
//! accounting the composite uses).
//!
//! Which round a grind boosts therefore depends on **where** the protocol
//! grinds. [`GrindingSites`] enumerates those sites so a protocol states them
//! as data instead of the crate hardcoding one placement per protocol.

use serde::Serialize;

use crate::error::ErrorBits;

/// Bits added to the soundness budget by a `pow_bits`-bit grinding round.
/// Equal to `pow_bits` when grinding is honest; provided as a function
/// so future tweaks (multi-round PoW, variable difficulty) stay local.
pub const fn grinding_bits(pow_bits: usize) -> f64 {
    pow_bits as f64
}

/// `error` boosted by a `pow_bits`-bit grind placed before the challenge that
/// round samples. A zero-bit grind is the identity.
pub const fn boost(error: ErrorBits, pow_bits: usize) -> ErrorBits {
    ErrorBits::from_log2(error.bits() + grinding_bits(pow_bits))
}

/// Where a STARK grinds, in bits per site.
///
/// Every field defaults to `0` — a protocol declares only the sites it
/// actually uses, and a default-constructed value is neutral.
///
/// Each site is consumed by the term whose round it opens:
///
/// - [`Self::query_phase`] and [`Self::ldt_commit_phase`] are consumed by the
///   [`crate::ldt::LowDegreeTest`] implementation, which already carries them
///   (`FriRegime::query_pow_bits` / `FriRegime::commit_pow_bits`). The
///   composite does not re-apply them, so they are never double-counted.
/// - [`Self::out_of_domain`] is applied to the DEEP-ALI term by
///   [`crate::stark::proven_security_report`] and
///   [`crate::stark::conjectured_security_report`].
/// - [`Self::lookup_challenge`] is applied to the LogUp fingerprint term by
///   [`crate::logup::security_term`].
///
/// A protocol grinding at a site this crate does not model builds its own
/// [`crate::report::SecurityTerm`], applies [`boost`] to it, and passes it
/// through `extras` — no change to this struct is needed.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct GrindingSites {
    /// Bits ground once before the low-degree test samples query positions.
    pub query_phase: usize,
    /// Bits ground at every low-degree-test commit-phase round, before its
    /// folding challenge.
    pub ldt_commit_phase: usize,
    /// Bits ground before the DEEP out-of-domain point is sampled.
    pub out_of_domain: usize,
    /// Bits ground before the lookup / permutation argument's challenges are
    /// sampled.
    pub lookup_challenge: usize,
}

impl GrindingSites {
    /// No grinding at any site — the neutral element, usable in `const`
    /// contexts where [`Default::default`] is not available.
    pub const NONE: Self = Self {
        query_phase: 0,
        ldt_commit_phase: 0,
        out_of_domain: 0,
        lookup_challenge: 0,
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boost_adds_pow_bits_and_zero_is_neutral() {
        let error = ErrorBits::from_log2(80.0);
        assert!((boost(error, 0).bits() - 80.0).abs() < 1e-12);
        assert!((boost(error, 16).bits() - 96.0).abs() < 1e-12);
    }

    #[test]
    fn default_sites_are_neutral() {
        assert_eq!(GrindingSites::default(), GrindingSites::NONE);
    }
}
