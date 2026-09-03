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
/// - [`Self::out_of_domain`] is applied to the DEEP-ALI term by
///   [`crate::stark::proven_security_report`] and
///   [`crate::stark::conjectured_security_report`].
/// - [`Self::batch_combination`] is applied to the batched-openings term by
///   [`crate::stark::proven_security_report`].
/// - [`Self::lookup_challenge`] is applied to the LogUp fingerprint term by
///   [`crate::logup::security_term`].
///
/// The low-degree test's own grinding sites (e.g. FRI's query- and
/// commit-phase proof-of-work) are **not** modeled here: a
/// [`crate::ldt::LowDegreeTest`] implementation carries those itself
/// (`FriRegime::query_pow_bits` / `FriRegime::commit_pow_bits`) and folds
/// them into the terms it returns, so the composite never re-applies them —
/// a site in this struct for them would be read back out unchanged, never
/// consulted.
///
/// A protocol grinding at a site this crate does not model builds its own
/// [`crate::report::SecurityTerm`], applies [`boost`] to it, and passes it
/// through `extras` — no change to this struct is needed.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct GrindingSites {
    /// Bits ground before the DEEP out-of-domain point is sampled.
    pub out_of_domain: usize,
    /// Bits ground before the challenge that random-linear-combines the
    /// committed codewords into the single low-degree-test instance.
    ///
    /// This is the opening-batching challenge of the polynomial commitment
    /// scheme (`alpha` in `p3_fri::TwoAdicFriPcs::open`), **not** a FRI
    /// folding challenge — those are `FriRegime::commit_pow_bits`, carried by
    /// the low-degree test itself. It is credited only to the term
    /// [`crate::report::BATCH_LABEL`] names, which exists only when more than
    /// one codeword is batched (`InstanceShape::num_batched_functions >= 2`);
    /// with nothing to batch there is no such round and these bits buy
    /// nothing.
    ///
    /// Only the proven path models the batched-openings round, so grinding
    /// here does not move the conjectured report — see
    /// [`crate::stark::conjectured_security_report`]'s "Not modeled".
    pub batch_combination: usize,
    /// Bits ground before the lookup / permutation argument's challenges are
    /// sampled.
    pub lookup_challenge: usize,
}

impl GrindingSites {
    /// No grinding at any site — the neutral element, usable in `const`
    /// contexts where [`Default::default`] is not available.
    pub const NONE: Self = Self {
        out_of_domain: 0,
        batch_combination: 0,
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
