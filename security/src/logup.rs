//! LogUp (logarithmic-derivative lookup) argument soundness.
//!
//! The `p3-lookup` crate fingerprints each bus message into a single
//! extension-field denominator
//!
//! ```text
//!     denom = alpha + (bus + 1)·beta^W − Σ_k beta^k · payload_k
//! ```
//!
//! and enforces a running sum `Σ ± multiplicity / denom = 0` over the trace.
//! Soundness is a Schwartz–Zippel argument in the Fiat–Shamir pair
//! `(alpha, beta)`, sampled after the main commitment: an unbalanced multiset
//! stays unbalanced unless `(alpha, beta)` is a root of the cleared
//! fingerprint polynomial.
//!
//! Cleared of denominators, that polynomial has degree ≤ `N` in `alpha` and
//! ≤ `N·W` in `beta`, where `N = num_interactions · trace_length` is the
//! number of denominator factors (interactions summed over all buses) and
//! `W = max_message_width` is the widest payload. Sampling both challenges
//! independently and uniformly from the challenge field `EF`, the union bound
//! gives
//!
//! ```text
//!     ε_logup ≤ N·(W + 1) / |EF|.
//! ```
//!
//! Bus batching is subsumed: `N` already sums interactions over every bus, and
//! the `beta^W` bus offset is injective by construction (`p3_lookup`'s
//! `Challenges`), so it contributes no separate error term. The multiplicity
//! height bound (`Σ wᵢ·hᵢ < p`) is enforced in-circuit and is not modeled here.
//!
//! The resulting term is passed to [`crate::stark::proven_security_report`] as
//! an `extras` [`SecurityTerm`] by the protocol call site; it is not wired into
//! the composite automatically, since not every STARK uses lookups.
//!
//! # References
//! - Haböck, *Multivariate lookups based on logarithmic derivatives*
//!   ([2022/1530](https://eprint.iacr.org/2022/1530)).

use libm::log2;

use crate::error::ErrorBits;
use crate::report::SecurityTerm;
use crate::shape::InstanceShape;

/// Label for the LogUp fingerprint term in a [`crate::report::SecurityReport`].
pub const LOGUP_LABEL: &str = "logup-fingerprint";

/// Shape of a LogUp lookup argument, as seen by its soundness bound.
#[derive(Copy, Clone, Debug)]
pub struct LogUpAir {
    /// Number of bus messages emitted per trace row, summed across all buses
    /// (global + local interactions). Each is one denominator factor.
    pub num_interactions: usize,
    /// Widest message payload tuple (`max_message_width`) — the degree of the
    /// `beta`-compression.
    pub max_message_width: usize,
}

/// `-log2(ε_logup)` in bits, following `ε_logup ≤ N·(W + 1) / |EF|` with
/// `N = num_interactions · 2^log_trace_length` and `W = max_message_width`.
///
/// Returns 0 bits for degenerate inputs (no interactions, or unknown field
/// size). Prefer [`security_term`], which omits the term entirely when there
/// are no interactions rather than reporting a binding 0-bit bound.
pub fn fingerprint_error(air: &LogUpAir, shape: &InstanceShape) -> ErrorBits {
    if air.num_interactions == 0 || shape.modulus_bits == 0 {
        return ErrorBits::from_log2(0.0);
    }
    let log_n = shape.log_trace_length as f64 + log2(air.num_interactions as f64);
    let width = air.max_message_width.max(1) as f64;
    let bits = shape.modulus_bits as f64 - log_n - log2(width + 1.0);
    ErrorBits::from_log2(bits.max(0.0))
}

/// The LogUp fingerprint term for use in `extras`, or `None` when the AIR has
/// no interactions (no lookup argument, hence no term).
pub fn security_term(air: &LogUpAir, shape: &InstanceShape) -> Option<SecurityTerm> {
    if air.num_interactions == 0 {
        return None;
    }
    Some(SecurityTerm::new(
        LOGUP_LABEL,
        fingerprint_error(air, shape),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape(modulus_bits: usize) -> InstanceShape {
        InstanceShape {
            log_trace_length: 20,
            modulus_bits,
            collision_resistance: 128,
            num_batched_functions: 1,
        }
    }

    /// `ε ≤ N·(W+1)/|EF|` → bits = |EF| − log2(N) − log2(W+1). For
    /// N = 2^20 · 16 = 2^24 and W = 3: 128 − 24 − log2(4) = 102.
    #[test]
    fn fingerprint_error_regression() {
        let air = LogUpAir {
            num_interactions: 16,
            max_message_width: 3,
        };
        let bits = fingerprint_error(&air, &shape(128)).bits();
        assert!((bits - 102.0).abs() < 1e-9, "got {bits}");
    }

    /// More interactions and wider messages both tighten (lower) the bound.
    #[test]
    fn fingerprint_error_is_monotone() {
        let base = LogUpAir {
            num_interactions: 16,
            max_message_width: 3,
        };
        let more_interactions = LogUpAir {
            num_interactions: 256,
            ..base
        };
        let wider = LogUpAir {
            max_message_width: 31,
            ..base
        };
        let s = shape(128);

        let b0 = fingerprint_error(&base, &s).bits();
        assert!(fingerprint_error(&more_interactions, &s).bits() < b0);
        assert!(fingerprint_error(&wider, &s).bits() < b0);
    }

    #[test]
    fn security_term_absent_without_interactions() {
        let none = LogUpAir {
            num_interactions: 0,
            max_message_width: 4,
        };
        assert!(security_term(&none, &shape(128)).is_none());

        let some = LogUpAir {
            num_interactions: 4,
            max_message_width: 4,
        };
        let term = security_term(&some, &shape(128)).expect("has interactions");
        assert_eq!(term.label, LOGUP_LABEL);
        assert_eq!(term.bits, fingerprint_error(&some, &shape(128)));
    }

    /// The term flows through `proven_security_report`'s `extras` and appears
    /// in the breakdown, only ever tightening the bound.
    #[test]
    fn logup_term_composes_as_extra() {
        use crate::fri::FriRegime;
        use crate::shape::StarkAirParams;
        use crate::stark::proven_security_report;

        let regime = FriRegime {
            log_blowup: 1,
            num_queries: 100,
            log_final_poly_len: 0,
            max_log_arity: 3,
            commit_pow_bits: 0,
            query_pow_bits: 16,
        };
        let air = StarkAirParams {
            num_constraints: 1,
            max_constraint_degree: 2,
            max_combo: 2,
        };
        // Small field so the lookup term is in range of the other terms.
        let s = shape(64);
        let logup = LogUpAir {
            num_interactions: 1 << 10,
            max_message_width: 8,
        };

        let baseline = proven_security_report(&regime, &air, &s, &[]);
        let term = security_term(&logup, &s).expect("has interactions");
        let with_logup = proven_security_report(&regime, &air, &s, &[term]);

        // Extras only tighten.
        assert!(with_logup.security_bits() <= baseline.security_bits());
        // The term is present in the UDR regime's labeled breakdown.
        assert!(
            with_logup
                .udr
                .terms()
                .iter()
                .any(|t| t.label == LOGUP_LABEL)
        );
    }
}
