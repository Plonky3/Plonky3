//! Typed low-degree-test boundary.
//!
//! [`crate::stark`]'s composite orchestration is generic over the LDT through
//! this trait: any protocol — FRI, WHIR, or an out-of-tree drop-in — can
//! implement [`LowDegreeTest`] and compose with the AIR + DEEP-ALI bounds via
//! [`crate::stark::proven_security_report`]. The FRI implementation lives in
//! [`crate::fri`]; [`crate::whir`] currently exposes only the underlying WHIR
//! error terms, not yet a [`LowDegreeTest`] impl.

use alloc::vec;
use alloc::vec::Vec;

use crate::error::ErrorBits;
use crate::report::{LDT_LABEL, SecurityTerm};
use crate::shape::{InstanceShape, StarkAirParams};

/// A low-degree test whose per-regime error terms compose with the AIR +
/// DEEP-ALI bounds at the protocol call site.
///
/// The proximity regimes mirror [`crate::proximity`]: unique decoding (list
/// size 1) and list decoding at an explicit proximity parameter `m`. An
/// implementation reports only the **LDT-only** error for each regime; the
/// composite ([`crate::stark::proven_security_report`]) folds in the ALI,
/// DEEP, extra, and commitment-collision terms.
pub trait LowDegreeTest {
    /// `log2` of the LDT blowup factor (rate ρ = 2^{−log_blowup}). The
    /// composite uses it to reconstruct the list-decoding list size for the
    /// regime returned by [`Self::best_ldr`].
    fn log_blowup(&self) -> usize;

    /// LDT-only error in the unique-decoding regime.
    fn proven_error_udr(&self, air: &StarkAirParams, shape: &InstanceShape) -> ErrorBits;

    /// Best list-decoding regime for this instance: the proximity parameter
    /// `m` maximising LDT-only security together with the error attained.
    /// `None` when no valid `m` exists (e.g. the trace is too small).
    fn best_ldr(&self, air: &StarkAirParams, shape: &InstanceShape) -> Option<(usize, ErrorBits)>;

    /// Every admissible list-decoding regime: the proximity parameter `m`
    /// paired with the LDT-only error attained there.
    ///
    /// [`crate::stark::proven_security_report`] evaluates the *full* composite
    /// at each candidate and keeps the best, instead of trusting the LDT's own
    /// optimum. The two disagree whenever a term outside the LDT depends on
    /// `m`: the batched-openings term grows as `(m + 1/2)⁵`, so an `m` chosen
    /// to maximise the LDT alone can sit far from the one that maximises the
    /// composite. Left to [`Self::best_ldr`], grinding that lets the LDT
    /// tolerate a larger `m` would then *lower* the reported level, breaking
    /// the invariant that proof of work never costs security.
    ///
    /// The default returns [`Self::best_ldr`]'s single choice, which leaves
    /// any existing implementation correct; override it to expose the whole
    /// admissible range.
    fn ldr_candidates(
        &self,
        air: &StarkAirParams,
        shape: &InstanceShape,
    ) -> Vec<(usize, ErrorBits)> {
        self.best_ldr(air, shape).into_iter().collect()
    }

    /// Conjectured LDT error (random-words / heuristic regime).
    fn conjectured_error(&self, shape: &InstanceShape) -> ErrorBits;

    /// The conjectured regime's LDT terms, labeled per phase.
    ///
    /// [`crate::stark::conjectured_security_report`] takes the minimum over
    /// these alongside the ALI, DEEP, and collision terms, so an implementation
    /// that reports its phases separately keeps each one inspectable rather
    /// than collapsing them into a single number — the same reason the proven
    /// path is a [`crate::report::RegimeReport`] and not a scalar.
    ///
    /// The default returns [`Self::conjectured_error`] under
    /// [`LDT_LABEL`], which is correct for any LDT whose conjectured error is
    /// already a single composed bound. Override it when the protocol has
    /// distinct rounds worth surfacing — [`crate::fri::FriRegime`] splits the
    /// query phase from the commit-phase folding rounds.
    ///
    /// Each returned term must bound a **distinct round** of the protocol.
    /// The composite takes the minimum rather than summing because an
    /// adversary who breaks any single round breaks the LDT, so the attained
    /// security is that of the weakest round — the round-by-round accounting
    /// of [2024/1553](https://eprint.iacr.org/2024/1553) §2. Returning two
    /// bounds on the *same* round would silently discard the tighter one.
    fn conjectured_terms(&self, shape: &InstanceShape) -> Vec<SecurityTerm> {
        vec![SecurityTerm::new(LDT_LABEL, self.conjectured_error(shape))]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A low-degree test that leaves every default in place, including
    /// [`LowDegreeTest::conjectured_terms`], so the default body has a
    /// dedicated test rather than relying on the first downstream `impl` —
    /// currently [`crate::fri::FriRegime`], which overrides it — to exercise
    /// it incidentally.
    struct StubLdt;

    impl LowDegreeTest for StubLdt {
        fn log_blowup(&self) -> usize {
            1
        }

        fn proven_error_udr(&self, _air: &StarkAirParams, _shape: &InstanceShape) -> ErrorBits {
            ErrorBits::from_log2(0.0)
        }

        fn best_ldr(
            &self,
            _air: &StarkAirParams,
            _shape: &InstanceShape,
        ) -> Option<(usize, ErrorBits)> {
            None
        }

        fn conjectured_error(&self, _shape: &InstanceShape) -> ErrorBits {
            ErrorBits::from_log2(42.0)
        }
    }

    #[test]
    fn default_conjectured_terms_wraps_conjectured_error_under_ldt_label() {
        let shape = InstanceShape {
            log_trace_length: 0,
            modulus_bits: 1,
            collision_resistance: 1,
            num_batched_functions: 1,
        };

        let terms = StubLdt.conjectured_terms(&shape);

        assert_eq!(terms.len(), 1);
        assert_eq!(terms[0].label, LDT_LABEL);
        assert!((terms[0].bits.bits() - 42.0).abs() < 1e-12);
    }
}
