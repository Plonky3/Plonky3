//! Typed low-degree-test boundary.
//!
//! [`crate::stark`]'s composite orchestration is generic over the LDT through
//! this trait: any protocol — FRI, WHIR, or an out-of-tree drop-in — can
//! implement [`LowDegreeTest`] and compose with the AIR + DEEP-ALI bounds via
//! [`crate::stark::proven_security_report`]. The FRI implementation lives in
//! [`crate::fri`]; [`crate::whir`] currently exposes only the underlying WHIR
//! error terms, not yet a [`LowDegreeTest`] impl.

use crate::error::ErrorBits;
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

    /// Conjectured LDT error (random-words / heuristic regime).
    fn conjectured_error(&self, shape: &InstanceShape) -> ErrorBits;
}
