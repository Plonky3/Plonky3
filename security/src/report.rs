//! Labeled soundness breakdown produced by the composite orchestration.
//!
//! [`SecurityReport`] is the public, audit-facing output of
//! [`crate::stark::proven_security_report`]. It carries every soundness
//! contribution as a named [`SecurityTerm`], per proximity regime, so the
//! binding term is inspectable rather than collapsed into a single number.

use alloc::vec::Vec;
use core::cmp::Ordering;

use serde::Serialize;

use crate::error::ErrorBits;

/// Label for the AIR-composition (ALI) term.
pub const ALI_LABEL: &str = "air-composition";
/// Label for the DEEP-ALI out-of-domain term.
pub const DEEP_LABEL: &str = "deep-ali";
/// Label for the low-degree-test term, when an implementation reports its
/// phases as one already-composed bound.
///
/// # Asymmetry with the conjectured path
///
/// The proven path reports this single label while the conjectured path
/// reports [`LDT_QUERY_LABEL`] and [`LDT_COMMIT_LABEL`] separately, so a
/// consumer diffing the two sees different label sets for the same protocol
/// phases. That is deliberate, not an oversight: the proven path's LDT error
/// is *already* a minimum by the time the composite sees it, because
/// [`crate::fri::best_ldr_m`] searches for the proximity parameter `m`
/// maximising `min(commit, query)` and returns only the winning value.
/// Splitting the label there would require the regime search to carry both
/// phases through, which changes what `best_ldr` optimises. The conjectured
/// path has no such search, so nothing forces the phases together.
pub const LDT_LABEL: &str = "low-degree-test";
/// Label for the low-degree test's query-phase term, when an implementation
/// reports its phases separately (see [`crate::ldt::LowDegreeTest::conjectured_terms`]).
pub const LDT_QUERY_LABEL: &str = "ldt-query-phase";
/// Label for the low-degree test's commit-phase (folding) term, when an
/// implementation reports its phases separately.
pub const LDT_COMMIT_LABEL: &str = "ldt-commit-phase";
/// Label for the batched-openings random-linear-combination term.
pub const BATCH_LABEL: &str = "batch-combination";
/// Label for the commitment-collision cap term.
pub const COLLISION_LABEL: &str = "commitment-collision";

/// A single named soundness contribution, in `−log2(error)` bits.
#[derive(Copy, Clone, Debug, PartialEq, Serialize)]
pub struct SecurityTerm {
    pub label: &'static str,
    pub bits: ErrorBits,
}

impl SecurityTerm {
    pub const fn new(label: &'static str, bits: ErrorBits) -> Self {
        Self { label, bits }
    }
}

/// The proximity regime a [`RegimeReport`] was evaluated in.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize)]
#[non_exhaustive]
pub enum Regime {
    /// Unique-decoding regime (list size 1).
    UniqueDecoding,
    /// List-decoding regime at proximity parameter `m`.
    ListDecoding { m: usize },
    /// Conjectured (random-words) regime: correlated agreement up to
    /// list-decoding capacity, at list size 1. See
    /// [`crate::proximity::list_size_conjectured`].
    Conjectured,
}

/// Full soundness breakdown within a single proximity regime.
///
/// `terms` holds every contribution — ALI, DEEP, LDT, any protocol extras,
/// and the commitment-collision cap. The attained security is the minimum
/// over all terms: a collision, or any single binding error, forges the
/// proof.
///
/// This is also the top-level output of
/// [`crate::stark::conjectured_security_report`], which has a single regime
/// and therefore no [`SecurityReport`] envelope to maximize over.
#[derive(Clone, Debug, Serialize)]
pub struct RegimeReport {
    pub regime: Regime,
    terms: Vec<SecurityTerm>,
}

impl RegimeReport {
    /// Builds a report from its labeled terms. `terms` must be non-empty —
    /// every regime carries at least the ALI, DEEP, LDT, and collision terms.
    pub(crate) fn new(regime: Regime, terms: Vec<SecurityTerm>) -> Self {
        debug_assert!(
            !terms.is_empty(),
            "a regime report must carry at least one term"
        );
        Self { regime, terms }
    }

    /// Every soundness contribution in this regime — ALI, DEEP, LDT, any
    /// protocol extras, and the commitment-collision cap.
    pub fn terms(&self) -> &[SecurityTerm] {
        &self.terms
    }

    /// The binding (minimum-bits) term. `terms` is always non-empty — every
    /// regime carries at least the ALI, DEEP, LDT, and collision terms.
    pub fn binding(&self) -> SecurityTerm {
        self.terms
            .iter()
            .copied()
            .min_by(|a, b| {
                a.bits
                    .bits()
                    .partial_cmp(&b.bits.bits())
                    .unwrap_or(Ordering::Equal)
            })
            .expect("a regime report always carries the ALI/DEEP/LDT/collision terms")
    }

    /// Attained security in this regime, in bits.
    pub fn security_bits(&self) -> f64 {
        self.binding().bits.bits()
    }
}

/// Proven-soundness report across both proximity regimes.
///
/// Each regime is an independent valid lower bound on round-by-round
/// soundness, so the attained security is the maximum of the two.
#[derive(Clone, Debug, Serialize)]
pub struct SecurityReport {
    pub udr: RegimeReport,
    /// `None` when no valid list-decoding regime exists for the instance.
    pub ldr: Option<RegimeReport>,
}

impl SecurityReport {
    /// Attained proven security in bits: the better of the two regimes.
    pub fn security_bits(&self) -> f64 {
        let ldr = self.ldr.as_ref().map_or(0.0, RegimeReport::security_bits);
        self.udr.security_bits().max(ldr)
    }

    /// The winning regime and its binding term.
    pub fn binding(&self) -> (Regime, SecurityTerm) {
        match &self.ldr {
            Some(ldr) if ldr.security_bits() > self.udr.security_bits() => {
                (ldr.regime, ldr.binding())
            }
            _ => (self.udr.regime, self.udr.binding()),
        }
    }
}
