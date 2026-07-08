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
/// Label for the low-degree-test term.
pub const LDT_LABEL: &str = "low-degree-test";
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
pub enum Regime {
    /// Unique-decoding regime (list size 1).
    UniqueDecoding,
    /// List-decoding regime at proximity parameter `m`.
    ListDecoding { m: usize },
}

/// Full soundness breakdown within a single proximity regime.
///
/// `terms` holds every contribution — ALI, DEEP, LDT, any protocol extras,
/// and the commitment-collision cap. The attained security is the minimum
/// over all terms: a collision, or any single binding error, forges the
/// proof.
#[derive(Clone, Debug, Serialize)]
pub struct RegimeReport {
    pub regime: Regime,
    pub terms: Vec<SecurityTerm>,
}

impl RegimeReport {
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
