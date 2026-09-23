//! Labeled soundness breakdown produced by the composite orchestration.
//!
//! [`SecurityReport`] is the audit-facing output of [`crate::stark::proven_security_report`].
//!
//! Every soundness contribution arrives as a named [`SecurityTerm`], one set per proximity regime.
//!
//! The binding term stays inspectable instead of collapsing into a single number.

use alloc::vec::Vec;
use core::cmp::Ordering;

use serde::Serialize;

use crate::error::ErrorBits;

/// Label for the AIR-composition (ALI) term.
pub const ALI_LABEL: &str = "air-composition";
/// Label for the DEEP-ALI out-of-domain term.
pub const DEEP_LABEL: &str = "deep-ali";
/// Label for the low-degree test, reported as one already-composed bound.
///
/// # Asymmetry with the conjectured path
///
/// The proven path reports one label here.
///
/// The conjectured path reports [`LDT_QUERY_LABEL`] and [`LDT_COMMIT_LABEL`] separately.
///
/// A consumer diffing the two sees different label sets for the same protocol phases.
///
/// That is deliberate.
///
/// The proven error is already a minimum by the time the composite sees it.
///
/// The regime search, [`crate::fri::best_ldr_m`], picks the proximity parameter that maximises the weaker phase.
///
/// Splitting the label would force that search to carry both phases through.
///
/// It would then be optimising something else.
///
/// The conjectured path runs no such search, so nothing forces its phases together.
pub const LDT_LABEL: &str = "low-degree-test";
/// Label for the low-degree test's query phase, when the phases are reported apart.
///
/// See [`crate::ldt::LowDegreeTest::conjectured_terms`].
pub const LDT_QUERY_LABEL: &str = "ldt-query-phase";
/// Label for the low-degree test's commit phase, when the phases are reported apart.
pub const LDT_COMMIT_LABEL: &str = "ldt-commit-phase";
/// Label for the batched-openings random-linear-combination term.
pub const BATCH_LABEL: &str = "batch-combination";
/// Label for the commitment-collision cap term.
pub const COLLISION_LABEL: &str = "commitment-collision";

/// A single named soundness contribution, in `−log2(error)` bits.
///
/// The label names the error source, and the crate charging it picks the name.
///
/// A protocol composing two instances of one scheme sees that label twice.
///
/// Naming the instance is what the component is for.
#[derive(Copy, Clone, Debug, PartialEq, Serialize)]
pub struct SecurityTerm {
    /// Error source this term charges.
    pub label: &'static str,
    /// Which of the composing protocol's parts charged it, when there is a choice.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub component: Option<&'static str>,
    /// The bound itself, in `−log2(error)` bits.
    pub bits: ErrorBits,
}

impl SecurityTerm {
    /// A term charged by whatever crate owns the error source.
    ///
    /// The composing protocol is the only side that can name a component, so this leaves none.
    pub const fn new(label: &'static str, bits: ErrorBits) -> Self {
        Self {
            label,
            component: None,
            bits,
        }
    }

    /// The same term, attributed to one part of the composing protocol.
    #[must_use]
    pub const fn in_component(mut self, component: &'static str) -> Self {
        self.component = Some(component);
        self
    }

    /// The same term, charged over every candidate a commitment still leaves open.
    ///
    /// A draw made before one candidate is named gives a prover one try per candidate.
    ///
    /// Subtracting the log of the set size is the union bound over those tries.
    ///
    /// This is the workspace's only implementation of that charge.
    ///
    /// The set is passed by value and comes back unspent, so the layer above charges it too.
    #[must_use]
    pub fn over_candidates(self, candidates: CandidateSet) -> Self {
        // An error above one is no bound at all, so the charge stops at zero bits.
        //
        // Zero bits reports "no bound" rather than hiding a shortfall under the floor.
        //
        // A union holding a zero-bit term composes to at most zero bits.
        //
        // Every caller grading a report against a positive target then fails closed.
        Self {
            bits: ErrorBits::from_log2((self.bits.bits() - candidates.log2_size()).max(0.0)),
            ..self
        }
    }
}

/// How many polynomials a commitment still leaves open.
///
/// The rule: a layer charges its own draws over the whole set, then passes the set on.
///
/// A charge never shrinks the set, which the commitment fixed before anything above drew.
///
/// Unique decoding leaves one candidate, which costs a later draw nothing.
///
/// A list-decoding argument leaves a list, and each earlier draw gets one try per member.
///
/// # What this type does and does not catch
///
/// It catches a count that is not a set size.
///
/// That is the one arithmetic error here that overstates a level.
///
/// A negative count would add bits to the term it is charged over.
///
/// Being copied rather than spent is what forwarding looks like in the signature.
///
/// It does not catch a draw nobody charged.
///
/// Nor a scheme reporting an open list as unique.
///
/// Both of those overstate a level, and no type can see either one.
///
/// A component supplying no evidence leaves the report with no number to give.
///
/// Past that, a missing term is caught by the tests that name each one.
///
/// An understated list is a claim about the proximity argument, checked where it lives.
///
/// # Example
///
/// Two stacked layers, over a commitment leaving sixteen candidates open.
///
/// ```
/// use p3_security::{CandidateSet, ErrorBits, SecurityTerm};
///
/// let candidates = CandidateSet::from_log2(4.0).unwrap();
///
/// // The inner layer charges its own reduction and forwards the set untouched.
/// let inner = SecurityTerm::new("inner", ErrorBits::from_log2(100.0)).over_candidates(candidates);
/// assert_eq!(inner.bits.bits(), 96.0);
///
/// // The outer layer charges its own draw over the same set, not over what is left of it.
/// let outer = SecurityTerm::new("outer", ErrorBits::from_log2(100.0)).over_candidates(candidates);
/// assert_eq!(outer.bits.bits(), 96.0);
/// ```
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd, Serialize)]
pub struct CandidateSet {
    /// Base-two logarithm of the set size.
    ///
    /// Finite and non-negative by construction.
    log2_size: f64,
}

impl CandidateSet {
    /// The commitment names one polynomial, so a later draw pays nothing.
    pub const UNIQUE: Self = Self { log2_size: 0.0 };

    /// A set of that many candidates, as a base-two logarithm.
    ///
    /// # Returns
    ///
    /// Nothing when the argument is not a set size.
    ///
    /// A set has at least one member and finitely many.
    ///
    /// So a negative, infinite, or undefined argument names no set at all.
    #[must_use]
    pub fn from_log2(log2_size: f64) -> Option<Self> {
        (log2_size.is_finite() && log2_size >= 0.0).then_some(Self { log2_size })
    }

    /// Base-two logarithm of how many candidates are left open.
    #[must_use]
    pub const fn log2_size(self) -> f64 {
        self.log2_size
    }

    /// Whether the commitment already names one polynomial.
    #[must_use]
    pub fn is_unique(self) -> bool {
        self.log2_size == 0.0
    }

    /// The set a draw faces when two independent commitments are both still open.
    ///
    /// A prover chooses one member of each, so the two choices multiply.
    #[must_use]
    pub fn product(self, other: Self) -> Self {
        Self {
            log2_size: self.log2_size + other.log2_size,
        }
    }
}

#[cfg(test)]
mod candidate_tests {
    use super::*;

    /// Sixteen candidates, the running example of the layering rule.
    fn sixteen() -> CandidateSet {
        CandidateSet::from_log2(4.0).expect("sixteen is a set size")
    }

    #[test]
    fn a_draw_before_a_candidate_is_named_pays_for_every_one_left_open() {
        // Sixteen candidates cost a draw four bits, and the label is carried through.
        let charged =
            SecurityTerm::new("r", ErrorBits::from_log2(100.0)).over_candidates(sixteen());
        assert_eq!(charged.bits.bits(), 96.0);
        assert_eq!(charged.label, "r");

        // A draw weaker than the candidate count is worth nothing, rather than negative.
        let drowned =
            SecurityTerm::new("weak", ErrorBits::from_log2(3.0)).over_candidates(sixteen());
        assert_eq!(drowned.bits.bits(), 0.0);
    }

    #[test]
    fn one_candidate_leaves_a_draw_at_its_own_strength() {
        // No choice is no advantage, so nothing is subtracted.
        let term = SecurityTerm::new("r", ErrorBits::from_log2(100.0));
        assert_eq!(
            term.over_candidates(CandidateSet::UNIQUE).bits.bits(),
            100.0
        );

        // Unique decoding is exactly the set a commitment that names one polynomial leaves.
        assert!(CandidateSet::UNIQUE.is_unique());
        assert_eq!(CandidateSet::from_log2(0.0), Some(CandidateSet::UNIQUE));
    }

    #[test]
    fn the_floor_reports_no_bound_rather_than_hiding_a_negative_margin() {
        // A draw three bits short of the candidate count has no bound left at all.
        //
        // ```text
        //     2^-3 error, 16 tries  ->  the prover expects to succeed
        // ```
        //
        // Zero bits says exactly that, and a union holding one composes to zero bits.
        //
        // So nothing downstream can read the shortfall as a passing margin.
        let drowned =
            SecurityTerm::new("weak", ErrorBits::from_log2(1.0)).over_candidates(sixteen());
        assert_eq!(drowned.bits.bits(), 0.0);

        let composed = ErrorBits::sum(&[drowned.bits, ErrorBits::from_log2(128.0)]);
        assert!(composed.bits() <= 0.0);
    }

    #[test]
    fn a_count_that_names_no_set_is_refused_before_it_can_be_charged() {
        // A negative count would add bits, which is the direction that overstates.
        //
        // An infinite or undefined one prices nothing, and neither names a set.
        for count in [-1.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
            assert_eq!(CandidateSet::from_log2(count), None);
        }
    }

    #[test]
    fn a_layer_charges_its_own_draw_and_forwards_the_set_untouched() {
        // Fixture state: one commitment leaving sixteen candidates, two layers above it.
        //
        // ```text
        //     commitment        its own error, already final
        //     inner reduction   drawn before a candidate is named  ->  pays 4 bits
        //     outer reduction   also drawn before one is named     ->  pays 4 bits
        // ```
        //
        // The outer layer charges the set the commitment fixed, not a smaller one.
        //
        // The inner union bound took nothing away from the prover.
        let candidates = sixteen();

        let commitment = SecurityTerm::new("commitment", ErrorBits::from_log2(90.0));
        let inner =
            SecurityTerm::new("inner", ErrorBits::from_log2(100.0)).over_candidates(candidates);
        let outer =
            SecurityTerm::new("outer", ErrorBits::from_log2(100.0)).over_candidates(candidates);

        assert_eq!(inner.bits.bits(), 96.0);
        assert_eq!(outer.bits.bits(), 96.0);

        // The commitment's own term is not a draw made before it, so it pays nothing.
        assert_eq!(commitment.bits.bits(), 90.0);

        // Charging the inner draw again would take the log of the same list off twice.
        //
        // That reports 92 bits for a draw worth 96, which understates the level.
        //
        // It is the harmless direction, so nothing in the type stands in its way.
        //
        // Each layer holding its own draws apart, and paying once, is what avoids it.
        assert_eq!(inner.over_candidates(candidates).bits.bits(), 92.0);
    }

    #[test]
    fn two_open_commitments_multiply_the_tries_a_single_draw_gets() {
        // A joint choice of one member from each list is a choice from the product.
        let main = CandidateSet::from_log2(4.0).unwrap();
        let preprocessed = CandidateSet::from_log2(3.0).unwrap();
        assert_eq!(main.product(preprocessed).log2_size(), 7.0);

        // Charging over the product is charging once, for both, as one union bound.
        let charged = SecurityTerm::new("r", ErrorBits::from_log2(100.0))
            .over_candidates(main.product(preprocessed));
        assert_eq!(charged.bits.bits(), 93.0);

        // Unique decoding on one side leaves the other side's set as it was.
        assert_eq!(main.product(CandidateSet::UNIQUE), main);
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
    /// Conjectured random-words regime, at list size one.
    ///
    /// Correlated agreement is assumed up to list-decoding capacity.
    ///
    /// See [`crate::proximity::list_size_conjectured`].
    Conjectured,
    /// Legacy conjectured regime, on the pre-random-words ethSTARK query bound.
    ///
    /// For FRI this omits the folding round.
    ///
    /// See [`crate::fri::legacy_conjectured_error`] and [`crate::stark::legacy_security_report`].
    Legacy,
}

/// Full soundness breakdown within a single proximity regime.
///
/// The terms hold every contribution, plus the commitment-collision cap.
///
/// The attained security is the minimum over all of them.
///
/// A collision, or any single binding error, forges the proof.
///
/// It is also the top-level output of [`crate::stark::conjectured_security_report`].
///
/// The same holds for [`crate::stark::legacy_security_report`].
///
/// Each has one regime, so neither needs a [`SecurityReport`] envelope to maximize over.
#[derive(Clone, Debug, Serialize)]
pub struct RegimeReport {
    pub regime: Regime,
    terms: Vec<SecurityTerm>,
}

impl RegimeReport {
    /// Builds a report from its labeled terms.
    ///
    /// The terms must be non-empty.
    ///
    /// Every regime carries at least [`ALI_LABEL`], [`DEEP_LABEL`], a low-degree-test term and [`COLLISION_LABEL`].
    pub(crate) fn new(regime: Regime, terms: Vec<SecurityTerm>) -> Self {
        debug_assert!(
            !terms.is_empty(),
            "a regime report must carry at least one term"
        );
        Self { regime, terms }
    }

    /// Every soundness contribution in this regime, cap included.
    pub fn terms(&self) -> &[SecurityTerm] {
        &self.terms
    }

    /// The binding term, which is the one with the fewest bits.
    ///
    /// A regime always carries at least four terms, so there is always one.
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
/// Each regime is an independent lower bound on round-by-round soundness.
///
/// So the attained security is the larger of the two.
#[derive(Clone, Debug, Serialize)]
pub struct SecurityReport {
    pub udr: RegimeReport,
    /// Absent when no valid list-decoding regime exists for the instance.
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
