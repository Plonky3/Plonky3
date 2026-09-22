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
///
/// The label names the error source, which the crate charging it chooses.
///
/// A protocol composing two instances of one scheme sees that label twice.
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
    /// A draw made before one candidate is named gives a prover that many tries at it,
    /// so the union bound over the whole set is what this subtracts.
    ///
    /// This is the one implementation of that charge. Every layer that prices a draw
    /// made between a commitment and its opening goes through here.
    ///
    /// The set is not consumed — see [`CandidateSet`] for the layering rule.
    ///
    /// The result is a [`ChargedTerm`], which has no charge of its own, so a second
    /// charge of the same term is not something a caller can write.
    #[must_use]
    pub fn over_candidates(self, candidates: CandidateSet) -> ChargedTerm {
        // An error above one is no bound at all, so the charge stops at zero bits.
        //
        // Zero bits is the honest report of "no bound", not a margin hidden by the floor:
        // a union bound containing a zero-bit term composes to at most zero bits, and
        // every caller grading a report against a positive target fails closed there.
        ChargedTerm {
            term: Self {
                bits: ErrorBits::from_log2((self.bits.bits() - candidates.log2_size()).max(0.0)),
                ..self
            },
            over: candidates,
        }
    }
}

/// How many polynomials a commitment still leaves open while later challenges are drawn.
///
/// A commitment in the unique-decoding regime names one polynomial, so its set is
/// [`CandidateSet::UNIQUE`] and costs a later draw nothing. A list-decoding argument
/// leaves a whole list open until its own opening phase names a member, so every draw
/// made in between hands a prover one try per member.
///
/// # A charge does not consume the set
///
/// This is the question three copies of the charge used to answer differently, so the
/// contract states it once, here.
///
/// The set is fixed by the commitment, once, before any layer above it draws. A layer
/// that union-bounds its own draws over the set takes nothing away from the prover's
/// freedom in the layer above, which therefore faces exactly the same set. The rule is:
///
/// ```text
///     a layer charges the draws it makes itself, over the whole set
///     a layer hands the same set, unchanged, to the layer above it
///     a term that has been charged is final, and is never charged again
/// ```
///
/// The types say so rather than the prose alone. `CandidateSet` is [`Copy`] and has no
/// operation that shrinks or spends it, so forwarding is the only thing a caller can do
/// with one. [`SecurityTerm::over_candidates`] hands back a [`ChargedTerm`], which has no
/// `over_candidates` of its own, so charging the same term twice does not typecheck.
///
/// # Example
///
/// A commitment leaving sixteen candidates open, charged by two stacked layers.
///
/// ```
/// use p3_security::{CandidateSet, ErrorBits, SecurityTerm};
///
/// let candidates = CandidateSet::from_log2(4.0).unwrap();
///
/// // The inner layer charges its own reduction and forwards the set untouched.
/// let inner = SecurityTerm::new("inner", ErrorBits::from_log2(100.0)).over_candidates(candidates);
/// assert_eq!(inner.bits().bits(), 96.0);
///
/// // The outer layer charges its own draw over the same set, not over what is left of it.
/// let outer = SecurityTerm::new("outer", ErrorBits::from_log2(100.0)).over_candidates(candidates);
/// assert_eq!(outer.bits().bits(), 96.0);
/// ```
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd, Serialize)]
pub struct CandidateSet {
    /// Base-two logarithm of the set size. Finite and non-negative by construction.
    log2_size: f64,
}

impl CandidateSet {
    /// The commitment names one polynomial, so a later draw pays nothing.
    pub const UNIQUE: Self = Self { log2_size: 0.0 };

    /// A set of `2^log2_size` candidates.
    ///
    /// # Returns
    ///
    /// Nothing when the argument is not a set size: a set has at least one member and
    /// finitely many, so anything negative, infinite, or NaN names no set at all.
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

/// A [`SecurityTerm`] that has already paid for the candidate set it was drawn against.
///
/// Its only purpose is to be a different type from an uncharged term. A charged term has
/// no `over_candidates`, so the second charge that would silently halve a reported level
/// is a compile error rather than a review finding.
///
/// [`Self::term`] unwraps it for composition into a report, and that unwrap is the single
/// visible place where a term re-enters the uncharged world.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ChargedTerm {
    term: SecurityTerm,
    over: CandidateSet,
}

impl ChargedTerm {
    /// The charged term, ready to be composed into a report.
    #[must_use]
    pub const fn term(self) -> SecurityTerm {
        self.term
    }

    /// The set this term was charged over, kept so a report can say what it paid for.
    #[must_use]
    pub const fn candidates(self) -> CandidateSet {
        self.over
    }

    /// The error source, which the charge leaves unchanged.
    #[must_use]
    pub const fn label(self) -> &'static str {
        self.term.label
    }

    /// The bound after the charge, in `-log2(error)` bits.
    #[must_use]
    pub const fn bits(self) -> ErrorBits {
        self.term.bits
    }
}

impl From<ChargedTerm> for SecurityTerm {
    fn from(charged: ChargedTerm) -> Self {
        charged.term
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
        assert_eq!(charged.bits().bits(), 96.0);
        assert_eq!(charged.label(), "r");

        // The charge records what it paid for, so a report can name the set.
        assert_eq!(charged.candidates(), sixteen());

        // A draw weaker than the candidate count is worth nothing, rather than negative.
        let drowned =
            SecurityTerm::new("weak", ErrorBits::from_log2(3.0)).over_candidates(sixteen());
        assert_eq!(drowned.bits().bits(), 0.0);
    }

    #[test]
    fn one_candidate_leaves_a_draw_at_its_own_strength() {
        // No choice is no advantage, so nothing is subtracted.
        let term = SecurityTerm::new("r", ErrorBits::from_log2(100.0));
        assert_eq!(
            term.over_candidates(CandidateSet::UNIQUE).bits().bits(),
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
        //     2^-3 error, 16 tries  ->  the prover expects to succeed
        //
        // Zero bits is what that is, and a union containing it composes to zero bits,
        // so nothing downstream can read the shortfall as a passing margin.
        let drowned =
            SecurityTerm::new("weak", ErrorBits::from_log2(1.0)).over_candidates(sixteen());
        assert_eq!(drowned.bits().bits(), 0.0);

        let composed = ErrorBits::sum(&[drowned.bits(), ErrorBits::from_log2(128.0)]);
        assert!(composed.bits() <= 0.0);
    }

    #[test]
    fn a_count_that_names_no_set_is_refused_before_it_can_be_charged() {
        // A negative count would *add* bits, which is the unsafe direction, and an
        // infinite or NaN one prices nothing. None of the three is a set size.
        for count in [-1.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
            assert_eq!(CandidateSet::from_log2(count), None);
        }
    }

    #[test]
    fn a_layer_charges_its_own_draw_and_forwards_the_set_untouched() {
        // Fixture state: one commitment leaving sixteen candidates, two layers above it.
        //
        //     commitment        its own error, already final
        //     inner reduction   drawn before a candidate is named  ->  pays 4 bits
        //     outer reduction   also drawn before one is named     ->  pays 4 bits
        //
        // The outer layer charges the set the commitment fixed, not a set the inner
        // layer somehow shrank: the inner union bound took nothing away from the prover.
        let candidates = sixteen();

        let commitment = SecurityTerm::new("commitment", ErrorBits::from_log2(90.0));
        let inner =
            SecurityTerm::new("inner", ErrorBits::from_log2(100.0)).over_candidates(candidates);
        let outer =
            SecurityTerm::new("outer", ErrorBits::from_log2(100.0)).over_candidates(candidates);

        assert_eq!(inner.bits().bits(), 96.0);
        assert_eq!(outer.bits().bits(), 96.0);

        // The set the inner layer forwarded is the one the outer layer charged.
        assert_eq!(inner.candidates(), outer.candidates());

        // The commitment's own term is not a draw made before it, so it pays nothing.
        let report = [commitment, inner.term(), outer.term()];
        assert_eq!(report[0].bits.bits(), 90.0);

        // Charging the inner term a second time would halve nothing here, because there
        // is no way to write it: `inner` is a `ChargedTerm` and has no `over_candidates`.
        //
        // The only route back is `term()`, which is the one visible unwrap, so a double
        // charge is a line a reviewer can point at rather than a silent default.
        let recharged = inner.term().over_candidates(candidates);
        assert_eq!(recharged.bits().bits(), 92.0);
        assert_ne!(recharged.bits().bits(), inner.bits().bits());
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
        assert_eq!(charged.bits().bits(), 93.0);

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
    /// Conjectured (random-words) regime: correlated agreement up to
    /// list-decoding capacity, at list size 1. See
    /// [`crate::proximity::list_size_conjectured`].
    Conjectured,
    /// Legacy conjectured regime using the pre-random-words ethSTARK
    /// query bound. For FRI this omits the folding round; see
    /// [`crate::fri::legacy_conjectured_error`] and
    /// [`crate::stark::legacy_security_report`].
    Legacy,
}

/// Full soundness breakdown within a single proximity regime.
///
/// `terms` holds every contribution — ALI, DEEP, LDT, any protocol extras,
/// and the commitment-collision cap. The attained security is the minimum
/// over all terms: a collision, or any single binding error, forges the
/// proof.
///
/// This is also the top-level output of
/// [`crate::stark::conjectured_security_report`] and
/// [`crate::stark::legacy_security_report`], which each have a single regime
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
