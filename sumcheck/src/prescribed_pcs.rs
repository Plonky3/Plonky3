//! Opening committed columns at caller-prescribed points.

use alloc::vec;
use alloc::vec::Vec;

use p3_challenger::{CanObserve, CanSampleUniformBits, FieldChallenger, GrindingChallenger};
use p3_commit::MultilinearPcs;
use p3_field::ExtensionField;
use p3_multilinear_util::point::Point;
use p3_security::{CandidateSet, ErrorBits, SecurityTerm};

use crate::table::{OpeningEvals, OpeningProtocol};

/// Conditional soundness evidence for a concrete prescribed-point opening.
///
/// The terms union-bound the probability that this opening accepts a wrong evaluation.
///
/// Otherwise an accepted evaluation agrees with one polynomial the commitment left open.
///
/// That set of candidates is fixed before the outer protocol samples anything.
///
/// The error arrives broken out by source rather than pre-summed.
///
/// An opening that stacks a reduction on a commitment charges both.
///
/// One number for the pair could not say which of the two is short.
///
/// # The candidate count is forwarded, not consumed
///
/// A charge here leaves the count exactly as it found it.
///
/// So a layer charges the draws it makes itself, then hands the same count on.
///
/// The set was fixed by the commitment, and one union bound does not shrink it.
///
/// This evidence is one link in a stack that relies on that:
///
/// ```text
///     WHIR commitment        fixes the set, charges its own proximity error
///     bit ring switch        charges its reduction over the set, forwards the count
///     column batching        charges its draw over the same set, forwards the count
///     multi-STARK report     charges its AIR, lookup and bus draws over the same set
/// ```
///
/// Terms already recorded here are final, each charged by the layer that drew it.
#[derive(Clone, Debug)]
pub struct PrescribedOpeningSecurity {
    /// Every labelled algebraic error the opening charges, composed by a union bound.
    ///
    /// Each has already paid what it owed, so a caller composes them as they are.
    pub terms: Vec<SecurityTerm>,
    /// Logarithm of the maximum candidate set size at commitment time.
    ///
    /// Read the checked view below before trusting any number derived from this.
    pub log2_max_candidates: f64,
}

impl PrescribedOpeningSecurity {
    /// Evidence whose whole algebraic error sits under one label.
    #[must_use]
    pub fn single(label: &'static str, error: ErrorBits, log2_max_candidates: f64) -> Self {
        Self {
            terms: vec![SecurityTerm::new(label, error)],
            log2_max_candidates,
        }
    }

    /// The candidate set the commitment leaves open, as a checked value.
    ///
    /// # Returns
    ///
    /// Nothing when the recorded count is negative, infinite, or undefined.
    ///
    /// A caller needing a number from this evidence fails closed on that.
    #[must_use]
    pub fn candidates(&self) -> Option<CandidateSet> {
        CandidateSet::from_log2(self.log2_max_candidates)
    }

    /// Charge one error the caller's own reduction draws after the commitment.
    ///
    /// A prover may pick its candidate after seeing those challenges.
    ///
    /// So the union bound over the whole candidate set is what this subtracts.
    ///
    /// The arithmetic itself lives in one place, and this only picks the set.
    ///
    /// The count is left untouched, so the layer above charges the same set.
    ///
    /// The term recorded here is final, and the layer above adds its own beside it.
    pub fn charge_reduction(&mut self, term: SecurityTerm) {
        // A count naming no set prices nothing, so the term is left with no bound.
        //
        // A negative count would otherwise raise it, which must never happen quietly.
        let unusable = SecurityTerm {
            bits: ErrorBits::from_log2(0.0),
            ..term
        };
        let charged = self
            .candidates()
            .map_or(unusable, |candidates| term.over_candidates(candidates));
        self.terms.push(charged);
    }

    /// Union of every term, which is the whole algebraic error of the opening.
    #[must_use]
    pub fn error(&self) -> ErrorBits {
        ErrorBits::sum(&self.terms.iter().map(|term| term.bits).collect::<Vec<_>>())
    }
}

/// A multilinear commitment scheme that opens columns at caller-chosen points.
///
/// The base opening path draws each evaluation point from the transcript.
///
/// An AIR proof instead fixes the point during its zerocheck.
///
/// The opening phase then opens the columns at that fixed point.
///
/// Shape agreement uses one table spec per committed table.
///
/// Each table spec carries its point-local column batches.
///
/// The caller supplies one point per batch instead of letting the transcript pick it.
///
/// The prescribed-point verifier does not absorb the commitment.
///
/// The outer protocol absorbs it once, before sampling any challenge.
///
/// # Fiat-Shamir / Soundness
///
/// In the sampled-point convention the transcript derives the point by construction.
///
/// The prover cannot influence it, and nothing further is required of the caller.
///
/// Prescribed mode carries no such guarantee from this trait alone.
///
/// Soundness rests entirely on the caller fixing the point through the shared transcript.
///
/// Deriving it from the zerocheck challenges does that, since those follow the commitment.
///
/// The point must be fixed before [`open_at`](PrescribedPointPcs::open_at) or `verify_at` is called.
///
/// A prover-influenceable point breaks the batched claim this opening feeds into.
pub trait PrescribedPointPcs<Challenge, Challenger>: MultilinearPcs<Challenge, Challenger>
where
    Challenge: ExtensionField<Self::Val>,
    Challenger: FieldChallenger<Self::Val>
        + GrindingChallenger<Witness = Self::Val>
        + CanSampleUniformBits<Self::Val>
        + CanObserve<Self::Commitment>,
{
    /// Soundness evidence for this exact opening protocol, or `None` when unknown.
    ///
    /// The bound must union-bound claim batching and every opening reduction.
    ///
    /// It inherits the implementation's documented proximity assumptions.
    ///
    /// It excludes hash and transcript collisions, which the outer protocol supplies.
    ///
    /// Returning the configured per-round target is not enough.
    ///
    /// The candidate count must apply before any outer challenge.
    ///
    /// That holds even when opening-time checks later cut the list to one polynomial.
    ///
    /// The default makes security-checked callers fail closed for unaudited backends.
    fn prescribed_security(
        &self,
        _protocol: &OpeningProtocol,
    ) -> Option<PrescribedOpeningSecurity> {
        None
    }

    /// Open the committed columns at caller-prescribed points instead of sampled ones.
    ///
    /// # Arguments
    ///
    /// - Prover data returned by the commitment phase.
    /// - Table shapes and per-point column batches.
    /// - One prescribed point per batch.
    /// - Fiat-Shamir transcript with the commitment already absorbed.
    ///
    /// Configuration and budget errors come back before the transcript moves.
    ///
    /// No private randomness is consumed either, as for [`MultilinearPcs::open`].
    ///
    /// # Panics
    ///
    /// Panics if the number of points differs from the number of opening batches.
    fn open_at(
        &self,
        prover_data: Self::ProverData,
        protocol: &OpeningProtocol,
        points: &[Point<Challenge>],
        challenger: &mut Challenger,
    ) -> Result<Self::Proof, Self::ProverError>;

    /// Verify a prescribed-point opening and return the opened column values.
    ///
    /// # Arguments
    ///
    /// - Commitment to the columns.
    /// - Opening proof.
    /// - Table shapes and column batches.
    /// - Prescribed points in opening order.
    /// - Fiat-Shamir transcript with the commitment already absorbed.
    ///
    /// # Returns
    ///
    /// One evaluation batch per opening batch.
    ///
    /// Each lists the direct column values, then the repeat-last successor-view values.
    ///
    /// # Errors
    ///
    /// Returns an error if any count or any shape disagrees.
    ///
    /// Returns an error if the proof fails to verify.
    ///
    /// # Panics
    ///
    /// Panics if the number of points differs from the number of opening batches.
    fn verify_at(
        &self,
        commitment: &Self::Commitment,
        proof: &Self::Proof,
        protocol: &OpeningProtocol,
        points: &[Point<Challenge>],
        challenger: &mut Challenger,
    ) -> Result<Vec<OpeningEvals<Challenge>>, Self::Error>;
}

#[cfg(test)]
mod tests {
    use p3_security::{CandidateSet, ErrorBits, SecurityTerm};

    use super::PrescribedOpeningSecurity;

    /// Evidence for a commitment leaving `2^log2` candidates and charging nothing itself.
    fn evidence(log2: f64) -> PrescribedOpeningSecurity {
        PrescribedOpeningSecurity {
            terms: alloc::vec::Vec::new(),
            log2_max_candidates: log2,
        }
    }

    #[test]
    fn a_reduction_pays_the_union_bound_over_the_candidate_set() {
        // A commitment leaving sixteen candidates costs a reduction four bits.
        let mut security = evidence(4.0);
        security.charge_reduction(SecurityTerm::new("r", ErrorBits::from_log2(100.0)));
        assert_eq!(security.terms[0].bits.bits(), 96.0);
        assert_eq!(security.terms[0].label, "r");

        // A reduction weaker than the candidate count is worth nothing, rather than negative.
        security.charge_reduction(SecurityTerm::new("weak", ErrorBits::from_log2(3.0)));
        assert_eq!(security.terms[1].bits.bits(), 0.0);
    }

    #[test]
    fn unique_decoding_leaves_a_reduction_at_its_own_strength() {
        // One candidate is no choice at all, so nothing is subtracted.
        let mut security = evidence(0.0);
        security.charge_reduction(SecurityTerm::new("r", ErrorBits::from_log2(100.0)));
        assert_eq!(security.terms[0].bits.bits(), 100.0);
        assert_eq!(security.candidates(), Some(CandidateSet::UNIQUE));
    }

    #[test]
    fn a_charge_forwards_the_candidate_count_to_the_layer_above() {
        // Fixture state: a commitment leaving sixteen candidates, two layers stacked on it.
        //
        // ```text
        //     inner layer   charges its own reduction  ->  100 - 4 = 96 bits
        //     outer layer   charges its own draw       ->  100 - 4 = 96 bits
        // ```
        //
        // The outer layer must see the same count.
        //
        // A consumed count would price its draw against a set the prover still has.
        //
        // The report would then come out four bits optimistic.
        let mut inner = evidence(4.0);
        inner.charge_reduction(SecurityTerm::new("inner", ErrorBits::from_log2(100.0)));
        assert_eq!(inner.log2_max_candidates, 4.0);

        let mut outer = inner.clone();
        outer.charge_reduction(SecurityTerm::new("outer", ErrorBits::from_log2(100.0)));
        assert_eq!(outer.log2_max_candidates, 4.0);

        assert_eq!(outer.terms[0].bits.bits(), 96.0);
        assert_eq!(outer.terms[1].bits.bits(), 96.0);
    }

    #[test]
    fn the_layer_above_never_charges_a_term_the_layer_below_already_charged() {
        // Same stack, read from the top: the outer layer adds one term.
        //
        // It leaves the inner one exactly where the inner layer left it.
        //
        // Charging it twice would report 92 bits for a draw that is worth 96.
        //
        // No test of the outer layer alone would notice, since both look like bounds.
        let mut security = evidence(4.0);
        security.charge_reduction(SecurityTerm::new("inner", ErrorBits::from_log2(100.0)));
        let after_inner = security.terms.clone();

        security.charge_reduction(SecurityTerm::new("outer", ErrorBits::from_log2(100.0)));
        assert_eq!(security.terms[..1], after_inner[..]);
        assert_eq!(security.terms[0].bits.bits(), 96.0);
        assert_ne!(security.terms[0].bits.bits(), 92.0);

        // The union of the two independent draws is what the opening as a whole charges.
        assert_eq!(
            security.error(),
            ErrorBits::sum(&[ErrorBits::from_log2(96.0), ErrorBits::from_log2(96.0)])
        );
    }

    #[test]
    fn a_count_that_names_no_set_leaves_a_reduction_with_no_bound() {
        // A negative count would add bits to the term, which is the unsafe direction.
        //
        // Nothing here guesses a set size, so the term is charged to zero bits.
        //
        // The checked view reports the evidence unusable, and a caller fails closed.
        for count in [-1.0, f64::INFINITY, f64::NAN] {
            let mut security = evidence(count);
            assert_eq!(security.candidates(), None);
            security.charge_reduction(SecurityTerm::new("r", ErrorBits::from_log2(100.0)));
            assert_eq!(security.terms[0].bits.bits(), 0.0);
            assert_eq!(security.terms[0].label, "r");
        }
    }
}
