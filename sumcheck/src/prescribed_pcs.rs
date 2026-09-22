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
/// Except with that probability, accepted evaluations agree with one of at most
/// `2^log2_max_candidates` polynomials of a set the commitment fixes.
///
/// That set is fixed before the outer protocol samples any challenge of its own.
///
/// Callers must union-bound their own reductions over it.
///
/// The error arrives broken out by source rather than pre-summed.
///
/// An opening that stacks a reduction on a commitment charges both.
///
/// A report showing one number for the pair cannot say which of them is short.
/// # The candidate count is forwarded, not consumed
///
/// [`Self::charge_reduction`] leaves `log2_max_candidates` exactly as it found it.
///
/// A layer charges the draws it makes itself, then hands the same count on, because the
/// set is fixed by the commitment and a union bound taken over it at one layer takes
/// nothing away from the prover's freedom at the next. [`CandidateSet`] states the rule
/// and carries it in the type; this struct is one link in the stack it describes.
///
/// ```text
///     WHIR commitment        fixes the set, charges its own proximity error
///     bit ring switch        charges its reduction over the set, forwards the count
///     column batching        charges its draw over the same set, forwards the count
///     multi-STARK report     charges its AIR and lookup draws over the same set
/// ```
///
/// Terms already in [`Self::terms`] are final: each was charged by the layer that drew
/// it, and no layer above re-charges them.
#[derive(Clone, Debug)]
pub struct PrescribedOpeningSecurity {
    /// Every labelled algebraic error the opening charges, composed by a union bound.
    ///
    /// Each term is already charged for whatever it had to pay, so a caller composes
    /// these as they are rather than charging them again.
    pub terms: Vec<SecurityTerm>,
    /// Logarithm of the maximum candidate set size at commitment time.
    ///
    /// [`Self::candidates`] is the checked view of this, and the one to read before
    /// trusting a number derived from it.
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
    /// Nothing when [`Self::log2_max_candidates`] names no set — negative, infinite, or
    /// NaN. A caller that needs a number from this evidence fails closed on that.
    #[must_use]
    pub fn candidates(&self) -> Option<CandidateSet> {
        CandidateSet::from_log2(self.log2_max_candidates)
    }

    /// Charge one error the caller's own reduction draws after the commitment.
    ///
    /// A prover may pick its candidate after seeing those challenges, so the union bound
    /// over the candidate set is what this subtracts.
    ///
    /// [`SecurityTerm::over_candidates`] is the one implementation of that arithmetic;
    /// this only decides which set the term is charged over.
    ///
    /// The count is left untouched, so the layer above charges the same set over the
    /// draws it makes itself. The term pushed here is final and is never charged again.
    pub fn charge_reduction(&mut self, term: SecurityTerm) {
        // A count naming no set prices nothing, so the term is left with no bound at all
        // rather than with a number resting on it. A negative count would otherwise
        // *raise* the term, which is the one direction that must not happen silently.
        let unusable = SecurityTerm {
            bits: ErrorBits::from_log2(0.0),
            ..term
        };
        let charged = self.candidates().map_or(unusable, |candidates| {
            term.over_candidates(candidates).term()
        });
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
/// An AIR proof fixes the point during its zerocheck.
/// The opening phase then opens the columns at that fixed point.
///
/// Shape agreement uses one table spec per committed table.
/// Each table spec carries its point-local column batches.
/// The caller supplies one point per batch instead of letting the transcript pick it.
///
/// The prescribed-point verifier does not absorb the commitment.
/// The outer protocol absorbs the commitment once.
/// That absorption happens before the outer protocol samples challenges.
///
/// # Fiat-Shamir / Soundness
///
/// In the sampled-point convention the opening point is transcript-derived by
/// construction, so it cannot be influenced by the prover. In prescribed mode there is no
/// such guarantee from this trait alone: soundness rests entirely on the caller fixing the
/// point via the shared transcript (e.g. deriving it from the AIR's zerocheck challenges,
/// which are themselves bound after the commitment is absorbed) *before* calling
/// [`open_at`](PrescribedPointPcs::open_at) / `verify_at`. A prover-influenceable,
/// non-transcript-bound point breaks the soundness of the alpha-batched claim this opening
/// feeds into.
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
    /// The bound must include claim batching and every opening reduction, composed by a
    /// union bound. It inherits the implementation's documented proximity assumptions.
    /// It excludes hash and transcript collision security, which the outer protocol must
    /// supply separately. Returning the configured per-round target is insufficient.
    /// The candidate count must apply before outer challenges, even if opening-time
    /// checks later reduce the list to a single polynomial.
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
    /// Configuration and budget errors are returned before any transcript interaction
    /// or consumption of private randomness, as for [`MultilinearPcs::open`].
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
    /// Each lists the direct column values, then the repeat-last successor-view values.
    ///
    /// # Errors
    ///
    /// Returns an error if any count disagrees.
    /// Returns an error if any shape disagrees.
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
        //     inner layer   charges its own reduction  ->  100 - 4 = 96 bits
        //     outer layer   charges its own draw       ->  100 - 4 = 96 bits
        //
        // The outer layer must see the same count. If the charge consumed it, the outer
        // draw would be priced against a set the prover still has, and the report would
        // come out four bits optimistic.
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
        // Same stack, read from the top: the outer layer adds one term and leaves the
        // inner one exactly where the inner layer left it.
        //
        // Charging it twice would report 92 bits for a draw that is worth 96, which no
        // test of the outer layer alone would notice — both numbers look like bounds.
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
        // Nothing here guesses a set size: the term is charged to zero bits, and
        // `candidates` reports that the evidence is unusable so a caller fails closed.
        for count in [-1.0, f64::INFINITY, f64::NAN] {
            let mut security = evidence(count);
            assert_eq!(security.candidates(), None);
            security.charge_reduction(SecurityTerm::new("r", ErrorBits::from_log2(100.0)));
            assert_eq!(security.terms[0].bits.bits(), 0.0);
            assert_eq!(security.terms[0].label, "r");
        }
    }
}
