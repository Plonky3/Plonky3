//! Opening committed columns at caller-prescribed points.

use alloc::vec::Vec;

use p3_challenger::{CanObserve, CanSampleUniformBits, FieldChallenger, GrindingChallenger};
use p3_commit::MultilinearPcs;
use p3_field::ExtensionField;
use p3_multilinear_util::point::Point;

use crate::table::{OpeningEvals, OpeningProtocol};

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
    /// Open the committed columns at caller-prescribed points instead of sampled ones.
    ///
    /// # Arguments
    ///
    /// - Prover data returned by the commitment phase.
    /// - Table shapes and per-point column batches.
    /// - One prescribed point per batch.
    /// - Fiat-Shamir transcript with the commitment already absorbed.
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
    ) -> Self::Proof;

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
