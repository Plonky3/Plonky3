//! Verify multilinear AIR SNARKs against trace commitments.

use alloc::vec::Vec;
use core::fmt::Debug;

use p3_challenger::{CanObserve, CanSampleUniformBits, FieldChallenger, GrindingChallenger};
use p3_sumcheck::PrescribedPointPcs;
use thiserror::Error;

use crate::VerifierInstances;
use crate::config::{Commitment, MultiStarkConfig, PcsError};
use crate::folder::VerifierAir;
use crate::lookup::{LookupError, verify_lookup};
use crate::opening::TableOpening;
use crate::proof::MultiStarkProof;
use crate::security::{SecurityError, security_report};
use crate::transcript::{
    MultiStarkShape, MultiStarkTranscriptFailure, MultiStarkVerifierTranscript,
};
use crate::zerocheck::{AirZerocheck, ZerocheckError};

/// Reasons the multilinear AIR verifier rejects a proof.
#[derive(Debug, Error)]
pub enum VerificationError<E>
where
    E: Debug,
{
    /// Security evidence is missing or the requested bound is not met.
    #[error("security: {0}")]
    Security(SecurityError),
    /// The zerocheck reduction or its closing constraint check failed.
    #[error("zerocheck: {0}")]
    Zerocheck(ZerocheckError),
    /// The lookup reduction is absent, unexpected, or failed its own checks.
    #[error("lookup: {0}")]
    Lookup(LookupError),
    /// The commitment opening failed to verify.
    #[error("opening: {0:?}")]
    Opening(E),
    /// The verifying key expects a preprocessed opening, but the proof carries none.
    #[error("preprocessed opening expected but absent")]
    MissingPreprocessedOpening,
    /// The proof carries a preprocessed opening, but the verifying key expects none.
    #[error("preprocessed opening present but not expected")]
    UnexpectedPreprocessedOpening,
    /// A statement-level transcript step could not be replayed.
    #[error("transcript: {0}")]
    Transcript(MultiStarkTranscriptFailure),
}

/// Verify only when the verifier's statement meets the requested security target.
///
/// The report uses trusted AIR declarations and verifier dimensions, never proof
/// counts. Missing PCS or collision evidence fails closed before the transcript
/// is touched. The result inherits the configured PCS assumptions; see
/// [`security_report`]. Other verifier preconditions are the same as [`verify`].
pub fn verify_with_security<'a, C, A>(
    config: &C,
    instances: VerifierInstances<'a, C, A>,
    proof: &MultiStarkProof<C>,
    pow_bits: usize,
    target_bits: usize,
    challenger: &mut C::Challenger,
) -> Result<(), VerificationError<PcsError<C>>>
where
    C: MultiStarkConfig,
    C::Pcs: PrescribedPointPcs<C::Challenge, C::Challenger>,
    C::Challenger: FieldChallenger<C::Val>
        + GrindingChallenger<Witness = C::Val>
        + CanSampleUniformBits<C::Val>
        + CanObserve<Commitment<C>>,
    Commitment<C>: Clone,
    A: VerifierAir<C::Val, C::Challenge>,
{
    security_report(config, &instances)
        .and_then(|report| report.require_security(target_bits))
        .map_err(VerificationError::Security)?;
    verify(config, instances, proof, pow_bits, challenger)
}

/// Verify a complete batched multilinear AIR proof.
///
/// This entry point enforces no minimum security level. Use [`verify_with_security`]
/// when verification must meet a security target, including the AIR and lookup reductions.
///
/// The verifier replays the prover's statement-level transcript in the same order:
///
/// ```text
///     1. replay batched preprocessed commitment (if any)
///     2. replay main commitment
///     3. replay public values, one step per instance
///     4. verify the lookup reduction (if any)  -> delegated
///     5. verify zerocheck sumcheck             -> delegated, yields bound point r
///     6. open main tables at r                 -> delegated, bound to the main commitment
///     7. open preprocessed tables at r (if any)
///                                              -> delegated, bound to the preprocessed commitment
///     8. recompute the batched constraint at r and match the reduced sum
/// ```
///
/// Both sides walk one pattern, and each driver checks only its own party against it:
///
/// ```text
///     this verifier misplaces a step  ->  its own driver refuses the call
///     the prover skips an absorb      ->  the sponges diverge, so a later check fails
///     the prover skips a bracket      ->  the delegated verification rejects on its own
/// ```
///
/// Why the brackets absorb nothing: a driver's opener and closer only append a marker to its own pattern record, and neither reaches the sponge.
///
/// That holds by construction on both sides rather than by test, so a skipped delegation is caught by the callee and never here.
///
/// Each AIR instance is evaluated at the suffix of the common point matching its
/// trace height. Main openings are returned in instance order. Preprocessed
/// openings are returned in setup order, skipping AIRs with no preprocessed columns.
///
/// # Soundness
///
/// - Opened values come from the commitment proofs.
/// - The proof body never supplies those values directly.
/// - The closing check therefore uses committed trace values.
/// - The point is the bound point returned by zerocheck.
///
/// # Arguments
///
/// - `config`: proof configuration selecting the commitment schemes.
/// - `instances`: AIRs, shared verifying key, trace heights, and public inputs.
/// - `proof`: batched proof to verify.
/// - `pow_bits`: grinding difficulty per sumcheck round.
/// - `challenger`: Fiat-Shamir transcript.
///
/// # Errors
///
/// Returns an error when the sumcheck fails.
/// Returns an error when the closing check fails.
/// Returns an error when either commitment opening fails.
/// Returns an error when the proof and key disagree on whether preprocessed data is opened.
/// Returns an error when the key and the AIRs disagree on whether a preprocessed trace exists.
/// Returns an error when an instance supplies a public-value count its AIR does not declare.
/// Returns an error when the proof and the AIRs disagree on whether a lookup exists.
///
/// # Panics
///
/// Panics if the instance list is empty.
/// Panics if the verifier instances do not all use the same verifying key.
/// Panics if the preprocessed key width disagrees with the AIR's declared preprocessed width.
#[tracing::instrument(skip_all)]
pub fn verify<'a, C, A>(
    config: &C,
    instances: VerifierInstances<'a, C, A>,
    proof: &MultiStarkProof<C>,
    pow_bits: usize,
    challenger: &mut C::Challenger,
) -> Result<(), VerificationError<PcsError<C>>>
where
    C: MultiStarkConfig,
    C::Pcs: PrescribedPointPcs<C::Challenge, C::Challenger>,
    C::Challenger: FieldChallenger<C::Val>
        + GrindingChallenger<Witness = C::Val>
        + CanSampleUniformBits<C::Val>
        + CanObserve<Commitment<C>>,
    Commitment<C>: Clone,
    A: VerifierAir<C::Val, C::Challenge>,
{
    assert!(!instances.is_empty());

    let (verifying_key, instances) = instances.into_parts();
    let preprocessed_commitment = verifying_key.preprocessed.as_ref();

    // The proof's preprocessed opening must match what the key expects.
    match (preprocessed_commitment, proof.preprocessed_opening.as_ref()) {
        (Some(_), None) => return Err(VerificationError::MissingPreprocessedOpening),
        (None, Some(_)) => return Err(VerificationError::UnexpectedPreprocessedOpening),
        _ => {}
    }

    // Describe the statement before replaying it.
    //
    // Invariant: every number describing the statement is one the caller already holds.
    //
    //     the AIRs     -> widths, public-value counts, preprocessed widths
    //     this caller  -> each instance's trace arity, and the grinding difficulty
    //
    // Nothing is read out of the proof.
    let airs = instances.airs();
    let log_heights = instances.num_variables();
    let public_values = instances.public_values();
    let mut transcript = MultiStarkVerifierTranscript::<C::Challenger, C::Val>::new(
        challenger,
        MultiStarkShape::new::<C::Val, A>(&airs, &log_heights, pow_bits),
    );

    // 1. Replay the reusable batched preprocessed commitment before any challenge
    // depends on it.
    transcript
        .preprocessed_commitment(preprocessed_commitment.cloned())
        .map_err(VerificationError::Transcript)?;

    // 2. Replay the absorb the commitment scheme performs inside the prover's commit phase.
    // The verifier never calls `commit`, so it absorbs the same commitment in the same bracket.
    transcript.main_commitment(|challenger| challenger.observe(proof.commitment.clone()));

    // 3. Replay the public values, one step per instance.
    transcript
        .public_values(&public_values)
        .map_err(VerificationError::Transcript)?;

    // 4. Verify the lookup reduction, inside the delegation bracket.
    // Its claim feeds the coupled AIR sumcheck below, so a rejection stops the replay here.
    let lookup = match transcript.lookup_argument(|challenger| {
        verify_lookup::<C::Val, C::Challenge, A, _>(
            &airs,
            &log_heights,
            proof.lookup.as_ref(),
            challenger,
        )
    }) {
        Ok(lookup) => lookup,
        // Releasing the completeness check keeps this rejection the only failure in flight.
        Err(error) => {
            transcript.abort();
            return Err(VerificationError::Lookup(error));
        }
    };

    // 5. Verify the batched zerocheck sumcheck, inside the delegation bracket.
    // It yields the common bound point and the reduced sum, which both openings need.
    let zerocheck = AirZerocheck::new(&airs, pow_bits);
    let reduction = match transcript.zerocheck(|challenger| {
        zerocheck.verify_reduction_with_lookup::<C::Val, C::Challenge, _>(
            &proof.sumcheck,
            &log_heights,
            &public_values,
            lookup.as_ref(),
            challenger,
        )
    }) {
        Ok(reduction) => reduction,
        Err(error) => {
            transcript.abort();
            return Err(VerificationError::Zerocheck(error));
        }
    };

    // Invariant: a return between here and the driver's `finish` must release the driver first.
    //
    //     main opening            -> Begin, the scheme's own run, End, on any outcome
    //     main rejection          -> abort, then return, with the preprocessed step unplayed
    //     preprocessed opening    -> the same bracket, when the batch describes one
    //     finish                  -> every described step replayed
    //     preprocessed rejection  -> travels past `finish`, since no described step follows it
    //
    // A rejected batch with preprocessed columns therefore costs one opening, not two.

    // 6. Open the committed main trace tables at their suffixes of the bound point.
    // The returned values are bound to the main commitment.
    let main_evals = match transcript.main_opening(|challenger| {
        config.pcs().verify_at(
            &proof.commitment,
            &proof.opening,
            &instances.opening_protocol(),
            &instances.main_points(&reduction.point),
            challenger,
        )
    }) {
        Ok(evals) => evals,
        // Nothing below can change this verdict, so the preprocessed opening never runs.
        Err(error) => {
            transcript.abort();
            return Err(VerificationError::Opening(error));
        }
    };

    // 7. Open the preprocessed tables at their suffixes of the same bound point.
    // The owned batches are kept local so the closing check can borrow them.
    let opened_preprocessed = transcript.preprocessed_opening(|challenger| {
        let commitment = preprocessed_commitment
            .expect("a described preprocessed commitment is checked before the replay");
        let opening = proof
            .preprocessed_opening
            .as_ref()
            .expect("missing preprocessed opening rejected before verification");
        config.preprocessed_pcs().verify_at(
            commitment,
            opening,
            &instances.preprocessed_opening_protocol(),
            &instances.preprocessed_points(&reduction.point),
            challenger,
        )
    });

    // Every described step has now been replayed.
    transcript.finish();

    let preprocessed_evals = opened_preprocessed
        .transpose()
        .map_err(VerificationError::Opening)?;

    let preprocessed_next_columns = instances.preprocessed_next_columns();
    let next_columns = instances.next_columns();
    let main_openings = main_evals
        .iter()
        .zip(next_columns.iter())
        .map(|(batch, next_columns)| TableOpening::new(batch.current(), next_columns, batch.next()))
        .collect::<Vec<_>>();

    // Build one preprocessed opening view per instance, in instance order.
    //
    // An AIR with no preprocessed columns gets an empty view.
    // Opened batches and their next-column lists share the non-empty order.
    // One iterator advances through them, stepping only for non-empty AIRs.
    // A shortfall yields an empty view, which the closing check rejects instead of panicking.
    let mut preprocessed_batches = preprocessed_evals
        .iter()
        .flatten()
        .zip(preprocessed_next_columns.iter());
    let preprocessed_openings = instances
        .iter()
        .map(|instance| {
            if instance.air.preprocessed_width() == 0 {
                TableOpening::empty()
            } else {
                preprocessed_batches.next().map_or_else(
                    TableOpening::empty,
                    |(batch, next_columns)| {
                        TableOpening::new(batch.current(), next_columns, batch.next())
                    },
                )
            }
        })
        .collect::<Vec<_>>();

    // 7. Close the zerocheck.
    // Recompute the batched constraint from commitment-bound values and match the reduced sum.
    zerocheck
        .check_constraint_with_lookup(
            &reduction,
            &main_openings,
            &preprocessed_openings,
            &log_heights,
            &public_values,
            lookup.as_ref(),
        )
        .map_err(VerificationError::Zerocheck)
}
