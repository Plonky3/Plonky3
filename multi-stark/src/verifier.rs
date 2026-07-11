//! Verify multilinear AIR SNARKs against trace commitments.

use alloc::vec::Vec;
use core::fmt::Debug;

use p3_air::{Air, BaseAir, SymbolicAirBuilder};
use p3_challenger::{CanObserve, CanSampleUniformBits, FieldChallenger, GrindingChallenger};
use p3_sumcheck::PrescribedPointPcs;
use thiserror::Error;

use crate::VerifierInstances;
use crate::config::{Commitment, MultiStarkConfig, PcsError};
use crate::folder::MultilinearFolder;
use crate::opening::TableOpening;
use crate::proof::MultiStarkProof;
use crate::zerocheck::{AirZerocheck, ZerocheckError};

/// Reasons the multilinear AIR verifier rejects a proof.
#[derive(Debug, Error)]
pub enum VerificationError<E>
where
    E: Debug,
{
    /// The zerocheck reduction or its closing constraint check failed.
    #[error("zerocheck: {0}")]
    Zerocheck(ZerocheckError),
    /// The commitment opening failed to verify.
    #[error("opening: {0:?}")]
    Opening(E),
    /// The verifying key expects a preprocessed opening, but the proof carries none.
    #[error("preprocessed opening expected but absent")]
    MissingPreprocessedOpening,
    /// The proof carries a preprocessed opening, but the verifying key expects none.
    #[error("preprocessed opening present but not expected")]
    UnexpectedPreprocessedOpening,
    /// The proof carries the wrong number of preprocessed opening slots.
    #[error("preprocessed opening count mismatch: expected {expected}, got {actual}")]
    PreprocessedOpeningCountMismatch {
        /// Number of verifier instances.
        expected: usize,
        /// Number of proof slots.
        actual: usize,
    },
}

/// Verify a complete batched multilinear AIR proof.
///
/// The verifier replays the prover's transcript in the same order:
///
/// ```text
///     1. absorb batched preprocessed commitment (if any)
///     2. absorb main commitment
///     3. verify zerocheck sumcheck -> common bound point r, reduced sum
///     4. open main tables at r     -> main values bound to the main commitment
///     5. open preprocessed tables at r (if any)
///                                  -> values bound to the preprocessed commitment
///     6. recompute g at r          -> match the reduced sum
/// ```
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
///
/// # Panics
///
/// Panics if the instance list is empty.
/// Panics if the verifier instances do not all use the same verifying key.
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
    A: for<'b> Air<MultilinearFolder<'b, C::Val, C::Challenge, C::Challenge>>
        + Air<SymbolicAirBuilder<C::Val, C::Challenge>>
        + BaseAir<C::Val>,
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

    // 1. Absorb the reusable batched preprocessed commitment before any challenge
    // depends on it.
    if let Some(commitment) = preprocessed_commitment {
        challenger.observe(commitment.clone());
    } else if instances
        .iter()
        .any(|instance| instance.air.preprocessed_width() != 0)
    {
        return Err(VerificationError::PreprocessedOpeningCountMismatch {
            expected: instances.len(),
            actual: 0,
        });
    }

    // 2. Absorb the main commitment, matching the prover's commit phase.
    challenger.observe(proof.commitment.clone());

    // 3. Verify the batched zerocheck sumcheck, yielding the common bound point
    // and the reduced sum.
    let airs = instances.airs();
    let log_heights = instances.num_variables();
    let public_values = instances.public_values();
    let zerocheck = AirZerocheck::new(&airs, pow_bits);
    let reduction = zerocheck
        .verify_reduction::<C::Val, C::Challenge, _>(
            &proof.sumcheck,
            &log_heights,
            &public_values,
            challenger,
        )
        .map_err(VerificationError::Zerocheck)?;

    // 4. Open the committed main trace tables at their suffixes of the bound
    // point. The returned values are bound to the main commitment.
    let main_points = instances.main_points(&reduction.point);
    let main_evals = config
        .pcs()
        .verify_at(
            &proof.commitment,
            &proof.opening,
            &instances.opening_protocol(),
            &main_points,
            challenger,
        )
        .map_err(VerificationError::Opening)?;

    // 5. Open the preprocessed tables at their suffixes of the same bound point.
    // The owned batches are kept local so the closing check can borrow them.
    let preprocessed_evals = if let Some(preprocessed_commitment) = preprocessed_commitment {
        let opening = proof
            .preprocessed_opening
            .as_ref()
            .expect("missing preprocessed opening rejected before verification");
        Some(
            config
                .preprocessed_pcs()
                .verify_at(
                    preprocessed_commitment,
                    opening,
                    &instances.preprocessed_opening_protocol(),
                    &instances.preprocessed_points(&reduction.point),
                    challenger,
                )
                .map_err(VerificationError::Opening)?,
        )
    } else {
        None
    };

    let preprocessed_next_columns = instances.preprocessed_next_columns();
    let next_columns = instances.next_columns();

    // Restore the true opened values from the committed corner-zeroed openings.
    // An AIR without boundary IO declares no cells, so its openings pass through unchanged.
    // The restored values equal what the prover folded.
    // The closing constraint recompute below therefore closes against the same value.
    let reconstructed = main_evals
        .iter()
        .enumerate()
        .map(|(i, batch)| {
            let cells = airs[i].public_boundary_io();
            crate::boundary::BoundaryIo::new(&cells).reconstruct(
                batch.current(),
                batch.next(),
                &next_columns[i],
                main_points[i].as_slice(),
                public_values[i],
            )
        })
        .collect::<Vec<_>>();
    let main_openings = reconstructed
        .iter()
        .zip(next_columns.iter())
        .map(|((current, next), next_columns)| TableOpening::new(current, next_columns, next))
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

    // 6. Close the zerocheck: recompute g from commitment-bound values and match
    // the reduced sum.
    zerocheck
        .check_constraint(
            &reduction,
            &main_openings,
            &preprocessed_openings,
            &log_heights,
            &public_values,
        )
        .map_err(VerificationError::Zerocheck)
}
