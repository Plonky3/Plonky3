//! Verify multilinear AIR SNARKs against trace commitments.

use alloc::vec::Vec;
use core::fmt::Debug;

use p3_air::{BoundaryIoError, boundary};
use p3_challenger::{CanObserve, CanSampleUniformBits, FieldChallenger, GrindingChallenger};
use p3_commit::MultilinearPcs;
use p3_lookup::TraceWindow;
use p3_multilinear_util::point::Point;
use p3_sumcheck::{OpeningEvals, PrescribedPointPcs};
use thiserror::Error;

use crate::VerifierInstances;
use crate::bus::transcript::{BusCompositionShape, BusCompositionVerifierTranscript};
use crate::bus::{BusBindingError, BusContext};
use crate::config::{Commitment, MultiStarkConfig, PcsError};
use crate::folder::VerifierAir;
use crate::indexed::IndexedPlan;
use crate::instance::{BatchRole, RunPoints, trace_suffix};
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
    /// The batch's indexed lookups do not describe a reduction.
    #[error("indexed lookup: {0}")]
    IndexedLookup(p3_lookup::IndexedLookupError),
    /// The indexed reduction failed its own checks.
    #[error("indexed reduction: {0}")]
    IndexedReduction(crate::logup_star::LogupStarError),
    /// A claim the indexed reduction closed on is not what the commitment opens.
    #[error("an indexed claim is not the value its trace opens to")]
    IndexedClaimsUnopened,
    /// The proof and the AIRs disagree on whether an indexed reduction exists.
    #[error("indexed reduction present but not expected, or absent but described")]
    UnexpectedIndexedReduction,
    /// Binary-native bus declarations do not define a supported statement.
    #[error("binary-bus planning failed: {0}")]
    BusPlan(#[from] p3_bus::BusPlanError),
    /// The product-tree bus proof is malformed or inconsistent.
    #[error("binary-bus product reduction failed: {0}")]
    BusArgument(#[from] p3_bus::BusArgumentError),
    /// The bus composition sumcheck is malformed or inconsistent.
    #[error("binary-bus composition sumcheck failed: {0}")]
    BusSumcheck(#[from] p3_sumcheck::generic_degree::GenericDegreeError),
    /// The bus terminal identity is not authenticated by committed openings.
    #[error("binary-bus commitment binding failed: {0}")]
    BusBinding(#[from] BusBindingError),
    /// The proof and AIRs disagree on whether a binary-native bus exists.
    #[error("binary-bus proof present but not expected, or absent but described")]
    UnexpectedBus,
    /// An AIR names a public boundary cell or public value it does not have.
    #[error("instance {instance} boundary IO: {error}")]
    BoundaryIo {
        /// Index of the offending instance in verifier-instance order.
        instance: usize,
        /// What is wrong with the declaration.
        error: BoundaryIoError,
    },
}

/// Bind the reduced bus products to the composition sumcheck the proof carries.
///
/// Every rejection releases the typed driver first.
///
/// A malformed proof is therefore rejected rather than unwound through a drop-time panic.
fn bind_bus_composition<C, Val, Challenge, E>(
    challenger: &mut C,
    output: &p3_bus::BusReductionOutput<Challenge>,
    proof: &crate::proof::BusProof<Val, Challenge>,
    shape: BusCompositionShape,
) -> Result<(Challenge, Point<Challenge>, Challenge), VerificationError<E>>
where
    Val: p3_challenger::fs::TranscriptField,
    Challenge: p3_field::ExtensionField<Val>,
    C: FieldChallenger<Val> + GrindingChallenger<Witness = Val>,
    E: Debug,
{
    let mut composition =
        BusCompositionVerifierTranscript::<_, Val, Challenge>::new(challenger, shape);
    let direction = composition.direction_challenge();

    // The driver refuses to be dropped mid-pattern, so it is released before every return.
    let expected_claim = match output.batched_terminal_claim(direction) {
        Ok(claim) => claim,
        Err(error) => {
            composition.abort();
            return Err(error.into());
        }
    };

    // Comparing first costs one field equality instead of a whole sumcheck replay.
    if proof.composition.claimed_sum != expected_claim {
        composition.abort();
        return Err(VerificationError::BusBinding(
            BusBindingError::InitialClaimMismatch,
        ));
    }

    let verified = composition.sumcheck(|challenger| {
        proof.composition.verify(
            challenger,
            shape.num_variables,
            shape.degree,
            shape.pow_bits,
        )
    });
    let (point, terminal) = match verified {
        Ok(verified) => verified,
        Err(error) => {
            composition.abort();
            return Err(VerificationError::BusSumcheck(error));
        }
    };
    composition.finish();

    Ok((direction, point, terminal))
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
///     4. verify the binary bus (if any)        -> delegated, leaves one composition point
///     5. verify the lookup reduction (if any)  -> delegated
///     6. verify zerocheck sumcheck             -> delegated, yields bound point r
///     7. verify the indexed reduction (if any) -> delegated, closes on claims from the proof
///     8. open main tables                      -> delegated, binds every terminal claim
///     9. open preprocessed tables (if any)
///                                              -> delegated, bound to the preprocessed commitment
///    10. discharge bus and indexed claims, then close the zerocheck at r
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
/// - The zerocheck closes at the bound point it returned.
/// - An indexed claim is discharged at the point the reduction chose for it.
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
/// Returns an error when an AIR names a public boundary cell it does not have.
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

    // Reject a malformed public boundary declaration before the transcript is touched.
    // The pins the folder injects read columns and public values by those numbers.
    for (instance, air) in airs.iter().enumerate() {
        boundary::validate(
            air.public_boundary_io(),
            air.width(),
            air.num_public_values(),
        )
        .map_err(|error| VerificationError::BoundaryIo { instance, error })?;
    }

    // Indexed lookups change the described sequence, so the plan is settled first.
    //
    // Both sides derive it from the AIRs alone, so no proof value reaches it.
    let indexed_plan = IndexedPlan::build::<C::Val, C::Challenge, A>(&airs, &log_heights)
        .map_err(VerificationError::IndexedLookup)?;
    let bus = BusContext::<C::Val, C::Challenge>::build(&airs, &log_heights)?;

    let mut transcript = MultiStarkVerifierTranscript::<C::Challenger, C::Val>::new(
        challenger,
        MultiStarkShape::new::<C::Val, A>(
            &airs,
            &log_heights,
            pow_bits,
            indexed_plan.is_some(),
            bus.is_some(),
        ),
    );

    // 1. Replay the reusable batched preprocessed commitment before any challenge
    // depends on it.
    transcript
        .preprocessed_commitment(preprocessed_commitment.cloned())
        .map_err(VerificationError::Transcript)?;

    // 2. Replay the binding the commitment scheme performs inside the prover's commit phase.
    //
    // The scheme owns that binding, and asking it is what keeps the two sides together.
    //
    // A scheme whose commitment rides a typed phase replays that same phase here.
    transcript.main_commitment(|challenger| {
        config
            .pcs()
            .observe_commitment(&proof.commitment, challenger);
    });

    // 3. Replay the public values, one step per instance.
    transcript
        .public_values(&public_values)
        .map_err(VerificationError::Transcript)?;

    // 4. Reduce the bus products, then bind their terminal claims by composition sumcheck.
    let bus_reduction = match (bus.as_ref(), proof.bus.as_ref()) {
        (Some(context), Some(bus_proof)) => {
            let reduction = transcript
                .bus_argument(|challenger| {
                    let output = context
                        .plan()
                        .verify::<C::Val, C::Challenge, _>(&bus_proof.product, challenger)?;
                    let shape = BusCompositionShape {
                        num_variables: context.max_num_variables(),
                        degree: context.composition_degree(),
                        pow_bits,
                    };
                    let (direction, point, terminal) =
                        bind_bus_composition::<_, C::Val, C::Challenge, PcsError<C>>(
                            challenger, &output, bus_proof, shape,
                        )?;
                    Ok((output, direction, point, terminal))
                })
                .expect("the statement describes a bus delegation");
            match reduction {
                Ok(reduction) => Some(reduction),
                Err(error) => {
                    transcript.abort();
                    return Err(error);
                }
            }
        }
        (None, None) => None,
        _ => {
            transcript.abort();
            return Err(VerificationError::UnexpectedBus);
        }
    };

    // 5. Verify the lookup reduction, inside the delegation bracket.
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

    // 6. Verify the batched zerocheck sumcheck, inside the delegation bracket.
    // It yields the common bound point and the reduced sum, which both openings need.
    let zerocheck = AirZerocheck::with_profiles(&airs, &verifying_key.air_profiles, pow_bits);
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

    // 7. Verify the indexed reduction against the point the zerocheck bound.
    //
    // The claims come from the proof, since only the opening supplies committed values.
    //
    // The closing check below is what ties them to the commitment.
    let indexed = match (indexed_plan.as_ref(), proof.indexed.as_ref()) {
        (Some(plan), Some(round)) => {
            // Proof data decides no count here.
            //
            // A list of the wrong shape is rejected rather than measured against.
            let statement =
                match plan.statement_from_claims(&reduction.point, round.reader_claims.clone()) {
                    Ok(statement) => statement,
                    // The driver refuses to be dropped mid-pattern, so it is released first.
                    Err(error) => {
                        transcript.abort();
                        return Err(VerificationError::IndexedLookup(error));
                    }
                };
            let readers = statement.readers();
            let lookups = statement.lookups(&readers);

            match transcript
                .indexed_lookup(|challenger| round.reduction.verify(&lookups, challenger))
            {
                Ok(output) => Some((statement, output)),
                Err(error) => {
                    transcript.abort();
                    return Err(VerificationError::IndexedReduction(error));
                }
            }
        }
        (None, None) => None,
        // The shape describes the bracket exactly when the AIRs declare a read.
        //
        // Either direction of disagreement lands here.
        //
        //     the proof carries a section nobody asked for
        //     the AIRs declare a read the proof does not answer
        _ => {
            transcript.abort();
            return Err(VerificationError::UnexpectedIndexedReduction);
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

    // 8. Open the committed main trace tables at every point a claim was left at.
    // The returned values are bound to the main commitment.
    let indexed_output = indexed.as_ref().map(|(_, output)| output);
    let bus_point = bus_reduction.as_ref().map(|(_, _, point, _)| point);
    let points = RunPoints::new(&reduction.point, indexed_output, bus_point);
    let main_schedule =
        instances.main_schedule(indexed_plan.as_ref(), bus.as_ref(), |role, rows| {
            trace_suffix(points.at(role), rows)
        });
    let main_evals = match transcript.main_opening(|challenger| {
        config.pcs().verify_at(
            &proof.commitment,
            &proof.opening,
            main_schedule.protocol(),
            &main_schedule.against(),
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

    // 9. Open the preprocessed tables at every point a claim was left at.
    // The owned batches are kept local so the closing check can borrow them.
    let preprocessed_schedule =
        instances.preprocessed_schedule(indexed_plan.as_ref(), bus.as_ref(), |role, rows| {
            trace_suffix(points.at(role), rows)
        });
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
            preprocessed_schedule.protocol(),
            &preprocessed_schedule.against(),
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
    // An AIR reads its own columns out of the first batch its table is opened in.
    //
    // Walking the results in order agrees with the AIR order only while each table owns one.
    let main_openings = main_schedule
        .first_batch_per_table()
        .into_iter()
        .filter_map(|batch| main_evals.get(batch))
        .zip(next_columns.iter())
        .map(|(batch, next_columns)| TableOpening::new(batch.current(), next_columns, batch.next()))
        .collect::<Vec<_>>();

    // ProductGKR terminal values remain unauthenticated until this committed opening check.
    if let Some((output, direction, point, terminal)) = &bus_reduction {
        let context = bus.as_ref().expect("a bus reduction has a public bus plan");
        let empty = &[][..];
        let bus_main = (0..airs.len())
            .map(|air| {
                main_schedule
                    .batch_answering(BatchRole::Bus { air })
                    .and_then(|batch| main_evals.get(batch))
                    .map_or(empty, OpeningEvals::current)
            })
            .collect::<Vec<_>>();
        let opened_preprocessed = preprocessed_evals.iter().flatten().collect::<Vec<_>>();
        let bus_preprocessed = (0..airs.len())
            .map(|air| {
                preprocessed_schedule
                    .batch_answering(BatchRole::Bus { air })
                    .and_then(|batch| opened_preprocessed.get(batch).copied())
                    .map_or(empty, OpeningEvals::current)
            })
            .collect::<Vec<_>>();
        let expected = context.terminal_composition(
            output,
            *direction,
            point,
            &bus_main,
            &bus_preprocessed,
            &public_values,
        )?;
        if expected != *terminal {
            return Err(VerificationError::BusBinding(
                BusBindingError::TerminalMismatch,
            ));
        }
    }

    // The reduction is a statement about claims, and two sets of them arrive unauthenticated.
    //
    // The reader claims came out of the proof, and the reduction's output claims out of the
    // reduction.
    //
    // The openings carry the committed values at every point both were taken at.
    //
    // Discharging both is what makes this a statement about the committed traces.
    if let Some((statement, output)) = &indexed {
        let plan = indexed_plan
            .as_ref()
            .expect("an indexed reduction comes from an indexed plan");

        // What each reader claims it pulled, against its own payload columns at the bound
        // point.
        //
        // Skipping this lets a prover claim values its trace never held.
        let opened = plan.statement(&reduction.point, &main_openings);
        if statement.claims() != opened.claims() {
            return Err(VerificationError::IndexedClaimsUnopened);
        }

        // What the reduction closed on, against the batches opened at its own two points.
        //
        // Skipping this lets a prover reduce against a table nobody committed.
        for (table, planned) in plan.tables().iter().enumerate() {
            let claims = &output.tables[table];

            for (reader, claimed) in claims.position_claims.iter().enumerate() {
                let role = BatchRole::Position { table, reader };
                let opened = main_schedule
                    .batch_answering(role)
                    .and_then(|batch| main_evals.get(batch))
                    .and_then(|batch| batch.current().first());
                if opened != Some(claimed) {
                    return Err(VerificationError::IndexedClaimsUnopened);
                }
            }

            // A table's columns live in whichever window its AIR committed them to.
            let role = BatchRole::TableColumns { table };
            let opened = match planned.table.window {
                TraceWindow::Main => main_schedule
                    .batch_answering(role)
                    .and_then(|batch| main_evals.get(batch))
                    .map(OpeningEvals::current),
                TraceWindow::Preprocessed => preprocessed_schedule
                    .batch_answering(role)
                    .and_then(|batch| preprocessed_evals.iter().flatten().nth(batch))
                    .map(OpeningEvals::current),
            };
            if opened != Some(claims.column_claims.as_slice()) {
                return Err(VerificationError::IndexedClaimsUnopened);
            }
        }
    }

    // Build one preprocessed opening view per instance, in instance order.
    //
    // An AIR with no preprocessed columns gets an empty view.
    // Opened batches and their next-column lists share the non-empty order.
    // One iterator advances through them, stepping only for non-empty AIRs.
    // A shortfall yields an empty view, which the closing check rejects instead of panicking.
    // As on the main side, a table's own columns are the first batch it owns.
    let preprocessed_own_batches = preprocessed_schedule.first_batch_per_table();
    let opened = preprocessed_evals.iter().flatten().collect::<Vec<_>>();
    let mut preprocessed_batches = preprocessed_own_batches
        .iter()
        .filter_map(|&batch| opened.get(batch).copied())
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

    // 10. Close the zerocheck.
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

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_bus::{
        BusArgumentError, BusChallenges, BusReductionOutput, ProductGkrOutput, ProductGkrProof,
    };
    use p3_challenger::DuplexChallenger;
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;
    use p3_sumcheck::generic_degree::GenericDegreeProof;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use super::*;
    use crate::proof::BusProof;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;
    type Perm = Poseidon2BabyBear<16>;
    type Chal = DuplexChallenger<F, Perm, 16, 8>;

    fn challenger() -> Chal {
        let mut rng = SmallRng::seed_from_u64(0xD15EA5E);
        Chal::new(Perm::new_from_rng_128(&mut rng))
    }

    fn reduction(values: Vec<EF>) -> BusReductionOutput<EF> {
        BusReductionOutput {
            challenges: BusChallenges {
                fingerprint: Vec::new(),
                offset: EF::ZERO,
            },
            product: ProductGkrOutput {
                roots: Vec::new(),
                point: Vec::new(),
                values,
            },
        }
    }

    fn bus_proof(claimed_sum: EF) -> BusProof<F, EF> {
        BusProof {
            product: p3_bus::BusProof {
                product: ProductGkrProof {
                    roots: Vec::new(),
                    layers: Vec::new(),
                },
            },
            composition: GenericDegreeProof {
                claimed_sum,
                round_polys: Vec::new(),
                pow_witnesses: Vec::new(),
            },
        }
    }

    #[test]
    fn a_noncanonical_terminal_value_count_rejects_without_unwinding() {
        // The direction challenge has been drawn, so an early return would abandon the driver.
        let error = bind_bus_composition::<_, F, EF, core::convert::Infallible>(
            &mut challenger(),
            &reduction(vec![EF::ONE]),
            &bus_proof(EF::ZERO),
            BusCompositionShape {
                num_variables: 2,
                degree: 2,
                pow_bits: 0,
            },
        )
        .expect_err("one terminal value has no push-then-pull reading");

        assert!(matches!(
            error,
            VerificationError::BusArgument(BusArgumentError::TerminalValueCount {
                expected: 2,
                actual: 1,
            })
        ));
    }

    #[test]
    fn a_tampered_initial_claim_rejects_before_the_sumcheck_runs() {
        // An empty round list would fail the sumcheck's own shape check if it ever ran.
        let error = bind_bus_composition::<_, F, EF, core::convert::Infallible>(
            &mut challenger(),
            &reduction(vec![EF::ONE, EF::ONE]),
            &bus_proof(EF::from_u8(7)),
            BusCompositionShape {
                num_variables: 2,
                degree: 2,
                pow_bits: 0,
            },
        )
        .expect_err("the batched terminal claim of two identity roots is zero");

        assert!(matches!(
            error,
            VerificationError::BusBinding(BusBindingError::InitialClaimMismatch)
        ));
    }
}
