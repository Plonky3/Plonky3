//! Verify multilinear AIR SNARKs against trace commitments.

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::fmt::Debug;

use p3_air::symbolic::AirLayout;
use p3_air::{Air, BoundaryIoError, boundary};
use p3_challenger::{CanObserve, CanSampleUniformBits, FieldChallenger, GrindingChallenger};
use p3_commit::MultilinearPcs;
use p3_field::{ExtensionField, Field};
use p3_lookup::{
    IndexedLookupError, IndexedLookups, InteractionSymbolicBuilder, Lookups, TraceWindow,
};
use p3_sumcheck::{OpeningEvals, PrescribedPointPcs};
use thiserror::Error;

use crate::VerifierInstances;
use crate::bus::{BusBindingError, BusContext};
use crate::circuit::{CircuitError, fits};
use crate::config::{Commitment, MultiStarkConfig, PcsError};
use crate::folder::VerifierAir;
use crate::indexed::IndexedPlan;
use crate::instance::{BatchRole, RunPoints, trace_suffix};
use crate::lookup::{LookupError, LookupPlan, verify_lookup};
use crate::opening::TableOpening;
use crate::proof::MultiStarkProof;
use crate::security::{SecurityError, security_report};
use crate::transcript::{
    MultiStarkShape, MultiStarkTranscriptFailure, MultiStarkVerifierTranscript,
};
use crate::zerocheck::{AirZerocheck, BusFamily, ZerocheckError};

/// Reasons the multilinear AIR verifier rejects a proof.
#[derive(Debug, Error)]
#[non_exhaustive]
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
    /// The instances do not have the preprocessed tables the verifying key was set up with.
    #[error(
        "preprocessed tables have {got:?} row variables, but the key was set up for {expected:?}"
    )]
    PreprocessedHeights {
        /// Row variables of each committed table, as recorded at setup.
        expected: Vec<usize>,
        /// Row variables of each instance with preprocessed columns, in batch order.
        got: Vec<usize>,
    },
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
    /// The product-tree bus proof is malformed or inconsistent.
    #[error("binary-bus product reduction failed: {0}")]
    BusArgument(#[from] p3_bus::BusArgumentError),
    /// The bus declarations do not describe a supported statement.
    #[error("binary-bus commitment binding failed: {0}")]
    BusBinding(#[from] BusBindingError),
    /// The proof and AIRs disagree on whether a binary-native bus exists.
    #[error("binary-bus proof present but not expected, or absent but described")]
    UnexpectedBus,
    /// A circuit table's AIR reads a successor row or assumes a bit, or an indexed read names it.
    #[error("instance {instance} is a circuit table its AIR or an indexed read misuses")]
    CircuitShape {
        /// Position of the circuit table in batch order.
        instance: usize,
    },
    /// The proof carries a different number of circuit arguments than the batch has circuit tables.
    #[error("{actual} circuit arguments for {expected} circuit tables")]
    CircuitCount {
        /// Circuit tables in the batch.
        expected: usize,
        /// Arguments the proof carries.
        actual: usize,
    },
    /// A circuit table's argument was rejected.
    #[error("circuit table {instance}: {error}")]
    Circuit {
        /// Position of the circuit table in batch order.
        instance: usize,
        /// Why its argument was rejected, boxed so the common error stays small.
        error: Box<CircuitError>,
    },
    /// The main commitment opens no weighted claim, which a circuit table's witness needs.
    #[error("the main commitment cannot open a circuit table's witness")]
    UnsupportedCircuits,
    /// An AIR names a public boundary cell or public value it does not have.
    #[error("instance {instance} boundary IO: {error}")]
    BoundaryIo {
        /// Index of the offending instance in verifier-instance order.
        instance: usize,
        /// What is wrong with the declaration.
        error: BoundaryIoError,
    },
    /// An AIR leaves bits to the commitment that the commitment does not hold as bits.
    #[error("instance {instance} assumes {assumed} bit columns, the commitment holds {committed}")]
    UncommittedBits {
        /// Index of the offending instance in verifier-instance order.
        instance: usize,
        /// Leading main columns the AIR assumes hold bits.
        assumed: usize,
        /// Leading main columns the commitment holds one bit per cell.
        committed: usize,
    },
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
///     4. verify the binary bus (if any)        -> delegated, leaves two terminal claims
///     5. verify the lookup reduction (if any)  -> delegated
///     6. verify the shared sumcheck            -> delegated, yields bound point r
///     7. verify the indexed reduction (if any) -> delegated, closes on claims from the proof
///     8. open main tables                      -> delegated, binds every terminal claim
///     9. open preprocessed tables (if any)
///                                              -> delegated, bound to the preprocessed commitment
///    10. discharge indexed claims, then close the AIR and bus families at r
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
/// Returns an error when the worst-case lookup multiplicity sum reaches the field characteristic.
/// Returns an error when a characteristic-two AIR declares an active lookup.
/// Returns an error when two declarations on one named bus disagree on payload width.
/// Returns an error when an AIR declares mutually exclusive interactions.
/// Returns an error when a lookup expression reads a periodic column.
/// Returns an error when every declared lookup tuple is empty.
///
/// The lookup statement errors are returned before the transcript is touched.
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

    // The preprocessed commitment binds the stacked tables, not which AIR owns each one.
    //
    //     key:   a -> 2^8 rows, b -> 2^7 rows
    //     claim: a -> 2^7 rows, b -> 2^8 rows    same stacked polynomial
    //
    // Each AIR would then be checked against the other's fixed columns.
    //
    // Invariant: every preprocessed height matches the one recorded at setup.
    let preprocessed_log_heights = airs
        .iter()
        .zip(&log_heights)
        .filter(|(air, _)| air.preprocessed_width() != 0)
        .map(|(_, &log_height)| log_height)
        .collect::<Vec<_>>();
    if preprocessed_log_heights != verifying_key.preprocessed_log_heights {
        return Err(VerificationError::PreprocessedHeights {
            expected: verifying_key.preprocessed_log_heights.clone(),
            got: preprocessed_log_heights,
        });
    }

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

    // A circuit table's AIR reads its port words on the current row, and nothing commits them.
    //
    // A port claim one row ahead, or a port assumed to be a bit, would bind nothing.
    //
    // Its AIR reads one column per port word, and its rows must make a batch Flock proves.
    //
    // Flock and the word claim read the witness as bits, so the scheme must commit it as bits.
    for (instance, (air, circuit)) in airs.iter().zip(instances.circuits()).enumerate() {
        if circuit.is_some_and(|argument| {
            !air.main_next_row_columns().is_empty()
                || air.boolean_columns() != 0
                || !fits(argument.circuit(), log_heights[instance], air.width())
                || config.pcs().bit_region(instance, 1) == 0
        }) {
            return Err(VerificationError::CircuitShape { instance });
        }
    }

    // An AIR may drop the booleanity of its leading columns and leave it to the commitment.
    //
    // A scheme that commits field elements there accepts any value, and no constraint refuses it.
    //
    // Invariant: every column an AIR assumes holds a bit is one the scheme commits as a bit.
    for (instance, air) in airs.iter().enumerate() {
        let assumed = air.boolean_columns();
        if assumed == 0 {
            continue;
        }
        let committed = config.pcs().bit_region(instance, air.width());
        if assumed > committed {
            return Err(VerificationError::UncommittedBits {
                instance,
                assumed,
                committed,
            });
        }
    }

    // One symbolic pass per AIR serves both lookup plans.
    //
    // Both sides derive them from the AIRs alone, so no proof value reaches them.
    let (indexed, lookups) =
        declarations::<C::Val, C::Challenge, A>(&airs).map_err(VerificationError::IndexedLookup)?;

    // Indexed lookups change the described sequence, so the plan is settled first.
    let indexed_plan = IndexedPlan::from_declared::<C::Val>(&indexed, &log_heights)
        .map_err(VerificationError::IndexedLookup)?;
    // An indexed read opens its table's columns where a circuit table commits none.
    let circuits = instances.circuits().collect::<Vec<_>>();
    if let Some(instance) = indexed_plan
        .as_ref()
        .and_then(|plan| plan.airs().find(|&air| circuits[air].is_some()))
    {
        return Err(VerificationError::CircuitShape { instance });
    }
    let circuit_tables = circuits
        .iter()
        .enumerate()
        .filter_map(|(table, circuit)| circuit.map(|circuit| (table, circuit)))
        .collect::<Vec<_>>();
    if proof.circuits.len() != circuit_tables.len() {
        return Err(VerificationError::CircuitCount {
            expected: circuit_tables.len(),
            actual: proof.circuits.len(),
        });
    }
    let bus = BusContext::<C::Val, C::Challenge>::build(&airs, &log_heights)?;

    // The lookup layout comes from the AIRs and the trace heights alone.
    // An unsupported declaration is a statement error, refused before the transcript is touched.
    let lookup_plan =
        LookupPlan::from_lookups(lookups, &log_heights).map_err(VerificationError::Lookup)?;

    let mut transcript = MultiStarkVerifierTranscript::<C::Challenger, C::Val>::new(
        challenger,
        MultiStarkShape::new::<C::Val, A>(
            &airs,
            &log_heights,
            pow_bits,
            indexed_plan.is_some(),
            bus.is_some(),
        )
        .with_circuits(instances.circuit_statements()),
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

    // 4. Reduce the bus products to their two terminal claims.
    // They stay unauthenticated until the shared sumcheck closes on committed openings.
    let bus_output = match (bus.as_ref(), proof.bus.as_ref()) {
        (Some(context), Some(bus_proof)) => {
            let output = transcript
                .bus_argument(|challenger| {
                    context
                        .plan()
                        .verify::<C::Val, C::Challenge, _>(bus_proof, challenger)
                })
                .expect("the statement describes a bus delegation");
            match output {
                Ok(output) => Some(output),
                Err(error) => {
                    transcript.abort();
                    return Err(VerificationError::BusArgument(error));
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
        verify_lookup::<C::Val, C::Challenge, _>(lookup_plan, proof.lookup.as_ref(), challenger)
    }) {
        Ok(lookup) => lookup,
        // Releasing the completeness check keeps this rejection the only failure in flight.
        Err(error) => {
            transcript.abort();
            return Err(VerificationError::Lookup(error));
        }
    };

    // 6. Verify the shared sumcheck, inside the delegation bracket.
    // It yields the common bound point and the reduced sum, which both openings need.
    let bus_family = bus
        .as_ref()
        .zip(bus_output.as_ref())
        .map(|(context, output)| BusFamily { context, output });
    let zerocheck = AirZerocheck::with_profiles(&airs, &verifying_key.air_profiles, pow_bits);
    let reduction = match transcript.zerocheck(|challenger| {
        zerocheck.verify_reduction_with_lookup_and_bus::<C::Val, C::Challenge, _>(
            &proof.sumcheck,
            &log_heights,
            &public_values,
            lookup.as_ref(),
            bus_family,
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

    // 8. Replay every circuit table's argument, binding its port values at the bound point first.
    //
    // Each leaves one claim on its witness column, which the opening below covers.
    let circuit_claims = transcript
        .circuit_argument(|challenger| {
            circuit_tables
                .iter()
                .zip(&proof.circuits)
                .map(|(&(table, circuit), circuit_proof)| {
                    circuit
                        .verify(
                            circuit_proof,
                            &trace_suffix(&reduction.point, log_heights[table]),
                            table,
                            pow_bits,
                            challenger,
                        )
                        .map_err(|error| VerificationError::Circuit {
                            instance: table,
                            error: Box::new(error),
                        })
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .transpose();
    // Each circuit table leaves two claims on its witness column.
    let weighted = match circuit_claims {
        Ok(claims) => claims
            .unwrap_or_default()
            .into_iter()
            .flatten()
            .collect::<Vec<_>>(),
        Err(error) => {
            transcript.abort();
            return Err(error);
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

    // 9. Open the committed main trace tables at every point a claim was left at.
    // The returned values are bound to the main commitment.
    let indexed_output = indexed.as_ref().map(|(_, output)| output);
    let points = RunPoints::new(&reduction.point, indexed_output);
    let main_schedule = instances.main_schedule(indexed_plan.as_ref(), |role, rows| {
        trace_suffix(points.at(role), rows)
    });
    let main_evals = match transcript.main_opening(|challenger| {
        config.pcs().verify_at_weighted(
            &proof.commitment,
            &proof.opening,
            main_schedule.protocol(),
            &main_schedule.against(),
            &weighted,
            challenger,
        )
    }) {
        Some(Ok(evals)) => evals,
        // Nothing below can change this verdict, so the preprocessed opening never runs.
        Some(Err(error)) => {
            transcript.abort();
            return Err(VerificationError::Opening(error));
        }
        None => {
            transcript.abort();
            return Err(VerificationError::UnsupportedCircuits);
        }
    };

    // 10. Open the preprocessed tables at every point a claim was left at.
    // The owned batches are kept local so the closing check can borrow them.
    let preprocessed_schedule = instances
        .preprocessed_schedule(indexed_plan.as_ref(), |role, rows| {
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
    // A circuit table's AIR reads its port values instead, which the weighted claims bound.
    //
    // A missing batch yields an empty view, which the closing check rejects instead of panicking.
    let mut circuit_ports = proof.circuits.iter();
    let main_openings = main_schedule
        .first_batch_per_table()
        .into_iter()
        .zip(&circuits)
        .zip(next_columns.iter())
        .map(|((batch, circuit), next_columns)| match circuit {
            Some(_) => circuit_ports
                .next()
                .map_or_else(TableOpening::empty, |circuit_proof| {
                    TableOpening::new(&circuit_proof.ports, next_columns, &[])
                }),
            None => batch
                .and_then(|batch| main_evals.get(batch))
                .map_or_else(TableOpening::empty, |batch| {
                    TableOpening::new(batch.current(), next_columns, batch.next())
                }),
        })
        .collect::<Vec<_>>();

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
        .flatten()
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

    // 11. Close the shared sumcheck.
    // Recompute the AIR and bus families from commitment-bound values and match the reduced sum.
    // ProductGKR terminal values are authenticated only by this check.
    zerocheck
        .check_constraint_with_lookup_and_bus(
            &reduction,
            &main_openings,
            &preprocessed_openings,
            &log_heights,
            &public_values,
            lookup.as_ref(),
            bus_family,
        )
        .map_err(VerificationError::Zerocheck)
}

/// What every AIR declares to the two lookup plans, from one symbolic pass per AIR.
///
/// A pass costs about as much as the AIR is wide, and both plans read the same builder.
///
/// # Errors
///
/// Returns an error when an indexed declaration names a column its AIR does not have.
fn declarations<F, EF, A>(
    airs: &[&A],
) -> Result<(Vec<IndexedLookups>, Vec<Lookups<F>>), IndexedLookupError>
where
    F: Field,
    EF: ExtensionField<F>,
    A: Air<InteractionSymbolicBuilder<F, EF>>,
{
    airs.iter()
        .map(|&air| {
            let layout = AirLayout::from_air(air);
            let builder = InteractionSymbolicBuilder::<F, EF>::from_air(air, layout);
            let indexed = IndexedLookups::from_builder(&builder, &layout)?;
            Ok((indexed, Lookups::from_builder(&builder)))
        })
        .collect::<Result<Vec<_>, _>>()
        .map(|declared| declared.into_iter().unzip())
}
