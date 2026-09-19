//! Prove that AIR instances are satisfied by committed traces.

use alloc::vec::Vec;

use p3_air::{Air, BaseAir, boundary};
use p3_challenger::{CanObserve, CanSampleUniformBits, FieldChallenger, GrindingChallenger};
use p3_commit::MultilinearPcs;
use p3_field::{ExtensionField, Field};
use p3_lookup::InteractionSymbolicBuilder;
use p3_sumcheck::PrescribedPointPcs;
use p3_sumcheck::generic_degree::RoundProver;

use crate::ProverInstances;
use crate::backend::{GenericBackend, ZerocheckBackend};
use crate::bus::{BusBindingError, BusContext};
use crate::bus_composition::BusCompositionProver;
use crate::bus_transcript::BusCompositionProverTranscript;
use crate::config::{Commitment, MultiStarkConfig, PcsProverError, ProverData};
use crate::folder::ProverAir;
use crate::indexed::{IndexedPlan, IndexedWitness};
use crate::instance::{ProverParts, RunPoints, trace_suffix};
use crate::logup_star::LogupStarProof;
use crate::lookup::prove_lookup;
use crate::opening::TableOpening;
use crate::proof::{BusProof, IndexedLookupProof, MultiStarkProof};
use crate::security::{SecurityError, assess_statement};
use crate::transcript::{MultiStarkProverTranscript, MultiStarkShape};
use crate::zerocheck::AirZerocheck;

/// What a test may substitute for what the indexed reduction reads.
///
/// A prover reached through the public API cannot reduce against one table while
/// committing another.
///
/// No test written against that API can drive the verifier's discharge of the reduction's
/// own claims.
///
/// This exists for those tests, and compiles only under test.
#[cfg(test)]
#[derive(Clone, Debug, Default)]
pub(crate) struct Forgery {
    /// A table, one of its readers, and the entries that reader's rows name.
    pub(crate) positions: Option<(usize, usize, Vec<usize>)>,
    /// A table in plan order, and the entries its first column carries.
    pub(crate) columns: Option<(usize, Vec<u64>)>,
    /// A reader in plan order, and the columns its claims are read off.
    pub(crate) claims: Option<(usize, Vec<usize>)>,
}

/// A proving-time PCS budget failure or a failed statement security assessment.
#[derive(Debug, thiserror::Error)]
pub enum ProvingError<E> {
    /// The PCS rejected a commitment or an opening budget.
    #[error("PCS {phase} failed: {source:?}")]
    Pcs { phase: &'static str, source: E },
    /// The requested complete-statement security target was not met.
    #[error(transparent)]
    Security(#[from] SecurityError),
    /// Binary-native bus declarations do not define a supported statement.
    #[error("binary-bus planning failed: {0}")]
    BusPlan(#[from] p3_bus::BusPlanError),
    /// The product-tree bus reduction rejected the committed witness.
    #[error("binary-bus product reduction failed: {0}")]
    BusArgument(#[from] p3_bus::BusArgumentError),
    /// The bus composition statement could not be derived.
    #[error("binary-bus commitment binding failed: {0}")]
    BusBinding(#[from] BusBindingError),
}

/// Prove only when the complete statement's security assessment meets `target_bits`.
///
/// Missing PCS or collision evidence is an error. Assessment happens before any
/// commitment, grinding, or transcript mutation. The bound inherits the configured
/// PCS assumptions; see [`crate::security_report`]. Other prover preconditions
/// and their panics are the same as [`prove`].
pub fn prove_with_security<'a, C, A>(
    config: &C,
    instances: ProverInstances<'a, C, A>,
    pow_bits: usize,
    target_bits: usize,
    challenger: &mut C::Challenger,
) -> Result<MultiStarkProof<C>, ProvingError<PcsProverError<C>>>
where
    C: MultiStarkConfig,
    C::Pcs: PrescribedPointPcs<C::Challenge, C::Challenger>,
    C::Challenger: Clone
        + FieldChallenger<C::Val>
        + GrindingChallenger<Witness = C::Val>
        + CanSampleUniformBits<C::Val>
        + CanObserve<Commitment<C>>,
    Commitment<C>: Clone,
    ProverData<C>: Clone,
    A: ProverAir<C::Val, C::Challenge>,
    <C::Challenge as ExtensionField<C::Val>>::ExtensionPacking:
        From<C::Challenge> + From<<C::Val as Field>::Packing>,
{
    assess_statement(config, &instances.statement())?.require_security(target_bits)?;
    prove(config, instances, pow_bits, challenger)
}

/// Prove that a batch of AIR instances is satisfied by committed execution traces.
///
/// This entry point enforces no minimum security level. Use [`prove_with_security`]
/// to assess all reduction and opening terms and reject unsupported or weak parameters.
///
/// The phases share one statement-level transcript, which records every one of them:
///
/// ```text
///     1. bind batched preprocessed commitment (if any)
///     2. commit(main trace tables)  -> the scheme absorbs the main commitment
///     3. bind public values, one step per instance
///     4. binary bus (if any)        -> delegated, leaves one composition point
///     5. lookup reduction (if any)  -> delegated
///     6. zerocheck reduction        -> delegated, yields bound point r
///     7. indexed reduction (if any) -> delegated, leaves claims at two further points
///     8. open main tables           -> delegated, openings bind every terminal claim
///     9. open preprocessed tables (if any)
///                                   -> delegated, bound to the preprocessed commitment
/// ```
///
/// Each table is opened at every point a claim was left at, not at the bound point alone.
///
/// Phases 2 and 4 through 8 run inside a recorded `Begin`/`End` bracket.
/// The pattern player therefore rejects a run that skips one or reorders two.
///
/// Main trace tables are committed together in input-instance order.
///
/// A table's own columns open at the suffix of the zerocheck point matching its height.
///
/// The batches an indexed reduction adds open at the points that reduction closes on.
///
/// The preprocessed commitment lives in the proving key, committed once at setup.
/// All non-empty preprocessed traces are stacked in AIR-instance order, skipping
/// AIRs with no preprocessed columns. Each proof clones the committed data to open
/// it at this proof's point without rebuilding the preprocessed commitment.
///
/// # Arguments
///
/// - `config`: proof configuration selecting the commitment schemes.
/// - `instances`: AIRs, transposed main trace tables, shared proving key, and public inputs.
/// - `pow_bits`: grinding difficulty per sumcheck round.
/// - `challenger`: Fiat-Shamir transcript.
///
/// # Errors
///
/// Returns the PCS configuration or budget error with its proving phase. The supplied
/// challenger is published only on success and is unchanged on a returned error.
/// This transcript transaction does not roll back earlier PCS randomness or commitments.
///
/// # Panics
///
/// - The instance list must not be empty.
/// - The trace width must match the AIR width.
/// - The trace arity must meet the commitment scheme's padding floor.
/// - This keeps the committed successor view in the same frame as zerocheck.
/// - The prover instances must all use the same proving key.
/// - The proving key must carry a preprocessed commitment exactly when an AIR declares columns.
/// - Every instance must supply the public-value count its AIR declares.
/// - The preprocessed key width must match the AIR's declared preprocessed width.
/// - A preprocessed key, when present, must have the same height as the main trace.
/// - A periodic column's period must be a power of two dividing the trace height.
/// - A lookup-active trace must meet the prover's SIMD packing width.
/// - An AIR's public boundary declaration must name only cells and values it has.
pub fn prove<'a, C, A>(
    config: &C,
    instances: ProverInstances<'a, C, A>,
    pow_bits: usize,
    caller_challenger: &mut C::Challenger,
) -> Result<MultiStarkProof<C>, ProvingError<PcsProverError<C>>>
where
    C: MultiStarkConfig,
    C::Pcs: PrescribedPointPcs<C::Challenge, C::Challenger>,
    C::Challenger: Clone
        + FieldChallenger<C::Val>
        + GrindingChallenger<Witness = C::Val>
        + CanSampleUniformBits<C::Val>
        + CanObserve<Commitment<C>>,
    Commitment<C>: Clone,
    ProverData<C>: Clone,
    A: ProverAir<C::Val, C::Challenge>,
    <C::Challenge as ExtensionField<C::Val>>::ExtensionPacking:
        From<C::Challenge> + From<<C::Val as Field>::Packing>,
{
    prove_with_backend::<C, A, GenericBackend>(config, instances, pow_bits, caller_challenger)
}

/// Prove as [`prove`] does, with the zerocheck rounds computed by backend `B`.
///
/// The backend chooses how each round polynomial, fold, and opening is computed.
/// Transcript, proof, errors, and panics are those of [`prove`] for every backend.
/// [`GenericBackend`] is the backend [`prove`] uses.
///
/// # Arguments
///
/// - `config`: proof configuration selecting the commitment schemes.
/// - `instances`: AIRs, transposed main trace tables, shared proving key, and public inputs.
/// - `pow_bits`: grinding difficulty per sumcheck round.
/// - `caller_challenger`: Fiat-Shamir transcript.
#[tracing::instrument(name = "prove", skip_all)]
pub fn prove_with_backend<'a, C, A, B>(
    config: &C,
    instances: ProverInstances<'a, C, A>,
    pow_bits: usize,
    caller_challenger: &mut C::Challenger,
) -> Result<MultiStarkProof<C>, ProvingError<PcsProverError<C>>>
where
    C: MultiStarkConfig,
    C::Pcs: PrescribedPointPcs<C::Challenge, C::Challenger>,
    C::Challenger: Clone
        + FieldChallenger<C::Val>
        + GrindingChallenger<Witness = C::Val>
        + CanSampleUniformBits<C::Val>
        + CanObserve<Commitment<C>>,
    Commitment<C>: Clone,
    ProverData<C>: Clone,
    A: BaseAir<C::Val>
        + Air<InteractionSymbolicBuilder<C::Val, C::Challenge>>
        + Air<p3_bus::BusSymbolicBuilder<C::Val, C::Challenge>>,
    B: ZerocheckBackend<C::Val, C::Challenge, A>,
{
    prove_forged::<C, A, B>(config, instances, pow_bits, caller_challenger, None)
}

/// The proving flow, with what the indexed reduction reads open to substitution.
///
/// Callers reach this through the entry points above, which substitute nothing.
pub(crate) fn prove_forged<'a, C, A, B>(
    config: &C,
    instances: ProverInstances<'a, C, A>,
    pow_bits: usize,
    caller_challenger: &mut C::Challenger,
    #[cfg(test)] forgery: Option<&Forgery>,
    #[cfg(not(test))] _forgery: Option<core::convert::Infallible>,
) -> Result<MultiStarkProof<C>, ProvingError<PcsProverError<C>>>
where
    C: MultiStarkConfig,
    C::Pcs: PrescribedPointPcs<C::Challenge, C::Challenger>,
    C::Challenger: Clone
        + FieldChallenger<C::Val>
        + GrindingChallenger<Witness = C::Val>
        + CanSampleUniformBits<C::Val>
        + CanObserve<Commitment<C>>,
    Commitment<C>: Clone,
    ProverData<C>: Clone,
    A: BaseAir<C::Val>
        + Air<InteractionSymbolicBuilder<C::Val, C::Challenge>>
        + Air<p3_bus::BusSymbolicBuilder<C::Val, C::Challenge>>,
    B: ZerocheckBackend<C::Val, C::Challenge, A>,
{
    let mut candidate = caller_challenger.clone();
    let challenger = &mut candidate;
    assert!(!instances.is_empty());

    let ProverParts {
        proving_key,
        tables,
        instances,
    } = instances.into_parts();

    // Every committed table must meet the scheme's padding floor.
    //
    // A table below the floor is zero-padded before commitment.
    // Padding moves the repeated boundary row into the pad.
    // The committed successor view then reads a pad row instead of the last row.
    // That disagrees with the zerocheck's repeat-last successor convention.
    assert!(
        tables
            .iter()
            .all(|table| table.num_variables() >= config.min_num_variables()),
        "every trace arity must be at least the commitment scheme's padding floor"
    );

    // Reject a malformed public boundary declaration before anything indexes by it.
    // The pins the folder injects read columns and public values by those numbers.
    let airs = instances.airs();
    for (instance, air) in airs.iter().enumerate() {
        boundary::validate(
            air.public_boundary_io(),
            air.width(),
            air.num_public_values(),
        )
        .unwrap_or_else(|error| panic!("instance {instance} boundary IO: {error}"));
    }

    // Indexed lookups change the described sequence, so the plan is settled first.
    let indexed_plan =
        IndexedPlan::build::<C::Val, C::Challenge, A>(&airs, &instances.num_variables())
            .expect("an indexed lookup the statement cannot plan is a caller error");
    let bus = BusContext::<C::Val, C::Challenge>::build(&airs, &instances.num_variables())?;

    // IndexedWitness currently borrows dense field slices for both payload and position columns.
    // Reject a packed source before the statement transcript or commitment can mutate the caller's
    // challenger; silently decoding here would expand the complete indexed payload.
    if indexed_plan.is_some() {
        assert!(
            tables.iter().all(|table| table.packed_bits().is_none()),
            "packed Boolean source tables are unsupported for active indexed lookups"
        );
        let preprocessed_data = proving_key
            .preprocessed
            .as_ref()
            .map(|preprocessed| &preprocessed.prover_data);
        let mut next_preprocessed = 0;
        for instance in instances.iter() {
            if instance.air.preprocessed_width() != 0 {
                let data = preprocessed_data.expect(
                    "preprocessed proving key is missing for an AIR with preprocessed columns",
                );
                let table = config.committed_table(data, next_preprocessed);
                next_preprocessed += 1;
                assert!(
                    table.packed_bits().is_none(),
                    "packed Boolean preprocessed tables are unsupported for active indexed lookups"
                );
            }
        }
    }

    // Describe the statement before binding anything into it.
    //
    // Every number comes from the AIRs, from the tables this caller holds, and from `pow_bits`.
    // No proof exists yet, so none of them can come from one.
    let num_instances = instances.len();
    let public_values = instances.public_values();
    let mut transcript = MultiStarkProverTranscript::<C::Challenger, C::Val>::new(
        challenger,
        MultiStarkShape::new::<C::Val, A>(
            &airs,
            &instances.num_variables(),
            pow_bits,
            indexed_plan.is_some(),
            bus.is_some(),
        ),
    );

    // 1. Bind the reusable batched preprocessed commitment before any challenge depends on it.
    transcript.preprocessed_commitment(
        proving_key
            .preprocessed
            .as_ref()
            .map(|preprocessed| preprocessed.commitment.clone()),
    );

    // 2. Commit all main trace tables in instance order, inside the delegation bracket.
    // The scheme absorbs the commitment it produces, so the bracket records where that lands.
    let witness = config.build_witness(tables);
    let (commitment, prover_data) = transcript
        .main_commitment(|challenger| config.pcs().commit(witness, challenger))
        .inspect_err(|_| transcript.abort())
        .map_err(|source| ProvingError::Pcs {
            phase: "main commitment",
            source,
        })?;

    // Keep commitment-bound table views for zerocheck, one per instance.
    let tables = (0..num_instances)
        .map(|table_index| config.committed_table(&prover_data, table_index))
        .collect::<Vec<_>>();

    // One entry per instance, in instance order.
    // An AIR with preprocessed columns takes the next committed table in setup order.
    // An AIR without them takes `None`.
    // A missing preprocessed key is valid only when no AIR declares preprocessed columns.
    let preprocessed_data = proving_key.preprocessed.as_ref().map(|p| &p.prover_data);
    let mut next_table = 0;
    let preprocessed_tables = instances
        .iter()
        .map(|instance| {
            (instance.air.preprocessed_width() != 0).then(|| {
                let data = preprocessed_data.expect(
                    "preprocessed proving key is missing for an AIR with preprocessed columns",
                );
                let table = config.committed_table(data, next_table);
                next_table += 1;
                table
            })
        })
        .collect::<Vec<_>>();

    // 3. Bind the public values, one step per instance.
    // They belong to the whole statement, so they land before either phase samples anything.
    transcript.public_values(&public_values);

    // 4. Reduce each planned bus product and bind its terminal claims by composition sumcheck.
    let bus_round = transcript
        .bus_argument(|challenger| {
            let context = bus
                .as_ref()
                .expect("the transcript describes a bus argument");
            let (product, output) = context.plan().prove::<C::Val, C::Challenge, _>(
                |challenges| {
                    context.materialize(&tables, &preprocessed_tables, &public_values, challenges)
                },
                challenger,
            )?;
            let degree = BusCompositionProver::degree(context);
            let num_variables = context.max_num_variables();
            let mut composition_transcript = BusCompositionProverTranscript::<
                _,
                C::Val,
                C::Challenge,
            >::new(
                challenger, num_variables, degree, pow_bits
            );
            let direction = composition_transcript.direction_challenge();
            let claimed_sum = context.composition_claim(&output, direction)?;
            let mut prover = BusCompositionProver::new(
                context,
                &output,
                &tables,
                &preprocessed_tables,
                &public_values,
                direction,
            );
            let (composition, point) = composition_transcript.sumcheck(|challenger| {
                prover.prove::<C::Val, _>(challenger, num_variables, degree, pow_bits, claimed_sum)
            });
            composition_transcript.finish();
            Ok::<_, ProvingError<PcsProverError<C>>>((
                BusProof {
                    product,
                    composition,
                },
                output,
                point,
            ))
        })
        .transpose();
    let (bus_proof, bus_point) = match bus_round {
        Ok(Some((proof, _output, point))) => (Some(proof), Some(point)),
        Ok(None) => (None, None),
        Err(error) => {
            transcript.abort();
            return Err(error);
        }
    };

    // 5. Materialize the lookup fractions and reduce them, inside the delegation bracket.
    // The resulting claim feeds the coupled AIR sumcheck below.
    let (lookup_proof, lookup_data) = transcript.lookup_argument(|challenger| {
        prove_lookup::<C::Val, C::Challenge, A, _>(
            &airs,
            &tables,
            &preprocessed_tables,
            &public_values,
            challenger,
        )
    });

    // 6. Reduce all AIR constraints to one batched sumcheck and one bound point.
    // The committed prover opens columns through the commitment schemes below, so
    // the zerocheck's own opened values are not used as the final proof openings.
    let zerocheck = AirZerocheck::with_profiles(&airs, &proving_key.air_profiles, pow_bits);
    let (zerocheck_proof, point) = transcript.zerocheck(|challenger| {
        zerocheck.prove_with_lookup::<C::Val, C::Challenge, B, _>(
            &preprocessed_tables,
            &tables,
            &public_values,
            lookup_data,
            config.sliced_rounds(),
            challenger,
        )
    });
    // 7. Reduce every indexed lookup against the point the zerocheck bound.
    //
    // The reduction needs what each reader pulled there.
    //
    // The zerocheck has just produced exactly those values.
    //
    // It leaves claims at two further points, which the opening below covers.
    let indexed_round = indexed_plan.as_ref().map(|plan| {
        let next_columns = instances.next_columns();
        let openings = zerocheck_proof
            .local
            .iter()
            .zip(&zerocheck_proof.next)
            .zip(&next_columns)
            .map(|((local, next), next_columns)| TableOpening::new(local, next_columns, next))
            .collect::<Vec<_>>();
        let statement = plan.statement(&point, &openings);

        // Under test the claims may be read off columns the reader never declared, so a
        // proof can carry values its payload column does not hold.
        #[cfg(test)]
        let statement = forgery.and_then(|forgery| forgery.claims.as_ref()).map_or(
            statement,
            |&(forged, ref substitute)| {
                let claims = plan
                    .tables()
                    .iter()
                    .flat_map(|table| &table.readers)
                    .enumerate()
                    .map(|(reader, placement)| {
                        let opened = openings[placement.air].local;
                        let columns = if reader == forged {
                            substitute
                        } else {
                            &placement.payload
                        };
                        columns.iter().map(|&column| opened[column]).collect()
                    })
                    .collect();
                plan.statement_from_claims(&point, claims)
                    .expect("a substituted claim list still describes this plan's readers")
            },
        );

        let readers = statement.readers();
        let lookups = statement.lookups(&readers);
        let witness = IndexedWitness::build(plan, &tables, &preprocessed_tables);

        // Under test a substitution may stand in for what the commitment holds, so the
        // verifier's discharge of these claims can be driven.
        #[cfg(test)]
        let witness = match forgery {
            Some(forgery) => witness.forge(forgery.positions.clone(), forgery.columns.clone()),
            None => witness,
        };

        let reader_views = witness.readers();
        let table_views = witness.tables(&reader_views);

        let (reduction, output) = transcript.indexed_lookup(|challenger| {
            // A forging prover skips its own statement check, so a test can reach the
            // verifier with a proof an honest prover would refuse to build.
            #[cfg(test)]
            if forgery.is_some() {
                return LogupStarProof::prove_unchecked(&lookups, &table_views, challenger);
            }
            LogupStarProof::prove(&lookups, &table_views, challenger)
        });
        (
            IndexedLookupProof {
                reader_claims: statement.claims().to_vec(),
                reduction,
            },
            output,
        )
    });
    let (indexed_round, indexed_output) = match indexed_round {
        Some((round, output)) => (Some(round), Some(output)),
        None => (None, None),
    };

    let sumcheck = zerocheck_proof.sumcheck;

    drop(tables);
    drop(preprocessed_tables);

    // 8. Open each main trace table at every point a claim was left at.
    let points = RunPoints::new(&point, indexed_output.as_ref(), bus_point.as_ref());
    let opening = transcript.main_opening(|challenger| {
        let schedule =
            instances.main_schedule(indexed_plan.as_ref(), bus.as_ref(), |role, rows| {
                trace_suffix(points.at(role), rows)
            });
        config.pcs().open_at(
            prover_data,
            schedule.protocol(),
            &schedule.against(),
            challenger,
        )
    });

    let opening = opening
        .inspect_err(|_| transcript.abort())
        .map_err(|source| ProvingError::Pcs {
            phase: "main opening",
            source,
        })?;

    // 9. Open each non-empty preprocessed table at every point a claim was left at.
    // The setup commitment data is reused rather than rebuilt.
    let preprocessed_opening = transcript.preprocessed_opening(|challenger| {
        let preprocessed = proving_key
            .preprocessed
            .as_ref()
            .expect("preprocessed proving key is missing for an AIR with preprocessed columns");
        let schedule =
            instances.preprocessed_schedule(indexed_plan.as_ref(), bus.as_ref(), |role, rows| {
                trace_suffix(points.at(role), rows)
            });
        config.preprocessed_pcs().open_at(
            preprocessed.prover_data.clone(),
            schedule.protocol(),
            &schedule.against(),
            challenger,
        )
    });

    let preprocessed_opening = preprocessed_opening
        .transpose()
        .inspect_err(|_| transcript.abort())
        .map_err(|source| ProvingError::Pcs {
            phase: "preprocessed opening",
            source,
        })?;

    // Every described step has now been played.
    transcript.finish();
    *caller_challenger = candidate;

    Ok(MultiStarkProof {
        commitment,
        lookup: lookup_proof,
        indexed: indexed_round,
        bus: bus_proof,
        sumcheck,
        opening,
        preprocessed_opening,
    })
}

#[cfg(test)]
mod tests {
    extern crate std;

    use alloc::string::String;
    use alloc::vec;
    use core::cell::Cell;
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::sync::OnceLock;

    use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_bus::{BusActivation, BusDirection, BusInteractionBuilder};
    use p3_challenger::{CanSample, DuplexChallenger};
    use p3_dft::Radix2DFTSmallBatch;
    use p3_field::extension::BinomialExtensionField;
    use p3_field::{PackedValue, PrimeCharacteristicRing};
    use p3_lookup::{IndexedLookupBuilder, TraceWindow};
    use p3_matrix::dense::RowMajorMatrix;
    use p3_merkle_tree::MerkleTreeMmcs;
    use p3_sumcheck::layout::{Layout, PrefixProver, Table, Witness};
    use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
    use p3_util::{log2_ceil_usize, log2_strict_usize};
    use p3_whir::{FoldingFactor, ProtocolParameters, SecurityAssumption, WhirConfig, WhirProver};
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use super::*;
    use crate::config::PcsError;
    use crate::verifier::{VerificationError, verify};
    use crate::{ProverInstance, ProverInstances, VerifierInstance, VerifierInstances, setup};

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;
    type Perm = Poseidon2BabyBear<16>;
    type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
    type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
    type MyChallenger = DuplexChallenger<F, Perm, 16, 8>;
    type PackedF = <F as Field>::Packing;
    type MyMmcs = MerkleTreeMmcs<PackedF, PackedF, MyHash, MyCompress, 2, 8>;
    type MyDft = Radix2DFTSmallBatch<F>;
    type L = PrefixProver<F, EF>;
    type TestPcs = WhirProver<EF, F, MyDft, MyMmcs, MyChallenger, L>;

    /// First-round folding factor, and the per-table padding floor.
    const FOLDING: usize = 2;

    struct TestConfig {
        /// Scheme sized for the stacked main traces.
        pcs: TestPcs,
        /// Scheme sized for the stacked preprocessed traces.
        preprocessed_pcs: TestPcs,
        /// Test-only hook for exposing a packed retained table to indexed proving.
        committed_table_override: Option<&'static Table<F>>,
        /// Number of main PCS accesses, used to prove early rejection.
        main_pcs_uses: Cell<usize>,
    }

    impl MultiStarkConfig for TestConfig {
        type Val = F;
        type Challenge = EF;
        type Challenger = MyChallenger;
        type Pcs = TestPcs;

        fn pcs(&self) -> &TestPcs {
            self.main_pcs_uses.set(self.main_pcs_uses.get() + 1);
            &self.pcs
        }

        fn collision_resistance_bits(&self) -> Option<usize> {
            None
        }

        fn preprocessed_pcs(&self) -> &TestPcs {
            &self.preprocessed_pcs
        }

        fn min_num_variables(&self) -> usize {
            FOLDING
        }

        fn build_witness(&self, tables: Vec<Table<F>>) -> Witness<F> {
            L::new_witness(tables, FOLDING)
        }

        fn committed_table<'a>(
            &self,
            prover_data: &'a p3_whir::WhirProverData<F, EF, MyMmcs, L>,
            table_index: usize,
        ) -> &'a Table<F> {
            self.committed_table_override
                .map_or_else(|| prover_data.table(table_index), |table| table)
        }
    }

    fn perm() -> Perm {
        let mut rng = SmallRng::seed_from_u64(0xD15EA5E);
        Perm::new_from_rng_128(&mut rng)
    }

    fn challenger() -> MyChallenger {
        MyChallenger::new(perm())
    }

    fn pcs(stacked_num_variables: usize) -> TestPcs {
        let folding_factor = FoldingFactor::Constant(FOLDING);
        let schedule = folding_factor
            .compute_folding_schedule(stacked_num_variables)
            .expect("valid folding schedule");
        let num_rounds = schedule.len().saturating_sub(1);
        let mut rates = Vec::with_capacity(num_rounds);
        let mut rate = 1;
        for &folding in schedule.iter().take(num_rounds) {
            rate += folding - 1;
            rates.push(rate);
        }

        let params = ProtocolParameters {
            security_level: 32,
            pow_bits: 0,
            round_log_inv_rates: rates,
            folding_factor,
            soundness_type: SecurityAssumption::CapacityBound,
            starting_log_inv_rate: 1,
        };
        let whir = WhirConfig::new(stacked_num_variables, params).unwrap();
        TestPcs::new(
            whir,
            MyDft::default(),
            MyMmcs::new(MyHash::new(perm()), MyCompress::new(perm()), 0),
        )
    }

    /// One scheme per commitment, each sized for the values stacked into it.
    fn config(main: usize, preprocessed: usize) -> TestConfig {
        TestConfig {
            pcs: pcs(main),
            preprocessed_pcs: pcs(preprocessed),
            committed_table_override: None,
            main_pcs_uses: Cell::new(0),
        }
    }

    /// AIR with a nonlinear payload and a nonconstant Boolean bus activation.
    #[derive(Clone, Copy)]
    struct ConditionalBusAir {
        /// Multiset side receiving this table's selected rows.
        direction: BusDirection,
        /// Whether the second column selects active rows.
        conditional: bool,
    }

    impl BaseAir<F> for ConditionalBusAir {
        fn width(&self) -> usize {
            2
        }
    }

    impl<AB> Air<AB> for ConditionalBusAir
    where
        AB: BusInteractionBuilder<F = F>,
    {
        fn eval(&self, builder: &mut AB) {
            let value: AB::Expr = builder.main().current_slice()[0].into();
            let selector: AB::Expr = builder.main().current_slice()[1].into();
            let activation = if self.conditional {
                BusActivation::Boolean(selector)
            } else {
                BusActivation::Always
            };
            builder.push_bus_interaction(
                "conditional-square",
                self.direction,
                [value.clone() * value],
                activation,
            );
        }
    }

    /// AIR whose bus payload comes from either a fixed or committed column.
    struct PreprocessedBusAir {
        /// Multiset side receiving this table's rows.
        direction: BusDirection,
        /// Fixed values supplied through the verifying key.
        fixed: Option<Vec<F>>,
    }

    impl BaseAir<F> for PreprocessedBusAir {
        fn width(&self) -> usize {
            1
        }

        fn preprocessed_width(&self) -> usize {
            usize::from(self.fixed.is_some())
        }

        fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
            // Setup commits the fixed payload independently of the prover trace.
            self.fixed
                .as_ref()
                .map(|values| RowMajorMatrix::new(values.clone(), 1))
        }
    }

    impl<AB> Air<AB> for PreprocessedBusAir
    where
        AB: BusInteractionBuilder<F = F>,
    {
        fn eval(&self, builder: &mut AB) {
            let main = builder.main().current_slice()[0];

            // Both sides expose the same one-coordinate tuple through different commitments.
            let value: AB::Expr = if self.fixed.is_some() {
                // The otherwise unused main column keeps an ordinary zerocheck constraint.
                builder.assert_zero(main);
                builder.preprocessed().current_slice()[0].into()
            } else {
                // Pin one committed cell so this AIR also contributes an ordinary constraint.
                builder.when_first_row().assert_one(main);
                main.into()
            };
            builder.push_bus_interaction(
                "preprocessed-payload",
                self.direction,
                [value],
                BusActivation::Always,
            );
        }
    }

    /// Prove one balanced conditional bus and return everything mutation tests reuse.
    fn conditional_bus_fixture() -> (
        TestConfig,
        ConditionalBusAir,
        ConditionalBusAir,
        crate::ProvingKey<TestConfig>,
        crate::VerifyingKey<TestConfig>,
        MultiStarkProof<TestConfig>,
    ) {
        let push = ConditionalBusAir {
            direction: BusDirection::Push,
            conditional: true,
        };
        let pull = ConditionalBusAir {
            direction: BusDirection::Pull,
            conditional: true,
        };
        let config = config(4, FOLDING);
        let (pk, vk) = setup(&config, &[&push, &pull], &mut challenger()).unwrap();
        let columns = vec![
            F::from_u64(1),
            F::from_u64(2),
            F::from_u64(3),
            F::from_u64(4),
            F::ZERO,
            F::ONE,
            F::ONE,
            F::ZERO,
        ];
        let table = || Table::new(RowMajorMatrix::new(columns.clone(), 4));
        let proof = prove(
            &config,
            ProverInstances::new(vec![
                ProverInstance::new(&push, table(), &pk, &[]),
                ProverInstance::new(&pull, table(), &pk, &[]),
            ]),
            0,
            &mut challenger(),
        )
        .unwrap();
        (config, push, pull, pk, vk, proof)
    }

    /// One AIR provides a named table, one reads it.
    enum Squares {
        /// Provides the named table from its main trace.
        Table(&'static str),
        /// Provides the named table from a preprocessed trace holding these entries.
        Fixed(&'static str, Vec<u64>),
        /// Names an entry of the table per row, and carries the value pulled.
        Reader(&'static str),
    }

    impl BaseAir<F> for Squares {
        fn width(&self) -> usize {
            match self {
                Self::Table(_) | Self::Fixed(..) => 1,
                Self::Reader(_) => 2,
            }
        }

        fn preprocessed_width(&self) -> usize {
            match self {
                Self::Fixed(..) => 1,
                _ => 0,
            }
        }

        fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
            match self {
                Self::Fixed(_, entries) => Some(RowMajorMatrix::new(
                    entries.iter().copied().map(F::from_u64).collect(),
                    1,
                )),
                _ => None,
            }
        }

        fn main_next_row_columns(&self) -> Vec<usize> {
            Vec::new()
        }
    }

    impl<AB> Air<AB> for Squares
    where
        AB: AirBuilder<F = F> + IndexedLookupBuilder,
    {
        fn eval(&self, builder: &mut AB) {
            match self {
                Self::Table(name) => {
                    // The table's own first entry is pinned, so the trace is constrained.
                    let main = builder.main();
                    builder
                        .when_first_row()
                        .assert_zero(main.current_slice()[0]);
                    builder.push_indexed_table(name, TraceWindow::Main, [0]);
                }
                Self::Fixed(name, _) => {
                    // The key already fixes the entries, so the main trace carries nothing
                    // and is pinned to zero to give this AIR a constraint of its own.
                    let main = builder.main();
                    builder.assert_zero(main.current_slice()[0]);
                    builder.push_indexed_table(name, TraceWindow::Preprocessed, [0]);
                }
                Self::Reader(name) => {
                    let main = builder.main();
                    builder
                        .when_first_row()
                        .assert_zero(main.current_slice()[0]);
                    builder.push_indexed_read(name, 0, [1]);
                }
            }
        }
    }

    /// One table, and every reader pulling from it.
    struct Lookup {
        /// Name both sides resolve this table by.
        name: &'static str,
        /// Trace the provider commits the entries to.
        window: TraceWindow,
        /// The table's entries.
        entries: Vec<u64>,
        /// Per reader: the entry each row names, and the value it claims to have pulled.
        readers: Vec<(Vec<u64>, Vec<u64>)>,
    }

    /// Entries a table needs before a commitment of its own opens in packed form.
    ///
    /// The packed opening wants a full element per prefix variable below the padding floor.
    ///
    /// A narrower table leaves it short of a lane, which is a panic on wide targets and
    /// invisible on scalar ones.
    fn packed_floor() -> usize {
        (1 << FOLDING) * PackedF::WIDTH
    }

    /// A table of squares wide enough to carry its own commitment on every target.
    fn squares() -> Vec<u64> {
        (0..packed_floor() as u64)
            .map(|entry| entry * entry)
            .collect()
    }

    /// Prove a batch whose committed traces agree with the forged reduction inputs.
    ///
    /// The reduction then accepts its own statement, so the only thing left to reject the
    /// proof is the comparison of its claims against the commitment.
    ///
    /// Providers come first in instance order, then every reader in table order.
    ///
    /// # Arguments
    ///
    /// - `lookups`: the tables to commit, and what each of their readers commits to.
    /// - `forgery`: what the reduction reads in place of the committed traces, or nothing.
    fn verdict(
        lookups: &[Lookup],
        forgery: Option<&Forgery>,
    ) -> Result<(), VerificationError<PcsError<TestConfig>>> {
        // A preprocessed provider carries its entries in the key, so its main trace is one
        // unconstrained column of the same height.
        let providers = lookups
            .iter()
            .map(|lookup| match lookup.window {
                TraceWindow::Main => (
                    Squares::Table(lookup.name),
                    lookup.entries.iter().copied().map(F::from_u64).collect(),
                ),
                TraceWindow::Preprocessed => (
                    Squares::Fixed(lookup.name, lookup.entries.clone()),
                    F::zero_vec(lookup.entries.len()),
                ),
            })
            .collect::<Vec<_>>();

        // Each reader commits its position column beside its payload column.
        let readers = lookups
            .iter()
            .flat_map(|lookup| {
                lookup.readers.iter().map(|(named, pulled)| {
                    let rows = named
                        .iter()
                        .zip(pulled)
                        .flat_map(|(&entry, &value)| [F::from_u64(entry), F::from_u64(value)])
                        .collect::<Vec<_>>();
                    (Squares::Reader(lookup.name), rows, named.len())
                })
            })
            .collect::<Vec<_>>();

        let airs = providers
            .iter()
            .map(|(air, _)| air)
            .chain(readers.iter().map(|(air, _, _)| air))
            .collect::<Vec<_>>();

        // One scheme per commitment, each sized for the values stacked into it.
        let main_values = providers.iter().map(|(_, rows)| rows.len()).sum::<usize>()
            + readers.iter().map(|(_, rows, _)| rows.len()).sum::<usize>();
        let preprocessed_values = lookups
            .iter()
            .filter(|lookup| lookup.window == TraceWindow::Preprocessed)
            .map(|lookup| lookup.entries.len())
            .sum::<usize>();
        let config = config(
            log2_ceil_usize(main_values),
            log2_ceil_usize(preprocessed_values).max(FOLDING),
        );
        let (pk, vk) = setup(&config, &airs, &mut challenger()).unwrap();

        let committed =
            |rows: &[F], width| Table::new(RowMajorMatrix::new(rows.to_vec(), width).transpose());
        let proving = providers
            .iter()
            .map(|(air, rows)| ProverInstance::new(air, committed(rows, 1), &pk, &[]))
            .chain(
                readers
                    .iter()
                    .map(|(air, rows, _)| ProverInstance::new(air, committed(rows, 2), &pk, &[])),
            )
            .collect();

        let proof = prove_forged::<_, _, GenericBackend>(
            &config,
            ProverInstances::new(proving),
            0,
            &mut challenger(),
            forgery,
        )
        .expect("a forged reduction input still produces a proof");

        let verifying = providers
            .iter()
            .zip(lookups)
            .map(|((air, _), lookup)| {
                VerifierInstance::new(air, &vk, log2_strict_usize(lookup.entries.len()), &[])
            })
            .chain(readers.iter().map(|(air, _, rows)| {
                VerifierInstance::new(air, &vk, log2_strict_usize(*rows), &[])
            }))
            .collect();

        verify(
            &config,
            VerifierInstances::new(verifying),
            &proof,
            0,
            &mut challenger(),
        )
    }

    #[test]
    fn a_reduction_run_against_an_uncommitted_table_is_rejected() {
        // Two tables, and the forgery moves the second one in plan order.
        //
        //     table "t0"   committed and reduced against, both honest
        //     table "t1"   committed 0 1 4 9 ..., reduced against 0 1 5 9 ...
        //
        // Its reader commits to having pulled the forged entry, so the reduction accepts
        // its own statement and only the comparison against the table's batch separates
        // the two.
        //
        // A comparison that stopped after "t0" would take this proof.
        let entries = squares();
        let mut forged = entries.clone();
        forged[2] = 5;

        let honest = |name| Lookup {
            name,
            window: TraceWindow::Main,
            entries: entries.clone(),
            readers: vec![(vec![0, 1, 2, 3, 3, 2, 1, 0], vec![0, 1, 4, 9, 9, 4, 1, 0])],
        };
        let mut second = honest("t1");
        second.readers = vec![(vec![0, 1, 2, 3, 3, 2, 1, 0], vec![0, 1, 5, 9, 9, 5, 1, 0])];

        let verdict = verdict(
            &[honest("t0"), second],
            Some(&Forgery {
                columns: Some((1, forged)),
                ..Forgery::default()
            }),
        );
        assert!(matches!(
            verdict,
            Err(VerificationError::IndexedClaimsUnopened)
        ));
    }

    #[test]
    fn packed_indexed_sources_are_rejected_before_commitment_and_transcript() {
        let table_air = Squares::Table("t");
        let reader_air = Squares::Reader("t");
        let config = config(FOLDING + 2, FOLDING);
        let (proving_key, _) =
            setup(&config, &[&table_air, &reader_air], &mut challenger()).unwrap();
        let packed_table = Table::from_packed_bits(RowMajorMatrix::new(vec![0u64], 1), FOLDING);
        let reader_table = Table::zero(2, FOLDING);
        let proving_instances = ProverInstances::new(vec![
            ProverInstance::new(&table_air, packed_table, &proving_key, &[]),
            ProverInstance::new(&reader_air, reader_table, &proving_key, &[]),
        ]);
        let mut challenger = challenger();
        let mut expected = challenger.clone();
        let panic = match catch_unwind(AssertUnwindSafe(|| {
            prove_forged::<_, _, GenericBackend>(
                &config,
                proving_instances,
                0,
                &mut challenger,
                None,
            )
        })) {
            Ok(_) => panic!("packed indexed input must be rejected"),
            Err(panic) => panic,
        };
        let message = panic
            .downcast_ref::<&str>()
            .copied()
            .or_else(|| panic.downcast_ref::<String>().map(String::as_str))
            .unwrap_or_default();
        assert!(
            message.contains("packed Boolean source tables"),
            "{message}"
        );
        assert_eq!(
            config.main_pcs_uses.get(),
            0,
            "packed indexed rejection must precede the main PCS commitment"
        );
        for _ in 0..4 {
            assert_eq!(
                CanSample::<F>::sample(&mut challenger),
                CanSample::<F>::sample(&mut expected),
                "caller challenger changed before packed indexed rejection"
            );
        }
    }

    #[test]
    fn packed_indexed_preprocessed_sources_are_rejected_before_pcs_and_transcript() {
        let height = packed_floor();
        let log_height = log2_strict_usize(height);
        let fixed_air = Squares::Fixed("t", vec![0; height]);
        let reader_air = Squares::Reader("t");
        let airs = [&fixed_air, &reader_air];
        let mut config = config(log_height + 2, log_height);
        let (proving_key, _) = setup(&config, &airs, &mut challenger()).unwrap();
        static PACKED_PREPROCESSED: OnceLock<Table<F>> = OnceLock::new();
        let packed_preprocessed = PACKED_PREPROCESSED.get_or_init(|| {
            let height = packed_floor();
            Table::from_packed_bits(
                RowMajorMatrix::new(vec![0u64; height.div_ceil(64)], 1),
                log_height,
            )
        });
        config.committed_table_override = Some(packed_preprocessed);

        let fixed_table = Table::zero(1, log_height);
        let reader_table = Table::zero(2, log_height);
        let proving_instances = ProverInstances::new(vec![
            ProverInstance::new(&fixed_air, fixed_table, &proving_key, &[]),
            ProverInstance::new(&reader_air, reader_table, &proving_key, &[]),
        ]);
        let mut challenger = challenger();
        let mut expected = challenger.clone();
        let panic = match catch_unwind(AssertUnwindSafe(|| {
            prove_forged::<_, _, GenericBackend>(
                &config,
                proving_instances,
                0,
                &mut challenger,
                None,
            )
        })) {
            Ok(_) => panic!("packed indexed preprocessed input must be rejected"),
            Err(panic) => panic,
        };
        let message = panic
            .downcast_ref::<&str>()
            .copied()
            .or_else(|| panic.downcast_ref::<String>().map(String::as_str))
            .unwrap_or_default();
        assert!(
            message.contains("packed Boolean preprocessed tables"),
            "{message}"
        );
        assert_eq!(
            config.main_pcs_uses.get(),
            0,
            "packed preprocessed rejection must precede the main PCS commitment"
        );
        for _ in 0..4 {
            assert_eq!(
                CanSample::<F>::sample(&mut challenger),
                CanSample::<F>::sample(&mut expected),
                "caller challenger changed before packed preprocessed rejection"
            );
        }
    }

    #[test]
    fn a_reduction_run_against_uncommitted_positions_is_rejected() {
        // One table with two readers, and the forgery moves the second reader.
        //
        //     reader 0   names 0 1 2 3 3 2 1 0, honest throughout
        //     reader 1   names 0 1 2 3 3 2 1 0, reduced against ... 1
        //                pulls 0 1 4 9 9 4 1 1, matching the entry it was reduced against
        //
        // Every pull agrees with the entry the reduction was told about, so the reduction
        // accepts its own statement.
        //
        // A loop that stopped after reader 0 would take this proof.
        let verdict = verdict(
            &[Lookup {
                name: "t0",
                window: TraceWindow::Main,
                entries: squares(),
                readers: vec![
                    (vec![0, 1, 2, 3, 3, 2, 1, 0], vec![0, 1, 4, 9, 9, 4, 1, 0]),
                    (vec![0, 1, 2, 3, 3, 2, 1, 0], vec![0, 1, 4, 9, 9, 4, 1, 1]),
                ],
            }],
            Some(&Forgery {
                positions: Some((0, 1, vec![0, 1, 2, 3, 3, 2, 1, 1])),
                ..Forgery::default()
            }),
        );
        assert!(matches!(
            verdict,
            Err(VerificationError::IndexedClaimsUnopened)
        ));
    }

    #[test]
    fn claims_the_reader_never_committed_to_are_rejected() {
        // The identity table, so what a row pulls is the entry it names.
        //
        //     reader 0   names 0 1 2 3 3 2 1 0, pulls the same, honest throughout
        //     reader 1   names 0 1 2 3 3 2 1 0, but commits pulls 0 7 7 7 7 7 7 0
        //
        // Reader 1's claims are read off its position column instead of its payload
        // column, so they describe an honest reduction over the committed table.
        //
        // A comparison that looked only at reader 0 would take this proof.
        let identity = (0..packed_floor() as u64).collect();
        let named = vec![0, 1, 2, 3, 3, 2, 1, 0];

        let verdict = verdict(
            &[Lookup {
                name: "t0",
                window: TraceWindow::Main,
                entries: identity,
                readers: vec![
                    (named.clone(), named.clone()),
                    (named, vec![0, 7, 7, 7, 7, 7, 7, 0]),
                ],
            }],
            Some(&Forgery {
                claims: Some((1, vec![0])),
                ..Forgery::default()
            }),
        );
        assert!(matches!(
            verdict,
            Err(VerificationError::IndexedClaimsUnopened)
        ));
    }

    #[test]
    fn a_reduction_run_against_an_uncommitted_preprocessed_table_is_rejected() {
        // The same substitution as for a table in the main trace, except that the provider
        // fixes its entries in the verifying key.
        //
        // The table claims are then discharged against the preprocessed commitment rather
        // than the main one, and that is the path this pins.
        let entries = squares();
        let mut forged = entries.clone();
        forged[2] = 5;

        let verdict = verdict(
            &[Lookup {
                name: "t0",
                window: TraceWindow::Preprocessed,
                entries,
                readers: vec![(vec![0, 1, 2, 3, 3, 2, 1, 0], vec![0, 1, 5, 9, 9, 5, 1, 0])],
            }],
            Some(&Forgery {
                columns: Some((0, forged)),
                ..Forgery::default()
            }),
        );
        assert!(matches!(
            verdict,
            Err(VerificationError::IndexedClaimsUnopened)
        ));
    }

    #[test]
    fn an_honest_batch_reading_a_preprocessed_table_verifies() {
        // Every rejection above is also what a verifier that never opens the preprocessed
        // table produces, since a missing batch leaves nothing for the claims to match.
        //
        // This is the case that tells the two apart: the reader pulls what the key holds,
        // and nothing is substituted.
        let entries = squares();
        let pulled = [0, 1, 2, 3, 3, 2, 1, 0]
            .iter()
            .map(|&entry: &usize| entries[entry])
            .collect();

        verdict(
            &[Lookup {
                name: "t0",
                window: TraceWindow::Preprocessed,
                entries,
                readers: vec![(vec![0, 1, 2, 3, 3, 2, 1, 0], pulled)],
            }],
            None,
        )
        .expect("a reader pulling what the key holds must verify");
    }

    #[test]
    fn conditional_nonlinear_bus_is_bound_to_committed_openings() {
        let (config, push, pull, _pk, vk, mut proof) = conditional_bus_fixture();
        let verify_with = |proof: &MultiStarkProof<TestConfig>, push, pull, height| {
            verify(
                &config,
                VerifierInstances::new(vec![
                    VerifierInstance::new(push, &vk, height, &[]),
                    VerifierInstance::new(pull, &vk, height, &[]),
                ]),
                proof,
                0,
                &mut challenger(),
            )
        };

        // The new composition sumcheck accepts a balanced bus with both nonlinear payload
        // and nonconstant activation.
        verify_with(&proof, &push, &pull, 2).unwrap();

        // Directly composing separately opened MLEs is not the MLE of the rowwise product.
        let r = EF::from_u64(3);
        let value_at_r = EF::from_u64(1) + (EF::from_u64(2) - EF::from_u64(1)) * r;
        let selector_at_r = r;
        let old_shortcut = selector_at_r * value_at_r.square();
        let true_factor_mle = EF::from_u64(4) * r;
        assert_ne!(old_shortcut, true_factor_mle);

        // Every proof-controlled layer message is bound by ProductGKR or composition sumcheck.
        proof.bus.as_mut().unwrap().product.product.roots[0] += EF::ONE;
        assert!(verify_with(&proof, &push, &pull, 2).is_err());
        proof.bus.as_mut().unwrap().product.product.roots[0] -= EF::ONE;

        proof.bus.as_mut().unwrap().composition.claimed_sum += EF::ONE;
        assert!(verify_with(&proof, &push, &pull, 2).is_err());
        proof.bus.as_mut().unwrap().composition.claimed_sum -= EF::ONE;

        proof.bus.as_mut().unwrap().composition.round_polys[0][0] += EF::ONE;
        assert!(verify_with(&proof, &push, &pull, 2).is_err());
        proof.bus.as_mut().unwrap().composition.round_polys[0][0] -= EF::ONE;

        // Direction, activation, and block geometry are verifier statement metadata.
        let wrong_direction = ConditionalBusAir {
            direction: BusDirection::Push,
            conditional: true,
        };
        assert!(verify_with(&proof, &push, &wrong_direction, 2).is_err());
        let wrong_activation = ConditionalBusAir {
            direction: BusDirection::Pull,
            conditional: false,
        };
        assert!(verify_with(&proof, &push, &wrong_activation, 2).is_err());

        // A height change moves the block prefix while retaining the same AIR declarations.
        assert!(matches!(
            verify_with(&proof, &push, &pull, 3),
            Err(VerificationError::BusArgument(_))
        ));

        // Alter only the prescribed bus-opening answer while retaining its proof shape.
        let batch = proof.opening.evals.last_mut().unwrap();
        let original = batch.clone();
        let mut current = batch.current().to_vec();
        current[0] += EF::ONE;
        *batch = p3_sumcheck::OpeningBatch::new(current, batch.next().to_vec());
        assert!(verify_with(&proof, &push, &pull, 2).is_err());
        *proof.opening.evals.last_mut().unwrap() = original;
    }

    #[test]
    fn mixed_height_bus_uses_the_fixed_all_one_vertex_selector() {
        let push = ConditionalBusAir {
            direction: BusDirection::Push,
            conditional: true,
        };
        let pull = ConditionalBusAir {
            direction: BusDirection::Pull,
            conditional: true,
        };
        let config = config(5, FOLDING);
        let (pk, vk) = setup(&config, &[&push, &pull], &mut challenger()).unwrap();
        let push_values = vec![1, 2, 3, 4, 9, 10, 11, 12]
            .into_iter()
            .map(F::from_u64)
            .chain([
                F::ONE,
                F::ONE,
                F::ONE,
                F::ONE,
                F::ZERO,
                F::ZERO,
                F::ZERO,
                F::ZERO,
            ])
            .collect();
        let pull_values = vec![1, 2, 3, 4]
            .into_iter()
            .map(F::from_u64)
            .chain([F::ONE; 4])
            .collect();
        let proof = prove(
            &config,
            ProverInstances::new(vec![
                ProverInstance::new(
                    &push,
                    Table::new(RowMajorMatrix::new(push_values, 8)),
                    &pk,
                    &[],
                ),
                ProverInstance::new(
                    &pull,
                    Table::new(RowMajorMatrix::new(pull_values, 4)),
                    &pk,
                    &[],
                ),
            ]),
            0,
            &mut challenger(),
        )
        .unwrap();

        verify(
            &config,
            VerifierInstances::new(vec![
                VerifierInstance::new(&push, &vk, 3, &[]),
                VerifierInstance::new(&pull, &vk, 2, &[]),
            ]),
            &proof,
            0,
            &mut challenger(),
        )
        .unwrap();
    }

    #[test]
    fn bus_payload_from_preprocessed_column_is_opened() {
        // The push side is fixed in the verifying key.
        // The pull side commits the same payload in the main trace.
        let height = packed_floor();
        let log_height = log2_strict_usize(height);
        let values = (1..=height as u64).map(F::from_u64).collect::<Vec<_>>();
        let push = PreprocessedBusAir {
            direction: BusDirection::Push,
            fixed: Some(values.clone()),
        };
        let pull = PreprocessedBusAir {
            direction: BusDirection::Pull,
            fixed: None,
        };
        let config = config(log_height + 1, log_height);
        let (pk, vk) = setup(&config, &[&push, &pull], &mut challenger()).unwrap();

        // The fixed provider still has one main column because every AIR table must be nonempty.
        let proof = prove(
            &config,
            ProverInstances::new(vec![
                ProverInstance::new(
                    &push,
                    Table::new(RowMajorMatrix::new(F::zero_vec(height), height)),
                    &pk,
                    &[],
                ),
                ProverInstance::new(
                    &pull,
                    Table::new(RowMajorMatrix::new(values, height)),
                    &pk,
                    &[],
                ),
            ]),
            0,
            &mut challenger(),
        )
        .unwrap();

        // Verification must consume both prescribed opening batches.
        verify(
            &config,
            VerifierInstances::new(vec![
                VerifierInstance::new(&push, &vk, log_height, &[]),
                VerifierInstance::new(&pull, &vk, log_height, &[]),
            ]),
            &proof,
            0,
            &mut challenger(),
        )
        .unwrap();
    }
}
