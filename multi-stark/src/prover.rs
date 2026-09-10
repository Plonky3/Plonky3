//! Prove that AIR instances are satisfied by committed traces.

use alloc::vec::Vec;

use p3_challenger::{CanObserve, CanSampleUniformBits, FieldChallenger, GrindingChallenger};
use p3_commit::MultilinearPcs;
use p3_field::{ExtensionField, Field};
use p3_sumcheck::PrescribedPointPcs;

use crate::ProverInstances;
use crate::config::{Commitment, MultiStarkConfig, PcsProverError, ProverData};
use crate::folder::ProverAir;
use crate::instance::ProverParts;
use crate::lookup::prove_lookup;
use crate::proof::MultiStarkProof;
use crate::security::{SecurityError, assess_statement};
use crate::transcript::{MultiStarkProverTranscript, MultiStarkShape};
use crate::zerocheck::AirZerocheck;

/// A proving-time PCS budget failure or a failed statement security assessment.
#[derive(Debug, thiserror::Error)]
pub enum ProvingError<E> {
    /// The PCS rejected a commitment or an opening budget.
    #[error("PCS {phase} failed: {source:?}")]
    Pcs { phase: &'static str, source: E },
    /// The requested complete-statement security target was not met.
    #[error(transparent)]
    Security(#[from] SecurityError),
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
///     4. lookup reduction (if any)  -> delegated
///     5. zerocheck reduction        -> delegated, yields bound point r
///     6. open main tables at r      -> delegated, openings bound to the main commitment
///     7. open preprocessed tables at r (if any)
///                                   -> delegated, bound to the preprocessed commitment
/// ```
///
/// Phases 2 and 4 through 7 run inside a recorded `Begin`/`End` bracket.
/// The pattern player therefore rejects a run that skips one or reorders two.
///
/// Main trace tables are committed together in input-instance order. Each table
/// is still opened at the suffix of the common zerocheck point matching that
/// instance's height.
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
#[tracing::instrument(skip_all)]
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

    // Describe the statement before binding anything into it.
    //
    // Every number comes from the AIRs, from the tables this caller holds, and from `pow_bits`.
    // No proof exists yet, so none of them can come from one.
    let num_instances = instances.len();
    let airs = instances.airs();
    let public_values = instances.public_values();
    let mut transcript = MultiStarkProverTranscript::<C::Challenger, C::Val>::new(
        challenger,
        MultiStarkShape::new::<C::Val, A>(&airs, &instances.num_variables(), pow_bits),
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

    // 4. Materialize the lookup fractions and reduce them, inside the delegation bracket.
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

    // 5. Reduce all AIR constraints to one batched sumcheck and one bound point.
    // The committed prover opens columns through the commitment schemes below, so
    // the zerocheck's own opened values are not used as the final proof openings.
    let zerocheck = AirZerocheck::new(&airs, pow_bits);
    let (zerocheck_proof, point) = transcript.zerocheck(|challenger| {
        zerocheck.prove_with_lookup::<C::Val, C::Challenge, _>(
            &preprocessed_tables,
            &tables,
            &public_values,
            lookup_data,
            challenger,
        )
    });
    let sumcheck = zerocheck_proof.sumcheck;

    drop(tables);
    drop(preprocessed_tables);

    // 6. Open each main trace table at its suffix of the common bound point.
    let opening = transcript.main_opening(|challenger| {
        config.pcs().open_at(
            prover_data,
            &instances.opening_protocol(),
            &instances.main_points(&point),
            challenger,
        )
    });

    let opening = opening
        .inspect_err(|_| transcript.abort())
        .map_err(|source| ProvingError::Pcs {
            phase: "main opening",
            source,
        })?;

    // 7. Open each non-empty preprocessed table at its suffix of the same bound point.
    // The setup commitment data is reused rather than rebuilt.
    let preprocessed_opening = transcript.preprocessed_opening(|challenger| {
        let preprocessed = proving_key
            .preprocessed
            .as_ref()
            .expect("preprocessed proving key is missing for an AIR with preprocessed columns");
        config.preprocessed_pcs().open_at(
            preprocessed.prover_data.clone(),
            &instances.preprocessed_opening_protocol(),
            &instances.preprocessed_points(&point),
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
        sumcheck,
        opening,
        preprocessed_opening,
    })
}
