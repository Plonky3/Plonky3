mod data;

use alloc::vec;
use alloc::vec::Vec;

pub use data::{BatchArith, VerifierData};
use p3_air::symbolic::{AirLayout, SymbolicExpressionExt};
use p3_air::{Air, BaseAir};
use p3_challenger::GrindingChallenger;
use p3_commit::{CommitmentOpening, Pcs, PeriodicColumns, PolynomialSpace, UnivariateStarkPcs};
use p3_field::{Algebra, BasedVectorSpace, PrimeCharacteristicRing, PrimeField64};
use p3_lookup::logup::LogUpGadget;
use p3_lookup::{InteractionSymbolicBuilder, LookupError, check_multiplicity_height_bound};
use p3_uni_stark::{
    ConstraintInputs, InvalidProofShapeError, StarkArith, VerificationError, validate_degree_bits,
};
use p3_util::checked_log_size_sum;
use p3_util::zip_eq::zip_eq;
use p3_verifier_arith::{Native, recompose_ext};
use tracing::{info_span, instrument};

use crate::common::CommonData;
use crate::config::{Challenge, Commitment, Domain, PcsError, StarkGenericConfig as SGC, Val};
use crate::error::BatchVerificationError;
use crate::folder::VerifierConstraintFolderWithLookups;
use crate::proof::{BatchCommitments, BatchOpenedValues, BatchProof};
use crate::symbolic::get_log_num_quotient_chunks_for_domain;
use crate::transcript::{BatchShape, BatchVerifierTranscript, lay_out_lookup_challenges};

// Keep these checks shared without moving them across the entry points' other shape checks.
fn validate_trace_next<F, A: BaseAir<F>, EF>(
    air: &A,
    opened: &p3_uni_stark::OpenedValues<EF>,
    i: usize,
) -> Result<(), InvalidProofShapeError> {
    if !air.main_next_row_columns().is_empty() {
        if opened
            .trace_next
            .as_ref()
            .is_none_or(|next| next.len() != air.width())
        {
            return Err(InvalidProofShapeError::TraceNextMismatch { air: Some(i) });
        }
    } else if opened.trace_next.is_some() {
        return Err(InvalidProofShapeError::UnexpectedTraceNext { air: Some(i) });
    }
    Ok(())
}

fn validate_preprocessed_opening<F, A: BaseAir<F>, EF>(
    air: &A,
    opened: &p3_uni_stark::OpenedValues<EF>,
    width: usize,
    i: usize,
) -> Result<(), InvalidProofShapeError> {
    let next_used = !air.preprocessed_next_row_columns().is_empty();
    // Presence matters even for zero-width rows: an empty opening is still an opening.
    match &opened.preprocessed {
        Some(_) if width == 0 => {
            return Err(InvalidProofShapeError::UnexpectedPreprocessedValues { air: Some(i) });
        }
        Some(preprocessed) => {
            let expected_next = if next_used { width } else { 0 };
            if preprocessed.local.len() != width
                || preprocessed.next.as_ref().map_or(0, Vec::len) != expected_next
            {
                return Err(InvalidProofShapeError::PreprocessedWidthMismatch { air: i });
            }
            if !next_used && preprocessed.next.is_some() {
                return Err(InvalidProofShapeError::UnexpectedPreprocessedNext { air: Some(i) });
            }
        }
        None if width > 0 => {
            return Err(InvalidProofShapeError::PreprocessedWidthMismatch { air: i });
        }
        None => {}
    }
    Ok(())
}

fn validate_randomization_presence<SC: SGC, Com, EF>(
    commitments: &BatchCommitments<Com>,
    opened_values: &BatchOpenedValues<EF>,
) -> Result<(), VerificationError<PcsError<SC>>> {
    if commitments.random.is_some() != SC::Pcs::ZK
        || opened_values
            .instances
            .iter()
            .any(|instance| instance.base_opened_values.random.is_some() != SC::Pcs::ZK)
    {
        return Err(VerificationError::RandomizationError);
    }
    Ok(())
}

fn validate_random_opening_dimension<SC: SGC, EF>(
    opened: &p3_uni_stark::OpenedValues<EF>,
) -> Result<(), VerificationError<PcsError<SC>>> {
    if opened
        .random
        .as_ref()
        .is_some_and(|values| values.len() != Challenge::<SC>::DIMENSION)
    {
        return Err(VerificationError::RandomizationError);
    }
    Ok(())
}

/// The opening argument of a batch, with each instance's quotient-chunk domains.
///
/// The last entry is the position of the preprocessed claim, taken from trusted metadata.
pub type OpeningArgumentWithQuotientDomains<SC, EF, Com> = (
    Vec<CommitmentOpening<EF, Com, Domain<SC>>>,
    Vec<Vec<Domain<SC>>>,
    Option<usize>,
);

/// Builds the claims a batch-STARK proof's PCS opening argument is checked against.
///
/// One round per commitment, in commit order:
///
/// - the ZK randomization round, when the PCS hides;
/// - the trace round, at `zeta` and at the point after it when an AIR reads the next row;
/// - the quotient-chunks round;
/// - the preprocessed round, when the batch commits to preprocessed columns;
/// - the permutation round, when the batch declares lookups.
///
/// Performs no transcript interaction.
#[expect(clippy::too_many_arguments, clippy::type_complexity)]
pub fn commitments_with_opening_points<SC, A, AR>(
    arith: &mut AR,
    config: &SC,
    airs: &[A],
    zeta: &AR::EF,
    commitments: &BatchCommitments<AR::Commitment>,
    preprocessed_commitment: Option<&AR::Commitment>,
    opened_values: &BatchOpenedValues<AR::EF>,
    common: &CommonData<SC>,
    degree_bits: &[usize],
    preprocessed_widths: &[usize],
    log_num_quotient_chunks: &[usize],
) -> Result<
    OpeningArgumentWithQuotientDomains<SC, AR::EF, AR::Commitment>,
    BatchVerificationError<PcsError<SC>>,
>
where
    SC: SGC,
    A: BaseAir<Val<SC>>,
    AR: StarkArith<SC>,
{
    let pcs = config.pcs();
    let is_zk = config.is_zk();

    // Check parallel collections before any zip can truncate or index can panic.
    let num_instances = airs.len();
    if opened_values.instances.len() != num_instances
        || degree_bits.len() != num_instances
        || preprocessed_widths.len() != num_instances
        || log_num_quotient_chunks.len() != num_instances
        || common.lookups.len() != num_instances
        || common
            .preprocessed
            .as_ref()
            .is_some_and(|global| global.instances.len() != num_instances)
    {
        return Err(InvalidProofShapeError::InstanceCountMismatch.into());
    }
    validate_randomization_presence::<SC, _, _>(commitments, opened_values)?;
    for instance in &opened_values.instances {
        validate_random_opening_dimension::<SC, _>(&instance.base_opened_values)?;
    }
    if let Some(global) = &common.preprocessed {
        for (matrix_index, &instance_index) in global.matrix_to_instance.iter().enumerate() {
            if global
                .instances
                .get(instance_index)
                .and_then(Option::as_ref)
                .is_none_or(|meta| meta.matrix_index != matrix_index)
            {
                return Err(InvalidProofShapeError::PreprocessedMetadataMismatch {
                    air: instance_index,
                }
                .into());
            }
        }
    }
    for (i, (air, instance)) in airs.iter().zip(&opened_values.instances).enumerate() {
        let opened = &instance.base_opened_values;
        validate_trace_next(air, opened, i)?;

        let width = preprocessed_widths[i];
        let meta = common
            .preprocessed
            .as_ref()
            .and_then(|global| global.instances[i].as_ref());
        if meta.map_or(0, |meta| meta.width) != width
            || meta.is_some_and(|meta| {
                meta.width == 0
                    || meta.degree_bits != degree_bits[i]
                    || common
                        .preprocessed
                        .as_ref()
                        .and_then(|global| global.matrix_to_instance.get(meta.matrix_index))
                        != Some(&i)
            })
        {
            return Err(InvalidProofShapeError::PreprocessedMetadataMismatch { air: i }.into());
        }
        validate_preprocessed_opening(air, opened, width, i)?;
    }
    let mut coms_to_verify = vec![];

    // Trace round: per instance, open at zeta and zeta_next.
    //
    // The extended domain size is `1 << degree_bits[i]`.
    //
    // `validate_degree_bits` gates it, so an out-of-range degree is an error, not an overflow.
    let trace_domain_pairs = degree_bits
        .iter()
        .enumerate()
        .map(|(i, &ext_db)| {
            let (_, ext_domain_size) = validate_degree_bits(
                Some(i),
                ext_db,
                is_zk,
                pcs.log_min_trace_height(),
                pcs.log_max_trace_height(),
            )?;
            Ok((
                pcs.natural_domain_for_degree(ext_domain_size >> is_zk),
                pcs.natural_domain_for_degree(ext_domain_size),
            ))
        })
        .collect::<Result<Vec<_>, InvalidProofShapeError>>()?;
    let (trace_domains, ext_trace_domains): (Vec<Domain<SC>>, Vec<Domain<SC>>) =
        trace_domain_pairs.into_iter().unzip();

    // The point after `zeta` in one instance's trace domain.
    let zeta_next = |arith: &mut AR, i: usize| {
        arith
            .next_point(&trace_domains[i], zeta)
            .ok_or(VerificationError::NextPointUnavailable)
    };

    if let Some(random_commit) = &commitments.random {
        coms_to_verify.push((
            random_commit.clone(),
            ext_trace_domains
                .iter()
                .zip(opened_values.instances.iter())
                .map(|(domain, inst_opened_vals)| {
                    // We already checked that random is present for each instance when ZK is enabled.
                    let random_vals = inst_opened_vals.base_opened_values.random.as_ref().unwrap();
                    (*domain, vec![(zeta.clone(), random_vals.clone())])
                })
                .collect::<Vec<_>>(),
        ));
    }

    let mut trace_round = Vec::with_capacity(num_instances);
    for (i, (ext_dom, inst_opened_vals)) in ext_trace_domains
        .iter()
        .zip(opened_values.instances.iter())
        .enumerate()
    {
        let mut points = vec![(
            zeta.clone(),
            inst_opened_vals.base_opened_values.trace_local.clone(),
        )];
        if !airs[i].main_next_row_columns().is_empty() {
            points.push((
                zeta_next(arith, i)?,
                inst_opened_vals
                    .base_opened_values
                    .trace_next
                    .clone()
                    .expect("checked in shape validation"),
            ));
        }
        trace_round.push((*ext_dom, points));
    }
    coms_to_verify.push((commitments.main.clone(), trace_round));

    // Quotient chunks round: flatten per-instance chunks to match commit order.
    // Use extended domains for the outer commit domain, with size = base_degree * num_quotient_chunks.
    let quotient_domains: Vec<Vec<Domain<SC>>> = (0..degree_bits.len())
        .map(|i| {
            let ext_db = degree_bits[i];
            let log_num_chunks = log_num_quotient_chunks[i];
            // The committed chunk count, ZK randomization included.
            let (_, n_chunks) = checked_log_size_sum(log_num_chunks, is_zk).ok_or_else(|| {
                InvalidProofShapeError::QuotientDomainTooLarge {
                    air: Some(i),
                    maximum: usize::BITS as usize - 1,
                    got: log_num_chunks.saturating_add(is_zk),
                }
            })?;
            let ext_dom = ext_trace_domains[i];
            let (quotient_domain_log_size, quotient_domain_size) =
                checked_log_size_sum(ext_db, log_num_chunks).ok_or_else(|| {
                    InvalidProofShapeError::QuotientDomainTooLarge {
                        air: Some(i),
                        maximum: usize::BITS as usize - 1,
                        got: ext_db.saturating_add(log_num_chunks),
                    }
                })?;
            let qdom = ext_dom
                .try_create_disjoint_domain(quotient_domain_size)
                .ok_or_else(|| InvalidProofShapeError::QuotientDomainTooLarge {
                    air: Some(i),
                    maximum: pcs.log_max_trace_height(),
                    got: quotient_domain_log_size,
                })?;
            Ok(qdom.split_domains(n_chunks))
        })
        .collect::<Result<Vec<_>, InvalidProofShapeError>>()?;

    // When ZK is enabled, the size of the quotient chunks' domains doubles.
    let randomized_quotient_chunks_domains = quotient_domains
        .iter()
        .map(|doms| {
            doms.iter()
                .map(|dom| pcs.natural_domain_for_degree(dom.size() << is_zk))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    // Build the per-matrix openings for the aggregated quotient commitment.
    let mut qc_round = Vec::new();
    for (i, domains) in randomized_quotient_chunks_domains.iter().enumerate() {
        let inst_qcs = &opened_values.instances[i]
            .base_opened_values
            .quotient_chunks;
        for (d, vals) in zip_eq(
            domains.iter(),
            inst_qcs,
            VerificationError::from(InvalidProofShapeError::QuotientDomainsCountMismatch {
                air: i,
            }),
        )? {
            qc_round.push((*d, vec![(zeta.clone(), vals.clone())]));
        }
    }
    coms_to_verify.push((commitments.quotient_chunks.clone(), qc_round));

    // Preprocessed rounds: a single global commitment with one matrix per
    // instance that has preprocessed columns.
    let preprocessed_index = common.preprocessed.as_ref().map(|_| coms_to_verify.len());
    if let (Some(global), Some(preprocessed_commitment)) =
        (&common.preprocessed, preprocessed_commitment)
    {
        let mut pre_round = Vec::new();

        for (matrix_index, &inst_idx) in global.matrix_to_instance.iter().enumerate() {
            // The preflight validated the metadata bijection and opening shapes.
            debug_assert_ne!(preprocessed_widths[inst_idx], 0);
            let inst = &opened_values.instances[inst_idx];
            let preprocessed = inst
                .base_opened_values
                .preprocessed
                .as_ref()
                .expect("checked in shape validation");
            let local = &preprocessed.local;
            let meta = global.instances[inst_idx]
                .as_ref()
                .expect("checked in shape validation");
            debug_assert_eq!(meta.matrix_index, matrix_index);
            debug_assert_eq!(meta.degree_bits, degree_bits[inst_idx]);

            let pre_domain = pcs.natural_domain_for_degree(1 << meta.degree_bits);
            if !airs[inst_idx].preprocessed_next_row_columns().is_empty() {
                let next = preprocessed
                    .next
                    .as_ref()
                    .expect("checked in shape validation");
                pre_round.push((
                    pre_domain,
                    vec![
                        (zeta.clone(), local.clone()),
                        (zeta_next(arith, inst_idx)?, next.clone()),
                    ],
                ));
            } else {
                pre_round.push((pre_domain, vec![(zeta.clone(), local.clone())]));
            }
        }

        coms_to_verify.push((preprocessed_commitment.clone(), pre_round));
    }

    if let Some(permutation_commit) = &commitments.permutation {
        let mut permutation_round = Vec::new();
        for (i, (ext_dom, inst_opened_vals)) in ext_trace_domains
            .iter()
            .zip(opened_values.instances.iter())
            .enumerate()
        {
            if inst_opened_vals.permutation_local.len() != inst_opened_vals.permutation_next.len() {
                return Err(LookupError::PermutationLengthMismatch { air: i }.into());
            }
            if !inst_opened_vals.permutation_local.is_empty() {
                permutation_round.push((
                    *ext_dom,
                    vec![
                        (zeta.clone(), inst_opened_vals.permutation_local.clone()),
                        (
                            zeta_next(arith, i)?,
                            inst_opened_vals.permutation_next.clone(),
                        ),
                    ],
                ));
            }
        }
        coms_to_verify.push((permutation_commit.clone(), permutation_round));
    }

    Ok((
        coms_to_verify.into_iter().map(Into::into).collect(),
        quotient_domains,
        preprocessed_index,
    ))
}

/// Verifies a batch-STARK proof.
///
/// # Errors
///
/// When the proof is malformed or does not verify.
#[instrument(skip_all)]
pub fn verify_batch<SC, A>(
    config: &SC,
    airs: &[A],
    proof: &BatchProof<SC>,
    public_values: &[Vec<Val<SC>>],
    common: &CommonData<SC>,
) -> Result<(), BatchVerificationError<PcsError<SC>>>
where
    SC: SGC,
    Val<SC>: PrimeField64,
    SymbolicExpressionExt<Val<SC>, SC::Challenge>: Algebra<SC::Challenge>,
    A: Air<InteractionSymbolicBuilder<Val<SC>, SC::Challenge>>
        + for<'a> Air<VerifierConstraintFolderWithLookups<'a, SC>>,
    Challenge<SC>: BasedVectorSpace<Val<SC>>,
    SC::Challenger: GrindingChallenger<Witness = Val<SC>>,
{
    verify_batch_in(
        &mut Native::new(),
        config,
        airs,
        proof,
        public_values,
        common,
    )
}

/// Verifies a batch-STARK proof within an arithmetic.
///
/// The public values are lifted into the arithmetic.
///
/// Every shape check reads the native proof, so both arithmetics reject the same proofs.
///
/// # Errors
///
/// When the proof is malformed or does not verify.
pub fn verify_batch_in<AR, SC, A>(
    arith: &mut AR,
    config: &SC,
    airs: &[A],
    proof: &BatchProof<SC>,
    public_values: &[Vec<AR::F>],
    common: &CommonData<SC>,
) -> Result<(), BatchVerificationError<PcsError<SC>>>
where
    SC: SGC,
    Val<SC>: PrimeField64,
    SymbolicExpressionExt<Val<SC>, SC::Challenge>: Algebra<SC::Challenge>,
    A: Air<InteractionSymbolicBuilder<Val<SC>, SC::Challenge>>,
    AR: BatchArith<SC, A>,
    Challenge<SC>: BasedVectorSpace<Val<SC>>,
    SC::Challenger: GrindingChallenger<Witness = Val<SC>>,
{
    // Public inputs reach this proof only through AIR constraints.
    // A cell listed for backend binding would go completely unbound.
    assert!(
        airs.iter().all(|air| air.public_boundary_io().is_empty()),
        "batch-stark does not support boundary-IO public values; bind them with AIR constraints"
    );

    // TODO: Extend if additional lookup gadgets are added.
    let lookup_gadget = LogUpGadget::new();

    let BatchProof {
        commitments,
        opened_values,
        opening_proof,
        lookup_terminals,
        degree_bits,
        lookup_pow_witness,
        ood_pow_witness,
    } = proof;

    let all_lookups = &common.lookups;

    let pcs = config.pcs();

    // Sanity checks
    if airs.len() != opened_values.instances.len()
        || airs.len() != public_values.len()
        || airs.len() != degree_bits.len()
        || airs.len() != lookup_terminals.len()
        || airs.len() != all_lookups.len()
        || common
            .preprocessed
            .as_ref()
            .is_some_and(|global| global.instances.len() != airs.len())
    {
        return Err(InvalidProofShapeError::InstanceCountMismatch.into());
    }

    validate_randomization_presence::<SC, _, _>(commitments, opened_values)?;

    // Validate opened values shape per instance and observe per-instance binding data.
    // Precompute per-instance preprocessed widths and number of quotient chunks.
    let mut preprocessed_widths = Vec::with_capacity(airs.len());
    // Number of quotient chunks per instance before ZK randomization.
    let mut log_num_quotient_chunks = Vec::with_capacity(airs.len());
    // The total number of quotient chunks, including ZK randomization.
    let mut num_quotient_chunks = Vec::with_capacity(airs.len());
    let mut base_degree_bits = Vec::with_capacity(airs.len());
    let mut ext_domain_sizes = Vec::with_capacity(airs.len());

    for (i, air) in airs.iter().enumerate() {
        let (base_db, ext_domain_size) = validate_degree_bits(
            Some(i),
            degree_bits[i],
            config.is_zk(),
            pcs.log_min_trace_height(),
            pcs.log_max_trace_height(),
        )?;
        base_degree_bits.push(base_db);
        ext_domain_sizes.push(ext_domain_size);

        let pre_w = common
            .preprocessed
            .as_ref()
            .and_then(|g| g.instances[i].as_ref().map(|m| m.width))
            .unwrap_or(0);
        preprocessed_widths.push(pre_w);

        let layout = AirLayout {
            preprocessed_width: pre_w,
            main_width: air.width(),
            num_public_values: air.num_public_values(),
            num_periodic_columns: air.num_periodic_columns(),
            ..Default::default()
        };
        let log_num_chunks =
            info_span!("infer log of constraint degree", air_idx = i).in_scope(|| {
                get_log_num_quotient_chunks_for_domain::<_, SC::Challenge, A, _>(
                    air,
                    layout,
                    pcs.natural_domain_for_degree(1usize << base_db),
                    &all_lookups[i],
                    config.is_zk(),
                    &lookup_gadget,
                )
            });
        log_num_quotient_chunks.push(log_num_chunks);

        let (_, n_chunks) =
            checked_log_size_sum(log_num_chunks, config.is_zk()).ok_or_else(|| {
                InvalidProofShapeError::QuotientDomainTooLarge {
                    air: Some(i),
                    maximum: usize::BITS as usize - 1,
                    got: log_num_chunks.saturating_add(config.is_zk()),
                }
            })?;
        num_quotient_chunks.push(n_chunks);
    }

    // Soundness: bound LogUp multiplicities so none wraps modulo p.
    // Base degree bits give the same heights the prover used.
    let trace_heights: Vec<usize> = base_degree_bits.iter().map(|&b| 1usize << b).collect();
    check_multiplicity_height_bound(all_lookups, &trace_heights)?;

    for (i, air) in airs.iter().enumerate() {
        let air_width = A::width(air);
        let expected_public_values_len = air.num_public_values();
        let got_public_values_len = public_values[i].len();
        let inst_opened_vals = &opened_values.instances[i];
        let inst_base_opened_vals = &inst_opened_vals.base_opened_values;

        if got_public_values_len != expected_public_values_len {
            return Err(InvalidProofShapeError::PublicValuesLengthMismatch {
                expected: expected_public_values_len,
                got: got_public_values_len,
            }
            .into());
        }

        // Validate trace widths match the AIR
        if inst_base_opened_vals.trace_local.len() != air_width {
            return Err(InvalidProofShapeError::TraceLocalWidthMismatch {
                air: i,
                expected: air_width,
                got: inst_base_opened_vals.trace_local.len(),
            }
            .into());
        }
        validate_trace_next(air, inst_base_opened_vals, i)?;

        // Validate quotient chunks structure
        let n_chunks = num_quotient_chunks[i];
        if inst_base_opened_vals.quotient_chunks.len() != n_chunks {
            return Err(InvalidProofShapeError::QuotientChunksCountMismatch {
                air: i,
                expected: n_chunks,
                got: inst_base_opened_vals.quotient_chunks.len(),
            }
            .into());
        }

        for chunk in &inst_base_opened_vals.quotient_chunks {
            if chunk.len() != Challenge::<SC>::DIMENSION {
                return Err(
                    InvalidProofShapeError::QuotientChunkDimensionMismatch { air: i }.into(),
                );
            }
        }

        validate_random_opening_dimension::<SC, _>(inst_base_opened_vals)?;
        validate_preprocessed_opening(air, inst_base_opened_vals, preprocessed_widths[i], i)?;

        // Width of the two lookup permutation rows, under the single-terminal layout:
        //
        //     no lookups -> no permutation columns at all
        //     k lookups  -> one shared accumulator column plus one fraction column per lookup
        //
        // Each column holds extension-field elements, flattened into base-field coefficients.
        // The opened row is therefore that column count times the extension degree.
        let expected_perm_len = if all_lookups[i].is_empty() {
            0
        } else {
            (all_lookups[i].len() + 1) * Challenge::<SC>::DIMENSION
        };

        // Checked alongside every other proof-carried width.
        // A malformed shape is then rejected before any transcript work begins.
        if inst_opened_vals.permutation_local.len() != expected_perm_len
            || inst_opened_vals.permutation_next.len() != expected_perm_len
        {
            return Err(LookupError::PermutationWidthMismatch {
                air: i,
                expected: expected_perm_len,
            }
            .into());
        }

        // One terminal per AIR with lookups; none otherwise.
        let expected_present = !all_lookups[i].is_empty();
        let got_present = lookup_terminals[i].is_some();
        if expected_present != got_present {
            return Err(LookupError::TerminalPresenceMismatch {
                air: i,
                expected_present,
                got_present,
            }
            .into());
        }
    }

    // Validate the shape of the lookup commitment.
    let num_lookup_instances = all_lookups.iter().filter(|c| !c.is_empty()).count();

    if commitments.permutation.is_some() != (num_lookup_instances > 0) {
        return Err(LookupError::CommitmentMismatch.into());
    }

    // Lift the proof into the arithmetic, once.
    //
    // The shape checks above read the native proof, so every lifted value has a checked shape.
    let lifted_commitments = commitments.map(|commitment| arith.load_commitment(pcs, commitment));
    let preprocessed_commitment = common
        .preprocessed
        .as_ref()
        .map(|global| arith.load_key_commitment(pcs, &global.commitment));
    let lifted_opened = opened_values.map(|value| arith.load_ext(*value));
    let terminals: Vec<Option<AR::EF>> = lookup_terminals
        .iter()
        .map(|terminal| terminal.as_ref().map(|t| arith.load_ext(t.0)))
        .collect();

    // Announce every value the transcript absorbs, in absorb order.
    //
    // Degree bits are part of the statement, so they enter as constants.
    for &bits in degree_bits {
        arith.will_observe_ext(&AR::EF::from(SC::Challenge::from_usize(bits)));
    }
    arith.will_observe_commitment(pcs, &lifted_commitments.main);
    for values in public_values {
        arith.will_observe(values.clone());
    }
    if let Some(commitment) = &preprocessed_commitment {
        arith.will_observe_commitment(pcs, commitment);
    }
    if num_lookup_instances > 0
        && config.lookup_proof_of_work_bits() > 0
        && let Some(witness) = *lookup_pow_witness
    {
        let witness = arith.load(witness);
        arith.will_observe(vec![witness]);
    }
    if let Some(commitment) = &lifted_commitments.permutation {
        arith.will_observe_commitment(pcs, commitment);
    }
    for terminal in terminals.iter().flatten() {
        arith.will_observe_ext(terminal);
    }
    arith.will_observe_commitment(pcs, &lifted_commitments.quotient_chunks);
    if let Some(commitment) = &lifted_commitments.random {
        arith.will_observe_commitment(pcs, commitment);
    }
    if config.ood_proof_of_work_bits() > 0 {
        let witness = arith.load(*ood_pow_witness);
        arith.will_observe(vec![witness]);
    }

    // Transcript: describe the run, then seed a driver from that description.
    //
    // Every number here comes from the AIRs, the common data, or the config.
    // Every proof-carried count the description relies on was rejected above.
    let shape = BatchShape {
        trace_widths: airs.iter().map(A::width).collect(),
        public_value_counts: airs.iter().map(BaseAir::num_public_values).collect(),
        preprocessed_widths: preprocessed_widths.clone(),
        has_preprocessed_commitment: common.preprocessed.is_some(),
        num_lookup_instances,
        lookup_pow_bits: config.lookup_proof_of_work_bits(),
        has_randomization_commitment: SC::Pcs::ZK,
        ood_pow_bits: config.ood_proof_of_work_bits(),
    };

    let mut challenger = arith.challenger(config);
    let mut transcript = arith.seeding(|| {
        BatchVerifierTranscript::<AR::Challenger, Val<SC>, Challenge<SC>, Commitment<SC>>::new(
            &mut challenger,
            shape,
        )
    });

    // The size of each instance is the prover's to choose, so it is replayed.
    transcript.instance_bindings(degree_bits);

    let native_public_values: Vec<Vec<Val<SC>>> = public_values
        .iter()
        .map(|values| values.iter().map(|value| arith.value(value)).collect())
        .collect();
    transcript.main_phase(commitments.main.clone(), &native_public_values);
    transcript.preprocessed_phase(common.preprocessed.as_ref().map(|g| g.commitment.clone()));

    // Replay the grind guarding the lookup challenges, then redraw them.
    //
    // Why: the check runs before the first squeeze.
    // A bad witness is therefore rejected without the challenges ever being drawn.
    let lookup_pair = transcript.lookup_phase(*lookup_pow_witness)?;
    let challenges_per_instance = match lookup_pair {
        Some((alpha, beta)) => {
            let alpha = arith.sampled_ext(alpha);
            let beta = arith.sampled_ext(beta);
            lay_out_lookup_challenges(all_lookups, &lookup_gadget, alpha, beta)
        }
        None => all_lookups.iter().map(|_| Vec::new()).collect(),
    };

    let terminal_values: Vec<Challenge<SC>> =
        lookup_terminals.iter().flatten().map(|t| t.0).collect();
    let alpha = transcript.permutation_phase(commitments.permutation.clone(), &terminal_values);
    let alpha = arith.sampled_ext(alpha);

    transcript.quotient_phase(
        commitments.quotient_chunks.clone(),
        commitments.random.clone(),
    );

    // Replay the grind guarding the out-of-domain point, then redraw it.
    //
    // Why: the check runs before the sample.
    // A bad witness is therefore rejected without zeta ever being drawn.
    let zeta = transcript.ood_phase(*ood_pow_witness)?;
    let zeta = arith.sampled_ext(zeta);

    // Assembling the opening argument can still reject the proof's shape.
    //
    // The driver is mid-pattern here, and dropping it unfinished panics.
    // Releasing its completeness check first keeps the rejection the only failure.
    let opening_argument = commitments_with_opening_points(
        arith,
        config,
        airs,
        &zeta,
        &lifted_commitments,
        preprocessed_commitment.as_ref(),
        &lifted_opened,
        common,
        degree_bits,
        &preprocessed_widths,
        &log_num_quotient_chunks,
    );
    let (coms_to_verify, quotient_domains, preprocessed_index) = match opening_argument {
        Ok(argument) => argument,
        Err(err) => {
            transcript.abort();
            return Err(err);
        }
    };

    // Verify all openings via PCS, on the same sponge and under its own description.
    let opening_result = transcript.delegate(|challenger| {
        arith.verify_opening(
            pcs,
            coms_to_verify,
            opening_proof,
            challenger,
            preprocessed_index,
        )
    });

    // The bracket is closed either way, so the driver finishes before the rejection travels.
    transcript.finish();
    opening_result.map_err(VerificationError::InvalidOpeningArgument)?;

    // Un-extended trace domains, one per instance, for selectors and periodic columns.
    let trace_domains: Vec<Domain<SC>> = ext_domain_sizes
        .iter()
        .map(|&ext_size| pcs.natural_domain_for_degree(ext_size >> config.is_zk()))
        .collect();

    // Now check constraint equality per instance.
    // For each instance, recombine quotient from chunks at zeta and compare to folded constraints.
    for (i, air) in airs.iter().enumerate() {
        let _air_span = info_span!("verify constraints", air_idx = i).entered();
        let opened = &lifted_opened.instances[i];

        // Recompose quotient(zeta) from chunks.
        let quotient = arith.recompose_quotient(
            &quotient_domains[i],
            &opened.base_opened_values.quotient_chunks,
            &zeta,
        )?;

        // Recompose permutation openings into extension-field columns.
        //
        // Shapes:
        // - commitment           — base-flattened matrix, width = aux_width * DIMENSION.
        // - constraint evaluator — extension-field matrix, width = aux_width.
        //
        // The flattened openings hold one extension column per `DIMENSION` coefficients.
        let recompose = |flat: &[AR::EF]| -> Vec<AR::EF> {
            if all_lookups[i].is_empty() {
                return vec![];
            }
            flat.chunks_exact(Challenge::<SC>::DIMENSION)
                .map(recompose_ext::<AR>)
                .collect()
        };
        let perm_local_ext = recompose(&opened.permutation_local);
        let perm_next_ext = recompose(&opened.permutation_next);

        let trace_next = opened
            .base_opened_values
            .trace_next
            .clone()
            .unwrap_or_else(|| vec![AR::EF::ZERO; A::width(air)]);
        let preprocessed_next = opened
            .base_opened_values
            .preprocessed_next()
            .map_or_else(|| vec![AR::EF::ZERO; preprocessed_widths[i]], <[_]>::to_vec);
        let perm_vals: Vec<AR::EF> = terminals[i].iter().cloned().collect();

        // Periodic columns are AIR logic; a malformed one must error, not panic.
        let declared = air.periodic_columns();
        let periodic_columns = PeriodicColumns::new(&declared, trace_domains[i].size())?;
        let periodic_values = arith.periodic_values(&trace_domains[i], periodic_columns, &zeta);

        let selectors = arith
            .selectors(&trace_domains[i], &zeta)
            .ok_or(VerificationError::OodPointInDomain)?;
        let verifier_data = VerifierData {
            inputs: ConstraintInputs {
                trace_local: &opened.base_opened_values.trace_local,
                trace_next: &trace_next,
                preprocessed_local: opened
                    .base_opened_values
                    .preprocessed_local()
                    .unwrap_or(&[]),
                preprocessed_next: &preprocessed_next,
                periodic_values: &periodic_values,
                public_values: &public_values[i],
                selectors: &selectors,
                alpha: alpha.clone(),
            },
            permutation_local: &perm_local_ext,
            permutation_next: &perm_next_ext,
            permutation_challenges: &challenges_per_instance[i],
            permutation_values: &perm_vals,
            lookups: &all_lookups[i],
        };
        let folded = arith.fold_constraints(air, &verifier_data, &lookup_gadget);

        // Check that constraints(zeta) / Z_H(zeta) = quotient(zeta).
        arith.assert_zero(folded * selectors.inv_vanishing - quotient, || {
            VerificationError::OodEvaluationMismatch { index: Some(i) }
        })?;
    }

    // Single-terminal LogUp cross-AIR check.
    //
    // Each AIR with lookups commits one terminal: the sum of every per-row
    // rational contribution across its trace.
    //
    // Soundness:
    // - The total across the batch must be zero.
    // - Bus challenges are sampled after the main commitment, so any
    //   per-bus imbalance survives the collapse with overwhelming probability.
    let total: AR::EF = terminals.iter().flatten().cloned().sum();
    arith.assert_zero(total, || LookupError::TerminalSumNonZero)?;

    Ok(())
}
