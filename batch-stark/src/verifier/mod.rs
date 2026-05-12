mod data;

use alloc::string::String;
use alloc::vec::Vec;
use alloc::{format, vec};

pub use data::VerifierData;
use hashbrown::HashMap;
use p3_air::Air;
use p3_air::symbolic::{AirLayout, SymbolicExpressionExt};
use p3_commit::{Pcs, PolynomialSpace};
use p3_field::{Algebra, BasedVectorSpace, ExtensionField, PrimeCharacteristicRing};
use p3_lookup::folder::VerifierConstraintFolderWithLookups;
use p3_lookup::logup::LogUpGadget;
use p3_lookup::{InteractionSymbolicBuilder, Kind, LookupProtocol};
use p3_uni_stark::{
    InvalidProofShapeError, VerificationError, recompose_quotient_from_chunks, validate_degree_bits,
};
use p3_util::checked_log_size_sum;
use p3_util::zip_eq::zip_eq;
use tracing::{info_span, instrument};

use crate::common::CommonData;
use crate::config::{Challenge, Domain, PcsError, StarkGenericConfig as SGC, Val};
use crate::proof::BatchProof;
use crate::symbolic::get_log_num_quotient_chunks;
use crate::transcript::BatchTranscript;

#[instrument(skip_all)]
pub fn verify_batch<SC, A>(
    config: &SC,
    airs: &[A],
    proof: &BatchProof<SC>,
    public_values: &[Vec<Val<SC>>],
    common: &CommonData<SC>,
) -> Result<(), VerificationError<PcsError<SC>>>
where
    SC: SGC,
    SymbolicExpressionExt<Val<SC>, SC::Challenge>: Algebra<SC::Challenge>,
    A: Air<InteractionSymbolicBuilder<Val<SC>, SC::Challenge>>
        + for<'a> Air<VerifierConstraintFolderWithLookups<'a, SC>>,
    Challenge<SC>: BasedVectorSpace<Val<SC>>,
{
    // TODO: Extend if additional lookup gadgets are added.
    let lookup_gadget = LogUpGadget::new();

    let BatchProof {
        commitments,
        opened_values,
        opening_proof,
        global_lookup_data,
        degree_bits,
    } = proof;

    let all_lookups = &common.lookups;

    let pcs = config.pcs();
    let mut transcript = BatchTranscript::<SC>::new(config.initialise_challenger());

    // Sanity checks
    if airs.len() != opened_values.instances.len()
        || airs.len() != public_values.len()
        || airs.len() != degree_bits.len()
        || airs.len() != global_lookup_data.len()
        || airs.len() != all_lookups.len()
        || common
            .preprocessed
            .as_ref()
            .is_some_and(|global| global.instances.len() != airs.len())
    {
        return Err(InvalidProofShapeError::InstanceCountMismatch.into());
    }

    // Check that the random commitments are/are not present depending on the ZK setting.
    // - If ZK is enabled, the prover should have random commitments.
    // - If ZK is not enabled, the prover should not have random commitments.
    if (opened_values
        .instances
        .iter()
        .any(|ov| ov.base_opened_values.random.is_some() != SC::Pcs::ZK))
        || (commitments.random.is_some() != SC::Pcs::ZK)
    {
        return Err(VerificationError::RandomizationError);
    }

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
        let (base_db, ext_domain_size) =
            validate_degree_bits(Some(i), degree_bits[i], config.is_zk())?;
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
                get_log_num_quotient_chunks::<Val<SC>, SC::Challenge, A, LogUpGadget>(
                    air,
                    layout,
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

    // Observe the instance count up front to match the prover's transcript.
    transcript.observe_instance_count(airs.len());

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
        if !airs[i].main_next_row_columns().is_empty() {
            if inst_base_opened_vals
                .trace_next
                .as_ref()
                .is_none_or(|v| v.len() != air_width)
            {
                return Err(InvalidProofShapeError::TraceNextMismatch { air: i }.into());
            }
        } else if inst_base_opened_vals.trace_next.is_some() {
            return Err(InvalidProofShapeError::UnexpectedTraceNext { air: i }.into());
        }

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

        // Validate random commit
        if inst_opened_vals
            .base_opened_values
            .random
            .as_ref()
            .is_some_and(|r_comm| r_comm.len() != SC::Challenge::DIMENSION)
        {
            return Err(VerificationError::RandomizationError);
        }

        // Validate that any preprocessed width implied by CommonData matches the opened shapes.
        let pre_w = preprocessed_widths[i];
        let pre_local_len = inst_base_opened_vals
            .preprocessed_local
            .as_ref()
            .map_or(0, |v| v.len());
        let pre_next_len = inst_base_opened_vals
            .preprocessed_next
            .as_ref()
            .map_or(0, |v| v.len());
        if pre_w == 0 {
            if pre_local_len != 0 || pre_next_len != 0 {
                return Err(InvalidProofShapeError::UnexpectedPreprocessedValues { air: i }.into());
            }
        } else if !airs[i].preprocessed_next_row_columns().is_empty() {
            if pre_local_len != pre_w || pre_next_len != pre_w {
                return Err(InvalidProofShapeError::PreprocessedWidthMismatch { air: i }.into());
            }
        } else if pre_local_len != pre_w || pre_next_len != 0 {
            return Err(InvalidProofShapeError::PreprocessedWidthMismatch { air: i }.into());
        }

        let expected_global_lookup_entries: Vec<_> = all_lookups[i]
            .iter()
            .filter_map(|l| match &l.kind {
                Kind::Global(name) => Some((name, l.column)),
                Kind::Local => None,
            })
            .collect();
        let expected_global_lookup_data_len = expected_global_lookup_entries.len();
        let got_global_lookup_data_len = global_lookup_data[i].len();
        if got_global_lookup_data_len != expected_global_lookup_data_len {
            return Err(InvalidProofShapeError::GlobalLookupDataCountMismatch {
                air: i,
                expected: expected_global_lookup_data_len,
                got: got_global_lookup_data_len,
            }
            .into());
        }
        for (lookup_idx, ((expected_name, expected_aux_column), data)) in
            expected_global_lookup_entries
                .into_iter()
                .zip(global_lookup_data[i].iter())
                .enumerate()
        {
            if data.name != *expected_name || data.aux_column != expected_aux_column {
                return Err(InvalidProofShapeError::GlobalLookupDataMetadataMismatch {
                    air: i,
                    lookup: lookup_idx,
                    expected_name: expected_name.clone(),
                    got_name: data.name.clone(),
                    expected_aux_column,
                    got_aux_column: data.aux_column,
                }
                .into());
            }
        }

        // Observe per-instance binding data.
        let ext_db = degree_bits[i];
        let base_db = base_degree_bits[i];
        let width = air.width();
        let n_chunks = num_quotient_chunks[i];
        transcript.observe_instance_binding(ext_db, base_db, width, n_chunks);
    }

    // Observe main commitment and public values, then preprocessed data.
    transcript.observe_main(&commitments.main, public_values);
    transcript.observe_preprocessed(&preprocessed_widths, common.preprocessed.as_ref());

    // Validate the shape of the lookup commitment.
    let is_lookup = commitments.permutation.is_some();

    if is_lookup != all_lookups.iter().any(|c| !c.is_empty()) {
        return Err(InvalidProofShapeError::LookupCommitmentMismatch.into());
    }

    // Sample permutation challenges and alpha.
    let challenges_per_instance = transcript.sample_perm_challenges(all_lookups, &lookup_gadget);
    let alpha: Challenge<SC> = transcript
        .observe_perm_and_sample_alpha(commitments.permutation.as_ref(), global_lookup_data);

    // Observe quotient chunks and optional random commitment.
    transcript.observe_quotient_commitment(&commitments.quotient_chunks);
    if let Some(r_commit) = &commitments.random {
        transcript.observe_random_commitment(r_commit);
    }

    // Sample OOD point.
    let zeta = transcript.sample_zeta();

    // Build commitments_with_opening_points to verify openings.
    let mut coms_to_verify = vec![];

    // Trace round: per instance, open at zeta and zeta_next
    let (trace_domains, ext_trace_domains): (Vec<Domain<SC>>, Vec<Domain<SC>>) = ext_domain_sizes
        .iter()
        .map(|&ext_size| {
            (
                pcs.natural_domain_for_degree(ext_size >> config.is_zk()),
                pcs.natural_domain_for_degree(ext_size),
            )
        })
        .unzip();

    if let Some(random_commit) = &commitments.random {
        coms_to_verify.push((
            random_commit.clone(),
            ext_trace_domains
                .iter()
                .zip(opened_values.instances.iter())
                .map(|(domain, inst_opened_vals)| {
                    // We already checked that random is present for each instance when ZK is enabled.
                    let random_vals = inst_opened_vals.base_opened_values.random.as_ref().unwrap();
                    (*domain, vec![(zeta, random_vals.clone())])
                })
                .collect::<Vec<_>>(),
        ));
    }

    let trace_round: Vec<_> = ext_trace_domains
        .iter()
        .zip(opened_values.instances.iter())
        .enumerate()
        .map(|(i, (ext_dom, inst_opened_vals))| {
            let mut points = vec![(
                zeta,
                inst_opened_vals.base_opened_values.trace_local.clone(),
            )];
            if !airs[i].main_next_row_columns().is_empty() {
                let zeta_next = trace_domains[i]
                    .next_point(zeta)
                    .ok_or(VerificationError::NextPointUnavailable)?;
                points.push((
                    zeta_next,
                    inst_opened_vals
                        .base_opened_values
                        .trace_next
                        .clone()
                        .expect("checked in shape validation"),
                ));
            }
            Ok((*ext_dom, points))
        })
        .collect::<Result<Vec<_>, VerificationError<PcsError<SC>>>>()?;
    coms_to_verify.push((commitments.main.clone(), trace_round));

    // Quotient chunks round: flatten per-instance chunks to match commit order.
    // Use extended domains for the outer commit domain, with size = base_degree * num_quotient_chunks.
    let quotient_domains: Vec<Vec<Domain<SC>>> = (0..degree_bits.len())
        .map(|i| {
            let ext_db = degree_bits[i];
            let log_num_chunks = log_num_quotient_chunks[i];
            let n_chunks = num_quotient_chunks[i];
            let ext_dom = ext_trace_domains[i];
            let (_, quotient_domain_size) = checked_log_size_sum(ext_db, log_num_chunks)
                .ok_or_else(|| InvalidProofShapeError::QuotientDomainTooLarge {
                    air: Some(i),
                    maximum: usize::BITS as usize - 1,
                    got: ext_db.saturating_add(log_num_chunks),
                })?;
            let qdom = ext_dom.create_disjoint_domain(quotient_domain_size);
            Ok(qdom.split_domains(n_chunks))
        })
        .collect::<Result<Vec<_>, InvalidProofShapeError>>()?;

    // When ZK is enabled, the size of the quotient chunks' domains doubles.
    let randomized_quotient_chunks_domains = quotient_domains
        .iter()
        .map(|doms| {
            doms.iter()
                .map(|dom| pcs.natural_domain_for_degree(dom.size() << (config.is_zk())))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    // Build the per-matrix openings for the aggregated quotient commitment.
    let mut qc_round = Vec::new();
    for (i, domains) in randomized_quotient_chunks_domains.iter().enumerate() {
        let inst_qcs = &opened_values.instances[i]
            .base_opened_values
            .quotient_chunks;
        if inst_qcs.len() != domains.len() {
            return Err(InvalidProofShapeError::QuotientDomainsCountMismatch { air: i }.into());
        }
        for (d, vals) in zip_eq(
            domains.iter(),
            inst_qcs,
            VerificationError::from(InvalidProofShapeError::QuotientDomainsCountMismatch {
                air: i,
            }),
        )? {
            qc_round.push((*d, vec![(zeta, vals.clone())]));
        }
    }
    coms_to_verify.push((commitments.quotient_chunks.clone(), qc_round));

    // Preprocessed rounds: a single global commitment with one matrix per
    // instance that has preprocessed columns.
    if let Some(global) = &common.preprocessed {
        let mut pre_round = Vec::new();

        for (matrix_index, &inst_idx) in global.matrix_to_instance.iter().enumerate() {
            let pre_w = preprocessed_widths[inst_idx];
            if pre_w == 0 {
                return Err(
                    InvalidProofShapeError::PreprocessedMetadataMismatch { air: inst_idx }.into(),
                );
            }

            let inst = &opened_values.instances[inst_idx];
            let local = inst
                .base_opened_values
                .preprocessed_local
                .as_ref()
                .ok_or_else(|| {
                    VerificationError::from(InvalidProofShapeError::MissingPreprocessedValues {
                        air: inst_idx,
                    })
                })?;

            // Validate that the preprocessed data's extended degree matches what we expect.
            let ext_db = degree_bits[inst_idx];

            let meta = global.instances[inst_idx].as_ref().ok_or_else(|| {
                VerificationError::from(InvalidProofShapeError::PreprocessedMetadataMismatch {
                    air: inst_idx,
                })
            })?;
            if meta.matrix_index != matrix_index || meta.degree_bits != ext_db {
                return Err(
                    InvalidProofShapeError::PreprocessedMetadataMismatch { air: inst_idx }.into(),
                );
            }

            let meta_db = meta.degree_bits;
            let pre_domain = pcs.natural_domain_for_degree(1 << meta_db);
            if !airs[inst_idx].preprocessed_next_row_columns().is_empty() {
                let next = inst
                    .base_opened_values
                    .preprocessed_next
                    .as_ref()
                    .ok_or_else(|| {
                        VerificationError::from(InvalidProofShapeError::MissingPreprocessedValues {
                            air: inst_idx,
                        })
                    })?;
                let zeta_next_i = trace_domains[inst_idx]
                    .next_point(zeta)
                    .ok_or(VerificationError::NextPointUnavailable)?;

                pre_round.push((
                    pre_domain,
                    vec![(zeta, local.clone()), (zeta_next_i, next.clone())],
                ));
            } else {
                pre_round.push((pre_domain, vec![(zeta, local.clone())]));
            }
        }

        coms_to_verify.push((global.commitment.clone(), pre_round));
    }

    if is_lookup {
        let permutation_commit = commitments.permutation.clone().unwrap();
        let mut permutation_round = Vec::new();
        for (i, (ext_dom, inst_opened_vals)) in ext_trace_domains
            .iter()
            .zip(opened_values.instances.iter())
            .enumerate()
        {
            if inst_opened_vals.permutation_local.len() != inst_opened_vals.permutation_next.len() {
                return Err(InvalidProofShapeError::PermutationLengthMismatch { air: i }.into());
            }
            if !inst_opened_vals.permutation_local.is_empty() {
                let zeta_next = trace_domains[i]
                    .next_point(zeta)
                    .ok_or(VerificationError::NextPointUnavailable)?;
                permutation_round.push((
                    *ext_dom,
                    vec![
                        (zeta, inst_opened_vals.permutation_local.clone()),
                        (zeta_next, inst_opened_vals.permutation_next.clone()),
                    ],
                ));
            }
        }
        coms_to_verify.push((permutation_commit, permutation_round));
    }

    // Verify all openings via PCS.
    pcs.verify(coms_to_verify, opening_proof, &mut transcript.challenger)
        .map_err(VerificationError::InvalidOpeningArgument)?;

    // Now check constraint equality per instance.
    // For each instance, recombine quotient from chunks at zeta and compare to folded constraints.
    for (i, air) in airs.iter().enumerate() {
        let _air_span = info_span!("verify constraints", air_idx = i).entered();

        let qc_domains = &quotient_domains[i];

        // Recompose quotient(zeta) from chunks using utility function.
        let quotient = recompose_quotient_from_chunks::<SC>(
            qc_domains,
            &opened_values.instances[i]
                .base_opened_values
                .quotient_chunks,
            zeta,
        );

        // Recompose permutation openings from base-flattened columns into extension field columns.
        // The permutation commitment is a base-flattened matrix with `width = aux_width * DIMENSION`.
        // For constraint evaluation, we need an extension field matrix with width `aux_width``.
        let aux_width = all_lookups[i]
            .iter()
            .map(|ctx| ctx.column)
            .max()
            .map(|m| m + 1)
            .unwrap_or(0);

        let ext_degree = Challenge::<SC>::DIMENSION;
        let expected_perm_len = aux_width * ext_degree;
        if opened_values.instances[i].permutation_local.len() != expected_perm_len
            || opened_values.instances[i].permutation_next.len() != expected_perm_len
        {
            return Err(InvalidProofShapeError::PermutationWidthMismatch {
                air: i,
                expected: expected_perm_len,
            }
            .into());
        }

        let recompose = |flat: &[Challenge<SC>]| -> Vec<Challenge<SC>> {
            if aux_width == 0 {
                return vec![];
            }
            // Each `ext_degree`-chunk holds the basis coefficients (in EF) of one EF element.
            // chunks_exact yields chunks of exactly `ext_degree` = DIMENSION, so the unwrap
            // below cannot panic.
            flat.chunks_exact(ext_degree)
                .map(|chunk| {
                    Challenge::<SC>::from_ext_basis_coefficients(chunk)
                        .expect("chunk length matches DIMENSION by construction")
                })
                .collect()
        };

        let perm_local_ext = recompose(&opened_values.instances[i].permutation_local);
        let perm_next_ext = recompose(&opened_values.instances[i].permutation_next);

        // Verify constraints at zeta using utility function.
        let init_trace_domain = trace_domains[i];
        let trace_next_zeros;
        let trace_next_ref = match &opened_values.instances[i].base_opened_values.trace_next {
            Some(v) => v.as_slice(),
            None => {
                trace_next_zeros = SC::Challenge::zero_vec(A::width(air));
                &trace_next_zeros
            }
        };
        let pre_next_zeros;
        let pre_next_ref = match &opened_values.instances[i]
            .base_opened_values
            .preprocessed_next
        {
            Some(v) => v.as_slice(),
            None => {
                pre_next_zeros = SC::Challenge::zero_vec(preprocessed_widths[i]);
                &pre_next_zeros
            }
        };
        let perm_vals: Vec<SC::Challenge> = global_lookup_data[i]
            .iter()
            .map(|ld| ld.cumulative_sum)
            .collect();
        let periodic_values: Vec<Challenge<SC>> = air
            .periodic_columns()
            .iter()
            .map(|col| trace_domains[i].evaluate_periodic_column_at(col, zeta))
            .collect();
        let verifier_data = VerifierData {
            trace_local: &opened_values.instances[i].base_opened_values.trace_local,
            trace_next: trace_next_ref,
            preprocessed_local: opened_values.instances[i]
                .base_opened_values
                .preprocessed_local
                .as_ref()
                .map_or(&[], |v| v),
            preprocessed_next: pre_next_ref,
            permutation_local: &perm_local_ext,
            permutation_next: &perm_next_ext,
            permutation_challenges: &challenges_per_instance[i],
            permutation_values: &perm_vals,
            periodic_values: &periodic_values,
            lookups: &all_lookups[i],
            public_values: &public_values[i],
            trace_domain: init_trace_domain,
            zeta,
            alpha,
            quotient,
        };

        verifier_data
            .verify_constraints_with_lookups::<A, LogUpGadget, PcsError<SC>>(air, &lookup_gadget)
            .map_err(|e| match e {
                VerificationError::OodEvaluationMismatch { .. } => {
                    VerificationError::OodEvaluationMismatch { index: Some(i) }
                }
                other => other,
            })?;
    }

    let mut global_cumulative = HashMap::<&String, Vec<_>>::new();
    for (lookups, data_for_instance) in all_lookups.iter().zip(global_lookup_data.iter()) {
        let global_lookups = lookups.iter().filter(|l| matches!(l.kind, Kind::Global(_)));
        debug_assert_eq!(global_lookups.clone().count(), data_for_instance.len());
        for (lookup, data) in global_lookups.zip(data_for_instance.iter()) {
            let name = match &lookup.kind {
                Kind::Global(n) => n,
                Kind::Local => unreachable!(),
            };
            global_cumulative
                .entry(name)
                .or_default()
                .push(data.cumulative_sum);
        }
    }

    for (name, all_expected_cumulative) in global_cumulative {
        lookup_gadget
            .verify_global_sum(&all_expected_cumulative)
            .map_err(|e| VerificationError::LookupError(format!("{e:?}: {name}")))?;
    }

    Ok(())
}
