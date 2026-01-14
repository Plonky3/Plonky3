use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt::Debug;

use hashbrown::HashMap;
use p3_air::Air;
use p3_challenger::{CanObserve, FieldChallenger};
use p3_commit::{Pcs, PolynomialSpace};
use p3_field::{BasedVectorSpace, PrimeCharacteristicRing};
use p3_lookup::folder::VerifierConstraintFolderWithLookups;
use p3_lookup::logup::LogUpGadget;
use p3_lookup::lookup_traits::{
    Lookup, LookupData, LookupError, LookupGadget, lookup_data_to_expr,
};
use p3_matrix::dense::RowMajorMatrixView;
use p3_matrix::stack::VerticalPair;
use p3_uni_stark::{
    SymbolicAirBuilder, SymbolicExpression, VerificationError, VerifierConstraintFolder,
    recompose_quotient_from_chunks,
};
use p3_util::zip_eq::zip_eq;
use tracing::instrument;

use crate::common::{CommonData, get_perm_challenges};
use crate::config::{
    Challenge, Domain, PcsError, StarkGenericConfig as SGC, Val, observe_instance_binding,
};
use crate::proof::BatchProof;
use crate::symbolic::get_log_num_quotient_chunks;

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
    SymbolicExpression<SC::Challenge>: From<SymbolicExpression<Val<SC>>>,
    A: Air<SymbolicAirBuilder<Val<SC>, SC::Challenge>>
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
    let mut challenger = config.initialise_challenger();

    // Sanity checks
    if airs.len() != opened_values.instances.len()
        || airs.len() != public_values.len()
        || airs.len() != degree_bits.len()
        || airs.len() != global_lookup_data.len()
    {
        return Err(VerificationError::InvalidProofShape);
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

    // Observe the number of instances up front to match the prover's transcript.
    let n_instances = airs.len();
    challenger.observe_base_as_algebra_element::<Challenge<SC>>(Val::<SC>::from_usize(n_instances));

    // Validate opened values shape per instance and observe per-instance binding data.
    // Precompute per-instance preprocessed widths and number of quotient chunks.
    let mut preprocessed_widths = Vec::with_capacity(airs.len());
    // Number of quotient chunks per instance before ZK randomization.
    let mut log_num_quotient_chunks = Vec::with_capacity(airs.len());
    // The total number of quotient chunks, including ZK randomization.
    let mut num_quotient_chunks = Vec::with_capacity(airs.len());

    for (i, air) in airs.iter().enumerate() {
        let pre_w = common
            .preprocessed
            .as_ref()
            .and_then(|g| g.instances[i].as_ref().map(|m| m.width))
            .unwrap_or(0);
        preprocessed_widths.push(pre_w);

        let log_num_chunks = get_log_num_quotient_chunks::<Val<SC>, SC::Challenge, A, LogUpGadget>(
            air,
            pre_w,
            public_values[i].len(),
            &all_lookups[i],
            &lookup_data_to_expr(&global_lookup_data[i]),
            config.is_zk(),
            &lookup_gadget,
        );
        log_num_quotient_chunks.push(log_num_chunks);

        let n_chunks = 1 << (log_num_chunks + config.is_zk());
        num_quotient_chunks.push(n_chunks);
    }

    for (i, air) in airs.iter().enumerate() {
        let air_width = A::width(air);
        let inst_opened_vals = &opened_values.instances[i];
        let inst_base_opened_vals = &inst_opened_vals.base_opened_values;

        // Validate trace widths match the AIR
        if inst_base_opened_vals.trace_local.len() != air_width
            || inst_base_opened_vals.trace_next.len() != air_width
        {
            return Err(VerificationError::InvalidProofShape);
        }

        // Validate quotient chunks structure
        let n_chunks = num_quotient_chunks[i];
        if inst_base_opened_vals.quotient_chunks.len() != n_chunks {
            return Err(VerificationError::InvalidProofShape);
        }

        for chunk in &inst_base_opened_vals.quotient_chunks {
            if chunk.len() != Challenge::<SC>::DIMENSION {
                return Err(VerificationError::InvalidProofShape);
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
        if pre_w != pre_local_len || pre_w != pre_next_len {
            return Err(VerificationError::InvalidProofShape);
        }

        // Observe per-instance binding data: (log_ext_degree, log_degree), width, num quotient chunks.
        let ext_db = degree_bits[i];
        let base_db = ext_db - config.is_zk();
        let width = A::width(air);
        observe_instance_binding::<SC>(&mut challenger, ext_db, base_db, width, n_chunks);
    }

    // Observe main commitment and public values (in instance order).
    challenger.observe(commitments.main.clone());
    for pv in public_values {
        challenger.observe_slice(pv);
    }

    // Observe preprocessed widths for each instance. If a global
    // preprocessed commitment exists, observe it once.
    for &pre_w in preprocessed_widths.iter() {
        challenger.observe_base_as_algebra_element::<Challenge<SC>>(Val::<SC>::from_usize(pre_w));
    }
    if let Some(global) = &common.preprocessed {
        challenger.observe(global.commitment.clone());
    }

    // Validate the shape of the lookup commitment.
    let is_lookup = commitments.permutation.is_some();

    if is_lookup != all_lookups.iter().any(|c| !c.is_empty()) {
        return Err(VerificationError::InvalidProofShape);
    }

    // Fetch lookups and sample their challenges.
    let challenges_per_instance =
        get_perm_challenges::<SC, LogUpGadget>(&mut challenger, all_lookups, &lookup_gadget);

    // Then, observe the permutation tables, if any.
    if is_lookup {
        challenger.observe(
            commitments
                .permutation
                .clone()
                .expect("We checked that the commitment exists"),
        );
    }

    // Sample alpha for constraint folding
    let alpha = challenger.sample_algebra_element();

    // Observe quotient chunks commitment
    challenger.observe(commitments.quotient_chunks.clone());

    // We've already checked that commitments.random is present if and only if ZK is enabled.
    // Observe the random commitment if it is present.
    if let Some(r_commit) = commitments.random.clone() {
        challenger.observe(r_commit);
    }

    // Sample OOD point
    let zeta = challenger.sample_algebra_element();

    // Build commitments_with_opening_points to verify openings.
    let mut coms_to_verify = vec![];

    // Trace round: per instance, open at zeta and zeta_next
    let (trace_domains, ext_trace_domains): (Vec<Domain<SC>>, Vec<Domain<SC>>) = degree_bits
        .iter()
        .map(|&ext_db| {
            let base_db = ext_db - config.is_zk();
            (
                pcs.natural_domain_for_degree(1 << base_db),
                pcs.natural_domain_for_degree(1 << ext_db),
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
            let zeta_next = trace_domains[i]
                .next_point(zeta)
                .ok_or(VerificationError::NextPointUnavailable)?;

            Ok((
                *ext_dom,
                vec![
                    (
                        zeta,
                        inst_opened_vals.base_opened_values.trace_local.clone(),
                    ),
                    (
                        zeta_next,
                        inst_opened_vals.base_opened_values.trace_next.clone(),
                    ),
                ],
            ))
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
            let qdom = ext_dom.create_disjoint_domain(1 << (ext_db + log_num_chunks));
            qdom.split_domains(n_chunks)
        })
        .collect();

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
            return Err(VerificationError::InvalidProofShape);
        }
        for (d, vals) in zip_eq(
            domains.iter(),
            inst_qcs,
            VerificationError::InvalidProofShape,
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
                return Err(VerificationError::InvalidProofShape);
            }

            let inst = &opened_values.instances[inst_idx];
            let local = inst
                .base_opened_values
                .preprocessed_local
                .as_ref()
                .ok_or(VerificationError::InvalidProofShape)?;
            let next = inst
                .base_opened_values
                .preprocessed_next
                .as_ref()
                .ok_or(VerificationError::InvalidProofShape)?;

            // Validate that the preprocessed data's base degree matches what we expect.
            let ext_db = degree_bits[inst_idx];

            let meta = global.instances[inst_idx]
                .as_ref()
                .ok_or(VerificationError::InvalidProofShape)?;
            if meta.matrix_index != matrix_index || meta.degree_bits != ext_db {
                return Err(VerificationError::InvalidProofShape);
            }

            let meta_db = meta.degree_bits;
            let pre_domain = pcs.natural_domain_for_degree(1 << meta_db);
            let zeta_next_i = trace_domains[inst_idx]
                .next_point(zeta)
                .ok_or(VerificationError::NextPointUnavailable)?;

            pre_round.push((
                pre_domain,
                vec![(zeta, local.clone()), (zeta_next_i, next.clone())],
            ));
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
                return Err(VerificationError::InvalidProofShape);
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
    pcs.verify(coms_to_verify, opening_proof, &mut challenger)
        .map_err(VerificationError::InvalidOpeningArgument)?;

    // Now check constraint equality per instance.
    // For each instance, recombine quotient from chunks at zeta and compare to folded constraints.
    for (i, air) in airs.iter().enumerate() {
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
            .flat_map(|ctx| ctx.columns.iter().cloned())
            .max()
            .map(|m| m + 1)
            .unwrap_or(0);

        let recompose = |flat: &[Challenge<SC>]| -> Vec<Challenge<SC>> {
            if aux_width == 0 {
                return vec![];
            }
            let ext_degree = Challenge::<SC>::DIMENSION;
            assert!(
                flat.len() == aux_width * ext_degree,
                "flattened permutation opening length ({}) must equal aux_width ({}) * DIMENSION ({})",
                flat.len(),
                aux_width,
                ext_degree
            );
            // Chunk the flattened coefficients into groups of size `dim`.
            // Each chunk represents the coefficients of one extension field element.
            flat.chunks_exact(ext_degree)
                .map(|coeffs| {
                    // Dot product: sum(coeff_j * basis_j)
                    coeffs
                        .iter()
                        .enumerate()
                        .map(|(j, &coeff)| {
                            coeff
                                * Challenge::<SC>::ith_basis_element(j)
                                    .expect("Basis element should exist")
                        })
                        .sum()
                })
                .collect()
        };

        let perm_local_ext = recompose(&opened_values.instances[i].permutation_local);
        let perm_next_ext = recompose(&opened_values.instances[i].permutation_next);

        // Verify constraints at zeta using utility function.
        let init_trace_domain = trace_domains[i];
        let verifier_data = VerifierData {
            trace_local: &opened_values.instances[i].base_opened_values.trace_local,
            trace_next: &opened_values.instances[i].base_opened_values.trace_next,
            preprocessed_local: opened_values.instances[i]
                .base_opened_values
                .preprocessed_local
                .as_ref()
                .map_or(&[], |v| v),
            preprocessed_next: opened_values.instances[i]
                .base_opened_values
                .preprocessed_next
                .as_ref()
                .map_or(&[], |v| v),
            permutation_local: &perm_local_ext,
            permutation_next: &perm_next_ext,
            permutation_challenges: &challenges_per_instance[i],
            lookup_data: &proof.global_lookup_data[i],
            lookups: &all_lookups[i],
            public_values: &public_values[i],
            trace_domain: init_trace_domain,
            zeta,
            alpha,
            quotient,
        };

        verify_constraints_with_lookups::<SC, A, LogUpGadget, PcsError<SC>>(
            air,
            &verifier_data,
            &lookup_gadget,
        )
        .map_err(|e| match e {
            VerificationError::OodEvaluationMismatch { .. } => {
                VerificationError::OodEvaluationMismatch { index: Some(i) }
            }
            other => other,
        })?;
    }

    let mut global_cumulative = HashMap::<&String, Vec<_>>::new();
    for data in global_lookup_data.iter().flatten() {
        global_cumulative
            .entry(&data.name)
            .or_default()
            .push(data.expected_cumulated);
    }

    for (name, all_expected_cumulative) in global_cumulative {
        lookup_gadget
            .verify_global_final_value(&all_expected_cumulative)
            .map_err(|_| {
                VerificationError::LookupError(LookupError::GlobalCumulativeMismatch(Some(
                    name.clone(),
                )))
            })?;
    }

    Ok(())
}

/// Structure storing all data needed for verifying one instance's constraints at the out-of-domain point.
pub struct VerifierData<'a, SC: SGC> {
    // Out-of-domain point at which constraints are evaluated
    zeta: SC::Challenge,
    // Challenge used to fold constraints
    alpha: SC::Challenge,
    // Main trace evaluated at `zeta`
    trace_local: &'a [SC::Challenge],
    // Main trace evaluated at the following point `g * zeta`, where `g` is the subgroup generator
    trace_next: &'a [SC::Challenge],
    // Preprocessed trace evaluated at `zeta`
    preprocessed_local: &'a [SC::Challenge],
    // Preprocessed trace evaluated at the following point `g * zeta`, where `g` is the subgroup generator
    preprocessed_next: &'a [SC::Challenge],
    // Permutation trace evaluated at `zeta`
    permutation_local: &'a [SC::Challenge],
    // Permutation trace evaluated at the following point `g * zeta`, where `g` is the subgroup generator
    permutation_next: &'a [SC::Challenge],
    // Challenges used for the lookup argument
    permutation_challenges: &'a [SC::Challenge],
    // Lookup data needed for global lookup verification
    lookup_data: &'a [LookupData<SC::Challenge>],
    // Lookup contexts for this instance
    lookups: &'a [Lookup<Val<SC>>],
    // Public values for this instance
    public_values: &'a [Val<SC>],
    // Trace domain for this instance
    trace_domain: Domain<SC>,
    // Quotient polynomial evaluated at `zeta`
    quotient: SC::Challenge,
}

/// Verifies that the folded constraints match the quotient polynomial at zeta.
///
/// This evaluates the AIR constraints at the out-of-domain point and checks
/// that constraints(zeta) / Z_H(zeta) = quotient(zeta).
#[allow(clippy::too_many_arguments)]
pub fn verify_constraints_with_lookups<'a, SC, A, LG: LookupGadget, PcsErr: Debug>(
    air: &A,
    verifier_data: &VerifierData<'a, SC>,
    lookup_gadget: &LG,
) -> Result<(), VerificationError<PcsErr>>
where
    SC: SGC,
    A: for<'b> Air<VerifierConstraintFolderWithLookups<'b, SC>>,
{
    let VerifierData {
        trace_local,
        trace_next,
        preprocessed_local,
        preprocessed_next,
        permutation_local,
        permutation_next,
        permutation_challenges,
        lookup_data,
        lookups,
        public_values,
        trace_domain,
        zeta,
        alpha,
        quotient,
    } = verifier_data;

    let sels = trace_domain.selectors_at_point(*zeta);

    let main = VerticalPair::new(
        RowMajorMatrixView::new_row(trace_local),
        RowMajorMatrixView::new_row(trace_next),
    );

    let preprocessed = VerticalPair::new(
        RowMajorMatrixView::new_row(preprocessed_local),
        RowMajorMatrixView::new_row(preprocessed_next),
    );

    let inner_folder = VerifierConstraintFolder {
        main,
        preprocessed: if preprocessed_local.is_empty() {
            None
        } else {
            Some(preprocessed)
        },
        public_values,
        is_first_row: sels.is_first_row,
        is_last_row: sels.is_last_row,
        is_transition: sels.is_transition,
        alpha: *alpha,
        accumulator: SC::Challenge::ZERO,
        periodic_values: vec![], // batch-stark doesn't support periodic columns yet
    };
    let mut folder = VerifierConstraintFolderWithLookups {
        inner: inner_folder,
        permutation: VerticalPair::new(
            RowMajorMatrixView::new_row(permutation_local),
            RowMajorMatrixView::new_row(permutation_next),
        ),
        permutation_challenges,
    };
    // Evaluate AIR and lookup constraints.
    A::eval_with_lookups(air, &mut folder, lookups, lookup_data, lookup_gadget);
    let folded_constraints = folder.inner.accumulator;

    // Check that constraints(zeta) / Z_H(zeta) = quotient(zeta)
    if folded_constraints * sels.inv_vanishing != *quotient {
        return Err(VerificationError::OodEvaluationMismatch { index: None });
    }

    Ok(())
}
