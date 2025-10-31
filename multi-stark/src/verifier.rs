use alloc::vec;
use alloc::vec::Vec;

use hashbrown::HashMap;
use itertools::Itertools;
use p3_air::Air;
use p3_challenger::{CanObserve, FieldChallenger};
use p3_commit::{Pcs, PolynomialSpace};
use p3_field::{BasedVectorSpace, PrimeCharacteristicRing};
use p3_lookup::folders::VerifierConstraintFolderWithLookups;
use p3_lookup::lookup_traits::{
    AirLookupHandler, AirNoLookup, EmptyLookupGadget, Kind, Lookup, LookupData, LookupGadget,
};
use p3_matrix::dense::RowMajorMatrixView;
use p3_matrix::stack::VerticalPair;
use p3_uni_stark::{
    SymbolicAirBuilder, SymbolicExpression, VerificationError, VerifierConstraintFolder,
    recompose_quotient_from_chunks,
};
use p3_util::zip_eq::zip_eq;
use tracing::instrument;

use crate::config::{
    Challenge, Domain, PcsError, StarkGenericConfig as SGC, Val, observe_base_as_ext,
    observe_instance_binding,
};
use crate::proof::MultiProof;
use crate::symbolic::get_log_quotient_degree;

#[instrument(skip_all)]
pub fn verify_multi<SC, A, LG>(
    config: &SC,
    airs: &mut [A],
    proof: &MultiProof<SC>,
    public_values: &[Vec<Val<SC>>],
    lookup_gadget: &LG,
) -> Result<(), VerificationError<PcsError<SC>>>
where
    SC: SGC,
    SymbolicExpression<SC::Challenge>: From<SymbolicExpression<Val<SC>>>,
    A: AirLookupHandler<SymbolicAirBuilder<Val<SC>, SC::Challenge>>
        + for<'a> AirLookupHandler<VerifierConstraintFolderWithLookups<'a, SC>>,
    Challenge<SC>: BasedVectorSpace<Val<SC>>,
    LG: LookupGadget,
{
    let MultiProof {
        commitments,
        opened_values,
        opening_proof,
        global_lookup_data,
        degree_bits,
    } = proof;

    let pcs = config.pcs();
    let mut challenger = config.initialise_challenger();

    // ZK mode is not supported yet
    if config.is_zk() != 0 {
        panic!("p3-multi-stark: ZK mode is not supported yet");
    }

    // ZK mode is not supported yet
    if config.is_zk() != 0 {
        panic!("p3-multi-stark: ZK mode is not supported yet");
    }

    // Sanity checks
    if airs.len() != opened_values.instances.len()
        || airs.len() != public_values.len()
        || airs.len() != degree_bits.len()
        || airs.len() != global_lookup_data.len()
    {
        return Err(VerificationError::InvalidProofShape);
    }

    // Observe the number of instances up front to match the prover's transcript.
    let n_instances = airs.len();
    observe_base_as_ext::<SC>(&mut challenger, Val::<SC>::from_usize(n_instances));

    let all_lookups = airs
        .iter_mut()
        .map(|air| {
            <A as AirLookupHandler<VerifierConstraintFolderWithLookups<'_, SC>>>::get_lookups(air)
        })
        .collect::<Vec<_>>();

    // Validate opened values shape per instance and observe per-instance binding data.
    // Precompute per-instance log_quotient_degrees and quotient_degrees in one pass.
    let (log_quotient_degrees, quotient_degrees): (Vec<usize>, Vec<usize>) = airs
        .iter()
        .zip_eq(public_values.iter())
        .zip_eq(all_lookups.iter())
        .zip_eq(global_lookup_data.iter())
        .map(|(((air, pv), contexts), lookup_data)| {
            let lqd = get_log_quotient_degree::<Val<SC>, SC::Challenge, A, LG>(
                air,
                0,
                pv.len(),
                contexts,
                lookup_data,
                config.is_zk(),
                lookup_gadget,
            );
            let qd = 1 << (lqd + config.is_zk());
            (lqd, qd)
        })
        .unzip();

    for (i, air) in airs.iter().enumerate() {
        let air_width = A::width(air);
        let inst_opened_vals = &opened_values.instances[i];

        // Validate trace widths match the AIR
        if inst_opened_vals.base_opened_values.trace_local.len() != air_width
            || inst_opened_vals.base_opened_values.trace_next.len() != air_width
        {
            return Err(VerificationError::InvalidProofShape);
        }

        // Validate quotient chunks structure
        let quotient_degree = quotient_degrees[i];
        if inst_opened_vals.base_opened_values.quotient_chunks.len() != quotient_degree {
            return Err(VerificationError::InvalidProofShape);
        }

        for chunk in &inst_opened_vals.base_opened_values.quotient_chunks {
            if chunk.len() != Challenge::<SC>::DIMENSION {
                return Err(VerificationError::InvalidProofShape);
            }
        }

        // Observe per-instance binding data: (log_ext_degree, log_degree), width, num quotient chunks.
        let ext_db = degree_bits[i];
        let base_db = ext_db - config.is_zk();
        let width = A::width(air);
        observe_instance_binding::<SC>(&mut challenger, ext_db, base_db, width, quotient_degree);
    }

    // Observe main commitment and public values (in instance order).
    challenger.observe(commitments.main.clone());
    for pv in public_values {
        challenger.observe_slice(pv);
    }

    // Validate the shape of the lookup commitment.
    let is_lookup = commitments.permutation.is_some();

    if is_lookup != all_lookups.iter().any(|c| !c.is_empty()) {
        return Err(VerificationError::InvalidProofShape);
    }

    // Fetch lookups and sample their challenges.
    let mut global_perm_challenges = HashMap::new();
    let mut challenges_per_instance = Vec::with_capacity(airs.len());
    for contexts in &all_lookups {
        let num_challenges = contexts.len() * lookup_gadget.num_challenges();
        let mut instance_challenges = Vec::with_capacity(num_challenges);
        for context in contexts {
            let cs = match &context.kind {
                Kind::Global(name) => {
                    let cs = global_perm_challenges.entry(name).or_insert_with(|| {
                        vec![
                            challenger.sample_algebra_element::<Challenge<SC>>(),
                            challenger.sample_algebra_element(),
                        ]
                    });
                    cs.clone()
                }
                Kind::Local => {
                    vec![
                        challenger.sample_algebra_element(),
                        challenger.sample_algebra_element(),
                    ]
                }
            };
            instance_challenges.extend(cs);
        }
        challenges_per_instance.push(instance_challenges);
    }

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
    let trace_round: Vec<_> = ext_trace_domains
        .iter()
        .zip(opened_values.instances.iter())
        .map(|(ext_dom, inst_opened_vals)| {
            let zeta_next = ext_dom
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

    if is_lookup {
        let permutation_commit = commitments.permutation.clone().unwrap();
        let mut permutation_round = Vec::new();
        for (ext_dom, inst_opened_vals) in
            ext_trace_domains.iter().zip(opened_values.instances.iter())
        {
            if inst_opened_vals.permutation_local.len() != inst_opened_vals.permutation_next.len() {
                return Err(VerificationError::InvalidProofShape);
            }
            if !inst_opened_vals.permutation_local.is_empty() {
                let zeta_next = ext_dom
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
        coms_to_verify.push((permutation_commit.clone(), permutation_round));
    }

    // Quotient chunks round: flatten per-instance chunks to match commit order.
    // Use extended domains for the outer commit domain, with size 2^(base_db + lqd + zk), and split into 2^(lqd+zk) chunks.
    let quotient_domains: Vec<Vec<Domain<SC>>> = (0..degree_bits.len())
        .map(|i| {
            let ext_db = degree_bits[i];
            let base_db = ext_db - config.is_zk();
            let lqd = log_quotient_degrees[i];
            let quotient_degree = quotient_degrees[i];
            let ext_dom = ext_trace_domains[i];
            let qdom = ext_dom.create_disjoint_domain(1 << (base_db + lqd + config.is_zk()));
            qdom.split_domains(quotient_degree)
        })
        .collect();

    // Build the per-matrix openings for the aggregated quotient commitment.
    let mut qc_round = Vec::new();
    for (i, domains) in quotient_domains.iter().enumerate() {
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

        // Verify constraints at zeta using utility function.
        let init_trace_domain = trace_domains[i];
        verify_constraints_with_lookups::<SC, A, LG, PcsError<SC>>(
            air,
            &opened_values.instances[i].base_opened_values.trace_local,
            &opened_values.instances[i].base_opened_values.trace_next,
            &opened_values.instances[i].permutation_local,
            &opened_values.instances[i].permutation_next,
            &challenges_per_instance[i],
            &proof.global_lookup_data[i],
            &all_lookups[i],
            lookup_gadget,
            &public_values[i],
            init_trace_domain,
            zeta,
            alpha,
            quotient,
        )
        .map_err(|e| match e {
            VerificationError::OodEvaluationMismatch { .. } => {
                VerificationError::OodEvaluationMismatch { index: Some(i) }
            }
            other => other,
        })?;
    }

    Ok(())
}

/// Verifies that the folded constraints match the quotient polynomial at zeta.
///
/// This evaluates the AIR constraints at the out-of-domain point and checks
/// that constraints(zeta) / Z_H(zeta) = quotient(zeta).
#[allow(clippy::too_many_arguments)]
pub fn verify_constraints_with_lookups<SC, A, LG: LookupGadget, PcsErr>(
    air: &A,
    trace_local: &[SC::Challenge],
    trace_next: &[SC::Challenge],
    permutation_local: &[SC::Challenge],
    permutation_next: &[SC::Challenge],
    permutation_challenges: &[SC::Challenge],
    lookup_data: &[LookupData<SC::Challenge>],
    lookups: &[Lookup<Val<SC>>],
    lookup_gadget: &LG,
    public_values: &Vec<Val<SC>>,
    trace_domain: Domain<SC>,
    zeta: SC::Challenge,
    alpha: SC::Challenge,
    quotient: SC::Challenge,
) -> Result<(), VerificationError<PcsErr>>
where
    SC: SGC,
    A: for<'a> AirLookupHandler<VerifierConstraintFolderWithLookups<'a, SC>>,
{
    let sels = trace_domain.selectors_at_point(zeta);

    let main = VerticalPair::new(
        RowMajorMatrixView::new_row(trace_local),
        RowMajorMatrixView::new_row(trace_next),
    );

    let base_folder = VerifierConstraintFolder {
        main,
        public_values,
        is_first_row: sels.is_first_row,
        is_last_row: sels.is_last_row,
        is_transition: sels.is_transition,
        alpha,
        accumulator: SC::Challenge::ZERO,
    };
    let mut folder = VerifierConstraintFolderWithLookups {
        base: base_folder,
        permutation: VerticalPair::new(
            RowMajorMatrixView::new_row(permutation_local),
            RowMajorMatrixView::new_row(permutation_next),
        ),
        permutation_challenges,
    };
    // Evaluate AIR and lookup constraints.
    <A as AirLookupHandler<_>>::eval(air, &mut folder, lookups, lookup_data, lookup_gadget);
    let folded_constraints = folder.base.accumulator;

    // Check that constraints(zeta) / Z_H(zeta) = quotient(zeta)
    if folded_constraints * sels.inv_vanishing != quotient {
        return Err(VerificationError::OodEvaluationMismatch { index: None });
    }

    Ok(())
}

pub fn verify_multi_no_lookups<SC, A>(
    config: &SC,
    airs: &[A],
    proof: &MultiProof<SC>,
    public_values: &[Vec<Val<SC>>],
) -> Result<(), VerificationError<PcsError<SC>>>
where
    SC: SGC,
    SymbolicExpression<SC::Challenge>: From<SymbolicExpression<Val<SC>>>,
    A: Air<SymbolicAirBuilder<Val<SC>, SC::Challenge>>
        + for<'a> Air<VerifierConstraintFolderWithLookups<'a, SC>>
        + Clone,
    Challenge<SC>: BasedVectorSpace<Val<SC>>,
{
    let mut no_lookup_airs = airs
        .iter()
        .map(|a| AirNoLookup::new(a.clone()))
        .collect::<Vec<_>>();

    let empty_lookup_gadget = EmptyLookupGadget {};

    verify_multi(
        config,
        &mut no_lookup_airs,
        proof,
        public_values,
        &empty_lookup_gadget,
    )
}
