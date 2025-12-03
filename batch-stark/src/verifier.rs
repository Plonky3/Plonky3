use alloc::vec;
use alloc::vec::Vec;

use p3_air::Air;
use p3_challenger::{CanObserve, FieldChallenger};
use p3_commit::{Pcs, PolynomialSpace};
use p3_field::{BasedVectorSpace, PrimeCharacteristicRing};
use p3_uni_stark::{
    SymbolicAirBuilder, VerificationError, VerifierConstraintFolder,
    recompose_quotient_from_chunks, verify_constraints,
};
use p3_util::zip_eq::zip_eq;
use tracing::instrument;

use crate::common::CommonData;
use crate::config::{
    Challenge, Domain, PcsError, StarkGenericConfig as SGC, Val, observe_instance_binding,
};
use crate::proof::BatchProof;

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
    A: Air<SymbolicAirBuilder<Val<SC>>> + for<'a> Air<VerifierConstraintFolder<'a, SC>>,
    Challenge<SC>: BasedVectorSpace<Val<SC>>,
{
    let BatchProof {
        commitments,
        opened_values,
        opening_proof,
        degree_bits,
    } = proof;

    let pcs = config.pcs();
    let mut challenger = config.initialise_challenger();

    // Sanity checks
    if airs.len() != opened_values.instances.len()
        || airs.len() != public_values.len()
        || airs.len() != degree_bits.len()
    {
        return Err(VerificationError::InvalidProofShape);
    }

    // Check that the random commitments are/are not present depending on the ZK setting.
    // - If ZK is enabled, the prover should have random commitments.
    // - If ZK is not enabled, the prover should not have random commitments.
    if (opened_values
        .instances
        .iter()
        .any(|ov| ov.random.is_some() != SC::Pcs::ZK))
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
    let mut num_quotient_chunks = Vec::with_capacity(airs.len());

    for (i, _air) in airs.iter().enumerate() {
        let pre_w = common
            .preprocessed
            .as_ref()
            .and_then(|g| g.instances[i].as_ref().map(|m| m.width))
            .unwrap_or(0);
        preprocessed_widths.push(pre_w);

        // Derive the number of quotient chunks directly from the proof shape.
        let n_chunks = opened_values.instances[i].quotient_chunks.len();
        num_quotient_chunks.push(n_chunks);
    }

    for (i, air) in airs.iter().enumerate() {
        let air_width = A::width(air);
        let inst_opened_vals = &opened_values.instances[i];

        // Validate trace widths match the AIR
        if inst_opened_vals.trace_local.len() != air_width
            || inst_opened_vals.trace_next.len() != air_width
        {
            return Err(VerificationError::InvalidProofShape);
        }

        // Validate quotient chunks structure
        let n_chunks = num_quotient_chunks[i];
        if inst_opened_vals.quotient_chunks.len() != n_chunks {
            return Err(VerificationError::InvalidProofShape);
        }

        for chunk in &inst_opened_vals.quotient_chunks {
            if chunk.len() != Challenge::<SC>::DIMENSION {
                return Err(VerificationError::InvalidProofShape);
            }
        }

        // Validate random commit
        if !inst_opened_vals
            .random
            .as_ref()
            .is_none_or(|r_comm| r_comm.len() == SC::Challenge::DIMENSION)
        {
            return Err(VerificationError::RandomizationError);
        }

        // Validate that any preprocessed width implied by CommonData matches the opened shapes.
        let pre_w = preprocessed_widths[i];
        let pre_local_len = inst_opened_vals
            .preprocessed_local
            .as_ref()
            .map_or(0, |v| v.len());
        let pre_next_len = inst_opened_vals
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

    let mut coms_to_verify = vec![];
    if let Some(random_commit) = &commitments.random {
        coms_to_verify.push((
            random_commit.clone(),
            ext_trace_domains
                .iter()
                .zip(opened_values.instances.iter())
                .map(|(domain, inst_opened_vals)| {
                    let random_vals = inst_opened_vals.random.as_ref().unwrap(); // Safe unwrap due to earlier checks
                    (*domain, vec![(zeta, random_vals.clone())])
                })
                .collect::<Vec<_>>(),
        ));
    }

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
                    (zeta, inst_opened_vals.trace_local.clone()),
                    (zeta_next, inst_opened_vals.trace_next.clone()),
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
            let n_chunks = num_quotient_chunks[i];
            let ext_dom = ext_trace_domains[i];
            let qdom = ext_dom.create_disjoint_domain((1 << ext_db) * n_chunks);
            qdom.split_domains(n_chunks)
        })
        .collect();

    // Build the per-matrix openings for the aggregated quotient commitment.
    let mut qc_round = Vec::new();
    for (i, domains) in quotient_domains.iter().enumerate() {
        let inst_qcs = &opened_values.instances[i].quotient_chunks;
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
                .preprocessed_local
                .as_ref()
                .ok_or(VerificationError::InvalidProofShape)?;
            let next = inst
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

            let base_db = meta.degree_bits;
            let pre_domain = pcs.natural_domain_for_degree(1 << base_db);
            let zeta_next_i = ext_trace_domains[inst_idx]
                .next_point(zeta)
                .ok_or(VerificationError::NextPointUnavailable)?;

            pre_round.push((
                pre_domain,
                vec![(zeta, local.clone()), (zeta_next_i, next.clone())],
            ));
        }

        coms_to_verify.push((global.commitment.clone(), pre_round));
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
            &opened_values.instances[i].quotient_chunks,
            zeta,
        );

        // Verify constraints at zeta using utility function.
        let init_trace_domain = trace_domains[i];
        verify_constraints::<SC, A, PcsError<SC>>(
            air,
            &opened_values.instances[i].trace_local,
            &opened_values.instances[i].trace_next,
            opened_values.instances[i].preprocessed_local.as_deref(),
            opened_values.instances[i].preprocessed_next.as_deref(),
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
