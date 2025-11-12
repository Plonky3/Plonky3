use alloc::vec;
use alloc::vec::Vec;

use p3_air::Air;
use p3_challenger::{CanObserve, FieldChallenger};
use p3_commit::{Pcs, PolynomialSpace};
use p3_field::{BasedVectorSpace, PrimeCharacteristicRing};
use p3_uni_stark::{
    SymbolicAirBuilder, VerificationError, VerifierConstraintFolder, get_log_quotient_degree,
    recompose_quotient_from_chunks, verify_constraints,
};
use p3_util::zip_eq::zip_eq;
use tracing::instrument;

use crate::config::{
    Challenge, Domain, PcsError, StarkGenericConfig as SGC, Val, observe_base_as_ext,
    observe_instance_binding,
};
use crate::proof::BatchProof;

#[instrument(skip_all)]
pub fn verify_batch<SC, A>(
    config: &SC,
    airs: &[A],
    proof: &BatchProof<SC>,
    public_values: &[Vec<Val<SC>>],
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

    // ZK mode is not supported yet
    if config.is_zk() != 0 {
        panic!("p3-batch-stark: ZK mode is not supported yet");
    }

    // Sanity checks
    if airs.len() != opened_values.instances.len()
        || airs.len() != public_values.len()
        || airs.len() != degree_bits.len()
    {
        return Err(VerificationError::InvalidProofShape);
    }

    // Observe the number of instances up front to match the prover's transcript.
    let n_instances = airs.len();
    observe_base_as_ext::<SC>(&mut challenger, Val::<SC>::from_usize(n_instances));

    // Validate opened values shape per instance and observe per-instance binding data.
    // Precompute per-instance log_quotient_degrees and quotient_degrees in one pass.
    let (log_quotient_degrees, quotient_degrees): (Vec<usize>, Vec<usize>) = airs
        .iter()
        .zip(public_values.iter())
        .map(|(air, pv)| {
            let lqd = get_log_quotient_degree::<Val<SC>, A>(air, 0, pv.len(), config.is_zk());
            let qd = 1 << (lqd + config.is_zk());
            (lqd, qd)
        })
        .unzip();

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
        let quotient_degree = quotient_degrees[i];
        if inst_opened_vals.quotient_chunks.len() != quotient_degree {
            return Err(VerificationError::InvalidProofShape);
        }

        for chunk in &inst_opened_vals.quotient_chunks {
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
                    (zeta, inst_opened_vals.trace_local.clone()),
                    (zeta_next, inst_opened_vals.trace_next.clone()),
                ],
            ))
        })
        .collect::<Result<Vec<_>, VerificationError<PcsError<SC>>>>()?;
    coms_to_verify.push((commitments.main.clone(), trace_round));

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
