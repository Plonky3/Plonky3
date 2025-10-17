use alloc::vec;
use alloc::vec::Vec;

use itertools::Itertools;
use p3_air::Air;
use p3_challenger::{CanObserve, FieldChallenger};
use p3_commit::{Pcs, PolynomialSpace};
use p3_field::{BasedVectorSpace, Field, PrimeCharacteristicRing, TwoAdicField};
use p3_matrix::dense::RowMajorMatrixView;
use p3_matrix::stack::VerticalPair;
use p3_uni_stark::{SymbolicAirBuilder, VerifierConstraintFolder, get_log_quotient_degree};
use p3_util::log2_strict_usize;
use p3_util::zip_eq::zip_eq;
use tracing::instrument;

use crate::config::{Domain, MultiStarkGenericConfig as MSGC, PcsError, Val};
use crate::proof::MultiProof;

#[derive(Debug)]
pub enum MultiVerificationError<PcsErr> {
    InvalidProofShape,
    InvalidOpeningArgument(PcsErr),
    OodEvaluationMismatch {
        index: usize,
    },
    /// The OOD point zeta fell into a base domain, which is a completeness edge case.
    OodPointInDomain,
    /// The log degree is too large for the field's two-adicity.
    InvalidLogDegree {
        index: usize,
        log_degree: usize,
        log_quotient_degree: usize,
    },
}

#[instrument(skip_all)]
pub fn verify_multi<SC, A>(
    config: &SC,
    airs: &[A],
    proof: &MultiProof<SC>,
    public_values: &[Vec<Val<SC>>],
) -> Result<(), MultiVerificationError<PcsError<SC>>>
where
    SC: MSGC,
    A: Air<SymbolicAirBuilder<Val<SC>>> + for<'a> Air<VerifierConstraintFolder<'a, SC>>,
    Val<SC>: TwoAdicField,
{
    let MultiProof {
        commitments,
        opened_values,
        opening_proof,
        degree_bits,
    } = proof;

    let pcs = config.pcs();
    let mut challenger = config.initialise_challenger();

    // ZK mode is not supported yet
    if config.is_zk() != 0 {
        panic!("p3-multi-stark: ZK mode is not supported yet");
    }

    // Sanity checks
    if airs.len() != opened_values.instances.len()
        || airs.len() != public_values.len()
        || airs.len() != degree_bits.len()
    {
        return Err(MultiVerificationError::InvalidProofShape);
    }

    // Observe the number of instances up front to match the prover's transcript.
    let n_instances = airs.len();
    challenger.observe(Val::<SC>::from_usize(n_instances));

    // Validate opened values shape per instance (critical soundness check)
    let log_quotient_degrees: Vec<usize> = airs
        .iter()
        .zip(public_values.iter())
        .map(|(air, pv)| get_log_quotient_degree::<Val<SC>, A>(air, 0, pv.len(), config.is_zk()))
        .collect();

    for (i, air) in airs.iter().enumerate() {
        let air_width = A::width(air);
        let inst_vals = &opened_values.instances[i];

        // Validate that log_degree + log_quotient_degree doesn't exceed field's two-adicity
        let log_degree = degree_bits[i] - config.is_zk();
        let log_quotient_degree = log_quotient_degrees[i];
        if log_degree.saturating_add(log_quotient_degree) > Val::<SC>::TWO_ADICITY {
            return Err(MultiVerificationError::InvalidLogDegree {
                index: i,
                log_degree,
                log_quotient_degree,
            });
        }

        // Validate trace widths match the AIR
        if inst_vals.trace_local.len() != air_width || inst_vals.trace_next.len() != air_width {
            return Err(MultiVerificationError::InvalidProofShape);
        }

        // Validate quotient chunks structure
        let expected_chunks = 1 << (log_quotient_degree + config.is_zk());
        if inst_vals.quotient_chunks.len() != expected_chunks {
            return Err(MultiVerificationError::InvalidProofShape);
        }

        for chunk in &inst_vals.quotient_chunks {
            if chunk.len() != SC::Challenge::DIMENSION {
                return Err(MultiVerificationError::InvalidProofShape);
            }
        }
    }

    // Observe per-instance binding data: (log_ext_degree, log_degree), width, num public values, num quotient chunks.
    for (i, air) in airs.iter().enumerate() {
        let ext_db = degree_bits[i];
        let base_db = ext_db - config.is_zk();
        challenger.observe(Val::<SC>::from_usize(ext_db));
        challenger.observe(Val::<SC>::from_usize(base_db));
        let width = A::width(air);
        challenger.observe(Val::<SC>::from_usize(width));
        let pv_len = public_values[i].len();
        challenger.observe(Val::<SC>::from_usize(pv_len));
        let num_chunks = 1 << (log_quotient_degrees[i] + config.is_zk());
        challenger.observe(Val::<SC>::from_usize(num_chunks));
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
    let trace_domains = degree_bits
        .iter()
        .map(|&ext_db| pcs.natural_domain_for_degree(1 << (ext_db - config.is_zk())))
        .collect::<Vec<_>>();
    let ext_trace_domains = degree_bits
        .iter()
        .map(|&ext_db| pcs.natural_domain_for_degree(1 << ext_db))
        .collect::<Vec<_>>();
    let trace_round: Vec<_> = ext_trace_domains
        .iter()
        .zip(trace_domains.iter())
        .zip(opened_values.instances.iter())
        .map(|((ext_dom, base_dom), inst_vals)| {
            let zeta_next = base_dom
                .next_point(zeta)
                .ok_or(MultiVerificationError::OodPointInDomain)?;
            Ok((
                *ext_dom,
                vec![
                    (zeta, inst_vals.trace_local.clone()),
                    (zeta_next, inst_vals.trace_next.clone()),
                ],
            ))
        })
        .collect::<Result<Vec<_>, MultiVerificationError<PcsError<SC>>>>()?;
    coms_to_verify.push((commitments.main.clone(), trace_round));

    // Quotient chunks round: flatten per-instance chunks to match commit order.
    // Use extended domains for the outer commit domain, with size 2^(base_db + lqd + zk), and split into 2^(lqd+zk) chunks.
    let quotient_domains: Vec<Vec<Domain<SC>>> = ext_trace_domains
        .iter()
        .zip(trace_domains.iter())
        .zip(log_quotient_degrees.iter())
        .map(|((ext_dom, base_dom), &lqd)| {
            let base_db = log2_strict_usize(base_dom.size());
            let qdom = ext_dom.create_disjoint_domain(1 << (base_db + lqd + config.is_zk()));
            let num_chunks = 1 << (lqd + config.is_zk());
            qdom.split_domains(num_chunks)
        })
        .collect();

    // Build the per-matrix openings for the aggregated quotient commitment.
    let mut qc_round = Vec::new();
    for (i, domains) in quotient_domains.iter().enumerate() {
        let inst_qcs = &opened_values.instances[i].quotient_chunks;
        if inst_qcs.len() != domains.len() {
            return Err(MultiVerificationError::InvalidProofShape);
        }
        for (d, vals) in zip_eq(
            domains.iter(),
            inst_qcs,
            MultiVerificationError::InvalidProofShape,
        )? {
            qc_round.push((*d, vec![(zeta, vals.clone())]));
        }
    }
    coms_to_verify.push((commitments.quotient_chunks.clone(), qc_round));

    // Verify all openings via PCS.
    pcs.verify(coms_to_verify, opening_proof, &mut challenger)
        .map_err(MultiVerificationError::InvalidOpeningArgument)?;

    // Now check constraint equality per instance.
    // For each instance, recombine quotient from chunks at zeta and compare to folded constraints.
    for (i, air) in airs.iter().enumerate() {
        let lqd = log_quotient_degrees[i];
        let qdeg = 1 << (lqd + config.is_zk());

        // Build per-chunk Lagrange scaling factors at zeta, exactly like uni-stark.
        let ext_db = degree_bits[i];
        let base_db = ext_db - config.is_zk();
        let ext_dom = pcs.natural_domain_for_degree(1 << ext_db);
        let qdom = ext_dom.create_disjoint_domain(1 << (base_db + lqd + config.is_zk()));
        let qc_domains = qdom.split_domains(qdeg);

        let zps = qc_domains
            .iter()
            .enumerate()
            .map(|(ch_i, domain)| {
                qc_domains
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| *j != ch_i)
                    .map(|(_, other_domain)| {
                        other_domain.vanishing_poly_at_point(zeta)
                            * other_domain
                                .vanishing_poly_at_point(domain.first_point())
                                .inverse()
                    })
                    .product::<SC::Challenge>()
            })
            .collect_vec();

        // Recompose quotient(zeta) from chunks:
        let quotient = opened_values.instances[i]
            .quotient_chunks
            .iter()
            .enumerate()
            .map(|(ch_i, ch)| {
                zps[ch_i]
                    * ch.iter()
                        .enumerate()
                        .map(|(e_i, &c)| SC::Challenge::ith_basis_element(e_i).unwrap() * c)
                        .sum::<SC::Challenge>()
            })
            .sum::<SC::Challenge>();

        // Fold constraints at zeta on (local,next)
        let init_trace_domain = pcs.natural_domain_for_degree(1 << base_db);
        let sels = init_trace_domain.selectors_at_point(zeta);

        let main = VerticalPair::new(
            RowMajorMatrixView::new_row(&opened_values.instances[i].trace_local),
            RowMajorMatrixView::new_row(&opened_values.instances[i].trace_next),
        );

        let mut folder = p3_uni_stark::VerifierConstraintFolder {
            main,
            public_values: &public_values[i],
            is_first_row: sels.is_first_row,
            is_last_row: sels.is_last_row,
            is_transition: sels.is_transition,
            alpha,
            accumulator: SC::Challenge::ZERO,
        };
        air.eval(&mut folder);
        let folded_constraints = folder.accumulator;

        if folded_constraints * sels.inv_vanishing != quotient {
            return Err(MultiVerificationError::OodEvaluationMismatch { index: i });
        }
    }

    Ok(())
}
