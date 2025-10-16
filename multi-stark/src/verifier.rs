use alloc::vec;
use alloc::vec::Vec;

use itertools::Itertools;
use p3_air::Air;
use p3_challenger::{CanObserve, FieldChallenger};
use p3_commit::{Pcs, PolynomialSpace};
use p3_field::{BasedVectorSpace, PrimeCharacteristicRing, Field};
use p3_matrix::dense::RowMajorMatrixView;
use p3_matrix::stack::VerticalPair;
use p3_util::{zip_eq::zip_eq, log2_strict_usize};
use tracing::instrument;

use p3_uni_stark::{SymbolicAirBuilder, VerifierConstraintFolder, get_log_quotient_degree};

use crate::config::{MultiStarkGenericConfig as MSGC, Val, Domain};
use crate::proof::{InstanceOpenedValues, MultiCommitments, MultiOpenedValues, MultiProof};

#[derive(Debug)]
pub enum MultiVerificationError<PcsErr> {
    InvalidProofShape,
    InvalidOpeningArgument(PcsErr),
    OodEvaluationMismatch { index: usize },
    RandomizationError,
}

#[instrument(skip_all)]
pub fn verify_multi<SC, A>(
    config: &SC,
    airs: &[A],
    proof: &MultiProof<SC>,
    public_values: &Vec<Vec<Val<SC>>>,
) -> Result<(), MultiVerificationError<<SC::Pcs as Pcs<<SC as p3_uni_stark::StarkGenericConfig>::Challenge, SC::Challenger>>::Error>>
where
    SC: MSGC,
    A: Air<SymbolicAirBuilder<Val<SC>>> + for<'a> Air<VerifierConstraintFolder<'a, SC>>,
{
    let MultiProof {
        commitments,
        opened_values,
        opening_proof,
        degree_bits,
    } = proof;

    let pcs = config.pcs();
    let mut challenger = config.initialise_challenger();

    // Sanity checks
    if airs.len() != opened_values.instances.len() || airs.len() != public_values.len() || airs.len() != degree_bits.len() {
        return Err(MultiVerificationError::InvalidProofShape);
    }

    // Observe per-instance degree bits (bind shape). No ZK, so base degree twice.
    for &db in degree_bits {
        challenger.observe(Val::<SC>::from_usize(db));
        challenger.observe(Val::<SC>::from_usize(db));
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
        .map(|&db| pcs.natural_domain_for_degree(1 << db))
        .collect::<Vec<_>>();
    let trace_round: Vec<_> = (
        trace_domains
            .iter()
            .zip(opened_values.instances.iter())
            .map(|(dom, inst_vals)| {
                let zeta_next = dom.next_point(zeta).unwrap();
                (
                    *dom,
                    vec![
                        (zeta, inst_vals.trace_local.clone()),
                        (zeta_next, inst_vals.trace_next.clone()),
                    ],
                )
            })
            .collect::<Vec<_>>()
    );
    coms_to_verify.push((commitments.main.clone(), trace_round));

    // Quotient chunks round: flatten per-instance chunks to match commit order.
    let log_quotient_degrees: Vec<usize> = airs
        .iter()
        .zip(public_values.iter())
        .map(|(air, pv)| get_log_quotient_degree::<Val<SC>, A>(air, 0, pv.len(), 0))
        .collect();
    let quotient_domains: Vec<Vec<Domain<SC>>> = trace_domains
        .iter()
        .zip(log_quotient_degrees.iter())
        .map(|(dom, &lqd)| {
            let qdom = dom.create_disjoint_domain(1 << (log2_strict_usize(dom.size()) + lqd));
            let num_chunks = 1 << lqd;
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
        for (d, vals) in zip_eq(domains.iter(), inst_qcs, MultiVerificationError::InvalidProofShape)? {
            qc_round.push((*d, vec![(zeta, vals.clone())]));
        }
    }
    coms_to_verify.push((commitments.quotient_chunks.clone(), qc_round));

    // Verify all openings via PCS.
    pcs.verify(coms_to_verify, &proof.opening_proof, &mut challenger)
        .map_err(MultiVerificationError::InvalidOpeningArgument)?;

    // Now check constraint equality per instance.
    // For each instance, recombine quotient from chunks at zeta and compare to folded constraints.
    let mut quotient_chunk_idx = 0usize;
    for (i, air) in airs.iter().enumerate() {
        let lqd = log_quotient_degrees[i];
        let qdeg = 1 << lqd;

        // Build per-chunk Lagrange scaling factors at zeta, exactly like uni-stark.
        let db = degree_bits[i];
        let trace_dom = pcs.natural_domain_for_degree(1 << db);
        let qdom = trace_dom.create_disjoint_domain(1 << (db + lqd));
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
        let init_trace_domain = pcs.natural_domain_for_degree(1 << db);
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

        quotient_chunk_idx += qdeg;
    }

    Ok(())
}
