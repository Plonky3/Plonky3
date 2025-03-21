use alloc::vec;
use alloc::vec::Vec;

use itertools::Itertools;
use p3_air::{Air, BaseAir};
use p3_challenger::{CanObserve, CanSample, FieldChallenger};
use p3_commit::{Pcs, PolynomialSpace};
use p3_field::{BasedVectorSpace, Field, PrimeCharacteristicRing};
use p3_matrix::dense::RowMajorMatrixView;
use p3_matrix::stack::VerticalPair;
use tracing::instrument;

use crate::symbolic_builder::{SymbolicAirBuilder, get_log_quotient_degree};
use crate::{PcsError, Proof, StarkGenericConfig, Val, VerifierConstraintFolder};

#[instrument(skip_all)]
pub fn verify<SC, A>(
    config: &SC,
    air: &A,
    challenger: &mut SC::Challenger,
    proof: &Proof<SC>,
    public_values: &Vec<Val<SC>>,
) -> Result<(), VerificationError<PcsError<SC>>>
where
    SC: StarkGenericConfig,
    A: Air<SymbolicAirBuilder<Val<SC>>> + for<'a> Air<VerifierConstraintFolder<'a, SC>>,
{
    let Proof {
        commitments,
        opened_values,
        opening_proof,
        degree_bits,
    } = proof;

    let pcs = config.pcs();

    let degree = 1 << degree_bits;
    let log_quotient_degree =
        get_log_quotient_degree::<Val<SC>, A>(air, 0, public_values.len(), config.is_zk());
    let quotient_degree = 1 << log_quotient_degree;

    let trace_domain = pcs.natural_domain_for_degree(degree);
    let init_trace_domain = pcs.natural_domain_for_degree_zk_init(degree);

    let num_chunks = pcs.get_num_chunks(quotient_degree);
    let quotient_domain =
        trace_domain.create_disjoint_domain(1 << (degree_bits + log_quotient_degree));
    let quotient_chunks_domains = quotient_domain.split_domains(num_chunks);

    let randomized_quotient_chunks_domains = quotient_chunks_domains
        .iter()
        .map(|domain| pcs.natural_domain_for_degree_zk_ext(domain.size()))
        .collect_vec();

    let air_width = <A as BaseAir<Val<SC>>>::width(air);
    let valid_shape = opened_values.trace_local.len() == air_width
        && opened_values.trace_next.len() == air_width
        && opened_values.quotient_chunks.len() == num_chunks
        && opened_values
            .quotient_chunks
            .iter()
            .all(|qc| qc.len() == <SC::Challenge as BasedVectorSpace<Val<SC>>>::DIMENSION)
        && if let Some(r_comm) = opened_values.random.clone() {
            r_comm.len() == SC::Challenge::DIMENSION
        } else {
            true
        };
    if !valid_shape {
        return Err(VerificationError::InvalidProofShape);
    }

    // Observe the instance.
    challenger.observe(Val::<SC>::from_usize(proof.degree_bits));
    challenger.observe(Val::<SC>::from_usize(
        proof.degree_bits - config.is_zk() as usize,
    ));
    // TODO: Might be best practice to include other instance data here in the transcript, like some
    // encoding of the AIR. This protects against transcript collisions between distinct instances.
    // Practically speaking though, the only related known attack is from failing to include public
    // values. It's not clear if failing to include other instance data could enable a transcript
    // collision, since most such changes would completely change the set of satisfying witnesses.

    challenger.observe(commitments.trace.clone());
    challenger.observe_slice(public_values);
    let alpha: SC::Challenge = challenger.sample_algebra_element();
    challenger.observe(commitments.quotient_chunks.clone());
    if let Some(r_commit) = commitments.random.clone() {
        challenger.observe(r_commit);
    }

    let zeta: SC::Challenge = challenger.sample();
    let zeta_next = init_trace_domain.next_point(zeta).unwrap();

    let mut coms_to_verify = if let Some(random_commit) = &commitments.random {
        let random_values = opened_values
            .random
            .clone()
            .expect("There should be opened random values in zk.");
        vec![(
            random_commit.clone(),
            vec![(trace_domain, vec![(zeta, random_values.clone())])],
        )]
    } else {
        vec![]
    };
    coms_to_verify.extend(vec![
        (
            commitments.trace.clone(),
            vec![(
                trace_domain,
                vec![
                    (zeta, opened_values.trace_local.clone()),
                    (zeta_next, opened_values.trace_next.clone()),
                ],
            )],
        ),
        (
            commitments.quotient_chunks.clone(),
            // Check the commitment on the randomized domains.
            randomized_quotient_chunks_domains
                .iter()
                .zip(&opened_values.quotient_chunks)
                .map(|(domain, values)| (*domain, vec![(zeta, values.clone())]))
                .collect_vec(),
        ),
    ]);

    pcs.verify(coms_to_verify, opening_proof, challenger)
        .map_err(VerificationError::InvalidOpeningArgument)?;

    let zps = quotient_chunks_domains
        .iter()
        .enumerate()
        .map(|(i, domain)| {
            quotient_chunks_domains
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, other_domain)| {
                    other_domain.vanishing_poly_at_point(zeta)
                        * other_domain
                            .vanishing_poly_at_point(domain.first_point())
                            .inverse()
                })
                .product::<SC::Challenge>()
        })
        .collect_vec();

    let quotient = opened_values
        .quotient_chunks
        .iter()
        .enumerate()
        .map(|(ch_i, ch)| {
            zps[ch_i]
                * ch.iter()
                    .enumerate()
                    .map(|(e_i, &c)| SC::Challenge::ith_basis_element(e_i) * c)
                    .sum::<SC::Challenge>()
        })
        .sum::<SC::Challenge>();

    let sels = init_trace_domain.selectors_at_point(zeta);

    let main = VerticalPair::new(
        RowMajorMatrixView::new_row(&opened_values.trace_local),
        RowMajorMatrixView::new_row(&opened_values.trace_next),
    );

    let mut folder = VerifierConstraintFolder {
        main,
        public_values,
        is_first_row: sels.is_first_row,
        is_last_row: sels.is_last_row,
        is_transition: sels.is_transition,
        alpha,
        accumulator: SC::Challenge::ZERO,
    };
    air.eval(&mut folder);
    let folded_constraints = folder.accumulator;

    // Finally, check that
    //     folded_constraints(zeta) / Z_H(zeta) = quotient(zeta)
    if folded_constraints * sels.inv_vanishing != quotient {
        return Err(VerificationError::OodEvaluationMismatch);
    }

    Ok(())
}

#[derive(Debug)]
pub enum VerificationError<PcsErr> {
    InvalidProofShape,
    /// An error occurred while verifying the claimed openings.
    InvalidOpeningArgument(PcsErr),
    /// Out-of-domain evaluation mismatch, i.e. `constraints(zeta)` did not match
    /// `quotient(zeta) Z_H(zeta)`.
    OodEvaluationMismatch,
}
