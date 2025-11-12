//! See `prover.rs` for an overview of the protocol and a more detailed soundness analysis.

use alloc::vec;
use alloc::vec::Vec;

use itertools::Itertools;
use p3_air::Air;
use p3_challenger::{CanObserve, FieldChallenger};
use p3_commit::{Pcs, PolynomialSpace};
use p3_field::{BasedVectorSpace, Field, PrimeCharacteristicRing};
use p3_matrix::dense::RowMajorMatrixView;
use p3_matrix::stack::VerticalPair;
use p3_util::zip_eq::zip_eq;
use tracing::{debug_span, instrument};

use crate::symbolic_builder::{SymbolicAirBuilder, get_log_quotient_degree};
use crate::{Domain, PcsError, Proof, StarkGenericConfig, Val, VerifierConstraintFolder};

/// Recomposes the quotient polynomial from its chunks evaluated at a point.
///
/// Given quotient chunks and their domains, this computes the Lagrange
/// interpolation coefficients (zps) and reconstructs quotient(zeta).
pub fn recompose_quotient_from_chunks<SC>(
    quotient_chunks_domains: &[Domain<SC>],
    quotient_chunks: &[Vec<SC::Challenge>],
    zeta: SC::Challenge,
) -> SC::Challenge
where
    SC: StarkGenericConfig,
{
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

    quotient_chunks
        .iter()
        .enumerate()
        .map(|(ch_i, ch)| {
            // We checked in valid_shape the length of "ch" is equal to
            // <SC::Challenge as BasedVectorSpace<Val<SC>>>::DIMENSION. Hence
            // the unwrap() will never panic.
            zps[ch_i]
                * ch.iter()
                    .enumerate()
                    .map(|(e_i, &c)| SC::Challenge::ith_basis_element(e_i).unwrap() * c)
                    .sum::<SC::Challenge>()
        })
        .sum::<SC::Challenge>()
}

/// Verifies that the folded constraints match the quotient polynomial at zeta.
///
/// This evaluates the AIR constraints at the out-of-domain point and checks
/// that constraints(zeta) / Z_H(zeta) = quotient(zeta).
#[allow(clippy::too_many_arguments)]
pub fn verify_constraints<SC, A, PcsErr>(
    air: &A,
    trace_local: &[SC::Challenge],
    trace_next: &[SC::Challenge],
    preprocessed_local: Option<&[SC::Challenge]>,
    preprocessed_next: Option<&[SC::Challenge]>,
    public_values: &[Val<SC>],
    trace_domain: Domain<SC>,
    zeta: SC::Challenge,
    alpha: SC::Challenge,
    quotient: SC::Challenge,
) -> Result<(), VerificationError<PcsErr>>
where
    SC: StarkGenericConfig,
    A: for<'a> Air<VerifierConstraintFolder<'a, SC>>,
{
    let sels = trace_domain.selectors_at_point(zeta);

    let main = VerticalPair::new(
        RowMajorMatrixView::new_row(trace_local),
        RowMajorMatrixView::new_row(trace_next),
    );

    let preprocessed = match (preprocessed_local, preprocessed_next) {
        (Some(local), Some(next)) => Some(VerticalPair::new(
            RowMajorMatrixView::new_row(local),
            RowMajorMatrixView::new_row(next),
        )),
        _ => None,
    };

    let mut folder = VerifierConstraintFolder {
        main,
        preprocessed,
        public_values,
        is_first_row: sels.is_first_row,
        is_last_row: sels.is_last_row,
        is_transition: sels.is_transition,
        alpha,
        accumulator: SC::Challenge::ZERO,
    };
    air.eval(&mut folder);
    let folded_constraints = folder.accumulator;

    // Check that constraints(zeta) / Z_H(zeta) = quotient(zeta)
    if folded_constraints * sels.inv_vanishing != quotient {
        return Err(VerificationError::OodEvaluationMismatch { index: None });
    }

    Ok(())
}

/// Validates and commits the preprocessed trace if present.
/// Returns the preprocessed width and its commitment hash (available iff width > 0).
#[allow(clippy::type_complexity)]
fn process_preprocessed_trace<SC, A>(
    air: &A,
    opened_values: &crate::proof::OpenedValues<SC::Challenge>,
    pcs: &SC::Pcs,
    trace_domain: <SC::Pcs as Pcs<SC::Challenge, SC::Challenger>>::Domain,
    is_zk: usize,
) -> Result<
    (
        usize,
        Option<<SC::Pcs as Pcs<SC::Challenge, SC::Challenger>>::Commitment>,
    ),
    VerificationError<PcsError<SC>>,
>
where
    SC: StarkGenericConfig,
    A: for<'a> Air<VerifierConstraintFolder<'a, SC>>,
{
    // If verifier asked for preprocessed trace, then proof should have it
    let preprocessed = air.preprocessed_trace();
    let preprocessed_width = preprocessed.as_ref().map(|m| m.width).unwrap_or(0);
    let preprocessed_local_len = opened_values
        .preprocessed_local
        .as_ref()
        .map_or(0, |v| v.len());
    let preprocessed_next_len = opened_values
        .preprocessed_next
        .as_ref()
        .map_or(0, |v| v.len());
    if preprocessed_width != preprocessed_local_len || preprocessed_width != preprocessed_next_len {
        // Verifier expects preprocessed trace while proof does not have it, or vice versa
        return Err(VerificationError::InvalidProofShape);
    }

    if preprocessed_width > 0 {
        assert_eq!(is_zk, 0); // TODO: preprocessed columns not supported in zk mode
        let height = preprocessed.as_ref().unwrap().values.len() / preprocessed_width;
        assert_eq!(
            height,
            trace_domain.size(),
            "Verifier's preprocessed trace height must be equal to trace domain size"
        );
        let (preprocessed_commit, _) = debug_span!("process preprocessed trace")
            .in_scope(|| pcs.commit([(trace_domain, preprocessed.unwrap())]));
        Ok((preprocessed_width, Some(preprocessed_commit)))
    } else {
        Ok((preprocessed_width, None))
    }
}

#[instrument(skip_all)]
pub fn verify<SC, A>(
    config: &SC,
    air: &A,
    proof: &Proof<SC>,
    public_values: &[Val<SC>],
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
    let trace_domain = pcs.natural_domain_for_degree(degree);
    // TODO: allow moving preprocessed commitment to preprocess time, if known in advance
    let (preprocessed_width, preprocessed_commit) =
        process_preprocessed_trace::<SC, A>(air, opened_values, pcs, trace_domain, config.is_zk())?;

    let log_quotient_degree = get_log_quotient_degree::<Val<SC>, A>(
        air,
        preprocessed_width,
        public_values.len(),
        config.is_zk(),
    );
    let quotient_degree = 1 << (log_quotient_degree + config.is_zk());
    let mut challenger = config.initialise_challenger();
    let init_trace_domain = pcs.natural_domain_for_degree(degree >> (config.is_zk()));

    let quotient_domain =
        trace_domain.create_disjoint_domain(1 << (degree_bits + log_quotient_degree));
    let quotient_chunks_domains = quotient_domain.split_domains(quotient_degree);

    let randomized_quotient_chunks_domains = quotient_chunks_domains
        .iter()
        .map(|domain| pcs.natural_domain_for_degree(domain.size() << (config.is_zk())))
        .collect_vec();
    // Check that the random commitments are/are not present depending on the ZK setting.
    // - If ZK is enabled, the prover should have random commitments.
    // - If ZK is not enabled, the prover should not have random commitments.
    if (opened_values.random.is_some() != SC::Pcs::ZK)
        || (commitments.random.is_some() != SC::Pcs::ZK)
    {
        return Err(VerificationError::RandomizationError);
    }

    let air_width = A::width(air);
    let valid_shape = opened_values.trace_local.len() == air_width
        && opened_values.trace_next.len() == air_width
        && opened_values.quotient_chunks.len() == quotient_degree
        && opened_values
            .quotient_chunks
            .iter()
            .all(|qc| qc.len() == SC::Challenge::DIMENSION)
        // We've already checked that opened_values.random is present if and only if ZK is enabled.
        && opened_values.random.as_ref().is_none_or(|r_comm| r_comm.len() == SC::Challenge::DIMENSION);
    if !valid_shape {
        return Err(VerificationError::InvalidProofShape);
    }

    // Observe the instance.
    challenger.observe(Val::<SC>::from_usize(proof.degree_bits));
    challenger.observe(Val::<SC>::from_usize(proof.degree_bits - config.is_zk()));
    challenger.observe(Val::<SC>::from_usize(preprocessed_width));
    // TODO: Might be best practice to include other instance data here in the transcript, like some
    // encoding of the AIR. This protects against transcript collisions between distinct instances.
    // Practically speaking though, the only related known attack is from failing to include public
    // values. It's not clear if failing to include other instance data could enable a transcript
    // collision, since most such changes would completely change the set of satisfying witnesses.
    challenger.observe(commitments.trace.clone());
    if preprocessed_width > 0 {
        challenger.observe(preprocessed_commit.as_ref().unwrap().clone());
    }
    challenger.observe_slice(public_values);

    // Get the first Fiat Shamir challenge which will be used to combine all constraint polynomials
    // into a single polynomial.
    //
    // Soundness Error: n/|EF| where n is the number of constraints.
    let alpha = challenger.sample_algebra_element();
    challenger.observe(commitments.quotient_chunks.clone());

    // We've already checked that commitments.random is present if and only if ZK is enabled.
    // Observe the random commitment if it is present.
    if let Some(r_commit) = commitments.random.clone() {
        challenger.observe(r_commit);
    }

    // Get an out-of-domain point to open our values at.
    //
    // Soundness Error: dN/|EF| where `N` is the trace length and our constraint polynomial has degree `d`.
    let zeta = challenger.sample_algebra_element();
    let zeta_next = init_trace_domain
        .next_point(zeta)
        .ok_or(VerificationError::NextPointUnavailable)?;

    // We've already checked that commitments.random and opened_values.random are present if and only if ZK is enabled.
    let mut coms_to_verify = if let Some(random_commit) = &commitments.random {
        let random_values = opened_values
            .random
            .as_ref()
            .ok_or(VerificationError::RandomizationError)?;
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
            zip_eq(
                randomized_quotient_chunks_domains.iter(),
                &opened_values.quotient_chunks,
                VerificationError::InvalidProofShape,
            )?
            .map(|(domain, values)| (*domain, vec![(zeta, values.clone())]))
            .collect_vec(),
        ),
    ]);

    // Add preprocessed commitment verification if present
    if preprocessed_width > 0 {
        coms_to_verify.push((
            preprocessed_commit.unwrap(),
            vec![(
                trace_domain,
                vec![
                    (zeta, opened_values.preprocessed_local.clone().unwrap()),
                    (zeta_next, opened_values.preprocessed_next.clone().unwrap()),
                ],
            )],
        ));
    }

    pcs.verify(coms_to_verify, opening_proof, &mut challenger)
        .map_err(VerificationError::InvalidOpeningArgument)?;

    let quotient = recompose_quotient_from_chunks::<SC>(
        &quotient_chunks_domains,
        &opened_values.quotient_chunks,
        zeta,
    );

    verify_constraints::<SC, A, PcsError<SC>>(
        air,
        &opened_values.trace_local,
        &opened_values.trace_next,
        opened_values.preprocessed_local.as_deref(),
        opened_values.preprocessed_next.as_deref(),
        public_values,
        init_trace_domain,
        zeta,
        alpha,
        quotient,
    )?;

    Ok(())
}

#[derive(Debug)]
pub enum VerificationError<PcsErr> {
    InvalidProofShape,
    /// An error occurred while verifying the claimed openings.
    InvalidOpeningArgument(PcsErr),
    /// Out-of-domain evaluation mismatch, i.e. `constraints(zeta)` did not match
    /// `quotient(zeta) Z_H(zeta)`.
    OodEvaluationMismatch {
        index: Option<usize>,
    },
    /// The FRI batch randomization does not correspond to the ZK setting.
    RandomizationError,
    /// The domain does not support computing the next point algebraically.
    NextPointUnavailable,
}
