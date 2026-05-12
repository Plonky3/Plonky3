//! See [`crate::prover`] for an overview of the protocol and a more detailed soundness analysis.

use alloc::vec;
use alloc::vec::Vec;

use itertools::Itertools;
use p3_air::symbolic::SymbolicAirBuilder;
use p3_air::{Air, RowWindow};
use p3_challenger::{CanObserve, FieldChallenger};
use p3_commit::{Pcs, PolynomialSpace};
use p3_field::{BasedVectorSpace, ExtensionField, Field, PrimeCharacteristicRing};
use p3_matrix::dense::RowMajorMatrixView;
use p3_matrix::stack::VerticalPair;
use p3_util::zip_eq::zip_eq;
use p3_util::{checked_log_size_sum, checked_pow2};
use tracing::instrument;

use crate::error::{InvalidProofShapeError, VerificationError};
use crate::symbolic::get_log_num_quotient_chunks;
use crate::{
    AirLayout, Domain, PcsError, PreprocessedVerifierKey, Proof, StarkGenericConfig, Val,
    VerifierConstraintFolder,
};

pub fn validate_degree_bits(
    air: Option<usize>,
    degree_bits: usize,
    is_zk: usize,
) -> Result<(usize, usize), InvalidProofShapeError> {
    if degree_bits < is_zk {
        return Err(InvalidProofShapeError::DegreeBitsTooSmall {
            air,
            minimum: is_zk,
            got: degree_bits,
        });
    }

    let degree = checked_pow2(degree_bits).ok_or(InvalidProofShapeError::DegreeBitsTooLarge {
        air,
        maximum: usize::BITS as usize - 1,
        got: degree_bits,
    })?;
    Ok((degree_bits - is_zk, degree))
}

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

    // valid_shape checks each ch has length <SC::Challenge as BasedVectorSpace<Val<SC>>>::DIMENSION,
    // so from_ext_basis_coefficients won't return None.
    quotient_chunks
        .iter()
        .enumerate()
        .map(|(ch_i, ch)| {
            zps[ch_i]
                * SC::Challenge::from_ext_basis_coefficients(ch)
                    .expect("quotient chunk length checked in valid_shape")
        })
        .sum::<SC::Challenge>()
}

/// Verifies that the folded constraints match the quotient polynomial at zeta.
///
/// This evaluates the [`Air`] constraints at the out-of-domain point and checks
/// that constraints(zeta) / Z_H(zeta) = quotient(zeta).
#[allow(clippy::too_many_arguments)]
pub fn verify_constraints<SC, A, PcsErr>(
    air: &A,
    trace_local: &[SC::Challenge],
    trace_next: &[SC::Challenge],
    preprocessed_local: Option<&[SC::Challenge]>,
    preprocessed_next: Option<&[SC::Challenge]>,
    periodic_values: &[SC::Challenge],
    public_values: &[Val<SC>],
    trace_domain: Domain<SC>,
    zeta: SC::Challenge,
    alpha: SC::Challenge,
    quotient: SC::Challenge,
) -> Result<(), VerificationError<PcsErr>>
where
    SC: StarkGenericConfig,
    A: for<'a> Air<VerifierConstraintFolder<'a, SC>>,
    PcsErr: core::fmt::Debug,
{
    let sels = trace_domain.selectors_at_point(zeta);

    let main = VerticalPair::new(
        RowMajorMatrixView::new_row(trace_local),
        RowMajorMatrixView::new_row(trace_next),
    );

    let preprocessed = match (preprocessed_local, preprocessed_next) {
        (Some(local), Some(next)) => VerticalPair::new(
            RowMajorMatrixView::new_row(local),
            RowMajorMatrixView::new_row(next),
        ),
        _ => VerticalPair::new(
            RowMajorMatrixView::new(&[], 0),
            RowMajorMatrixView::new(&[], 0),
        ),
    };

    let preprocessed_window =
        RowWindow::from_two_rows(preprocessed.top.values, preprocessed.bottom.values);
    let mut folder = VerifierConstraintFolder {
        main,
        preprocessed,
        preprocessed_window,
        periodic_values,
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
    preprocessed_vk: Option<&PreprocessedVerifierKey<SC>>,
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
    // Determine expected preprocessed width.
    // - If a verifier key is provided, trust its width.
    // - Otherwise, derive width from the AIR's preprocessed trace (if any).
    let preprocessed_width = preprocessed_vk
        .map(|vk| vk.width)
        .unwrap_or_else(|| air.preprocessed_width());

    // Check that the proof's opened preprocessed values match the expected width.
    let preprocessed_local_len = opened_values
        .preprocessed_local
        .as_ref()
        .map_or(0, |v| v.len());
    let preprocessed_next_len = opened_values
        .preprocessed_next
        .as_ref()
        .map_or(0, |v| v.len());
    let expected_next_len = if !air.preprocessed_next_row_columns().is_empty() {
        preprocessed_width
    } else {
        0
    };
    if preprocessed_width != preprocessed_local_len || expected_next_len != preprocessed_next_len {
        return Err(InvalidProofShapeError::PreprocessedTraceWidthMismatch {
            expected_local: preprocessed_width,
            expected_next: expected_next_len,
            got_local: preprocessed_local_len,
            got_next: preprocessed_next_len,
        }
        .into());
    }

    // Validate consistency between width, verifier key, and zk settings.
    match (preprocessed_width, preprocessed_vk) {
        // Case: No preprocessed columns.
        //
        // Valid only if no verifier key is provided.
        (0, None) => Ok((0, None)),

        // Case: Preprocessed columns exist.
        //
        // Valid only if VK exists, widths match, and we are NOT in zk mode.
        (w, Some(vk)) if w == vk.width => Ok((w, Some(vk.commitment.clone()))),

        // Catch-all for invalid states, such as:
        // - Width is 0 but VK is provided.
        // - Width > 0 but VK is missing.
        // - Width > 0 but VK width mismatches the expected width.
        _ => Err(InvalidProofShapeError::PreprocessedVerifierKeyInconsistency.into()),
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
    verify_with_preprocessed(config, air, proof, public_values, None)
}

#[instrument(skip_all)]
pub fn verify_with_preprocessed<SC, A>(
    config: &SC,
    air: &A,
    proof: &Proof<SC>,
    public_values: &[Val<SC>],
    preprocessed_vk: Option<&PreprocessedVerifierKey<SC>>,
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
    let degree_bits = *degree_bits;

    let pcs = config.pcs();
    let (base_degree_bits, degree) = validate_degree_bits(None, degree_bits, config.is_zk())?;
    let trace_domain = pcs.natural_domain_for_degree(degree);
    // TODO: allow moving preprocessed commitment to preprocess time, if known in advance
    let (preprocessed_width, preprocessed_commit) =
        process_preprocessed_trace::<SC, A>(air, opened_values, preprocessed_vk)?;

    // Ensure the preprocessed trace and main trace have the same height.
    if let Some(vk) = preprocessed_vk
        && preprocessed_width > 0
        && vk.degree_bits != degree_bits
    {
        return Err(InvalidProofShapeError::PreprocessedDegreeMismatch {
            vk_degree_bits: vk.degree_bits,
            proof_degree_bits: degree_bits,
        }
        .into());
    }

    let layout = AirLayout {
        preprocessed_width,
        main_width: air.width(),
        num_public_values: air.num_public_values(),
        num_periodic_columns: air.num_periodic_columns(),
        ..Default::default()
    };
    let log_num_quotient_chunks =
        get_log_num_quotient_chunks::<Val<SC>, A>(air, layout, config.is_zk());
    let (_, num_quotient_chunks) = checked_log_size_sum(log_num_quotient_chunks, config.is_zk())
        .ok_or_else(|| InvalidProofShapeError::QuotientDomainTooLarge {
            air: None,
            maximum: usize::BITS as usize - 1,
            got: log_num_quotient_chunks.saturating_add(config.is_zk()),
        })?;
    let mut challenger = config.initialise_challenger();
    let init_trace_domain = pcs.natural_domain_for_degree(degree >> config.is_zk());

    let (_, quotient_domain_size) = checked_log_size_sum(degree_bits, log_num_quotient_chunks)
        .ok_or_else(|| InvalidProofShapeError::QuotientDomainTooLarge {
            air: None,
            maximum: usize::BITS as usize - 1,
            got: degree_bits.saturating_add(log_num_quotient_chunks),
        })?;
    let quotient_domain = trace_domain.create_disjoint_domain(quotient_domain_size);
    let quotient_chunks_domains = quotient_domain.split_domains(num_quotient_chunks);

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
    let expected_public_values_len = air.num_public_values();
    if public_values.len() != expected_public_values_len {
        return Err(InvalidProofShapeError::PublicValuesLengthMismatch {
            expected: expected_public_values_len,
            got: public_values.len(),
        }
        .into());
    }

    let main_next = !air.main_next_row_columns().is_empty();
    let pre_next = !air.preprocessed_next_row_columns().is_empty();
    let trace_next_ok = if main_next {
        opened_values
            .trace_next
            .as_ref()
            .is_some_and(|v| v.len() == air_width)
    } else {
        opened_values.trace_next.is_none()
    };
    let valid_shape = opened_values.trace_local.len() == air_width
        && trace_next_ok
        && opened_values.quotient_chunks.len() == num_quotient_chunks
        && opened_values
            .quotient_chunks
            .iter()
            .all(|qc| qc.len() == SC::Challenge::DIMENSION)
        // We've already checked that opened_values.random is present if and only if ZK is enabled.
        && opened_values.random.as_ref().is_none_or(|r_comm| r_comm.len() == SC::Challenge::DIMENSION);
    if !valid_shape {
        return Err(InvalidProofShapeError::OpenedValuesDimensionMismatch.into());
    }

    // Observe the instance.
    challenger.observe(Val::<SC>::from_usize(degree_bits));
    challenger.observe(Val::<SC>::from_usize(base_degree_bits));
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
    let periodic_values: Vec<SC::Challenge> = air
        .periodic_columns()
        .iter()
        .map(|periodic_col| init_trace_domain.evaluate_periodic_column_at(periodic_col, zeta))
        .collect();

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
    let trace_round = {
        let mut trace_points = vec![(zeta, opened_values.trace_local.clone())];
        if main_next {
            trace_points.push((
                zeta_next,
                opened_values
                    .trace_next
                    .clone()
                    .expect("checked in shape validation"),
            ));
        }
        (
            commitments.trace.clone(),
            vec![(trace_domain, trace_points)],
        )
    };
    coms_to_verify.extend(vec![
        trace_round,
        (
            commitments.quotient_chunks.clone(),
            // Check the commitment on the randomized domains.
            zip_eq(
                randomized_quotient_chunks_domains.iter(),
                &opened_values.quotient_chunks,
                VerificationError::from(InvalidProofShapeError::QuotientDomainsCountMismatch {
                    air: 0,
                }),
            )?
            .map(|(domain, values)| (*domain, vec![(zeta, values.clone())]))
            .collect_vec(),
        ),
    ]);

    // Add preprocessed commitment verification if present
    if preprocessed_width > 0 {
        let mut pre_points = vec![(zeta, opened_values.preprocessed_local.clone().unwrap())];
        if pre_next {
            pre_points.push((zeta_next, opened_values.preprocessed_next.clone().unwrap()));
        }
        coms_to_verify.push((
            preprocessed_commit.unwrap(),
            vec![(trace_domain, pre_points)],
        ));
    }

    pcs.verify(coms_to_verify, opening_proof, &mut challenger)
        .map_err(VerificationError::InvalidOpeningArgument)?;

    let quotient = recompose_quotient_from_chunks::<SC>(
        &quotient_chunks_domains,
        &opened_values.quotient_chunks,
        zeta,
    );

    let zeros;
    let trace_next_slice = match &opened_values.trace_next {
        Some(v) => v.as_slice(),
        None => {
            zeros = SC::Challenge::zero_vec(air_width);
            &zeros
        }
    };
    let pre_next_zeros;
    let preprocessed_next_for_verify = match &opened_values.preprocessed_next {
        Some(v) => Some(v.as_slice()),
        None if preprocessed_width > 0 => {
            pre_next_zeros = SC::Challenge::zero_vec(preprocessed_width);
            Some(pre_next_zeros.as_slice())
        }
        None => None,
    };
    verify_constraints::<SC, A, PcsError<SC>>(
        air,
        &opened_values.trace_local,
        trace_next_slice,
        opened_values.preprocessed_local.as_deref(),
        preprocessed_next_for_verify,
        &periodic_values,
        public_values,
        init_trace_domain,
        zeta,
        alpha,
        quotient,
    )?;

    Ok(())
}
