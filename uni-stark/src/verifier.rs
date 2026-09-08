//! See [`crate::prover`] for an overview of the protocol and a more detailed soundness analysis.

use alloc::vec;
use alloc::vec::Vec;

use itertools::Itertools;
use p3_air::symbolic::SymbolicAirBuilder;
use p3_air::{Air, RowWindow};
use p3_challenger::GrindingChallenger;
use p3_commit::{CommitmentWithOpeningPoints, Pcs, PolynomialSpace};
use p3_field::{BasedVectorSpace, ExtensionField, Field, PrimeCharacteristicRing};
use p3_matrix::dense::RowMajorMatrixView;
use p3_matrix::stack::VerticalPair;
use p3_util::zip_eq::zip_eq;
use p3_util::{checked_log_size_sum, checked_pow2};
use tracing::instrument;

use crate::error::{InvalidProofShapeError, PeriodicColumnError, VerificationError};
use crate::symbolic::get_log_num_quotient_chunks_for_domain;
use crate::{
    AirLayout, Com, Commitments, Domain, PcsError, PreprocessedVerifierKey, Proof,
    StarkGenericConfig, StarkShape, StarkVerifierTranscript, Val, VerifierConstraintFolder,
};

/// Reject periodic columns the verifier cannot evaluate over the trace domain.
///
/// - Evaluation samples a subdomain whose size is the column length.
/// - Both verifiers call this before evaluating.
/// - A malformed AIR therefore errors instead of panicking.
///
/// # Arguments
///
/// - `periodic_columns` — the periodic columns declared by the AIR.
/// - `trace_length` — the number of rows the columns repeat over.
///
/// # Errors
///
/// - A length that is not a power of two has no evaluation subdomain.
/// - A length larger than the trace cannot sit inside the trace domain.
pub fn check_periodic_column_lengths<F>(
    periodic_columns: &[Vec<F>],
    trace_length: usize,
) -> Result<(), PeriodicColumnError> {
    for col in periodic_columns {
        let period = col.len();

        // A subdomain of size `period` exists only for powers of two.
        if !period.is_power_of_two() {
            return Err(PeriodicColumnError::LengthNotPowerOfTwo { got: period });
        }

        // That subdomain must sit inside the trace domain.
        if period > trace_length {
            return Err(PeriodicColumnError::LengthTooLarge {
                maximum: trace_length,
                got: period,
            });
        }
    }

    Ok(())
}

pub fn validate_degree_bits(
    air: Option<usize>,
    degree_bits: usize,
    is_zk: usize,
    max_log_degree: usize,
) -> Result<(usize, usize), InvalidProofShapeError> {
    if degree_bits < is_zk {
        return Err(InvalidProofShapeError::DegreeBitsTooSmall {
            air,
            minimum: is_zk,
            got: degree_bits,
        });
    }

    if degree_bits > max_log_degree {
        return Err(InvalidProofShapeError::DegreeBitsTooLarge {
            air,
            maximum: max_log_degree,
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

/// Everything the opening argument needs, once the out-of-domain point is known.
struct OpeningClaims<SC: StarkGenericConfig> {
    /// The AIR's periodic columns evaluated at the out-of-domain point.
    periodic_values: Vec<SC::Challenge>,
    /// One entry per commitment, with the points and values of each of its matrices.
    claims: Vec<CommitmentWithOpeningPoints<SC::Challenge, Com<SC>, Domain<SC>>>,
}

/// Assemble the claims the opening argument has to answer at `zeta`.
///
/// Every rejection reachable here happens while a transcript driver is live.
///
/// The caller therefore carries this result past the driver rather than returning it.
/// A driver dropped mid-pattern panics, and a panic during an unwind aborts the process.
///
/// # Arguments
///
/// - `air`: the AIR being verified, read for its periodic columns and its next-row usage.
/// - `commitments`: the commitments the proof carries.
/// - `opened_values`: the claimed evaluations the proof carries.
/// - `preprocessed_commit`: the preprocessed commitment, when the width in force is positive.
/// - `trace_domain`: the domain every committed matrix is defined over.
/// - `init_trace_domain`: the trace domain before any zero-knowledge extension.
/// - `randomized_quotient_chunks_domains`: one domain per committed quotient chunk.
/// - `zeta`: the out-of-domain point.
///
/// # Errors
///
/// - `zeta` landed inside the trace domain, where the selector inverse is undefined.
/// - The AIR declares a periodic column the trace domain cannot evaluate.
/// - The domain cannot compute the next point algebraically.
/// - A randomization opening is absent under a zero-knowledge PCS.
/// - The quotient openings do not pair one for one with the quotient chunk domains.
#[allow(clippy::too_many_arguments)]
fn prepare_opening_claims<SC, A>(
    air: &A,
    commitments: &Commitments<Com<SC>>,
    opened_values: &crate::proof::OpenedValues<SC::Challenge>,
    preprocessed_commit: Option<Com<SC>>,
    trace_domain: Domain<SC>,
    init_trace_domain: Domain<SC>,
    randomized_quotient_chunks_domains: &[Domain<SC>],
    zeta: SC::Challenge,
) -> Result<OpeningClaims<SC>, VerificationError<PcsError<SC>>>
where
    SC: StarkGenericConfig,
    A: for<'a> Air<VerifierConstraintFolder<'a, SC>>,
{
    // The opening at zeta divides by the vanishing polynomial of the trace domain.
    // Reject any zeta on the domain, where that polynomial is zero and the inverse panics.
    // Honest Fiat-Shamir sampling reaches this only with probability |H| / |EF|.
    if init_trace_domain.vanishing_poly_at_point(zeta).is_zero() {
        return Err(VerificationError::OodPointInDomain);
    }

    // Periodic columns are AIR logic; a malformed one must error, not panic.
    let periodic_columns = air.periodic_columns();
    check_periodic_column_lengths(&periodic_columns, init_trace_domain.size())?;

    let periodic_values: Vec<SC::Challenge> =
        init_trace_domain.evaluate_periodic_columns_at(&periodic_columns, zeta);

    let zeta_next = init_trace_domain
        .next_point(zeta)
        .ok_or(VerificationError::NextPointUnavailable)?;

    // A randomization commitment is present exactly when the PCS is zero-knowledge.
    // The caller has already checked that, so this branch only unpacks it.
    let mut claims = if let Some(random_commit) = &commitments.random {
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
        if !air.main_next_row_columns().is_empty() {
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

    claims.extend(vec![
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

    // Add the preprocessed commitment when the AIR declares preprocessed columns.
    if let Some(preprocessed_commit) = preprocessed_commit {
        let mut pre_points = vec![(zeta, opened_values.preprocessed_local.clone().unwrap())];
        if !air.preprocessed_next_row_columns().is_empty() {
            pre_points.push((zeta_next, opened_values.preprocessed_next.clone().unwrap()));
        }
        claims.push((preprocessed_commit, vec![(trace_domain, pre_points)]));
    }

    Ok(OpeningClaims {
        periodic_values,
        claims,
    })
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
    SC::Challenger: GrindingChallenger<Witness = Val<SC>>,
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
    SC::Challenger: GrindingChallenger<Witness = Val<SC>>,
    A: Air<SymbolicAirBuilder<Val<SC>>> + for<'a> Air<VerifierConstraintFolder<'a, SC>>,
{
    let Proof {
        commitments,
        opened_values,
        opening_proof,
        degree_bits,
        ood_pow_witness,
    } = proof;
    let degree_bits = *degree_bits;

    let pcs = config.pcs();
    let (base_degree_bits, degree) =
        validate_degree_bits(None, degree_bits, config.is_zk(), pcs.log_max_lde_height())?;
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
    // Base trace length `N` (before any ZK extension); the quotient degree model
    // measures trace columns as degree-`(N - 1)` polynomials and accounts for ZK
    // separately via `is_zk`.
    let base_degree = 1usize << base_degree_bits;
    let log_num_quotient_chunks = get_log_num_quotient_chunks_for_domain::<Val<SC>, A>(
        air,
        layout,
        pcs.natural_domain_for_degree(base_degree),
        config.is_zk(),
    );
    let (_, num_quotient_chunks) = checked_log_size_sum(log_num_quotient_chunks, config.is_zk())
        .ok_or_else(|| InvalidProofShapeError::QuotientDomainTooLarge {
            air: None,
            maximum: usize::BITS as usize - 1,
            got: log_num_quotient_chunks.saturating_add(config.is_zk()),
        })?;
    let init_trace_domain = pcs.natural_domain_for_degree(degree >> config.is_zk());

    let (quotient_domain_log_size, quotient_domain_size) =
        checked_log_size_sum(degree_bits, log_num_quotient_chunks).ok_or_else(|| {
            InvalidProofShapeError::QuotientDomainTooLarge {
                air: None,
                maximum: usize::BITS as usize - 1,
                got: degree_bits.saturating_add(log_num_quotient_chunks),
            }
        })?;
    let quotient_domain = trace_domain
        .try_create_disjoint_domain(quotient_domain_size)
        .ok_or_else(|| InvalidProofShapeError::QuotientDomainTooLarge {
            air: None,
            maximum: pcs.log_max_lde_height(),
            got: quotient_domain_log_size,
        })?;
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

    // A preprocessed commitment is bound only when the width in force is positive.
    let preprocessed_commit = preprocessed_commit.filter(|_| preprocessed_width > 0);

    // Describe the transcript before replaying it.
    //
    // Every number comes from the configuration and from the AIR, never from the proof.
    // The two the proof does supply, both trace heights, are validated above first.
    let mut challenger = config.initialise_challenger();
    let mut transcript = StarkVerifierTranscript::<SC::Challenger, Val<SC>, SC::Challenge>::new(
        &mut challenger,
        StarkShape::new::<Val<SC>, A>(
            air,
            preprocessed_width,
            degree_bits,
            base_degree_bits,
            num_quotient_chunks,
            SC::Pcs::ZK,
            config.ood_proof_of_work_bits(),
        ),
    );

    // Replay both committed traces and the public values, then redraw the batching challenge.
    //
    // Soundness Error: n/|EF| where n is the number of constraints.
    let alpha = transcript.constraint_phase::<Com<SC>>(
        commitments.trace.clone(),
        preprocessed_commit.clone(),
        public_values,
    )?;

    // Replay the quotient commitment and the grind, then redraw the out-of-domain point.
    //
    // Soundness Error: dN/|EF| where `N` is the trace length and our constraint polynomial has
    // degree `d`, plus `ood_proof_of_work_bits` from the grind checked here.
    let zeta = transcript.ood_phase::<Com<SC>>(
        commitments.quotient_chunks.clone(),
        commitments.random.clone(),
        *ood_pow_witness,
    )?;

    // Invariant: no early return may cross the span from here to the driver's `finish`.
    //
    //     prepare   -> a Result, carried rather than returned
    //     delegate  -> Begin, opening argument, End, on any outcome
    //     finish    -> every described step replayed
    //     rejection -> propagated afterwards, never across a live driver
    //
    // Dropping a driver that has not replayed its pattern panics.
    // A panic raised while an error unwinds aborts the process instead of reporting it.
    let prepared = prepare_opening_claims::<SC, A>(
        air,
        commitments,
        opened_values,
        preprocessed_commit,
        trace_domain,
        init_trace_domain,
        &randomized_quotient_chunks_domains,
        zeta,
    );

    // Run the opening argument inside the bracket, lending it the sponge.
    let checked: Result<_, VerificationError<PcsError<SC>>> = transcript.delegate(|challenger| {
        let OpeningClaims {
            periodic_values,
            claims,
        } = prepared?;
        pcs.verify(claims, opening_proof, challenger)
            .map_err(VerificationError::InvalidOpeningArgument)?;
        Ok(periodic_values)
    });

    // Every described step has now been replayed.
    transcript.finish();

    let periodic_values = checked?;

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
