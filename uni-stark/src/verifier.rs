//! See [`crate::prover`] for an overview of the protocol and a more detailed soundness analysis.

use alloc::vec;
use alloc::vec::Vec;

use itertools::Itertools;
use p3_air::symbolic::SymbolicAirBuilder;
use p3_air::{Air, BaseAir};
use p3_challenger::GrindingChallenger;
use p3_commit::{CommitmentOpening, Pcs, PeriodicColumns, PolynomialSpace, UnivariateStarkPcs};
use p3_field::{
    BasedVectorSpace, ExtensionField, Field, PrimeCharacteristicRing, batch_multiplicative_inverse,
};
use p3_util::zip_eq::zip_eq;
use p3_util::{checked_log_size_sum, checked_pow2};
use p3_verifier_arith::Native;
use tracing::instrument;

use crate::error::{InvalidProofShapeError, VerificationError};
use crate::symbolic::get_log_num_quotient_chunks_for_domain;
use crate::{
    AirLayout, Com, Commitments, ConstraintArith, ConstraintInputs, Domain, PcsError,
    PreprocessedOpenedValues, PreprocessedVerifierKey, Proof, StarkArith, StarkGenericConfig,
    StarkShape, StarkVerifierTranscript, Val, VerifierConstraintFolder,
};

pub fn validate_degree_bits(
    air: Option<usize>,
    degree_bits: usize,
    is_zk: usize,
    min_log_degree: usize,
    max_log_degree: usize,
) -> Result<(usize, usize), InvalidProofShapeError> {
    // A claimed degree is the height of the extended trace domain, in bits.
    // Removing the zero-knowledge blowup bit leaves the base trace domain.
    // That is the domain the selectors and the periodic columns are evaluated over.
    //
    //     extended domain : degree_bits
    //     base domain     : degree_bits - is_zk
    //
    // Why: the base domain is built from this proof-supplied height before the opening runs.
    // A height the backend's domain arithmetic cannot handle has to be caught here.
    let minimum = is_zk + min_log_degree;
    if degree_bits < minimum {
        return Err(InvalidProofShapeError::DegreeBitsTooSmall {
            air,
            minimum,
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
///
/// # Errors
///
/// Returns a dimension-mismatch shape error unless:
///
/// - there is exactly one chunk per domain,
/// - each chunk holds the coefficients of one extension element.
///
/// Recursive verifiers call this directly, so the shape is checked here.
pub fn recompose_quotient_from_chunks<SC>(
    quotient_chunks_domains: &[Domain<SC>],
    quotient_chunks: &[Vec<SC::Challenge>],
    zeta: SC::Challenge,
) -> Result<SC::Challenge, InvalidProofShapeError>
where
    SC: StarkGenericConfig,
{
    if quotient_chunks.len() != quotient_chunks_domains.len() {
        return Err(InvalidProofShapeError::OpenedValuesDimensionMismatch);
    }

    match quotient_chunks.len() {
        0 => return Ok(SC::Challenge::ZERO),
        1 => {
            return SC::Challenge::from_ext_basis_coefficients(&quotient_chunks[0])
                .ok_or(InvalidProofShapeError::OpenedValuesDimensionMismatch);
        }
        _ => {}
    }

    // For two chunks, direct products avoid the temporary product buffers.
    let zps = if quotient_chunks.len() == 2 {
        quotient_chunks_domains
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
            .collect_vec()
    } else {
        let vanishing_values = quotient_chunks_domains
            .iter()
            .map(|domain| domain.vanishing_poly_at_point(zeta))
            .collect_vec();

        // Prefix/suffix products give every leave-one-out numerator in linear time.
        // Dividing the total product by Z_i(zeta) would fail at a chunk-domain point.
        let mut prefix = SC::Challenge::ONE;
        let mut zps = vanishing_values
            .iter()
            .map(|&z| {
                let before = prefix;
                prefix *= z;
                before
            })
            .collect_vec();
        let mut suffix = SC::Challenge::ONE;
        for (zp, &z) in zps.iter_mut().zip(&vanishing_values).rev() {
            *zp *= suffix;
            suffix *= z;
        }

        // Domain points and denominator products are in the base field. Invert one
        // product per chunk in a batch instead of each pair's denominator separately.
        let denominators = quotient_chunks_domains
            .iter()
            .enumerate()
            .map(|(i, domain)| {
                quotient_chunks_domains
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| *j != i)
                    .map(|(_, other)| other.vanishing_poly_at_point(domain.first_point()))
                    .product::<Val<SC>>()
            })
            .collect_vec();
        for (zp, inverse) in zps
            .iter_mut()
            .zip(batch_multiplicative_inverse(&denominators))
        {
            *zp *= inverse;
        }
        zps
    };

    // A chunk of the wrong length has no extension element, and the proof is rejected.
    quotient_chunks
        .iter()
        .zip(zps)
        .try_fold(SC::Challenge::ZERO, |acc, (ch, zp)| {
            SC::Challenge::from_ext_basis_coefficients(ch)
                .map(|chunk| acc + zp * chunk)
                .ok_or(InvalidProofShapeError::OpenedValuesDimensionMismatch)
        })
}

/// Verifies that the folded constraints match the quotient polynomial at zeta.
///
/// This evaluates the [`Air`] constraints at the out-of-domain point and checks
/// that constraints(zeta) / Z_H(zeta) = quotient(zeta).
///
/// `preprocessed_width` must come from the verifier key (when supplied), or the AIR, rather than from the opened rows.
///
/// Both preprocessed rows must have that width.
///
/// Callers whose AIR does not read the next row supply a zero-filled next row.
///
/// # Errors
///
/// - A shape error for mismatched rows.
/// - [`VerificationError::OodPointInDomain`] when `zeta` is on `trace_domain`.
/// - [`VerificationError::OodEvaluationMismatch`] when the constraints miss the quotient.
#[allow(clippy::too_many_arguments)]
pub fn verify_constraints<SC, A, AR, PcsErr>(
    arith: &mut AR,
    air: &A,
    preprocessed_width: usize,
    trace_local: &[AR::EF],
    trace_next: &[AR::EF],
    preprocessed_local: Option<&[AR::EF]>,
    preprocessed_next: Option<&[AR::EF]>,
    periodic_values: &[AR::EF],
    public_values: &[AR::F],
    trace_domain: Domain<SC>,
    zeta: &AR::EF,
    alpha: &AR::EF,
    quotient: AR::EF,
) -> Result<(), VerificationError<PcsErr>>
where
    SC: StarkGenericConfig,
    A: BaseAir<Val<SC>>,
    AR: ConstraintArith<SC, A>,
    PcsErr: core::fmt::Debug,
{
    // Public inputs reach this proof only through AIR constraints.
    // A cell listed for backend binding would go completely unbound.
    assert!(
        air.public_boundary_io().is_empty(),
        "uni-stark does not support boundary-IO public values; bind them with AIR constraints"
    );

    // The constraint folder indexes these inputs by column.
    //
    //     current and next rows -> the AIR's width
    //     public values         -> the AIR's public value count
    //     periodic values       -> one per periodic column the AIR declares
    //
    // Why: a short input would panic during evaluation instead of rejecting the proof.
    //
    // Recursive verifiers reach this public entry point directly, so the shapes are checked here.
    let air_width = A::width(air);
    if trace_local.len() != air_width {
        return Err(InvalidProofShapeError::OpenedValuesDimensionMismatch.into());
    }
    if trace_next.len() != air_width {
        return Err(InvalidProofShapeError::TraceNextMismatch { air: None }.into());
    }
    let expected_public_values_len = air.num_public_values();
    if public_values.len() != expected_public_values_len {
        return Err(InvalidProofShapeError::PublicValuesLengthMismatch {
            expected: expected_public_values_len,
            got: public_values.len(),
        }
        .into());
    }
    if periodic_values.len() != air.periodic_columns().len() {
        return Err(InvalidProofShapeError::OpenedValuesDimensionMismatch.into());
    }

    let selectors = arith
        .selectors(&trace_domain, zeta)
        .ok_or(VerificationError::OodPointInDomain)?;

    // The width comes from trusted metadata, so two equally short rows cannot
    // pass validation and then panic when the AIR indexes preprocessed columns.
    let preprocessed_local_len = preprocessed_local.map_or(0, <[_]>::len);
    let preprocessed_next_len = preprocessed_next.map_or(0, <[_]>::len);
    if preprocessed_local_len != preprocessed_width || preprocessed_next_len != preprocessed_width {
        return Err(InvalidProofShapeError::PreprocessedTraceWidthMismatch {
            expected_local: preprocessed_width,
            expected_next: preprocessed_width,
            got_local: preprocessed_local_len,
            got_next: preprocessed_next_len,
        }
        .into());
    }

    let inputs = ConstraintInputs {
        trace_local,
        trace_next,
        preprocessed_local: preprocessed_local.unwrap_or(&[]),
        preprocessed_next: preprocessed_next.unwrap_or(&[]),
        periodic_values,
        public_values,
        selectors: &selectors,
        alpha: alpha.clone(),
    };
    let folded_constraints = arith.fold_constraints(air, &inputs);

    // Check that constraints(zeta) / Z_H(zeta) = quotient(zeta)
    arith.assert_zero(
        folded_constraints * selectors.inv_vanishing.clone() - quotient,
        || VerificationError::OodEvaluationMismatch { index: None },
    )
}

/// Validates the preprocessed trace and pairs its commitment with its opening.
///
/// # Returns
///
/// - The preprocessed width in force, taken from the verifier key when one is supplied.
/// - The commitment and the opening it binds, present together exactly when that width is positive.
#[allow(clippy::type_complexity)]
fn process_preprocessed_trace<'a, SC, A>(
    air: &A,
    opened_values: &'a crate::proof::OpenedValues<SC::Challenge>,
    preprocessed_vk: Option<&PreprocessedVerifierKey<SC>>,
) -> Result<
    (
        usize,
        Option<(
            <SC::Pcs as Pcs<SC::Challenge, SC::Challenger>>::Commitment,
            &'a PreprocessedOpenedValues<SC::Challenge>,
        )>,
    ),
    VerificationError<PcsError<SC>>,
>
where
    SC: StarkGenericConfig,
    A: BaseAir<Val<SC>>,
{
    // Determine expected preprocessed width.
    // - If a verifier key is provided, trust its width.
    // - Otherwise, derive width from the AIR's preprocessed trace (if any).
    let preprocessed_width = preprocessed_vk
        .map(|vk| vk.width)
        .unwrap_or_else(|| air.preprocessed_width());

    // The proof carries a preprocessed opening exactly when the AIR has preprocessed columns,
    // and it reads the next row exactly when the AIR does.
    //
    //     no preprocessed columns    -> no opening at all
    //     reads the next row         -> next row present, as wide as the current row
    //     does not read the next row -> next row absent
    //
    // Why: a present-but-empty next row is zero columns wide yet still a present opening.
    // Nothing in the opening argument covers it.
    // Its value is therefore unbound, under a full-width current row.
    let preprocessed_next_used = !air.preprocessed_next_row_columns().is_empty();
    let preprocessed = match &opened_values.preprocessed {
        Some(_) if preprocessed_width == 0 => {
            return Err(InvalidProofShapeError::UnexpectedPreprocessedValues { air: None }.into());
        }
        Some(preprocessed) => {
            let expected_next_len = if preprocessed_next_used {
                preprocessed_width
            } else {
                0
            };
            let got_next = preprocessed.next.as_ref().map_or(0, Vec::len);
            if preprocessed.local.len() != preprocessed_width || got_next != expected_next_len {
                return Err(InvalidProofShapeError::PreprocessedTraceWidthMismatch {
                    expected_local: preprocessed_width,
                    expected_next: expected_next_len,
                    got_local: preprocessed.local.len(),
                    got_next,
                }
                .into());
            }
            if !preprocessed_next_used && preprocessed.next.is_some() {
                return Err(
                    InvalidProofShapeError::UnexpectedPreprocessedNext { air: None }.into(),
                );
            }
            Some(preprocessed)
        }
        None if preprocessed_width > 0 => {
            return Err(InvalidProofShapeError::PreprocessedTraceWidthMismatch {
                expected_local: preprocessed_width,
                expected_next: if preprocessed_next_used {
                    preprocessed_width
                } else {
                    0
                },
                got_local: 0,
                got_next: 0,
            }
            .into());
        }
        None => None,
    };

    // Validate consistency between width, verifier key, and zk settings.
    match (preprocessed_width, preprocessed_vk) {
        // Case: No preprocessed columns.
        //
        // Valid only if no verifier key is provided.
        (0, None) => Ok((0, None)),

        // Case: Preprocessed columns exist.
        //
        // Valid only if VK exists, widths match, and we are NOT in zk mode.
        //
        // A zero width leaves the opening absent.
        // The commitment then binds nothing.
        // It is dropped rather than carried forward.
        (w, Some(vk)) if w == vk.width => Ok((w, preprocessed.map(|p| (vk.commitment.clone(), p)))),

        // Catch-all for invalid states, such as:
        // - Width is 0 but VK is provided.
        // - Width > 0 but VK is missing.
        // - Width > 0 but VK width mismatches the expected width.
        _ => Err(InvalidProofShapeError::PreprocessedVerifierKeyInconsistency.into()),
    }
}

/// Everything the opening argument needs, once the out-of-domain point is known.
struct OpeningClaims<SC: StarkGenericConfig, AR: StarkArith<SC>> {
    /// The AIR's periodic columns evaluated at the out-of-domain point.
    periodic_values: Vec<AR::EF>,
    /// One entry per commitment, with the points and values of each of its matrices.
    claims: Vec<CommitmentOpening<AR::EF, AR::Commitment, Domain<SC>>>,
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
/// - `preprocessed`: the preprocessed commitment and the opening it binds, when there is one.
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
fn prepare_opening_claims<SC, A, AR>(
    arith: &mut AR,
    air: &A,
    commitments: &Commitments<AR::Commitment>,
    opened_values: &crate::proof::OpenedValues<AR::EF>,
    preprocessed: Option<(AR::Commitment, &PreprocessedOpenedValues<AR::EF>)>,
    trace_domain: Domain<SC>,
    init_trace_domain: Domain<SC>,
    randomized_quotient_chunks_domains: &[Domain<SC>],
    zeta: &AR::EF,
) -> Result<OpeningClaims<SC, AR>, VerificationError<PcsError<SC>>>
where
    SC: StarkGenericConfig,
    A: BaseAir<Val<SC>>,
    AR: StarkArith<SC>,
{
    // The opening at zeta divides by the vanishing polynomial of the trace domain.
    // Reject any zeta on the domain, where that polynomial is zero and the inverse panics.
    // Honest Fiat-Shamir sampling reaches this only with probability |H| / |EF|.
    if arith.selectors(&init_trace_domain, zeta).is_none() {
        return Err(VerificationError::OodPointInDomain);
    }

    // Periodic columns are AIR logic; a malformed one must error, not panic.
    let declared = air.periodic_columns();
    let periodic_columns = PeriodicColumns::new(&declared, init_trace_domain.size())?;

    let periodic_values = arith.periodic_values(&init_trace_domain, periodic_columns, zeta);

    let zeta_next = arith
        .next_point(&init_trace_domain, zeta)
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
            vec![(trace_domain, vec![(zeta.clone(), random_values.clone())])],
        )]
    } else {
        vec![]
    };

    let trace_round = {
        let mut trace_points = vec![(zeta.clone(), opened_values.trace_local.clone())];
        if !air.main_next_row_columns().is_empty() {
            trace_points.push((
                zeta_next.clone(),
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
            .map(|(domain, values)| (*domain, vec![(zeta.clone(), values.clone())]))
            .collect_vec(),
        ),
    ]);

    // Add the preprocessed commitment when the AIR declares preprocessed columns.
    //
    // The shape check that produced this pair already matched the rows to the AIR.
    if let Some((preprocessed_commit, preprocessed)) = preprocessed {
        let mut pre_points = vec![(zeta.clone(), preprocessed.local.clone())];
        if let Some(next) = &preprocessed.next {
            pre_points.push((zeta_next, next.clone()));
        }
        claims.push((preprocessed_commit, vec![(trace_domain, pre_points)]));
    }

    Ok(OpeningClaims {
        periodic_values,
        claims: claims.into_iter().map(Into::into).collect(),
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
    verify_in(
        &mut Native::new(),
        config,
        air,
        proof,
        public_values,
        preprocessed_vk,
    )
}

/// Verifies a STARK proof within an arithmetic.
///
/// The public values are lifted into the arithmetic.
///
/// Every shape check reads the native proof, so both arithmetics reject the same proofs.
///
/// # Errors
///
/// When the proof is malformed or does not verify.
pub fn verify_in<AR, SC, A>(
    arith: &mut AR,
    config: &SC,
    air: &A,
    proof: &Proof<SC>,
    public_values: &[AR::F],
    preprocessed_vk: Option<&PreprocessedVerifierKey<SC>>,
) -> Result<(), VerificationError<PcsError<SC>>>
where
    SC: StarkGenericConfig,
    SC::Challenger: GrindingChallenger<Witness = Val<SC>>,
    A: Air<SymbolicAirBuilder<Val<SC>>>,
    AR: ConstraintArith<SC, A>,
{
    // Public inputs reach this proof only through AIR constraints.
    // A cell listed for backend binding would go completely unbound.
    assert!(
        air.public_boundary_io().is_empty(),
        "uni-stark does not support boundary-IO public values; bind them with AIR constraints"
    );

    let Proof {
        commitments,
        opened_values,
        opening_proof,
        degree_bits,
        ood_pow_witness,
    } = proof;
    let degree_bits = *degree_bits;

    let pcs = config.pcs();
    let (base_degree_bits, degree) = validate_degree_bits(
        None,
        degree_bits,
        config.is_zk(),
        pcs.log_min_trace_height(),
        pcs.log_max_trace_height(),
    )?;
    let trace_domain = pcs.natural_domain_for_degree(degree);
    // TODO: allow moving preprocessed commitment to preprocess time, if known in advance
    let (preprocessed_width, preprocessed) =
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
            maximum: pcs.log_max_trace_height(),
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

    // An AIR either reads the main trace on the next row or it does not.
    //
    //     reads the next row         -> opening present, as wide as the main trace
    //     does not read the next row -> opening absent
    //
    // Why: the opening argument never covers a row the constraints do not index into.
    // Nothing then binds its value.
    //
    // The two failures are reported separately, so a rejection names the rule it broke.
    if !air.main_next_row_columns().is_empty() {
        if opened_values
            .trace_next
            .as_ref()
            .is_none_or(|v| v.len() != air_width)
        {
            return Err(InvalidProofShapeError::TraceNextMismatch { air: None }.into());
        }
    } else if opened_values.trace_next.is_some() {
        return Err(InvalidProofShapeError::UnexpectedTraceNext { air: None }.into());
    }

    let valid_shape = opened_values.trace_local.len() == air_width
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

    // A zero grinding budget leaves the out-of-domain witness unread, so its value is
    // pinned here rather than by the grind.
    //
    //     bits = 0 -> prover emits zero, verifier reads nothing -> pin the field here
    //     bits > 0 -> prover grinds,     verifier resamples     -> the grind pins it
    if config.ood_proof_of_work_bits() == 0 && *ood_pow_witness != Val::<SC>::ZERO {
        return Err(InvalidProofShapeError::NonCanonicalOodPowWitness.into());
    }

    // Lift the proof into the arithmetic, once.
    //
    // The shape checks above read the native proof, so every lifted value has a checked shape.
    let lifted_commitments = Commitments {
        trace: arith.load_commitment(pcs, &commitments.trace),
        quotient_chunks: arith.load_commitment(pcs, &commitments.quotient_chunks),
        random: commitments
            .random
            .as_ref()
            .map(|random| arith.load_commitment(pcs, random)),
    };
    let lifted_opened = opened_values.map(|value| arith.load_ext(*value));
    let lifted_preprocessed = preprocessed
        .as_ref()
        .map(|(commit, _)| arith.load_key_commitment(pcs, commit));

    // Announce every value the transcript absorbs, in absorb order.
    arith.will_observe_commitment(pcs, &lifted_commitments.trace);
    if let Some(commit) = &lifted_preprocessed {
        arith.will_observe_commitment(pcs, commit);
    }
    arith.will_observe(public_values.to_vec());
    arith.will_observe_commitment(pcs, &lifted_commitments.quotient_chunks);
    if let Some(random) = &lifted_commitments.random {
        arith.will_observe_commitment(pcs, random);
    }
    if config.ood_proof_of_work_bits() > 0 {
        let witness = arith.load(*ood_pow_witness);
        arith.will_observe(vec![witness]);
    }

    // The transcript binds the commitment alone.
    // The opening travels with the claims instead.
    let preprocessed_commit = preprocessed.as_ref().map(|(commit, _)| commit.clone());
    let preprocessed_index = preprocessed
        .as_ref()
        .map(|_| crate::StarkOpeningLayout::new(SC::Pcs::ZK).preprocessed);

    // Describe the transcript before replaying it.
    //
    // Every number comes from the configuration and from the AIR, never from the proof.
    // The two the proof does supply, both trace heights, are validated above first.
    let mut challenger = arith.challenger(config);
    let mut transcript = arith.seeding(|| {
        StarkVerifierTranscript::<AR::Challenger, Val<SC>, SC::Challenge>::new(
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
        )
    });

    // Replay both committed traces and the public values, then redraw the batching challenge.
    //
    // Soundness Error: n/|EF| where n is the number of constraints.
    let native_public_values: Vec<Val<SC>> = public_values
        .iter()
        .map(|value| arith.value(value))
        .collect();
    let alpha = transcript.constraint_phase::<Com<SC>>(
        commitments.trace.clone(),
        preprocessed_commit,
        &native_public_values,
    )?;
    let alpha = arith.sampled_ext(alpha);

    // Replay the quotient commitment and the grind, then redraw the out-of-domain point.
    //
    // Soundness Error: dN/|EF| where `N` is the trace length and our constraint polynomial has
    // degree `d`, plus `ood_proof_of_work_bits` from the grind checked here.
    let zeta = transcript.ood_phase::<Com<SC>>(
        commitments.quotient_chunks.clone(),
        commitments.random.clone(),
        *ood_pow_witness,
    )?;
    let zeta = arith.sampled_ext(zeta);

    // Invariant: no early return may cross the span from here to the driver's `finish`.
    //
    //     prepare   -> a Result, carried rather than returned
    //     delegate  -> Begin, opening argument, End, on any outcome
    //     finish    -> every described step replayed
    //     rejection -> propagated afterwards, never across a live driver
    //
    // Dropping a driver that has not replayed its pattern panics.
    // A panic raised while an error unwinds aborts the process instead of reporting it.
    let lifted_preprocessed_opening = lifted_opened.preprocessed.as_ref();
    let prepared = prepare_opening_claims::<SC, A, AR>(
        arith,
        air,
        &lifted_commitments,
        &lifted_opened,
        lifted_preprocessed.zip(lifted_preprocessed_opening),
        trace_domain,
        init_trace_domain,
        &randomized_quotient_chunks_domains,
        &zeta,
    );

    // Run the opening argument inside the bracket, lending it the sponge.
    let checked: Result<_, VerificationError<PcsError<SC>>> = transcript.delegate(|challenger| {
        let OpeningClaims {
            periodic_values,
            claims,
        } = prepared?;
        arith
            .verify_opening(pcs, claims, opening_proof, challenger, preprocessed_index)
            .map_err(VerificationError::InvalidOpeningArgument)?;
        Ok(periodic_values)
    });

    // Every described step has now been replayed.
    transcript.finish();

    let periodic_values = checked?;

    let quotient = arith.recompose_quotient(
        &quotient_chunks_domains,
        &lifted_opened.quotient_chunks,
        &zeta,
    )?;

    let trace_next = lifted_opened
        .trace_next
        .clone()
        .unwrap_or_else(|| vec![AR::EF::ZERO; air_width]);
    let preprocessed_next = match lifted_opened.preprocessed_next() {
        Some(next) => Some(next.to_vec()),
        None if preprocessed_width > 0 => Some(vec![AR::EF::ZERO; preprocessed_width]),
        None => None,
    };
    verify_constraints::<SC, A, AR, PcsError<SC>>(
        arith,
        air,
        preprocessed_width,
        &lifted_opened.trace_local,
        &trace_next,
        lifted_opened.preprocessed_local(),
        preprocessed_next.as_deref(),
        &periodic_values,
        public_values,
        init_trace_domain,
        &zeta,
        &alpha,
        quotient,
    )?;

    Ok(())
}
