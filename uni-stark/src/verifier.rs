//! See [`crate::prover`] for an overview of the protocol and a more detailed soundness analysis.

use alloc::string::String;
use alloc::vec::Vec;
use alloc::{format, vec};

use itertools::Itertools;
use p3_air::Air;
use p3_challenger::{CanObserve, FieldChallenger};
use p3_commit::{Pcs, PolynomialSpace};
use p3_field::{BasedVectorSpace, Field, PrimeCharacteristicRing};
use p3_matrix::dense::RowMajorMatrixView;
use p3_matrix::stack::VerticalPair;
use p3_util::zip_eq::zip_eq;
use thiserror::Error;
use tracing::instrument;

use crate::symbolic_builder::{SymbolicAirBuilder, get_log_num_quotient_chunks};
use crate::{
    Domain, PcsError, PreprocessedVerifierKey, Proof, StarkGenericConfig, Val,
    VerifierConstraintFolder,
};

/// All challenges generated during the STARK verification process.
///
/// This struct contains both the STARK-level challenges (alpha, zeta) and the
/// FRI-level challenges that are generated during the PCS verification.
/// It is useful for recursion, where we need to obtain all challenge values
/// without running the full verifier circuit.
#[derive(Clone, Debug)]
pub struct StarkChallenges<Challenge> {
    /// The random challenge used to combine all constraint polynomials.
    pub alpha: Challenge,
    /// The out-of-domain evaluation point.
    pub zeta: Challenge,
    /// The next point after zeta in the trace domain.
    pub zeta_next: Challenge,
}

/// Generates all challenges needed for STARK verification without performing the actual verification.
///
/// This function replicates the challenge generation logic from `verify_with_preprocessed`,
/// allowing callers to obtain all Fiat-Shamir challenges for use in recursive verification
/// circuits without running the full verifier.
///
/// # Arguments
/// * `config` - The STARK configuration
/// * `air` - The AIR being verified
/// * `proof` - The proof to generate challenges for
/// * `public_values` - The public inputs
/// * `preprocessed_vk` - Optional preprocessed verifier key
///
/// # Returns
/// A tuple containing:
/// - `StarkChallenges` with alpha, zeta, and zeta_next
/// - The challenger state after observing all proof data and generating challenges
///   (this can be passed to the PCS for FRI challenge generation)
#[allow(clippy::type_complexity)]
#[instrument(skip_all)]
pub fn generate_challenges<SC, A>(
    config: &SC,
    air: &A,
    proof: &Proof<SC>,
    public_values: &[Val<SC>],
    preprocessed_vk: Option<&PreprocessedVerifierKey<SC>>,
) -> Result<(StarkChallenges<SC::Challenge>, SC::Challenger), VerificationError<PcsError<SC>>>
where
    SC: StarkGenericConfig,
    A: Air<SymbolicAirBuilder<Val<SC>>> + for<'a> Air<VerifierConstraintFolder<'a, SC>>,
{
    let Proof {
        commitments,
        opened_values,
        degree_bits,
        ..
    } = proof;

    let pcs = config.pcs();
    let degree = 1 << degree_bits;

    // Determine expected preprocessed width.
    let preprocessed_width = preprocessed_vk
        .map(|vk| vk.width)
        .or_else(|| air.preprocessed_trace().as_ref().map(|m| m.width))
        .unwrap_or(0);

    // Validate preprocessed trace consistency
    let preprocessed_local_len = opened_values
        .preprocessed_local
        .as_ref()
        .map_or(0, |v| v.len());
    let preprocessed_next_len = opened_values
        .preprocessed_next
        .as_ref()
        .map_or(0, |v| v.len());
    if preprocessed_width != preprocessed_local_len || preprocessed_width != preprocessed_next_len {
        return Err(VerificationError::InvalidProofShape);
    }

    // Get preprocessed commitment if present
    let preprocessed_commit = match (preprocessed_width, preprocessed_vk) {
        (0, None) => None,
        (w, Some(vk)) if w == vk.width => Some(vk.commitment.clone()),
        _ => return Err(VerificationError::InvalidProofShape),
    };

    // Ensure the preprocessed trace and main trace have the same height.
    if let Some(vk) = preprocessed_vk
        && preprocessed_width > 0
        && vk.degree_bits != *degree_bits
    {
        return Err(VerificationError::InvalidProofShape);
    }

    // Initialize challenger and observe instance data
    let mut challenger = config.initialise_challenger();

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

    // Observe the random commitment if it is present.
    if let Some(r_commit) = commitments.random.clone() {
        challenger.observe(r_commit);
    }

    // Get an out-of-domain point to open our values at.
    //
    // Soundness Error: dN/|EF| where `N` is the trace length and our constraint polynomial has degree `d`.
    let zeta = challenger.sample_algebra_element();

    // Compute zeta_next
    let init_trace_domain = pcs.natural_domain_for_degree(degree >> config.is_zk());
    let zeta_next = init_trace_domain
        .next_point(zeta)
        .ok_or(VerificationError::NextPointUnavailable)?;

    Ok((
        StarkChallenges {
            alpha,
            zeta,
            zeta_next,
        },
        challenger,
    ))
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
/// This evaluates the [`Air`] constraints at the out-of-domain point and checks
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
    PcsErr: core::fmt::Debug,
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
        .or_else(|| air.preprocessed_trace().as_ref().map(|m| m.width))
        .unwrap_or(0);

    // Check that the proof's opened preprocessed values match the expected width.
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
        _ => Err(VerificationError::InvalidProofShape),
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
    // Generate all STARK-level challenges using the shared function
    let (challenges, mut challenger) =
        generate_challenges(config, air, proof, public_values, preprocessed_vk)?;

    let StarkChallenges {
        alpha,
        zeta,
        zeta_next,
    } = challenges;

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
        process_preprocessed_trace::<SC, A>(air, opened_values, preprocessed_vk)?;

    let log_num_quotient_chunks = get_log_num_quotient_chunks::<Val<SC>, A>(
        air,
        preprocessed_width,
        public_values.len(),
        config.is_zk(),
    );
    let num_quotient_chunks = 1 << (log_num_quotient_chunks + config.is_zk());
    let init_trace_domain = pcs.natural_domain_for_degree(degree >> config.is_zk());

    let quotient_domain =
        trace_domain.create_disjoint_domain(1 << (degree_bits + log_num_quotient_chunks));
    let quotient_chunks_domains = quotient_domain.split_domains(num_quotient_chunks);

    let randomized_quotient_chunks_domains = quotient_chunks_domains
        .iter()
        .map(|domain| pcs.natural_domain_for_degree(domain.size() << config.is_zk()))
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
        && opened_values.quotient_chunks.len() == num_quotient_chunks
        && opened_values
            .quotient_chunks
            .iter()
            .all(|qc| qc.len() == SC::Challenge::DIMENSION)
        // We've already checked that opened_values.random is present if and only if ZK is enabled.
        && opened_values
            .random
            .as_ref()
            .is_none_or(|r_comm| r_comm.len() == SC::Challenge::DIMENSION);
    if !valid_shape {
        return Err(VerificationError::InvalidProofShape);
    }

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

    // Verify PCS openings (this will generate and use FRI challenges internally)
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

/// Defines errors that can occur during lookup verification.
#[derive(Debug)]
pub enum LookupError {
    /// Error indicating that the global cumulative sum is incorrect.
    GlobalCumulativeMismatch(Option<String>),
}

#[derive(Debug, Error)]
pub enum VerificationError<PcsErr>
where
    PcsErr: core::fmt::Debug,
{
    #[error("invalid proof shape")]
    InvalidProofShape,
    /// An error occurred while verifying the claimed openings.
    #[error("invalid opening argument: {0:?}")]
    InvalidOpeningArgument(PcsErr),
    /// Out-of-domain evaluation mismatch, i.e. `constraints(zeta)` did not match
    /// `quotient(zeta) Z_H(zeta)`.
    #[error("out-of-domain evaluation mismatch{}", .index.map(|i| format!(" at index {}", i)).unwrap_or_default())]
    OodEvaluationMismatch { index: Option<usize> },
    /// The FRI batch randomization does not correspond to the ZK setting.
    #[error("randomization error: FRI batch randomization does not match ZK setting")]
    RandomizationError,
    /// The domain does not support computing the next point algebraically.
    #[error(
        "next point unavailable: domain does not support computing the next point algebraically"
    )]
    NextPointUnavailable,
    /// Lookup related error
    #[error("lookup error: {0:?}")]
    LookupError(LookupError),
}
