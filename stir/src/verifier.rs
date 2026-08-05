//! STIR verifier implementation (Construction 5.2).

use alloc::vec::Vec;

use p3_challenger::{CanObserve, CanSampleUniformBits, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_field::{
    BasedVectorSpace, ExtensionField, Field, TwoAdicField, batch_multiplicative_inverse,
};
use p3_matrix::Dimensions;
use thiserror::Error;

use crate::config::StirConfig;
use crate::proof::{StirProof, StirQueryOpenings, StirRoundProof};
use crate::utils::{
    check_shake_consistency, eval_degree_correction, eval_poly, eval_poly_at_base,
    fold_domain_params, lagrange_eval_at, next_domain_shift, reduce_mod_x_pow_minus_c,
    sample_ood_points, vanishing_poly_from_roots,
};

/// `(index, row)` pairs for a round's queries, in draw order.
type FirstRoundPairs<EF> = Vec<(usize, Vec<EF>)>;

#[derive(Clone)]
struct VirtualRoundContext<EF> {
    ans_poly: Vec<EF>,
    all_points: Vec<EF>,
    /// Coefficient form of `prod_{y in all_points} (X - y)`, built once per round so that each
    /// query can reduce it rather than re-multiplying every root.
    vanishing_coeffs: Vec<EF>,
    r_comb: EF,
}

/// Translate one opened row of the previous round's oracle into values of the current
/// virtual oracle `DegCor((g - Ans) / Z)`.
///
/// `subgroup_points` are the fiber's coordinates on the domain's subgroup; the fiber itself
/// lives at `shift * subgroup_points`. Those are **base-field** points, so every evaluation
/// here is extension-by-base.
///
/// The fiber is a coset of the `arity`-th roots of unity, hence every point shares the same
/// `x^arity`. Reducing `Ans` and the vanishing polynomial modulo `X^arity - x^arity` once per
/// query leaves `arity` coefficients to evaluate at `arity` points, turning `O(arity * t)`
/// work per query into `O(t + arity^2)`.
fn materialize_virtual_fiber<F, EF>(
    row_evals: &[EF],
    subgroup_points: &[F],
    shift: F,
    prev_ctx: Option<&VirtualRoundContext<EF>>,
) -> Option<Vec<EF>>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField + BasedVectorSpace<F>,
{
    let Some(ctx) = prev_ctx else {
        return Some(row_evals.to_vec());
    };

    let arity = row_evals.len();
    let points: Vec<F> = subgroup_points.iter().map(|&x| shift * x).collect();
    let common_power = points[0].exp_u64(arity as u64);

    let ans_rem = reduce_mod_x_pow_minus_c(&ctx.ans_poly, arity, common_power);
    let vanishing_rem = reduce_mod_x_pow_minus_c(&ctx.vanishing_coeffs, arity, common_power);

    let vanishing_values: Vec<EF> = points
        .iter()
        .map(|&x| eval_poly_at_base(&vanishing_rem, x))
        .collect();
    if vanishing_values.contains(&EF::ZERO) {
        return None;
    }
    let vanishing_inverses = batch_multiplicative_inverse(&vanishing_values);

    Some(
        row_evals
            .iter()
            .zip(points)
            .zip(vanishing_inverses)
            .map(|((&g_value, x), vanishing_inverse)| {
                let quotient = (g_value - eval_poly_at_base(&ans_rem, x)) * vanishing_inverse;
                eval_degree_correction(quotient, EF::from(x), ctx.r_comb, ctx.all_points.len())
            })
            .collect(),
    )
}

/// Wraps each opened row as a single-matrix batch for [`Mmcs::verify_multi_batch`].
///
/// `verify_multi_batch` takes `opened_values[query][matrix]` to support batches spanning
/// several committed matrices at once; STIR only ever commits one matrix per round, so every
/// inner `Vec` here always holds exactly one row slice.
fn single_matrix_opened_values<EF>(row_evals: &[Vec<EF>]) -> Vec<Vec<&[EF]>> {
    row_evals
        .iter()
        .map(|row| alloc::vec![row.as_slice()])
        .collect()
}

/// Fetch an external initial oracle's queried fibers and lay them out in draw order.
///
/// Queries are sampled with replacement, so `indices` may repeat; the source is asked once for
/// the sorted unique indices and must answer in that same order.
fn external_fibers_in_draw_order<EF: Clone, MmcsError, InputError>(
    indices: &[usize],
    arity: usize,
    source: impl FnOnce(&[usize]) -> Result<Vec<Vec<EF>>, StirError<MmcsError, InputError>>,
) -> Result<Vec<Vec<EF>>, StirError<MmcsError, InputError>> {
    let mut unique = indices.to_vec();
    unique.sort_unstable();
    unique.dedup();

    let fibers = source(&unique)?;
    if fibers.len() != unique.len() || fibers.iter().any(|fiber| fiber.len() != arity) {
        return Err(StirError::InvalidProofShape);
    }

    Ok(indices
        .iter()
        .map(|j| {
            let pos = unique
                .binary_search(j)
                .expect("every draw appears in the deduplicated index list");
            fibers[pos].clone()
        })
        .collect())
}

/// Static parameters for fetching one round's queried oracle rows via [`fetch_round_rows`].
struct RoundRowsRequest<'a, EF: Field, M: Mmcs<EF>> {
    query_openings: Option<&'a StirQueryOpenings<EF, M>>,
    query_indices: &'a [usize],
    arity: usize,
    expected_num_queries: usize,
    commitment: Option<&'a M::Commitment>,
    dimensions: &'a [Dimensions],
    error_round: usize,
}

/// Fetch a round's queried oracle rows: an external initial oracle answered by the caller's
/// own binding, or a committed oracle authenticated by one shared Merkle multi-opening proof.
///
/// Shared by the intermediate-round and final-round query-fetch steps, which differ only in
/// which openings, commitment, and dimensions apply.
fn fetch_round_rows<EF, M, Src, IE>(
    mmcs: &M,
    is_external: bool,
    external_fibers: &mut Option<Src>,
    req: &RoundRowsRequest<'_, EF, M>,
) -> Result<Vec<Vec<EF>>, StirError<M::Error, IE>>
where
    EF: Field,
    M: Mmcs<EF>,
    Src: FnOnce(&[usize]) -> Result<Vec<Vec<EF>>, StirError<M::Error, IE>>,
{
    if is_external {
        if req.query_openings.is_some() {
            return Err(StirError::InvalidProofShape);
        }
        let source = external_fibers
            .take()
            .expect("the external source is consumed exactly once");
        return external_fibers_in_draw_order(req.query_indices, req.arity, source);
    }

    let openings = req.query_openings.ok_or(StirError::InvalidProofShape)?;
    if openings.row_evals.len() != req.expected_num_queries
        || openings.row_evals.iter().any(|row| row.len() != req.arity)
    {
        return Err(StirError::InvalidProofShape);
    }
    let commit = req.commitment.ok_or(StirError::InvalidProofShape)?;
    let opened_values = single_matrix_opened_values(&openings.row_evals);
    mmcs.verify_multi_batch(
        commit,
        req.dimensions,
        req.query_indices,
        &opened_values,
        &openings.opening_proof,
    )
    .map_err(|source| StirError::InvalidMmcsProof {
        round: req.error_round,
        source,
    })?;
    Ok(openings.row_evals.clone())
}

/// Compute one query's expected fold value against the current virtual oracle.
///
/// Builds the fiber's subgroup coordinates from `domain_gen`/`fiber_step`/`j`, materializes
/// the opened row through the previous round's virtual-oracle transform (verbatim in round
/// 0), and evaluates the fold at `fold_beta`. Shared by the intermediate-round and
/// final-round query loops, which differ only in what they do with the resulting value.
#[allow(clippy::too_many_arguments)]
fn query_fold_value<F, EF, MmcsErr, InputErr>(
    row_evals: &[EF],
    j: usize,
    domain_gen: F,
    fiber_step: F,
    arity: usize,
    current_shift: F,
    fold_beta: EF,
    prev_ctx: Option<&VirtualRoundContext<EF>>,
    round: usize,
    query: usize,
) -> Result<EF, StirError<MmcsErr, InputErr>>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField + BasedVectorSpace<F>,
{
    let subgroup_points: Vec<F> = fiber_step
        .shifted_powers(domain_gen.exp_u64(j as u64))
        .take(arity)
        .collect();

    let current_fiber =
        materialize_virtual_fiber::<F, EF>(row_evals, &subgroup_points, current_shift, prev_ctx)
            .ok_or(StirError::InvalidRoundConsistency { round, query })?;

    Ok(lagrange_eval_at(
        &subgroup_points,
        &current_fiber,
        fold_beta,
    ))
}

/// Output of verifying one intermediate STIR round.
struct RoundVerifyOutput<F, EF> {
    ctx: VirtualRoundContext<EF>,
    next_shift: F,
    next_log_domain: usize,
    /// `(index, row)` pairs for round-0 queries, in draw order. Empty unless this was round 0.
    first_round_pairs: FirstRoundPairs<EF>,
}

/// Verify one intermediate STIR round (Construction 5.2) against the current virtual oracle,
/// producing the virtual-oracle context and domain state the next round (or the final round)
/// builds on.
#[allow(clippy::too_many_arguments)]
fn verify_round<F, EF, M, Challenger, IE, Src>(
    config: &StirConfig<F, EF, M, Challenger>,
    round: usize,
    rp: &StirRoundProof<EF, M, F>,
    challenger: &mut Challenger,
    current_shift: F,
    current_log_domain: usize,
    prev_ctx: Option<&VirtualRoundContext<EF>>,
    is_external: bool,
    external_fibers: &mut Option<Src>,
    commitment: Option<&M::Commitment>,
) -> Result<RoundVerifyOutput<F, EF>, StirError<M::Error, IE>>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField + BasedVectorSpace<F>,
    M: Mmcs<EF>,
    Challenger: FieldChallenger<F>
        + CanObserve<M::Commitment>
        + GrindingChallenger<Witness = F>
        + CanSampleUniformBits<F>,
    Src: FnOnce(&[usize]) -> Result<Vec<Vec<EF>>, StirError<M::Error, IE>>,
{
    let rc = &config.round_configs[round];
    let log_arity = rc.log_folding_factor;
    let arity = 1 << log_arity;

    let (fold_log_domain, fold_shift) =
        fold_domain_params(current_shift, current_log_domain, log_arity);
    let fold_height = 1usize << fold_log_domain;
    let next_log_domain = current_log_domain - 1;
    let next_shift = next_domain_shift(current_shift, log_arity);

    // Step 1: folding PoW, folding challenge gamma, and folded-oracle commitment.
    if !challenger.check_witness(rc.folding_pow_bits, rp.folding_pow_witness) {
        return Err(StirError::InvalidPowWitness { round });
    }

    let gamma: EF = challenger.sample_algebra_element();
    challenger.observe(rp.commitment.clone());

    // Mirror the prover: fold at coset coordinates via `gamma / current_shift`
    // (`fold_fiber` interpolates at subgroup coordinates).
    let fold_beta = gamma * EF::from(current_shift.inverse());

    // Step 2: OOD sampling and answer observation.
    if rp.ood_answers.len() != rc.num_ood_samples {
        return Err(StirError::InvalidProofShape);
    }

    let ood_points: Vec<EF> = sample_ood_points(
        challenger,
        [
            (current_shift, current_log_domain),
            (next_shift, next_log_domain),
            (fold_shift, fold_log_domain),
        ],
        rc.num_ood_samples,
    );

    challenger.observe_algebra_slice(&rp.ood_answers);

    // Step 3: query-phase PoW. It protects only the immediately following combination
    // challenge and query indices; configuration soundness gives no PoW credit to the
    // earlier OOD samples or the later shake challenge.
    if !challenger.check_witness(rc.pow_bits, rp.pow_witness) {
        return Err(StirError::InvalidPowWitness { round });
    }

    // Step 4: combination challenge, query sampling, and fiber verification.

    let fold_gen = F::two_adic_generator(fold_log_domain);

    let cur_dimensions = alloc::vec![Dimensions {
        height: fold_height,
        width: arity
    }];

    let r_comb: EF = challenger.sample_algebra_element();

    // Step 4a: sample every query index first, in draw order, mirroring the prover's
    // unbiased-sampling policy so the Fiat-Shamir transcript stays in sync. Merkle
    // verification is deferred to a single shared multi-opening check below.
    let mut query_indices: Vec<usize> = Vec::with_capacity(rc.num_queries);
    for _ in 0..rc.num_queries {
        let j = challenger
            .sample_uniform_bits::<true>(fold_log_domain)
            .expect("RESAMPLE = true: rejection loops internally, never errors");
        query_indices.push(j);
    }

    // Step 4b: obtain this round's oracle rows. An external initial oracle is answered by
    // the caller against its own binding; every committed oracle has one shared, pruned
    // Merkle multi-opening proof authenticating all of its query rows at once.
    let round_rows = fetch_round_rows(
        &config.mmcs,
        is_external,
        external_fibers,
        &RoundRowsRequest {
            query_openings: rp.query_openings.as_ref(),
            query_indices: &query_indices,
            arity,
            expected_num_queries: rc.num_queries,
            commitment,
            dimensions: &cur_dimensions,
            error_round: round,
        },
    )?;

    // Step 4c: per-query virtual-oracle materialization and folding.
    let mut query_points: Vec<EF> = Vec::with_capacity(rc.num_queries);
    let mut query_answers: Vec<EF> = Vec::with_capacity(rc.num_queries);
    let mut first_round_pairs: FirstRoundPairs<EF> = Vec::new();

    let mut seen_query_indices: alloc::collections::BTreeSet<usize> =
        alloc::collections::BTreeSet::new();

    // The fiber of query `j` sits at subgroup coordinates `g^j * (g^fold_height)^l`, a
    // coset of the arity-th roots of unity. Deriving it once per query serves both the
    // virtual-oracle materialization and the fold.
    let domain_gen = F::two_adic_generator(current_log_domain);
    let fiber_step = domain_gen.exp_power_of_2(fold_log_domain);

    for (q, (&j, row_evals)) in query_indices.iter().zip(&round_rows).enumerate() {
        let fold_point = EF::from(fold_shift) * EF::from(fold_gen.exp_u64(j as u64));

        let fold_val = query_fold_value(
            row_evals,
            j,
            domain_gen,
            fiber_step,
            arity,
            current_shift,
            fold_beta,
            prev_ctx,
            round,
            q,
        )?;

        if seen_query_indices.insert(j) {
            query_points.push(fold_point);
            query_answers.push(fold_val);
            if round == 0 {
                first_round_pairs.push((j, row_evals.clone()));
            }
        }
    }

    // Step 4: ans + shake polynomial observation and consistency check.
    let all_points: Vec<EF> = ood_points
        .iter()
        .chain(query_points.iter())
        .copied()
        .collect();
    let all_values: Vec<EF> = rp
        .ood_answers
        .iter()
        .chain(query_answers.iter())
        .copied()
        .collect();

    // Ans interpolates |all_points| values, so its degree is `< all_points.len()`. The
    // prover may have stripped trailing zeros, so accept any length up to that bound; reject
    // anything larger as malformed. Shake has degree `< all_points.len() - 1`.
    let max_ans_len = all_points.len();
    if rp.ans_polynomial.len() > max_ans_len
        || rp.shake_polynomial.len() > max_ans_len.saturating_sub(1)
    {
        return Err(StirError::InvalidProofShape);
    }

    // Bind ans_poly into the transcript BEFORE rho. The shake identity is a one-point check;
    // observing both polys first means the prover commits to Ans before learning rho.
    challenger.observe_algebra_slice(&rp.ans_polynomial);
    challenger.observe_algebra_slice(&rp.shake_polynomial);

    let rho: EF = challenger.sample_algebra_element();

    if !check_shake_consistency(
        &rp.ans_polynomial,
        &rp.shake_polynomial,
        &all_points,
        &all_values,
        rho,
    ) {
        return Err(StirError::InvalidShakeConsistency { round });
    }

    Ok(RoundVerifyOutput {
        ctx: VirtualRoundContext {
            ans_poly: rp.ans_polynomial.clone(),
            vanishing_coeffs: vanishing_poly_from_roots(&all_points),
            all_points,
            r_comb,
        },
        next_shift,
        next_log_domain,
        first_round_pairs,
    })
}

/// Verify the final STIR round: the last fold is checked directly against the sent final
/// polynomial rather than committed and queried again like an intermediate round.
#[allow(clippy::too_many_arguments)]
fn verify_final_round<F, EF, M, Challenger, IE, Src>(
    config: &StirConfig<F, EF, M, Challenger>,
    proof: &StirProof<EF, M, F>,
    num_rounds: usize,
    challenger: &mut Challenger,
    current_shift: F,
    current_log_domain: usize,
    prev_ctx: Option<&VirtualRoundContext<EF>>,
    is_external: bool,
    external_fibers: &mut Option<Src>,
    commitment: Option<&M::Commitment>,
) -> Result<FirstRoundPairs<EF>, StirError<M::Error, IE>>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField + BasedVectorSpace<F>,
    M: Mmcs<EF>,
    Challenger: FieldChallenger<F>
        + CanObserve<M::Commitment>
        + GrindingChallenger<Witness = F>
        + CanSampleUniformBits<F>,
    Src: FnOnce(&[usize]) -> Result<Vec<Vec<EF>>, StirError<M::Error, IE>>,
{
    let final_log_arity = config.log_folding_factor;
    let final_arity = 1usize << final_log_arity;
    let (final_new_log_domain, final_new_shift) =
        fold_domain_params(current_shift, current_log_domain, final_log_arity);
    let final_new_height = 1usize << final_new_log_domain;

    if !challenger.check_witness(
        config.final_folding_pow_bits,
        proof.final_folding_pow_witness,
    ) {
        return Err(StirError::InvalidPowWitness { round: num_rounds });
    }

    let final_gamma: EF = challenger.sample_algebra_element();
    // See the round-fold note: coset fold at `final_gamma` via `final_gamma / current_shift`.
    let final_fold_beta = final_gamma * EF::from(current_shift.inverse());

    let expected_final_len = config.final_poly_len();
    if proof.final_polynomial.len() != expected_final_len {
        return Err(StirError::InvalidProofShape);
    }

    challenger.observe_algebra_slice(&proof.final_polynomial);

    if !challenger.check_witness(config.final_pow_bits, proof.final_pow_witness) {
        return Err(StirError::InvalidPowWitness { round: num_rounds });
    }

    let final_dimensions = alloc::vec![Dimensions {
        height: final_new_height,
        width: final_arity,
    }];
    let final_gen = F::two_adic_generator(final_new_log_domain);

    // Sample every final-round query index first, deferring Merkle verification to a single
    // shared multi-opening check below.
    let mut final_indices: Vec<usize> = Vec::with_capacity(config.final_queries);
    for _ in 0..config.final_queries {
        let j = challenger
            .sample_uniform_bits::<true>(final_new_log_domain)
            .expect("RESAMPLE = true: rejection loops internally, never errors");
        final_indices.push(j);
    }

    // With no intermediate rounds these queries read the initial oracle, so they follow the
    // same external-or-committed split as an intermediate round.
    let final_rows = fetch_round_rows(
        &config.mmcs,
        is_external,
        external_fibers,
        &RoundRowsRequest {
            query_openings: proof.final_query_openings.as_ref(),
            query_indices: &final_indices,
            arity: final_arity,
            expected_num_queries: config.final_queries,
            commitment,
            dimensions: &final_dimensions,
            error_round: num_rounds,
        },
    )?;

    // When num_rounds == 0 the final queries also serve as the PCS first-round binding.
    // Track them with the same dedup-on-first-occurrence rule as the intermediate-round path.
    let mut final_seen: alloc::collections::BTreeSet<usize> = alloc::collections::BTreeSet::new();
    let mut first_round_pairs: FirstRoundPairs<EF> = Vec::new();

    let final_domain_gen = F::two_adic_generator(current_log_domain);
    let final_fiber_step = final_domain_gen.exp_power_of_2(final_new_log_domain);

    for (q, (&j, row_evals)) in final_indices.iter().zip(&final_rows).enumerate() {
        let fold_val = query_fold_value(
            row_evals,
            j,
            final_domain_gen,
            final_fiber_step,
            final_arity,
            current_shift,
            final_fold_beta,
            prev_ctx,
            num_rounds,
            q,
        )?;

        let x_j = EF::from(final_new_shift) * EF::from(final_gen.exp_u64(j as u64));

        let expected = eval_poly(&proof.final_polynomial, x_j);
        if fold_val != expected {
            return Err(StirError::FinalPolyMismatch);
        }

        if num_rounds == 0 && final_seen.insert(j) {
            first_round_pairs.push((j, row_evals.clone()));
        }
    }

    Ok(first_round_pairs)
}

/// Errors returned by [`verify_stir`].
#[derive(Debug, Error, PartialEq, Eq)]
pub enum StirError<MmcsError, InputError = ()> {
    /// A proof-of-work witness failed verification.
    #[error("Invalid proof-of-work witness in round {round}")]
    InvalidPowWitness { round: usize },

    /// A Merkle multi-opening proof failed for a round's queries.
    #[error("Invalid MMCS opening proof in round {round}")]
    InvalidMmcsProof {
        round: usize,
        #[source]
        source: MmcsError,
    },

    /// The shake polynomial identity failed at the random evaluation point.
    #[error("Shake polynomial consistency check failed in round {round}")]
    InvalidShakeConsistency { round: usize },

    /// A virtual-oracle evaluation landed in the prior round's challenge set.
    #[error("Invalid virtual-oracle query in round {round}, query {query}")]
    InvalidRoundConsistency { round: usize, query: usize },

    /// The final polynomial does not evaluate consistently with the last committed codeword.
    #[error("Final polynomial evaluation mismatch")]
    FinalPolyMismatch,

    /// The proof has the wrong number of rounds, queries, or OOD answers.
    #[error("Invalid proof shape")]
    InvalidProofShape,

    /// An error propagated from the input polynomial commitment scheme.
    #[error("Input error")]
    InputError(InputError),
}

impl<E1, IE1> StirError<E1, IE1> {
    /// Map the `InputError` variant to a different type.
    pub fn map_input_err<IE2>(self, f: impl FnOnce(IE1) -> IE2) -> StirError<E1, IE2> {
        match self {
            Self::InvalidPowWitness { round } => StirError::InvalidPowWitness { round },
            Self::InvalidMmcsProof { round, source } => {
                StirError::InvalidMmcsProof { round, source }
            }
            Self::InvalidShakeConsistency { round } => StirError::InvalidShakeConsistency { round },
            Self::InvalidRoundConsistency { round, query } => {
                StirError::InvalidRoundConsistency { round, query }
            }
            Self::FinalPolyMismatch => StirError::FinalPolyMismatch,
            Self::InvalidProofShape => StirError::InvalidProofShape,
            Self::InputError(e) => StirError::InputError(f(e)),
        }
    }
}

/// Side outputs of a successful STIR verification.
///
/// A caller binding its own commitment to the STIR proof needs the fold-domain positions the
/// verifier sampled in the first round (or the final round, when `num_rounds == 0`) and the
/// oracle values there. Returning that view directly saves the caller from replaying the
/// Fiat-Shamir transcript by hand, which would be fragile against any future change to the
/// transcript order. `first_round_fiber_evals[i]` corresponds to `first_round_indices[i]`, and
/// the indices are sorted in ascending order.
#[derive(Debug, Clone)]
pub struct StirVerifyOutputs<EF> {
    /// Sorted (ascending) unique fold-domain query indices from the first round (or the final
    /// round when `num_rounds == 0`). Length ≤ `num_queries` (or `final_queries`) due to
    /// deduplication.
    pub first_round_indices: Vec<usize>,
    /// Row evaluations for each unique query, aligned with `first_round_indices`.
    pub first_round_fiber_evals: Vec<Vec<EF>>,
}

/// Source of the queried fibers of an external initial oracle, when there is none.
type NoExternalFibers<EF, MmcsError> = fn(&[usize]) -> Result<Vec<Vec<EF>>, StirError<MmcsError>>;

/// Verify a STIR proof (Construction 5.2).
pub fn verify_stir<F, EF, M, Challenger>(
    config: &StirConfig<F, EF, M, Challenger>,
    proof: &StirProof<EF, M, Challenger::Witness>,
    challenger: &mut Challenger,
) -> Result<StirVerifyOutputs<EF>, StirError<M::Error>>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField + BasedVectorSpace<F>,
    M: Mmcs<EF>,
    Challenger: FieldChallenger<F>
        + CanObserve<M::Commitment>
        + GrindingChallenger<Witness = F>
        + CanSampleUniformBits<F>,
{
    verify_stir_inner(
        config,
        proof,
        challenger,
        None::<NoExternalFibers<EF, M::Error>>,
    )
}

/// Verify a STIR proof whose initial oracle is external.
///
/// The proof carries no commitment to the initial codeword and no openings against it. Instead
/// `initial_fibers` is called once with the sorted unique fold-domain indices STIR sampled for
/// the round that reads the initial oracle; it must return the matching fibers, each of length
/// `2^log_folding_factor` and ordered by fiber column, as authenticated by whatever commitment
/// the caller has already bound into the transcript.
///
/// # Soundness requirement
///
/// See [`prove_stir_from_external_codeword`]: the initial codeword must already be pinned by
/// data absorbed into the challenger before proving, and the returned fibers must be derived
/// from that same binding.
///
/// [`prove_stir_from_external_codeword`]: crate::prover::prove_stir_from_external_codeword
pub fn verify_stir_with_external_initial<F, EF, M, Challenger, IE>(
    config: &StirConfig<F, EF, M, Challenger>,
    proof: &StirProof<EF, M, Challenger::Witness>,
    challenger: &mut Challenger,
    initial_fibers: impl FnOnce(&[usize]) -> Result<Vec<Vec<EF>>, StirError<M::Error, IE>>,
) -> Result<StirVerifyOutputs<EF>, StirError<M::Error, IE>>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField + BasedVectorSpace<F>,
    M: Mmcs<EF>,
    Challenger: FieldChallenger<F>
        + CanObserve<M::Commitment>
        + GrindingChallenger<Witness = F>
        + CanSampleUniformBits<F>,
{
    verify_stir_inner(config, proof, challenger, Some(initial_fibers))
}

/// Shared body of [`verify_stir`] and [`verify_stir_with_external_initial`].
fn verify_stir_inner<F, EF, M, Challenger, IE, Src>(
    config: &StirConfig<F, EF, M, Challenger>,
    proof: &StirProof<EF, M, Challenger::Witness>,
    challenger: &mut Challenger,
    external_fibers: Option<Src>,
) -> Result<StirVerifyOutputs<EF>, StirError<M::Error, IE>>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField + BasedVectorSpace<F>,
    M: Mmcs<EF>,
    Challenger: FieldChallenger<F>
        + CanObserve<M::Commitment>
        + GrindingChallenger<Witness = F>
        + CanSampleUniformBits<F>,
    Src: FnOnce(&[usize]) -> Result<Vec<Vec<EF>>, StirError<M::Error, IE>>,
{
    let num_rounds = config.num_rounds();

    if proof.round_proofs.len() != num_rounds {
        return Err(StirError::InvalidProofShape);
    }

    // The initial oracle is either committed by STIR — in which case its commitment enters the
    // transcript here — or external, in which case the caller has already bound it and the
    // proof must not carry a commitment at all.
    let mut external_fibers = external_fibers;
    let initial_is_external = external_fibers.is_some();
    match (&proof.initial_commitment, initial_is_external) {
        (Some(commitment), false) => challenger.observe(commitment.clone()),
        (None, true) => {}
        _ => return Err(StirError::InvalidProofShape),
    }

    // Initial domain shift is always F::GENERATOR; round_configs[0].domain_shift mirrors it
    // when num_rounds > 0.
    let mut current_shift = F::GENERATOR;
    let mut current_log_domain = config.log_starting_domain_size();
    let mut prev_ctx: Option<VirtualRoundContext<EF>> = None;

    // Round-0 query view, recorded once and returned for the PCS input-binding step.
    // Pairs are inserted in challenger-sample (insertion) order; sorted by index at the end.
    let mut first_round_pairs: FirstRoundPairs<EF> = Vec::new();

    // Commitment holding the oracle round `r` reads. Round 0 reads the initial oracle, which
    // has no commitment when it is external — callers must not reach this for that case.
    let round_commitment = |r: usize| -> Option<&M::Commitment> {
        if r == 0 {
            proof.initial_commitment.as_ref()
        } else {
            Some(&proof.round_proofs[r - 1].commitment)
        }
    };

    for (round, rp) in proof.round_proofs.iter().enumerate() {
        let commitment = round_commitment(round);
        let output = verify_round(
            config,
            round,
            rp,
            challenger,
            current_shift,
            current_log_domain,
            prev_ctx.as_ref(),
            round == 0 && initial_is_external,
            &mut external_fibers,
            commitment,
        )?;

        first_round_pairs.extend(output.first_round_pairs);
        prev_ctx = Some(output.ctx);
        current_shift = output.next_shift;
        current_log_domain = output.next_log_domain;
    }

    // Final round: verify the final fold against the last virtual oracle.
    let final_commitment = round_commitment(num_rounds);
    let final_pairs = verify_final_round(
        config,
        proof,
        num_rounds,
        challenger,
        current_shift,
        current_log_domain,
        prev_ctx.as_ref(),
        num_rounds == 0 && initial_is_external,
        &mut external_fibers,
        final_commitment,
    )?;
    first_round_pairs.extend(final_pairs);

    // Sort by index (ascending) so the PCS layer's output ordering is deterministic and
    // matches the prover-side `first_round_query_indices` which is also sorted.
    first_round_pairs.sort_by_key(|(j, _)| *j);
    let (first_round_indices, first_round_fiber_evals): (Vec<_>, Vec<_>) =
        first_round_pairs.into_iter().unzip();

    Ok(StirVerifyOutputs {
        first_round_indices,
        first_round_fiber_evals,
    })
}
