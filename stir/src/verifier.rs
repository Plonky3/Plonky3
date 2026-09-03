//! STIR verifier implementation (Construction 5.2).

use alloc::vec::Vec;

use itertools::izip;
use p3_challenger::{CanObserve, CanSampleUniformBits, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_field::{
    BasedVectorSpace, ExtensionField, Field, TwoAdicField, batch_multiplicative_inverse,
};
use p3_matrix::Dimensions;

use crate::config::{StirConfig, StirRoundConfig};
use crate::error::{ExternalSourceError, GrindStage, ProofShapeError, RoundLabel, StirError};
use crate::proof::{StirProof, StirQueryOpenings, StirRoundProof};
use crate::utils::{
    check_shake_consistency, eval_poly, eval_poly_at_base, fold_domain_params,
    lagrange_interpolate_at, next_domain_shift, reduce_mod_x_pow_minus_c, sample_ood_points,
    vanishing_poly_from_roots,
};

/// `(index, row)` pairs for a round's queries, in draw order.
type FirstRoundPairs<EF> = Vec<(usize, Vec<EF>)>;

#[derive(Clone)]
struct VirtualRoundContext<EF> {
    ans_poly: Vec<EF>,
    /// Coefficient form of `prod_{y in all_points} (X - y)`, built once per round so that each
    /// query can reduce it rather than re-multiplying every root.
    vanishing_coeffs: Vec<EF>,
    r_comb: EF,
    /// The round's degree-correction gap: the number of points `ans` interpolates. Held here
    /// rather than recovered from the point list at every use, so that it and the cached
    /// `r_comb_pow_gap1` below are always derived from the same value.
    gap: usize,
    /// `r_comb^(gap + 1)`, the degree-correction geometric sum's `r_comb` factor at the
    /// round's fixed gap. Computed once per round rather than as part of a fresh
    /// `(point * r_comb)^(gap + 1)` exponentiation per fiber point.
    r_comb_pow_gap1: EF,
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
    let gap = ctx.gap;

    let ans_rem = reduce_mod_x_pow_minus_c(&ctx.ans_poly, arity, common_power);
    let vanishing_rem = reduce_mod_x_pow_minus_c(&ctx.vanishing_coeffs, arity, common_power);

    let mut denoms: Vec<EF> = points
        .iter()
        .map(|&x| eval_poly_at_base(&vanishing_rem, x))
        .collect();
    // A vanishing evaluation of zero means this fiber meets the round's interpolation nodes,
    // which is a malformed round. Checked before the degree-correction denominator is folded
    // in, so it can never be confused with the `step == 1` case handled below.
    if denoms.contains(&EF::ZERO) {
        return None;
    }

    // Degree-correction geometric sum `(1 - step^(gap+1)) / (1 - step)` per point, `step :=
    // point * r_comb`. `step^(gap+1) = x^(gap+1) * r_comb^(gap+1)` splits into a cheap
    // per-point base-field exponentiation and the round-constant `r_comb_pow_gap1`
    // (`VirtualRoundContext::finish`), instead of a fresh extension-field exponentiation of
    // `step` at every point.
    //
    // The quotient's vanishing denominator and the geometric sum's `(1 - step)` denominator
    // are formed in place as a single product and inverted together, mirroring the way
    // `RoundProver::finish` fuses the same two factors. `step == 1` (density `1/|EF|`) needs
    // the sum's `gap + 1` closed form rather than a division by zero, so that lane's
    // vanishing denominator is left unscaled and the closed form is applied to it directly.
    let steps: Vec<EF> = points.iter().map(|&x| ctx.r_comb * x).collect();
    for (denom, &step) in denoms.iter_mut().zip(&steps) {
        if step != EF::ONE {
            *denom *= EF::ONE - step;
        }
    }
    let inverses = batch_multiplicative_inverse(&denoms);

    Some(
        izip!(row_evals, &points, &steps, &inverses)
            .map(|(&row_eval, &x, &step, &inverse)| {
                let quotient = (row_eval - eval_poly_at_base(&ans_rem, x)) * inverse;
                if step == EF::ONE {
                    quotient * EF::from_usize(gap + 1)
                } else {
                    let x_pow = x.exp_u64((gap + 1) as u64);
                    quotient * (EF::ONE - ctx.r_comb_pow_gap1 * x_pow)
                }
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

/// Reject an `Ans`/shake pair longer than the round's degree bounds allow.
///
/// - `Ans` interpolates `max_ans_len` points, so its degree is below that.
/// - The shake polynomial is one degree lower.
/// - A shorter pair is legitimate: the prover may strip trailing zeros.
fn check_ans_and_shake_lengths<EF, MmcsError, InputError>(
    round: RoundLabel,
    ans_polynomial: &[EF],
    shake_polynomial: &[EF],
    max_ans_len: usize,
) -> Result<(), StirError<MmcsError, InputError>> {
    if ans_polynomial.len() > max_ans_len {
        return Err(ProofShapeError::AnsPolynomialTooLong {
            round,
            maximum: max_ans_len,
            got: ans_polynomial.len(),
        }
        .into());
    }
    let max_shake_len = max_ans_len.saturating_sub(1);
    if shake_polynomial.len() > max_shake_len {
        return Err(ProofShapeError::ShakePolynomialTooLong {
            round,
            maximum: max_shake_len,
            got: shake_polynomial.len(),
        }
        .into());
    }
    Ok(())
}

/// Fetch an external initial oracle's queried fibers and lay them out in draw order.
///
/// Queries are sampled with replacement, so `indices` may repeat; the source is asked once for
/// the sorted unique indices and must answer in that same order.
///
/// `round` labels the reported errors: whichever round reads the initial oracle.
fn external_fibers_in_draw_order<EF: Clone, MmcsError, InputError>(
    indices: &[usize],
    arity: usize,
    round: RoundLabel,
    source: impl FnOnce(&[usize]) -> Result<Vec<Vec<EF>>, StirError<MmcsError, InputError>>,
) -> Result<Vec<Vec<EF>>, StirError<MmcsError, InputError>> {
    let mut unique = indices.to_vec();
    unique.sort_unstable();
    unique.dedup();

    let fibers = source(&unique)?;
    if fibers.len() != unique.len() {
        return Err(ExternalSourceError::FiberCount {
            round,
            expected: unique.len(),
            got: fibers.len(),
        }
        .into());
    }
    if let Some((fiber, evals)) = fibers.iter().enumerate().find(|(_, f)| f.len() != arity) {
        return Err(ExternalSourceError::FiberArity {
            round,
            fiber,
            expected: arity,
            got: evals.len(),
        }
        .into());
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
    round: RoundLabel,
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
            return Err(ProofShapeError::UnexpectedQueryOpenings { round: req.round }.into());
        }
        let source = external_fibers
            .take()
            .expect("the external source is consumed exactly once");
        return external_fibers_in_draw_order(req.query_indices, req.arity, req.round, source);
    }

    let openings = req
        .query_openings
        .ok_or(ProofShapeError::MissingQueryOpenings { round: req.round })?;
    if openings.row_evals.len() != req.expected_num_queries {
        return Err(ProofShapeError::QueryOpeningCount {
            round: req.round,
            expected: req.expected_num_queries,
            got: openings.row_evals.len(),
        }
        .into());
    }
    if let Some((query, row)) = openings
        .row_evals
        .iter()
        .enumerate()
        .find(|(_, row)| row.len() != req.arity)
    {
        return Err(ProofShapeError::OpenedRowArity {
            round: req.round,
            query,
            expected: req.arity,
            got: row.len(),
        }
        .into());
    }
    let commit = req
        .commitment
        .ok_or(ProofShapeError::MissingCommitment { round: req.round })?;
    let opened_values = single_matrix_opened_values(&openings.row_evals);
    mmcs.verify_multi_batch(
        commit,
        req.dimensions,
        req.query_indices,
        &opened_values,
        &openings.opening_proof,
    )
    .map_err(|source| StirError::InvalidMmcsProof {
        round: req.round,
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
    round: RoundLabel,
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

    Ok(lagrange_interpolate_at(
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

/// One instance's in-flight intermediate round, advanced in the order the transcript demands.
///
/// Mirrors `prover::RoundProver`: every grind sits immediately before a challenge
/// block with no prover message in between, so holding the round's virtual-oracle state open
/// across those boundaries lets a caller drive several instances through the same boundaries,
/// sharing one grind per site.
struct RoundVerifier<F, EF: Field> {
    arity: usize,
    fold_log_domain: usize,
    fold_shift: F,
    next_log_domain: usize,
    next_shift: F,

    fold_beta: EF,
    ood_points: Vec<EF>,
    query_points: Vec<EF>,
    query_answers: Vec<EF>,
    first_round_pairs: FirstRoundPairs<EF>,
    r_comb: EF,
}

impl<F, EF> RoundVerifier<F, EF>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField + BasedVectorSpace<F>,
{
    /// Fix the round's domain geometry. Touches no transcript state.
    fn new(rc: &StirRoundConfig<F>, current_shift: F, current_log_domain: usize) -> Self {
        let log_arity = rc.log_folding_factor;
        let (fold_log_domain, fold_shift) =
            fold_domain_params(current_shift, current_log_domain, log_arity);
        Self {
            arity: 1usize << log_arity,
            fold_log_domain,
            fold_shift,
            next_log_domain: current_log_domain - 1,
            next_shift: next_domain_shift(current_shift, log_arity),
            fold_beta: EF::ZERO,
            ood_points: Vec::new(),
            query_points: Vec::new(),
            query_answers: Vec::new(),
            first_round_pairs: Vec::new(),
            r_comb: EF::ZERO,
        }
    }

    /// The domains an OOD point must avoid, matching `prover::RoundProver::ood_excluded_domains`.
    const fn excluded_domains(
        &self,
        current_shift: F,
        current_log_domain: usize,
    ) -> [(F, usize); 3] {
        [
            (current_shift, current_log_domain),
            (self.next_shift, self.next_log_domain),
            (self.fold_shift, self.fold_log_domain),
        ]
    }

    /// Consume the folding challenge, deriving the coset fold point.
    fn set_gamma(&mut self, gamma: EF, current_shift: F) {
        self.fold_beta = gamma * EF::from(current_shift.inverse());
    }

    /// Fetch this round's oracle rows, translate each into the current virtual oracle, and
    /// fold. Records the round-0 `(index, row)` pairs the PCS layer needs for input binding.
    #[allow(clippy::too_many_arguments)]
    fn fetch_and_fold<M, Challenger, IE, Src>(
        &mut self,
        config: &StirConfig<F, EF, M, Challenger>,
        round: usize,
        rp: &StirRoundProof<EF, M, F>,
        current_shift: F,
        current_log_domain: usize,
        prev_ctx: Option<&VirtualRoundContext<EF>>,
        is_external: bool,
        external_fibers: &mut Option<Src>,
        commitment: Option<&M::Commitment>,
        query_indices: &[usize],
    ) -> Result<(), StirError<M::Error, IE>>
    where
        M: Mmcs<EF>,
        Src: FnOnce(&[usize]) -> Result<Vec<Vec<EF>>, StirError<M::Error, IE>>,
    {
        let fold_height = 1usize << self.fold_log_domain;
        let cur_dimensions = alloc::vec![Dimensions {
            height: fold_height,
            width: self.arity
        }];

        let round_rows = fetch_round_rows(
            &config.mmcs,
            is_external,
            external_fibers,
            &RoundRowsRequest {
                query_openings: rp.query_openings.as_ref(),
                query_indices,
                arity: self.arity,
                expected_num_queries: query_indices.len(),
                commitment,
                dimensions: &cur_dimensions,
                round: RoundLabel::Round(round),
            },
        )?;

        let fold_gen = F::two_adic_generator(self.fold_log_domain);
        // The fiber of query `j` sits at subgroup coordinates `g^j * (g^fold_height)^l`, a
        // coset of the arity-th roots of unity. Deriving it once per query serves both the
        // virtual-oracle materialization and the fold.
        let domain_gen = F::two_adic_generator(current_log_domain);
        let fiber_step = domain_gen.exp_power_of_2(self.fold_log_domain);

        let mut seen_query_indices: alloc::collections::BTreeSet<usize> =
            alloc::collections::BTreeSet::new();

        for (q, (&j, row_evals)) in query_indices.iter().zip(&round_rows).enumerate() {
            let fold_point = EF::from(self.fold_shift) * EF::from(fold_gen.exp_u64(j as u64));

            let fold_val = query_fold_value(
                row_evals,
                j,
                domain_gen,
                fiber_step,
                self.arity,
                current_shift,
                self.fold_beta,
                prev_ctx,
                RoundLabel::Round(round),
                q,
            )?;

            if seen_query_indices.insert(j) {
                self.query_points.push(fold_point);
                self.query_answers.push(fold_val);
                if round == 0 {
                    self.first_round_pairs.push((j, row_evals.clone()));
                }
            }
        }
        Ok(())
    }

    /// The OOD + query points and their claimed values, in the order `Ans` interpolates them.
    fn all_points_and_values(&self, ood_answers: &[EF]) -> (Vec<EF>, Vec<EF>) {
        let all_points: Vec<EF> = self
            .ood_points
            .iter()
            .chain(self.query_points.iter())
            .copied()
            .collect();
        let all_values: Vec<EF> = ood_answers
            .iter()
            .chain(self.query_answers.iter())
            .copied()
            .collect();
        (all_points, all_values)
    }

    /// Touches no transcript state, so it may run after the caller has moved on to other
    /// instances.
    fn finish(self, ans_polynomial: Vec<EF>, all_points: &[EF]) -> RoundVerifyOutput<F, EF> {
        let gap = all_points.len();
        let r_comb_pow_gap1 = self.r_comb.exp_u64((gap + 1) as u64);
        RoundVerifyOutput {
            ctx: VirtualRoundContext {
                vanishing_coeffs: vanishing_poly_from_roots(all_points),
                ans_poly: ans_polynomial,
                r_comb: self.r_comb,
                gap,
                r_comb_pow_gap1,
            },
            next_shift: self.next_shift,
            next_log_domain: self.next_log_domain,
            first_round_pairs: self.first_round_pairs,
        }
    }
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
    let mut rv = RoundVerifier::<F, EF>::new(rc, current_shift, current_log_domain);

    // Step 1: folding PoW, folding challenge gamma, and folded-oracle commitment.
    if !challenger.check_witness(rc.folding_pow_bits, rp.folding_pow_witness) {
        return Err(StirError::InvalidPowWitness {
            round: RoundLabel::Round(round),
        });
    }

    let gamma: EF = challenger.sample_algebra_element();
    challenger.observe(rp.commitment.clone());
    // Mirror the prover: fold at coset coordinates via `gamma / current_shift`
    // (`fold_fiber` interpolates at subgroup coordinates).
    rv.set_gamma(gamma, current_shift);

    // Step 2: OOD sampling and answer observation.
    if rp.ood_answers.len() != rc.num_ood_samples {
        return Err(ProofShapeError::OodAnswerCount {
            round: RoundLabel::Round(round),
            expected: rc.num_ood_samples,
            got: rp.ood_answers.len(),
        }
        .into());
    }

    rv.ood_points = sample_ood_points(
        challenger,
        rv.excluded_domains(current_shift, current_log_domain),
        rc.num_ood_samples,
    );

    challenger.observe_algebra_slice(&rp.ood_answers);

    // Step 3: query-phase PoW. It protects only the immediately following combination
    // challenge and query indices; configuration soundness gives no PoW credit to the
    // earlier OOD samples or the later shake challenge.
    if !challenger.check_witness(rc.pow_bits, rp.pow_witness) {
        return Err(StirError::InvalidPowWitness {
            round: RoundLabel::Round(round),
        });
    }

    // Step 4: combination challenge, query sampling, and fiber verification.
    let r_comb: EF = challenger.sample_algebra_element();
    rv.r_comb = r_comb;

    // Step 4a: sample every query index first, in draw order, mirroring the prover's
    // unbiased-sampling policy so the Fiat-Shamir transcript stays in sync. Merkle
    // verification is deferred to a single shared multi-opening check below.
    let mut query_indices: Vec<usize> = Vec::with_capacity(rc.num_queries);
    for _ in 0..rc.num_queries {
        let j = challenger
            .sample_uniform_bits::<true>(rv.fold_log_domain)
            .expect("RESAMPLE = true: rejection loops internally, never errors");
        query_indices.push(j);
    }

    // Step 4b/4c: obtain this round's oracle rows and fold each query against the current
    // virtual oracle.
    rv.fetch_and_fold(
        config,
        round,
        rp,
        current_shift,
        current_log_domain,
        prev_ctx,
        is_external,
        external_fibers,
        commitment,
        &query_indices,
    )?;

    // Step 4: ans + shake polynomial observation and consistency check.
    let (all_points, all_values) = rv.all_points_and_values(&rp.ood_answers);

    // Ans interpolates |all_points| values, bounding both polynomials' lengths.
    let max_ans_len = all_points.len();
    check_ans_and_shake_lengths(
        RoundLabel::Round(round),
        &rp.ans_polynomial,
        &rp.shake_polynomial,
        max_ans_len,
    )?;

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
        return Err(StirError::InvalidShakeConsistency {
            round: RoundLabel::Round(round),
        });
    }

    Ok(rv.finish(rp.ans_polynomial.clone(), &all_points))
}

/// One instance's in-flight final round, advanced in the order the transcript demands.
///
/// Mirrors `prover::FinalRoundProver`.
struct FinalRoundVerifier<F, EF: Field> {
    final_arity: usize,
    final_new_log_domain: usize,
    final_new_shift: F,
    fold_beta: EF,
}

impl<F, EF> FinalRoundVerifier<F, EF>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField + BasedVectorSpace<F>,
{
    /// Fix the final round's domain geometry. Touches no transcript state.
    fn new(log_folding_factor: usize, current_shift: F, current_log_domain: usize) -> Self {
        let (final_new_log_domain, final_new_shift) =
            fold_domain_params(current_shift, current_log_domain, log_folding_factor);
        Self {
            final_arity: 1usize << log_folding_factor,
            final_new_log_domain,
            final_new_shift,
            fold_beta: EF::ZERO,
        }
    }

    /// Consume the final folding challenge, deriving the coset fold point.
    fn set_gamma(&mut self, final_gamma: EF, current_shift: F) {
        self.fold_beta = final_gamma * EF::from(current_shift.inverse());
    }

    /// Fetch the final-round oracle rows, fold each query against the current virtual oracle,
    /// and check the fold against the sent final polynomial. Returns the round-0 `(index, row)`
    /// pairs the PCS layer needs for input binding, when this is also the first round.
    #[allow(clippy::too_many_arguments)]
    fn fetch_and_check<M, Challenger, IE, Src>(
        &self,
        config: &StirConfig<F, EF, M, Challenger>,
        proof: &StirProof<EF, M, F>,
        num_rounds: usize,
        current_shift: F,
        current_log_domain: usize,
        prev_ctx: Option<&VirtualRoundContext<EF>>,
        is_external: bool,
        external_fibers: &mut Option<Src>,
        commitment: Option<&M::Commitment>,
        final_indices: &[usize],
    ) -> Result<FirstRoundPairs<EF>, StirError<M::Error, IE>>
    where
        M: Mmcs<EF>,
        Src: FnOnce(&[usize]) -> Result<Vec<Vec<EF>>, StirError<M::Error, IE>>,
    {
        let final_new_height = 1usize << self.final_new_log_domain;
        let final_dimensions = alloc::vec![Dimensions {
            height: final_new_height,
            width: self.final_arity,
        }];
        let final_gen = F::two_adic_generator(self.final_new_log_domain);

        // With no intermediate rounds these queries read the initial oracle, so they follow
        // the same external-or-committed split as an intermediate round.
        let final_rows = fetch_round_rows(
            &config.mmcs,
            is_external,
            external_fibers,
            &RoundRowsRequest {
                query_openings: proof.final_query_openings.as_ref(),
                query_indices: final_indices,
                arity: self.final_arity,
                expected_num_queries: final_indices.len(),
                commitment,
                dimensions: &final_dimensions,
                round: RoundLabel::Final,
            },
        )?;

        // When num_rounds == 0 the final queries also serve as the PCS first-round binding.
        // Track them with the same dedup-on-first-occurrence rule as the intermediate-round
        // path.
        let mut final_seen: alloc::collections::BTreeSet<usize> =
            alloc::collections::BTreeSet::new();
        let mut first_round_pairs: FirstRoundPairs<EF> = Vec::new();

        let final_domain_gen = F::two_adic_generator(current_log_domain);
        let final_fiber_step = final_domain_gen.exp_power_of_2(self.final_new_log_domain);

        for (q, (&j, row_evals)) in final_indices.iter().zip(&final_rows).enumerate() {
            let fold_val = query_fold_value(
                row_evals,
                j,
                final_domain_gen,
                final_fiber_step,
                self.final_arity,
                current_shift,
                self.fold_beta,
                prev_ctx,
                RoundLabel::Final,
                q,
            )?;

            let x_j = EF::from(self.final_new_shift) * EF::from(final_gen.exp_u64(j as u64));

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
    let mut fv = FinalRoundVerifier::<F, EF>::new(
        config.final_log_folding_factor(),
        current_shift,
        current_log_domain,
    );

    if !challenger.check_witness(
        config.final_folding_pow_bits,
        proof.final_folding_pow_witness,
    ) {
        return Err(StirError::InvalidPowWitness {
            round: RoundLabel::Final,
        });
    }

    let final_gamma: EF = challenger.sample_algebra_element();
    fv.set_gamma(final_gamma, current_shift);

    let expected_final_len = config.final_poly_len();
    if proof.final_polynomial.len() != expected_final_len {
        return Err(ProofShapeError::FinalPolynomialLength {
            expected: expected_final_len,
            got: proof.final_polynomial.len(),
        }
        .into());
    }

    challenger.observe_algebra_slice(&proof.final_polynomial);

    if !challenger.check_witness(config.final_pow_bits, proof.final_pow_witness) {
        return Err(StirError::InvalidPowWitness {
            round: RoundLabel::Final,
        });
    }

    // Sample every final-round query index first, deferring Merkle verification to a single
    // shared multi-opening check below.
    let mut final_indices: Vec<usize> = Vec::with_capacity(config.final_queries);
    for _ in 0..config.final_queries {
        let j = challenger
            .sample_uniform_bits::<true>(fv.final_new_log_domain)
            .expect("RESAMPLE = true: rejection loops internally, never errors");
        final_indices.push(j);
    }

    fv.fetch_and_check(
        config,
        proof,
        num_rounds,
        current_shift,
        current_log_domain,
        prev_ctx,
        is_external,
        external_fibers,
        commitment,
        &final_indices,
    )
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
        return Err(ProofShapeError::RoundCount {
            instance: None,
            expected: num_rounds,
            got: proof.round_proofs.len(),
        }
        .into());
    }

    // The initial oracle is either committed by STIR — in which case its commitment enters the
    // transcript here — or external, in which case the caller has already bound it and the
    // proof must not carry a commitment at all.
    let mut external_fibers = external_fibers;
    let initial_is_external = external_fibers.is_some();
    match (&proof.initial_commitment, initial_is_external) {
        (Some(commitment), false) => challenger.observe(commitment.clone()),
        (None, true) => {}
        (Some(_), true) => return Err(ProofShapeError::UnexpectedInitialCommitment.into()),
        (None, false) => return Err(ProofShapeError::MissingInitialCommitment.into()),
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

/// Source of the queried fibers of `B` external initial oracles, when there are none.
type NoExternalFibersMulti<EF, MmcsError> =
    fn(&[usize]) -> Result<Vec<Vec<EF>>, StirError<MmcsError>>;

/// Verify `B` STIR proofs of possibly different degrees in lockstep, mirroring
/// [`crate::prover::prove_stir_multi`].
pub fn verify_stir_multi<F, EF, M, Challenger>(
    configs: &[&StirConfig<F, EF, M, Challenger>],
    proofs: &[&StirProof<EF, M, Challenger::Witness>],
    challenger: &mut Challenger,
) -> Result<Vec<StirVerifyOutputs<EF>>, StirError<M::Error>>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField + BasedVectorSpace<F>,
    M: Mmcs<EF>,
    Challenger: FieldChallenger<F>
        + CanObserve<M::Commitment>
        + GrindingChallenger<Witness = F>
        + CanSampleUniformBits<F>,
{
    verify_stir_multi_inner(
        configs,
        proofs,
        challenger,
        None::<Vec<NoExternalFibersMulti<EF, M::Error>>>,
    )
}

/// Verify `B` STIR proofs whose initial oracles are external, mirroring
/// [`crate::prover::prove_stir_multi_from_external_codewords`].
///
/// `initial_fibers[i]` is called once with instance `i`'s sorted unique fold-domain indices,
/// exactly as [`verify_stir_with_external_initial`]'s single-instance callback.
pub fn verify_stir_multi_with_external_initial<F, EF, M, Challenger, IE, Src>(
    configs: &[&StirConfig<F, EF, M, Challenger>],
    proofs: &[&StirProof<EF, M, Challenger::Witness>],
    challenger: &mut Challenger,
    initial_fibers: Vec<Src>,
) -> Result<Vec<StirVerifyOutputs<EF>>, StirError<M::Error, IE>>
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
    verify_stir_multi_inner(configs, proofs, challenger, Some(initial_fibers))
}

/// Shared body of [`verify_stir_multi`] and [`verify_stir_multi_with_external_initial`].
///
/// Mirrors `prover::prove_stir_multi_inner`'s right-aligned lockstep schedule.
///
/// At each global round:
/// - every active instance's replicated grind witness must agree,
/// - else [`ProofShapeError::ReplicatedWitnessMismatch`],
/// - then the shared grind is checked once, at the max of the active instances' bits.
fn verify_stir_multi_inner<F, EF, M, Challenger, IE, Src>(
    configs: &[&StirConfig<F, EF, M, Challenger>],
    proofs: &[&StirProof<EF, M, Challenger::Witness>],
    challenger: &mut Challenger,
    external_fibers: Option<Vec<Src>>,
) -> Result<Vec<StirVerifyOutputs<EF>>, StirError<M::Error, IE>>
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
    let b = configs.len();
    if proofs.len() != b {
        return Err(ProofShapeError::InstanceCount {
            expected: b,
            got: proofs.len(),
        }
        .into());
    }

    // Why: an empty batch is a transcript no-op on both sides.
    //   prover:   shared grinds fall back to zero bits
    //   verifier: check_witness(0, _) observes nothing
    if b == 0 {
        return Ok(Vec::new());
    }

    for i in 0..b {
        if proofs[i].round_proofs.len() != configs[i].num_rounds() {
            return Err(ProofShapeError::RoundCount {
                instance: Some(i),
                expected: configs[i].num_rounds(),
                got: proofs[i].round_proofs.len(),
            }
            .into());
        }
    }

    let initial_is_external = external_fibers.is_some();
    let mut external_fibers: Vec<Option<Src>> = match external_fibers {
        None => (0..b).map(|_| None).collect(),
        Some(sources) => {
            if sources.len() != b {
                return Err(ExternalSourceError::SourceCount {
                    expected: b,
                    got: sources.len(),
                }
                .into());
            }
            sources.into_iter().map(Some).collect()
        }
    };

    // The initial oracle is either committed by STIR — in which case its commitment enters the
    // transcript here — or external, in which case the caller has already bound it and every
    // proof must carry no commitment at all.
    let mut shifts = Vec::with_capacity(b);
    let mut log_domains = Vec::with_capacity(b);
    for i in 0..b {
        match (&proofs[i].initial_commitment, initial_is_external) {
            (Some(commitment), false) => challenger.observe(commitment.clone()),
            (None, true) => {}
            (Some(_), true) => return Err(ProofShapeError::UnexpectedInitialCommitment.into()),
            (None, false) => return Err(ProofShapeError::MissingInitialCommitment.into()),
        }
        shifts.push(F::GENERATOR);
        log_domains.push(configs[i].log_starting_domain_size());
    }

    let max_m = configs.iter().map(|c| c.num_rounds()).max().unwrap_or(0);
    let offset = |i: usize| max_m - configs[i].num_rounds();

    let mut prev_ctx: Vec<Option<VirtualRoundContext<EF>>> = (0..b).map(|_| None).collect();
    let mut first_round_pairs: Vec<FirstRoundPairs<EF>> = (0..b).map(|_| Vec::new()).collect();

    for r in 0..max_m {
        let active: Vec<usize> = (0..b).filter(|&i| offset(i) <= r).collect();

        let mut rvs: Vec<RoundVerifier<F, EF>> = active
            .iter()
            .map(|&i| {
                RoundVerifier::<F, EF>::new(
                    &configs[i].round_configs[r - offset(i)],
                    shifts[i],
                    log_domains[i],
                )
            })
            .collect();

        // [grind folding_pow_bits]: every active instance's replicated witness must agree.
        let folding_witnesses: Vec<F> = active
            .iter()
            .map(|&i| proofs[i].round_proofs[r - offset(i)].folding_pow_witness)
            .collect();
        if folding_witnesses.windows(2).any(|w| w[0] != w[1]) {
            return Err(ProofShapeError::ReplicatedWitnessMismatch {
                round: RoundLabel::Round(r),
                stage: GrindStage::Folding,
            }
            .into());
        }
        let shared_folding_bits = active
            .iter()
            .map(|&i| configs[i].round_configs[r - offset(i)].folding_pow_bits)
            .max()
            .expect("`active` is non-empty for r < max_m");
        if !challenger.check_witness(shared_folding_bits, folding_witnesses[0]) {
            return Err(StirError::InvalidPowWitness {
                round: RoundLabel::Round(r),
            });
        }

        // Phase 1: per-instance folding challenge and commitment absorb.
        for (&i, rv) in active.iter().zip(rvs.iter_mut()) {
            let gamma: EF = challenger.sample_algebra_element();
            challenger.observe(proofs[i].round_proofs[r - offset(i)].commitment.clone());
            rv.set_gamma(gamma, shifts[i]);
        }

        // Phase 2: per-instance OOD sampling and answer absorb.
        for (&i, rv) in active.iter().zip(rvs.iter_mut()) {
            let local_r = r - offset(i);
            let rc = &configs[i].round_configs[local_r];
            let rp = &proofs[i].round_proofs[local_r];
            if rp.ood_answers.len() != rc.num_ood_samples {
                return Err(ProofShapeError::OodAnswerCount {
                    round: RoundLabel::Round(local_r),
                    expected: rc.num_ood_samples,
                    got: rp.ood_answers.len(),
                }
                .into());
            }
            rv.ood_points = sample_ood_points(
                challenger,
                rv.excluded_domains(shifts[i], log_domains[i]),
                rc.num_ood_samples,
            );
            challenger.observe_algebra_slice(&rp.ood_answers);
        }

        // [grind pow_bits]: every active instance's replicated witness must agree.
        let query_witnesses: Vec<F> = active
            .iter()
            .map(|&i| proofs[i].round_proofs[r - offset(i)].pow_witness)
            .collect();
        if query_witnesses.windows(2).any(|w| w[0] != w[1]) {
            return Err(ProofShapeError::ReplicatedWitnessMismatch {
                round: RoundLabel::Round(r),
                stage: GrindStage::Query,
            }
            .into());
        }
        let shared_pow_bits = active
            .iter()
            .map(|&i| configs[i].round_configs[r - offset(i)].pow_bits)
            .max()
            .expect("`active` is non-empty for r < max_m");
        if !challenger.check_witness(shared_pow_bits, query_witnesses[0]) {
            return Err(StirError::InvalidPowWitness {
                round: RoundLabel::Round(r),
            });
        }

        // Phase 3: per-instance combination challenge, query sampling, and fiber
        // materialization/folding.
        for (&i, rv) in active.iter().zip(rvs.iter_mut()) {
            let local_r = r - offset(i);
            let rc = &configs[i].round_configs[local_r];
            let r_comb: EF = challenger.sample_algebra_element();
            rv.r_comb = r_comb;

            let mut query_indices: Vec<usize> = Vec::with_capacity(rc.num_queries);
            for _ in 0..rc.num_queries {
                let j = challenger
                    .sample_uniform_bits::<true>(rv.fold_log_domain)
                    .expect("RESAMPLE = true: rejection loops internally, never errors");
                query_indices.push(j);
            }

            let commitment = if local_r == 0 {
                proofs[i].initial_commitment.as_ref()
            } else {
                Some(&proofs[i].round_proofs[local_r - 1].commitment)
            };
            let is_external = local_r == 0 && initial_is_external;

            rv.fetch_and_fold(
                configs[i],
                local_r,
                &proofs[i].round_proofs[local_r],
                shifts[i],
                log_domains[i],
                prev_ctx[i].as_ref(),
                is_external,
                &mut external_fibers[i],
                commitment,
                &query_indices,
            )?;
        }

        // Phase 4: per-instance ans/shake absorb, shake-check challenge, and consistency.
        let mut finishes: Vec<(usize, RoundVerifyOutput<F, EF>)> = Vec::with_capacity(active.len());
        for (&i, rv) in active.iter().zip(rvs) {
            let local_r = r - offset(i);
            let rp = &proofs[i].round_proofs[local_r];
            let (all_points, all_values) = rv.all_points_and_values(&rp.ood_answers);

            let max_ans_len = all_points.len();
            check_ans_and_shake_lengths(
                RoundLabel::Round(local_r),
                &rp.ans_polynomial,
                &rp.shake_polynomial,
                max_ans_len,
            )?;

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
                return Err(StirError::InvalidShakeConsistency {
                    round: RoundLabel::Round(local_r),
                });
            }

            finishes.push((i, rv.finish(rp.ans_polynomial.clone(), &all_points)));
        }

        // Phase 5: touches no transcript state, so instance order no longer matters.
        for (i, output) in finishes {
            first_round_pairs[i].extend(output.first_round_pairs);
            prev_ctx[i] = Some(output.ctx);
            shifts[i] = output.next_shift;
            log_domains[i] = output.next_log_domain;
        }
    }

    // Final round: every instance reaches it on this same global step (right-alignment).
    let final_folding_witnesses: Vec<F> =
        proofs.iter().map(|p| p.final_folding_pow_witness).collect();
    if final_folding_witnesses.windows(2).any(|w| w[0] != w[1]) {
        return Err(ProofShapeError::ReplicatedWitnessMismatch {
            round: RoundLabel::Final,
            stage: GrindStage::Folding,
        }
        .into());
    }
    let shared_final_folding_bits = configs
        .iter()
        .map(|c| c.final_folding_pow_bits)
        .max()
        .unwrap_or(0);
    if !challenger.check_witness(shared_final_folding_bits, final_folding_witnesses[0]) {
        return Err(StirError::InvalidPowWitness {
            round: RoundLabel::Final,
        });
    }

    let mut fvs: Vec<FinalRoundVerifier<F, EF>> = (0..b)
        .map(|i| {
            FinalRoundVerifier::<F, EF>::new(
                configs[i].final_log_folding_factor(),
                shifts[i],
                log_domains[i],
            )
        })
        .collect();

    for i in 0..b {
        let final_gamma: EF = challenger.sample_algebra_element();
        fvs[i].set_gamma(final_gamma, shifts[i]);

        let expected_final_len = configs[i].final_poly_len();
        if proofs[i].final_polynomial.len() != expected_final_len {
            return Err(ProofShapeError::FinalPolynomialLength {
                expected: expected_final_len,
                got: proofs[i].final_polynomial.len(),
            }
            .into());
        }
        challenger.observe_algebra_slice(&proofs[i].final_polynomial);
    }

    let final_query_witnesses: Vec<F> = proofs.iter().map(|p| p.final_pow_witness).collect();
    if final_query_witnesses.windows(2).any(|w| w[0] != w[1]) {
        return Err(ProofShapeError::ReplicatedWitnessMismatch {
            round: RoundLabel::Final,
            stage: GrindStage::Query,
        }
        .into());
    }
    let shared_final_pow_bits = configs.iter().map(|c| c.final_pow_bits).max().unwrap_or(0);
    if !challenger.check_witness(shared_final_pow_bits, final_query_witnesses[0]) {
        return Err(StirError::InvalidPowWitness {
            round: RoundLabel::Final,
        });
    }

    for i in 0..b {
        let mut final_indices: Vec<usize> = Vec::with_capacity(configs[i].final_queries);
        for _ in 0..configs[i].final_queries {
            let j = challenger
                .sample_uniform_bits::<true>(fvs[i].final_new_log_domain)
                .expect("RESAMPLE = true: rejection loops internally, never errors");
            final_indices.push(j);
        }

        let commitment = if configs[i].num_rounds() == 0 {
            proofs[i].initial_commitment.as_ref()
        } else {
            Some(
                &proofs[i]
                    .round_proofs
                    .last()
                    .expect("num_rounds() > 0")
                    .commitment,
            )
        };
        let is_external = configs[i].num_rounds() == 0 && initial_is_external;

        let final_pairs = fvs[i].fetch_and_check(
            configs[i],
            proofs[i],
            configs[i].num_rounds(),
            shifts[i],
            log_domains[i],
            prev_ctx[i].as_ref(),
            is_external,
            &mut external_fibers[i],
            commitment,
            &final_indices,
        )?;
        first_round_pairs[i].extend(final_pairs);
    }

    Ok(first_round_pairs
        .into_iter()
        .map(|mut pairs| {
            pairs.sort_by_key(|(j, _)| *j);
            let (first_round_indices, first_round_fiber_evals): (Vec<_>, Vec<_>) =
                pairs.into_iter().unzip();
            StirVerifyOutputs {
                first_round_indices,
                first_round_fiber_evals,
            }
        })
        .collect())
}
