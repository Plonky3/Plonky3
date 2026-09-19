//! Suffix-mode stacked-sumcheck prover.

use alloc::vec;
use alloc::vec::Vec;

use p3_challenger::fs::TranscriptField;
use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{Algebra, ExtensionField, Field, dot_product};
use p3_maybe_rayon::prelude::*;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use p3_multilinear_util::split_eq::SplitEq;

use crate::lagrange::lagrange_weights_01inf_multi;
use crate::layout::opening::{EqSvoPartials, NextSvoPartials, Opening, ProverMultiClaim};
use crate::layout::prover::{Layout, StackedClaims, SuffixResidualProver};
use crate::layout::witness::{Table, column_slots};
use crate::layout::{LayoutStrategy, Witness};
use crate::product_polynomial::ProductPolynomial;
use crate::strategy::{
    Basis, FromTable, IntoTranscriptField, ReprSumcheckProver, SumcheckProver, VariableOrder,
};
use crate::svo::{SvoPoint, calculate_accumulators_batch};
use crate::table::{OpeningBatch, OpeningEvals, OpeningRequest};
use crate::transcript::{ProverTranscript, SumcheckShape};
use crate::{Claim, SumcheckData, extrapolate_01inf};

/// Largest table arity whose openings share one dense weight table per batch.
///
/// A dense table holds one extension element per row, so it is capped here.
const SHARED_WEIGHTS_MAX_VARIABLES: usize = 20;

/// Rows per chunk of a weighted column sum.
///
/// A column up to this length is summed serially; a longer one sums its chunks in parallel.
const WEIGHTED_SUM_CHUNK: usize = 1 << 12;

/// Entries per chunk when a residual factor fills one column slot.
///
/// A slot up to this length is filled serially; a longer one fills its chunks in parallel, so a
/// single tall column is not left to one thread.
const SLOT_CHUNK: usize = 1 << 12;

/// One claim's residual weight tables at unit scale, over a single column slot.
struct ClaimWeightTables<EF> {
    /// Equality weights, present when the claim opens a column directly.
    current: Option<Vec<EF>>,
    /// Repeat-last successor weights, present when the claim opens a successor view.
    next: Option<Vec<EF>>,
}

/// One source table's contributions to the residual weight polynomial, indexed by column.
///
/// Each contribution is `(claim index, is successor, alpha power)`.
type ColumnWeights<EF> = Vec<Vec<(usize, bool, EF)>>;

/// Every source table's [`ColumnWeights`], indexed by source table.
type WeightPlan<EF> = Vec<ColumnWeights<EF>>;

impl<EF: Copy + Send + Sync> ClaimWeightTables<EF> {
    /// Both tables, moved into another field a whole table at a time.
    fn into_image<R: FromTable<EF>>(self) -> ClaimWeightTables<R> {
        ClaimWeightTables {
            current: self.current.map(R::from_table),
            next: self.next.map(R::from_table),
        }
    }
}

/// Stacked-sumcheck prover with suffix-first variable binding.
///
/// # Flow
///
/// - SVO accumulators are precomputed at claim-recording time.
/// - Each preprocessing round reads its slice of those accumulators.
/// - The residual product polynomial is built once, after all rounds.
#[derive(Debug, Clone)]
pub struct SuffixProver<F: Field, EF: ExtensionField<F>> {
    /// Recorded opening claims and the layout context that batches them.
    ///
    /// - Suffix binding walks the per-table data directly.
    /// - No separate copy of the stacked polynomial is kept.
    pub(crate) claims: StackedClaims<F, EF>,
}

impl<F: Field, EF: ExtensionField<F>> Layout<F, EF> for SuffixProver<F, EF> {
    fn from_witness(witness: Witness<F>) -> Self {
        // Move the witness fields out so the prover owns them outright.
        // The stacked polynomial is intentionally discarded: every suffix-mode
        // primitive walks the per-table data instead.
        let parts = witness.into_parts();
        Self {
            claims: StackedClaims::new(
                parts.tables,
                parts.placements,
                parts.num_variables,
                parts.folding,
            ),
        }
    }

    fn new_witness(tables: Vec<Table<F>>, folding: usize) -> Witness<F> {
        Witness::new(tables, folding)
    }

    fn claims(&self) -> &StackedClaims<F, EF> {
        &self.claims
    }

    /// Evaluates the selected columns of one table at a point and records the claim.
    ///
    /// All requested columns share the one supplied local opening point.
    ///
    /// - Direct openings evaluate a column at that point.
    /// - Successor openings evaluate the repeat-last view at that point.
    /// - Returned evaluations list every direct opening first.
    ///
    /// # Arguments
    ///
    /// - Source table index.
    /// - Columns opened directly and through the successor view.
    /// - Local-frame opening point, one coordinate per table variable.
    ///
    /// # Performance
    ///
    /// - The point is factorised once and reused by every selected column.
    /// - Each column is an independent linear pass, so columns run in parallel.
    /// - Without SVO rounds, for a table of at most `2^SHARED_WEIGHTS_MAX_VARIABLES` rows and a
    ///   batch opening more than one column, the equality and successor weight tables are built
    ///   once per call. Each column is then one weighted sum against them, free of products on
    ///   bit-valued rows.
    #[tracing::instrument(skip_all, level = "debug")]
    fn record_opening(
        &mut self,
        table_idx: usize,
        batch: &OpeningRequest,
        point: &Point<EF>,
    ) -> OpeningEvals<EF> {
        // Split the request into its two column groups.
        let current = batch.current();
        let next = batch.next();

        let table = &self.claims.tables[table_idx];
        // The opening point lives in the table's local frame, one coordinate per variable.
        debug_assert_eq!(point.num_variables(), table.num_variables());

        // Without SVO rounds an opening carries no per-round residuals.
        // Each column then reduces to one weighted sum against a table shared by the whole batch.
        // Taller tables keep the factored evaluation, whose weights stay at the square root size.
        // So does a single-column batch, which cannot amortize a dense table.
        if self.claims.folding == 0
            && table.num_variables() <= SHARED_WEIGHTS_MAX_VARIABLES
            && current.len() + next.len() >= 2
        {
            // Equality weights of the point over every row of the table.
            let eq = Poly::new_from_point(point.as_slice(), EF::ONE);
            // Repeat-last successor weights, derived from the equality weights when needed.
            let next_weights = (!next.is_empty()).then(|| successor_weights(eq.as_slice()));

            let (current_openings, current_evals): (Vec<_>, Vec<EF>) = current
                .into_par_iter()
                .copied()
                .map(|poly_idx| {
                    let eval = weighted_sum(eq.as_slice(), table.poly(poly_idx).as_slice());
                    let partial_evals = EqSvoPartials::new(Vec::new());
                    (Opening::new_with_data(poly_idx, eval, partial_evals), eval)
                })
                .unzip();

            let (next_openings, next_evals): (Vec<_>, Vec<EF>) = next
                .into_par_iter()
                .copied()
                .map(|poly_idx| {
                    let weights = next_weights.as_deref().unwrap();
                    let eval = weighted_sum(weights, table.poly(poly_idx).as_slice());
                    let partial_evals = NextSvoPartials::new(Vec::new());
                    (Opening::new_with_data(poly_idx, eval, partial_evals), eval)
                })
                .unzip();

            // Record the claim at the zero-round factorisation of the point.
            self.claims.claim_map[table_idx].push(ProverMultiClaim::new(
                SvoPoint::new_unpacked(0, point, VariableOrder::Suffix),
                current_openings,
                next_openings,
            ));

            return OpeningBatch::new(current_evals, next_evals);
        }

        // Factorise the point with the suffix split; every selected column reuses it.
        let point = SvoPoint::new_unpacked(self.claims.folding, point, VariableOrder::Suffix);

        // Current group: evaluate each column at the point.
        // Each entry yields an opening (carrying preprocessing residuals) plus the bare eval.
        // Each column is an independent O(2^n) pass, so columns evaluate in parallel.
        let (current_openings, current_evals): (Vec<_>, Vec<EF>) = current
            .into_par_iter()
            .copied()
            .map(|poly_idx| {
                let (eval, partial_evals) = point.eval(table.poly(poly_idx));
                (Opening::new_with_data(poly_idx, eval, partial_evals), eval)
            })
            .unzip();

        // Next group: evaluate the repeat-last successor view at the same point.
        // `current_openings` is fully built above and only read here, so this is safe to
        // run in parallel alongside it.
        let (next_openings, next_evals): (Vec<_>, Vec<EF>) = next
            .into_par_iter()
            .copied()
            .map(|poly_idx| {
                // Reuse: if this column was already opened in the current group,
                // its last-round equality residual feeds the successor computation
                // for free, avoiding a redundant pass over the column.
                //
                // The per-round list is empty when there are no SVO rounds (folding
                // is zero), so fall back to recomputing the residual rather than
                // unwrapping an absent last round.
                let d_eq = current_openings
                    .iter()
                    .find(|opening| opening.poly_idx() == Some(poly_idx))
                    .and_then(|opening| opening.data().rounds().last())
                    .map(|round| round.poly());
                let (eval, partial_evals) = point.eval_next_suffix(table.poly(poly_idx), d_eq);
                (Opening::new_with_data(poly_idx, eval, partial_evals), eval)
            })
            .unzip();

        // Store the batch with its shared SVO point.
        self.claims.claim_map[table_idx].push(ProverMultiClaim::new(
            point,
            current_openings,
            next_openings,
        ));

        // Return both eval groups in the canonical current-then-next order.
        OpeningBatch::new(current_evals, next_evals)
    }

    /// Evaluates the full stacked polynomial at a point and records the claim.
    ///
    /// # Overview
    ///
    /// WHIR pins the stacked polynomial at a fresh point for soundness amplification.
    ///
    /// The stacked evaluation factors per column through the slot selector:
    ///
    /// ```text
    ///     stacked(point) = sum_{i}  eq(selector_i, point_selector_part)
    ///                               * col_i(point_local_part)
    /// ```
    ///
    /// # Flow
    ///
    /// - Each column is evaluated at its local sub-point.
    /// - Per-column partials are collected on the fly.
    /// - Those partials feed the preprocessing accumulator batcher.
    ///
    /// # Arguments
    ///
    /// - Point covering every stacked variable.
    #[tracing::instrument(skip_all)]
    fn record_virtual(&mut self, point: &Point<EF>) -> EF {
        // Per-column accumulation state:
        //
        // - eval    : running stacked evaluation.
        // - openings: one virtual opening per column, carrying SVO partials.
        // - weights : per-column selector-equality scalars.
        let mut eval = EF::ZERO;
        let mut openings = Vec::new();
        let mut weights = Vec::new();

        for placement in &self.claims.placements {
            for (poly_idx, selector) in placement.selectors().iter().enumerate() {
                // Source column behind this slot.
                let poly = self.claims.tables[placement.idx()].poly(poly_idx);

                // Split the challenge into (selector_bits, local_bits).
                let (selector_part, local_part) = point.split_at(selector.num_variables());

                // Scalar weight: eq(selector, selector_part) for this column.
                let weight =
                    Point::eval_eq::<EF>(selector.point().as_slice(), selector_part.as_slice());

                // Factorise the local part with the suffix split, then evaluate.
                let local_svo =
                    SvoPoint::new_unpacked(self.claims.folding, &local_part, VariableOrder::Suffix);
                let (column_eval, partial_evals) = local_svo.eval(poly);

                // Record a virtual opening (no source column tag) with partials.
                let opening = Opening {
                    poly_idx: None,
                    eval: column_eval,
                    data: partial_evals,
                };

                // Add the weighted column evaluation into the stacked total.
                eval += weight * column_eval;

                // Stash opening and weight for the accumulator-batcher call.
                openings.push(opening);
                weights.push(weight);
            }
        }

        // Batch every per-column opening into per-round SVO accumulators.
        let accumulators = calculate_accumulators_batch(
            &ProverMultiClaim::new(
                SvoPoint::new_unpacked(self.claims.folding, point, VariableOrder::Suffix),
                openings,
                Vec::new(),
            ),
            &weights,
        );

        // Debug-only consistency check:
        //
        // - hand-rolled weighted sum must equal the direct stacked evaluation.
        // - accumulators batched per column must equal the single-opening batch.
        #[cfg(debug_assertions)]
        {
            // Materialise the stacked polynomial with no challenges applied.
            let poly = &self.compress_stacked(&Point::default());
            // Check 1: weighted sum equals the direct evaluation.
            assert_eq!(eval, poly.eval_base(point));

            // Build the reference opening by evaluating the materialised poly directly.
            let ref_svo =
                SvoPoint::<EF, EF>::new_unpacked(self.claims.folding, point, VariableOrder::Suffix);
            let (ref_eval, ref_partials) = ref_svo.eval(poly.as_view());
            let opening = Opening {
                poly_idx: None,
                eval: ref_eval,
                data: ref_partials,
            };
            // Check 2: the reference evaluation matches the weighted one.
            assert_eq!(eval, ref_eval);
            // Check 3: accumulators from per-column batching match the single-opening batch.
            assert_eq!(
                accumulators,
                calculate_accumulators_batch(
                    &ProverMultiClaim::new(
                        SvoPoint::new_unpacked(self.claims.folding, point, VariableOrder::Suffix),
                        vec![opening],
                        Vec::new(),
                    ),
                    &[EF::ONE],
                ),
            );
        }

        // Record the claim so the folding rounds can read its accumulators.
        self.claims.virtual_claims.push(Claim {
            point: point.clone(),
            eval,
            data: accumulators,
        });

        eval
    }

    /// Finalises SVO preprocessing and returns the residual sumcheck prover.
    ///
    /// # Returns
    ///
    /// - Residual sumcheck prover over the unpacked product polynomial.
    /// - Folding challenges sampled during preprocessing.
    ///
    /// # Algorithm
    ///
    /// ```text
    ///     Phase | Action
    ///     ------+------------------------------------------------------------
    ///       1   | Sample batching challenge  a; flatten alphas by opening_idx.
    ///       2   | Pre-batch per-claim accumulators with the a-powers.
    ///       3   | Loop over preprocessing rounds:
    ///               a. (h(0), h(inf)) = dot(accumulators, Lagrange weights).
    ///               b. Sample challenge r; extrapolate the running sum.
    ///       4   | Compose the residual product polynomial from compressed slots.
    /// ```
    #[tracing::instrument(skip_all)]
    fn into_sumcheck<Ch>(
        self,
        sumcheck_data: &mut SumcheckData<F, EF>,
        pow_bits: usize,
        challenger: &mut Ch,
    ) -> (SumcheckProver<F, EF>, Point<EF>)
    where
        F: TranscriptField,
        Ch: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        let (alpha, sum, rs) = self.preprocess(sumcheck_data, pow_bits, challenger);

        // Stage D: materialise the residual product polynomial.
        //
        // - Suffix binding folds variables in reverse.
        // - The residual poly therefore lives in the reversed-challenges frame.
        let reversed = rs.reversed();
        // Factor 1 of the product: the compressed stacked poly at rs.
        // No external scaling here; the plain path keeps the running sum unchanged.
        let compressed = self.compress_stacked(&reversed);
        // Factor 2 of the product: the batched equality-weight poly.
        let weights = self.combine_weights(&reversed, alpha);
        // Pair them; the product polynomial drives the remaining rounds.
        let poly = ProductPolynomial::new_unpacked(VariableOrder::Suffix, compressed, weights);
        // Cross-check: the dot product of the two factors must equal the
        // running sum accumulated across the preprocessing rounds.
        debug_assert_eq!(poly.dot_product(), sum);

        (SumcheckProver::new(poly, sum), rs)
    }

    fn strategy() -> LayoutStrategy {
        LayoutStrategy::new(false, VariableOrder::Suffix)
    }
}

impl<F: Field, EF: ExtensionField<F>> SuffixProver<F, EF> {
    /// Runs the SVO preprocessing rounds.
    ///
    /// # Returns
    ///
    /// - Batching challenge that weights the recorded openings.
    /// - Running claimed sum after the preprocessing rounds.
    /// - Folding challenges, in sampling order.
    fn preprocess<Ch>(
        &self,
        sumcheck_data: &mut SumcheckData<F, EF>,
        pow_bits: usize,
        challenger: &mut Ch,
    ) -> (EF, EF, Point<EF>)
    where
        F: TranscriptField,
        Ch: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        // Sanity: preprocessing cannot consume more rounds than the stacked arity.
        assert!(self.claims.folding <= self.claims.num_variables);

        // The batching challenge seeds a sub-transcript of its own.
        //
        // Both claim counts therefore reach the sponge before the challenge is drawn.
        let alpha: EF = self.batching_challenge(challenger);
        let n_claims = self.num_claims();

        // Stage A: batch per-claim accumulators using insertion-order alpha powers.
        //
        // - Iteration order is placement order, matching `sum` and `combine_weights`.
        // - Each claim consumes exactly `claim.len()` consecutive powers from
        //   the shared iterator, so the per-claim alpha vector is aligned with
        //   the claim's opening list by construction.
        let mut alphas = alpha.powers();
        let accumulators: Vec<_> = self
            .claims
            .concrete_claims()
            .map(|claim| {
                let per_claim: Vec<EF> = alphas.by_ref().take(claim.len()).collect();
                calculate_accumulators_batch(claim, &per_claim)
            })
            .collect();

        // Stage C: drive the preprocessing rounds from the accumulators.
        let mut sum = self.claims.sum(alpha);
        let mut rs: Vec<EF> = vec![];

        // First alpha power assigned to the virtual claims, sitting just past the concrete claims.
        // The claim count is fixed for the whole fold, so this exponentiation is loop-invariant.
        let alpha_base = alpha.exp_u64(n_claims as u64);

        // One driver spans the whole preprocessing batch, so the description is walked exactly once.
        let shape = SumcheckShape::new(self.claims.folding, pow_bits, Basis::Evaluation);
        let mut transcript = ProverTranscript::<Ch, F, EF>::new(challenger, shape);

        for round_idx in 0..self.claims.folding {
            // Lagrange weights at the challenges sampled so far.
            let weights = lagrange_weights_01inf_multi(&rs);

            // Round-coefficient identity (linearity of the dot product):
            //
            //     c0    = sum_c  dot(claim_c.accs[0], weights)
            //           + sum_v  alpha_v * dot(virtual_v.accs[0], weights)
            //     c_inf = same with accs[1]
            //
            // - Concrete claims carry alpha pre-batched in stage B.
            // - Virtual claims keep a separate scalar per claim.
            // - No intermediate element-wise accumulator is needed.
            let mut c0 = EF::ZERO;
            let mut c_inf = EF::ZERO;

            for accs in &accumulators {
                c0 += dot_product::<EF, _, _>(
                    accs[round_idx][0].iter().copied(),
                    weights.iter().copied(),
                );
                c_inf += dot_product::<EF, _, _>(
                    accs[round_idx][1].iter().copied(),
                    weights.iter().copied(),
                );
            }

            // Virtual-claim contributions: scale each claim's dot by its alpha power.
            for (vc, alpha_i) in self
                .claims
                .virtual_claims
                .iter()
                .zip(alpha.shifted_powers(alpha_base))
            {
                let vc_accs = &vc.data;
                c0 += alpha_i
                    * dot_product::<EF, _, _>(
                        vc_accs[round_idx][0].iter().copied(),
                        weights.iter().copied(),
                    );
                c_inf += alpha_i
                    * dot_product::<EF, _, _>(
                        vc_accs[round_idx][1].iter().copied(),
                        weights.iter().copied(),
                    );
            }

            // Observe coefficients, sample r, extrapolate the running sum.
            let r = sumcheck_data.observe_and_sample(&mut transcript, c0, c_inf);
            sum = extrapolate_01inf(c0, sum - c0, c_inf, r);
            rs.push(r);
        }

        // Require that every described step was played.
        transcript.finish();

        (alpha, sum, Point::new(rs))
    }

    /// Finalises SVO preprocessing and returns the residual prover over tables in `R`.
    ///
    /// Plays exactly the transcript [`Layout::into_sumcheck`] plays; only the field
    /// the residual tables are held in differs.
    ///
    /// Takes the row-first route of [`SuffixResidualProver`] when there is more than one column,
    /// no preprocessing round, no virtual claim, every table has the same height, and every
    /// column's weights are one shared row table times a scale of its own. Takes the dense route
    /// otherwise.
    ///
    /// # Returns
    ///
    /// - Residual sumcheck prover whose tables live in `R`.
    /// - Folding challenges sampled during preprocessing.
    #[tracing::instrument(skip_all)]
    pub fn into_sumcheck_in<R, Ch>(
        self,
        sumcheck_data: &mut SumcheckData<F, EF>,
        pow_bits: usize,
        challenger: &mut Ch,
    ) -> (SuffixResidualProver<F, EF, R>, Point<EF>)
    where
        F: TranscriptField,
        R: IntoTranscriptField<EF> + Algebra<F>,
        Ch: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        let (alpha, sum, rs) = self.preprocess(sumcheck_data, pow_bits, challenger);

        // Suffix binding folds variables in reverse, so the residual factors live in the
        // reversed-challenges frame.
        let reversed = rs.reversed();
        let (claim_tables, column_weights) = self.weight_plan(&reversed, alpha);

        // A single column gains nothing from the row-first route: its rows are the whole stacked
        // space, so both routes play every round over tables of the same size.
        let row_first = self
            .claims
            .tables
            .iter()
            .map(Table::num_polys)
            .sum::<usize>()
            > 1;
        if let Some((row_weights, scales)) = row_first
            .then(|| self.shared_row_weights(&claim_tables, &column_weights))
            .flatten()
        {
            let StackedClaims {
                tables,
                placements,
                num_variables,
                ..
            } = self.claims;
            let prover = SuffixResidualProver::row_first(
                tables,
                placements,
                scales,
                row_weights,
                sum,
                num_variables,
            );
            return (prover, rs);
        }

        let prover = self.dense_residual(&reversed, alpha, sum, claim_tables, column_weights);
        (SuffixResidualProver::dense(prover), rs)
    }

    /// Builds the residual prover over the whole stacked space from a weight plan.
    ///
    /// # Arguments
    ///
    /// - `rs` — suffix challenges already sampled, in the reversed frame.
    /// - `alpha` — the batching challenge.
    /// - `sum` — the running claim after preprocessing.
    /// - `claim_tables`, `column_weights` — the [`Self::weight_plan`] at `rs` and `alpha`.
    fn dense_residual<R>(
        &self,
        rs: &Point<EF>,
        alpha: EF,
        sum: EF,
        claim_tables: Vec<ClaimWeightTables<EF>>,
        column_weights: WeightPlan<EF>,
    ) -> ReprSumcheckProver<F, EF, R>
    where
        R: IntoTranscriptField<EF>,
    {
        // Factor 1 of the product: the compressed stacked poly at rs.
        // No external scaling here; the plain path keeps the running sum unchanged.
        let compressed = self.compress_stacked(rs);
        // Factor 2 of the product: the batched equality-weight poly, accumulated in `R`.
        let weights = self.combine_weights_in::<R>(rs, alpha, claim_tables, column_weights);

        ReprSumcheckProver::from_tables(VariableOrder::Suffix, compressed, weights, sum)
    }

    /// Splits the residual weights into one row table every column shares and a scale per column.
    ///
    /// ```text
    ///     W(c, x) = s_c * T(x)
    /// ```
    ///
    /// The reference is the first opened column. Every other opened column must list the same
    /// claim tables in the same order, with coefficients proportional to the reference's.
    ///
    /// # Returns
    ///
    /// - The row table `T`, normalised so the reference's first coefficient is one.
    /// - Per source table and column, the scale `s_c`; zero for a column never opened.
    ///
    /// `None` when any of these fails, since the row-first route needs all of them:
    ///
    /// - no preprocessing rounds and no virtual claims,
    /// - every source table has the same height,
    /// - at least one opening, and the reference's first coefficient is nonzero,
    /// - every opened column's coefficients are proportional to the reference's.
    fn shared_row_weights(
        &self,
        claim_tables: &[ClaimWeightTables<EF>],
        column_weights: &WeightPlan<EF>,
    ) -> Option<(Vec<EF>, Vec<Vec<EF>>)> {
        if self.claims.folding != 0 || !self.claims.virtual_claims.is_empty() {
            return None;
        }
        let row_variables = self.claims.tables.first()?.num_variables();
        if self
            .claims
            .tables
            .iter()
            .any(|table| table.num_variables() != row_variables)
        {
            return None;
        }

        let reference = column_weights
            .iter()
            .flatten()
            .find(|terms| !terms.is_empty())?;
        let lead = reference[0].2;
        let lead_inv = lead.try_inverse()?;

        // Proportionality is checked by cross-multiplying, so only the reference needs an inverse.
        //
        //     terms[j] / terms[0] == reference[j] / reference[0]
        let proportional = |terms: &[(usize, bool, EF)], scale: EF| {
            terms.len() == reference.len()
                && terms.iter().zip(reference).all(|(term, reference_term)| {
                    term.0 == reference_term.0
                        && term.1 == reference_term.1
                        && term.2 * lead == reference_term.2 * scale
                })
        };
        let scales = column_weights
            .iter()
            .map(|columns| {
                columns
                    .iter()
                    .map(|terms| match terms.first() {
                        None => Some(EF::ZERO),
                        Some(&(_, _, scale)) => proportional(terms, scale).then_some(scale),
                    })
                    .collect::<Option<Vec<EF>>>()
            })
            .collect::<Option<Vec<Vec<EF>>>>()?;

        // The shared row table: the reference's own tables, over its first coefficient.
        //
        //     T = table_0 + sum_{j > 0}  (reference[j] / reference[0]) * table_j
        let table = |&(claim_idx, is_next, _): &(usize, bool, EF)| {
            let claim_tables = &claim_tables[claim_idx];
            let table = if is_next {
                &claim_tables.next
            } else {
                &claim_tables.current
            };
            table.as_deref().unwrap()
        };
        let mut row_weights = table(&reference[0]).to_vec();
        for term in &reference[1..] {
            let ratio = term.2 * lead_inv;
            row_weights
                .par_iter_mut()
                .zip(table(term))
                .for_each(|(out, &weight)| *out += ratio * weight);
        }

        Some((row_weights, scales))
    }

    /// Compress every stacked-table slot by fixing the suffix challenges.
    #[tracing::instrument(skip_all)]
    pub(crate) fn compress_stacked(&self, rs: &Point<EF>) -> Poly<EF> {
        self.compress_stacked_scaled(rs, EF::ONE)
    }

    /// Compress every stacked-table slot, folding `scale` into the equality table.
    ///
    /// ```text
    ///     out[slot, x_rest] = sum_{y in {0,1}^|r|}  scale * eq(r, y) * col(x_rest, y)
    /// ```
    ///
    /// One output slot per column.
    /// Writes never overlap.
    /// Output arity equals the stacked arity minus the challenge count.
    ///
    /// # Arguments
    ///
    /// - `rs` — suffix challenges already sampled.
    /// - `scale` — extra factor folded into the equality table.
    ///
    /// # Why a scale parameter
    ///
    /// - A non-unit `scale` lets the caller absorb a combining challenge into
    ///   the residual factor without a second pass.
    ///
    /// # Panics
    ///
    /// - `scale` is zero: a zero scale silently zeroes the residual.
    #[tracing::instrument(skip_all)]
    pub(crate) fn compress_stacked_scaled(&self, rs: &Point<EF>, scale: EF) -> Poly<EF> {
        assert!(rs.num_variables() <= self.claims.num_variables);
        assert!(scale != EF::ZERO, "compress scale must be non-zero");
        // Output spans the residual stacked space.
        // Size is 2^(num_variables - |rs|).
        let mut out = Poly::<EF>::zero(self.claims.num_variables - rs.num_variables());
        let num_folded = rs.num_variables();
        // Bake the scalar into the prefix-half equality table.
        // Each slot compression then returns scale * eq(r, y) * col(...) in one pass.
        let rs = SplitEq::new_unpacked(rs, scale);

        // Slots are disjoint, so columns compress independently.
        // A slot filled from its column alone is also split into chunks, so a tall column spreads.
        column_slots(
            &self.claims.placements,
            &self.claims.tables,
            num_folded,
            out.as_mut_slice(),
        )
        .into_par_iter()
        .for_each(|(slot, table_idx, poly_idx)| {
            let poly = self.claims.tables[table_idx].poly(poly_idx);
            if num_folded == 0 {
                // Nothing to fold: the slot is the column itself, times the scale.
                let chunks = slot
                    .par_chunks_mut(SLOT_CHUNK)
                    .zip(poly.as_slice().par_chunks(SLOT_CHUNK));
                if scale == EF::ONE {
                    chunks.for_each(|(slot, values)| {
                        slot.iter_mut()
                            .zip(values)
                            .for_each(|(out, &value)| *out = value.into());
                    });
                } else {
                    chunks.for_each(|(slot, values)| {
                        slot.iter_mut()
                            .zip(values)
                            .for_each(|(out, &value)| *out = scale * value);
                    });
                }
            } else {
                rs.compress_suffix_into(slot, poly.as_view());
            }
        });
        out
    }

    /// Builds the residual weight polynomial after the SVO rounds.
    ///
    /// # Contributions
    ///
    /// - Concrete claim: factored equality table scaled by
    ///   `alpha^i * eq(svo_part, rs)`, written into the owning slot only.
    /// - Virtual claim: scaled equality table written across the full output.
    #[tracing::instrument(skip_all)]
    pub(crate) fn combine_weights(&self, rs: &Point<EF>, alpha: EF) -> Poly<EF> {
        let (tables, column_weights) = self.weight_plan(rs, alpha);
        self.combine_weights_from_plan::<F, EF>(rs, alpha, &tables, &column_weights)
    }

    /// Builds the residual weight polynomial from a weight plan, every accumulation in `R`.
    ///
    /// Each entry is the `R` image of the same entry [`Self::combine_weights`] builds, since
    /// `R::from` is a field isomorphism and every entry is a polynomial in the converted inputs.
    ///
    /// Each per-claim table spans a single column slot, so the crossing costs one pass over a
    /// slot per claim, against the whole output the products it feeds then cover.
    ///
    /// Packs the virtual-claim equality tables over `R` itself. That is only the wide-SIMD
    /// choice where `R::Packing` already is: a binary field crossing into its own
    /// polynomial-basis representation, for instance. A caller crossing into a prime-field
    /// extension instead, where `R::Packing` collapses to `R`, should pack over `R`'s base field
    /// the way [`Self::combine_weights`] packs over `F`.
    ///
    /// # Arguments
    ///
    /// - `rs` — suffix challenges already sampled.
    /// - `alpha` — the batching challenge.
    /// - `tables`, `column_weights` — the [`Self::weight_plan`] at `rs` and `alpha`.
    #[tracing::instrument(skip_all)]
    fn combine_weights_in<R>(
        &self,
        rs: &Point<EF>,
        alpha: EF,
        tables: Vec<ClaimWeightTables<EF>>,
        column_weights: WeightPlan<EF>,
    ) -> Poly<R>
    where
        R: Field + FromTable<EF>,
    {
        // Each source table is released as its image appears.
        let tables: Vec<ClaimWeightTables<R>> = tables
            .into_iter()
            .map(ClaimWeightTables::into_image)
            .collect();
        let column_weights: WeightPlan<R> = column_weights
            .into_iter()
            .map(|table| {
                table
                    .into_iter()
                    .map(|column| {
                        column
                            .into_iter()
                            .map(|(claim_idx, is_next, scale)| (claim_idx, is_next, R::from(scale)))
                            .collect()
                    })
                    .collect()
            })
            .collect();
        self.combine_weights_from_plan::<R, R>(rs, alpha, &tables, &column_weights)
    }

    /// Builds each claim's residual weight tables and the per-column batching coefficients.
    ///
    /// # Returns
    ///
    /// - One [`ClaimWeightTables`] per concrete claim, in the batching walk order.
    /// - Per source table and column: `(claim index, is successor, alpha power)` per contribution.
    fn weight_plan(
        &self,
        rs: &Point<EF>,
        alpha: EF,
    ) -> (Vec<ClaimWeightTables<EF>>, WeightPlan<EF>) {
        // Preconditions: challenge count matches the folding depth.
        assert_eq!(rs.num_variables(), self.claims.folding);

        // Every opening of a claim shares that claim's point, so its residual weight table is
        // the same for every column it opens. Build each claim's current and successor tables
        // once, at unit scale; both accumulations are linear in the scale.
        //
        // Walk order matches the batched claim, so alpha powers stay aligned:
        // placements, then claims, then current openings, then successor openings.
        let mut alphas = alpha.powers();
        let mut tables: Vec<ClaimWeightTables<EF>> = Vec::new();
        // Per column: `(table index into tables, is successor, alpha power)`.
        let mut column_weights: WeightPlan<EF> = self
            .claims
            .tables
            .iter()
            .map(|table| vec![Vec::new(); table.num_polys()])
            .collect();
        for placement in &self.claims.placements {
            let len = 1usize << (self.num_variables_table(placement.idx()) - rs.num_variables());
            for claim in &self.claims.claim_map[placement.idx()] {
                let claim_idx = tables.len();
                let current = (!claim.current_openings().is_empty()).then(|| {
                    let mut table = EF::zero_vec(len);
                    claim.point().accumulate_into(&mut table, rs, EF::ONE);
                    table
                });
                let next = (!claim.next_openings().is_empty()).then(|| {
                    let mut table = EF::zero_vec(len);
                    claim
                        .point()
                        .accumulate_next_suffix_into(&mut table, rs, EF::ONE);
                    table
                });
                tables.push(ClaimWeightTables { current, next });

                let weights = &mut column_weights[placement.idx()];
                for opening in claim.current_openings() {
                    weights[opening.poly_idx().unwrap()].push((
                        claim_idx,
                        false,
                        alphas.next().unwrap(),
                    ));
                }
                for opening in claim.next_openings() {
                    weights[opening.poly_idx().unwrap()].push((
                        claim_idx,
                        true,
                        alphas.next().unwrap(),
                    ));
                }
            }
        }

        (tables, column_weights)
    }

    /// Accumulates a weight plan into the residual weight polynomial over `R`.
    ///
    /// `B` is the base field the virtual claims' factored equality tables are packed over. Pass
    /// the narrowest field `R` extends whose packing is wide, not `R` itself, unless
    /// `R::Packing` already is `R` and there is no narrower field to gain from.
    ///
    /// # Contributions
    ///
    /// - Concrete claim: its residual table scaled by the column's coefficient, into that
    ///   column's slot only.
    /// - Virtual claim: scaled equality table written across the full output.
    fn combine_weights_from_plan<B, R>(
        &self,
        rs: &Point<EF>,
        alpha: EF,
        tables: &[ClaimWeightTables<R>],
        column_weights: &[ColumnWeights<R>],
    ) -> Poly<R>
    where
        B: Field,
        R: ExtensionField<B> + From<EF>,
    {
        // Output arity: stacked arity minus the folded challenges.
        let mut out = Poly::<R>::zero(self.claims.num_variables - rs.num_variables());

        // A column's weight entry resolves to its claim's table and alpha power.
        let resolve = |&(claim_idx, is_next, scale): &(usize, bool, R)| {
            let claim_tables = &tables[claim_idx];
            let table = if is_next {
                &claim_tables.next
            } else {
                &claim_tables.current
            };
            (table.as_deref().unwrap(), scale)
        };

        // Concrete claims: each column's slot is independent, so slots fill in parallel.
        // Each slot is also split into chunks, so a tall column spreads across threads.
        // Two contributions share one pass over the slot; any other count takes one pass each.
        column_slots(
            &self.claims.placements,
            &self.claims.tables,
            rs.num_variables(),
            out.as_mut_slice(),
        )
        .into_par_iter()
        .for_each(|(slot, table_idx, poly_idx)| {
            match column_weights[table_idx][poly_idx].as_slice() {
                [term0, term1] => {
                    let ((table0, scale0), (table1, scale1)) = (resolve(term0), resolve(term1));
                    slot.par_chunks_mut(SLOT_CHUNK)
                        .zip(table0.par_chunks(SLOT_CHUNK))
                        .zip(table1.par_chunks(SLOT_CHUNK))
                        .for_each(|((slot, table0), table1)| {
                            slot.iter_mut().zip(table0.iter().zip(table1)).for_each(
                                |(out, (&weight0, &weight1))| {
                                    *out += scale0 * weight0 + scale1 * weight1;
                                },
                            );
                        });
                }
                terms => {
                    for term in terms {
                        let (table, scale) = resolve(term);
                        slot.par_chunks_mut(SLOT_CHUNK)
                            .zip(table.par_chunks(SLOT_CHUNK))
                            .for_each(|(slot, table)| {
                                slot.iter_mut()
                                    .zip(table)
                                    .for_each(|(out, &weight)| *out += scale * weight);
                            });
                    }
                }
            }
        });

        // Virtual claims: span the full output; alpha continues after concrete ones.
        let mut alpha_i = alpha.exp_u64(self.num_claims() as u64);
        for claim in &self.claims.virtual_claims {
            // Split the claim point into (rest-of-space, svo-sub-point).
            let (rest, svo) = claim
                .point
                .as_slice()
                .split_at(claim.point.num_variables() - rs.num_variables());
            // Scalar weight: alpha^i times the equality between svo part and rs.
            let scale = alpha_i * Point::eval_eq(svo, rs.as_slice());
            // The equality table is a polynomial in the point, so it commutes with `R::from`.
            let rest = Point::new(rest.iter().copied().map(R::from).collect());
            // Contribute the scaled equality table across the whole output.
            SplitEq::<B, R>::new_packed(&rest, R::from(scale))
                .accumulate_into(out.as_mut_slice(), None);
            // Advance alpha for the next virtual claim.
            alpha_i *= alpha;
        }

        out
    }
}

/// Sums `weights[i] * values[i]` over every row.
///
/// # Performance
///
/// - A zero row is skipped and a one row adds its weight, so a column of bits costs no product.
/// - From the first row holding any other value, the rest of its chunk pays one product per row.
/// - A column longer than one chunk sums its chunks in parallel.
///
/// # Panics
///
/// - The two slices differ in length.
pub(super) fn weighted_sum<F: Field, EF: Field + Algebra<F>>(weights: &[EF], values: &[F]) -> EF {
    assert_eq!(weights.len(), values.len());
    if values.len() <= WEIGHTED_SUM_CHUNK {
        return weighted_sum_chunk(weights, values);
    }
    weights
        .par_chunks(WEIGHTED_SUM_CHUNK)
        .zip(values.par_chunks(WEIGHTED_SUM_CHUNK))
        .map(|(weights, values)| weighted_sum_chunk(weights, values))
        .sum()
}

/// Sums `weights[i] * values[i]` over one chunk, adding weights directly while every row is a bit.
fn weighted_sum_chunk<F: Field, EF: Field + Algebra<F>>(weights: &[EF], values: &[F]) -> EF {
    let mut sum = EF::ZERO;
    for (row, (&weight, &value)) in weights.iter().zip(values).enumerate() {
        if value == F::ONE {
            sum += weight;
        } else if value != F::ZERO {
            // Rows before this one are all bits; every row from here pays one product.
            return sum
                + dot_product::<EF, _, _>(
                    weights[row..].iter().copied(),
                    values[row..].iter().copied(),
                );
        }
    }
    sum
}

/// Builds the repeat-last successor weights from the equality weights of the same point.
///
/// Row `x` of the successor view reads the column at `x + 1`, and the last row reads itself.
/// Each column row therefore collects its predecessor's weight, and the last row keeps its own:
///
/// ```text
///     next = [0, eq[0], eq[1], ..., eq[last - 2], eq[last - 1] + eq[last]]
/// ```
fn successor_weights<EF: Field>(eq: &[EF]) -> Vec<EF> {
    let last = eq.len() - 1;
    let mut next = EF::zero_vec(eq.len());
    next[1..].copy_from_slice(&eq[..last]);
    next[last] += eq[last];
    next
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use itertools::Itertools;
    use p3_baby_bear::BabyBear;
    use p3_binary_field::{BinaryChallenger, BinaryField128, Ghash128};
    use p3_challenger::HashChallenger;
    use p3_field::extension::BinomialExtensionField;
    use p3_field::{ExtensionField, Field, PrimeCharacteristicRing, dot_product};
    use p3_keccak::Keccak256Hash;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_multilinear_util::point::Point;
    use p3_multilinear_util::poly::Poly;
    use rand::distr::{Distribution, StandardUniform};
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;

    /// Degree-4 binomial extension of BabyBear.
    type BabyBearExt4 = BinomialExtensionField<BabyBear, 4>;

    /// Number of column value patterns produced by `column`.
    const NUM_COLUMN_KINDS: usize = 7;

    /// Table arities opened together in one witness: empty, tiny, small, and multi-chunk tables.
    const ARITIES: [usize; 5] = [0, 1, 2, 5, 13];

    /// Draws a field element that is neither zero nor one.
    fn non_bit<F: Field>(rng: &mut SmallRng) -> F
    where
        StandardUniform: Distribution<F>,
    {
        loop {
            let value: F = rng.random();
            if value != F::ZERO && value != F::ONE {
                return value;
            }
        }
    }

    /// Builds one column of `len` rows whose value pattern is selected by `kind`.
    ///
    /// - 0: uniformly random values.
    /// - 1: random bits.
    /// - 2: all zeros.
    /// - 3: all ones.
    /// - 4: random bits with a non-bit at the first row.
    /// - 5: random bits with a non-bit just past the midpoint.
    /// - 6: random bits with a non-bit at the last row.
    fn column<F: Field>(rng: &mut SmallRng, kind: usize, len: usize) -> Vec<F>
    where
        StandardUniform: Distribution<F>,
    {
        let kind = kind % NUM_COLUMN_KINDS;
        if kind == 0 {
            return (0..len).map(|_| rng.random()).collect();
        }
        let mut values: Vec<F> = (0..len)
            .map(|_| F::from_bool(rng.random_bool(0.5)))
            .collect();
        match kind {
            2 => values.fill(F::ZERO),
            3 => values.fill(F::ONE),
            4 => values[0] = non_bit(rng),
            5 => values[(len / 2 + 1).min(len - 1)] = non_bit(rng),
            6 => values[len - 1] = non_bit(rng),
            _ => {}
        }
        values
    }

    /// Builds a table whose columns cycle through every value pattern.
    fn table<F: Field>(rng: &mut SmallRng, num_variables: usize, num_polys: usize) -> Table<F>
    where
        StandardUniform: Distribution<F>,
    {
        let len = 1 << num_variables;
        let values = (0..num_polys)
            .flat_map(|kind| column::<F>(rng, kind, len))
            .collect();
        Table::new(RowMajorMatrix::new(values, len))
    }

    /// Builds one table per arity, each holding every column pattern.
    fn tables<F: Field>(seed: u64) -> Vec<Table<F>>
    where
        StandardUniform: Distribution<F>,
    {
        let mut rng = SmallRng::seed_from_u64(seed);
        ARITIES
            .iter()
            .map(|&num_variables| table(&mut rng, num_variables, NUM_COLUMN_KINDS))
            .collect()
    }

    /// Records openings at zero folding and checks them against the zero-round SVO routines.
    ///
    /// # Schedule
    ///
    /// Every table is opened six times, each at a fresh random point:
    ///
    /// - every column directly and through the successor view,
    /// - every column directly, in reverse order,
    /// - every column through the successor view, in reverse order,
    /// - the odd columns directly and every column through the successor view,
    /// - the first column directly,
    /// - the last column through the successor view.
    ///
    /// The last two batches open a single column, so they skip the shared weight tables.
    ///
    /// # Checks
    ///
    /// - Each returned and stored evaluation equals the `SvoPoint` evaluation of its view.
    /// - Each stored opening names its column and carries no per-round residuals.
    /// - The batched claimed sum equals the stacked columns dotted with the combined weights.
    fn assert_unfolded_openings_match_svo<F, EF>(tables: Vec<Table<F>>, seed: u64)
    where
        F: Field,
        EF: ExtensionField<F>,
        StandardUniform: Distribution<EF>,
    {
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut prover =
            SuffixProver::<F, EF>::from_witness(SuffixProver::<F, EF>::new_witness(tables, 0));

        for table_idx in 0..prover.claims.tables.len() {
            let num_polys = prover.claims.tables[table_idx].num_polys();
            let num_variables = prover.claims.tables[table_idx].num_variables();
            let columns = (0..num_polys).collect::<Vec<_>>();
            let reversed = columns.iter().rev().copied().collect::<Vec<_>>();
            let odd = columns
                .iter()
                .copied()
                .filter(|poly_idx| poly_idx % 2 == 1)
                .collect::<Vec<_>>();
            let requests = [
                OpeningBatch::new(columns.clone(), columns.clone()),
                OpeningBatch::new(reversed.clone(), Vec::new()),
                OpeningBatch::new(Vec::new(), reversed),
                OpeningBatch::new(odd, columns),
                OpeningBatch::new(vec![0], Vec::new()),
                OpeningBatch::new(Vec::new(), vec![num_polys - 1]),
            ];

            for request in &requests {
                let point = Point::<EF>::rand(&mut rng, num_variables);
                let evals = prover.record_opening(table_idx, request, &point);

                // Reference: the zero-round SVO evaluation of each requested view.
                let table = &prover.claims.tables[table_idx];
                let svo_point = SvoPoint::<F, EF>::new_unpacked(0, &point, VariableOrder::Suffix);
                let expected_current = request
                    .current()
                    .iter()
                    .map(|&poly_idx| {
                        let (eval, partial_evals) = svo_point.eval(table.poly(poly_idx));
                        assert!(partial_evals.rounds().is_empty());
                        eval
                    })
                    .collect::<Vec<_>>();
                let expected_next = request
                    .next()
                    .iter()
                    .map(|&poly_idx| {
                        let (eval, partial_evals) =
                            svo_point.eval_next_suffix(table.poly(poly_idx), None);
                        assert!(partial_evals.rounds().is_empty());
                        eval
                    })
                    .collect::<Vec<_>>();
                assert_eq!(evals.current(), expected_current.as_slice());
                assert_eq!(evals.next(), expected_next.as_slice());

                // The stored claim holds the same point and the same openings, without residuals.
                let claim = prover.claims.claim_map[table_idx].last().unwrap();
                assert_eq!(claim.point().z_svo(), svo_point.z_svo());
                assert_eq!(
                    claim.point().z_split().materialize(),
                    svo_point.z_split().materialize()
                );
                for ((opening, &poly_idx), &eval) in claim
                    .current_openings()
                    .iter()
                    .zip_eq(request.current())
                    .zip_eq(&expected_current)
                {
                    assert_eq!(opening.poly_idx(), Some(poly_idx));
                    assert_eq!(opening.eval(), eval);
                    assert!(opening.data().rounds().is_empty());
                }
                for ((opening, &poly_idx), &eval) in claim
                    .next_openings()
                    .iter()
                    .zip_eq(request.next())
                    .zip_eq(&expected_next)
                {
                    assert_eq!(opening.poly_idx(), Some(poly_idx));
                    assert_eq!(opening.eval(), eval);
                    assert!(opening.data().rounds().is_empty());
                }
            }
        }

        // The residual factors built from the stored claims reproduce the batched claimed sum.
        let alpha: EF = rng.random();
        let rs = Point::default();
        let stacked = prover.compress_stacked(&rs);
        let weights = prover.combine_weights(&rs, alpha);
        assert_eq!(
            prover.claims.sum(alpha),
            dot_product::<EF, _, _>(stacked.iter().copied(), weights.iter().copied())
        );
    }

    /// The challenge field as its own representation, taking tables through the default map.
    impl FromTable<Self> for BabyBearExt4 {}

    /// Records a claim set that exercises every column arity of the weight combiner, then checks
    /// that combining in `R` lands on the image of combining in `EF`.
    ///
    /// # Claims
    ///
    /// Two claims on the first table, both opening columns directly and through the successor
    /// view, plus a third naming one column; the second table takes a claim with a single
    /// opening, and the stacked polynomial takes one virtual claim.
    ///
    /// Columns therefore carry one, two and three contributions, and the second table's unopened
    /// columns carry none.
    ///
    /// At `folding == 0`, the depth the binary PCS runs at, the challenges are empty and the
    /// virtual claim's SVO half is too.
    fn assert_combine_weights_in_matches_image<F, EF, R>(folding: usize, seed: u64)
    where
        F: Field,
        EF: ExtensionField<F>,
        R: Field + FromTable<EF>,
        StandardUniform: Distribution<F> + Distribution<EF>,
    {
        let mut rng = SmallRng::seed_from_u64(seed);
        let tables = vec![table::<F>(&mut rng, 4, 4), table::<F>(&mut rng, 3, 3)];
        let mut prover = SuffixProver::<F, EF>::from_witness(SuffixProver::<F, EF>::new_witness(
            tables, folding,
        ));

        let requests = [
            (0, OpeningBatch::new(vec![0, 1], vec![1, 2])),
            (0, OpeningBatch::new(vec![2], vec![0])),
            (0, OpeningBatch::new(vec![0], Vec::new())),
            (1, OpeningBatch::new(vec![0], Vec::new())),
        ];
        for (table_idx, request) in &requests {
            let num_variables = prover.claims.tables[*table_idx].num_variables();
            let point = Point::<EF>::rand(&mut rng, num_variables);
            prover.record_opening(*table_idx, request, &point);
        }
        let virtual_point = Point::<EF>::rand(&mut rng, prover.claims.num_variables);
        prover.record_virtual(&virtual_point);

        let rs = Point::<EF>::rand(&mut rng, folding);
        let alpha: EF = rng.random();
        let expected = prover.combine_weights(&rs, alpha);
        let (tables, column_weights) = prover.weight_plan(&rs, alpha);
        let combined = prover.combine_weights_in::<R>(&rs, alpha, tables, column_weights);

        assert_eq!(combined.num_variables(), expected.num_variables());
        // Guard against a vacuous comparison of two zero tables.
        assert!(expected.iter().any(|&weight| weight != EF::ZERO));
        for (&combined, &expected) in combined.iter().zip_eq(expected.iter()) {
            assert_eq!(combined, R::from(expected));
        }
    }

    #[test]
    fn combine_weights_in_the_field_itself_matches_the_challenge_field() {
        for folding in [0, 2] {
            assert_combine_weights_in_matches_image::<BabyBear, BabyBearExt4, BabyBearExt4>(
                folding, 5,
            );
        }
    }

    #[test]
    fn combine_weights_in_the_polynomial_basis_matches_the_tower() {
        for folding in [0, 2] {
            assert_combine_weights_in_matches_image::<BinaryField128, BinaryField128, Ghash128>(
                folding, 7,
            );
        }
    }

    #[test]
    fn unfolded_openings_match_svo_over_binary_field() {
        assert_unfolded_openings_match_svo::<BinaryField128, BinaryField128>(tables(1), 2);
    }

    #[test]
    fn unfolded_openings_match_svo_over_extension_of_prime_field() {
        assert_unfolded_openings_match_svo::<BabyBear, BabyBearExt4>(tables(3), 4);
    }

    #[test]
    fn weighted_sum_matches_dot_product() {
        fn check<F: Field, EF: ExtensionField<F>>(rng: &mut SmallRng)
        where
            StandardUniform: Distribution<F> + Distribution<EF>,
        {
            // Lengths straddle the chunk size, including partial trailing chunks.
            for len in [
                1,
                2,
                WEIGHTED_SUM_CHUNK - 1,
                WEIGHTED_SUM_CHUNK,
                WEIGHTED_SUM_CHUNK + 1,
                3 * WEIGHTED_SUM_CHUNK + 5,
            ] {
                let weights = (0..len).map(|_| rng.random()).collect::<Vec<EF>>();
                for kind in 0..NUM_COLUMN_KINDS {
                    let values = column::<F>(rng, kind, len);
                    let expected =
                        dot_product::<EF, _, _>(weights.iter().copied(), values.iter().copied());
                    assert_eq!(
                        weighted_sum(&weights, &values),
                        expected,
                        "len={len}, kind={kind}"
                    );
                }
            }
        }

        let mut rng = SmallRng::seed_from_u64(5);
        check::<BinaryField128, BinaryField128>(&mut rng);
        check::<BabyBear, BabyBearExt4>(&mut rng);
    }

    /// One opening batch: `(table index, request)`, taken at a fresh random point.
    type Batch = (usize, OpeningRequest);

    /// The Fiat-Shamir transcript the binary-field cases are driven with.
    type BinaryTranscript = BinaryChallenger<BinaryField128, HashChallenger<u8, Keccak256Hash, 32>>;

    /// Records every batch of `schedule` at zero folding, each at a fresh random point.
    fn recorded<F, EF>(tables: Vec<Table<F>>, schedule: &[Batch], seed: u64) -> SuffixProver<F, EF>
    where
        F: Field,
        EF: ExtensionField<F>,
        StandardUniform: Distribution<EF>,
    {
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut prover =
            SuffixProver::<F, EF>::from_witness(SuffixProver::<F, EF>::new_witness(tables, 0));
        for (table_idx, request) in schedule {
            let num_variables = prover.claims.tables[*table_idx].num_variables();
            let point = Point::<EF>::rand(&mut rng, num_variables);
            prover.record_opening(*table_idx, request, &point);
        }
        prover
    }

    /// Witnesses and schedules whose weights all split into one shared row table.
    ///
    /// - Every column directly and through the successor view, in one order.
    /// - Every column directly only.
    /// - Two claims at distinct points, each opening every column in the same order.
    /// - A subset of the columns; the rest are never opened.
    /// - A second table of the same height, never opened, so its slots carry zero scales.
    /// - A table past one aggregate block and one weighted-sum chunk.
    /// - One row per column, so no row round precedes the column rounds.
    ///
    /// Columns cycle through every value pattern, so bit-valued and arbitrary rows both occur.
    fn row_first_cases<F: Field>(seed: u64) -> Vec<(Vec<Table<F>>, Vec<Batch>)>
    where
        StandardUniform: Distribution<F>,
    {
        let mut rng = SmallRng::seed_from_u64(seed);
        let every = |width: usize| (0..width).collect::<Vec<_>>();
        vec![
            (
                vec![table(&mut rng, 4, 5)],
                vec![(0, OpeningBatch::new(every(5), every(5)))],
            ),
            (
                vec![table(&mut rng, 4, 5)],
                vec![(0, OpeningBatch::new(every(5), Vec::new()))],
            ),
            (
                vec![table(&mut rng, 3, 3)],
                vec![
                    (0, OpeningBatch::new(every(3), every(3))),
                    (0, OpeningBatch::new(every(3), every(3))),
                ],
            ),
            (
                vec![table(&mut rng, 4, 5)],
                vec![(0, OpeningBatch::new(vec![0, 2, 3], vec![0, 2, 3]))],
            ),
            (
                vec![table(&mut rng, 4, 3), table(&mut rng, 4, 2)],
                vec![(0, OpeningBatch::new(every(3), every(3)))],
            ),
            (
                vec![table(&mut rng, 13, 3)],
                vec![(0, OpeningBatch::new(every(3), every(3)))],
            ),
            (
                vec![table(&mut rng, 0, 5)],
                vec![(0, OpeningBatch::new(every(5), Vec::new()))],
            ),
        ]
    }

    /// Plays every residual round of both routes from one recorded prover and equal transcripts.
    ///
    /// The row-first route must be the one [`SuffixProver::into_sumcheck_in`] takes. The dense
    /// reference runs through the same preprocessing, then both play `rounds_per_call` rounds
    /// per call, settling after each call when `settle` is set.
    ///
    /// # Checks
    ///
    /// - Equal arities, challenges and running claims after every call.
    /// - Equal round messages over the whole run.
    /// - Equal sponge states once every round is played.
    fn assert_row_first_matches_dense<F, EF, R, Ch>(
        prover: SuffixProver<F, EF>,
        challenger: Ch,
        rounds_per_call: usize,
        settle: bool,
    ) where
        F: TranscriptField,
        EF: ExtensionField<F> + From<R>,
        R: Field + FromTable<EF> + Algebra<F>,
        Ch: FieldChallenger<F> + GrindingChallenger<Witness = F> + Clone,
    {
        let mut dense_challenger = challenger.clone();
        let mut dense_data = SumcheckData::default();
        let (alpha, sum, rs) = prover.preprocess(&mut dense_data, 0, &mut dense_challenger);
        let (claim_tables, column_weights) = prover.weight_plan(&rs, alpha);
        assert!(
            prover
                .shared_row_weights(&claim_tables, &column_weights)
                .is_some()
        );
        let dense = prover.dense_residual::<R>(&rs, alpha, sum, claim_tables, column_weights);
        let mut dense = SuffixResidualProver::dense(dense);

        let mut row_first_challenger = challenger;
        let mut row_first_data = SumcheckData::default();
        let (mut row_first, row_first_rs) =
            prover.into_sumcheck_in::<R, _>(&mut row_first_data, 0, &mut row_first_challenger);
        assert!(row_first.is_row_first());
        assert_eq!(row_first_rs, rs);

        while dense.num_variables() > 0 {
            assert_eq!(row_first.num_variables(), dense.num_variables());
            let rounds = rounds_per_call.min(dense.num_variables());
            let expected = dense.compute_sumcheck_polynomials(
                &mut dense_data,
                &mut dense_challenger,
                rounds,
                0,
            );
            let challenges = row_first.compute_sumcheck_polynomials(
                &mut row_first_data,
                &mut row_first_challenger,
                rounds,
                0,
            );
            assert_eq!(challenges, expected);
            assert_eq!(row_first.claimed_sum(), dense.claimed_sum());
            if settle {
                dense.settle();
                row_first.settle();
            }
        }
        assert_eq!(row_first.num_variables(), 0);
        assert_eq!(
            row_first_data.polynomial_evaluations,
            dense_data.polynomial_evaluations
        );
        assert_eq!(
            row_first_challenger.sample_algebra_element::<EF>(),
            dense_challenger.sample_algebra_element::<EF>()
        );

        // A debug build checks the last held binding against the pair it lands on.
        row_first.settle();
    }

    #[test]
    fn row_first_route_plays_the_dense_transcript_over_binary_field() {
        for (seed, (tables, schedule)) in (0..).zip(row_first_cases::<BinaryField128>(11)) {
            for (rounds_per_call, settle) in [(1, false), (1, true), (3, false)] {
                let prover =
                    recorded::<BinaryField128, BinaryField128>(tables.clone(), &schedule, seed);
                let challenger = BinaryTranscript::from_hasher(Vec::new(), Keccak256Hash);
                assert_row_first_matches_dense::<_, _, Ghash128, _>(
                    prover,
                    challenger,
                    rounds_per_call,
                    settle,
                );
            }
        }
    }

    #[test]
    fn row_first_route_plays_the_dense_transcript_over_extension_of_prime_field() {
        for (seed, (tables, schedule)) in (0..).zip(row_first_cases::<BabyBear>(12)) {
            for (rounds_per_call, settle) in [(1, false), (1, true), (3, false)] {
                let prover = recorded::<BabyBear, BabyBearExt4>(tables.clone(), &schedule, seed);
                assert_row_first_matches_dense::<_, _, BabyBearExt4, _>(
                    prover,
                    crate::tests::challenger(),
                    rounds_per_call,
                    settle,
                );
            }
        }
    }

    #[test]
    fn a_single_column_takes_the_dense_route() {
        type F = BinaryField128;

        // The weights split, yet the one column already spans the whole stacked space.
        let schedule = vec![(0, OpeningBatch::new(vec![0], vec![0]))];
        let tables = vec![table::<F>(&mut SmallRng::seed_from_u64(14), 5, 1)];
        let prover = recorded::<F, F>(tables, &schedule, 15);

        let mut challenger = BinaryTranscript::from_hasher(Vec::new(), Keccak256Hash);
        let (residual, _) = prover.into_sumcheck_in::<Ghash128, _>(
            &mut SumcheckData::default(),
            0,
            &mut challenger,
        );
        assert!(!residual.is_row_first());
    }

    #[test]
    fn shared_row_weights_refuses_weights_it_cannot_split() {
        type F = BinaryField128;

        // Whether the weight plan of a recorded prover splits, at fresh challenges.
        fn splits(prover: &SuffixProver<F, F>, rng: &mut SmallRng) -> bool {
            let rs = Point::<F>::rand(rng, prover.claims.folding);
            let alpha: F = rng.random();
            let (claim_tables, column_weights) = prover.weight_plan(&rs, alpha);
            prover
                .shared_row_weights(&claim_tables, &column_weights)
                .is_some()
        }

        let mut rng = SmallRng::seed_from_u64(13);
        let every = |width: usize| (0..width).collect::<Vec<_>>();

        // Positive control: the shape every refusal below departs from.
        let keccak_shape = vec![(0, OpeningBatch::new(every(5), every(5)))];
        let tables = vec![table::<F>(&mut rng, 4, 5)];
        assert!(splits(
            &recorded::<F, F>(tables.clone(), &keccak_shape, 1),
            &mut rng
        ));

        // Only some columns open through the successor view: two row tables, no common one.
        let partial = vec![(0, OpeningBatch::new(every(5), vec![1, 3]))];
        assert!(!splits(
            &recorded::<F, F>(tables.clone(), &partial, 2),
            &mut rng
        ));

        // A virtual claim weighs the stacked space by a table no column shares.
        let mut prover = recorded::<F, F>(tables.clone(), &keccak_shape, 3);
        let point = Point::<F>::rand(&mut rng, prover.claims.num_variables);
        let _ = prover.record_virtual(&point);
        assert!(!splits(&prover, &mut rng));

        // Tables of different heights have no row variables in common.
        let uneven = vec![table::<F>(&mut rng, 4, 3), table::<F>(&mut rng, 3, 2)];
        let schedule = vec![
            (0, OpeningBatch::new(every(3), Vec::new())),
            (1, OpeningBatch::new(every(2), Vec::new())),
        ];
        assert!(!splits(&recorded::<F, F>(uneven, &schedule, 4), &mut rng));

        // Equal heights, but each table's columns reference a claim of their own.
        let even = vec![table::<F>(&mut rng, 3, 3), table::<F>(&mut rng, 3, 2)];
        assert!(!splits(&recorded::<F, F>(even, &schedule, 5), &mut rng));

        // Preprocessing rounds leave nothing for the row-first route to start from.
        let mut prover =
            SuffixProver::<F, F>::from_witness(SuffixProver::<F, F>::new_witness(tables, 2));
        let point = Point::<F>::rand(&mut rng, 4);
        prover.record_opening(0, &keccak_shape[0].1, &point);
        assert!(!splits(&prover, &mut rng));
    }

    #[test]
    fn successor_weights_match_dense_successor_table() {
        let mut rng = SmallRng::seed_from_u64(6);
        for num_variables in 0..=8 {
            let point = Point::<BabyBearExt4>::rand(&mut rng, num_variables);
            let eq = Poly::new_from_point(point.as_slice(), BabyBearExt4::ONE);
            assert_eq!(
                successor_weights(eq.as_slice()),
                Poly::new_next_from_point(point.as_slice()).as_slice()
            );
        }
    }
}
