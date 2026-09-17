//! Suffix-mode stacked-sumcheck prover.

use alloc::vec;
use alloc::vec::Vec;

use p3_challenger::fs::TranscriptField;
use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, Field, dot_product};
use p3_maybe_rayon::prelude::*;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use p3_multilinear_util::split_eq::SplitEq;

use crate::lagrange::lagrange_weights_01inf_multi;
use crate::layout::opening::{Opening, ProverMultiClaim};
use crate::layout::prover::{Layout, StackedClaims};
use crate::layout::witness::Table;
use crate::layout::{LayoutStrategy, Witness};
use crate::product_polynomial::ProductPolynomial;
use crate::strategy::{Basis, SumcheckProver, VariableOrder};
use crate::svo::{SvoPoint, calculate_accumulators_batch};
use crate::table::{OpeningBatch, OpeningEvals, OpeningRequest};
use crate::transcript::{ProverTranscript, SumcheckShape};
use crate::{Claim, SumcheckData, extrapolate_01inf};

/// One claim's residual weight tables at unit scale, over a single column slot.
struct ClaimWeightTables<EF> {
    /// Equality weights, present when the claim opens a column directly.
    current: Option<Vec<EF>>,
    /// Repeat-last successor weights, present when the claim opens a successor view.
    next: Option<Vec<EF>>,
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
    #[tracing::instrument(skip_all)]
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

        // Stage D: materialise the residual product polynomial.
        //
        // - Suffix binding folds variables in reverse.
        // - The residual poly therefore lives in the reversed-challenges frame.
        let rs = Point::new(rs);
        // Reverse the challenges before handing them to the compressors.
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
        // A column is usually too short to parallelize inside, so the parallelism is across columns.
        self.column_slots(out.as_mut_slice(), num_folded)
            .into_par_iter()
            .for_each(|(slot, table_idx, poly_idx)| {
                let poly = self.claims.tables[table_idx].poly(poly_idx);
                if num_folded == 0 {
                    // Nothing to fold: the slot is the column itself, times the scale.
                    if scale == EF::ONE {
                        slot.iter_mut()
                            .zip(poly.as_slice())
                            .for_each(|(out, &value)| *out = value.into());
                    } else {
                        slot.iter_mut()
                            .zip(poly.as_slice())
                            .for_each(|(out, &value)| *out = scale * value);
                    }
                } else {
                    rs.compress_suffix_into(slot, poly.as_view());
                }
            });
        out
    }

    /// Splits `out` into one disjoint slot per column, after `num_folded` suffix variables.
    ///
    /// Column `(table, poly)` owns the slot starting at `selector.index() << (n - num_folded)`,
    /// of length `2^(n - num_folded)`, where `n` is the table's arity.
    ///
    /// # Returns
    ///
    /// One `(slot, table index, column index)` triple per column, in increasing slot order.
    fn column_slots<'a>(
        &self,
        out: &'a mut [EF],
        num_folded: usize,
    ) -> Vec<(&'a mut [EF], usize, usize)> {
        let mut ranges: Vec<(usize, usize, usize, usize)> = self
            .claims
            .placements
            .iter()
            .flat_map(|placement| {
                let num_variables_table = self.claims.num_variables_table(placement.idx());
                assert!(num_folded <= num_variables_table);
                let log_len = num_variables_table - num_folded;
                placement
                    .selectors()
                    .iter()
                    .enumerate()
                    .map(move |(poly_idx, selector)| {
                        (
                            selector.index() << log_len,
                            1 << log_len,
                            placement.idx(),
                            poly_idx,
                        )
                    })
            })
            .collect();
        ranges.sort_unstable_by_key(|&(offset, ..)| offset);

        let mut slots = Vec::with_capacity(ranges.len());
        let mut rest = out;
        let mut consumed = 0;
        for (offset, len, table_idx, poly_idx) in ranges {
            let (_, tail) = core::mem::take(&mut rest).split_at_mut(offset - consumed);
            let (slot, tail) = tail.split_at_mut(len);
            slots.push((slot, table_idx, poly_idx));
            rest = tail;
            consumed = offset + len;
        }
        slots
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
        // Preconditions: challenge count matches the folding depth.
        assert_eq!(rs.num_variables(), self.claims.folding);
        // Output arity: stacked arity minus the folded challenges.
        let mut out = Poly::<EF>::zero(self.claims.num_variables - rs.num_variables());

        // Every opening of a claim shares that claim's point, so its residual weight table is
        // the same for every column it opens. Build each claim's current and successor tables
        // once, at unit scale; both accumulations are linear in the scale.
        //
        // Walk order matches the batched claim, so alpha powers stay aligned:
        // placements, then claims, then current openings, then successor openings.
        let mut alphas = alpha.powers();
        let mut tables: Vec<ClaimWeightTables<EF>> = Vec::new();
        // Per column: `(table index into tables, is successor, alpha power)`.
        let mut column_weights: Vec<Vec<Vec<(usize, bool, EF)>>> = self
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

        // Concrete claims: each column's slot is independent, so slots fill in parallel.
        // Every contribution to a slot is summed in one pass over it.
        self.column_slots(out.as_mut_slice(), rs.num_variables())
            .into_par_iter()
            .for_each(|(slot, table_idx, poly_idx)| {
                let terms: Vec<(&[EF], EF)> = column_weights[table_idx][poly_idx]
                    .iter()
                    .map(|&(claim_idx, is_next, scale)| {
                        let claim_tables = &tables[claim_idx];
                        let table = if is_next {
                            &claim_tables.next
                        } else {
                            &claim_tables.current
                        };
                        (table.as_deref().unwrap(), scale)
                    })
                    .collect();
                match terms.as_slice() {
                    [] => {}
                    [(table, scale)] => slot
                        .iter_mut()
                        .zip(*table)
                        .for_each(|(out, &weight)| *out += *scale * weight),
                    [(table0, scale0), (table1, scale1)] => slot
                        .iter_mut()
                        .zip(table0.iter().zip(*table1))
                        .for_each(|(out, (&weight0, &weight1))| {
                            *out += *scale0 * weight0 + *scale1 * weight1;
                        }),
                    _ => slot.iter_mut().enumerate().for_each(|(row, out)| {
                        *out += terms
                            .iter()
                            .map(|(table, scale)| *scale * table[row])
                            .sum::<EF>();
                    }),
                }
            });

        // Virtual claims: span the full output; alpha continues after concrete ones.
        let mut alpha_i = alpha.exp_u64(self.num_claims() as u64);
        for claim in &self.claims.virtual_claims {
            // Split the claim point into (rest-of-space, svo-sub-point).
            let (rest, svo) = claim
                .point
                .split_at(claim.point.num_variables() - rs.num_variables());
            // Scalar weight: alpha^i times the equality between svo part and rs.
            let scale = alpha_i * Point::eval_eq(svo.as_slice(), rs.as_slice());
            // Contribute the scaled equality table across the whole output.
            SplitEq::new_packed(&rest, scale).accumulate_into(out.as_mut_slice(), None);
            // Advance alpha for the next virtual claim.
            alpha_i *= alpha;
        }

        out
    }
}
