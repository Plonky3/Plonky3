//! Suffix-mode stacked-sumcheck prover.

use alloc::vec;
use alloc::vec::Vec;

use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, Field, dot_product};
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use p3_multilinear_util::split_eq::SplitEq;

use crate::sumcheck::lagrange::lagrange_weights_01inf_multi;
use crate::sumcheck::layout::opening::{MultiClaim, Opening, SuffixMultiClaim, SuffixVirtualClaim};
use crate::sumcheck::layout::witness::{Table, TablePlacement};
use crate::sumcheck::product_polynomial::ProductPolynomial;
use crate::sumcheck::strategy::{SumcheckProver, VariableOrder};
use crate::sumcheck::svo::{SvoPoint, calculate_accumulators_batch};
use crate::sumcheck::{Claim, SumcheckData, extrapolate_01inf};

/// Stacked-sumcheck prover with suffix-first variable binding.
///
/// # Flow
///
/// - SVO accumulators are precomputed at claim-recording time.
/// - Each preprocessing round reads its slice of those accumulators.
/// - The residual product polynomial is built once, after all rounds.
#[derive(Debug, Clone)]
pub struct SuffixProver<F: Field, EF: ExtensionField<F>> {
    /// Source tables behind the stacked polynomial.
    pub(crate) tables: Vec<Table<F>>,
    /// Per-table placement metadata inside the stacked polynomial.
    pub(crate) placements: Vec<TablePlacement>,
    /// Number of variables of the stacked polynomial.
    pub(crate) num_variables: usize,
    /// Number of preprocessing rounds consumed before residual sumcheck.
    pub(crate) folding: usize,
    /// Stacked committed polynomial.
    pub(crate) poly: Poly<F>,
    /// Concrete claims recorded per source table (carries per-round SVO partials).
    ///
    /// # Invariants
    ///
    /// - Every opening stored here is tied to a concrete source column.
    /// - Virtual openings never enter this map.
    /// - Claims are appended in insertion order.
    pub(crate) claim_map: Vec<Vec<SuffixMultiClaim<F, EF>>>,
    /// Virtual claims carrying precomputed SVO accumulators.
    pub(crate) virtual_claims: Vec<SuffixVirtualClaim<EF>>,
}

impl<F: Field, EF: ExtensionField<F>> SuffixProver<F, EF> {
    /// Returns the number of variables of the stacked polynomial.
    pub const fn num_variables(&self) -> usize {
        self.num_variables
    }

    /// Returns the number of variables of table `id`.
    pub fn num_variables_table(&self, id: usize) -> usize {
        self.tables[id].num_variables()
    }

    /// Returns the stacked committed polynomial.
    pub const fn poly(&self) -> &Poly<F> {
        &self.poly
    }

    /// Records opening claims for the selected columns of `table_idx`.
    ///
    /// # Arguments
    ///
    /// - `table_idx`  — source table index.
    /// - `polys`      — columns to open; must be non-empty.
    /// - `challenger` — Fiat–Shamir transcript.
    ///
    /// # Fiat–Shamir
    ///
    /// - Samples the opening point internally from the challenger.
    /// - Absorbs the evaluations into the transcript before returning.
    /// - The verifier's `add_claim` performs the symmetric absorption.
    ///
    /// # Panics
    ///
    /// - Columns list must be non-empty.
    #[tracing::instrument(skip_all)]
    pub fn eval<Ch>(&mut self, table_idx: usize, polys: &[usize], challenger: &mut Ch) -> Vec<EF>
    where
        Ch: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        // Precondition: opening nothing would silently push an empty MultiClaim.
        assert!(
            !polys.is_empty(),
            "opening schedule must name at least one column"
        );

        // Sample the local-frame opening point from the transcript.
        let table = &self.tables[table_idx];
        let point = Point::expand_from_univariate(
            challenger.sample_algebra_element(),
            table.num_variables(),
        );

        // Factorise the point with the suffix split; every selected column reuses it.
        let point = SvoPoint::new_unpacked(self.folding, &point, VariableOrder::Suffix);

        // Evaluate each requested column and split into (opening, eval) in a single pass.
        let (openings, evals): (Vec<_>, Vec<EF>) = polys
            .iter()
            .map(|&poly_idx| {
                // Per-column eval plus the per-round partial-eval polynomials.
                let (eval, partial_evals) = point.eval(table.poly(poly_idx));
                // Wrap the outputs as a concrete opening on this column.
                let opening = Opening {
                    poly_idx: Some(poly_idx),
                    eval,
                    data: partial_evals,
                };
                (opening, eval)
            })
            .unzip();

        // Bind the evaluations into the transcript; the verifier absorbs the same bytes.
        challenger.observe_algebra_slice(&evals);

        // Store the batch with its shared SVO point.
        self.claim_map[table_idx].push(SuffixMultiClaim::new(point, openings));

        evals
    }

    /// Samples a virtual evaluation on the full stacked polynomial.
    ///
    /// # Why heavier than prefix binding
    ///
    /// The stacked evaluation factors per column via the selector:
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
    /// - Those partials feed the SVO accumulator batcher.
    #[tracing::instrument(skip_all)]
    pub fn add_virtual_eval<Ch>(&mut self, challenger: &mut Ch) -> EF
    where
        Ch: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        // Sample a challenge point covering every stacked variable.
        let point =
            Point::expand_from_univariate(challenger.sample_algebra_element(), self.num_variables);

        // Per-column accumulation state:
        //
        // - eval    : running stacked evaluation.
        // - openings: one virtual opening per column, carrying SVO partials.
        // - weights : per-column selector-equality scalars.
        let mut eval = EF::ZERO;
        let mut openings = Vec::new();
        let mut weights = Vec::new();

        for placement in &self.placements {
            for (poly_idx, selector) in placement.selectors().iter().enumerate() {
                // Source column behind this slot.
                let poly = self.tables[placement.idx()].poly(poly_idx);

                // Split the challenge into (selector_bits, local_bits).
                let (selector_part, local_part) = point.split_at(selector.num_variables());

                // Scalar weight: eq(selector, selector_part) for this column.
                let weight =
                    Point::eval_eq::<EF>(selector.point().as_slice(), selector_part.as_slice());

                // Factorise the local part with the suffix split, then evaluate.
                let local_svo =
                    SvoPoint::new_unpacked(self.folding, &local_part, VariableOrder::Suffix);
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
            &SuffixMultiClaim::new(
                SvoPoint::new_unpacked(self.folding, &point, VariableOrder::Suffix),
                openings,
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
            assert_eq!(eval, poly.eval_base(&point));

            // Build the reference opening by evaluating the materialised poly directly.
            let ref_svo =
                SvoPoint::<EF, EF>::new_unpacked(self.folding, &point, VariableOrder::Suffix);
            let (ref_eval, ref_partials) = ref_svo.eval(poly);
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
                    &SuffixMultiClaim::new(
                        SvoPoint::new_unpacked(self.folding, &point, VariableOrder::Suffix),
                        vec![opening],
                    ),
                    &[EF::ONE],
                ),
            );
        }

        // Commit the evaluation to the transcript and record the claim.
        challenger.observe_algebra_element(eval);
        self.virtual_claims.push(Claim {
            point,
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
    pub fn into_sumcheck<Ch>(
        self,
        sumcheck_data: &mut SumcheckData<F, EF>,
        pow_bits: usize,
        challenger: &mut Ch,
    ) -> (SumcheckProver<F, EF>, Point<EF>)
    where
        Ch: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        // Sanity: preprocessing cannot consume more rounds than the stacked arity.
        assert!(self.folding <= self.num_variables);
        let alpha: EF = challenger.sample_algebra_element();
        let n_claims = self.num_claims();

        // Stage A: batch per-claim accumulators using insertion-order alpha powers.
        //
        // - Iteration order is placement order, matching `sum` and `combine_eqs`.
        // - Each claim consumes exactly `claim.len()` consecutive powers from
        //   the shared iterator, so the per-claim alpha vector is aligned with
        //   the claim's opening list by construction.
        let mut alphas = alpha.powers();
        let accumulators: Vec<_> = self
            .placements
            .iter()
            .flat_map(|placement| self.claim_map[placement.idx()].iter())
            .map(|claim| {
                let per_claim: Vec<EF> = alphas.by_ref().take(claim.len()).collect();
                calculate_accumulators_batch(claim, &per_claim)
            })
            .collect();

        // Stage C: drive the preprocessing rounds from the accumulators.
        let mut sum = self.sum(alpha);
        let mut rs: Vec<EF> = vec![];

        for round_idx in 0..self.folding {
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
                .virtual_claims
                .iter()
                .zip(alpha.powers().skip(n_claims))
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
            let r = sumcheck_data.observe_and_sample(challenger, c0, c_inf, pow_bits);
            sum = extrapolate_01inf(c0, sum - c0, c_inf, r);
            rs.push(r);
        }

        // Stage D: materialise the residual product polynomial.
        //
        // - Suffix binding folds variables in reverse.
        // - The residual poly therefore lives in the reversed-challenges frame.
        let rs = Point::new(rs);
        // Reverse the challenges before handing them to the compressors.
        let reversed = rs.reversed();
        // Factor 1 of the product: the compressed stacked poly at rs.
        let compressed = self.compress_stacked(&reversed);
        // Factor 2 of the product: the batched equality-weight poly.
        let weights = self.combine_eqs(&reversed, alpha);
        // Pair them; the product polynomial drives the remaining rounds.
        let poly = ProductPolynomial::new_unpacked(VariableOrder::Suffix, compressed, weights);
        // Cross-check: the dot product of the two factors must equal the
        // running sum accumulated across the preprocessing rounds.
        debug_assert_eq!(poly.dot_product(), sum);

        (SumcheckProver::new(poly, sum), rs)
    }

    /// Returns the total number of concrete openings recorded so far.
    fn num_claims(&self) -> usize {
        self.claim_map
            .iter()
            .flat_map(|claims| claims.iter().map(MultiClaim::len))
            .sum()
    }

    /// Computes the batched claimed sum from concrete and virtual openings.
    ///
    /// # Identity
    ///
    /// ```text
    ///     sum = sum_{i}  alpha^i * eval_i
    /// ```
    ///
    /// # Alpha ordering
    ///
    /// Powers of `alpha` are handed out in insertion order:
    ///
    /// - Outer: placements, in the order the witness laid them out.
    /// - Middle: claims recorded against that placement's source table.
    /// - Inner: openings inside each claim, in the order they were recorded.
    ///
    /// # Virtual claims
    ///
    /// - Virtual evaluations continue the same alpha sequence.
    /// - They start at `alpha^n`, with `n` the total concrete opening count.
    ///
    /// # Verifier agreement
    ///
    /// The verifier walks its claim registry with the same three-loop order,
    /// so both sides assign the same `alpha^i` to the same claim point.
    fn sum(&self, alpha: EF) -> EF {
        let mut sum = EF::ZERO;
        let mut alphas = alpha.powers();

        // Concrete openings: three loops, no filter.
        for placement in &self.placements {
            for claim in &self.claim_map[placement.idx()] {
                for opening in claim.openings() {
                    sum += opening.eval() * alphas.next().unwrap();
                }
            }
        }

        // Virtual claims continue the alpha sequence right after the concrete ones.
        sum += dot_product::<EF, _, _>(
            self.virtual_claims.iter().map(Claim::eval),
            alpha.powers().skip(self.num_claims()),
        );

        sum
    }

    /// Compresses every stacked-table slot by fixing the suffix challenges.
    ///
    /// ```text
    ///     out[slot_idx, x_rest] = sum_{y in {0,1}^|rs|}  eq(rs, y) * col(x_rest, y)
    /// ```
    ///
    /// # Layout
    ///
    /// - One output slot per column; writes never overlap.
    /// - Output arity is the stacked arity minus the number of challenges.
    fn compress_stacked(&self, rs: &Point<EF>) -> Poly<EF> {
        assert!(rs.num_variables() <= self.num_variables);
        // Output: residual stacked space of size 2^(num_variables - |rs|).
        let mut out = Poly::<EF>::zero(self.num_variables - rs.num_variables());
        let rs = SplitEq::new_unpacked(rs, EF::ONE);

        for placement in &self.placements {
            for (poly_idx, selector) in placement.selectors().iter().enumerate() {
                let poly = self.tables[placement.idx()].poly(poly_idx);
                assert!(rs.num_variables() <= poly.num_variables());
                // Slot start in the compressed output.
                let off = selector.index() << (poly.num_variables() - rs.num_variables());
                // Write this column's compression into its own slot.
                rs.compress_suffix_into(
                    &mut out.as_mut_slice()
                        [off..off + (1 << (poly.num_variables() - rs.num_variables()))],
                    poly,
                );
            }
        }
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
    fn combine_eqs(&self, rs: &Point<EF>, alpha: EF) -> Poly<EF> {
        // Preconditions: challenge count matches the folding depth.
        assert_eq!(rs.num_variables(), self.folding);
        // Output arity: stacked arity minus the folded challenges.
        let mut out = Poly::<EF>::zero(self.num_variables - rs.num_variables());

        let mut alphas = alpha.powers();

        // Concrete claims: write each into the slot its column's selector addresses.
        for placement in &self.placements {
            let num_variables_table = self.num_variables_table(placement.idx());
            let slot_size = 1usize << num_variables_table;
            for claim in &self.claim_map[placement.idx()] {
                for opening in claim.openings() {
                    // The opening's column tells us which selector picks the slot.
                    let col = opening.poly_idx().unwrap();
                    let off = placement.selectors()[col].index() << num_variables_table;
                    // Fold the scalar slot range down by the SVO depth.
                    let folded_range = (off >> self.folding)..((off + slot_size) >> self.folding);
                    claim.point().accumulate_into(
                        &mut out.as_mut_slice()[folded_range],
                        rs,
                        alphas.next().unwrap(),
                    );
                }
            }
        }

        // Virtual claims: span the full output; alpha continues after concrete ones.
        let mut alpha_i = alpha.exp_u64(self.num_claims() as u64);
        for claim in &self.virtual_claims {
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

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use proptest::prelude::*;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::SuffixProver;
    use crate::sumcheck::layout::prover::test_utils::{
        ASCENDING_POLYS, NON_ASCENDING_POLYS, arb_opening_schedule, arb_witness_and_schedule,
        build_witness, build_witness_from_shape, run_suffix_roundtrip, run_suffix_roundtrip_with,
        table_shapes_from,
    };
    use crate::sumcheck::tests::*;

    #[test]
    fn suffix_num_claims_counts_every_recorded_opening() {
        // Invariant:
        //     Concrete opening count equals the sum over every (table, claim).
        //
        // Fixture state:
        //     fresh prover:                  0 concrete openings
        //     after eval(table 0, [0, 1]):   2 concrete openings
        //     after eval(table 1, [0]):      3 concrete openings
        let witness = build_witness();
        let mut prover: SuffixProver<F, EF> = witness.as_suffix_prover();
        // Fresh prover: no claims recorded yet.
        assert_eq!(prover.num_claims(), 0);

        // Fresh Fiat-Shamir transcript; eval samples its own point and absorbs.
        let mut ch = challenger();

        // Record two openings on table 0 → count advances from 0 to 2.
        prover.eval(0, &[0, 1], &mut ch);
        assert_eq!(prover.num_claims(), 2);

        // Record one more opening on table 1 → count advances from 2 to 3.
        prover.eval(1, &[0], &mut ch);
        assert_eq!(prover.num_claims(), 3);
    }

    #[test]
    fn suffix_sum_matches_weighted_eval_sum() {
        // Invariant:
        //     sum(alpha) = eval_0 * alpha^0 + eval_1 * alpha^1
        //     for a single claim opening two columns in traversal order.
        //
        // Fixture state:
        //     one eval call recording two openings on one claim.
        //     traversal order: column 0 before column 1.
        let witness = build_witness();
        let mut prover: SuffixProver<F, EF> = witness.as_suffix_prover();

        // Fresh Fiat-Shamir transcript.
        let mut ch = challenger();

        // Record two openings; evals[0], evals[1] line up with columns 0, 1.
        let evals = prover.eval(0, &[0, 1], &mut ch);

        // Deterministic alpha for hand-rolled comparison.
        let mut rng = SmallRng::seed_from_u64(7);
        let alpha: EF = rng.random();
        let expected = evals[0] + alpha * evals[1];

        // Check: the prover's sum helper matches the hand-rolled formula.
        assert_eq!(prover.sum(alpha), expected);
    }

    #[test]
    fn suffix_roundtrip_ascending_polys() {
        // Baseline: columns opened in ascending order inside each claim.
        run_suffix_roundtrip(ASCENDING_POLYS);
    }

    #[test]
    fn suffix_roundtrip_non_ascending_polys() {
        // Regression:
        //     Alphas and partial evals used to desync when opening order
        //     within a claim was not sorted ascending.
        run_suffix_roundtrip(NON_ASCENDING_POLYS);
    }

    proptest! {
        #![proptest_config(ProptestConfig { cases: 16, ..ProptestConfig::default() })]

        // Invariant:
        //     Every valid opening schedule roundtrips through the protocol
        //     without the prover and verifier diverging.
        //
        // Coverage: includes non-ascending column orders that previously
        // exposed the alpha / partial-eval alignment bug.
        #[test]
        fn suffix_roundtrip_proptest(schedule in arb_opening_schedule()) {
            // Adapt the owned schedule to the slice-pair shape the runner expects.
            let borrowed: Vec<(usize, &[usize])> = schedule
                .iter()
                .map(|(t, polys)| (*t, polys.as_slice()))
                .collect();
            // Drive both sides; the runner asserts agreement internally.
            run_suffix_roundtrip(&borrowed);
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig { cases: 8, ..ProptestConfig::default() })]

        // Invariant:
        //     Roundtrip agreement holds for ANY valid witness shape, not just
        //     the fixed two-table fixture.
        //
        // Strategy:
        //     Random witness shape (1..=3 tables, arity in [5, 8], 1..=3 cols per
        //     table) paired with a matching opening schedule over that shape.
        #[test]
        fn suffix_roundtrip_shape_proptest(
            (shape, schedule) in arb_witness_and_schedule(),
        ) {
            // Build a matching witness and verifier-side shapes for this case.
            let witness = build_witness_from_shape(&shape);
            let shapes = table_shapes_from(&shape);
            // Adapt the owned schedule to the slice-pair shape the runner expects.
            let borrowed: Vec<(usize, &[usize])> = schedule
                .iter()
                .map(|(t, polys)| (*t, polys.as_slice()))
                .collect();
            // Drive both sides on the generated shape; runner asserts agreement.
            run_suffix_roundtrip_with(witness, &shapes, &borrowed);
        }
    }
}
