//! Prefix-mode stacked-sumcheck prover.

use alloc::vec::Vec;

use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{
    ExtensionField, Field, PackedFieldExtension, PackedValue, TwoAdicField, dot_product,
};
use p3_matrix::dense::DenseMatrix;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use p3_multilinear_util::split_eq::SplitEq;
use p3_util::log2_strict_usize;

use crate::commit::commit_base;
use crate::lagrange::lagrange_weights_01inf_multi;
use crate::layout::opening::Opening;
use crate::layout::prover::Layout;
use crate::layout::witness::{Table, TablePlacement};
use crate::layout::{LayoutStrategy, ProverMultiClaim, ProverVirtualClaim, Witness};
use crate::product_polynomial::ProductPolynomial;
use crate::strategy::{SumcheckProver, VariableOrder};
use crate::svo::{SvoPoint, calculate_accumulators_batch};
use crate::table::{OpeningBatch, OpeningEvals, OpeningRequest};
use crate::{Claim, SumcheckData, extrapolate_01inf};

/// Stacked-sumcheck prover with prefix-first variable binding.
///
/// # Flow
///
/// - Round one runs in SIMD-packed form.
/// - Every later round runs on the residual product polynomial.
#[derive(Debug, Clone)]
pub struct PrefixProver<F: Field, EF: ExtensionField<F>> {
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
    /// Concrete claims recorded per source table.
    ///
    /// # Invariants
    ///
    /// - Every opening stored here is tied to a concrete source column.
    /// - Virtual openings never enter this map.
    /// - Claims are appended in insertion order.
    pub(crate) claim_map: Vec<Vec<ProverMultiClaim<F, EF>>>,
    /// Virtual claims sampled directly on the stacked polynomial.
    pub(crate) virtual_claims: Vec<ProverVirtualClaim<EF>>,
}

impl<F: TwoAdicField, EF: ExtensionField<F>> Layout<F, EF> for PrefixProver<F, EF> {
    fn from_witness(witness: Witness<F>) -> Self {
        // Move the witness fields out so the prover owns them outright.
        let parts = witness.into_parts();
        // One claim list per source table; virtual claims live in their own bucket.
        let num_tables = parts.tables.len();
        Self {
            tables: parts.tables,
            placements: parts.placements,
            num_variables: parts.num_variables,
            folding: parts.folding,
            poly: parts.poly,
            claim_map: (0..num_tables).map(|_| Vec::new()).collect(),
            virtual_claims: Vec::new(),
        }
    }

    fn new_witness(tables: Vec<Table<F>>, folding: usize) -> Witness<F> {
        Witness::new_interleaved(tables, folding)
    }

    fn commit<Dft, MT, Challenger>(
        dft: &Dft,
        mmcs: &MT,
        challenger: &mut Challenger,
        witness: Witness<F>,
        folding: usize,
        starting_log_inv_rate: usize,
    ) -> (Self, MT::Commitment, MT::ProverData<DenseMatrix<F>>)
    where
        Dft: TwoAdicSubgroupDft<F>,
        MT: Mmcs<F>,
        Challenger: CanObserve<MT::Commitment>,
    {
        let (root, prover_data) = commit_base(
            Self::variable_order(),
            dft,
            mmcs,
            challenger,
            &witness.poly,
            folding,
            starting_log_inv_rate,
        );

        (Self::from_witness(witness), root, prover_data)
    }

    fn folding(&self) -> usize {
        self.folding
    }

    /// Returns the number of variables of the stacked polynomial.
    fn num_variables(&self) -> usize {
        self.num_variables
    }

    /// Returns the number of variables of table `id`.
    fn num_variables_table(&self, id: usize) -> usize {
        self.tables[id].num_variables()
    }

    /// Records opening claims for the selected columns of one table.
    ///
    /// All requested columns share one sampled local opening point.
    ///
    /// - Current openings evaluate a column at that point.
    /// - Next openings evaluate the repeat-last successor view at that point.
    /// - Returned evaluations list all current openings first, then all next openings.
    ///
    /// # Arguments
    ///
    /// - `table_idx`  — source table index.
    /// - `batch`      — current and next columns opened at this point.
    /// - `challenger` — Fiat-Shamir transcript.
    ///
    /// # Fiat-Shamir
    ///
    /// - Samples the opening point internally from the transcript.
    /// - Absorbs the evaluations before returning.
    /// - The verifier performs the symmetric absorption.
    ///
    /// # Panics
    ///
    /// - At least one current or next column must be requested.
    #[tracing::instrument(skip_all)]
    fn eval<Ch>(
        &mut self,
        table_idx: usize,
        batch: &OpeningRequest,
        challenger: &mut Ch,
    ) -> OpeningEvals<EF>
    where
        Ch: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        // Split the request into its two column groups.
        let current = batch.current();
        let next = batch.next();
        // Precondition: opening nothing would silently push an empty claim.
        assert!(
            !batch.is_empty(),
            "opening schedule must name at least one column"
        );
        // Sample the local-frame opening point from the transcript.
        let table = &self.tables[table_idx];
        let point = Point::expand_from_univariate(
            challenger.sample_algebra_element(),
            table.num_variables(),
        );

        // Factorise the point once; every selected column reuses it.
        let point = SvoPoint::new_packed(self.folding, &point);

        // Current group: evaluate each column at the point.
        // Each entry yields an opening (carrying preprocessing residuals) plus the bare eval.
        let (current_openings, current_evals): (Vec<_>, Vec<EF>) = current
            .iter()
            .copied()
            .map(|poly_idx| {
                let (eval, partial_evals) = point.eval(table.poly(poly_idx));
                (Opening::new_with_data(poly_idx, eval, partial_evals), eval)
            })
            .unzip();

        // Next group: evaluate the repeat-last successor view at the same point.
        // The prefix layout folds the leading variables, so the successor is taken accordingly.
        let (next_openings, next_evals): (Vec<_>, Vec<EF>) = next
            .iter()
            .copied()
            .map(|poly_idx| {
                let (eval, partial_evals) = point.eval_next_prefix(table.poly(poly_idx));
                (Opening::new_with_data(poly_idx, eval, partial_evals), eval)
            })
            .unzip();

        // Bind the evaluations into the transcript, current group first then next.
        // The verifier absorbs the same bytes in the same order.
        challenger.observe_algebra_slice(&current_evals);
        challenger.observe_algebra_slice(&next_evals);

        // Store the batch for the later sumcheck reduction.
        self.claim_map[table_idx].push(ProverMultiClaim::new(
            point,
            current_openings,
            next_openings,
        ));

        // Return both eval groups in the canonical current-then-next order.
        OpeningBatch::new(current_evals, next_evals)
    }

    /// Samples a virtual evaluation on the full stacked polynomial.
    ///
    /// # Why
    ///
    /// The WHIR protocol occasionally pins the stacked polynomial at a fresh
    /// random point for soundness amplification. Prefix mode evaluates the
    /// stacked polynomial directly — no per-column weighting needed.
    #[tracing::instrument(skip_all)]
    fn add_virtual_eval<Ch>(&mut self, challenger: &mut Ch) -> EF
    where
        Ch: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        // Sample a challenge point covering every stacked variable.
        let point =
            Point::expand_from_univariate(challenger.sample_algebra_element(), self.num_variables);

        let mut eval = EF::ZERO;
        let mut openings = Vec::new();
        let mut weights = Vec::new();

        for placement in &self.placements {
            let table = &self.tables[placement.idx()];
            for (poly_idx, selector) in placement.selectors().iter().enumerate() {
                let poly = table.poly(poly_idx);

                let (local_part, selector_part) = point.split_at(table.num_variables());

                let weight =
                    Point::eval_eq::<EF>(selector.point().as_slice(), selector_part.as_slice());

                let local_svo = SvoPoint::new_packed(self.folding, &local_part);
                let (column_eval, partial_evals) = local_svo.eval(poly);

                eval += weight * column_eval;
                openings.push(Opening {
                    poly_idx: None,
                    eval: column_eval,
                    data: partial_evals,
                });
                weights.push(weight);
            }
        }

        let accumulators = calculate_accumulators_batch(
            &ProverMultiClaim::new(
                SvoPoint::new_unpacked(self.folding, &point, VariableOrder::Prefix),
                openings,
                Vec::new(),
            ),
            &weights,
        );

        // Commit the evaluation to the transcript.
        challenger.observe_algebra_element(eval);
        self.virtual_claims.push(Claim {
            point,
            eval,
            data: accumulators,
        });

        eval
    }

    /// Finalises preprocessing and returns the residual sumcheck prover.
    ///
    /// # Returns
    ///
    /// - Residual sumcheck prover over the packed product polynomial.
    /// - Folding challenges sampled during preprocessing.
    ///
    /// # Algorithm
    ///
    /// ```text
    ///     Phase | Action
    ///     ------+-----------------------------------------------
    ///       1   | Sample the batching challenge  a.
    ///       2   | running sum  = sum_{i}  a^i * eval_i.
    ///       3   | weight poly  = sum_{i}  a^i * eq(z_i, X).
    ///       4   | Fold round 1 in SIMD-packed arithmetic.
    ///       5   | Drive rounds 2..folding on the product polynomial.
    /// ```
    ///
    /// # Precondition
    ///
    /// - Each table's arity is at least  log_2(W), with W the packing width.
    /// - Guarantees every per-slot packed accumulation spans a whole packed element.
    #[tracing::instrument(skip_all)]
    fn into_sumcheck<Ch>(
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

        let mut sum = self.sum(alpha);
        let mut rs = Vec::new();

        for round_idx in 0..self.folding {
            let weights = lagrange_weights_01inf_multi(&rs);

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

            for (vc, alpha_i) in self
                .virtual_claims
                .iter()
                .zip(alpha.shifted_powers(alpha.exp_u64(n_claims as u64)))
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

            let r = sumcheck_data.observe_and_sample(challenger, c0, c_inf, pow_bits);
            sum = extrapolate_01inf(c0, sum - c0, c_inf, r);
            rs.push(r);
        }

        let rs = Point::new(rs);
        let compressed = tracing::info_span!("compress_prefix_to_packed")
            .in_scope(|| self.poly.compress_prefix_to_packed(&rs, EF::ONE));

        let weights = self.residual_weights_packed(&rs, alpha);
        let prod_poly =
            ProductPolynomial::<F, EF>::new_packed(VariableOrder::Prefix, compressed, weights);
        debug_assert_eq!(prod_poly.dot_product(), sum);

        (SumcheckProver::new(prod_poly, sum), rs)
    }

    /// Returns the total number of concrete openings recorded so far.
    fn num_claims(&self) -> usize {
        self.claim_map
            .iter()
            .flat_map(|claims| claims.iter().map(ProverMultiClaim::len))
            .sum()
    }

    fn strategy() -> LayoutStrategy {
        LayoutStrategy::new(true, VariableOrder::Prefix)
    }
}

impl<F: TwoAdicField, EF: ExtensionField<F>> PrefixProver<F, EF> {
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
    pub(crate) fn sum(&self, alpha: EF) -> EF {
        let mut sum = EF::ZERO;
        let mut alphas = alpha.powers();

        // Walk every concrete opening in the canonical insertion order.
        //     placements -> claims -> current openings -> next openings
        // Each opening consumes the next power of alpha, matching the verifier.
        for placement in &self.placements {
            for claim in &self.claim_map[placement.idx()] {
                // Current group first.
                for opening in claim.current_openings() {
                    sum += opening.eval() * alphas.next().unwrap();
                }
                // Next group second, continuing the same power sequence.
                for opening in claim.next_openings() {
                    sum += opening.eval() * alphas.next().unwrap();
                }
            }
        }

        // Virtual claims continue the alpha sequence right after the concrete ones.
        sum += dot_product::<EF, _, _>(
            self.virtual_claims.iter().map(Claim::eval),
            alpha.shifted_powers(alpha.exp_u64(self.num_claims() as u64)),
        );

        sum
    }

    /// Builds the residual equality weights in packed form.
    ///
    /// Two routes produce the identical polynomial; each call takes the cheaper one:
    ///
    /// - Scatter: write directly into the packed buffer, visiting only claimed slots.
    /// - Pack: build the full scalar table, then transpose-copy it into packed form.
    ///
    /// The scatter wins on a sparsely claimed residual.
    /// The vectorized transpose wins when the claims fill most of the space.
    pub(crate) fn residual_weights_packed(
        &self,
        rs: &Point<EF>,
        alpha: EF,
    ) -> Poly<EF::ExtensionPacking> {
        if self.scatter_beats_pack(rs) {
            self.combine_weights_packed(rs, alpha)
        } else {
            self.combine_weights(rs, alpha).pack::<F, EF>()
        }
    }

    /// Decides whether the scatter route beats the transpose route.
    ///
    /// True when the concrete claims occupy at most a third of the residual space:
    ///
    /// ```text
    ///     occupied = sum over openings of 2^(table_arity - folding)
    ///     residual = 2^(num_variables - folding)
    ///     scatter  <=>  3 * occupied <= residual
    /// ```
    ///
    /// The crossover sits near half occupancy:
    ///
    /// - At a quarter, the scatter is clearly ahead.
    /// - At a half, the two routes tie.
    /// - Above a half, the transpose pulls ahead.
    ///
    /// The one-third cutoff keeps a margin below the crossover.
    /// So the chosen route is never slower, across machines and SIMD widths.
    ///
    /// Virtual claims span the whole space and cost the same either way.
    /// They do not enter the decision.
    fn scatter_beats_pack(&self, rs: &Point<EF>) -> bool {
        let residual = 1usize << (self.num_variables - rs.num_variables());
        let occupied: usize = self
            .placements
            .iter()
            .map(|placement| {
                let local =
                    1usize << (self.num_variables_table(placement.idx()) - rs.num_variables());
                self.claim_map[placement.idx()]
                    .iter()
                    .map(|claim| claim.len() * local)
                    .sum::<usize>()
            })
            .sum();
        occupied.saturating_mul(3) <= residual
    }

    /// Builds the residual weight polynomial left after the folded prefix rounds.
    ///
    /// This is the dense-residual route: it materializes the full scalar table.
    /// The packed scatter route shares the same accumulation but skips this buffer.
    ///
    /// # Overview
    ///
    /// - Each concrete opening contributes a slot-local equality weight.
    /// - The slot's selector lifts that weight into the stacked variable space.
    /// - All contributions are summed with the same alpha powers as the batched claim.
    ///
    /// # Arguments
    ///
    /// - `rs`    — folded challenges from the rounds already bound.
    /// - `alpha` — batching challenge whose powers weight each opening.
    ///
    /// # Panics
    ///
    /// - The challenge count must equal the folding depth.
    #[tracing::instrument(skip_all)]
    pub(crate) fn combine_weights(&self, rs: &Point<EF>, alpha: EF) -> Poly<EF> {
        // Invariant: one folded challenge per folded round.
        assert_eq!(rs.num_variables(), self.folding);
        // Output spans the stacked space minus the already-folded variables.
        let mut out = Poly::<EF>::zero(self.num_variables - rs.num_variables());

        let mut alphas = alpha.powers();

        // Same canonical walk as the batched claim, so powers stay aligned.
        for placement in &self.placements {
            // Variables left in each slot after removing the folded ones.
            let local_rest_variables =
                self.num_variables_table(placement.idx()) - rs.num_variables();
            for claim in &self.claim_map[placement.idx()] {
                // Current group: equality weight of the column at the claim point.
                for opening in claim.current_openings() {
                    // The column picks the selector that names this slot.
                    let col = opening.poly_idx().unwrap();
                    let selector = &placement.selectors()[col];
                    // Build the slot-local weight scaled by this opening's alpha power.
                    let mut local = Poly::<EF>::zero(local_rest_variables);
                    claim
                        .point()
                        .accumulate_into(local.as_mut_slice(), rs, alphas.next().unwrap());

                    // Scatter the slot-local weight into its stacked positions.
                    //     dst = (local_idx << selector_vars) | selector_index
                    for (local_idx, &value) in local.as_slice().iter().enumerate() {
                        let dst = (local_idx << selector.num_variables()) | selector.index();
                        out.as_mut_slice()[dst] += value;
                    }
                }
                // Next group: same scatter, but using the repeat-last successor weight.
                for opening in claim.next_openings() {
                    let col = opening.poly_idx().unwrap();
                    let selector = &placement.selectors()[col];
                    let mut local = Poly::<EF>::zero(local_rest_variables);
                    claim.point().accumulate_next_prefix_into(
                        local.as_mut_slice(),
                        rs,
                        alphas.next().unwrap(),
                    );

                    for (local_idx, &value) in local.as_slice().iter().enumerate() {
                        let dst = (local_idx << selector.num_variables()) | selector.index();
                        out.as_mut_slice()[dst] += value;
                    }
                }
            }
        }

        let mut alpha_i = alpha.exp_u64(self.num_claims() as u64);
        for claim in &self.virtual_claims {
            let (svo, rest) = claim.point.split_at(rs.num_variables());
            let scale = alpha_i * Point::eval_eq(svo.as_slice(), rs.as_slice());
            SplitEq::new_unpacked(&rest, scale).accumulate_into(out.as_mut_slice(), None);
            alpha_i *= alpha;
        }

        out
    }

    /// Builds the residual equality weights straight into the packed buffer.
    ///
    /// Scatters each claim's contribution into packed lanes directly.
    /// This skips the scalar table and the transpose-copy the dense route pays.
    ///
    /// # Identity
    ///
    /// For a concrete opening on a column at point `z = (z_svo, z_rest)`:
    ///
    /// ```text
    ///     out[(y << s) | v] += alpha^i * eq(z_svo, rs) * eq(z_rest, y)
    /// ```
    ///
    /// - `(s, v)` is the column's selector: `s` selector bits, slot index `v`.
    /// - `y` ranges over the local table.
    /// - scalar index `(y << s) | v` maps to packed word `idx >> k_pack`, lane `idx & (W - 1)`.
    ///
    /// Virtual claims span the whole residual space.
    /// They continue the alpha sequence after the concrete openings.
    ///
    /// # Panics
    ///
    /// - `rs` must have exactly `folding` variables.
    /// - The residual space must hold at least one packed element.
    #[tracing::instrument(skip_all)]
    pub(crate) fn combine_weights_packed(
        &self,
        rs: &Point<EF>,
        alpha: EF,
    ) -> Poly<EF::ExtensionPacking> {
        assert_eq!(rs.num_variables(), self.folding);
        let k_pack = log2_strict_usize(F::Packing::WIDTH);
        let out_variables = self.num_variables - rs.num_variables();
        assert!(out_variables >= k_pack);

        let mut out = Poly::<EF::ExtensionPacking>::zero(out_variables - k_pack);
        let lane_mask = F::Packing::WIDTH - 1;

        // Scatters one slot-local weight table into its packed positions.
        //     dst = (local_idx << selector_vars) | selector_index
        // The destination splits into packed word `dst >> k_pack` and lane `dst & mask`.
        let scatter = |out: &mut Poly<EF::ExtensionPacking>, local: &[EF], s: usize, v: usize| {
            for (y, &value) in local.iter().enumerate() {
                let idx = (y << s) | v;
                out.as_mut_slice()[idx >> k_pack].add_assign_lane(idx & lane_mask, value);
            }
        };

        // Concrete claims: scatter each column's local table into its packed slot.
        // Alpha powers run in (placement, claim, current openings, next openings) order,
        // matching the verifier and the scalar route.
        let mut alphas = alpha.powers();
        for placement in &self.placements {
            let local_rest_variables =
                self.num_variables_table(placement.idx()) - rs.num_variables();
            for claim in &self.claim_map[placement.idx()] {
                // Current group: equality weight of the column at the claim point.
                for opening in claim.current_openings() {
                    let col = opening.poly_idx().unwrap();
                    let selector = &placement.selectors()[col];
                    // Materialize alpha^i * eq(z_svo, rs) * eq(z_rest, .) for this column.
                    let mut local = Poly::<EF>::zero(local_rest_variables);
                    claim
                        .point()
                        .accumulate_into(local.as_mut_slice(), rs, alphas.next().unwrap());
                    scatter(
                        &mut out,
                        local.as_slice(),
                        selector.num_variables(),
                        selector.index(),
                    );
                }
                // Next group: same scatter, but using the repeat-last successor weight.
                for opening in claim.next_openings() {
                    let col = opening.poly_idx().unwrap();
                    let selector = &placement.selectors()[col];
                    let mut local = Poly::<EF>::zero(local_rest_variables);
                    claim.point().accumulate_next_prefix_into(
                        local.as_mut_slice(),
                        rs,
                        alphas.next().unwrap(),
                    );
                    scatter(
                        &mut out,
                        local.as_slice(),
                        selector.num_variables(),
                        selector.index(),
                    );
                }
            }
        }

        // Each virtual claim spans the full residual space and adds one packed term.
        //
        //     virtual point = [ svo vars | residual vars ]
        //                      \_______/
        //                      fixed to rs by the SVO rounds
        //
        // The packed accumulation folds eq(svo, rs) into a scalar weight.
        // That weight scales the residual equality table added to the output.

        // Virtual claims take the alpha powers right after the concrete claims.
        let mut alpha_i = alpha.exp_u64(self.num_claims() as u64);
        for claim in &self.virtual_claims {
            // Split the claim point into its leading SVO variables and the residual variables.
            let svo_point = SvoPoint::new_packed(rs.num_variables(), &claim.point);
            // Add eq(svo, rs) * eq(residual, .) into the packed buffer, scaled by this claim's power.
            svo_point.accumulate_into_packed(out.as_mut_slice(), rs, alpha_i);
            // Step to the next power of alpha for the following claim.
            alpha_i *= alpha;
        }

        out
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_field::PrimeCharacteristicRing;
    use proptest::prelude::*;

    use super::*;
    use crate::layout::prover::test_utils::{
        FOLDING, arb_opening_schedule, arb_witness_and_schedule, build_tables, tables_from_shape,
    };
    use crate::tests::{EF, F, challenger};

    /// Replays a schedule plus virtual claims, then pins packed against scalar.
    fn assert_packed_matches_scalar(
        witness: Witness<F>,
        schedule: &[(usize, Vec<usize>)],
        num_virtual: usize,
    ) {
        let mut prover = PrefixProver::<F, EF>::from_witness(witness);
        let mut ch = challenger();
        // Record concrete openings; `eval` samples points and absorbs evals internally.
        // These tests exercise current openings only, so the next group stays empty.
        for (table_idx, polys) in schedule {
            prover.eval(
                *table_idx,
                &OpeningBatch::new(polys.clone(), Vec::new()),
                &mut ch,
            );
        }
        // Record virtual claims; they continue the alpha sequence after concrete ones.
        for _ in 0..num_virtual {
            let _ = prover.add_virtual_eval(&mut ch);
        }
        // Random batching challenge and SVO folding point.
        let alpha: EF = ch.sample_algebra_element();
        let rs = Point::expand_from_univariate(ch.sample_algebra_element(), FOLDING);
        // The packed path must equal the scalar reference followed by packing.
        let scalar = prover.combine_weights(&rs, alpha).pack::<F, EF>();
        let packed = prover.combine_weights_packed(&rs, alpha);
        assert_eq!(scalar, packed);
        // The adaptive dispatcher must return that same polynomial on either branch.
        assert_eq!(prover.residual_weights_packed(&rs, alpha), packed);
    }

    #[test]
    fn scatter_beats_pack_routes_on_occupancy() {
        let mut ch = challenger();
        let rs = Point::expand_from_univariate(ch.sample_algebra_element(), FOLDING);

        // Builds a four-column table and opens the first `open` of them.
        // Four columns tile a power-of-two space, so `open` columns is `open/4` occupancy.
        let mut routed = |open: usize| {
            let mut prover =
                PrefixProver::<F, EF>::from_witness(PrefixProver::<F, EF>::new_witness(
                    vec![Table::new((0..4).map(|_| Poly::<F>::zero(8)).collect())],
                    FOLDING,
                ));
            let cols: Vec<usize> = (0..open).collect();
            prover.eval(0, &OpeningBatch::new(cols, Vec::new()), &mut ch);
            prover.scatter_beats_pack(&rs)
        };

        // The threshold routes to the scatter only below a third occupancy.
        assert!(routed(1), "25% occupancy must pick scatter");
        assert!(!routed(2), "50% occupancy must fall back to pack");
        assert!(!routed(4), "100% occupancy must fall back to pack");

        // A holey layout (one of five columns) sits far below the threshold.
        let mut sparse = PrefixProver::<F, EF>::from_witness(PrefixProver::<F, EF>::new_witness(
            vec![Table::new((0..5).map(|_| Poly::<F>::zero(8)).collect())],
            FOLDING,
        ));
        sparse.eval(0, &OpeningBatch::new(vec![2], Vec::new()), &mut ch);
        assert!(
            sparse.scatter_beats_pack(&rs),
            "holey layout must pick scatter"
        );
    }

    #[test]
    fn combine_weights_packed_no_claims_is_zero() {
        // No claims at all: the scatter is skipped and every weight is zero.
        let witness = PrefixProver::<F, EF>::new_witness(build_tables(), FOLDING);
        let prover = PrefixProver::<F, EF>::from_witness(witness);
        let mut ch = challenger();
        let alpha: EF = ch.sample_algebra_element();
        let rs = Point::expand_from_univariate(ch.sample_algebra_element(), FOLDING);
        let packed = prover.combine_weights_packed(&rs, alpha);
        assert!(
            packed
                .iter()
                .all(|&w| w == <EF as ExtensionField<F>>::ExtensionPacking::ZERO)
        );
    }

    #[test]
    fn combine_weights_packed_virtual_claims_only() {
        // No concrete openings: only the virtual-claim accumulation contributes.
        let witness = PrefixProver::<F, EF>::new_witness(build_tables(), FOLDING);
        assert_packed_matches_scalar(witness, &[], 2);
    }

    proptest! {
        #![proptest_config(ProptestConfig { cases: 32, ..ProptestConfig::default() })]

        // Invariant: the packed scatter equals the scalar build followed by pack.
        //
        //     fixed two-table witness: mixed selector widths, padding holes
        //     coverage: openings sharing a slot, plus 0..=2 virtual claims
        #[test]
        fn combine_weights_packed_matches_scalar(
            schedule in arb_opening_schedule(),
            num_virtual in 0usize..=2,
        ) {
            let witness = PrefixProver::<F, EF>::new_witness(build_tables(), FOLDING);
            assert_packed_matches_scalar(witness, &schedule, num_virtual);
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig { cases: 16, ..ProptestConfig::default() })]

        // Invariant: packed and scalar agree on random witness shapes.
        //
        //     includes single-table layouts with zero selector bits
        #[test]
        fn combine_weights_packed_matches_scalar_shapes(
            (shape, schedule) in arb_witness_and_schedule(),
            num_virtual in 0usize..=1,
        ) {
            let witness = PrefixProver::<F, EF>::new_witness(tables_from_shape(&shape), FOLDING);
            // Packing needs at least one full SIMD lane in the residual space.
            let k_pack = log2_strict_usize(<F as Field>::Packing::WIDTH);
            prop_assume!(witness.num_variables() >= FOLDING + k_pack);
            assert_packed_matches_scalar(witness, &schedule, num_virtual);
        }
    }
}
