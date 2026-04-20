//! Prefix-mode stacked-sumcheck prover.

use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, Field, PackedFieldExtension, PackedValue, dot_product};
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use p3_multilinear_util::split_eq::SplitEq;
use p3_util::log2_strict_usize;

use crate::sumcheck::layout::opening::{MultiClaim, Opening, PrefixMultiClaim, PrefixVirtualClaim};
use crate::sumcheck::layout::witness::{Table, TablePlacement};
use crate::sumcheck::product_polynomial::ProductPolynomial;
use crate::sumcheck::strategy::{SumcheckProver, VariableOrder};
use crate::sumcheck::{Claim, SumcheckData, extrapolate_01inf};

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
    pub(crate) claim_map: Vec<Vec<PrefixMultiClaim<F, EF>>>,
    /// Virtual claims sampled directly on the stacked polynomial.
    pub(crate) virtual_claims: Vec<PrefixVirtualClaim<F, EF>>,
}

impl<F: Field, EF: ExtensionField<F>> PrefixProver<F, EF> {
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

        // Factorise the point once; every selected column reuses it.
        let point = SplitEq::new_packed(&point, EF::ONE);

        // Evaluate each column at the factored point; split into (opening, eval).
        let (openings, evals): (Vec<_>, Vec<EF>) = polys
            .iter()
            .map(|&poly_idx| {
                let eval = point.eval_base(table.poly(poly_idx));
                let opening = Opening {
                    poly_idx: Some(poly_idx),
                    eval,
                    data: (),
                };
                (opening, eval)
            })
            .unzip();

        // Bind the evaluations into the transcript; the verifier absorbs the same bytes.
        challenger.observe_algebra_slice(&evals);

        // Store the batch for the later sumcheck reduction.
        self.claim_map[table_idx].push(PrefixMultiClaim::new(point, openings));

        evals
    }

    /// Samples a virtual evaluation on the full stacked polynomial.
    ///
    /// # Why
    ///
    /// The WHIR protocol occasionally pins the stacked polynomial at a fresh
    /// random point for soundness amplification. Prefix mode evaluates the
    /// stacked polynomial directly — no per-column weighting needed.
    #[tracing::instrument(skip_all)]
    pub fn add_virtual_eval<Ch>(&mut self, challenger: &mut Ch) -> EF
    where
        Ch: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        // Sample a challenge point covering every stacked variable.
        let point =
            Point::expand_from_univariate(challenger.sample_algebra_element(), self.num_variables);

        // Evaluate the stacked polynomial directly.
        let eval = self.poly.eval_base(&point);

        // Commit the evaluation to the transcript.
        challenger.observe_algebra_element(eval);

        // Record the claim; data payload is unit in prefix mode.
        self.virtual_claims.push(Claim {
            point,
            eval,
            data: (),
            _marker: PhantomData,
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

        // Precondition for packed per-slot accumulation.
        let k_pack = log2_strict_usize(F::Packing::WIDTH);
        assert!(
            self.placements
                .iter()
                .all(|p| self.num_variables_table(p.idx()) >= k_pack),
            "prefix mode requires num_variables_table >= log2(packing width) for every table",
        );

        // Batching challenge for combining every recorded claim.
        let folding = self.folding;
        let alpha: EF = challenger.sample_algebra_element();

        // Phase 1: build the initial (sum, weights, poly) triple.
        let mut sum = self.sum(alpha);
        let mut weights = self.combine_eqs(alpha);
        let poly = F::Packing::pack_slice(self.poly.as_slice());

        // First-round coefficients in packed arithmetic.
        let (c0, c_inf) = VariableOrder::Prefix.sumcheck_coefficients(poly, weights.as_slice());
        // Horizontal reduction across SIMD lanes.
        let c0 = EF::ExtensionPacking::to_ext_iter([c0]).sum();
        let c_inf = EF::ExtensionPacking::to_ext_iter([c_inf]).sum();

        // Phase 2: run round 1 (observe coefficients, sample r, fold everything).
        let r = sumcheck_data.observe_and_sample(challenger, c0, c_inf, pow_bits);
        weights.fix_prefix_var_mut(r);
        let poly = self.poly.fix_prefix_var_to_packed(r);
        sum = extrapolate_01inf(c0, sum - c0, c_inf, r);

        // Phase 3: pack (poly, weights) into a product polynomial for the residual rounds.
        let mut prod_poly =
            ProductPolynomial::<F, EF>::new_packed(VariableOrder::Prefix, poly, weights);
        debug_assert_eq!(prod_poly.dot_product(), sum);

        // Phase 4: rounds 2..folding run on the product polynomial directly.
        let rs = core::iter::once(r)
            .chain(
                (1..folding)
                    .map(|_| prod_poly.round(sumcheck_data, challenger, &mut sum, pow_bits)),
            )
            .collect();

        (SumcheckProver::new(prod_poly, sum), Point::new(rs))
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

    /// Builds the packed equality-weight polynomial for the first packed round.
    ///
    /// ```text
    ///     out(x) = sum_{i}  alpha^i * eq(z_i, x)
    /// ```
    ///
    /// - Concrete openings write into their owning slot, found via the
    ///   selector for the opening's column.
    /// - Virtual claims span the entire stacked output buffer.
    #[tracing::instrument(skip_all)]
    fn combine_eqs(&self, alpha: EF) -> Poly<EF::ExtensionPacking> {
        // Packed output has 2^(num_variables - k_pack) entries; each holds WIDTH scalars.
        let k_pack = log2_strict_usize(F::Packing::WIDTH);
        let mut out = Poly::<EF::ExtensionPacking>::zero(self.num_variables - k_pack);

        let mut alphas = alpha.powers();

        // Concrete openings: write each into the slot its column's selector addresses.
        for placement in &self.placements {
            let num_variables_table = self.num_variables_table(placement.idx());
            let slot_size = 1usize << num_variables_table;
            for claim in &self.claim_map[placement.idx()] {
                for opening in claim.openings() {
                    // The opening's column tells us which selector picks the slot.
                    let col = opening.poly_idx().unwrap();
                    let off = placement.selectors()[col].index() << num_variables_table;
                    let packed_range = (off >> k_pack)..((off + slot_size) >> k_pack);
                    claim.point().accumulate_into_packed(
                        &mut out.as_mut_slice()[packed_range],
                        Some(alphas.next().unwrap()),
                    );
                }
            }
        }

        // Virtual claims contribute across the whole output.
        // Start alpha where the concrete openings stopped.
        let mut alpha_i = alpha.exp_u64(self.num_claims() as u64);
        for claim in &self.virtual_claims {
            SplitEq::new_packed(&claim.point, alpha_i)
                .accumulate_into_packed(out.as_mut_slice(), None);
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

    use super::PrefixProver;
    use crate::sumcheck::layout::prover::test_utils::{
        ASCENDING_POLYS, NON_ASCENDING_POLYS, arb_opening_schedule, arb_witness_and_schedule,
        build_witness, build_witness_from_shape, run_prefix_roundtrip, run_prefix_roundtrip_with,
        table_shapes_from,
    };
    use crate::sumcheck::tests::*;

    #[test]
    fn prefix_num_claims_counts_every_recorded_opening() {
        // Invariant:
        //     Concrete opening count equals the sum over every (table, claim).
        //
        // Fixture state:
        //     fresh prover:                  0 concrete openings
        //     after eval(table 0, [0, 1]):   2 concrete openings
        //     after eval(table 1, [0]):      3 concrete openings
        let witness = build_witness();
        let mut prover: PrefixProver<F, EF> = witness.as_prefix_prover();
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
    fn prefix_sum_matches_weighted_eval_sum() {
        // Invariant:
        //     sum(alpha) = eval_0 * alpha^0 + eval_1 * alpha^1
        //     for a single claim opening two columns in traversal order.
        //
        // Fixture state:
        //     one eval call recording two openings on one claim.
        //     traversal order: column 0 before column 1.
        let witness = build_witness();
        let mut prover: PrefixProver<F, EF> = witness.as_prefix_prover();

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
    fn prefix_roundtrip_ascending_polys() {
        // Baseline: columns opened in ascending order inside each claim.
        run_prefix_roundtrip(ASCENDING_POLYS);
    }

    #[test]
    fn prefix_roundtrip_non_ascending_polys() {
        // Opening order within a claim must not affect correctness.
        run_prefix_roundtrip(NON_ASCENDING_POLYS);
    }

    proptest! {
        #![proptest_config(ProptestConfig { cases: 16, ..ProptestConfig::default() })]

        // Invariant:
        //     Every valid opening schedule roundtrips through the protocol
        //     without the prover and verifier diverging.
        //
        // Strategy: 1..=6 random calls over the fixed two-table witness.
        #[test]
        fn prefix_roundtrip_proptest(schedule in arb_opening_schedule()) {
            // Adapt the owned schedule to the slice-pair shape the runner expects.
            let borrowed: Vec<(usize, &[usize])> = schedule
                .iter()
                .map(|(t, polys)| (*t, polys.as_slice()))
                .collect();
            // Drive both sides; the runner asserts agreement internally.
            run_prefix_roundtrip(&borrowed);
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
        fn prefix_roundtrip_shape_proptest(
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
            run_prefix_roundtrip_with(witness, &shapes, &borrowed);
        }
    }
}
