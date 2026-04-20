//! Prefix-mode stacked-sumcheck prover.

use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, Field, PackedFieldExtension, PackedValue, dot_product};
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use p3_multilinear_util::split_eq::SplitEq;
use p3_util::log2_strict_usize;

use super::util::traverse_openings;
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
    pub(crate) num_vars: usize,
    /// Number of preprocessing rounds consumed before residual sumcheck.
    pub(crate) folding: usize,
    /// Stacked committed polynomial.
    pub(crate) poly: Poly<F>,
    /// Concrete claims recorded per source table.
    pub(crate) claim_map: Vec<Vec<PrefixMultiClaim<F, EF>>>,
    /// Virtual claims sampled directly on the stacked polynomial.
    pub(crate) virtual_claims: Vec<PrefixVirtualClaim<F, EF>>,
}

impl<F: Field, EF: ExtensionField<F>> PrefixProver<F, EF> {
    /// Returns the number of variables of the stacked polynomial.
    pub const fn num_vars(&self) -> usize {
        self.num_vars
    }

    /// Returns the number of variables of table `id`.
    pub fn num_vars_table(&self, id: usize) -> usize {
        self.tables[id].num_vars()
    }

    /// Returns the stacked committed polynomial.
    pub const fn poly(&self) -> &Poly<F> {
        &self.poly
    }

    /// Records opening claims for the selected columns at `point`.
    ///
    /// # Arguments
    ///
    /// - `point`      — evaluation point in the table's local variable space.
    /// - `table_idx`  — source table index.
    /// - `polys`      — columns to open.
    #[tracing::instrument(skip_all)]
    pub fn eval(&mut self, point: &Point<EF>, table_idx: usize, polys: &[usize]) -> Vec<EF> {
        // Invariant: the evaluation point lives in the table's local frame.
        let table = &self.tables[table_idx];
        assert_eq!(point.num_vars(), table.num_vars());

        // Factorise the point once; every selected column reuses it.
        let point = SplitEq::new_packed(point, EF::ONE);

        // Evaluate each column at the factored point; split into (opening, eval).
        let (openings, evals) = polys
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
            Point::expand_from_univariate(challenger.sample_algebra_element(), self.num_vars);

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
        assert!(self.folding <= self.num_vars);

        // Precondition for packed per-slot accumulation.
        let k_pack = log2_strict_usize(F::Packing::WIDTH);
        assert!(
            self.placements
                .iter()
                .all(|p| self.num_vars_table(p.idx()) >= k_pack),
            "prefix mode requires num_vars_table >= log2(packing width) for every table",
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
    fn sum(&self, alpha: EF) -> EF {
        let mut sum = EF::ZERO;

        // Concrete openings: scale each by its alpha power and accumulate.
        traverse_openings(
            &self.placements,
            |id| self.num_vars_table(id),
            &self.claim_map,
            alpha,
            |v| sum += v.opening.eval() * v.alpha,
        );

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
    /// Concrete claims contribute into their owning slot only.
    /// Virtual claims span the entire stacked output buffer.
    #[tracing::instrument(skip_all)]
    fn combine_eqs(&self, alpha: EF) -> Poly<EF::ExtensionPacking> {
        // Packed output has 2^(num_vars - k_pack) entries; each holds WIDTH scalars.
        let k_pack = log2_strict_usize(F::Packing::WIDTH);
        let mut out = Poly::<EF::ExtensionPacking>::zero(self.num_vars - k_pack);

        // Concrete claims write into their per-slot packed range.
        traverse_openings(
            &self.placements,
            |id| self.num_vars_table(id),
            &self.claim_map,
            alpha,
            |v| {
                let packed_range = (v.slot.start >> k_pack)..(v.slot.end >> k_pack);
                v.claim
                    .point()
                    .accumulate_into_packed(&mut out.as_mut_slice()[packed_range], Some(v.alpha));
            },
        );

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

    use p3_multilinear_util::point::Point;
    use proptest::prelude::*;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::PrefixProver;
    use crate::sumcheck::layout::prover::test_utils::{
        ASCENDING_POLYS, NON_ASCENDING_POLYS, arb_opening_schedule, build_witness,
        run_prefix_roundtrip,
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

        // Deterministic RNG so point values stay reproducible across runs.
        let mut rng = SmallRng::seed_from_u64(42);
        let point_a = Point::<EF>::rand(&mut rng, prover.num_vars_table(0));
        let point_b = Point::<EF>::rand(&mut rng, prover.num_vars_table(1));

        // Record two openings on table 0 → count advances from 0 to 2.
        prover.eval(&point_a, 0, &[0, 1]);
        assert_eq!(prover.num_claims(), 2);

        // Record one more opening on table 1 → count advances from 2 to 3.
        prover.eval(&point_b, 1, &[0]);
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

        // Sample a deterministic opening point with the table's arity.
        let mut rng = SmallRng::seed_from_u64(7);
        let point = Point::<EF>::rand(&mut rng, prover.num_vars_table(0));

        // Record two openings; evals[0], evals[1] line up with columns 0, 1.
        let evals = prover.eval(&point, 0, &[0, 1]);

        // Pick a random batching challenge and compute the expected sum by hand.
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
}
