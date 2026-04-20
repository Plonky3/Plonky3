use alloc::vec::Vec;

use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, Field, PackedFieldExtension, PackedValue};
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use p3_multilinear_util::split_eq::SplitEq;
use p3_util::log2_strict_usize;

use crate::sumcheck::layout::{
    LayoutStrategy, MultiClaimPrefixLayout as MultiClaim, OpeningPrefixLayout as Opening,
    ProverLayout, VirtualClaimPrefixLayout as VirtualClaim,
};
use crate::sumcheck::product_polynomial::ProductPolynomial;
use crate::sumcheck::strategy::{PrefixSumcheck, SumcheckProver, SumcheckStrategy};
use crate::sumcheck::{SumcheckData, extrapolate_01inf};

/// Layout strategy where claimed variables are processed in prefix order.
#[derive(Debug, Clone)]
pub struct PrefixLayout;

/// Concrete prover layout for the prefix strategy.
pub type ProverPrefixLayout<F, EF> = ProverLayout<F, EF, PrefixLayout>;

impl PrefixLayout {
    /// Builds the packed equality-weight polynomial used after the first packed round.
    #[tracing::instrument(skip_all)]
    fn combine_eqs<F: Field, EF: ExtensionField<F>>(
        l: &ProverLayout<F, EF, Self>,
        alpha: EF,
    ) -> Poly<EF::ExtensionPacking> {
        let k_pack = log2_strict_usize(F::Packing::WIDTH);
        let mut out = Poly::<EF::ExtensionPacking>::zero(l.num_vars - k_pack);
        l.for_each_opening(alpha, |_, _, _, claim, _, alpha_i, range| {
            let packed_range = (range.start >> k_pack)..(range.end >> k_pack);
            claim.accumulate_into_packed(&mut out.as_mut_slice()[packed_range], Some(alpha_i));
        });

        let mut alpha_i = alpha.exp_u64(l.num_claims() as u64);
        l.virtual_claims.iter().for_each(|claim| {
            SplitEq::new_packed(&claim.point, alpha_i)
                .accumulate_into_packed(out.as_mut_slice(), None);
            alpha_i *= alpha;
        });

        out
    }
}

impl LayoutStrategy for PrefixLayout {
    /// Prefix sumcheck continues on the residual product polynomial.
    type SumcheckStrategy = PrefixSumcheck;
    /// Prefix openings are represented by split-equality tables.
    type Point<F: Field, EF: ExtensionField<F>> = SplitEq<F, EF>;
    /// Prefix openings need no extra per-opening prover data.
    type DataOpening<EF: Field> = ();
    /// Prefix virtual claims carry only their evaluation.
    type DataVirtual<EF: Field> = ();

    /// Records concrete openings for one table at the given local point.
    fn eval<F: Field, EF: ExtensionField<F>>(
        l: &mut ProverLayout<F, EF, Self>,
        point: &Point<EF>,
        table_idx: usize,
        polys: Vec<usize>,
    ) -> Vec<EF> {
        let table = &l.tables[table_idx];
        assert_eq!(point.num_vars(), table.num_vars());
        let point = SplitEq::new_packed(point, EF::ONE);
        let openings = polys
            .iter()
            .map(|&poly_idx| table.eval(poly_idx, &point))
            .collect::<Vec<_>>();
        let evals = openings.iter().map(Opening::eval).collect::<Vec<_>>();
        l.claim_map[table_idx].push(MultiClaim::new(point, openings));
        evals
    }

    /// Samples and records a virtual opening on the full stacked polynomial.
    fn add_virtual_eval<Challenger, F: Field, EF: ExtensionField<F>>(
        l: &mut ProverLayout<F, EF, Self>,
        challenger: &mut Challenger,
    ) -> EF
    where
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        let point =
            Point::expand_from_univariate(challenger.sample_algebra_element(), l.num_vars());
        let eval = l.poly.eval_base(&point);
        challenger.observe_algebra_element(eval);
        l.virtual_claims.push(VirtualClaim::new(point, eval));
        eval
    }

    /// Builds the initial prefix-round product polynomial and consumes `folding` rounds.
    fn new_prover<Challenger, F: Field, EF: ExtensionField<F>>(
        l: ProverLayout<F, EF, Self>,
        sumcheck_data: &mut SumcheckData<F, EF>,
        pow_bits: usize,
        challenger: &mut Challenger,
    ) -> (SumcheckProver<F, EF, Self::SumcheckStrategy>, Point<EF>)
    where
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        assert!(l.folding() <= l.num_vars());
        let folding = l.folding;
        let alpha: EF = challenger.sample_algebra_element();

        let mut sum = l.sum(alpha);
        let mut weights = Self::combine_eqs(&l, alpha);
        let poly = F::Packing::pack_slice(l.poly.as_slice());

        let (c0, c_inf) = PrefixSumcheck::sumcheck_coefficients(poly, weights.as_slice());
        let c0 = EF::ExtensionPacking::to_ext_iter([c0]).sum();
        let c_inf = EF::ExtensionPacking::to_ext_iter([c_inf]).sum();

        let r = sumcheck_data.observe_and_sample(challenger, c0, c_inf, pow_bits);
        weights.fix_prefix_var_mut(r);
        let poly = l.poly.fix_prefix_var_to_packed(r);
        sum = extrapolate_01inf(c0, sum - c0, c_inf, r);

        let mut prod_poly = ProductPolynomial::<F, EF, PrefixSumcheck>::new_packed(poly, weights);
        debug_assert_eq!(prod_poly.dot_product(), sum);

        let rs = core::iter::once(r)
            .chain(
                (1..folding)
                    .map(|_| prod_poly.round(sumcheck_data, challenger, &mut sum, pow_bits)),
            )
            .collect::<Vec<_>>();

        (SumcheckProver::new(prod_poly, sum), Point::new(rs))
    }
}
