use alloc::vec;
use alloc::vec::Vec;

use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, Field, dot_product};
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use p3_multilinear_util::split_eq::SplitEq;

use crate::sumcheck::lagrange::lagrange_weights_01inf_multi;
use crate::sumcheck::layout::{
    LayoutStrategy, MultiClaimSuffixLayout as MultiClaim, OpeningSuffixLayout as Opening,
    ProverLayout, VirtualClaimSuffixLayout as VirtualClaim,
};
use crate::sumcheck::product_polynomial::ProductPolynomial;
use crate::sumcheck::strategy::{SuffixSumcheck, SumcheckProver, VariableOrder};
use crate::sumcheck::svo::{SvoPoint, calculate_accumulators_batch};
use crate::sumcheck::{SumcheckData, extrapolate_01inf};

/// Layout strategy where the first preprocessing rounds eliminate suffix variables.
#[derive(Debug, Clone)]
pub struct SuffixLayout;

/// Concrete prover layout for the suffix strategy.
pub type ProverSuffixLayout<F, EF> = ProverLayout<F, EF, SuffixLayout>;

impl SuffixLayout {
    /// Compresses every stacked table polynomial by fixing the suffix challenges `rs`.
    fn compress_stacked<F: Field, EF: ExtensionField<F>>(
        l: &ProverLayout<F, EF, Self>,
        rs: &Point<EF>,
    ) -> Poly<EF> {
        assert!(rs.num_vars() <= l.num_vars());
        let mut out = Poly::<EF>::zero(l.num_vars() - rs.num_vars());
        let rs = SplitEq::new_unpacked(rs, EF::ONE);
        for table_layout in l.layout.iter() {
            for (poly_idx, selector) in table_layout.selectors.iter().enumerate() {
                let poly = l.tables[table_layout.idx].poly(poly_idx);
                assert!(rs.num_vars() <= poly.num_vars());
                let off = selector.index << (poly.num_vars() - rs.num_vars());
                rs.compress_suffix_into(
                    &mut out.as_mut_slice()[off..off + (1 << (poly.num_vars() - rs.num_vars()))],
                    poly,
                );
            }
        }
        out
    }

    /// Combines all concrete and virtual suffix claims into one residual weight polynomial.
    #[tracing::instrument(skip_all)]
    fn combine_eqs<F: Field, EF: ExtensionField<F>>(
        l: &ProverLayout<F, EF, Self>,
        rs: &Point<EF>,
        alpha: EF,
    ) -> Poly<EF> {
        assert_eq!(rs.num_vars(), l.folding);
        let mut out = Poly::<EF>::zero(l.num_vars - rs.num_vars());
        l.for_each_opening(alpha, |_, _, _, claim, _, alpha_i, range| {
            let folded_range = (range.start >> l.folding)..(range.end >> l.folding);
            claim.accumulate_into(&mut out.as_mut_slice()[folded_range], rs, alpha_i);
        });

        let mut alpha_i = alpha.exp_u64(l.num_claims() as u64);
        l.virtual_claims.iter().for_each(|claim| {
            let (rest, svo) = claim.point.split_at(claim.point.num_vars() - rs.num_vars());
            let scale = alpha_i * Point::eval_eq(svo.as_slice(), rs.as_slice());
            SplitEq::new_packed(&rest, scale).accumulate_into(out.as_mut_slice(), None);
            alpha_i *= alpha;
        });

        out
    }

    /// Constructs the residual product polynomial after all SVO suffix rounds are fixed.
    #[tracing::instrument(skip_all)]
    fn prod_poly<F: Field, EF: ExtensionField<F>>(
        l: &ProverLayout<F, EF, Self>,
        rs: &Point<EF>,
        alpha: EF,
    ) -> ProductPolynomial<F, EF, SuffixSumcheck> {
        let poly = Self::compress_stacked(l, rs);
        let weights = Self::combine_eqs(l, rs, alpha);
        ProductPolynomial::new_unpacked(poly, weights)
    }
}

impl LayoutStrategy for SuffixLayout {
    /// Suffix sumcheck continues on the residual unpacked product polynomial.
    type SumcheckStrategy = SuffixSumcheck;
    /// Suffix openings use `SvoPoint` to track the SVO split and partials.
    type Point<F: Field, EF: ExtensionField<F>> = SvoPoint<F, EF>;
    /// Suffix openings store one partial evaluation per preprocessing round.
    type DataOpening<EF: Field> = Vec<Poly<EF>>;
    /// Suffix virtual claims also carry precomputed SVO accumulators.
    type DataVirtual<EF: Field> = crate::sumcheck::svo::SvoAccumulators<EF>;

    /// Records concrete openings for one table at the given local point.
    fn eval<F: Field, EF: ExtensionField<F>>(
        l: &mut ProverLayout<F, EF, Self>,
        point: &Point<EF>,
        table_idx: usize,
        polys: Vec<usize>,
    ) -> Vec<EF> {
        let table = &l.tables[table_idx];
        assert_eq!(point.num_vars(), table.num_vars());
        let point = SvoPoint::new_unpacked(l.folding, point, VariableOrder::Suffix);
        let openings = polys
            .iter()
            .map(|&poly_idx| table.eval_svo(poly_idx, &point))
            .collect::<Vec<_>>();
        let evals = openings.iter().map(Opening::eval).collect::<Vec<_>>();
        l.claim_map[table_idx].push(MultiClaim::new(point, openings));
        evals
    }

    /// Samples and records a virtual opening together with its batched SVO accumulators.
    fn add_virtual_eval<Challenger, F: Field, EF: ExtensionField<F>>(
        l: &mut ProverLayout<F, EF, Self>,
        challenger: &mut Challenger,
    ) -> EF
    where
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        let point =
            Point::expand_from_univariate(challenger.sample_algebra_element(), l.num_vars());
        let mut eval = EF::ZERO;
        let mut openings = Vec::new();
        let mut weights = Vec::new();

        for table_layout in l.layout.iter() {
            for (poly_idx, selector) in table_layout.selectors.iter().enumerate() {
                let poly = l.tables[table_layout.idx].poly(poly_idx);
                let (selector_part, local_part) = point.split_at(selector.num_vars);
                let weight =
                    Point::eval_eq::<EF>(selector.point().as_slice(), selector_part.as_slice());
                let opening = Opening::eval_poly(
                    None,
                    &SvoPoint::new_unpacked(l.folding(), &local_part, VariableOrder::Suffix),
                    poly,
                );
                eval += weight * opening.eval();
                openings.push(opening);
                weights.push(weight);
            }
        }

        let accumulators = calculate_accumulators_batch(
            &MultiClaim::new(
                SvoPoint::new_unpacked(l.folding, &point, VariableOrder::Suffix),
                openings,
            ),
            &weights,
        );

        #[cfg(debug_assertions)]
        {
            let poly = &Self::compress_stacked(l, &Point::default());
            assert_eq!(eval, poly.eval_base(&point));

            let opening = Opening::eval_poly::<EF>(
                None,
                &SvoPoint::<EF, EF>::new_unpacked(l.folding, &point, VariableOrder::Suffix),
                poly,
            );
            assert_eq!(eval, opening.eval());
            assert_eq!(
                accumulators,
                calculate_accumulators_batch(
                    &MultiClaim::new(
                        SvoPoint::new_unpacked(l.folding, &point, VariableOrder::Suffix),
                        vec![opening],
                    ),
                    &[EF::ONE]
                )
            );
        }

        challenger.observe_algebra_element(eval);
        l.virtual_claims
            .push(VirtualClaim::new(point, eval, accumulators));
        eval
    }

    /// Consumes the explicit suffix rounds, then returns a prover for the residual product.
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
        let alpha: EF = challenger.sample_algebra_element();
        let n_claims = l.num_claims();

        // Pre-size each inner vector to the claim's opening count.
        // Indexing by opening_idx keeps alphas aligned with natural-order
        // partial evals; for_each_opening fires in poly_idx-sorted order,
        // so push-based accumulation would swap alphas on non-ascending polys.
        let mut claim_alphas = l
            .claim_map
            .iter()
            .map(|claims| {
                claims
                    .iter()
                    .map(|claim| vec![EF::ZERO; claim.len()])
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        l.for_each_opening(alpha, |table_idx, claim_idx, opening_idx, _, _, alpha_i, _| {
            claim_alphas[table_idx][claim_idx][opening_idx] = alpha_i;
        });

        let claim_alphas = &claim_alphas;
        let accumulators = l
            .claim_map
            .iter()
            .enumerate()
            .flat_map(|(table_idx, claims)| {
                claims.iter().enumerate().map(move |(claim_idx, claim)| {
                    calculate_accumulators_batch(claim, &claim_alphas[table_idx][claim_idx])
                })
            })
            .collect::<Vec<_>>();

        let mut sum = l.sum(alpha);
        let mut rs = vec![];

        for round_idx in 0..l.folding {
            let weights = lagrange_weights_01inf_multi(rs.as_slice());
            let mut acc0 = vec![EF::ZERO; weights.len()];
            let mut acc_inf = vec![EF::ZERO; weights.len()];

            for accumulators in accumulators.iter() {
                acc0.iter_mut()
                    .zip(accumulators[round_idx][0].iter())
                    .for_each(|(acc, &w)| *acc += w);
                acc_inf
                    .iter_mut()
                    .zip(accumulators[round_idx][1].iter())
                    .for_each(|(acc, &w)| *acc += w);
            }

            for (vc, alpha_i) in l.virtual_claims.iter().zip(alpha.powers().skip(n_claims)) {
                let vc_accumulators = &vc.data;
                acc0.iter_mut()
                    .zip(vc_accumulators[round_idx][0].iter())
                    .for_each(|(acc, &w)| *acc += alpha_i * w);
                acc_inf
                    .iter_mut()
                    .zip(vc_accumulators[round_idx][1].iter())
                    .for_each(|(acc, &w)| *acc += alpha_i * w);
            }

            let c0 = dot_product::<EF, _, _>(acc0.iter().copied(), weights.iter().copied());
            let c_inf = dot_product::<EF, _, _>(acc_inf.iter().copied(), weights.iter().copied());

            let r = sumcheck_data.observe_and_sample(challenger, c0, c_inf, pow_bits);
            sum = extrapolate_01inf(c0, sum - c0, c_inf, r);
            rs.push(r);
        }

        let rs = Point::new(rs);
        let poly = Self::prod_poly(&l, &rs.reversed(), alpha);
        debug_assert_eq!(poly.dot_product(), sum);
        (SumcheckProver::new(poly, sum), rs)
    }
}
