//! Product accumulators on the ternary grid and per-round batch assembly.
//!
//! Each SVO round emits the round polynomial evaluated at `0` and at `inf`.
//! These accumulators are built from pointwise products on the first and last
//! thirds of the `{0, 1, inf}^l` grid.

use alloc::vec;
use alloc::vec::Vec;

use p3_field::{ExtensionField, Field};

use super::grid::evals_01inf_grid_into;
use crate::layout::{EqPartials, NextPartials, ProverMultiClaim};
use crate::strategy::VariableOrder;

/// Per-round accumulator slices for the SVO rounds.
///
/// For round `i + 1`, the entry stores:
/// - `[0]`: accumulator values on `{0,1,2}^i x {0}`
/// - `[1]`: accumulator values on `{0,1,2}^i x {2}`
///
/// We do not store the `x {1}` face because the verifier can derive `h(1)` from the
/// claimed sum and `h(0)`.
pub(crate) type SvoAccumulators<EF> = Vec<[Vec<EF>; 2]>;

/// Builds the SVO round accumulators for a batch of openings.
///
/// # Overview
///
/// - Produces, for each SVO round, the round polynomial evaluated at `0` and at `inf`.
/// - Combines equality-weighted openings and repeat-last-successor openings with the batching coefficients.
/// - Keeps only the first and last thirds of the ternary grid, which reconstruct the quadratic round polynomial.
///
/// # Arguments
///
/// - `claim`: the batched opening claim carrying the point layout and per-opening payloads.
/// - `alphas`: one batching coefficient per opening, equality openings first.
///
/// # Panics
///
/// - If the number of batching coefficients does not match the number of openings.
pub(crate) fn calculate_accumulators_batch<F: Field, EF: ExtensionField<F>>(
    claim: &ProverMultiClaim<F, EF>,
    alphas: &[EF],
) -> SvoAccumulators<EF> {
    // One batching coefficient is needed per opening in the claim.
    assert_eq!(claim.len(), alphas.len());
    // Number of SVO active variables, which is also the number of SVO sumcheck rounds.
    let k = claim.point().num_variables_svo();

    // Emit one accumulator pair per SVO round.
    (0..k)
        .map(|round_idx| {
            // Round l folds the first l active coordinates.
            let l = round_idx + 1;
            // Separate running payloads for equality openings and repeat-last-successor openings.
            let mut eq_acc = EqPartials::zero(l);
            let mut next_acc = NextPartials::zero(l);

            // Batch every equality opening's round-l payload with its coefficient.
            for (opening, &alpha) in claim.current_openings().iter().zip(alphas.iter()) {
                eq_acc.accumulate(&opening.data().rounds()[round_idx], alpha);
            }

            // Repeat-last-successor openings consume the coefficients after the equality ones.
            for (opening, &alpha) in claim
                .next_openings()
                .iter()
                .zip(alphas.iter().skip(claim.current_openings().len()))
            {
                next_acc.accumulate(&opening.data().rounds()[round_idx], alpha);
            }

            // Each output third spans 3^round_idx rows over the trailing active coordinates.
            let stride = 3usize.pow(round_idx as u32);
            // Accumulator for the round polynomial at the active coordinate fixed to 0.
            let mut acc0 = EF::zero_vec(stride);
            // Accumulator for the round polynomial at the active coordinate fixed to inf.
            let mut acc_inf = EF::zero_vec(stride);

            match claim.point().var_order() {
                // Active SVO variables are the low bits: take the leading l point coordinates.
                VariableOrder::Prefix => {
                    let (svo_active, _) = claim.point().z_svo().split_at(l);

                    // Fold equality openings only when any are present.
                    if !claim.current_openings().is_empty() {
                        eq_acc.accumulate_prefix(svo_active.as_slice(), &mut acc0, &mut acc_inf);
                    }

                    // Fold repeat-last-successor openings only when any are present.
                    if !claim.next_openings().is_empty() {
                        next_acc.accumulate_prefix(svo_active.as_slice(), &mut acc0, &mut acc_inf);
                    }
                }
                // Active SVO variables are the high bits: take the trailing l point coordinates.
                VariableOrder::Suffix => {
                    let (_, svo_active) = claim.point().z_svo().split_at(k - l);

                    // Fold equality openings only when any are present.
                    if !claim.current_openings().is_empty() {
                        eq_acc.accumulate_suffix(svo_active.as_slice(), &mut acc0, &mut acc_inf);
                    }

                    // Fold repeat-last-successor openings only when any are present.
                    if !claim.next_openings().is_empty() {
                        next_acc.accumulate_suffix(svo_active.as_slice(), &mut acc0, &mut acc_inf);
                    }
                }
            }

            // Pair of evaluations at 0 and inf for this round.
            [acc0, acc_inf]
        })
        .collect()
}

/// Computes the `0` and `inf` accumulators of a pointwise product of two multilinear tables.
///
/// # Overview
///
/// - Both inputs are evaluation tables over the same `2^l` Boolean hypercube.
/// - The output is the ternary grid of their product, restricted to the first and last thirds.
/// - The first third fixes the leading active coordinate to `0`, the last third to `inf`.
///
/// # Arguments
///
/// - `l`: the number of active coordinates shared by both tables.
/// - `left`: the first multilinear evaluation table.
/// - `right`: the second multilinear evaluation table.
///
/// # Panics
///
/// - If either table does not have length `2^l`.
///
/// # Performance
///
/// - For `l <= 3` straightline specializations avoid loop, buffer, and parallelization overhead.
/// - Larger `l` falls back to the general grid expansion at `O(3^l)` field operations.
pub(crate) fn calculate_product_accumulator<F: Field>(
    l: usize,
    left: &[F],
    right: &[F],
) -> [Vec<F>; 2] {
    // Each input must be a full multilinear table over l Boolean coordinates.
    assert_eq!(left.len(), 1 << l);
    assert_eq!(right.len(), 1 << l);

    // Dispatch to the fastest path available for this number of active coordinates.
    // The asserts above pin both lengths to `2^l`, so each array conversion is infallible.
    match l {
        // Single active coordinate: closed-form on two values.
        1 => calculate_accumulator_1(left.try_into().unwrap(), right.try_into().unwrap()),
        // Two active coordinates: closed-form on four values.
        2 => calculate_accumulator_2(left.try_into().unwrap(), right.try_into().unwrap()),
        // Three active coordinates: closed-form on eight values.
        3 => calculate_accumulator_3(left.try_into().unwrap(), right.try_into().unwrap()),
        // Otherwise expand both tables onto the ternary grid and multiply pointwise.
        _ => calculate_accumulator_general(l, left, right),
    }
}

/// Straightline accumulator computation for l=1.
///
/// ```text
///     eq0:     [e0, e1]          (2 values on {0,1})
///     reduced: [r0, r1]          (2 values on {0,1})
///
///     Grid on {0, 1, inf}:
///       f(0) = f[0],  f(1) = f[1],  f(inf) = f[1] - f[0]
///
///     acc0    = [eq0(0) * red(0)]       = [e0 * r0]             (1 value)
///     acc_inf = [eq0(inf) * red(inf)]   = [(e1-e0) * (r1-r0)]   (1 value)
/// ```
fn calculate_accumulator_1<EF: Field>(eq0: &[EF; 2], reduced: &[EF; 2]) -> [Vec<EF>; 2] {
    let &[e0, e1] = eq0;
    let &[r0, r1] = reduced;

    // Leading coefficients: f(inf) = f(1) - f(0).
    let e_inf = e1 - e0;
    let r_inf = r1 - r0;

    [vec![e0 * r0], vec![e_inf * r_inf]]
}

/// Straightline accumulator computation for l=2.
///
/// Grid layout after expansion (x_0 slowest, x_1 fastest):
///
/// ```text
///     grid[0..3] = x_0=0 group: f(0,0), f(0,1), f(0,inf)
///     grid[3..6] = x_0=1 group: f(1,0), f(1,1), f(1,inf)
///     grid[6..9] = x_0=inf group: f(inf,0), f(inf,1), f(inf,inf)
///
///     stride = 3^{l-1} = 3
///     acc0    = grid[0..3]  = x_0=0 slice, x_1 in {0, 1, inf}
///     acc_inf = grid[6..9]  = x_0=inf slice, x_1 in {0, 1, inf}
/// ```
fn calculate_accumulator_2<EF: Field>(eq0: &[EF; 4], reduced: &[EF; 4]) -> [Vec<EF>; 2] {
    // Boolean evaluations: index = x_0 + 2*x_1.
    let &[e00, e10, e01, e11] = eq0;
    let &[r00, r10, r01, r11] = reduced;

    // Extrapolate x_0 to get x_0=inf values.
    let e20 = e10 - e00;
    let e21 = e11 - e01;
    let r20 = r10 - r00;
    let r21 = r11 - r01;

    // Extrapolate x_1 to get x_1=inf values (only needed for x_0=0 and x_0=inf).
    let e02 = e01 - e00;
    let r02 = r01 - r00;

    // acc0: x_0=0 slice, indexed by x_1 in {0, 1, inf}.
    // acc_inf: x_0=inf slice, indexed by x_1 in {0, 1, inf}.
    [
        vec![e00 * r00, e01 * r01, e02 * r02],
        vec![e20 * r20, e21 * r21, (e21 - e20) * (r21 - r20)],
    ]
}

/// Straightline accumulator computation for l=3.
///
/// Grid layout after expansion (x_0 slowest, x_2 fastest):
///
/// ```text
///     stride = 3^{l-1} = 9
///     acc0 = grid[0..9]   = x_0=0 slice
///     acc_inf = grid[18..27] = x_0=inf slice
///
///     Within each x_0 group, 9 entries ordered by (x_1, x_2):
///       x_1=0:   f(x_0, 0, 0), f(x_0, 0, 1), f(x_0, 0, inf)
///       x_1=1:   f(x_0, 1, 0), f(x_0, 1, 1), f(x_0, 1, inf)
///       x_1=inf: f(x_0, inf, 0), f(x_0, inf, 1), f(x_0, inf, inf)
/// ```
fn calculate_accumulator_3<EF: Field>(eq0: &[EF; 8], reduced: &[EF; 8]) -> [Vec<EF>; 2] {
    // Boolean evaluations: index = x_0 + 2*x_1 + 4*x_2.
    // Name: e_ijk = eq0(x_0=i, x_1=j, x_2=k), similarly for r.
    let &[e_000, e_100, e_010, e_110, e_001, e_101, e_011, e_111] = eq0;
    let &[r_000, r_100, r_010, r_110, r_001, r_101, r_011, r_111] = reduced;

    // Extrapolate x_0: f(inf,j,k) = f(1,j,k) - f(0,j,k).
    let e_200 = e_100 - e_000;
    let e_210 = e_110 - e_010;
    let e_201 = e_101 - e_001;
    let e_211 = e_111 - e_011;
    let r_200 = r_100 - r_000;
    let r_210 = r_110 - r_010;
    let r_201 = r_101 - r_001;
    let r_211 = r_111 - r_011;

    // Extrapolate x_1: f(i,inf,k) = f(i,1,k) - f(i,0,k).
    // Only needed for x_0=0 and x_0=inf.
    let e_020 = e_010 - e_000;
    let e_220 = e_210 - e_200;
    let e_021 = e_011 - e_001;
    let e_221 = e_211 - e_201;
    let r_020 = r_010 - r_000;
    let r_220 = r_210 - r_200;
    let r_021 = r_011 - r_001;
    let r_221 = r_211 - r_201;

    // Extrapolate x_2: f(i,j,inf) = f(i,j,1) - f(i,j,0).
    // Only needed for x_0=0 and x_0=inf (the slices we read out).
    let e_002 = e_001 - e_000;
    let e_012 = e_011 - e_010;
    let e_022 = e_021 - e_020;
    let e_202 = e_201 - e_200;
    let e_212 = e_211 - e_210;
    let e_222 = e_221 - e_220;
    let r_002 = r_001 - r_000;
    let r_012 = r_011 - r_010;
    let r_022 = r_021 - r_020;
    let r_202 = r_201 - r_200;
    let r_212 = r_211 - r_210;
    let r_222 = r_221 - r_220;

    // acc0: x_0=0 slice, 9 entries ordered (x_1, x_2) with x_2 fastest.
    // acc_inf: x_0=inf slice, same layout.
    [
        vec![
            e_000 * r_000,
            e_001 * r_001,
            e_002 * r_002,
            e_010 * r_010,
            e_011 * r_011,
            e_012 * r_012,
            e_020 * r_020,
            e_021 * r_021,
            e_022 * r_022,
        ],
        vec![
            e_200 * r_200,
            e_201 * r_201,
            e_202 * r_202,
            e_210 * r_210,
            e_211 * r_211,
            e_212 * r_212,
            e_220 * r_220,
            e_221 * r_221,
            e_222 * r_222,
        ],
    ]
}

/// General accumulator computation for l >= 4.
///
/// Allocates `{0, 1, inf}^l` buffers and runs the staged grid expansion.
fn calculate_accumulator_general<F: Field, EF: ExtensionField<F>>(
    l: usize,
    eq0: &[EF],
    reduced_evals: &[EF],
) -> [Vec<EF>; 2] {
    let grid_len = 3usize.pow(l as u32);
    let mut eq0_grid = EF::zero_vec(grid_len);
    let mut reduced_grid = EF::zero_vec(grid_len);
    let mut scratch = EF::zero_vec(grid_len);

    evals_01inf_grid_into(eq0, &mut eq0_grid, &mut scratch);
    evals_01inf_grid_into(reduced_evals, &mut reduced_grid, &mut scratch);

    let stride = 3usize.pow((l - 1) as u32);
    // Both grids span the full ternary cube so the 0-face and inf-face slices are well defined.
    debug_assert_eq!(eq0_grid.len(), 3 * stride);
    debug_assert_eq!(reduced_grid.len(), 3 * stride);
    let acc0 = eq0_grid[..stride]
        .iter()
        .copied()
        .zip(reduced_grid[..stride].iter().copied())
        .map(|(eq, eval)| eq * eval)
        .collect();
    let acc_inf = eq0_grid[2 * stride..]
        .iter()
        .copied()
        .zip(reduced_grid[2 * stride..].iter().copied())
        .map(|(eq, eval)| eq * eval)
        .collect();

    [acc0, acc_inf]
}

#[cfg(test)]
mod test {
    use alloc::vec::Vec;

    use p3_field::extension::BinomialExtensionField;
    use p3_field::{ExtensionField, Field, PrimeCharacteristicRing, dot_product};
    use p3_koala_bear::KoalaBear;
    use p3_multilinear_util::point::Point;
    use p3_multilinear_util::poly::Poly;
    use p3_multilinear_util::split_eq::SplitEq;
    use p3_util::log2_strict_usize;
    use proptest::prelude::*;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;
    use crate::lagrange::lagrange_weights_01inf_multi;
    use crate::layout::Opening;
    use crate::strategy::VariableOrder;
    use crate::svo::SvoPoint;

    type F = KoalaBear;
    type EF = BinomialExtensionField<F, 4>;

    #[test]
    fn test_batch_svo_accumulators() {
        let k = 12;
        let n_polys = 3;
        let mut rng = SmallRng::seed_from_u64(0);
        let polys = (0..n_polys)
            .map(|_| Poly::<F>::rand(&mut rng, k))
            .collect::<Vec<_>>();
        let alphas = (0..polys.len())
            .map(|_| rng.random::<EF>())
            .collect::<Vec<_>>();
        let point = Point::<EF>::rand(&mut rng, k);

        for l0 in 0..=k / 2 {
            let svo_point = SvoPoint::<F, EF>::new_unpacked(l0, &point, VariableOrder::Suffix);
            let openings = polys
                .iter()
                .map(|poly| {
                    // Virtual opening: evaluate the column at the SVO point and
                    // carry the per-round partial evaluations as payload.
                    let (eval, partial_evals) = svo_point.eval(poly);
                    let opening = Opening {
                        poly_idx: None,
                        eval,
                        data: partial_evals,
                    };
                    assert_eq!(opening.eval(), poly.eval_base(&point));
                    opening
                })
                .collect::<Vec<_>>();
            let claim = ProverMultiClaim::new(svo_point, openings, Vec::new());

            let accumulators = calculate_accumulators_batch(&claim, &alphas);
            if l0 == 0 {
                assert!(accumulators.is_empty());
                continue;
            }

            let mut poly = Poly::<EF>::zero(l0);
            claim
                .current_openings()
                .iter()
                .zip(alphas.iter())
                .for_each(|(opening, &alpha)| {
                    let full_svo_poly = opening
                        .data()
                        .rounds()
                        .last()
                        .expect("l0 > 0 guarantees one SVO partial polynomial");
                    poly.as_mut_slice()
                        .iter_mut()
                        .zip(full_svo_poly.poly().iter())
                        .for_each(|(out, &value)| *out += alpha * value);
                });

            let mut eq = Poly::new_from_point(claim.point().z_svo().as_slice(), EF::ONE);
            let mut rs = Vec::with_capacity(l0);

            for [acc0, acc_inf] in accumulators.iter() {
                let weights = lagrange_weights_01inf_multi(rs.as_slice());
                let c0 = dot_product::<EF, _, _>(acc0.iter().copied(), weights.iter().copied());
                let cinf =
                    dot_product::<EF, _, _>(acc_inf.iter().copied(), weights.iter().copied());

                let (c0_ref, cinf_ref) =
                    VariableOrder::Suffix.sumcheck_coefficients(poly.as_slice(), eq.as_slice());

                assert_eq!(c0, c0_ref);
                assert_eq!(cinf, cinf_ref);

                let r: EF = rng.random();
                poly.fix_suffix_var_mut(r);
                eq.fix_suffix_var_mut(r);
                rs.push(r);
            }
        }

        for l0 in 0..=k / 2 {
            let svo_point = SvoPoint::<F, EF>::new_unpacked(l0, &point, VariableOrder::Prefix);
            let openings = polys
                .iter()
                .map(|poly| {
                    let (eval, partial_evals) = svo_point.eval(poly);
                    let opening = Opening {
                        poly_idx: None,
                        eval,
                        data: partial_evals,
                    };
                    assert_eq!(opening.eval(), poly.eval_base(&point));
                    opening
                })
                .collect::<Vec<_>>();
            let claim = ProverMultiClaim::new(svo_point, openings, Vec::new());

            let accumulators = calculate_accumulators_batch(&claim, &alphas);
            if l0 == 0 {
                assert!(accumulators.is_empty());
                continue;
            }

            let mut poly = Poly::<EF>::zero(l0);
            claim
                .current_openings()
                .iter()
                .zip(alphas.iter())
                .for_each(|(opening, &alpha)| {
                    let full_svo_poly = opening
                        .data()
                        .rounds()
                        .last()
                        .expect("l0 > 0 guarantees one SVO partial polynomial");
                    poly.as_mut_slice()
                        .iter_mut()
                        .zip(full_svo_poly.poly().iter())
                        .for_each(|(out, &value)| *out += alpha * value);
                });

            let mut eq = Poly::new_from_point(claim.point().z_svo().as_slice(), EF::ONE);
            let mut rs = Vec::with_capacity(l0);

            for [acc0, acc_inf] in accumulators.iter() {
                let weights = lagrange_weights_01inf_multi(rs.as_slice());
                let c0 = dot_product::<EF, _, _>(acc0.iter().copied(), weights.iter().copied());
                let cinf =
                    dot_product::<EF, _, _>(acc_inf.iter().copied(), weights.iter().copied());

                let (c0_ref, cinf_ref) =
                    VariableOrder::Prefix.sumcheck_coefficients(poly.as_slice(), eq.as_slice());

                assert_eq!(c0, c0_ref);
                assert_eq!(cinf, cinf_ref);

                let r: EF = rng.random();
                poly.fix_prefix_var_mut(r);
                eq.fix_prefix_var_mut(r);
                rs.push(r);
            }
        }
    }

    /// Computes the SVO accumulators using grid expansion.
    ///
    /// Rather than rebuilding every Lagrange basis vector independently (O(6^l)),
    /// this expands both the residual equality polynomial and the partially compressed
    /// multilinear polynomial over the entire `{0, 1, inf}^l` grid. The required
    /// accumulators are then simple pointwise products on the slices with
    /// final coordinate fixed to `0` or `inf`.
    ///
    /// For l <= 3, straightline specializations avoid all loop overhead, buffer
    /// allocation, and parallelization machinery.
    ///
    /// Total cost: O(3^l) field operations.
    fn calculate_accumulator<F: Field, EF: ExtensionField<F>>(
        l: usize,
        partial_evals: &[EF],
        point: &[EF],
    ) -> [Vec<EF>; 2] {
        let total_vars = log2_strict_usize(partial_evals.len());
        let offset = total_vars - l;

        // Split the challenge point into inner (z0) and outer (z1) components.
        // - z0 corresponds to the variables covered by the grid points.
        // - z1 corresponds to the remaining variables handled separately.
        let (z0, z1) = point.split_at(point.len() - offset);

        // Build equality polynomial evaluation tables for both components.
        // - eq0: evaluations of eq(z0, x) for x in {0,1}^{|z0|}
        // - eq1: evaluations of eq(z1, x) for x in {0,1}^{|z1|}
        let eq0 = Poly::new_from_point(z0, EF::ONE);
        let eq1 = Poly::new_from_point(z1, EF::ONE);

        // Reduce partial evaluations over the outer variables using eq1.
        //
        // This computes: sum_{x1} eq(z1, x1) * partial_evals[chunk_for_x1]
        let reduced_evals = partial_evals
            .chunks(eq1.num_evals())
            .map(|chunk| dot_product::<EF, _, _>(eq1.iter().copied(), chunk.iter().copied()))
            .collect::<Vec<_>>();

        // Dispatch through the production path so this reference exercises the same specializations.
        calculate_product_accumulator(l, eq0.as_slice(), &reduced_evals)
    }

    proptest! {
        /// Verify grid-expansion accumulators match the per-point Lagrange reference.
        #[test]
        fn prop_accumulators_specialization_matches_general(k in 10usize..=14) {
            let mut rng = SmallRng::seed_from_u64(k as u64);
            let poly = Poly::new((0..1 << k).map(|_| rng.random()).collect());
            let point = Point::<EF>::rand(&mut rng, k);

            // Split the point at half to get partial evaluations.
            let (z_svo, z_rest) = point.split_at(k / 2);
            let split_eq = SplitEq::<F, EF>::new_packed(&z_rest, EF::ONE);
            let partial_evals = split_eq.compress_suffix(&poly);

            // Compare the dispatch path (which uses straightline specializations
            // for l <= 3) against the general grid-expansion path.
            for l in 1..k / 2 {
                let dispatched =
                    calculate_accumulator::<F, EF>(l, partial_evals.as_slice(), z_svo.as_slice());

                let eq0 = Poly::new_from_point(&z_svo.as_slice()[..l], EF::ONE);
                let eq1 = Poly::new_from_point(&z_svo.as_slice()[l..], EF::ONE);
                let reduced: Vec<EF> = partial_evals
                    .as_slice()
                    .chunks(eq1.num_evals())
                    .map(|chunk| dot_product::<EF, _, _>(eq1.iter().copied(), chunk.iter().copied()))
                    .collect();
                let general = calculate_accumulator_general::<F, EF>(l, eq0.as_slice(), &reduced);

                prop_assert_eq!(dispatched, general);
            }
        }

        /// Verify l=1 straightline matches the general path.
        #[test]
        fn prop_accumulators_1_matches_general(seed in 0u64..1000) {
            let mut rng = SmallRng::seed_from_u64(seed);
            let eq0: [EF; 2] = rng.random();
            let reduced: [EF; 2] = rng.random();

            let fast = calculate_accumulator_1(&eq0, &reduced);
            let general = calculate_accumulator_general::<F, EF>(1, &eq0, &reduced);

            prop_assert_eq!(fast, general);
        }

        /// Verify l=2 straightline matches the general path.
        #[test]
        fn prop_accumulators_2_matches_general(seed in 0u64..1000) {
            let mut rng = SmallRng::seed_from_u64(seed);
            let eq0: [EF; 4] = rng.random();
            let reduced: [EF; 4] = rng.random();

            let fast = calculate_accumulator_2(&eq0, &reduced);
            let general = calculate_accumulator_general::<F, EF>(2, &eq0, &reduced);

            prop_assert_eq!(fast, general);
        }

        /// Verify l=3 straightline matches the general path.
        #[test]
        fn prop_accumulators_3_matches_general(seed in 0u64..1000) {
            let mut rng = SmallRng::seed_from_u64(seed);
            let eq0: [EF; 8] = rng.random();
            let reduced: [EF; 8] = rng.random();

            let fast = calculate_accumulator_3(&eq0, &reduced);
            let general = calculate_accumulator_general::<F, EF>(3, &eq0, &reduced);

            prop_assert_eq!(fast, general);
        }
    }
}
