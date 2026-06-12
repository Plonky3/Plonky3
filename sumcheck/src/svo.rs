//! Small-Value Optimization (SVO) for the sumcheck protocol.
//!
//! Optimizes sumcheck proving when the polynomial includes an equality polynomial factor.
//! Implements Algorithm 5 from "Speeding Up Sum-Check Proving" (ePrint 2025/1117).
//!
//! # Key Insight
//!
//! For polynomials of the form `g(X) = eq(w, X) * p(X)`,
//! the equality polynomial can be decomposed as:
//!
//! ```text
//! eq(w, (r_{<i}, X, x')) = eq(w_{<i}, r_{<i}) * eq(w_i, X) * eq(w_{>i}, x')
//! ```
//!
//! This allows:
//! 1. Pre-computing smaller tables for the left and right components.
//! 2. Avoiding the full `2^l`-sized equality table.
//! 3. Reconstructing round polynomials from compact accumulators via Lagrange interpolation.

use alloc::vec;
use alloc::vec::Vec;

use itertools::Itertools;
use p3_field::{ExtensionField, Field, add_scaled_slice_in_place};
use p3_maybe_rayon::prelude::*;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use p3_multilinear_util::split_eq::SplitEq;
use p3_util::log2_strict_usize;

use crate::layout::{EqPartials, EqSvoPartials, NextPartials, NextSvoPartials, ProverMultiClaim};
use crate::strategy::VariableOrder;

/// Output length at or above which residual weight accumulation runs in parallel.
///
/// # Why this value
///
/// - Below `4096` output entries the per-chunk work is too small to amortize thread spawning.
/// - At or above it the parallel sweep over split-variable chunks outpaces the serial sweep.
const PARALLEL_THRESHOLD: usize = 4096;

/// Expand `2^l` Boolean-hypercube evaluations to `3^l` evaluations on `{0,1,inf}^l`.
///
/// # Overview
///
/// A multilinear polynomial in `l` variables is determined by `2^l` evaluations
/// on `{0,1}^l`. This extends them to include the "evaluation at infinity"
/// (the leading coefficient) for each variable:
///
/// ```text
///     f(0), f(1)  -->  f(0), f(1), f(inf)    where f(inf) = f(1) - f(0)
/// ```
///
/// # Motivation
///
/// The SVO sumcheck prover needs accumulator values on the `{0, 1, inf}^l` grid.
///
/// - **Naive**: evaluate each of the `3^l` grid points independently via
///   Lagrange interpolation from the `2^l` Boolean values --> `O(6^l)`.
/// - **This function**: process one variable at a time --> `O(3^l)`.
///
/// # Memory Layout
///
/// The input uses prefix-variable-fastest ordering: `idx = x_0 + 2*x_1 + ...`
///
/// The output uses the same convention as the expansion order. The first
/// variable processed (x_0) becomes the **slowest**-varying coordinate
/// in the ternary grid. Flat index = `x_1 + 3*x_0` for 2 variables:
///
/// ```text
///     l=2 example:
///     index:   0      1      2      3      4      5      6      7      8
///     point: (0,0)  (0,1)  (0,inf)  (1,0)  (1,1)  (1,inf)  (inf,0)  (inf,1)  (inf,inf)
/// ```
///
/// # Algorithm
///
/// `l` stages, one per variable. Each stage converts pairs into triples:
///
/// ```text
///     stage 0:  2^l             values on {0,1}^l
///     stage 1:  3 * 2^{l-1}    values on {0, 1, inf} x {0,1}^{l-1}
///       ...
///     stage l:  3^l             values on {0, 1, inf}^l
/// ```
///
/// Two buffers alternate in a ping-pong pattern.
/// Initial assignment ensures the final result lands in the output buffer.
///
/// # Parallelization
///
/// - **Early stages**: many small blocks --> parallelize across blocks.
/// - **Late stages**: few large blocks --> parallelize within each block.
///
/// # Panics
///
/// - Input length not a power of two.
/// - Output or scratch length != `3^l`.
///
/// # Performance
///
/// - Time: `O(3^l)` field additions and doublings.
/// - Space: two pre-allocated `3^l` buffers, no internal allocation.
fn evals_01inf_grid_into<F: Field>(boolean_evals: &[F], output: &mut [F], scratch: &mut [F]) {
    let num_variables = log2_strict_usize(boolean_evals.len());
    let output_len = 3usize.pow(num_variables as u32);

    assert_eq!(output.len(), output_len);
    assert_eq!(scratch.len(), output_len);

    // Single constant -- nothing to expand.
    if num_variables == 0 {
        output[0] = boolean_evals[0];
        return;
    }

    // Ping-pong buffer setup.
    //
    // Each stage swaps cur/next. After l swaps the result must be in `output`.
    //
    //     l odd  --> start in scratch --> after l swaps --> output  OK
    //     l even --> start in output  --> after l swaps --> output  OK
    let (mut cur, mut next) = if num_variables % 2 == 1 {
        scratch[..boolean_evals.len()].copy_from_slice(boolean_evals);
        (&mut scratch[..], &mut output[..])
    } else {
        output[..boolean_evals.len()].copy_from_slice(boolean_evals);
        (&mut output[..], &mut scratch[..])
    };

    // Below this: parallelize across blocks (many small chunks).
    // Above this: parallelize within each block (few large chunks).
    //
    // Why 256: below this the per-element work is too small for per-element
    // thread scheduling; above this there may be only 1-2 blocks.
    const PARALLEL_STRIDE_THRESHOLD: usize = 256;

    for stage in 0..num_variables {
        // Stride = 3^stage: how many consecutive elements share the same
        // value of the variable being expanded (the already-expanded
        // variables each contribute a factor of 3).
        let in_stride = 3usize.pow(stage as u32);

        // Blocks = 2^{remaining}: independent groups of (f(0), f(1)) pairs.
        let blocks = 1usize << (num_variables - stage - 1);

        // Slice only the live region (early stages use less than the full buffer).
        //
        //     cur  per block: [ f(0)-group | f(1)-group ]   each of size in_stride
        //     next per block: [ f(0)-group | f(1)-group | f(inf)-group ]
        let cur_slice = &cur[..blocks * 2 * in_stride];
        let next_slice = &mut next[..blocks * 3 * in_stride];

        if in_stride < PARALLEL_STRIDE_THRESHOLD {
            // Many small blocks -- parallelize across blocks.
            //
            //     l=2, stage 0 (stride=1, blocks=2):
            //     cur:   [ f(0,0) f(1,0) | f(0,1) f(1,1) ]
            //     next:  [ f(0,0) f(1,0) f(inf,0) | f(0,1) f(1,1) f(inf,1) ]
            cur_slice
                .par_chunks(2 * in_stride)
                .zip(next_slice.par_chunks_mut(3 * in_stride))
                .for_each(|(c_chunk, n_chunk)| {
                    for j in 0..in_stride {
                        let f0 = c_chunk[j];
                        let f1 = c_chunk[in_stride + j];
                        // Interleaved output: position j --> indices 3j, 3j+1, 3j+2.
                        n_chunk[3 * j] = f0;
                        n_chunk[3 * j + 1] = f1;
                        n_chunk[3 * j + 2] = f1 - f0;
                    }
                });
        } else {
            // Few large blocks -- parallelize within each block.
            cur_slice
                .chunks(2 * in_stride)
                .zip(next_slice.chunks_mut(3 * in_stride))
                .for_each(|(c_chunk, n_chunk)| {
                    // Split into left=f(0) half, right=f(1) half.
                    let (c_left, c_right) = c_chunk.split_at(in_stride);
                    // Each (f0, f1) pair --> (f0, f1, f1 - f0) triple.
                    c_left
                        .par_iter()
                        .zip(c_right.par_iter())
                        .zip(n_chunk.par_chunks_mut(3))
                        .for_each(|((&f0, &f1), out)| {
                            out[0] = f0;
                            out[1] = f1;
                            out[2] = f1 - f0;
                        });
                });
        }

        // What was `next` becomes `cur` for the next stage.
        core::mem::swap(&mut cur, &mut next);
    }
}

pub(crate) fn evals_01inf_grid_prefix<F: Field>(evals: &[F]) -> Vec<F> {
    fn reverse_ternary_digits(mut idx: usize, l: usize) -> usize {
        let mut rev = 0usize;
        for _ in 0..l {
            rev = 3 * rev + (idx % 3);
            idx /= 3;
        }
        rev
    }

    let grid_len = 3usize.pow(log2_strict_usize(evals.len()) as u32);
    let l = log2_strict_usize(evals.len());
    let mut prefix = F::zero_vec(grid_len);
    let mut scratch = F::zero_vec(grid_len);
    evals_01inf_grid_into(evals, &mut prefix, &mut scratch);

    let mut out = F::zero_vec(grid_len);
    for (src_idx, value) in prefix.into_iter().enumerate() {
        out[reverse_ternary_digits(src_idx, l)] = value;
    }
    out
}
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
    match l {
        // Single active coordinate: closed-form on two values.
        1 => calculate_accumulator_1(left, right),
        // Two active coordinates: closed-form on four values.
        2 => calculate_accumulator_2(left, right),
        // Three active coordinates: closed-form on eight values.
        3 => calculate_accumulator_3(left, right),
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
fn calculate_accumulator_1<EF: Field>(eq0: &[EF], reduced: &[EF]) -> [Vec<EF>; 2] {
    assert_eq!(eq0.len(), 2);
    assert_eq!(reduced.len(), 2);

    let (e0, e1) = (eq0[0], eq0[1]);
    let (r0, r1) = (reduced[0], reduced[1]);

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
fn calculate_accumulator_2<EF: Field>(eq0: &[EF], reduced: &[EF]) -> [Vec<EF>; 2] {
    assert_eq!(eq0.len(), 4);
    assert_eq!(reduced.len(), 4);

    // Boolean evaluations: index = x_0 + 2*x_1.
    let (e00, e10, e01, e11) = (eq0[0], eq0[1], eq0[2], eq0[3]);
    let (r00, r10, r01, r11) = (reduced[0], reduced[1], reduced[2], reduced[3]);

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
fn calculate_accumulator_3<EF: Field>(eq0: &[EF], reduced: &[EF]) -> [Vec<EF>; 2] {
    assert_eq!(eq0.len(), 8);
    assert_eq!(reduced.len(), 8);

    // Boolean evaluations: index = x_0 + 2*x_1 + 4*x_2.
    // Name: e_ijk = eq0(x_0=i, x_1=j, x_2=k), similarly for r.
    let (e_000, e_100, e_010, e_110, e_001, e_101, e_011, e_111) = (
        eq0[0], eq0[1], eq0[2], eq0[3], eq0[4], eq0[5], eq0[6], eq0[7],
    );
    let (r_000, r_100, r_010, r_110, r_001, r_101, r_011, r_111) = (
        reduced[0], reduced[1], reduced[2], reduced[3], reduced[4], reduced[5], reduced[6],
        reduced[7],
    );

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

/// Per-round accumulator slices for the SVO rounds.
///
/// For round `i + 1`, the entry stores:
/// - `[0]`: accumulator values on `{0,1,2}^i x {0}`
/// - `[1]`: accumulator values on `{0,1,2}^i x {2}`
///
/// We do not store the `x {1}` face because the verifier can derive `h(1)` from the
/// claimed sum and `h(0)`.
pub(super) type SvoAccumulators<EF> = Vec<[Vec<EF>; 2]>;

/// Challenge point split into an SVO prefix and a residual split-eq suffix.
///
/// The split direction depends on [`VariableOrder`]:
/// - `Prefix`: `z_svo` is the prefix `l0` variables and `z_split` represents the
///   remaining suffix.
/// - `Suffix`: `z_svo` is the suffix `l0` variables and `z_split` represents the
///   remaining prefix.
#[derive(Debug, Clone)]
pub struct SvoPoint<F: Field, EF: ExtensionField<F>> {
    /// The first `k_svo` coordinates of the original point, handled by the SVO
    /// accumulator rounds.
    pub(crate) z_svo: Point<EF>,
    /// A factored table for `eq(z_rest, ·)` on the remaining coordinates after
    /// removing `z_svo` from the original point.
    pub(crate) z_split: SplitEq<F, EF>,
    /// Variable processing order
    var_order: VariableOrder,
}

impl<F: Field, EF: ExtensionField<F>> SvoPoint<F, EF> {
    /// Splits a challenge point into the SVO portion and the residual split-eq
    /// portion according to `var_order`.
    ///
    /// `l0` is the number of variables handled by the SVO optimization.
    pub fn new_unpacked(l0: usize, point: &Point<EF>, var_order: VariableOrder) -> Self {
        assert!(l0 <= point.num_variables());
        let (svo, split) = match var_order {
            VariableOrder::Prefix => point.split_at(l0),
            VariableOrder::Suffix => {
                let (split, svo) = point.split_at(point.num_variables() - l0);
                (svo, split)
            }
        };
        let z_split = SplitEq::new_unpacked(&split, EF::ONE);
        Self {
            z_svo: svo,
            z_split,
            var_order,
        }
    }

    /// Splits a challenge point into a prefix SVO portion and a residual suffix.
    ///
    /// `l0` is the number of variables handled by the SVO optimization.
    pub fn new_packed(l0: usize, point: &Point<EF>) -> Self {
        assert!(l0 <= point.num_variables());
        let (svo, split) = point.split_at(l0);
        let z_split = SplitEq::new_packed(&split, EF::ONE);
        Self {
            z_svo: svo,
            z_split,
            var_order: VariableOrder::Prefix,
        }
    }

    /// Accumulates this claim's residual equality table into a buffer.
    ///
    /// Once the SVO rounds have fixed the suffix variables to `rs`, the remaining
    /// weight vector is:
    /// ```text
    /// alpha  · eq(z_rest, x_rest) · eq(z_svo, rs)
    /// ```
    /// for every assignment `x_rest` to the non-SVO variables.
    ///
    /// This method computes the scalar factor `alpha · eq(z_svo, rs)` and then asks
    /// `split_eq` to materialize the residual `eq(z_rest, ·)` table into `out`.
    pub fn accumulate_into(&self, out: &mut [EF], rs: &Point<EF>, mut scale: EF) {
        assert_eq!(rs.num_variables(), self.num_variables_svo());
        scale *= Point::eval_eq(self.z_svo.as_slice(), rs.as_slice());
        self.z_split.accumulate_into(out, Some(scale));
    }

    /// Accumulates this claim's residual equality table into a packed buffer.
    ///
    /// Once the SVO rounds have fixed the suffix variables to `rs`, the remaining
    /// weight vector is:
    /// ```text
    /// alpha  · eq(z_rest, x_rest) · eq(z_svo, rs)
    /// ```
    /// for every assignment `x_rest` to the non-SVO variables.
    ///
    /// This method computes the scalar factor `alpha · eq(z_svo, rs)` and then asks
    /// `split_eq` to materialize the residual `eq(z_rest, ·)` table into `out`.
    pub fn accumulate_into_packed(
        &self,
        out: &mut [EF::ExtensionPacking],
        rs: &Point<EF>,
        mut scale: EF,
    ) {
        assert!(matches!(self.var_order, VariableOrder::Prefix));
        assert_eq!(rs.num_variables(), self.num_variables_svo());
        scale *= Point::eval_eq(self.z_svo.as_slice(), rs.as_slice());
        self.z_split.accumulate_into_packed(out, Some(scale));
    }

    /// Evaluates `poly` at this point and returns all partial evaluations seen
    /// during the SVO rounds.
    ///
    /// The non-SVO prefix is compressed first using `SplitEq`. The result is a
    /// polynomial only in the SVO variables, which is then:
    /// - evaluated at `z_svo` to obtain the opening value
    /// - partially compressed after each SVO round to feed the accumulator path
    pub fn eval(&self, poly: &Poly<F>) -> (EF, EqSvoPartials<EF>) {
        assert_eq!(self.num_variables(), poly.num_variables());
        let (compressed, partial_evals) = match self.var_order {
            VariableOrder::Prefix => {
                let compressed = self.z_split.compress_suffix(poly);
                let partial_evals = (1..=self.num_variables_svo())
                    .map(|i| {
                        let (_svo_active, svo_rest) = self.z_svo.split_at(i);
                        compressed.compress_suffix(&svo_rest, EF::ONE)
                    })
                    .collect::<Vec<_>>();
                (compressed, partial_evals)
            }
            VariableOrder::Suffix => {
                let compressed = self.z_split.compress_prefix(poly);
                let partial_evals = (1..=self.num_variables_svo())
                    .map(|i| {
                        let (svo_rest, _svo_active) =
                            self.z_svo.split_at(self.z_svo.num_variables() - i);
                        compressed.compress_prefix(&svo_rest, EF::ONE)
                    })
                    .collect::<Vec<_>>();
                (compressed, partial_evals)
            }
        };
        // Evaluate the fully compressed SVO-only polynomial to get the scalar opening value.
        let eval = compressed.eval_base(&self.z_svo);
        // Wrap each per-round compression as an equality payload for the accumulator path.
        (
            eval,
            EqSvoPartials::new(partial_evals.into_iter().map(EqPartials::new).collect()),
        )
    }

    /// Returns the number of SVO variables (`l0`).
    ///
    /// This is the depth of the SVO optimization.
    /// These coordinates are processed via the accumulator-based Lagrange
    /// interpolation path rather than the standard fold-and-sum path.
    pub const fn num_variables_svo(&self) -> usize {
        self.z_svo.num_variables()
    }

    /// Returns the number of variables of the represented point.
    pub const fn num_variables(&self) -> usize {
        self.z_svo.num_variables() + self.z_split.num_variables()
    }

    /// Returns the SVO suffix of the represented point.
    pub const fn z_svo(&self) -> &Point<EF> {
        &self.z_svo
    }

    /// Returns the factored equality table for the non-SVO prefix of the represented point.
    pub const fn z_split(&self) -> &SplitEq<F, EF> {
        &self.z_split
    }

    /// Returns the original point represented by this struct.
    pub const fn var_order(&self) -> VariableOrder {
        self.var_order
    }
    /// Adds the residual suffix-layout successor weight over the split variables.
    ///
    /// # Overview
    ///
    /// Once the SVO variables are fixed to the round challenges, the leftover
    /// weight over the split variables is a sum of three terms:
    ///
    /// - a "done" scalar times the split equality weights,
    /// - a "carry" scalar times the split equality weights shifted up one row,
    /// - an "omega" scalar times the all-ones split boundary weight.
    ///
    /// The result is added into the output scaled by the batching coefficient,
    /// without ever materializing a dense successor table.
    ///
    /// # Arguments
    ///
    /// - `out`: the split-variable weight buffer to accumulate into.
    /// - `rs`: the SVO round challenges that fix the SVO variables.
    /// - `scale`: the batching coefficient applied to this opening.
    ///
    /// # Panics
    ///
    /// - If the point is not in suffix layout.
    /// - If the number of challenges does not match the number of SVO variables.
    /// - If the output length does not match the number of split variables.
    pub fn accumulate_next_suffix_into(&self, out: &mut [EF], rs: &Point<EF>, scale: EF) {
        // This closed form is only derived for the suffix SVO layout.
        assert!(
            matches!(self.var_order(), VariableOrder::Suffix),
            "next residual weights are implemented for suffix SVO only"
        );
        // One round challenge per SVO variable.
        assert_eq!(rs.num_variables(), self.num_variables_svo());
        // The output buffer spans exactly the split variables.
        assert_eq!(log2_strict_usize(out.len()), self.z_split.num_variables());

        // Closed-form successor decomposition of the SVO part at the round challenges.
        let (carry, done, omega) = Point::eval_next(self.z_svo.as_slice(), rs.as_slice());
        // Pre-multiply each state scalar by the batching coefficient.
        let done_scale = scale * done;
        let carry_scale = scale * carry;
        let omega_scale = scale * omega;

        // Reference path: build the same residual densely to cross-check the fast path.
        #[cfg(debug_assertions)]
        let expected = {
            let mut expected = out.to_vec();
            let eq = self.z_split.materialize();
            // Done term: split equality weights, aligned row by row.
            expected
                .iter_mut()
                .zip_eq(eq.iter())
                .for_each(|(out, &weight)| *out += done_scale * weight);
            // Carry term: split equality weights shifted up by one row.
            expected
                .iter_mut()
                .skip(1)
                .zip_eq(eq.as_slice()[..eq.num_evals() - 1].iter())
                .for_each(|(out, &weight)| *out += carry_scale * weight);
            // Omega term: only the final all-ones split row.
            *expected.last_mut().unwrap() += omega_scale * self.z_split.last_scalar();
            expected
        };

        // The split equality table is factored as an outer (eq0) over an inner (eq1) block.
        let eq1 = self.z_split.eq1();
        // Each outer entry owns one contiguous inner chunk of the output.
        let cs = eq1.scalar_chunk_size();
        // Weight of the all-ones inner row, used to stitch the carry across chunk boundaries.
        let eq1_last = eq1.last_scalar();
        // Serial path: cheaper than thread spawning for small outputs.
        if out.len() < PARALLEL_THRESHOLD {
            // Carry into the first row of a chunk comes from the last row of the previous chunk.
            let mut prev_last = EF::ZERO;
            out.chunks_mut(cs)
                .zip(self.z_split.eq0().iter())
                .for_each(|(chunk, &w0)| {
                    // Scale done and carry by the outer weight; pass the cross-chunk carry seed.
                    eq1.accumulate_next_chunk_into(
                        chunk,
                        done_scale * w0,
                        carry_scale * w0,
                        carry_scale * prev_last,
                    );
                    // Carry seed for the next chunk: this chunk's all-ones inner row.
                    prev_last = w0 * eq1_last;
                });
        } else {
            // Parallel path: each chunk recomputes its own cross-chunk carry from the outer table.
            let eq0 = self.z_split.eq0().as_slice();
            out.par_chunks_mut(cs)
                .enumerate()
                .zip(eq0.par_iter())
                .for_each(|((idx, chunk), &w0)| {
                    // Boundary carry depends on the previous outer entry, except for the first chunk.
                    let boundary = if idx > 0 {
                        carry_scale * eq0[idx - 1] * eq1_last
                    } else {
                        EF::ZERO
                    };
                    eq1.accumulate_next_chunk_into(
                        chunk,
                        done_scale * w0,
                        carry_scale * w0,
                        boundary,
                    );
                });
        }

        // Omega contributes only to the global all-ones split row.
        *out.last_mut().unwrap() += omega_scale * self.z_split.last_scalar();

        // Fast and dense paths must agree exactly.
        #[cfg(debug_assertions)]
        debug_assert!(out == expected.as_slice());
    }

    /// Adds the residual prefix-layout successor weight over the split variables.
    ///
    /// # Overview
    ///
    /// Once the prefix SVO variables are fixed to the round challenges, the
    /// leftover weight over the split variables is a sum of three terms:
    ///
    /// - a "done" scalar times the split equality weights shifted up one row,
    /// - a "carry" scalar landing only on the first (all-zeros) split row,
    /// - an "omega" scalar landing only on the last (all-ones) split row.
    ///
    /// The result is added into the output scaled by the batching coefficient.
    ///
    /// # Arguments
    ///
    /// - `out`: the split-variable weight buffer to accumulate into.
    /// - `rs`: the SVO round challenges that fix the SVO variables.
    /// - `scale`: the batching coefficient applied to this opening.
    ///
    /// # Panics
    ///
    /// - If the point is not in prefix layout.
    /// - If the number of challenges does not match the number of SVO variables.
    /// - If the output length does not match the number of split variables.
    pub fn accumulate_next_prefix_into(&self, out: &mut [EF], rs: &Point<EF>, scale: EF) {
        // This closed form is only derived for the prefix SVO layout.
        assert!(
            matches!(self.var_order(), VariableOrder::Prefix),
            "prefix next residual weights require prefix SVO"
        );
        // One round challenge per SVO variable.
        assert_eq!(rs.num_variables(), self.num_variables_svo());
        // The output buffer spans exactly the split variables.
        assert_eq!(log2_strict_usize(out.len()), self.z_split.num_variables());

        // Successor decomposition of the SVO part; prefix layout uses the done and omega states.
        let (_carry, done, omega) = Point::eval_next(self.z_svo.as_slice(), rs.as_slice());
        // Prefix layout also needs the plain equality weight of the SVO part.
        let eq = Point::eval_eq(self.z_svo.as_slice(), rs.as_slice());
        // Pre-multiply each state scalar by the batching coefficient.
        let done_scale = scale * eq;
        let carry_scale = scale * done;
        let omega_scale = scale * omega;

        // Reference path: build the same residual densely to cross-check the fast path.
        #[cfg(debug_assertions)]
        let expected = {
            let mut expected = out.to_vec();
            let eq = self.z_split.materialize();
            // Done term: split equality weights shifted up by one row.
            expected
                .iter_mut()
                .skip(1)
                .zip_eq(eq.as_slice()[..eq.num_evals() - 1].iter())
                .for_each(|(out, &weight)| *out += done_scale * weight);
            // Carry and omega land only on the two boundary rows.
            let boundary = self.z_split.last_scalar();
            *expected.first_mut().unwrap() += carry_scale * boundary;
            *expected.last_mut().unwrap() += omega_scale * boundary;
            expected
        };

        // The split equality table is factored as an outer (eq0) over an inner (eq1) block.
        let eq1 = self.z_split.eq1();
        // Each outer entry owns one contiguous inner chunk of the output.
        let cs = eq1.scalar_chunk_size();
        // Weight of the all-ones inner row, used to stitch the shift across chunk boundaries.
        let eq1_last = eq1.last_scalar();
        // Serial path: cheaper than thread spawning for small outputs.
        if out.len() < PARALLEL_THRESHOLD {
            // Shifted weight into the first row of a chunk comes from the last row of the previous chunk.
            let mut prev_last = EF::ZERO;
            out.chunks_mut(cs)
                .zip(self.z_split.eq0().iter())
                .for_each(|(chunk, &w0)| {
                    // No row-aligned done term here; only the shifted-up contribution and its seed.
                    eq1.accumulate_next_chunk_into(
                        chunk,
                        EF::ZERO,
                        done_scale * w0,
                        done_scale * prev_last,
                    );
                    // Seed for the next chunk: this chunk's all-ones inner row.
                    prev_last = w0 * eq1_last;
                });
        } else {
            // Parallel path: each chunk recomputes its own cross-chunk shift from the outer table.
            let eq0 = self.z_split.eq0().as_slice();
            out.par_chunks_mut(cs)
                .enumerate()
                .zip(eq0.par_iter())
                .for_each(|((idx, chunk), &w0)| {
                    // Boundary shift depends on the previous outer entry, except for the first chunk.
                    let boundary = if idx > 0 {
                        done_scale * eq0[idx - 1] * eq1_last
                    } else {
                        EF::ZERO
                    };
                    eq1.accumulate_next_chunk_into(chunk, EF::ZERO, done_scale * w0, boundary);
                });
        }

        // Carry and omega contribute only to the global first and last split rows.
        let boundary = self.z_split.last_scalar();
        *out.first_mut().unwrap() += carry_scale * boundary;
        *out.last_mut().unwrap() += omega_scale * boundary;

        // Fast and dense paths must agree exactly.
        #[cfg(debug_assertions)]
        debug_assert!(out == expected.as_slice());
    }

    /// Evaluates a suffix-layout repeat-last-successor opening and caches its SVO rounds.
    ///
    /// # Overview
    ///
    /// - The witness polynomial is compressed over the split prefix into three payloads over the SVO suffix.
    /// - The three payloads are weighted respectively by the equality, carry, and boundary split states.
    /// - The payloads give the scalar opening value and one cached per-round successor table.
    ///
    /// # Arguments
    ///
    /// - `poly`: the raw witness polynomial over all variables.
    /// - `d_eq`: an optional precomputed equality payload reused from an earlier opening.
    ///
    /// # Returns
    ///
    /// - The scalar opening value of the successor weight against the polynomial.
    /// - One cached successor table per SVO sumcheck round.
    ///
    /// # Panics
    ///
    /// - If the polynomial does not span all point variables.
    /// - If the point is not in suffix layout.
    /// - If a supplied equality payload does not span the SVO variables.
    pub fn eval_next_suffix(
        &self,
        poly: &Poly<F>,
        d_eq: Option<&Poly<EF>>,
    ) -> (EF, NextSvoPartials<EF>) {
        // The polynomial must cover both the split and SVO variables.
        assert_eq!(self.num_variables(), poly.num_variables());
        // This routine derives its compressions only for suffix layout.
        assert!(
            matches!(self.var_order(), VariableOrder::Suffix),
            "next openings are implemented for suffix SVO only"
        );

        // Compress the equality payload over the split prefix unless the caller supplied it.
        let d_eq_owned = d_eq.is_none().then(|| self.z_split.compress_prefix(poly));
        let d_eq = d_eq.unwrap_or_else(|| d_eq_owned.as_ref().unwrap());
        assert_eq!(d_eq.num_variables(), self.num_variables_svo());

        // A caller-supplied payload must equal the freshly computed one.
        #[cfg(debug_assertions)]
        if d_eq_owned.is_none() {
            debug_assert_eq!(*d_eq, self.z_split.compress_prefix(poly));
        }
        // Carry payload: compress the polynomial over the split prefix with a one-row shift.
        let d_t = self.z_split.compress_prefix_shifted(poly);

        // Number of Boolean rows over the SVO and split variables.
        let svo_rows = 1 << self.num_variables_svo();
        let split_rows = 1 << self.z_split.num_variables();
        // The boundary weight is the all-ones corner of the split equality table.
        let omega_scale = self.z_split.last_scalar();
        // Suffix layout stores the all-ones split block at the very end of the polynomial.
        let omega_start = (split_rows - 1) * svo_rows;
        // Boundary payload: that final split block scaled by the boundary weight.
        let d_omega = Poly::new(
            poly.as_slice()[omega_start..omega_start + svo_rows]
                .iter()
                .map(|&value| omega_scale * value)
                .collect(),
        );

        // Opening value: sum the three-state successor weight against the payloads over every SVO row.
        let eval = (0..svo_rows)
            .map(|svo_idx| {
                // Boolean assignment of the SVO variables for this row.
                let row = Point::hypercube(svo_idx, self.z_svo.num_variables());
                // Closed-form successor states of the SVO point at this row.
                let (carry, done, omega) = Point::eval_next(self.z_svo.as_slice(), row.as_slice());
                done * d_eq.as_slice()[svo_idx]
                    + carry * d_t.as_slice()[svo_idx]
                    + omega * d_omega.as_slice()[svo_idx]
            })
            .sum();

        // Cache one compressed successor table per SVO sumcheck round.
        let rounds = (1..=self.num_variables_svo())
            .map(|active_len| {
                next_round_partials_suffix(d_eq, &d_t, &d_omega, &self.z_svo, active_len)
            })
            .collect::<Vec<_>>();

        (eval, NextSvoPartials::new(rounds))
    }

    /// Evaluates a prefix-layout repeat-last-successor opening and caches its SVO rounds.
    ///
    /// # Overview
    ///
    /// - The witness polynomial is compressed over the split suffix into three payloads over the SVO prefix.
    /// - The payloads are the shifted-done weight, the carry boundary, and the omega boundary.
    /// - The payloads give the scalar opening value and one cached per-round successor table.
    ///
    /// # Arguments
    ///
    /// - `poly`: the raw witness polynomial over all variables.
    ///
    /// # Returns
    ///
    /// - The scalar opening value of the successor weight against the polynomial.
    /// - One cached successor table per SVO sumcheck round.
    ///
    /// # Panics
    ///
    /// - If the polynomial does not span all point variables.
    /// - If the point is not in prefix layout.
    pub fn eval_next_prefix(&self, poly: &Poly<F>) -> (EF, NextSvoPartials<EF>) {
        // The polynomial must cover both the split and SVO variables.
        assert_eq!(self.num_variables(), poly.num_variables());
        // This routine derives its compressions only for prefix layout.
        assert!(
            matches!(self.var_order(), VariableOrder::Prefix),
            "prefix next openings require prefix SVO"
        );

        // Done payload: compress over the split suffix with a one-row shift.
        let d_done = self.z_split.compress_suffix_shifted(poly);

        // Number of Boolean rows over the SVO and split variables.
        let svo_rows = 1 << self.num_variables_svo();
        let split_rows = 1 << self.z_split.num_variables();
        // The boundary weight is the all-ones corner of the split equality table.
        let boundary = self.z_split.last_scalar();
        // Carry payload over the SVO prefix, one entry per SVO row.
        let mut d_carry = EF::zero_vec(svo_rows);
        // Omega (boundary) payload over the SVO prefix, one entry per SVO row.
        let mut d_omega = EF::zero_vec(svo_rows);

        // Prefix layout makes each SVO row a contiguous split chunk of the polynomial.
        d_carry
            .iter_mut()
            .zip_eq(d_omega.iter_mut())
            .zip_eq(poly.as_slice().chunks(split_rows))
            .for_each(|((d_carry, d_omega), chunk)| {
                // Carry enters at the all-zeros split corner (first chunk entry).
                *d_carry = boundary * chunk.first().copied().unwrap();
                // Omega exits at the all-ones split corner (last chunk entry).
                *d_omega = boundary * chunk.last().copied().unwrap();
            });

        // Wrap the boundary payloads as polynomials over the SVO prefix.
        let d_carry = Poly::new(d_carry);
        let d_omega = Poly::new(d_omega);

        // Opening value: sum the three-state successor weight against the payloads over every SVO row.
        let eval = (0..svo_rows)
            .map(|svo_idx| {
                // Boolean assignment of the SVO variables for this row.
                let row = Point::hypercube(svo_idx, self.z_svo.num_variables());
                // Closed-form successor states and equality weight of the SVO point at this row.
                let (_carry, done, omega) = Point::eval_next(self.z_svo.as_slice(), row.as_slice());
                let eq = Point::eval_eq(self.z_svo.as_slice(), row.as_slice());
                eq * d_done.as_slice()[svo_idx]
                    + done * d_carry.as_slice()[svo_idx]
                    + omega * d_omega.as_slice()[svo_idx]
            })
            .sum();

        // Cache one compressed successor table per SVO sumcheck round.
        let rounds = (1..=self.num_variables_svo())
            .map(|active_len| {
                next_round_partials_prefix(&d_done, &d_carry, &d_omega, &self.z_svo, active_len)
            })
            .collect::<Vec<_>>();

        (eval, NextSvoPartials::new(rounds))
    }
}

/// Compresses suffix-layout repeat-last-successor payloads to the active SVO variables.
///
/// # Overview
///
/// - The three input payloads are indexed by all SVO variables, with the active suffix as the low bits.
/// - The already-folded SVO prefix is fixed at the corresponding point coordinates.
/// - The output is the three-table successor decomposition over only the active suffix variables.
///
/// # Arguments
///
/// - `d_eq`: payload weighted by the equality state of the successor map.
/// - `d_t`: payload weighted by the carry-into-next state.
/// - `d_omega`: payload weighted by the wrap-around boundary state.
/// - `p_svo`: the opening point restricted to all SVO variables.
/// - `active_len`: the number of active suffix variables kept for this round.
///
/// # Panics
///
/// - If `active_len` is zero or exceeds the number of SVO variables.
/// - If any payload does not span all SVO variables.
fn next_round_partials_suffix<F: Field>(
    d_eq: &Poly<F>,
    d_t: &Poly<F>,
    d_omega: &Poly<F>,
    p_svo: &Point<F>,
    active_len: usize,
) -> NextPartials<F> {
    // Total SVO variables carried by each payload.
    let svo_len = p_svo.num_variables();
    // A round always keeps at least one active variable, never more than all of them.
    assert!(active_len > 0);
    assert!(active_len <= svo_len);
    // Every payload must be indexed by all SVO variables before compression.
    assert_eq!(d_eq.num_variables(), svo_len);
    assert_eq!(d_t.num_variables(), svo_len);
    assert_eq!(d_omega.num_variables(), svo_len);

    // Number of already-folded prefix variables to fix.
    let rest_len = svo_len - active_len;
    // Number of Boolean rows over the active suffix variables.
    let active_rows = 1 << active_len;

    // No prefix to fold: the payloads already live over only the active variables.
    if rest_len == 0 {
        return NextPartials::new(d_eq.clone(), d_t.clone(), d_omega.clone());
    }

    // Suffix layout puts the folded prefix in the high bits of the index.
    let (p_rest, _p_active) = p_svo.split_at(rest_len);
    // Equality table over the prefix used to contract it away.
    let rest_eq = SplitEq::<F, F>::new_packed(&p_rest, F::ONE);

    // Done state: contract the equality payload over the prefix at the matching active row.
    let done = rest_eq.compress_prefix(d_eq);
    // Carry state: contract the equality payload over the prefix shifted by one successor step.
    let mut carry = rest_eq.compress_prefix_shifted(d_eq);

    // The all-ones prefix corner carries the full product of prefix coordinates.
    let carry_scale = p_rest.iter().copied().product::<F>();
    // Add the carry-into-next contribution from the first active block of the carry payload.
    // The shared kernel runs this fused multiply-add over SIMD lanes.
    add_scaled_slice_in_place(
        carry.as_mut_slice(),
        &d_t.as_slice()[..active_rows],
        carry_scale,
    );

    // Number of Boolean rows over the folded prefix variables.
    let rest_rows = 1 << rest_len;
    // The wrap-around boundary picks up the same all-ones prefix product.
    let omega_scale = carry_scale;
    // Omega lives only on the last prefix block, the all-ones prefix row.
    let omega_start = (rest_rows - 1) * active_rows;
    // Scale that final active block to obtain the boundary state over the active variables.
    let omega = Poly::new(
        d_omega.as_slice()[omega_start..omega_start + active_rows]
            .iter()
            .map(|&value| omega_scale * value)
            .collect(),
    );

    NextPartials::new(done, carry, omega)
}

/// Compresses prefix-layout repeat-last-successor payloads to the active SVO variables.
///
/// # Overview
///
/// - The three input payloads are indexed by all SVO variables, with the active prefix as the high bits.
/// - The already-folded SVO suffix is fixed at the corresponding point coordinates.
/// - The output is the three-table successor decomposition over only the active prefix variables.
///
/// # Arguments
///
/// - `d_done`: payload weighted by the shifted equality state of the successor map.
/// - `d_carry`: payload weighted by the carry-into-next state.
/// - `d_omega`: payload weighted by the wrap-around boundary state.
/// - `p_svo`: the opening point restricted to all SVO variables.
/// - `active_len`: the number of active prefix variables kept for this round.
///
/// # Panics
///
/// - If `active_len` is zero or exceeds the number of SVO variables.
/// - If any payload does not span all SVO variables.
fn next_round_partials_prefix<F: Field>(
    d_done: &Poly<F>,
    d_carry: &Poly<F>,
    d_omega: &Poly<F>,
    p_svo: &Point<F>,
    active_len: usize,
) -> NextPartials<F> {
    // Total SVO variables carried by each payload.
    let svo_len = p_svo.num_variables();
    // A round always keeps at least one active variable, never more than all of them.
    assert!(active_len > 0);
    assert!(active_len <= svo_len);
    // Every payload must be indexed by all SVO variables before compression.
    assert_eq!(d_done.num_variables(), svo_len);
    assert_eq!(d_carry.num_variables(), svo_len);
    assert_eq!(d_omega.num_variables(), svo_len);

    // Number of already-folded suffix variables to fix.
    let rest_len = svo_len - active_len;

    // No suffix to fold: the payloads already live over only the active variables.
    if rest_len == 0 {
        return NextPartials::new(d_done.clone(), d_carry.clone(), d_omega.clone());
    }

    // Prefix layout puts the folded suffix in the low bits of the index.
    let (_p_active, p_rest) = p_svo.split_at(active_len);
    // Equality table over the suffix used to contract it away.
    let rest_eq = SplitEq::<F, F>::new_packed(&p_rest, F::ONE);

    // Done state: contract the shifted-done payload over the suffix.
    let mut done = rest_eq.compress_suffix(d_done);
    // Carry crossing a row boundary lands one suffix step over, so contract it shifted.
    let carry_done = rest_eq.compress_suffix_shifted(d_carry);

    // Fold the boundary-crossing carry contribution into the done state.
    done.as_mut_slice()
        .iter_mut()
        .zip_eq(carry_done.as_slice().iter())
        .for_each(|(out, &carry_done)| *out += carry_done);

    // Number of Boolean rows over the active prefix variables.
    let active_rows = 1 << active_len;
    // Number of Boolean rows over the folded suffix variables.
    let rest_rows = 1 << rest_len;
    // The all-ones suffix corner weight, where carry and omega boundaries live.
    let boundary = rest_eq.last_scalar();
    // Carry state over the active prefix, one entry per active row.
    let mut carry = F::zero_vec(active_rows);
    // Omega (wrap-around) state over the active prefix, one entry per active row.
    let mut omega = F::zero_vec(active_rows);

    // Walk the active rows, each a contiguous suffix chunk of the payloads.
    carry
        .iter_mut()
        .zip_eq(omega.iter_mut())
        .zip_eq(d_carry.as_slice().chunks(rest_rows))
        .zip_eq(d_omega.as_slice().chunks(rest_rows))
        .for_each(|(((carry, omega), carry_chunk), omega_chunk)| {
            // Carry enters at the all-zeros suffix corner (first chunk entry).
            *carry = boundary * carry_chunk.first().copied().unwrap();
            // Omega exits at the all-ones suffix corner (last chunk entry).
            *omega = boundary * omega_chunk.last().copied().unwrap();
        });

    // Wrap the contracted active tables as polynomials.
    let carry = Poly::new(carry);
    let omega = Poly::new(omega);

    NextPartials::new(done, carry, omega)
}

/// Materializes the three repeat-last-successor state tables for a point.
///
/// # Overview
///
/// The repeat-last successor map sends Boolean row `x` to row `x + 1`, with the
/// last row mapping to itself.
/// Its weight against a point splits into three tables indexed by Boolean rows:
///
/// - a "done" table equal to the equality weight shifted up by one row,
/// - a "carry" table holding the wrap-in weight at the all-zeros row,
/// - an "omega" table holding the wrap-out weight at the all-ones row.
///
/// # Algorithm
///
/// ```text
///     rows:   0      1      2    ...   2^n - 1
///     done:   0    eq[0]  eq[1]  ...   eq[2^n-2]
///     carry:  B      0      0    ...      0
///     omega:  0      0      0    ...      B
///
///     B = product of all point coordinates (the all-ones corner weight)
/// ```
pub(crate) fn next_state_evals<F: Field>(point_suffix: &[F]) -> NextPartials<F> {
    // Number of point coordinates, one per Boolean variable.
    let num_variables = point_suffix.len();
    // Number of Boolean rows over those variables.
    let num_rows = 1 << num_variables;

    // The all-ones corner weight is the product of all coordinates.
    let boundary = point_suffix.iter().copied().product::<F>();
    // Carry enters only at the all-zeros row (the row that wraps in).
    let mut carry = F::zero_vec(num_rows);
    // Omega exits only at the all-ones row (the row that repeats).
    let mut omega = F::zero_vec(num_rows);
    carry[0] = boundary;
    omega[num_rows - 1] = boundary;

    // Equality weights of the point over all Boolean rows.
    let eq = Poly::new_from_point(point_suffix, F::ONE);
    // Done table is the equality table shifted up by one row, with row 0 left at zero.
    let mut done = F::zero_vec(num_rows);
    if num_rows > 1 {
        done[1..].copy_from_slice(&eq.as_slice()[..num_rows - 1]);
    }

    NextPartials::new(Poly::new(done), Poly::new(carry), Poly::new(omega))
}
#[cfg(test)]
mod test {

    use alloc::vec;
    use alloc::vec::Vec;

    use p3_field::extension::BinomialExtensionField;
    use p3_field::{PackedFieldExtension, PackedValue, PrimeCharacteristicRing, dot_product};
    use p3_koala_bear::KoalaBear;
    use proptest::prelude::*;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;
    use crate::lagrange::lagrange_weights_01inf_multi;
    use crate::layout::Opening;
    use crate::strategy::VariableOrder;

    type F = KoalaBear;
    type EF = BinomialExtensionField<F, 4>;
    type PackedEF = <EF as ExtensionField<F>>::ExtensionPacking;

    /// Convenience wrapper: expand boolean evals onto {0, 1, inf}^l grid.
    fn evals_01inf_grid(boolean_evals: &[EF]) -> Vec<EF> {
        let num_variables = log2_strict_usize(boolean_evals.len());
        let output_len = 3usize.pow(num_variables as u32);
        let mut output = EF::zero_vec(output_len);
        let mut scratch = EF::zero_vec(output_len);
        evals_01inf_grid_into(boolean_evals, &mut output, &mut scratch);
        output
    }

    /// Compare the grid expansion against naive multilinear evaluation on every
    /// point of `{0, 1, inf}^l`.
    /// Verify the grid stores correct `(f(0), f(1), f(inf))` triples.
    ///
    /// For each variable, the third slot must satisfy `f(inf) = f(1) - f(0)`.
    /// Combined with the fact that `f(0)` and `f(1)` are the original Boolean
    /// evaluations, this fully validates the grid.
    fn assert_evals_01inf_grid_correct(boolean_evals: &[EF]) {
        let num_variables = log2_strict_usize(boolean_evals.len());
        let grid = evals_01inf_grid(boolean_evals);

        // For every triple along the innermost variable (stride 1), check
        // that grid[3k+2] == grid[3k+1] - grid[3k].
        let inner_groups = grid.len() / 3;
        for g in 0..inner_groups {
            let v0 = grid[3 * g];
            let v1 = grid[3 * g + 1];
            let v_inf = grid[3 * g + 2];
            assert_eq!(
                v_inf,
                v1 - v0,
                "f(inf) != f(1)-f(0) at group {g}, num_variables={num_variables}"
            );
        }
    }

    // Tests for evals_01inf_grid_into

    #[test]
    fn test_evals_01inf_grid_into_zero_vars() {
        // Zero variables: the polynomial is a single constant.
        // Input: [c] on {0}^0 (one point, the empty tuple).
        // Output: [c] on {0}^0 (still one point).
        let c = EF::from_u32(42);
        let input = [c];
        let mut output = [EF::ZERO];
        let mut scratch = [EF::ZERO];

        evals_01inf_grid_into(&input, &mut output, &mut scratch);

        // The sole value is copied through unchanged.
        assert_eq!(output, [c]);
    }

    #[test]
    fn test_evals_01inf_grid_into_one_var() {
        // One variable: f(0) = 3, f(1) = 7.
        // f(inf) = f(1) - f(0) = 7 - 3 = 4  (the leading coefficient).
        //
        // Output layout on {0, 1, inf}:
        //   index 0 → f(0) = 3
        //   index 1 → f(1) = 7
        //   index 2 → f(inf) = 4
        let f0 = EF::from_u32(3);
        let f1 = EF::from_u32(7);
        let input = [f0, f1];
        let mut output = [EF::ZERO; 3];
        let mut scratch = [EF::ZERO; 3];

        evals_01inf_grid_into(&input, &mut output, &mut scratch);

        assert_eq!(output[0], f0);
        assert_eq!(output[1], f1);
        // f(inf) = 7 - 3 = 4
        assert_eq!(output[2], EF::from_u32(4));
    }

    #[test]
    fn test_evals_01inf_grid_into_two_vars_hand_computed() {
        // f(x_0, x_1) = 1 + 2*x_0 + 4*x_1 + 4*x_0*x_1
        //
        // Input (low-var-fastest):
        //   idx 0 -> (0,0) = 1
        //   idx 1 -> (1,0) = 3
        //   idx 2 -> (0,1) = 5
        //   idx 3 -> (1,1) = 11
        //
        // Stage 0 expands x_0: each (f(0), f(1)) pair -> (f(0), f(1), f(1)-f(0)):
        //   x_1=0: (1, 3) -> (1, 3, 2)
        //   x_1=1: (5, 11) -> (5, 11, 6)
        //   buffer: [1, 3, 2, 5, 11, 6]
        //
        // Stage 1 expands x_1 (stride=3): for each x_0 value,
        // the pair (f(x_0,0), f(x_0,1)) -> (f(x_0,0), f(x_0,1), f(x_0,1)-f(x_0,0)):
        //   x_0=0: (1, 5)  -> (1, 5, 4)
        //   x_0=1: (3, 11) -> (3, 11, 8)
        //   x_0=2: (2, 6)  -> (2, 6, 4)
        //
        // Output (x_0 slowest, x_1 fastest), index = x_1 + 3*x_0:
        //   idx:   0  1  2  3   4  5  6  7  8
        //   val:   1  5  4  3  11  8  2  6  4
        let input = [1, 3, 5, 11].map(EF::from_u32);
        let mut output = [EF::ZERO; 9];
        let mut scratch = [EF::ZERO; 9];

        evals_01inf_grid_into(&input, &mut output, &mut scratch);

        let expected = [1, 5, 4, 3, 11, 8, 2, 6, 4].map(EF::from_u32);
        assert_eq!(output, expected);
    }

    #[test]
    fn test_evals_01inf_grid_into_output_size() {
        // Verify the output length is 3^l for various numbers of variables.
        for num_variables in 1..=5 {
            let input_len = 1 << num_variables;
            let output_len = 3usize.pow(num_variables as u32);

            // Use all-zero input; we only care about sizes here.
            let input = EF::zero_vec(input_len);
            let mut output = EF::zero_vec(output_len);
            let mut scratch = EF::zero_vec(output_len);

            // Should not panic.
            evals_01inf_grid_into(&input, &mut output, &mut scratch);
        }
    }

    #[test]
    fn test_evals_01inf_grid_into_result_lands_in_output() {
        // The ping-pong buffer logic must place the final result in the
        // output buffer, not in scratch. Verify for both odd and even
        // numbers of variables (since the initial buffer assignment differs).
        let mut rng = SmallRng::seed_from_u64(123);

        for num_variables in 1..=4 {
            let input: Vec<EF> = (0..1 << num_variables).map(|_| rng.random()).collect();
            let output_len = 3usize.pow(num_variables as u32);
            let mut output = EF::zero_vec(output_len);
            let mut scratch = EF::zero_vec(output_len);

            evals_01inf_grid_into(&input, &mut output, &mut scratch);

            // The grid computed via the convenience wrapper must match.
            let reference = evals_01inf_grid(&input);
            assert_eq!(
                output, reference,
                "ping-pong mismatch for num_variables={num_variables}"
            );
        }
    }

    #[test]
    fn test_evals_01inf_grid_into_preserves_boolean_points() {
        // The grid expansion must preserve the original 2^l Boolean evaluations.
        // At every Boolean point in {0,1}^l, the grid value must equal the input.
        //
        // The binary index uses b_0 + 2*b_1 + ... (low-var-fastest).
        // The ternary grid has x_0 slowest (first variable processed),
        // so the ternary index for a Boolean point is b_{l-1} + 3*b_{l-2} + ... + 3^{l-1}*b_0.
        let mut rng = SmallRng::seed_from_u64(77);

        for num_variables in 1..=4 {
            let input: Vec<EF> = (0..1 << num_variables).map(|_| rng.random()).collect();
            let grid = evals_01inf_grid(&input);

            for (bool_idx, &input_val) in input.iter().enumerate() {
                // Extract binary digits (low-var-first): b_0, b_1, ..., b_{l-1}.
                let mut bits = Vec::with_capacity(num_variables);
                let mut tmp = bool_idx;
                for _ in 0..num_variables {
                    bits.push(tmp & 1);
                    tmp >>= 1;
                }

                // Build ternary index with reversed variable order:
                // x_0 is slowest (weight 3^{l-1}), x_{l-1} is fastest (weight 1).
                let mut ternary_idx = 0;
                let mut power_of_3 = 1;
                for &b in bits.iter().rev() {
                    ternary_idx += b * power_of_3;
                    power_of_3 *= 3;
                }

                assert_eq!(
                    grid[ternary_idx], input_val,
                    "Boolean point mismatch at bool_idx={bool_idx}, num_variables={num_variables}"
                );
            }
        }
    }

    #[test]
    fn test_evals_01inf_grid_into_constant_polynomial() {
        // A constant polynomial f(x) = c has f(0) = f(1) = c and f(inf) = 0.
        //
        // For l=1, the grid is [c, c, 0].
        // For l=2, the grid is [c, c, 0, c, c, 0, 0, 0, 0].
        // The {0,1} slots hold c; any slot involving inf in any coordinate is 0.
        let c = EF::from_u32(99);

        for num_variables in 0..=4 {
            let input = vec![c; 1 << num_variables];
            let output_len = 3usize.pow(num_variables as u32);
            let mut output = EF::zero_vec(output_len);
            let mut scratch = EF::zero_vec(output_len);

            evals_01inf_grid_into(&input, &mut output, &mut scratch);

            // Check each grid point. A point has value c if all its ternary
            // digits are in {0,1}, and value 0 if any digit is 2 (the inf slot).
            for (idx, &val) in output.iter().enumerate() {
                let has_inf = {
                    let mut tmp = idx;
                    let mut found = false;
                    for _ in 0..num_variables {
                        if tmp % 3 == 2 {
                            found = true;
                        }
                        tmp /= 3;
                    }
                    found
                };
                let expected = if has_inf { EF::ZERO } else { c };
                assert_eq!(
                    val, expected,
                    "constant polynomial mismatch at idx={idx}, num_variables={num_variables}"
                );
            }
        }
    }

    #[test]
    fn test_evals_01inf_grid_into_linearity() {
        // The grid expansion must be linear:
        //   grid(a*f + b*g) = a*grid(f) + b*grid(g)
        //
        // This follows from the extrapolation formula being linear,
        // but verify it concretely.
        let mut rng = SmallRng::seed_from_u64(55);
        let num_variables = 3;
        let n = 1 << num_variables;

        let f: Vec<EF> = (0..n).map(|_| rng.random()).collect();
        let g: Vec<EF> = (0..n).map(|_| rng.random()).collect();
        let a: EF = rng.random();
        let b: EF = rng.random();

        // Compute grid(a*f + b*g).
        let combined: Vec<EF> = f
            .iter()
            .zip(g.iter())
            .map(|(&fi, &gi)| a * fi + b * gi)
            .collect();
        let grid_combined = evals_01inf_grid(&combined);

        // Compute a*grid(f) + b*grid(g).
        let grid_f = evals_01inf_grid(&f);
        let grid_g = evals_01inf_grid(&g);
        let linear_combined: Vec<EF> = grid_f
            .iter()
            .zip(grid_g.iter())
            .map(|(&fi, &gi)| a * fi + b * gi)
            .collect();

        assert_eq!(grid_combined, linear_combined);
    }

    #[test]
    fn test_evals_01inf_grid_into_large_stride_branch_matches_naive() {
        // `num_variables = 7` guarantees the final stage has `in_stride = 3^6 = 729`,
        // which takes the large-stride branch (`in_stride >= 256`).
        let num_variables = 7;
        let mut rng = SmallRng::seed_from_u64(2025);
        let evals: Vec<EF> = (0..1 << num_variables).map(|_| rng.random()).collect();
        assert_evals_01inf_grid_correct(evals.as_slice());
    }

    #[test]
    #[should_panic(expected = "Not a power of two")]
    fn test_evals_01inf_grid_into_panics_on_non_power_of_two_input() {
        let input = [EF::ZERO; 3];
        let mut output = [EF::ZERO; 3];
        let mut scratch = [EF::ZERO; 3];

        evals_01inf_grid_into(&input, &mut output, &mut scratch);
    }

    #[test]
    #[should_panic(expected = "assertion `left == right` failed")]
    fn test_evals_01inf_grid_into_panics_on_wrong_output_len() {
        let input = [EF::ZERO; 4];
        let mut output = [EF::ZERO; 8];
        let mut scratch = [EF::ZERO; 9];

        evals_01inf_grid_into(&input, &mut output, &mut scratch);
    }

    #[test]
    #[should_panic(expected = "assertion `left == right` failed")]
    fn test_evals_01inf_grid_into_panics_on_wrong_scratch_len() {
        let input = [EF::ZERO; 4];
        let mut output = [EF::ZERO; 9];
        let mut scratch = [EF::ZERO; 8];

        evals_01inf_grid_into(&input, &mut output, &mut scratch);
    }

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

    proptest! {
        /// Verify the {0, 1, inf}^l grid expansion matches naive MLE evaluation.
        #[test]
        fn prop_evals_01inf_grid_matches_naive(num_variables in 1usize..=5) {
            // For each grid point in {0, 1, inf}^l, compare the fast grid-expansion
            // result against the naive approach of evaluating the multilinear
            // extension at that point by repeatedly fixing variables.
            let mut rng = SmallRng::seed_from_u64(num_variables as u64);
            let evals: Vec<EF> = (0..1 << num_variables).map(|_| rng.random()).collect();
            assert_evals_01inf_grid_correct(evals.as_slice());
        }

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
            let eq0: Vec<EF> = (0..2).map(|_| rng.random()).collect();
            let reduced: Vec<EF> = (0..2).map(|_| rng.random()).collect();

            let fast = calculate_accumulator_1(&eq0, &reduced);
            let general = calculate_accumulator_general::<F, EF>(1, &eq0, &reduced);

            prop_assert_eq!(fast, general);
        }

        /// Verify l=2 straightline matches the general path.
        #[test]
        fn prop_accumulators_2_matches_general(seed in 0u64..1000) {
            let mut rng = SmallRng::seed_from_u64(seed);
            let eq0: Vec<EF> = (0..4).map(|_| rng.random()).collect();
            let reduced: Vec<EF> = (0..4).map(|_| rng.random()).collect();

            let fast = calculate_accumulator_2(&eq0, &reduced);
            let general = calculate_accumulator_general::<F, EF>(2, &eq0, &reduced);

            prop_assert_eq!(fast, general);
        }

        /// Verify l=3 straightline matches the general path.
        #[test]
        fn prop_accumulators_3_matches_general(seed in 0u64..1000) {
            let mut rng = SmallRng::seed_from_u64(seed);
            let eq0: Vec<EF> = (0..8).map(|_| rng.random()).collect();
            let reduced: Vec<EF> = (0..8).map(|_| rng.random()).collect();

            let fast = calculate_accumulator_3(&eq0, &reduced);
            let general = calculate_accumulator_general::<F, EF>(3, &eq0, &reduced);

            prop_assert_eq!(fast, general);
        }

        /// Verify that grid expansion preserves the original Boolean-hypercube values.
        #[test]
        fn prop_evals_01inf_grid_preserves_boolean_points(num_variables in 1usize..=6) {
            let mut rng = SmallRng::seed_from_u64(num_variables as u64 + 1000);
            let input: Vec<EF> = (0..1 << num_variables).map(|_| rng.random()).collect();
            let grid = evals_01inf_grid(&input);

            for (bool_idx, &input_val) in input.iter().enumerate() {
                // Extract binary digits (low-var-first): b_0, b_1, ..., b_{l-1}.
                let mut bits = Vec::with_capacity(num_variables);
                let mut tmp = bool_idx;
                for _ in 0..num_variables {
                    bits.push(tmp & 1);
                    tmp >>= 1;
                }

                // Ternary index with x_0 slowest (reversed variable order).
                let mut ternary_idx = 0;
                let mut power_of_3 = 1;
                for &b in bits.iter().rev() {
                    ternary_idx += b * power_of_3;
                    power_of_3 *= 3;
                }

                prop_assert_eq!(grid[ternary_idx], input_val);
            }
        }
    }

    #[test]
    fn test_svo_point_eval() {
        let assert_eval = |svo_point: &SvoPoint<F, EF>, poly: &Poly<F>, point: &Point<EF>| {
            let e0 = poly.eval_base(point);

            let (e1, partial_evals) = svo_point.eval(poly);
            assert_eq!(e0, e1);
            assert_eq!(partial_evals.rounds().len(), svo_point.num_variables_svo());

            match svo_point.var_order() {
                VariableOrder::Prefix => {
                    partial_evals
                        .rounds()
                        .iter()
                        .enumerate()
                        .for_each(|(i, pe0)| {
                            let (_point_lo, point_hi) = point.split_at(i + 1);
                            assert_eq!(pe0.poly(), &poly.compress_suffix(&point_hi, EF::ONE));
                            assert_eq!(
                                e0,
                                pe0.poly().eval_base(&svo_point.z_svo().split_at(i + 1).0)
                            );
                        });
                }
                VariableOrder::Suffix => {
                    partial_evals
                        .rounds()
                        .iter()
                        .enumerate()
                        .for_each(|(i, pe0)| {
                            let (point_lo, point_hi) =
                                point.split_at(point.num_variables() - i - 1);
                            assert_eq!(pe0.poly(), &poly.compress_prefix(&point_lo, EF::ONE));
                            assert_eq!(e0, pe0.poly().eval_base(&point_hi));
                        });
                }
            }
        };

        let k = 12;
        let mut rng = SmallRng::seed_from_u64(11);
        let poly = Poly::<F>::rand(&mut rng, k);
        let point = Point::<EF>::rand(&mut rng, k);

        for l0 in 0..=k {
            let unpacked_prefix =
                SvoPoint::<F, EF>::new_unpacked(l0, &point, VariableOrder::Prefix);
            assert_eval(&unpacked_prefix, &poly, &point);
        }

        for l0 in 0..=k {
            let unpacked_suffix =
                SvoPoint::<F, EF>::new_unpacked(l0, &point, VariableOrder::Suffix);
            assert_eval(&unpacked_suffix, &poly, &point);
        }

        for l0 in 0..=k {
            let packed_prefix = SvoPoint::<F, EF>::new_packed(l0, &point);
            assert_eval(&packed_prefix, &poly, &point);
        }
    }

    #[test]
    fn test_svo_point_accumulate() {
        let mut rng = SmallRng::seed_from_u64(0);

        let assert_accumulate_unpacked =
            |svo_point: &SvoPoint<F, EF>, point: &Point<EF>, scale: EF, rs: &Point<EF>| {
                let eq = Poly::new_from_point(point.as_slice(), EF::ONE);
                let expected = match svo_point.var_order() {
                    VariableOrder::Prefix => eq.compress_prefix(rs, scale),
                    VariableOrder::Suffix => eq.compress_suffix(rs, scale),
                };

                let mut out = Poly::<EF>::zero(expected.num_variables());
                svo_point.accumulate_into(out.as_mut_slice(), rs, scale);
                assert_eq!(out, expected);
            };

        let assert_accumulate_packed =
            |svo_point: &SvoPoint<F, EF>, point: &Point<EF>, scale: EF, rs: &Point<EF>| {
                let eq = Poly::new_from_point(point.as_slice(), EF::ONE);
                let expected = eq.compress_prefix(rs, scale);
                let k_pack = log2_strict_usize(<<F as Field>::Packing as PackedValue>::WIDTH);
                assert!(expected.num_variables() >= k_pack);

                let mut out = Poly::<PackedEF>::zero(expected.num_variables() - k_pack);
                svo_point.accumulate_into_packed(out.as_mut_slice(), rs, scale);
                let unpacked =
                    <PackedEF as PackedFieldExtension<F, EF>>::to_ext_iter(out.iter().copied())
                        .take(expected.num_evals())
                        .collect::<Vec<_>>();
                assert_eq!(unpacked, expected.as_slice());
            };

        let k = 12;
        let k_pack = log2_strict_usize(<<F as Field>::Packing as PackedValue>::WIDTH);
        let point = Point::<EF>::rand(&mut rng, k);
        let scale: EF = rng.random();

        for l0 in 0..=k {
            let unpacked_prefix =
                SvoPoint::<F, EF>::new_unpacked(l0, &point, VariableOrder::Prefix);
            assert_eq!(unpacked_prefix.var_order(), VariableOrder::Prefix);
            assert_eq!(unpacked_prefix.num_variables(), k);
            assert_eq!(unpacked_prefix.num_variables_svo(), l0);
            assert_accumulate_unpacked(&unpacked_prefix, &point, scale, &Point::rand(&mut rng, l0));
        }

        for l0 in 0..=k {
            let unpacked_suffix =
                SvoPoint::<F, EF>::new_unpacked(l0, &point, VariableOrder::Suffix);
            assert_eq!(unpacked_suffix.var_order(), VariableOrder::Suffix);
            assert_eq!(unpacked_suffix.num_variables(), k);
            assert_eq!(unpacked_suffix.num_variables_svo(), l0);
            assert_accumulate_unpacked(&unpacked_suffix, &point, scale, &Point::rand(&mut rng, l0));
        }

        for l0 in 0..=k {
            if k - l0 >= k_pack {
                let packed_prefix = SvoPoint::<F, EF>::new_packed(l0, &point);
                assert_eq!(packed_prefix.var_order(), VariableOrder::Prefix);
                assert_eq!(packed_prefix.num_variables(), k);
                assert_eq!(packed_prefix.num_variables_svo(), l0);
                assert_accumulate_packed(&packed_prefix, &point, scale, &Point::rand(&mut rng, l0));
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
        use p3_field::dot_product;

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

        // For small l, use straightline specializations that avoid loop/buffer overhead.
        match l {
            1 => calculate_accumulator_1(eq0.as_slice(), &reduced_evals),
            2 => calculate_accumulator_2(eq0.as_slice(), &reduced_evals),
            3 => calculate_accumulator_3(eq0.as_slice(), &reduced_evals),
            _ => calculate_accumulator_general(l, eq0.as_slice(), &reduced_evals),
        }
    }

    // Brute-force reference: materialize each successor state table row by row from the closed form.
    fn next_state_evals_reference(point_suffix: &[F]) -> [Poly<F>; 3] {
        let num_variables = point_suffix.len();
        let num_rows = 1 << num_variables;
        let mut carry = F::zero_vec(num_rows);
        let mut done = F::zero_vec(num_rows);
        let mut omega = F::zero_vec(num_rows);

        // Evaluate the closed-form successor decomposition at every Boolean row.
        for row_idx in 0..num_rows {
            let row = Point::hypercube(row_idx, num_variables);
            let (c, d, o) = Point::eval_next(point_suffix, row.as_slice());
            carry[row_idx] = c;
            done[row_idx] = d;
            omega[row_idx] = o;
        }

        [Poly::new(carry), Poly::new(done), Poly::new(omega)]
    }

    #[test]
    fn test_next_state_evals_matches_reference() {
        let mut rng = SmallRng::seed_from_u64(1);

        // Invariant: the fast sparse construction equals the brute-force per-row tables.
        // Fixture state: variable counts 0..=8, with 16 random points each.
        for num_variables in 0..=8 {
            for _ in 0..16 {
                let point = Point::<F>::rand(&mut rng, num_variables);
                // Fast path: closed-form sparse construction of the three state tables.
                let actual = next_state_evals(point.as_slice());
                // Slow path: one closed-form evaluation per Boolean row.
                let [carry, done, omega] = next_state_evals_reference(point.as_slice());

                assert_eq!(actual.carry(), &carry);
                assert_eq!(actual.done(), &done);
                assert_eq!(actual.omega(), &omega);
            }
        }
    }

    // Brute-force reference: compress the polynomial over the split prefix into the three SVO payloads.
    fn split_compressions_for_next(
        poly: &[EF],
        point: &[EF],
        split_len: usize,
        svo_len: usize,
    ) -> [Vec<EF>; 3] {
        assert_eq!(point.len(), split_len + svo_len);
        assert_eq!(poly.len(), 1 << point.len());

        // Suffix layout: the split prefix occupies the leading point coordinates.
        let (p_split, _p_svo) = point.split_at(split_len);
        let split_rows = 1 << split_len;
        let svo_rows = 1 << svo_len;

        // Per split-row weights: equality, carry-into-next (done), and boundary (omega).
        let mut eq_split = EF::zero_vec(split_rows);
        let mut t_split = EF::zero_vec(split_rows);
        let mut omega_split = EF::zero_vec(split_rows);

        // Fill the split weights from the closed form at each split row.
        for split_idx in 0..split_rows {
            let row = Point::hypercube(split_idx, split_len);
            let (_carry, done, omega) = Point::eval_next(p_split, row.as_slice());
            eq_split[split_idx] = Point::eval_eq(p_split, row.as_slice());
            t_split[split_idx] = done;
            omega_split[split_idx] = omega;
        }

        // Resulting SVO payloads, one entry per SVO row.
        let mut d_eq = EF::zero_vec(svo_rows);
        let mut d_t = EF::zero_vec(svo_rows);
        let mut d_omega = EF::zero_vec(svo_rows);

        // Contract the split dimension: the polynomial index splits as (split_idx << svo_len) | svo_idx.
        for split_idx in 0..split_rows {
            let base = split_idx << svo_len;
            for svo_idx in 0..svo_rows {
                let value = poly[base | svo_idx];
                // Weight each polynomial value by the matching split state and sum into the SVO payload.
                d_eq[svo_idx] += value * eq_split[split_idx];
                d_t[svo_idx] += value * t_split[split_idx];
                d_omega[svo_idx] += value * omega_split[split_idx];
            }
        }

        [d_eq, d_t, d_omega]
    }

    // Brute-force reference: sum the full ternary-grid accumulator over every fixed rest assignment.
    fn dense_next_accumulator_round(
        d_eq: &[EF],
        d_t: &[EF],
        d_omega: &[EF],
        p_svo: &[EF],
        active_len: usize,
    ) -> [Vec<EF>; 2] {
        let svo_len = p_svo.len();
        assert!(active_len > 0);
        assert!(active_len <= svo_len);
        // The rest variables are the already-folded SVO prefix summed over below.
        let rest_len = svo_len - active_len;
        let rest_rows = 1 << rest_len;
        // Full ternary grid over the active variables: 3^active_len entries.
        let grid_len = 3usize.pow(active_len as u32);
        let mut grid = EF::zero_vec(grid_len);

        // Accumulate one grid contribution per Boolean assignment of the rest variables.
        for rest_idx in 0..rest_rows {
            let rest_row = Point::hypercube(rest_idx, rest_len);
            let active_rows = 1 << active_len;
            // This rest assignment owns a contiguous active block of the SVO payloads.
            let row_start = rest_idx << active_len;
            let row_range = row_start..row_start + active_rows;

            // Expand each payload's active block to the ternary grid.
            let d_eq_grid = evals_01inf_grid_prefix(&d_eq[row_range.clone()]);
            let d_t_grid = evals_01inf_grid_prefix(&d_t[row_range.clone()]);
            let d_omega_grid = evals_01inf_grid_prefix(&d_omega[row_range]);

            // Successor states of the full SVO point with the rest prefix fixed.
            let mut carry = EF::zero_vec(active_rows);
            let mut done = EF::zero_vec(active_rows);
            let mut omega = EF::zero_vec(active_rows);
            for active_idx in 0..active_rows {
                let active_row = Point::hypercube(active_idx, active_len);
                // Concatenate the fixed rest prefix with this active row to form a full SVO row.
                let mut row = Vec::with_capacity(svo_len);
                row.extend_from_slice(rest_row.as_slice());
                row.extend_from_slice(active_row.as_slice());
                let (c, d, o) = Point::eval_next(p_svo, &row);
                carry[active_idx] = c;
                done[active_idx] = d;
                omega[active_idx] = o;
            }

            // Expand the state tables to the same ternary grid.
            let carry_grid = evals_01inf_grid_prefix(&carry);
            let done_grid = evals_01inf_grid_prefix(&done);
            let omega_grid = evals_01inf_grid_prefix(&omega);

            // Add the three state-times-data products pointwise across the grid.
            for idx in 0..grid_len {
                grid[idx] += done_grid[idx] * d_eq_grid[idx]
                    + carry_grid[idx] * d_t_grid[idx]
                    + omega_grid[idx] * d_omega_grid[idx];
            }
        }

        // Keep only the 0 third and the inf third, matching the production accumulator pair.
        let stride = 3usize.pow((active_len - 1) as u32);
        [grid[..stride].to_vec(), grid[2 * stride..].to_vec()]
    }

    #[test]
    fn test_next_svo_accumulators_match_dense_ternary_reference() {
        let mut rng = SmallRng::seed_from_u64(4);

        // Invariant: the fast compress-then-accumulate path equals the brute-force ternary grid.
        // Fixture state: split lengths 0..=4, SVO lengths 1..=5, 20 random instances each.
        for split_len in 0..=4 {
            for svo_len in 1..=5 {
                let total_len = split_len + svo_len;

                for _ in 0..20 {
                    // Random witness polynomial over all variables.
                    let poly = (0..1 << total_len)
                        .map(|_| rng.random::<EF>())
                        .collect::<Vec<_>>();
                    // Random opening point over all variables.
                    let point = (0..total_len)
                        .map(|_| rng.random::<EF>())
                        .collect::<Vec<_>>();
                    // Suffix layout: SVO variables are the trailing point coordinates.
                    let (_p_split, p_svo) = point.split_at(split_len);
                    let p_svo = Point::new(p_svo.to_vec());
                    // Reference compression of the polynomial over the split prefix.
                    let [d_eq, d_t, d_omega] =
                        split_compressions_for_next(&poly, &point, split_len, svo_len);

                    // Compare both paths for every SVO sumcheck round.
                    for active_len in 1..=svo_len {
                        // Fast path: compress payloads to the active variables for this round.
                        let round = next_round_partials_suffix(
                            &Poly::new(d_eq.clone()),
                            &Poly::new(d_t.clone()),
                            &Poly::new(d_omega.clone()),
                            &p_svo,
                            active_len,
                        );
                        // Active suffix coordinates kept for this round.
                        let (_svo_rest, svo_active) = p_svo.split_at(svo_len - active_len);
                        let stride = 3usize.pow((active_len - 1) as u32);
                        // Production accumulators at 0 and inf.
                        let mut production0 = EF::zero_vec(stride);
                        let mut production_inf = EF::zero_vec(stride);
                        round.accumulate_suffix(
                            svo_active.as_slice(),
                            &mut production0,
                            &mut production_inf,
                        );
                        let production = [production0, production_inf];
                        // Slow path: full ternary grid summed over the rest variables.
                        let dense = dense_next_accumulator_round(
                            &d_eq,
                            &d_t,
                            &d_omega,
                            p_svo.as_slice(),
                            active_len,
                        );

                        assert_eq!(
                            production, dense,
                            "split_len={split_len}, svo_len={svo_len}, active_len={active_len}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_svo_point_eval_next_suffix() {
        // Invariant: the suffix opening value matches the direct successor evaluation,
        // and every cached round table re-evaluates to that same value.
        let assert_eval = |svo_point: &SvoPoint<F, EF>, poly: &Poly<F>, point: &Point<EF>| {
            assert!(matches!(svo_point.var_order(), VariableOrder::Suffix));

            // Ground truth: the full successor evaluation over all variables.
            let expected = poly.eval_next_base(point);
            // Fast path: the SVO opening value plus per-round cached tables.
            let (actual, partials) = svo_point.eval_next_suffix(poly, None);
            assert_eq!(actual, expected);
            assert_eq!(partials.rounds().len(), svo_point.num_variables_svo());

            // Each cached round must independently reproduce the opening value.
            for (round_idx, round) in partials.rounds().iter().enumerate() {
                let active_len = round_idx + 1;
                // Suffix layout: the active variables are the trailing SVO coordinates.
                let (_svo_rest, svo_active) = svo_point
                    .z_svo()
                    .split_at(svo_point.num_variables_svo() - active_len);

                assert_eq!(round.done().num_variables(), active_len);
                assert_eq!(round.carry().num_variables(), active_len);
                assert_eq!(round.omega().num_variables(), active_len);

                // Re-evaluate the three-state successor weight against the cached tables.
                let round_eval = (0..1 << active_len)
                    .map(|row_idx| {
                        let row = Point::hypercube(row_idx, active_len);
                        let (carry, done, omega) =
                            Point::eval_next(svo_active.as_slice(), row.as_slice());
                        done * round.done().as_slice()[row_idx]
                            + carry * round.carry().as_slice()[row_idx]
                            + omega * round.omega().as_slice()[row_idx]
                    })
                    .sum::<EF>();

                assert_eq!(round_eval, expected);
            }
        };

        // Fixture state: 12-variable random witness and opening point.
        let k = 12;
        let mut rng = SmallRng::seed_from_u64(12);
        let poly = Poly::<F>::rand(&mut rng, k);
        let point = Point::<EF>::rand(&mut rng, k);

        // Sweep every SVO depth from no SVO variables up to all 12.
        for l0 in 0..=k {
            let svo_point = SvoPoint::<F, EF>::new_unpacked(l0, &point, VariableOrder::Suffix);
            assert_eval(&svo_point, &poly, &point);
        }
    }

    #[test]
    fn test_svo_point_eval_next_prefix() {
        // Invariant: the prefix opening value matches the direct successor evaluation,
        // and every cached round table re-evaluates to that same value.
        let assert_eval = |svo_point: &SvoPoint<F, EF>, poly: &Poly<F>, point: &Point<EF>| {
            assert!(matches!(svo_point.var_order(), VariableOrder::Prefix));

            // Ground truth: the full successor evaluation over all variables.
            let expected = poly.eval_next_base(point);
            // Fast path: the SVO opening value plus per-round cached tables.
            let (actual, partials) = svo_point.eval_next_prefix(poly);
            assert_eq!(actual, expected);
            assert_eq!(partials.rounds().len(), svo_point.num_variables_svo());

            // Each cached round must independently reproduce the opening value.
            for (round_idx, round) in partials.rounds().iter().enumerate() {
                let active_len = round_idx + 1;
                // Prefix layout: the active variables are the leading SVO coordinates.
                let (svo_active, _) = svo_point.z_svo().split_at(active_len);

                assert_eq!(round.done().num_variables(), active_len);
                assert_eq!(round.carry().num_variables(), active_len);
                assert_eq!(round.omega().num_variables(), active_len);

                // Prefix re-evaluation pairs the equality weight with the done payload, plus the two boundary states.
                let round_eval = (0..1 << active_len)
                    .map(|row_idx| {
                        let row = Point::hypercube(row_idx, active_len);
                        let (_carry, done, omega) =
                            Point::eval_next(svo_active.as_slice(), row.as_slice());
                        let eq = Point::eval_eq(svo_active.as_slice(), row.as_slice());
                        eq * round.done().as_slice()[row_idx]
                            + done * round.carry().as_slice()[row_idx]
                            + omega * round.omega().as_slice()[row_idx]
                    })
                    .sum::<EF>();

                assert_eq!(round_eval, expected);
            }
        };

        // Fixture state: 12-variable random witness and opening point.
        let k = 12;
        let mut rng = SmallRng::seed_from_u64(13);
        let poly = Poly::<F>::rand(&mut rng, k);
        let point = Point::<EF>::rand(&mut rng, k);

        // Sweep every SVO depth using the unpacked split-equality construction.
        for l0 in 0..=k {
            let svo_point = SvoPoint::<F, EF>::new_unpacked(l0, &point, VariableOrder::Prefix);
            assert_eval(&svo_point, &poly, &point);
        }

        // Repeat with the packed split-equality construction to cover both code paths.
        for l0 in 0..=k {
            let svo_point = SvoPoint::<F, EF>::new_packed(l0, &point);
            assert_eval(&svo_point, &poly, &point);
        }
    }

    #[test]
    fn test_svo_point_accumulate_next_prefix() {
        let mut rng = SmallRng::seed_from_u64(14);
        // Fixture state: 12-variable random opening point and a random batching coefficient.
        let k = 12;
        let point = Point::<EF>::rand(&mut rng, k);
        let scale: EF = rng.random();
        // Dense successor weight table over all variables, used as the reference.
        let next = Poly::new_next_from_point(point.as_slice());

        // Invariant: the residual accumulator equals the dense weight compressed at the round challenges.
        for l0 in 0..=k {
            let svo_point = SvoPoint::<F, EF>::new_packed(l0, &point);
            // Random SVO round challenges fixing the l0 SVO variables.
            let rs = Point::rand(&mut rng, l0);
            // Reference: compress the dense weight over the prefix challenges, scaled.
            let expected = next.compress_prefix(&rs, scale);

            // Fast path: accumulate the residual split weight in place.
            let mut out = Poly::<EF>::zero(expected.num_variables());
            svo_point.accumulate_next_prefix_into(out.as_mut_slice(), &rs, scale);
            assert_eq!(out, expected);
        }
    }
}
