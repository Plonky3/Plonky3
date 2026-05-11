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
use p3_field::{ExtensionField, Field, dot_product};
use p3_maybe_rayon::prelude::*;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use p3_multilinear_util::split_eq::SplitEq;
use p3_util::log2_strict_usize;

use crate::sumcheck::layout::ProverMultiClaim;
use crate::sumcheck::strategy::VariableOrder;

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

fn evals_01inf_grid_prefix<F: Field>(evals: &[F]) -> Vec<F> {
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

/// Builds the SVO accumulators for a batch of claims.
///
/// At round `l = round_idx + 1`, we first combine the round-`l` partial evaluations
/// from all openings using the provided batching coefficients `alphas`. We then
/// evaluate both that combined polynomial and the active equality polynomial on the
/// ternary grid `{0,1,inf}^l`, and keep only the first and last thirds of that
/// recursive grid, corresponding to the active coordinate fixed to `0` and `inf`.
/// Those are exactly the two values needed to reconstruct the quadratic round
/// polynomial.
pub(crate) fn calculate_accumulators_batch<F: Field, EF: ExtensionField<F>>(
    claim: &ProverMultiClaim<F, EF>,
    alphas: &[EF],
) -> SvoAccumulators<EF> {
    assert_eq!(claim.len(), alphas.len());
    let k = claim.point().num_variables_svo();

    (0..k)
        .map(|round_idx| {
            let l = round_idx + 1;
            let mut acc = Poly::<EF>::zero(l);

            // Pick each opening's round-`round_idx` partial evaluation and
            // accumulate `alpha * partial` into `acc`.
            claim
                .openings()
                .iter()
                .map(|opening| &opening.data()[round_idx])
                .zip(alphas.iter())
                .for_each(|(partial, &alpha)| {
                    acc.as_mut_slice()
                        .iter_mut()
                        .zip_eq(partial.iter())
                        .for_each(|(out, &f)| *out += alpha * f);
                });

            if matches!(claim.point().var_order(), VariableOrder::Prefix) {
                let (svo_active, _) = claim.point().z_svo().split_at(l);
                return calculate_accumulator::<F, EF>(l, acc.as_slice(), svo_active.as_slice());
            }

            let (_, svo_active) = claim.point().z_svo().split_at(k - l);
            let eq_grid = evals_01inf_grid_prefix(
                Poly::new_from_point(svo_active.as_slice(), EF::ONE).as_slice(),
            );
            let acc_grid = evals_01inf_grid_prefix(acc.as_slice());
            let stride = 3usize.pow(round_idx as u32);

            let acc0 = eq_grid[..stride]
                .iter()
                .zip(acc_grid[..stride].iter())
                .map(|(&eq, &eval)| eq * eval)
                .collect::<Vec<_>>();

            let acc_inf = eq_grid[2 * stride..]
                .iter()
                .zip(acc_grid[2 * stride..].iter())
                .map(|(&eq, &eval)| eq * eval)
                .collect::<Vec<_>>();

            [acc0, acc_inf]
        })
        .collect()
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

    // For small l, use straightline specializations that avoid loop/buffer overhead.
    match l {
        1 => calculate_accumulator_1(eq0.as_slice(), &reduced_evals),
        2 => calculate_accumulator_2(eq0.as_slice(), &reduced_evals),
        3 => calculate_accumulator_3(eq0.as_slice(), &reduced_evals),
        _ => calculate_accumulator_general(l, eq0.as_slice(), &reduced_evals),
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
        let split = SplitEq::new_unpacked(&split, EF::ONE);
        Self {
            z_svo: svo,
            z_split: split,
            var_order,
        }
    }

    /// Splits a challenge point into a prefix SVO portion and a residual suffix.
    ///
    /// `l0` is the number of variables handled by the SVO optimization.
    pub fn new_packed(l0: usize, point: &Point<EF>) -> Self {
        assert!(l0 <= point.num_variables());
        let (svo, split) = point.split_at(l0);
        let split = SplitEq::new_packed(&split, EF::ONE);
        Self {
            z_svo: svo,
            z_split: split,
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
    pub fn eval(&self, poly: &Poly<F>) -> (EF, Vec<Poly<EF>>) {
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
        let eval = compressed.eval_base(&self.z_svo);
        (eval, partial_evals)
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
    use crate::sumcheck::lagrange::lagrange_weights_01inf_multi;
    use crate::sumcheck::layout::Opening;
    use crate::sumcheck::strategy::VariableOrder;

    type F = KoalaBear;
    type EF = BinomialExtensionField<F, 4>;

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
            let claim = ProverMultiClaim::new(svo_point, openings);

            let accumulators = calculate_accumulators_batch(&claim, &alphas);
            if l0 == 0 {
                assert!(accumulators.is_empty());
                continue;
            }

            let mut poly = Poly::<EF>::zero(l0);
            claim
                .openings()
                .iter()
                .zip(alphas.iter())
                .for_each(|(opening, &alpha)| {
                    let full_svo_poly = opening
                        .data()
                        .last()
                        .expect("l0 > 0 guarantees one SVO partial polynomial");
                    poly.as_mut_slice()
                        .iter_mut()
                        .zip(full_svo_poly.iter())
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
            let claim = ProverMultiClaim::new(svo_point, openings);

            let accumulators = calculate_accumulators_batch(&claim, &alphas);
            if l0 == 0 {
                assert!(accumulators.is_empty());
                continue;
            }

            let mut poly = Poly::<EF>::zero(l0);
            claim
                .openings()
                .iter()
                .zip(alphas.iter())
                .for_each(|(opening, &alpha)| {
                    let full_svo_poly = opening
                        .data()
                        .last()
                        .expect("l0 > 0 guarantees one SVO partial polynomial");
                    poly.as_mut_slice()
                        .iter_mut()
                        .zip(full_svo_poly.iter())
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
            assert_eq!(partial_evals.len(), svo_point.num_variables_svo());

            match svo_point.var_order() {
                VariableOrder::Prefix => {
                    partial_evals.iter().enumerate().for_each(|(i, pe0)| {
                        let (_point_lo, point_hi) = point.split_at(i + 1);
                        assert_eq!(pe0, &poly.compress_suffix(&point_hi, EF::ONE));
                        assert_eq!(e0, pe0.eval_base(&svo_point.z_svo().split_at(i + 1).0));
                    });
                }
                VariableOrder::Suffix => {
                    partial_evals.iter().enumerate().for_each(|(i, pe0)| {
                        let (point_lo, point_hi) = point.split_at(point.num_variables() - i - 1);
                        assert_eq!(pe0, &poly.compress_prefix(&point_lo, EF::ONE));
                        assert_eq!(e0, pe0.eval_base(&point_hi));
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
        type F = KoalaBear;
        type EF = BinomialExtensionField<F, 4>;
        type PackedEF = <EF as ExtensionField<F>>::ExtensionPacking;

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
}
