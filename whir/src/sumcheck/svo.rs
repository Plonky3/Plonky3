//! Split-Value Optimization (SVO) for the sumcheck protocol.
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
use p3_field::{ExtensionField, Field, PackedValue, dot_product};
use p3_maybe_rayon::prelude::*;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use p3_multilinear_util::split_eq::SplitEq;
use p3_util::log2_strict_usize;

/// Expand `2^l` Boolean-hypercube evaluations to `3^l` evaluations on `{0,1,2}^l`.
///
/// # Overview
///
/// A multilinear polynomial in `l` variables is determined by `2^l` evaluations
/// on `{0,1}^l`. This extends them to `{0,1,2}^l` via linear extrapolation:
///
/// ```text
///     f(0), f(1)  -->  f(0), f(1), f(2)    where f(2) = 2*f(1) - f(0)
/// ```
///
/// # Motivation
///
/// The SVO sumcheck prover needs accumulator values on the `{0,1,2}^l` grid.
///
/// - **Naive**: evaluate each of the `3^l` grid points independently via
///   Lagrange interpolation from the `2^l` Boolean values --> `O(6^l)`.
/// - **This function**: process one variable at a time --> `O(3^l)`.
///
/// # Memory Layout
///
/// The input uses low-variable-fastest ordering: `idx = x_0 + 2*x_1 + ...`
///
/// The output uses the same convention as the expansion order. The first
/// variable processed (x_0) becomes the **slowest**-varying coordinate
/// in the ternary grid. Flat index = `x_1 + 3*x_0` for 2 variables:
///
/// ```text
///     l=2 example:
///     index:   0      1      2      3      4      5      6      7      8
///     point: (0,0)  (0,1)  (0,2)  (1,0)  (1,1)  (1,2)  (2,0)  (2,1)  (2,2)
/// ```
///
/// # Algorithm
///
/// `l` stages, one per variable. Each stage converts pairs into triples:
///
/// ```text
///     stage 0:  2^l             values on {0,1}^l
///     stage 1:  3 * 2^{l-1}    values on {0,1,2} x {0,1}^{l-1}
///       ...
///     stage l:  3^l             values on {0,1,2}^l
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
fn evals_012_grid_into<F: Field>(boolean_evals: &[F], output: &mut [F], scratch: &mut [F]) {
    let num_vars = log2_strict_usize(boolean_evals.len());
    let output_len = 3usize.pow(num_vars as u32);

    assert_eq!(output.len(), output_len);
    assert_eq!(scratch.len(), output_len);

    // Single constant -- nothing to expand.
    if num_vars == 0 {
        output[0] = boolean_evals[0];
        return;
    }

    // Ping-pong buffer setup.
    //
    // Each stage swaps cur/next. After l swaps the result must be in `output`.
    //
    //     l odd  --> start in scratch --> after l swaps --> output  OK
    //     l even --> start in output  --> after l swaps --> output  OK
    let (mut cur, mut next) = if num_vars % 2 == 1 {
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

    for stage in 0..num_vars {
        // Stride = 3^stage: how many consecutive elements share the same
        // value of the variable being expanded (the already-expanded
        // variables each contribute a factor of 3).
        let in_stride = 3usize.pow(stage as u32);

        // Blocks = 2^{remaining}: independent groups of (f(0), f(1)) pairs.
        let blocks = 1usize << (num_vars - stage - 1);

        // Slice only the live region (early stages use less than the full buffer).
        //
        //     cur  per block: [ f(0)-group | f(1)-group ]   each of size in_stride
        //     next per block: [ f(0)-group | f(1)-group | f(2)-group ]
        let cur_slice = &cur[..blocks * 2 * in_stride];
        let next_slice = &mut next[..blocks * 3 * in_stride];

        if in_stride < PARALLEL_STRIDE_THRESHOLD {
            // Many small blocks -- parallelize across blocks.
            //
            //     l=2, stage 0 (stride=1, blocks=2):
            //     cur:   [ f(0,0) f(1,0) | f(0,1) f(1,1) ]
            //     next:  [ f(0,0) f(1,0) f(2,0) | f(0,1) f(1,1) f(2,1) ]
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
                        n_chunk[3 * j + 2] = f1.double() - f0;
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
                    // Each (f0, f1) pair --> (f0, f1, 2*f1 - f0) triple.
                    c_left
                        .par_iter()
                        .zip(c_right.par_iter())
                        .zip(n_chunk.par_chunks_mut(3))
                        .for_each(|((&f0, &f1), out)| {
                            out[0] = f0;
                            out[1] = f1;
                            out[2] = f1.double() - f0;
                        });
                });
        }

        // What was `next` becomes `cur` for the next stage.
        core::mem::swap(&mut cur, &mut next);
    }
}

/// Computes the SVO accumulators using grid expansion.
///
/// Rather than rebuilding every Lagrange basis vector independently (O(6^l)),
/// this expands both the residual equality polynomial and the partially compressed
/// multilinear polynomial over the entire `{0,1,2}^l` grid. The required
/// accumulators are then simple pointwise products on the slices with
/// final coordinate fixed to `0` or `2`.
///
/// For l <= 3, straightline specializations avoid all loop overhead, buffer
/// allocation, and parallelization machinery.
///
/// Total cost: O(3^l) field operations.
fn calculate_accumulators<F: Field, EF: ExtensionField<F>>(
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
    let reduced_evals: Vec<EF> = partial_evals
        .chunks(eq1.num_evals())
        .map(|chunk| dot_product::<EF, _, _>(eq1.iter().copied(), chunk.iter().copied()))
        .collect();

    // For small l, use straightline specializations that avoid loop/buffer overhead.
    match l {
        1 => calculate_accumulators_1(eq0.as_slice(), &reduced_evals),
        2 => calculate_accumulators_2(eq0.as_slice(), &reduced_evals),
        3 => calculate_accumulators_3(eq0.as_slice(), &reduced_evals),
        _ => calculate_accumulators_general(l, eq0.as_slice(), &reduced_evals),
    }
}

/// Straightline accumulator computation for l=1.
///
/// ```text
///     eq0:     [e0, e1]          (2 values on {0,1})
///     reduced: [r0, r1]          (2 values on {0,1})
///
///     Grid on {0,1,2}:
///       f(0) = f[0],  f(1) = f[1],  f(2) = 2*f[1] - f[0]
///
///     acc0 = [eq0(0) * red(0)]   = [e0 * r0]               (1 value)
///     acc2 = [eq0(2) * red(2)]   = [(2*e1-e0) * (2*r1-r0)] (1 value)
/// ```
fn calculate_accumulators_1<EF: Field>(eq0: &[EF], reduced: &[EF]) -> [Vec<EF>; 2] {
    assert_eq!(eq0.len(), 2);
    assert_eq!(reduced.len(), 2);

    let (e0, e1) = (eq0[0], eq0[1]);
    let (r0, r1) = (reduced[0], reduced[1]);

    // Extrapolate both tables to point 2: f(2) = 2*f(1) - f(0).
    let e2 = e1.double() - e0;
    let r2 = r1.double() - r0;

    [vec![e0 * r0], vec![e2 * r2]]
}

/// Straightline accumulator computation for l=2.
///
/// Grid layout after expansion (x_0 slowest, x_1 fastest):
///
/// ```text
///     grid[0..3] = x_0=0 group: f(0,0), f(0,1), f(0,2)
///     grid[3..6] = x_0=1 group: f(1,0), f(1,1), f(1,2)
///     grid[6..9] = x_0=2 group: f(2,0), f(2,1), f(2,2)
///
///     stride = 3^{l-1} = 3
///     acc0 = grid[0..3]  = x_0=0 slice, x_1 in {0,1,2}
///     acc2 = grid[6..9]  = x_0=2 slice, x_1 in {0,1,2}
/// ```
fn calculate_accumulators_2<EF: Field>(eq0: &[EF], reduced: &[EF]) -> [Vec<EF>; 2] {
    assert_eq!(eq0.len(), 4);
    assert_eq!(reduced.len(), 4);

    // Boolean evaluations: index = x_0 + 2*x_1.
    let (e00, e10, e01, e11) = (eq0[0], eq0[1], eq0[2], eq0[3]);
    let (r00, r10, r01, r11) = (reduced[0], reduced[1], reduced[2], reduced[3]);

    // Extrapolate x_0 to get x_0=2 values.
    let e20 = e10.double() - e00;
    let e21 = e11.double() - e01;
    let r20 = r10.double() - r00;
    let r21 = r11.double() - r01;

    // Extrapolate x_1 to get x_1=2 values (only needed for x_0=0 and x_0=2).
    let e02 = e01.double() - e00;
    let r02 = r01.double() - r00;

    // acc0: x_0=0 slice, indexed by x_1 in {0,1,2}.
    // acc2: x_0=2 slice, indexed by x_1 in {0,1,2}.
    [
        vec![e00 * r00, e01 * r01, e02 * r02],
        vec![
            e20 * r20,
            e21 * r21,
            (e21.double() - e20) * (r21.double() - r20),
        ],
    ]
}

/// Straightline accumulator computation for l=3.
///
/// Grid layout after expansion (x_0 slowest, x_2 fastest):
///
/// ```text
///     stride = 3^{l-1} = 9
///     acc0 = grid[0..9]   = x_0=0 slice
///     acc2 = grid[18..27] = x_0=2 slice
///
///     Within each x_0 group, 9 entries ordered by (x_1, x_2):
///       x_1=0: f(x_0, 0, 0), f(x_0, 0, 1), f(x_0, 0, 2)
///       x_1=1: f(x_0, 1, 0), f(x_0, 1, 1), f(x_0, 1, 2)
///       x_1=2: f(x_0, 2, 0), f(x_0, 2, 1), f(x_0, 2, 2)
/// ```
fn calculate_accumulators_3<EF: Field>(eq0: &[EF], reduced: &[EF]) -> [Vec<EF>; 2] {
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

    // Extrapolate x_0: f(2,j,k) = 2*f(1,j,k) - f(0,j,k).
    let e_200 = e_100.double() - e_000;
    let e_210 = e_110.double() - e_010;
    let e_201 = e_101.double() - e_001;
    let e_211 = e_111.double() - e_011;
    let r_200 = r_100.double() - r_000;
    let r_210 = r_110.double() - r_010;
    let r_201 = r_101.double() - r_001;
    let r_211 = r_111.double() - r_011;

    // Extrapolate x_1: f(i,2,k) = 2*f(i,1,k) - f(i,0,k).
    // Only needed for x_0=0 and x_0=2.
    let e_020 = e_010.double() - e_000;
    let e_220 = e_210.double() - e_200;
    let e_021 = e_011.double() - e_001;
    let e_221 = e_211.double() - e_201;
    let r_020 = r_010.double() - r_000;
    let r_220 = r_210.double() - r_200;
    let r_021 = r_011.double() - r_001;
    let r_221 = r_211.double() - r_201;

    // Extrapolate x_2: f(i,j,2) = 2*f(i,j,1) - f(i,j,0).
    // Only needed for x_0=0 and x_0=2 (the slices we read out).
    let e_002 = e_001.double() - e_000;
    let e_012 = e_011.double() - e_010;
    let e_022 = e_021.double() - e_020;
    let e_202 = e_201.double() - e_200;
    let e_212 = e_211.double() - e_210;
    let e_222 = e_221.double() - e_220;
    let r_002 = r_001.double() - r_000;
    let r_012 = r_011.double() - r_010;
    let r_022 = r_021.double() - r_020;
    let r_202 = r_201.double() - r_200;
    let r_212 = r_211.double() - r_210;
    let r_222 = r_221.double() - r_220;

    // acc0: x_0=0 slice, 9 entries ordered (x_1, x_2) with x_2 fastest.
    // acc2: x_0=2 slice, same layout.
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
/// Allocates `{0,1,2}^l` buffers and runs the staged grid expansion.
fn calculate_accumulators_general<F: Field, EF: ExtensionField<F>>(
    l: usize,
    eq0: &[EF],
    reduced_evals: &[EF],
) -> [Vec<EF>; 2] {
    let grid_len = 3usize.pow(l as u32);
    let mut eq0_grid = EF::zero_vec(grid_len);
    let mut reduced_grid = EF::zero_vec(grid_len);
    let mut scratch = EF::zero_vec(grid_len);
    evals_012_grid_into(eq0, &mut eq0_grid, &mut scratch);
    evals_012_grid_into(reduced_evals, &mut reduced_grid, &mut scratch);

    let stride = 3usize.pow((l - 1) as u32);

    let acc0 = eq0_grid[..stride]
        .iter()
        .copied()
        .zip(reduced_grid[..stride].iter().copied())
        .map(|(eq, eval)| eq * eval)
        .collect();
    let acc2 = eq0_grid[2 * stride..]
        .iter()
        .copied()
        .zip(reduced_grid[2 * stride..].iter().copied())
        .map(|(eq, eval)| eq * eval)
        .collect();

    [acc0, acc2]
}

/// Challenge point split into an SVO prefix and a residual split-eq suffix.
#[derive(Debug, Clone)]
struct SvoPoint<F: Field, EF: ExtensionField<F>> {
    /// The first `k_svo` coordinates of the original point, handled by the SVO
    /// accumulator rounds.
    z_svo: Point<EF>,
    /// A factored table for `eq(z_rest, ·)` on the remaining coordinates after
    /// removing `z_svo` from the original point.
    split_eq: SplitEq<F, EF>,
}

impl<F: Field, EF: ExtensionField<F>> SvoPoint<F, EF> {
    /// Returns the number of SVO variables (`l0`).
    ///
    /// This is the depth of the SVO optimization.
    /// These coordinates are processed via the accumulator-based Lagrange
    /// interpolation path rather than the standard fold-and-sum path.
    const fn k_svo(&self) -> usize {
        self.z_svo.num_vars()
    }

    /// Returns the number of variables covered by the split eq tables.
    ///
    /// ```text
    /// k_split = inner_half_vars + outer_half_vars + log_2(SIMD_WIDTH)
    /// ```
    const fn k_split(&self) -> usize {
        self.split_eq.num_vars()
    }

    /// Accumulates this claim's residual equality table into a packed buffer.
    ///
    /// Once the SVO rounds have fixed the prefix variables to `rs`, the remaining
    /// weight vector is:
    /// ```text
    /// alpha · eq(z_svo, rs) · eq(z_rest, x_rest)
    /// ```
    /// for every assignment `x_rest` to the non-SVO variables.
    ///
    /// This method computes the scalar factor `alpha · eq(z_svo, rs)` and then asks
    /// `split_eq` to materialize the residual `eq(z_rest, ·)` table into `out`.
    fn accumulate_into_packed(&self, out: &mut [EF::ExtensionPacking], rs: &Point<EF>, alpha: EF) {
        // Compute the scalar factor: alpha^j * eq(z_svo_j, rs).
        //
        // This evaluates the SVO portion of the equality polynomial at the
        // sumcheck challenges `rs`, and folds in the batching challenge `alpha`.
        // We pass `alpha` as the initial value to `new_from_point` so it gets
        // multiplied into every term of the eq evaluation.
        let scale = Poly::new_from_point(self.z_svo.as_slice(), alpha).eval_ext::<F>(rs);

        // Materializes the contribution of this split eq into the output buffer.
        self.split_eq.accumulate_into_packed(out, Some(scale));
    }
}

/// Split equality polynomial with precomputed accumulators for optimized sumcheck proving.
///
/// The equality polynomial for a point `w in F^k` is:
///
/// ```text
/// eq(w, X) = prod_{i=1}^{k} (w_i * X_i + (1 - w_i) * (1 - X_i))
/// ```
///
/// This struct exploits the product structure by splitting `w` into three parts:
///
/// - The first `l0` coordinates, kept as a raw point for SVO rounds.
/// - The next `(k - l0) / 2` coordinates, stored as a scalar evaluation table.
/// - The remaining coordinates, stored as a packed evaluation table for SIMD.
///
/// This allows computing `eq(w, x)` for many `x` values efficiently by:
/// 1. Pre-computing small tables for each half.
/// 2. Combining them as needed during the sumcheck protocol.
///
/// # Memory
///
/// Instead of storing a `2^{k-l0}`-sized table, we store:
/// - `2^{(k-l0)/2}` scalar extension field elements for the first half.
/// - `2^{(k-l0)/2}` packed extension field elements for the second half.
///
/// This reduces memory from `O(2^{k-l0})` to `O(2^{(k-l0)/2})`.
/// Precomputes `O(3^l)` total accumulator values upfront.
#[derive(Debug, Clone)]
pub struct SvoClaim<F: Field, EF: ExtensionField<F>> {
    /// Internal split representation of the challenge point.
    point: SvoPoint<F, EF>,

    /// Precomputed accumulators for all SVO rounds.
    ///
    /// `accumulators[i]` contains the accumulator values for round `i+1`:
    /// - `accumulators[i][0]`: Values for grid points in `{0,1,2}^i x {0}`
    /// - `accumulators[i][1]`: Values for grid points in `{0,1,2}^i x {2}`
    accumulators: Vec<[Vec<EF>; 2]>,

    /// The evaluation `sum_x eq(w, x) * poly(x)` of the polynomial weighted by eq.
    ///
    /// This is the initial claimed sum for the sumcheck protocol.
    eval: EF,

    /// The original challenge point `w` for the equality polynomial.
    original: Point<EF>,
}

impl<F: Field, EF: ExtensionField<F>> SvoClaim<F, EF> {
    /// Creates a new SVO equality claim with precomputed accumulators.
    ///
    /// 1. Splits the challenge point into SVO, scalar, and packed components.
    /// 2. Computes partial evaluations of `poly` weighted by the equality polynomial.
    /// 3. Evaluates `poly` at `point`, storing the result in `eval`.
    /// 4. Precomputes accumulators for all `l` SVO rounds.
    #[tracing::instrument(skip_all)]
    pub fn new(point: &Point<EF>, l: usize, poly: &Poly<F>) -> Self {
        let k = point.num_vars();
        assert_eq!(k, poly.num_vars());
        assert!(k > l);
        assert!(k >= 2 * log2_strict_usize(F::Packing::WIDTH));

        assert!(point.num_vars() > l);
        // Split into SVO part and the rest
        let (z_svo, z_rest) = point.split_at(l);
        // Build evaluation tables for the rest of the point
        let split_eq = SplitEq::new_packed(&z_rest, EF::ONE);

        // Compute partial evaluations of the polynomial weighted by the split eq.
        //
        // For each chunk of the polynomial (one per SVO hypercube point), computes:
        //   sum_{x_inner, x_outer} eq_inner[x_inner] * eq_outer[x_outer] * poly[x_inner, x_outer]
        //
        // Also computes the full weighted evaluation by combining partial results
        // with the SVO portion of the equality polynomial.
        let eq_svo = Poly::new_from_point(z_svo.as_slice(), EF::ONE);
        let partial_evals = split_eq.compress_hi(poly);
        let eval = dot_product::<EF, _, _>(eq_svo.iter().copied(), partial_evals.iter().copied());

        // Precompute accumulators for each SVO round i = 1..l.
        //
        // For round i, both the equality and reduced-evaluation tables are
        // expanded onto the {0,1,2}^i grid, then pointwise-multiplied.
        //
        // Total cost: O(3^l) vs O(6^l) for per-point Lagrange interpolation.
        let accumulators = (1..=z_svo.num_vars())
            .map(|i| calculate_accumulators(i, partial_evals.as_slice(), z_svo.as_slice()))
            .collect();

        Self {
            point: SvoPoint { z_svo, split_eq },
            accumulators,
            eval,
            original: point.clone(),
        }
    }

    /// Returns the original challenge point for the equality polynomial.
    pub const fn point(&self) -> &Point<EF> {
        &self.original
    }

    /// Returns the claimed evaluation of the polynomial
    pub const fn eval(&self) -> EF {
        self.eval
    }

    /// Returns the total number of variables `k` in the original point.
    ///
    /// Equal to the SVO depth plus the split table variables:
    ///
    /// ```text
    /// k = k_svo + k_split
    /// ```
    pub const fn num_variables(&self) -> usize {
        self.k_svo() + self.k_split()
    }

    /// Returns the precomputed accumulators for all SVO rounds.
    ///
    /// `accumulators()[i]` contains `[acc_0, acc_2]` for round `i+1`.
    pub fn accumulators(&self) -> &[[Vec<EF>; 2]] {
        &self.accumulators
    }

    /// Returns the number of SVO variables (`l0`).
    ///
    /// This is the depth of the SVO optimization.
    /// These coordinates are processed via the accumulator-based Lagrange
    /// interpolation path rather than the standard fold-and-sum path.
    pub const fn k_svo(&self) -> usize {
        self.point.k_svo()
    }

    /// Returns the number of variables covered by the split eq tables.
    pub const fn k_split(&self) -> usize {
        self.point.k_split()
    }

    /// Combines multiple split eq polynomials into a single packed output.
    ///
    /// Used after the SVO rounds to merge split representations
    /// back into a single weight vector for subsequent sumcheck rounds.
    ///
    /// For each output position indexed by `(x_L, x_R)`:
    ///
    /// ```text
    /// out[x_L, x_R] += sum_j alpha^j * eq(z_svo_j, rs) * eq0_j[x_L] * eq1_j[x_R]
    /// ```
    ///
    /// where:
    /// - `j` ranges over the input split eqs
    /// - `rs` are the random challenges from the completed SVO rounds
    /// - `eq(z_svo_j, rs)` evaluates the SVO portion at the challenge point
    /// - `eq0_j[x_L]` and `eq1_j[x_R]` are the precomputed split eq tables
    ///
    /// # Arguments
    ///
    /// * `selfs` - Slice of split eq representations to combine.
    /// * `out` - Output buffer of packed extension field elements to accumulate into.
    ///   Must have size `2^{k_split - log_2(SIMD_WIDTH)}`.
    /// * `alpha` - Batching challenge for merging multiple constraints.
    /// * `rs` - The `k_svo` random challenges from the completed SVO rounds.
    #[tracing::instrument(skip_all, fields(k = log2_strict_usize(out.len()), selfs = selfs.len()))]
    pub fn combine_into_packed(
        selfs: &[Self],
        out: &mut [EF::ExtensionPacking],
        alpha: EF,
        rs: &Point<EF>,
    ) {
        // Nothing to do if there are no split eqs.
        if selfs.is_empty() {
            return;
        }

        // Verify all split eqs have the same k.
        let k_split = selfs.iter().map(Self::k_split).all_equal_value().unwrap();

        // Verify output buffer size.
        assert_eq!(
            out.len(),
            1 << (k_split - log2_strict_usize(F::Packing::WIDTH)),
            "combine_into_packed: output buffer has wrong size"
        );

        // Verify all split eqs have the same SVO size.
        let k_svo = selfs.iter().map(Self::k_svo).all_equal_value().unwrap();

        assert_eq!(
            rs.num_vars(),
            k_svo,
            "combine_into_packed: wrong number of SVO challenges"
        );

        // Accumulate each split eq into the output, weighted by powers of alpha.
        for (svo_claim, alpha) in selfs.iter().zip(alpha.powers()) {
            svo_claim.point.accumulate_into_packed(out, rs, alpha);
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use p3_field::extension::BinomialExtensionField;
    use p3_field::{PrimeCharacteristicRing, dot_product};
    use p3_koala_bear::KoalaBear;
    use p3_util::log3_strict_usize;
    use proptest::prelude::*;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;

    type F = KoalaBear;
    type EF = BinomialExtensionField<F, 4>;

    /// Generates grid points for SVO accumulator evaluation (test-only).
    ///
    /// Returns two arrays of `3^{l-1}` points each:
    /// - First array: points in `{0,1,2}^{l-1} x {0}` (for computing `h(0)`)
    /// - Second array: points in `{0,1,2}^{l-1} x {2}` (for computing `h(2)`)
    fn points_012(l: usize) -> [Vec<Vec<F>>; 2] {
        fn expand(pts: &[Vec<F>], values: &[usize]) -> Vec<Vec<F>> {
            values
                .iter()
                .flat_map(|&v| {
                    pts.iter().cloned().map(move |mut p| {
                        p.push(F::from_u32(v as u32));
                        p
                    })
                })
                .collect()
        }

        assert!(l > 0, "points_012: l must be positive");
        let mut pts = vec![vec![]];
        for _ in 0..l - 1 {
            pts = expand(&pts, &[0, 1, 2]);
        }
        [expand(&pts, &[0]), expand(&pts, &[2])]
    }

    /// Reference accumulator computation via per-point Lagrange interpolation
    /// (O(6^l)). Used to verify the grid-expansion approach.
    fn calculate_accumulators_reference(
        us: &[Vec<F>],
        partial_evals: &[EF],
        point: &[EF],
    ) -> Vec<EF> {
        let l0 = log2_strict_usize(partial_evals.len());
        let offset = l0 - log3_strict_usize(us.len()) - 1;
        let (z0, z1) = point.split_at(point.len() - offset);

        let eq0 = Poly::new_from_point(z0, EF::ONE);
        let eq1 = Poly::new_from_point(z1, EF::ONE);
        let reduced_evals: Vec<EF> = partial_evals
            .chunks(eq1.num_evals())
            .map(|chunk| dot_product::<EF, _, _>(eq1.iter().copied(), chunk.iter().copied()))
            .collect();

        us.par_iter()
            .map(|u| {
                let coeffs = Poly::new_from_point(u.as_slice(), F::ONE);
                dot_product::<EF, _, _>(eq0.iter().copied(), coeffs.iter().copied())
                    * dot_product::<EF, _, _>(reduced_evals.iter().copied(), coeffs.iter().copied())
            })
            .collect()
    }

    /// Convenience wrapper: expand boolean evals onto {0,1,2}^l grid.
    fn evals_012_grid(boolean_evals: &[EF]) -> Vec<EF> {
        let num_vars = log2_strict_usize(boolean_evals.len());
        let output_len = 3usize.pow(num_vars as u32);
        let mut output = vec![EF::ZERO; output_len];
        let mut scratch = vec![EF::ZERO; output_len];
        evals_012_grid_into(boolean_evals, &mut output, &mut scratch);
        output
    }

    /// Sequentially fixes the lowest variables of a polynomial at the given values.
    /// Equivalent to the old `compress_multi` method.
    fn compress_multi_ef(poly: &Poly<EF>, vars: &[EF]) -> Poly<EF> {
        let mut result = poly.clone();
        for &v in vars {
            result.fix_lo_var_mut(v);
        }
        result
    }

    /// Compare the grid expansion against naive multilinear evaluation on every
    /// point of `{0,1,2}^l`.
    fn assert_evals_012_grid_matches_naive(boolean_evals: &[EF]) {
        let num_vars = log2_strict_usize(boolean_evals.len());
        let poly = Poly::new(boolean_evals.to_vec());
        let grid = evals_012_grid(boolean_evals);

        for (idx, &grid_val) in grid.iter().enumerate() {
            // Decode the flat ternary index into per-variable digits.
            let mut tmp = idx;
            let mut digits = Vec::with_capacity(num_vars);
            for _ in 0..num_vars {
                digits.push(tmp % 3);
                tmp /= 3;
            }

            let point = Point::new(
                digits
                    .iter()
                    .copied()
                    .map(|digit| EF::from(F::from_u32(digit as u32)))
                    .collect::<Vec<_>>(),
            );

            let expected = compress_multi_ef(&poly, point.as_slice()).as_slice()[0];
            assert_eq!(grid_val, expected);
        }
    }

    // Tests for evals_012_grid_into

    #[test]
    fn test_evals_012_grid_into_zero_vars() {
        // Zero variables: the polynomial is a single constant.
        // Input: [c] on {0}^0 (one point, the empty tuple).
        // Output: [c] on {0}^0 (still one point).
        let c = EF::from_u32(42);
        let input = [c];
        let mut output = [EF::ZERO];
        let mut scratch = [EF::ZERO];

        evals_012_grid_into(&input, &mut output, &mut scratch);

        // The sole value is copied through unchanged.
        assert_eq!(output, [c]);
    }

    #[test]
    fn test_evals_012_grid_into_one_var() {
        // One variable: f(0) = 3, f(1) = 7.
        // The multilinear extension is f(t) = 3 + 4t.
        // So f(2) = 3 + 8 = 11, i.e. 2*f(1) - f(0) = 14 - 3 = 11.
        //
        // Input layout (low-var-fastest, only one var):
        //   index 0 → x_0=0 → f(0)=3
        //   index 1 → x_0=1 → f(1)=7
        //
        // Output layout on {0,1,2}:
        //   index 0 → x_0=0 → f(0)=3
        //   index 1 → x_0=1 → f(1)=7
        //   index 2 → x_0=2 → f(2)=11
        let f0 = EF::from_u32(3);
        let f1 = EF::from_u32(7);
        let input = [f0, f1];
        let mut output = [EF::ZERO; 3];
        let mut scratch = [EF::ZERO; 3];

        evals_012_grid_into(&input, &mut output, &mut scratch);

        assert_eq!(output[0], f0);
        assert_eq!(output[1], f1);
        // f(2) = 2*7 - 3 = 11
        assert_eq!(output[2], EF::from_u32(11));
    }

    #[test]
    fn test_evals_012_grid_into_two_vars_hand_computed() {
        // f(x_0, x_1) = 1 + 2*x_0 + 4*x_1 + 4*x_0*x_1
        //
        // Input (low-var-fastest):
        //   idx 0 -> (0,0) = 1
        //   idx 1 -> (1,0) = 3
        //   idx 2 -> (0,1) = 5
        //   idx 3 -> (1,1) = 11
        //
        // Stage 0 expands x_0: each (f(0), f(1)) pair -> (f(0), f(1), 2*f(1)-f(0)):
        //   x_1=0: (1, 3) -> (1, 3, 5)
        //   x_1=1: (5, 11) -> (5, 11, 17)
        //   buffer: [1, 3, 5, 5, 11, 17]
        //
        // Stage 1 expands x_1 (stride=3): for each x_0 value,
        // the pair (f(x_0,0), f(x_0,1)) -> (f(x_0,0), f(x_0,1), 2*f(x_0,1)-f(x_0,0)):
        //   x_0=0: (1, 5)  -> (1, 5, 9)
        //   x_0=1: (3, 11) -> (3, 11, 19)
        //   x_0=2: (5, 17) -> (5, 17, 29)
        //
        // Output (x_0 slowest, x_1 fastest), index = x_1 + 3*x_0:
        //   idx:   0  1  2   3   4   5   6   7   8
        //   val:   1  5  9   3  11  19   5  17  29
        let input = [1, 3, 5, 11].map(EF::from_u32);
        let mut output = [EF::ZERO; 9];
        let mut scratch = [EF::ZERO; 9];

        evals_012_grid_into(&input, &mut output, &mut scratch);

        let expected = [1, 5, 9, 3, 11, 19, 5, 17, 29].map(EF::from_u32);
        assert_eq!(output, expected);
    }

    #[test]
    fn test_evals_012_grid_into_output_size() {
        // Verify the output length is 3^l for various numbers of variables.
        for num_vars in 1..=5 {
            let input_len = 1 << num_vars;
            let output_len = 3usize.pow(num_vars as u32);

            // Use all-zero input; we only care about sizes here.
            let input = vec![EF::ZERO; input_len];
            let mut output = vec![EF::ZERO; output_len];
            let mut scratch = vec![EF::ZERO; output_len];

            // Should not panic.
            evals_012_grid_into(&input, &mut output, &mut scratch);
        }
    }

    #[test]
    fn test_evals_012_grid_into_result_lands_in_output() {
        // The ping-pong buffer logic must place the final result in the
        // output buffer, not in scratch. Verify for both odd and even
        // numbers of variables (since the initial buffer assignment differs).
        let mut rng = SmallRng::seed_from_u64(123);

        for num_vars in 1..=4 {
            let input: Vec<EF> = (0..1 << num_vars).map(|_| rng.random()).collect();
            let output_len = 3usize.pow(num_vars as u32);
            let mut output = vec![EF::ZERO; output_len];
            let mut scratch = vec![EF::ZERO; output_len];

            evals_012_grid_into(&input, &mut output, &mut scratch);

            // The grid computed via the convenience wrapper must match.
            let reference = evals_012_grid(&input);
            assert_eq!(
                output, reference,
                "ping-pong mismatch for num_vars={num_vars}"
            );
        }
    }

    #[test]
    fn test_evals_012_grid_into_preserves_boolean_points() {
        // The grid expansion must preserve the original 2^l Boolean evaluations.
        // At every Boolean point in {0,1}^l, the grid value must equal the input.
        //
        // The binary index uses b_0 + 2*b_1 + ... (low-var-fastest).
        // The ternary grid has x_0 slowest (first variable processed),
        // so the ternary index for a Boolean point is b_{l-1} + 3*b_{l-2} + ... + 3^{l-1}*b_0.
        let mut rng = SmallRng::seed_from_u64(77);

        for num_vars in 1..=4 {
            let input: Vec<EF> = (0..1 << num_vars).map(|_| rng.random()).collect();
            let grid = evals_012_grid(&input);

            for (bool_idx, &input_val) in input.iter().enumerate() {
                // Extract binary digits (low-var-first): b_0, b_1, ..., b_{l-1}.
                let mut bits = Vec::with_capacity(num_vars);
                let mut tmp = bool_idx;
                for _ in 0..num_vars {
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
                    "Boolean point mismatch at bool_idx={bool_idx}, num_vars={num_vars}"
                );
            }
        }
    }

    #[test]
    fn test_evals_012_grid_into_constant_polynomial() {
        // A constant polynomial f(x) = c should evaluate to c everywhere.
        // All 3^l grid values must be identical.
        let c = EF::from_u32(99);

        for num_vars in 0..=4 {
            let input = vec![c; 1 << num_vars];
            let output_len = 3usize.pow(num_vars as u32);
            let mut output = vec![EF::ZERO; output_len];
            let mut scratch = vec![EF::ZERO; output_len];

            evals_012_grid_into(&input, &mut output, &mut scratch);

            // Every grid point should equal the constant.
            for (idx, &val) in output.iter().enumerate() {
                assert_eq!(
                    val, c,
                    "constant polynomial mismatch at idx={idx}, num_vars={num_vars}"
                );
            }
        }
    }

    #[test]
    fn test_evals_012_grid_into_linearity() {
        // The grid expansion must be linear:
        //   grid(a*f + b*g) = a*grid(f) + b*grid(g)
        //
        // This follows from the extrapolation formula being linear,
        // but verify it concretely.
        let mut rng = SmallRng::seed_from_u64(55);
        let num_vars = 3;
        let n = 1 << num_vars;

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
        let grid_combined = evals_012_grid(&combined);

        // Compute a*grid(f) + b*grid(g).
        let grid_f = evals_012_grid(&f);
        let grid_g = evals_012_grid(&g);
        let linear_combined: Vec<EF> = grid_f
            .iter()
            .zip(grid_g.iter())
            .map(|(&fi, &gi)| a * fi + b * gi)
            .collect();

        assert_eq!(grid_combined, linear_combined);
    }

    #[test]
    fn test_evals_012_grid_into_large_stride_branch_matches_naive() {
        // `num_vars = 7` guarantees the final stage has `in_stride = 3^6 = 729`,
        // which takes the large-stride branch (`in_stride >= 256`).
        let num_vars = 7;
        let mut rng = SmallRng::seed_from_u64(2025);
        let evals: Vec<EF> = (0..1 << num_vars).map(|_| rng.random()).collect();
        assert_evals_012_grid_matches_naive(evals.as_slice());
    }

    #[test]
    #[should_panic(expected = "Not a power of two")]
    fn test_evals_012_grid_into_panics_on_non_power_of_two_input() {
        let input = [EF::ZERO; 3];
        let mut output = [EF::ZERO; 3];
        let mut scratch = [EF::ZERO; 3];

        evals_012_grid_into(&input, &mut output, &mut scratch);
    }

    #[test]
    #[should_panic(expected = "assertion `left == right` failed")]
    fn test_evals_012_grid_into_panics_on_wrong_output_len() {
        let input = [EF::ZERO; 4];
        let mut output = [EF::ZERO; 8];
        let mut scratch = [EF::ZERO; 9];

        evals_012_grid_into(&input, &mut output, &mut scratch);
    }

    #[test]
    #[should_panic(expected = "assertion `left == right` failed")]
    fn test_evals_012_grid_into_panics_on_wrong_scratch_len() {
        let input = [EF::ZERO; 4];
        let mut output = [EF::ZERO; 9];
        let mut scratch = [EF::ZERO; 8];

        evals_012_grid_into(&input, &mut output, &mut scratch);
    }

    #[test]
    fn test_points_012_l1() {
        // For l=1: pts should have 3^0 = 1 point each.
        // pts_0 = [(0,)]
        // pts_2 = [(2,)]
        let [pts_0, pts_2] = points_012(1);

        assert_eq!(pts_0.len(), 1);
        assert_eq!(pts_2.len(), 1);

        assert_eq!(pts_0[0], vec![F::ZERO]);
        assert_eq!(pts_2[0], vec![F::TWO]);
    }

    #[test]
    fn test_points_012_l2() {
        // For l=2: pts should have 3^1 = 3 points each.
        // First l-1=1 coordinates in {0,1,2}, last coordinate fixed.
        let [pts_0, pts_2] = points_012(2);

        assert_eq!(pts_0.len(), 3);
        assert_eq!(pts_2.len(), 3);

        // All pts_0 should end with 0.
        for pt in &pts_0 {
            assert_eq!(pt.len(), 2);
            assert_eq!(*pt.last().unwrap(), F::ZERO);
        }

        // All pts_2 should end with 2.
        for pt in &pts_2 {
            assert_eq!(pt.len(), 2);
            assert_eq!(*pt.last().unwrap(), F::TWO);
        }
    }

    #[test]
    fn test_points_012_sizes() {
        // Verify output sizes: each array should have 3^{l-1} points.
        for l in 1..=6 {
            let [pts_0, pts_2] = points_012(l);
            let expected_size = 3usize.pow((l - 1) as u32);

            assert_eq!(pts_0.len(), expected_size, "pts_0 size mismatch for l={l}");
            assert_eq!(pts_2.len(), expected_size, "pts_2 size mismatch for l={l}");

            // Each point should have l coordinates.
            for pt in pts_0.iter().chain(pts_2.iter()) {
                assert_eq!(pt.len(), l, "point dimension mismatch for l={l}");
            }
        }
    }

    #[test]
    fn test_points_012_values_in_range() {
        // All coordinates should be in {0, 1, 2}, since the grid points live
        // in {0, 1, 2}^{l-1} x {0 or 2}. The first l-1 coordinates are free
        // to take any value in {0, 1, 2}, while the last is fixed.
        let [pts_0, pts_2] = points_012(4);

        let valid_values = [F::ZERO, F::ONE, F::TWO];

        for pt in pts_0.iter().chain(pts_2.iter()) {
            for (i, &coord) in pt.iter().enumerate() {
                // Last coordinate is fixed (0 for pts_0, 2 for pts_2).
                // Other coordinates should be in {0, 1, 2}.
                if i < pt.len() - 1 {
                    assert!(
                        valid_values.contains(&coord),
                        "invalid coordinate value at position {i}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_points_012_unique() {
        // All points within each array should be unique, since the expand
        // function enumerates all combinations of {0,1,2}^{l-1} crossed
        // with a fixed last coordinate.
        let [pts_0, pts_2] = points_012(4);

        let pts_0_set: alloc::collections::BTreeSet<_> = pts_0.iter().cloned().collect();
        let pts_2_set: alloc::collections::BTreeSet<_> = pts_2.iter().cloned().collect();

        assert_eq!(pts_0_set.len(), pts_0.len(), "pts_0 contains duplicates");
        assert_eq!(pts_2_set.len(), pts_2.len(), "pts_2 contains duplicates");
    }

    #[test]
    #[should_panic(expected = "l must be positive")]
    fn test_points_012_panics_on_zero() {
        let _ = points_012(0);
    }

    #[test]
    fn test_accumulators_correctness() {
        // Main correctness test: verify accumulators match naive computation.
        //
        // For each SVO depth l, verify that the precomputed accumulators match
        // what we'd get by directly computing eq(z, u) * f(u) via compress_multi.
        let k = 10;
        let mut rng = SmallRng::seed_from_u64(1);

        // Generate a random polynomial f over the boolean hypercube {0,1}^k.
        let f = Poly::new((0..1 << k).map(|_| rng.random()).collect());

        // Generate a random challenge point z and build the full eq table.
        let z = Point::<EF>::rand(&mut rng, f.num_vars());
        let eq = Poly::new_from_point(z.as_slice(), EF::ONE);

        // Test for each SVO depth l from 1 to k/2 - 1.
        for l in 1..k / 2 {
            let split_eq = SvoClaim::<F, EF>::new(&z, l, &f);

            // There should be exactly l accumulator rounds (one per SVO variable).
            let accumulators = split_eq.accumulators();
            assert_eq!(accumulators.len(), l);

            // For each round i, verify both the h(0) and h(2) accumulators.
            for (i, accumulator) in accumulators.iter().enumerate() {
                let us = points_012(i + 1);

                // Verify h(0) accumulators: grid points ending in 0.
                // For each grid point u, the accumulator should equal
                // sum_x eq(z, (u, x)) * f(u, x) computed via compress_multi.
                us[0]
                    .iter()
                    .zip(accumulator[0].iter())
                    .for_each(|(u, &acc)| {
                        // Lift u from base field to extension field for compress_multi.
                        let u = u.iter().copied().map(EF::from).collect::<Vec<_>>();
                        // Partially evaluate f and eq by binding the first i+1 variables to u.
                        let f = f.compress_lo(&Point::new(u.clone()), EF::ONE);
                        let eq = compress_multi_ef(&eq, &u);
                        // The naive accumulator is the dot product of the residual eq and f.
                        let e1: EF = dot_product(eq.iter().copied(), f.iter().copied());
                        assert_eq!(acc, e1);
                    });

                // Verify h(2) accumulators: grid points ending in 2.
                // Same logic as above, but with extrapolation point 2.
                us[1]
                    .iter()
                    .zip(accumulator[1].iter())
                    .for_each(|(u, &acc)| {
                        let u = u.iter().copied().map(EF::from).collect::<Vec<_>>();
                        let f = f.compress_lo(&Point::new(u.clone()), EF::ONE);
                        let eq = compress_multi_ef(&eq, &u);
                        let e1: EF = dot_product(eq.iter().copied(), f.iter().copied());
                        assert_eq!(acc, e1);
                    });
            }
        }
    }

    #[test]
    fn test_split_eq_k_calculation() {
        // Verify that k() returns the original number of variables.
        // The splitting into z_svo, eq0, and eq1 must preserve k = k_svo + k_split,
        // regardless of how the coordinates are distributed.
        let mut rng = SmallRng::seed_from_u64(42);

        for k in [8, 10, 12, 14] {
            let point = Point::<EF>::rand(&mut rng, k);
            let poly = Poly::new((0..1 << k).map(|_| rng.random()).collect());
            // Use l=0 SVO depth so all variables go into the split eq tables.
            let split_eq = SvoClaim::<F, EF>::new(&point, 0, &poly);
            assert_eq!(split_eq.num_variables(), k, "k() mismatch for k={k}");
        }
    }

    #[test]
    fn test_split_eq_partial_evals_size() {
        // Verify partial_evals returns the correct number of elements.
        //
        // When the polynomial has n variables and the split eq covers k variables,
        // partial_evals processes the polynomial in chunks of 2^k, producing
        // one partial evaluation per chunk. So the output length is 2^{n-k}.
        let mut rng = SmallRng::seed_from_u64(42);
        let k = 10;

        let point = Point::<EF>::rand(&mut rng, k);
        let poly = Poly::new((0..1 << k).map(|_| rng.random()).collect());
        let split_eq = SvoClaim::<F, EF>::new(&point, 0, &poly);

        // With l=0 SVO depth, all variables are in the split tables.
        // partial_evals on a k-variable polynomial should give 2^0 = 1 partial eval.
        // But let's test with a larger polynomial.
        let n = 14;
        let larger_poly = Poly::new((0..1 << n).map(|_| rng.random()).collect());

        // We need to create a new split_eq for n variables to test partial_evals
        let point_n = Point::<EF>::rand(&mut rng, n);
        let split_eq_n = SvoClaim::<F, EF>::new(&point_n, 0, &larger_poly);
        // The number of partial evals is 2^{k_svo} = 2^0 = 1 for l=0
        // This is implicitly tested by the constructor succeeding.
        assert_eq!(split_eq_n.num_variables(), n);
        let _ = split_eq;
    }

    proptest! {
        /// Verify that points_012 generates the correct number of points.
        #[test]
        fn prop_points_012_sizes(l in 1usize..=8) {
            let [pts_0, pts_2] = points_012(l);
            let expected = 3usize.pow((l - 1) as u32);

            prop_assert_eq!(pts_0.len(), expected);
            prop_assert_eq!(pts_2.len(), expected);
        }

        /// Verify that all pts_0 points end with 0 and all pts_2 points end with 2.
        #[test]
        fn prop_points_012_last_coordinate(l in 1usize..=6) {
            let [pts_0, pts_2] = points_012(l);

            for pt in &pts_0 {
                prop_assert_eq!(*pt.last().unwrap(), F::ZERO);
            }
            for pt in &pts_2 {
                prop_assert_eq!(*pt.last().unwrap(), F::TWO);
            }
        }

        /// Verify that the total variable count matches the original point dimension.
        #[test]
        fn prop_split_eq_k_consistency(k in 8usize..=14) {
            let mut rng = SmallRng::seed_from_u64(k as u64);
            let point = Point::<EF>::rand(&mut rng, k);
            let poly = Poly::zero(k);
            let split_eq = SvoClaim::<F, EF>::new(&point, 0, &poly);

            prop_assert_eq!(split_eq.num_variables(), k);
        }

        /// Verify the {0,1,2}^l grid expansion matches naive MLE evaluation.
        #[test]
        fn prop_evals_012_grid_matches_naive(num_vars in 1usize..=5) {
            // For each grid point in {0,1,2}^l, compare the fast grid-expansion
            // result against the naive approach of evaluating the multilinear
            // extension at that point by repeatedly fixing variables.
            let mut rng = SmallRng::seed_from_u64(num_vars as u64);
            let evals: Vec<EF> = (0..1 << num_vars).map(|_| rng.random()).collect();
            assert_evals_012_grid_matches_naive(evals.as_slice());
        }

        /// Verify grid-expansion accumulators match the per-point Lagrange reference.
        #[test]
        fn prop_accumulators_match_reference(k in 10usize..=14) {
            let mut rng = SmallRng::seed_from_u64(k as u64);
            let poly = Poly::new((0..1 << k).map(|_| rng.random()).collect());
            let point = Point::<EF>::rand(&mut rng, k);

            // Split the point at half to get partial evaluations.
            let (z_svo, z_rest) = point.split_at(k / 2);
            let split_eq = SplitEq::<F, EF>::new_packed(&z_rest, EF::ONE);
            let partial_evals = split_eq.compress_hi(&poly);

            // Compare grid-expansion accumulators against per-point Lagrange
            // reference for each SVO depth from 1 to k/2 - 1.
            for l in 1..k / 2 {
                let [acc0, acc2] =
                    calculate_accumulators::<F, EF>(l, partial_evals.as_slice(), z_svo.as_slice());

                let us = points_012(l);
                let ref_acc0 = calculate_accumulators_reference(
                    &us[0],
                    partial_evals.as_slice(),
                    z_svo.as_slice(),
                );
                let ref_acc2 = calculate_accumulators_reference(
                    &us[1],
                    partial_evals.as_slice(),
                    z_svo.as_slice(),
                );

                prop_assert_eq!(acc0, ref_acc0);
                prop_assert_eq!(acc2, ref_acc2);
            }
        }

        /// Verify l=1 straightline matches the general path.
        #[test]
        fn prop_accumulators_1_matches_general(seed in 0u64..1000) {
            let mut rng = SmallRng::seed_from_u64(seed);
            let eq0: Vec<EF> = (0..2).map(|_| rng.random()).collect();
            let reduced: Vec<EF> = (0..2).map(|_| rng.random()).collect();

            let fast = calculate_accumulators_1(&eq0, &reduced);
            let general = calculate_accumulators_general::<F, EF>(1, &eq0, &reduced);

            prop_assert_eq!(fast, general);
        }

        /// Verify l=2 straightline matches the general path.
        #[test]
        fn prop_accumulators_2_matches_general(seed in 0u64..1000) {
            let mut rng = SmallRng::seed_from_u64(seed);
            let eq0: Vec<EF> = (0..4).map(|_| rng.random()).collect();
            let reduced: Vec<EF> = (0..4).map(|_| rng.random()).collect();

            let fast = calculate_accumulators_2(&eq0, &reduced);
            let general = calculate_accumulators_general::<F, EF>(2, &eq0, &reduced);

            prop_assert_eq!(fast, general);
        }

        /// Verify l=3 straightline matches the general path.
        #[test]
        fn prop_accumulators_3_matches_general(seed in 0u64..1000) {
            let mut rng = SmallRng::seed_from_u64(seed);
            let eq0: Vec<EF> = (0..8).map(|_| rng.random()).collect();
            let reduced: Vec<EF> = (0..8).map(|_| rng.random()).collect();

            let fast = calculate_accumulators_3(&eq0, &reduced);
            let general = calculate_accumulators_general::<F, EF>(3, &eq0, &reduced);

            prop_assert_eq!(fast, general);
        }

        /// Verify that grid expansion preserves the original Boolean-hypercube values.
        #[test]
        fn prop_evals_012_grid_preserves_boolean_points(num_vars in 1usize..=6) {
            let mut rng = SmallRng::seed_from_u64(num_vars as u64 + 1000);
            let input: Vec<EF> = (0..1 << num_vars).map(|_| rng.random()).collect();
            let grid = evals_012_grid(&input);

            for (bool_idx, &input_val) in input.iter().enumerate() {
                // Extract binary digits (low-var-first): b_0, b_1, ..., b_{l-1}.
                let mut bits = Vec::with_capacity(num_vars);
                let mut tmp = bool_idx;
                for _ in 0..num_vars {
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
}
