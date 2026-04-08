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

/// Expands Boolean-hypercube evaluations onto the full `{0,1,2}^l` grid.
///
/// Given `2^l` evaluations on `{0,1}^l`, produces `3^l` evaluations on `{0,1,2}^l`
/// by linear extrapolation: for each variable, the pair `(f(0), f(1))` is extended
/// to the triple `(f(0), f(1), 2*f(1) - f(0))`.
///
/// The input is ordered with the low variable varying fastest. The output keeps
/// the same convention, so the last variable is the slowest-varying coordinate.
fn evals_012_grid_into<F: Field>(boolean_evals: &[F], output: &mut [F], scratch: &mut [F]) {
    let num_vars = log2_strict_usize(boolean_evals.len());
    let output_len = 3usize.pow(num_vars as u32);

    assert_eq!(output.len(), output_len);
    assert_eq!(scratch.len(), output_len);

    if num_vars == 0 {
        output[0] = boolean_evals[0];
        return;
    }

    let (mut cur, mut next) = if num_vars % 2 == 1 {
        scratch[..boolean_evals.len()].copy_from_slice(boolean_evals);
        (&mut scratch[..], &mut output[..])
    } else {
        output[..boolean_evals.len()].copy_from_slice(boolean_evals);
        (&mut output[..], &mut scratch[..])
    };

    // Parallelization threshold: for early stages with many small blocks,
    // parallelize over blocks; for late stages with few large blocks,
    // parallelize over the inner stride operations.
    const PARALLEL_STRIDE_THRESHOLD: usize = 256;

    for stage in 0..num_vars {
        let in_stride = 3usize.pow(stage as u32);
        let blocks = 1usize << (num_vars - stage - 1);

        // We slice exactly what we need to avoid operating on uninitialized ends.
        let cur_slice = &cur[..blocks * 2 * in_stride];
        let next_slice = &mut next[..blocks * 3 * in_stride];

        if in_stride < PARALLEL_STRIDE_THRESHOLD {
            // Early stages: many blocks, small stride.
            // Parallelize over the outer blocks, keep inner loop vectorizable.
            cur_slice
                .par_chunks(2 * in_stride)
                .zip(next_slice.par_chunks_mut(3 * in_stride))
                .for_each(|(c_chunk, n_chunk)| {
                    for j in 0..in_stride {
                        let f0 = c_chunk[j];
                        let f1 = c_chunk[in_stride + j];
                        n_chunk[3 * j] = f0;
                        n_chunk[3 * j + 1] = f1;
                        n_chunk[3 * j + 2] = f1.double() - f0;
                    }
                });
        } else {
            // Late stages: few blocks (often 1), massive stride.
            // Sequential over blocks, but parallelize the inner stride operations.
            cur_slice
                .chunks(2 * in_stride)
                .zip(next_slice.chunks_mut(3 * in_stride))
                .for_each(|(c_chunk, n_chunk)| {
                    let (c_left, c_right) = c_chunk.split_at(in_stride);
                    c_left
                        .par_iter()
                        .zip(c_right.par_iter())
                        .zip(n_chunk.par_chunks_mut(3))
                        .for_each(|((&f0, &f1), out_triple)| {
                            out_triple[0] = f0;
                            out_triple[1] = f1;
                            out_triple[2] = f1.double() - f0;
                        });
                });
        }

        core::mem::swap(&mut cur, &mut next);
    }
}

/// Computes the SVO accumulators using the Jolt small-grid pattern.
///
/// Rather than rebuilding every Lagrange basis vector independently (O(6^l)),
/// this expands both the residual equality polynomial and the partially compressed
/// multilinear polynomial over the entire `{0,1,2}^l` grid. The required
/// accumulators are then simple pointwise products on the slices with
/// final coordinate fixed to `0` or `2`.
///
/// Total cost: O(3^l) field operations.
fn calculate_accumulators<F: Field, EF: ExtensionField<F>>(
    l: usize,
    partial_evals: &[EF],
    point: &[EF],
) -> [Vec<EF>; 2] {
    let total_vars = log2_strict_usize(partial_evals.len());
    let offset = total_vars - l;
    let (z0, z1) = point.split_at(point.len() - offset);

    // Build equality polynomial evaluation tables for both components.
    let eq0 = Poly::new_from_point(z0, EF::ONE);
    let eq1 = Poly::new_from_point(z1, EF::ONE);

    // Reduce partial evaluations over the outer variables using eq1.
    let reduced_evals: Vec<EF> = partial_evals
        .chunks(eq1.num_evals())
        .map(|chunk| dot_product::<EF, _, _>(eq1.iter().copied(), chunk.iter().copied()))
        .collect();

    // Expand both tables onto the {0,1,2}^l grid.
    let grid_len = 3usize.pow(l as u32);
    let mut eq0_grid = vec![EF::ZERO; grid_len];
    let mut reduced_grid = vec![EF::ZERO; grid_len];
    let mut scratch = vec![EF::ZERO; grid_len];
    evals_012_grid_into(eq0.as_slice(), &mut eq0_grid, &mut scratch);
    evals_012_grid_into(reduced_evals.as_slice(), &mut reduced_grid, &mut scratch);

    // Slice out the accumulator values: pointwise products where the last
    // coordinate is fixed to 0 or 2.
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

        // Precompute accumulators for all SVO rounds using the Jolt small-grid
        // pattern. This expands the relevant multilinear tables onto {0,1,2}^i
        // grids and reads off pointwise products, achieving O(3^l) cost instead
        // of the O(6^l) Lagrange-basis approach.
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
    /// (O(6^l)). Used to verify the Jolt grid-expansion approach.
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
    fn compress_multi_ef(poly: &Poly<EF>, vars: &[EF]) -> Poly<EF> {
        let mut result = poly.clone();
        for &v in vars {
            result.fix_lo_var_mut(v);
        }
        result
    }

    #[test]
    fn test_points_012_l1() {
        let [pts_0, pts_2] = points_012(1);

        assert_eq!(pts_0.len(), 1);
        assert_eq!(pts_2.len(), 1);

        assert_eq!(pts_0[0], vec![F::ZERO]);
        assert_eq!(pts_2[0], vec![F::TWO]);
    }

    #[test]
    fn test_points_012_l2() {
        let [pts_0, pts_2] = points_012(2);

        assert_eq!(pts_0.len(), 3);
        assert_eq!(pts_2.len(), 3);

        for pt in &pts_0 {
            assert_eq!(pt.len(), 2);
            assert_eq!(*pt.last().unwrap(), F::ZERO);
        }
        for pt in &pts_2 {
            assert_eq!(pt.len(), 2);
            assert_eq!(*pt.last().unwrap(), F::TWO);
        }
    }

    #[test]
    fn test_points_012_sizes() {
        for l in 1..=6 {
            let [pts_0, pts_2] = points_012(l);
            let expected_size = 3usize.pow((l - 1) as u32);

            assert_eq!(pts_0.len(), expected_size, "pts_0 size mismatch for l={l}");
            assert_eq!(pts_2.len(), expected_size, "pts_2 size mismatch for l={l}");

            for pt in pts_0.iter().chain(pts_2.iter()) {
                assert_eq!(pt.len(), l, "point dimension mismatch for l={l}");
            }
        }
    }

    #[test]
    fn test_points_012_values_in_range() {
        let [pts_0, pts_2] = points_012(4);

        let valid_values = [F::ZERO, F::ONE, F::TWO];

        for pt in pts_0.iter().chain(pts_2.iter()) {
            for (i, &coord) in pt.iter().enumerate() {
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
                us[0]
                    .iter()
                    .zip(accumulator[0].iter())
                    .for_each(|(u, &acc)| {
                        let u = u.iter().copied().map(EF::from).collect::<Vec<_>>();
                        let f = f.compress_lo(&Point::new(u.clone()), EF::ONE);
                        let eq = compress_multi_ef(&eq, &u);
                        let e1: EF = dot_product(eq.iter().copied(), f.iter().copied());
                        assert_eq!(acc, e1);
                    });

                // Verify h(2) accumulators: grid points ending in 2.
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
    fn test_jolt_accumulators_match_reference() {
        // Verify the Jolt grid-expansion approach matches the reference
        // Lagrange-basis approach for accumulator computation.
        let k = 12;
        let mut rng = SmallRng::seed_from_u64(9);
        let poly = Poly::new((0..1 << k).map(|_| rng.random()).collect());
        let point = Point::<EF>::rand(&mut rng, k);

        let (z_svo, z_rest) = point.split_at(k / 2);
        let split_eq = SplitEq::<F, EF>::new_packed(&z_rest, EF::ONE);
        let partial_evals = split_eq.compress_hi(&poly);

        for l in 1..k / 2 {
            let [jolt_acc0, jolt_acc2] =
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

            assert_eq!(jolt_acc0, ref_acc0, "acc0 mismatch for l={l}");
            assert_eq!(jolt_acc2, ref_acc2, "acc2 mismatch for l={l}");
        }
    }

    #[test]
    fn test_split_eq_k_calculation() {
        // Verify that k() returns the original number of variables.
        let mut rng = SmallRng::seed_from_u64(42);

        for k in [8, 10, 12, 14] {
            let point = Point::<EF>::rand(&mut rng, k);
            let poly = Poly::new((0..1 << k).map(|_| rng.random()).collect());
            let split_eq = SvoClaim::<F, EF>::new(&point, 0, &poly);
            assert_eq!(split_eq.num_variables(), k, "k() mismatch for k={k}");
        }
    }

    #[test]
    fn test_split_eq_partial_evals_size() {
        let mut rng = SmallRng::seed_from_u64(42);
        let k = 10;

        let point = Point::<EF>::rand(&mut rng, k);
        let poly = Poly::new((0..1 << k).map(|_| rng.random()).collect());
        let split_eq = SvoClaim::<F, EF>::new(&point, 0, &poly);

        let n = 14;
        let larger_poly = Poly::new((0..1 << n).map(|_| rng.random()).collect());

        let point_n = Point::<EF>::rand(&mut rng, n);
        let split_eq_n = SvoClaim::<F, EF>::new(&point_n, 0, &larger_poly);
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
            let mut rng = SmallRng::seed_from_u64(num_vars as u64);
            let evals: Vec<EF> = (0..1 << num_vars).map(|_| rng.random()).collect();
            let poly = Poly::new(evals.clone());
            let grid = evals_012_grid(evals.as_slice());
            let total = 3usize.pow(num_vars as u32);

            for idx in 0..total {
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
                prop_assert_eq!(grid[idx], expected);
            }
        }
    }
}
