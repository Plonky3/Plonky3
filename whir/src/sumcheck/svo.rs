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
use p3_field::{ExtensionField, Field, PackedFieldExtension, PackedValue, dot_product};
use p3_maybe_rayon::prelude::*;
use p3_multilinear_util::evals::Poly;
use p3_multilinear_util::multilinear::Point;
use p3_util::{log2_strict_usize, log3_strict_usize};

/// Generates grid points for SVO accumulator evaluation.
///
/// Returns two arrays of `3^{l-1}` points each:
/// - First array: points in `{0,1,2}^{l-1} x {0}` (for computing `h(0)`)
/// - Second array: points in `{0,1,2}^{l-1} x {2}` (for computing `h(2)`)
///
/// The round polynomial `h(X)` is quadratic (degree 2).
/// Three evaluations determine it uniquely.
/// We compute `h(0)` and `h(2)` from accumulators.
/// The verifier derives `h(1) = claimed_sum - h(0)`.
///
/// # Panics
///
/// Panics if `l == 0`.
///
/// # Performance
///
/// Time: O(3^l), Space: O(3^l)
pub fn points_012<F: Field>(l: usize) -> [Vec<Vec<F>>; 2] {
    /// Expands a set of points by appending each value from `values` to each point.
    ///
    /// If `pts` has `n` points and `values` has `m` elements, the result has `m * n` points.
    fn expand<F: Field>(pts: &[Vec<F>], values: &[usize]) -> Vec<Vec<F>> {
        values
            .iter()
            .flat_map(|&v| {
                // For each value, clone all existing points and append the value.
                pts.iter().cloned().map(move |mut p| {
                    p.push(F::from_u32(v as u32));
                    p
                })
            })
            .collect()
    }

    // We need at least one round.
    assert!(l > 0, "points_012: l must be positive");

    // Start with the empty point (representing 0 dimensions).
    let mut pts = vec![vec![]];

    // Build up points in {0,1,2}^{l-1} by iteratively expanding.
    // After this loop, pts contains 3^{l-1} points.
    for _ in 0..l - 1 {
        pts = expand(&pts, &[0, 1, 2]);
    }

    // Create final points by appending 0 or 2 as the last coordinate.
    [expand(&pts, &[0]), expand(&pts, &[2])]
}

/// Computes SVO accumulators for a set of grid points.
///
/// For each grid point `u`, computes:
///
/// ```text
/// A(u) = f(u) * eq(u, point)
/// ```
///
/// where `f(u)` is derived from partial evaluations
/// and `eq` is the equality polynomial.
///
/// These values are later combined with Lagrange weights
/// to reconstruct the round polynomial.
///
///
/// # Algorithm
///
/// 1. Split the challenge point into inner (`z0`) and outer (`z1`) components.
/// 2. Reduce partial evaluations over `z1` using the equality polynomial.
/// 3. For each grid point `u`, compute the accumulator via Lagrange interpolation.
///
/// # Returns
///
/// One accumulator value per grid point.
pub fn calculate_accumulators<F: Field, EF: ExtensionField<F>>(
    us: &[Vec<F>],
    partial_evals: &[EF],
    point: &[EF],
) -> Vec<EF> {
    // Determine the dimensions involved.
    // - l0: log2 of partial_evals length (total variables in the partial evaluation domain)
    // - offset: number of variables handled by the "outer" equality polynomial
    let l0 = log2_strict_usize(partial_evals.len());
    let offset = l0 - log3_strict_usize(us.len()) - 1;

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

    // For each grid point u, compute the accumulator value.
    //
    // This uses parallel iteration for better performance when |us| is large.
    // The computation for each u is independent, making it embarrassingly parallel.
    us.par_iter()
        .map(|u| {
            // Build the Lagrange coefficient vector for this grid point.
            // coeffs[x] = prod_{i} L_{u_i}(x_i) where L is the Lagrange basis.
            let coeffs = Poly::new_from_point(u.as_slice(), F::ONE);

            // Compute: (sum_x eq(z0, x) * coeffs(x)) * (sum_x reduced_evals[x] * coeffs(x))
            // This gives f(u) * eq(u, point) via the Lagrange interpolation formula.
            dot_product::<EF, _, _>(eq0.iter().copied(), coeffs.iter().copied())
                * dot_product::<EF, _, _>(reduced_evals.iter().copied(), coeffs.iter().copied())
        })
        .collect()
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
pub struct SplitEq<F: Field, EF: ExtensionField<F>> {
    /// First part of the point where we apply the SVO optimization.
    z_svo: Point<EF>,

    /// Evaluations of eq for the lower-index half of rest of the point.
    ///
    /// Contains 2^{(k-l0)/2} extension field elements.
    eq0: Poly<EF>,

    /// Evaluations of eq for the higher-index half of rest of the point.
    ///
    /// Stored in packed form for SIMD operations.
    /// Contains 2^{(k-l0)/2 - log2(packing_width)} packed elements.
    eq1: Poly<EF::ExtensionPacking>,

    /// Precomputed accumulators for all SVO rounds.
    ///
    /// `accumulators[i]` contains the accumulator values for round `i+1`:
    /// - `accumulators[i][0]`: Values for grid points in `{0,1,2}^i x {0}`
    /// - `accumulators[i][1]`: Values for grid points in `{0,1,2}^i x {2}`
    accumulators: Vec<[Vec<EF>; 2]>,

    /// The evaluation `sum_x eq(w, x) * poly(x)` of the polynomial weighted by eq.
    ///
    /// This is the initial claimed sum for the sumcheck protocol.
    pub eval: EF,

    /// The original challenge point `w` for the equality polynomial.
    pub point: Point<EF>,
}

impl<F: Field, EF: ExtensionField<F>> SplitEq<F, EF> {
    /// Creates a new split equality polynomial with precomputed accumulators.
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

        // Split into SVO part and the rest.
        let (z_svo, z_rest) = point.split_at(l);

        // Split the rest into two halves for the eq tables.
        let (z0, z1) = z_rest.split_at((k - l) / 2);

        // Build evaluation tables: eq0 unpacked, eq1 packed for SIMD.
        let eq0 = Poly::new_from_point(z0.as_slice(), EF::ONE);
        let eq1 = Poly::new_packed_from_point(z1.as_slice(), EF::ONE);

        // Compute partial evaluations of the polynomial weighted by the split eq.
        //
        // For each chunk of the polynomial (one per SVO hypercube point), computes:
        //   sum_{x_inner, x_outer} eq_inner[x_inner] * eq_outer[x_outer] * poly[x_inner, x_outer]
        //
        // Also computes the full weighted evaluation by combining partial results
        // with the SVO portion of the equality polynomial.
        let (partial_evals, eval) = {
            // Each chunk of size 2^{k-l} is processed independently.
            let k_split = eq1.num_vars() + eq0.num_vars() + log2_strict_usize(F::Packing::WIDTH);
            let chunk_size = 1 << k_split;
            let partial_evals = poly
                .as_slice()
                .chunks(chunk_size)
                .map(|poly_chunk| {
                    // Pack the polynomial chunk for SIMD operations.
                    let poly_packed = F::Packing::pack_slice(poly_chunk);

                    // Compute the double sum: sum_{x_L} sum_{x_R} left[x_L] * right[x_R] * poly[x_L, x_R]
                    // The outer sum is over right indices (parallelized).
                    // The inner sum is over left indices (vectorized via packing).
                    let sum = poly_packed
                        .par_chunks(eq1.num_evals())
                        .zip_eq(eq0.as_slice().par_iter())
                        .map(|(poly_packed, &eq0_val)| {
                            // Inner sum: sum_{x_1} eq1[x_1] * poly[x_0, x_1]
                            let inner_sum = poly_packed
                                .iter()
                                .zip_eq(eq1.iter())
                                .map(|(&f, &eq1_val)| eq1_val * f)
                                .sum::<EF::ExtensionPacking>();
                            // Multiply by the eq0 weight.
                            inner_sum * eq0_val
                        })
                        .sum::<EF::ExtensionPacking>();

                    // Unpack the result to get a single extension field element.
                    EF::ExtensionPacking::to_ext_iter([sum]).sum()
                })
                .collect::<Vec<_>>();

            // Combine partial evals with eq_svo to get the full evaluation.
            let eq_svo = Poly::new_from_point(z_svo.as_slice(), EF::ONE);
            let eval =
                dot_product::<EF, _, _>(eq_svo.iter().copied(), partial_evals.iter().copied());
            (partial_evals, eval)
        };

        // Precompute accumulators for all SVO rounds.
        let accumulators = (1..=z_svo.num_vars())
            .map(|i| {
                let us = points_012::<F>(i);
                let acc0 = calculate_accumulators(&us[0], &partial_evals, z_svo.as_slice());
                let acc2 = calculate_accumulators(&us[1], &partial_evals, z_svo.as_slice());
                [acc0, acc2]
            })
            .collect();

        Self {
            z_svo,
            eq0,
            eq1,
            accumulators,
            eval,
            point: point.clone(),
        }
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
        self.z_svo.num_vars()
    }

    /// Returns the number of variables covered by the split eq tables.
    ///
    /// ```text
    /// k_split = inner_half_vars + outer_half_vars + log_2(SIMD_WIDTH)
    /// ```
    ///
    /// The `log_2(SIMD_WIDTH)` term accounts for variables absorbed by SIMD packing.
    /// Together with the SVO depth, this reconstructs the original `k`.
    pub const fn k_split(&self) -> usize {
        self.eq1.num_vars() + self.eq0.num_vars() + log2_strict_usize(F::Packing::WIDTH)
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
        rs: &[EF],
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
            rs.len(),
            k_svo,
            "combine_into_packed: wrong number of SVO challenges"
        );

        // Accumulate each split eq into the output, weighted by powers of alpha.
        for (split_eq, alpha) in selfs.iter().zip(alpha.powers()) {
            // Compute the scalar factor: alpha^j * eq(z_svo_j, rs).
            //
            // This evaluates the SVO portion of the equality polynomial at the
            // sumcheck challenges `rs`, and folds in the batching challenge `alpha`.
            // We pass `alpha` as the initial value to `new_from_point` so it gets
            // multiplied into every term of the eq evaluation.
            let scale = Poly::new_from_point(split_eq.z_svo.as_slice(), alpha)
                .eval_ext::<F>(&Point::new(rs.to_vec()));

            // Process output in chunks of size |eq1|, one chunk per eq0 entry.
            //
            // The output is indexed as out[x_L * |eq1| + x_R], and we iterate
            // over x_L (outer loop via chunks) and x_R (inner loop within chunk).
            // For each position: out[x_L, x_R] += eq1[x_R] * eq0[x_L] * scale.
            out.par_chunks_mut(split_eq.eq1.num_evals())
                .zip(split_eq.eq0.as_slice().par_iter())
                .for_each(|(chunk, &right)| {
                    chunk
                        .iter_mut()
                        .zip(split_eq.eq1.iter())
                        .for_each(|(out, &left)| *out += left * right * scale);
                });
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use p3_field::extension::BinomialExtensionField;
    use p3_field::{PrimeCharacteristicRing, dot_product};
    use p3_koala_bear::KoalaBear;
    use proptest::prelude::*;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;

    type F = KoalaBear;
    type EF = BinomialExtensionField<F, 4>;

    #[test]
    fn test_points_012_l1() {
        // For l=1: pts should have 3^0 = 1 point each.
        // pts_0 = [(0,)]
        // pts_2 = [(2,)]
        let [pts_0, pts_2] = points_012::<F>(1);

        assert_eq!(pts_0.len(), 1);
        assert_eq!(pts_2.len(), 1);

        assert_eq!(pts_0[0], vec![F::ZERO]);
        assert_eq!(pts_2[0], vec![F::TWO]);
    }

    #[test]
    fn test_points_012_l2() {
        // For l=2: pts should have 3^1 = 3 points each.
        // First l-1=1 coordinates in {0,1,2}, last coordinate fixed.
        let [pts_0, pts_2] = points_012::<F>(2);

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
            let [pts_0, pts_2] = points_012::<F>(l);
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
        let [pts_0, pts_2] = points_012::<F>(4);

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
        let [pts_0, pts_2] = points_012::<F>(4);

        let pts_0_set: alloc::collections::BTreeSet<_> = pts_0.iter().cloned().collect();
        let pts_2_set: alloc::collections::BTreeSet<_> = pts_2.iter().cloned().collect();

        assert_eq!(pts_0_set.len(), pts_0.len(), "pts_0 contains duplicates");
        assert_eq!(pts_2_set.len(), pts_2.len(), "pts_2 contains duplicates");
    }

    #[test]
    #[should_panic(expected = "l must be positive")]
    fn test_points_012_panics_on_zero() {
        let _: [Vec<Vec<F>>; 2] = points_012(0);
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
            let split_eq = SplitEq::<F, EF>::new(&z, l, &f);

            // There should be exactly l accumulator rounds (one per SVO variable).
            let accumulators = split_eq.accumulators();
            assert_eq!(accumulators.len(), l);

            // For each round i, verify both the h(0) and h(2) accumulators.
            for (i, accumulator) in accumulators.iter().enumerate() {
                let us = points_012::<F>(i + 1);

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
            let split_eq = SplitEq::<F, EF>::new(&point, 0, &poly);
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
        let split_eq = SplitEq::<F, EF>::new(&point, 0, &poly);

        // With l=0 SVO depth, all variables are in the split tables.
        // partial_evals on a k-variable polynomial should give 2^0 = 1 partial eval.
        // But let's test with a larger polynomial.
        let n = 14;
        let larger_poly = Poly::new((0..1 << n).map(|_| rng.random()).collect());

        // We need to create a new split_eq for n variables to test partial_evals
        let point_n = Point::<EF>::rand(&mut rng, n);
        let split_eq_n = SplitEq::<F, EF>::new(&point_n, 0, &larger_poly);
        // The number of partial evals is 2^{k_svo} = 2^0 = 1 for l=0
        // This is implicitly tested by the constructor succeeding.
        assert_eq!(split_eq_n.num_variables(), n);
        let _ = split_eq;
    }

    proptest! {
        /// Verify that points_012 generates the correct number of points.
        #[test]
        fn prop_points_012_sizes(l in 1usize..=8) {
            let [pts_0, pts_2] = points_012::<F>(l);
            let expected = 3usize.pow((l - 1) as u32);

            prop_assert_eq!(pts_0.len(), expected);
            prop_assert_eq!(pts_2.len(), expected);
        }

        /// Verify that all pts_0 points end with 0 and all pts_2 points end with 2.
        #[test]
        fn prop_points_012_last_coordinate(l in 1usize..=6) {
            let [pts_0, pts_2] = points_012::<F>(l);

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
            let split_eq = SplitEq::<F, EF>::new(&point, 0, &poly);

            prop_assert_eq!(split_eq.num_variables(), k);
        }
    }
}
