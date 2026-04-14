//! Barycentric Lagrange interpolation over two-adic cosets.
//!
//! Evaluates polynomials at out-of-domain points given their evaluations
//! over a power-of-two multiplicative coset.
//!
//! # Mathematical background
//!
//! Given evaluations of a polynomial f over a coset g * H of size N = 2^k,
//! the barycentric formula recovers f(z) for any z outside the coset:
//!
//! ```text
//!   f(z) = (z^N - g^N) / (N * g^N)  *  sum_i  g*h^i / (z - g*h^i)  *  f(g*h^i)
//! ```
//!
//! The per-element weight g*h^i / (z - g*h^i) normally requires N
//! extension-by-base multiplications. An algebraic identity eliminates them:
//!
//! ```text
//!   g*h^i / (z - g*h^i)  =  z * ( 1/(z - g*h^i) - 1/z )
//! ```
//!
//! So we define **adjusted weights** as the parenthesized difference, absorb the
//! extra factor of z into the global scalar, and feed the adjusted weights
//! straight into the SIMD-optimized dot product — zero per-element coset work.

use alloc::vec::Vec;

use p3_field::coset::TwoAdicMultiplicativeCoset;
use p3_field::{
    ExtensionField, Field, TwoAdicField, batch_multiplicative_inverse,
    scale_slice_in_place_single_core,
};
use p3_maybe_rayon::prelude::*;
use p3_util::log2_strict_usize;

use crate::Matrix;

/// Subtract z^{-1} from each inverse denominator to produce adjusted barycentric weights.
///
/// # Overview
///
/// Converts raw 1/(z - x_i) values into the form needed by the
/// zero-allocation interpolation path.
///
/// Intended to be called once per opening point z, then reused across
/// every matrix opened at that point.
///
/// # Performance
///
/// One extension-field inversion + N parallel extension-field subtractions.
pub fn compute_adjusted_weights<EF: Field>(point: EF, diff_invs: &[EF]) -> Vec<EF> {
    // Single inversion of z, amortised over all N weights.
    let point_inv = point.inverse();
    // Subtract z^{-1} from each 1/(z - x_i) in parallel.
    diff_invs.par_iter().map(|&d| d - point_inv).collect()
}

/// Barycentric Lagrange interpolation over two-adic cosets.
///
/// Blanket-implemented for every matrix over a two-adic field.
/// Import the trait, then call the methods directly on any matrix.
pub trait Interpolate<F: TwoAdicField>: Matrix<F> {
    /// Evaluate a batch of polynomials at a point outside the canonical subgroup.
    ///
    /// Convenience wrapper that uses shift = 1.
    ///
    /// # Safety
    ///
    /// The evaluation point must not lie in the subgroup.
    fn interpolate_subgroup<EF: ExtensionField<F>>(&self, point: EF) -> Vec<EF> {
        // Canonical subgroup has unit shift.
        self.interpolate_coset(F::ONE, point)
    }

    /// Evaluate a batch of polynomials at a point outside a shifted coset.
    ///
    /// Builds the coset, batch-inverts the denominators, converts to adjusted
    /// weights, and evaluates — all in one call.
    ///
    /// Evaluations must be in standard (not bit-reversed) order.
    ///
    /// # Safety
    ///
    /// The evaluation point must not lie in the coset.
    fn interpolate_coset<EF: ExtensionField<F>>(&self, shift: F, point: EF) -> Vec<EF> {
        let log_height = log2_strict_usize(self.height());

        // Materialise the coset so the diff computation can use parallel iteration.
        let coset: Vec<F> = TwoAdicMultiplicativeCoset::new(shift, log_height)
            .unwrap()
            .iter()
            .collect();

        // Compute z - x_i in parallel, then batch-invert in one shot
        // (Montgomery's trick: single field inversion + O(N) multiplications).
        let diffs: Vec<EF> = coset.par_iter().map(|&g| point - g).collect();
        let diff_invs = batch_multiplicative_inverse(&diffs);

        // Convert to adjusted weights and delegate to the zero-allocation hot path.
        let adjusted = compute_adjusted_weights(point, &diff_invs);
        self.interpolate_coset_with_adjusted_weights(shift, point, &adjusted)
    }

    /// Evaluate a batch of polynomials using precomputed inverse denominators.
    ///
    /// Accepts 1/(z - x_i) values computed by the caller. Converts them to
    /// adjusted weights internally, then delegates to the zero-allocation path.
    ///
    /// Prefer the adjusted-weights variant when the same denominators are reused
    /// across multiple matrices — it avoids re-subtracting z^{-1} on every call.
    ///
    /// # Safety
    ///
    /// - The evaluation point must not lie in the coset.
    /// - Coset and inverse-denominator slices must be consistent with row ordering.
    fn interpolate_coset_with_precomputation<EF: ExtensionField<F>>(
        &self,
        shift: F,
        point: EF,
        coset: &[F],
        diff_invs: &[EF],
    ) -> Vec<EF> {
        debug_assert_eq!(coset.len(), diff_invs.len());
        debug_assert_eq!(coset.len(), self.height());

        // Convert 1/(z - x_i) to adjusted form and delegate.
        let adjusted = compute_adjusted_weights(point, diff_invs);
        self.interpolate_coset_with_adjusted_weights(shift, point, &adjusted)
    }

    /// Fastest interpolation path — zero allocation beyond the result vector.
    ///
    /// # Overview
    ///
    /// Each adjusted weight encodes the identity:
    ///
    /// ```text
    ///   g*h^i / (z - g*h^i)  =  z * adjusted_i
    /// ```
    ///
    /// so the full barycentric formula becomes:
    ///
    /// ```text
    ///   f(z)  =  z * (z^N - g^N) / (N * g^N)  *  sum_i  adjusted_i * f(g*h^i)
    /// ```
    ///
    /// The inner sum is a single SIMD-optimized column-wise dot product.
    /// The outer scalar is computed with one base-field inversion.
    ///
    /// # Safety
    ///
    /// - The evaluation point must not lie in the coset.
    /// - Each weight must equal 1/(z - x_i) - 1/z for the corresponding coset element.
    ///
    /// # Performance
    ///
    /// - One base-field inversion (for N * g^N).
    /// - log_2(N) extension-field squarings (for z^N).
    /// - log_2(N) base-field squarings (for g^N).
    /// - One SIMD-parallel column-wise dot product over the full matrix.
    /// - No heap allocation except the result vector.
    fn interpolate_coset_with_adjusted_weights<EF: ExtensionField<F>>(
        &self,
        shift: F,
        point: EF,
        adjusted_weights: &[EF],
    ) -> Vec<EF> {
        debug_assert_eq!(adjusted_weights.len(), self.height());

        let log_height = log2_strict_usize(self.height());

        // Phase 1: Global scaling factor
        //
        //   s = z * (z^N - g^N) / (N * g^N)
        //
        // z^N via extension-field repeated squaring (expensive).
        let z_pow_n = point.exp_power_of_2(log_height);
        // g^N via base-field repeated squaring (cheap — single-word ops).
        let g_pow_n = shift.exp_power_of_2(log_height);
        // Combine denominator N * g^N and invert once (only base-field inversion).
        let denom_inv = g_pow_n.mul_2exp_u64(log_height as u64).inverse();
        // Assemble: z * (z^N - g^N) * 1/(N * g^N).
        let scaling_factor = point * (z_pow_n - g_pow_n) * denom_inv;

        // Phase 2: Weighted column sums via the SIMD-optimized dot product.
        //
        // Computes M^T * adjusted_weights, yielding one extension-field
        // result per column (polynomial).
        let mut evals = self.columnwise_dot_product(adjusted_weights);

        // Phase 3: Apply the global scalar to every column result.
        scale_slice_in_place_single_core(&mut evals, scaling_factor);
        evals
    }
}

impl<F: TwoAdicField, M: Matrix<F>> Interpolate<F> for M {}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use p3_baby_bear::BabyBear;
    use p3_field::extension::BinomialExtensionField;
    use p3_field::{
        ExtensionField, Field, PrimeCharacteristicRing, TwoAdicField, batch_multiplicative_inverse,
    };
    use p3_util::log2_strict_usize;
    use proptest::prelude::*;

    use super::Interpolate;
    use crate::dense::RowMajorMatrix;

    type F = BabyBear;
    type EF4 = BinomialExtensionField<BabyBear, 4>;

    fn eval_poly<EF: ExtensionField<F>>(coeffs: &[F], point: EF) -> EF {
        // Horner's method: fold from the highest-degree coefficient down.
        // f(z) = c_0 + z * (c_1 + z * (c_2 + ...))
        coeffs
            .iter()
            .rev()
            .fold(EF::ZERO, |acc, &c| acc * point + c)
    }

    fn eval_poly_on_coset<EF: ExtensionField<F>>(coeffs: &[F], shift: F, log_n: usize) -> Vec<EF> {
        let n = 1 << log_n;
        let subgroup_gen = F::two_adic_generator(log_n);
        (0..n)
            .map(|i| {
                // Coset element at index i: shift * generator^i.
                let coset_elem = shift * subgroup_gen.exp_u64(i as u64);
                eval_poly(coeffs, EF::from(coset_elem))
            })
            .collect()
    }

    #[test]
    fn test_interpolate_subgroup() {
        // Invariant: interpolating f(x) = x^2 + 2x + 3 at z = 100
        // must recover f(100) = 10203.
        //
        // Fixture: 8 precomputed evaluations over the canonical subgroup.
        let evals = [
            6, 886605102, 1443543107, 708307799, 2, 556938009, 569722818, 1874680944,
        ]
        .map(F::from_u32);

        // One polynomial (1 column), 8 evaluation points (8 rows).
        let evals_mat = RowMajorMatrix::new(evals.to_vec(), 1);

        // z = 100 lies outside the subgroup.
        let point = F::from_u16(100);
        let result = evals_mat.interpolate_subgroup(point);

        // 100^2 + 2*100 + 3 = 10203.
        assert_eq!(result, vec![F::from_u16(10203)]);
    }

    #[test]
    fn test_interpolate_coset() {
        // Invariant: both the full path and the precomputation path must
        // agree on f(100) = 10203 for f(x) = x^2 + 2x + 3.
        //
        // Fixture: 8-point coset shifted by the field's multiplicative generator.
        let shift = F::GENERATOR;

        let evals = [
            1026, 129027310, 457985035, 994890337, 902, 1988942953, 1555278970, 913671254,
        ]
        .map(F::from_u32);

        let evals_mat = RowMajorMatrix::new(evals.to_vec(), 1);

        // Part 1: full path (builds coset + batch-inverts internally).
        let point = F::from_u16(100);
        let result = evals_mat.interpolate_coset(shift, point);
        assert_eq!(result, vec![F::from_u16(10203)]);

        // Part 2: precomputation path (caller provides coset + inverse denominators).
        let n = evals.len();
        let k = log2_strict_usize(n);

        // Build the 8-element coset manually.
        let coset = F::two_adic_generator(k).shifted_powers(shift).collect_n(n);

        // Compute 1/(z - x_i) via batch inversion.
        let denom: Vec<_> = coset.iter().map(|&w| point - w).collect();
        let denom = batch_multiplicative_inverse(&denom);

        // Both paths must be bit-identical.
        let result = evals_mat.interpolate_coset_with_precomputation(shift, point, &coset, &denom);
        assert_eq!(result, vec![F::from_u16(10203)]);
    }

    #[test]
    fn test_interpolate_coset_single_point_identity() {
        // Invariant: a constant polynomial f(x) = c evaluates to c everywhere.
        // Interpolation at any external point must recover exactly c.
        let c = F::from_u32(42);

        // 8 identical evaluations => constant polynomial of degree 0.
        let evals = vec![c; 8];
        let evals_mat = RowMajorMatrix::new(evals, 1);

        let shift = F::GENERATOR;
        let point = F::from_u16(1337);

        let result = evals_mat.interpolate_coset(shift, point);
        assert_eq!(result, vec![c]);
    }

    #[test]
    fn test_interpolate_subgroup_degree_3_correctness() {
        // Invariant: a degree-3 polynomial over a quartic extension field is
        // uniquely determined by 4 = 2^2 evaluation points.
        // Interpolation must match direct evaluation.

        // f(x) = x^3 + 2*x^2 + 3*x + 4
        let poly = |x: EF4| x * x * x + x * x * F::TWO + x * F::from_u32(3) + F::from_u32(4);

        // Evaluate at the 4 elements of the canonical 2^2-subgroup.
        let subgroup = EF4::two_adic_generator(2).powers().collect_n(4);
        let evals: Vec<_> = subgroup.iter().map(|&x| poly(x)).collect();
        let evals_mat = RowMajorMatrix::new(evals, 1);

        // Interpolate at z = 5 and compare against direct Horner evaluation.
        let point = EF4::from_u16(5);
        let result = evals_mat.interpolate_subgroup(point);
        let expected = poly(point);
        assert_eq!(result[0], expected);
    }

    #[test]
    fn test_interpolate_coset_multiple_polynomials() {
        // Invariant: batch interpolation of K polynomials via a K-column matrix
        // must agree with interpolating each polynomial individually.
        //
        // Fixture: two polynomials over an 8-point coset.
        //
        //   f_1(x) = x^2 + 2x + 3
        //   f_2(x) = 4x^2 + 5x + 6
        //
        //   Matrix layout (8 rows x 2 columns):
        //
        //     row i = [ f_1(coset_i),  f_2(coset_i) ]

        // 8-point coset shifted by the quartic extension generator.
        let shift = EF4::GENERATOR;
        let coset = EF4::two_adic_generator(3)
            .shifted_powers(shift)
            .collect_n(8);

        let f1 = |x: EF4| x * x + x * F::TWO + F::from_u32(3);
        let f2 = |x: EF4| x * x * F::from_u32(4) + x * F::from_u32(5) + F::from_u32(6);

        // Interleave: [f1(c_0), f2(c_0), f1(c_1), f2(c_1), ...].
        let evals: Vec<_> = coset.iter().flat_map(|&x| vec![f1(x), f2(x)]).collect();
        let evals_mat = RowMajorMatrix::new(evals, 2);

        // Interpolate both columns at z = 77.
        let point = EF4::from_u32(77);
        let result = evals_mat.interpolate_coset(shift, point);

        // Each column must match direct evaluation.
        assert_eq!(result[0], f1(point));
        assert_eq!(result[1], f2(point));
    }

    #[test]
    fn test_interpolate_subgroup_multiple_columns() {
        // Invariant: the subgroup path (shift = 1) must produce the same
        // results as the coset path for multi-column matrices.
        //
        //   f_1(x) = x^2 + 2x + 3
        //   f_2(x) = 4x^2 + 5x + 6

        let f1 = |x: EF4| x * x + x * F::TWO + F::from_u32(3);
        let f2 = |x: EF4| x * x * F::from_u32(4) + x * F::from_u32(5) + F::from_u32(6);

        // Canonical 8-point subgroup.
        let subgroup_iter = EF4::two_adic_generator(3).powers().take(8);

        // Interleaved evaluations => 8 rows x 2 columns.
        let evals: Vec<_> = subgroup_iter.flat_map(|x| vec![f1(x), f2(x)]).collect();
        let evals_mat = RowMajorMatrix::new(evals, 2);

        // z = 77 lies outside the subgroup.
        let point = EF4::from_u32(77);
        let result = evals_mat.interpolate_subgroup(point);

        assert_eq!(result, vec![f1(point), f2(point)]);
    }

    proptest! {
        #[test]
        fn prop_roundtrip_subgroup(
            log_n in 1usize..=4,
            coeffs_raw in prop::collection::vec(0u32..2013265921, 1..=16),
            point_raw in 1u32..2013265921u32,
        ) {
            // Invariant: evaluate f on the canonical subgroup, interpolate at z,
            // and the result must equal direct Horner evaluation of f at z.
            //
            //   coeffs -> eval on {h^0, ..., h^{N-1}} -> interpolate at z
            //   coeffs -> Horner at z
            //   Both must agree.

            // Truncate to degree < N so the polynomial is uniquely determined.
            let n = 1usize << log_n;
            let coeffs: Vec<F> = coeffs_raw.iter().take(n).map(|&v| F::from_u32(v)).collect();

            // Evaluate on the canonical subgroup (shift = 1).
            let evals: Vec<F> = eval_poly_on_coset(&coeffs, F::ONE, log_n);
            let evals_mat = RowMajorMatrix::new(evals, 1);

            let point = EF4::from_u32(point_raw);

            let result = evals_mat.interpolate_subgroup(point);
            let expected = eval_poly(&coeffs, point);
            prop_assert_eq!(result[0], expected);
        }

        #[test]
        fn prop_roundtrip_coset(
            log_n in 1usize..=4,
            coeffs_raw in prop::collection::vec(0u32..2013265921, 1..=16),
            point_raw in 1u32..2013265921u32,
        ) {
            // Invariant: same round-trip as above, but over a shifted coset
            // with shift = multiplicative generator.

            let n = 1usize << log_n;
            let coeffs: Vec<F> = coeffs_raw.iter().take(n).map(|&v| F::from_u32(v)).collect();
            let shift = F::GENERATOR;

            let evals: Vec<F> = eval_poly_on_coset(&coeffs, shift, log_n);
            let evals_mat = RowMajorMatrix::new(evals, 1);
            let point = EF4::from_u32(point_raw);

            let result = evals_mat.interpolate_coset(shift, point);
            let expected = eval_poly(&coeffs, point);
            prop_assert_eq!(result[0], expected);
        }

        #[test]
        fn prop_precomputation_equivalence(
            log_n in 1usize..=4,
            coeffs_raw in prop::collection::vec(0u32..2013265921, 1..=16),
            point_raw in 1u32..2013265921u32,
        ) {
            // Invariant: the full path and the precomputation path must be
            // bit-identical — they compute the same barycentric formula.

            let n = 1usize << log_n;
            let coeffs: Vec<F> = coeffs_raw.iter().take(n).map(|&v| F::from_u32(v)).collect();
            let shift = F::GENERATOR;

            let evals: Vec<F> = eval_poly_on_coset(&coeffs, shift, log_n);
            let evals_mat = RowMajorMatrix::new(evals, 1);
            let point = EF4::from_u32(point_raw);

            // Full path (builds everything internally).
            let result_standard = evals_mat.interpolate_coset(shift, point);

            // Manual precomputation path (caller builds coset + inverse denoms).
            let subgroup_gen = F::two_adic_generator(log_n);
            let coset: Vec<F> =
                (0..n).map(|i| shift * subgroup_gen.exp_u64(i as u64)).collect();
            let diffs: Vec<EF4> = coset.iter().map(|&c| point - c).collect();
            let diff_invs = batch_multiplicative_inverse(&diffs);
            let result_precomp = evals_mat
                .interpolate_coset_with_precomputation(shift, point, &coset, &diff_invs);

            prop_assert_eq!(result_standard, result_precomp);
        }

        #[test]
        fn prop_constant_polynomial(
            log_n in 1usize..=4,
            c_raw in 0u32..2013265921u32,
            point_raw in 1u32..2013265921u32,
        ) {
            // Invariant: a constant polynomial evaluates to the same value
            // everywhere. Interpolation at any external point must recover it.

            let n = 1usize << log_n;
            let c = F::from_u32(c_raw);

            // N identical evaluations => degree-0 polynomial.
            let evals = vec![c; n];
            let evals_mat = RowMajorMatrix::new(evals, 1);
            let point = EF4::from_u32(point_raw);

            let result = evals_mat.interpolate_subgroup(point);
            prop_assert_eq!(result[0], EF4::from(c));
        }

        #[test]
        fn prop_linearity(
            log_n in 1usize..=3,
            f_raw in prop::collection::vec(0u32..2013265921, 1..=8),
            g_raw in prop::collection::vec(0u32..2013265921, 1..=8),
            a_raw in 0u32..2013265921u32,
            b_raw in 0u32..2013265921u32,
            point_raw in 1u32..2013265921u32,
        ) {
            // Invariant: barycentric interpolation is linear.
            //   interp(a*f + b*g, z) == a * interp(f, z) + b * interp(g, z)

            let n = 1usize << log_n;
            let f_coeffs: Vec<F> = f_raw.iter().take(n).map(|&v| F::from_u32(v)).collect();
            let g_coeffs: Vec<F> = g_raw.iter().take(n).map(|&v| F::from_u32(v)).collect();
            let a = F::from_u32(a_raw);
            let b = F::from_u32(b_raw);

            // Evaluate f, g, and the linear combination on the canonical subgroup.
            let f_evals: Vec<F> = eval_poly_on_coset(&f_coeffs, F::ONE, log_n);
            let g_evals: Vec<F> = eval_poly_on_coset(&g_coeffs, F::ONE, log_n);
            let combined_evals: Vec<F> = f_evals
                .iter()
                .zip(&g_evals)
                .map(|(&fe, &ge)| a * fe + b * ge)
                .collect();

            let f_mat = RowMajorMatrix::new(f_evals, 1);
            let g_mat = RowMajorMatrix::new(g_evals, 1);
            let combined_mat = RowMajorMatrix::new(combined_evals, 1);
            let point = EF4::from_u32(point_raw);

            // Interpolate individually and as a linear combination.
            let interp_f = f_mat.interpolate_subgroup(point)[0];
            let interp_g = g_mat.interpolate_subgroup(point)[0];
            let interp_combined = combined_mat.interpolate_subgroup(point)[0];

            let expected = EF4::from(a) * interp_f + EF4::from(b) * interp_g;
            prop_assert_eq!(interp_combined, expected);
        }

        #[test]
        fn prop_batch_equals_individual(
            log_n in 1usize..=3,
            f_raw in prop::collection::vec(0u32..2013265921, 1..=8),
            g_raw in prop::collection::vec(0u32..2013265921, 1..=8),
            point_raw in 1u32..2013265921u32,
        ) {
            // Invariant: a multi-column matrix must produce the same results
            // as interpolating each column individually.
            //
            //   batch (N x 2):  [ f(c_0)  g(c_0) ]  -> interpolate -> [ f(z), g(z) ]
            //                   [ f(c_1)  g(c_1) ]
            //                   ...
            //
            //   single (N x 1) each  -> interpolate -> f(z), g(z)

            let n = 1usize << log_n;
            let f_coeffs: Vec<F> = f_raw.iter().take(n).map(|&v| F::from_u32(v)).collect();
            let g_coeffs: Vec<F> = g_raw.iter().take(n).map(|&v| F::from_u32(v)).collect();
            let shift = F::GENERATOR;

            let f_evals: Vec<F> = eval_poly_on_coset(&f_coeffs, shift, log_n);
            let g_evals: Vec<F> = eval_poly_on_coset(&g_coeffs, shift, log_n);

            // Interleave into a 2-column batch matrix.
            let batch_evals: Vec<F> = f_evals
                .iter()
                .zip(&g_evals)
                .flat_map(|(&fe, &ge)| vec![fe, ge])
                .collect();
            let batch_mat = RowMajorMatrix::new(batch_evals, 2);

            // Individual single-column matrices.
            let f_mat = RowMajorMatrix::new(f_evals, 1);
            let g_mat = RowMajorMatrix::new(g_evals, 1);
            let point = EF4::from_u32(point_raw);

            let batch_result = batch_mat.interpolate_coset(shift, point);
            let f_result = f_mat.interpolate_coset(shift, point)[0];
            let g_result = g_mat.interpolate_coset(shift, point)[0];

            prop_assert_eq!(batch_result[0], f_result);
            prop_assert_eq!(batch_result[1], g_result);
        }
    }
}
