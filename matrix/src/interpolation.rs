//! Lagrange interpolation over two-adic cosets.

use alloc::vec::Vec;

use p3_field::coset::TwoAdicMultiplicativeCoset;
use p3_field::{
    ExtensionField, TwoAdicField, batch_multiplicative_inverse, scale_slice_in_place_single_core,
};
use p3_maybe_rayon::prelude::*;
use p3_util::log2_strict_usize;

use crate::Matrix;

/// Extension trait that adds Lagrange interpolation over two-adic cosets to any [`Matrix`].
///
/// This is automatically implemented for every `M: Matrix<F>` where `F: TwoAdicField`.
/// So there is nothing to implement manually — just import the trait.
pub trait Interpolate<F: TwoAdicField>: Matrix<F> {
    /// Given evaluations of a batch of polynomials over the canonical power-of-two subgroup,
    /// evaluate the polynomials at `point`.
    ///
    /// The canonical subgroup is the coset with shift = 1, so this delegates directly
    /// to the coset interpolation method.
    ///
    /// This assumes the point is not in the subgroup, otherwise the behavior is undefined.
    fn interpolate_subgroup<EF: ExtensionField<F>>(&self, point: EF) -> Vec<EF> {
        // The canonical subgroup is the coset g*H with g = 1.
        self.interpolate_coset(F::ONE, point)
    }

    /// Given evaluations of a batch of polynomials over the given coset of the canonical
    /// power-of-two subgroup, evaluate the polynomials at `point`.
    ///
    /// This assumes the point is not in the coset, otherwise the behavior is undefined.
    ///
    /// The evaluations must be given in standard (not bit-reversed) order.
    fn interpolate_coset<EF: ExtensionField<F>>(&self, shift: F, point: EF) -> Vec<EF> {
        // The matrix height equals the coset size N = 2^log_height.
        let log_height = log2_strict_usize(self.height());

        // Build the coset elements: {shift * h^0, shift * h^1, ..., shift * h^{N-1}}
        // where h is the 2-adic generator of order N.
        let coset = TwoAdicMultiplicativeCoset::new(shift, log_height)
            .unwrap()
            .iter()
            .collect();

        // Compute the difference (point - coset_element) for each coset element,
        // then batch-invert to get 1/(z - g*h^i) in a single Montgomery inversion.
        let diffs: Vec<_> = coset.par_iter().map(|&g| point - g).collect();
        let diff_invs = batch_multiplicative_inverse(&diffs);

        // Delegate to the precomputation variant with the coset and inverses we just built.
        self.interpolate_coset_with_precomputation(shift, point, &coset, &diff_invs)
    }

    /// Given evaluations of a batch of polynomials over the given coset of the canonical
    /// power-of-two subgroup, evaluate the polynomials at `point`.
    ///
    /// This assumes the point is not in the coset, otherwise the behavior is undefined.
    ///
    /// This method takes the precomputed `coset` points and `diff_invs` (the inverses of the
    /// differences between the evaluation point and each shifted subgroup element), and should
    /// be preferred over [`interpolate_coset`](Interpolate::interpolate_coset) when repeatedly
    /// called with the same subgroup and/or point.
    ///
    /// Unlike `interpolate_coset`, the parameters `coset` and `diff_invs` may use any indexing
    /// scheme, as long as they are consistent with the row ordering of `self`.
    fn interpolate_coset_with_precomputation<EF: ExtensionField<F>>(
        &self,
        shift: F,
        point: EF,
        coset: &[F],
        diff_invs: &[EF],
    ) -> Vec<EF> {
        // Slight variation of this approach: https://hackmd.io/@vbuterin/barycentric_evaluation
        debug_assert_eq!(coset.len(), diff_invs.len());
        debug_assert_eq!(coset.len(), self.height());

        // We start with the evaluations of a polynomial `f` over a coset `gH` of size `N`
        // and want to compute `f(z)`.
        //
        // Observe that `z^N - g^N` is equal to `0` at all points in the coset.
        // Thus `(z^N - g^N)/(z - gh^i)` is equal to `0` at all points except for `gh^i`
        // where it is equal to:
        //          `N * (gh^i)^{N - 1} = N * g^N * (gh^i)^{-1}.`
        //
        // Hence `L_i(z) = h^i * (z^N - g^N)/(N * g^{N - 1} * (z - gh^i))` will be equal
        // to `1` at `gh^i` and `0` at all other points in the coset. This means that we
        // can compute `f(z)` as:
        //   `\sum_i L_i(z) f(gh^i) = (z^N - g^N)/(N * g^N) * \sum_i gh^i/(z - gh^i) f(gh^i).`

        // TODO: It might be possible to speed this up by refactoring the code to instead compute:
        //          `((z/g)^N - 1)/N * \sum_i 1/(z/(gh^i) - 1) f(gh^i).`
        // This would remove the need for the multiplications and collections in `col_scale` and
        // let us remove one of the `exp_power_of_2` calls (which are somewhat expensive as they
        // are over the extension field). We could also remove the .inverse() in scale_vec.

        // N = 2^log_height is the coset size.
        let log_height = log2_strict_usize(self.height());

        // Phase 1: Build the per-element barycentric weights.
        //
        // For each coset element g*h^i, compute the weight:
        //     w_i = g*h^i / (z - g*h^i)
        //
        // These are the numerator terms in the barycentric Lagrange formula.
        let col_scale: Vec<_> = coset
            .par_iter()
            .zip(diff_invs)
            .map(|(&sg, &diff_inv)| diff_inv * sg)
            .collect();

        // Phase 2: Compute the global scaling factor.
        //
        // The barycentric formula gives:
        //     f(z) = Z_{gH}(z) / (N * g^N) * \sum_i w_i * f(g*h^i)
        //
        // where Z_{gH}(z) = z^N - g^N is the vanishing polynomial of the coset.

        // Raise point and shift to the N-th power via repeated squaring (log_height steps).
        let point_pow_height = point.exp_power_of_2(log_height);
        let shift_pow_height = shift.exp_power_of_2(log_height);

        // Vanishing polynomial evaluated at z: Z_{gH}(z) = z^N - g^N.
        let vanishing_polynomial = point_pow_height - shift_pow_height;

        // Denominator: N * g^N. Multiplying by 2^log_height is equivalent to * N in a
        // two-adic field (since N is a power of two, this is a left-shift).
        let denominator = shift_pow_height.mul_2exp_u64(log_height as u64);

        // Global scaling factor: s = Z_{gH}(z) / (N * g^N).
        let scaling_factor = vanishing_polynomial * denominator.inverse();

        // Phase 3: Evaluate all column polynomials at once.
        //
        // M^T * col_scale computes \sum_i w_i * f_j(g*h^i) for each column j.
        // Then multiply each result by the global scaling factor s.
        let mut evals = self.columnwise_dot_product(&col_scale);
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
        ExtensionField, Field, PrimeCharacteristicRing, TwoAdicField,
        batch_multiplicative_inverse,
    };
    use p3_util::log2_strict_usize;
    use proptest::prelude::*;

    use super::Interpolate;
    use crate::dense::RowMajorMatrix;

    type F = BabyBear;
    type EF4 = BinomialExtensionField<BabyBear, 4>;

    /// Evaluate a polynomial (given by coefficients) at a point using Horner's method.
    fn eval_poly<EF: ExtensionField<F>>(coeffs: &[F], point: EF) -> EF {
        // Horner's method: f(z) = c_0 + z*(c_1 + z*(c_2 + ...))
        // Process coefficients from highest degree down to constant term.
        coeffs
            .iter()
            .rev()
            .fold(EF::ZERO, |acc, &c| acc * point + c)
    }

    /// Evaluate a polynomial at all points of a coset, returning the evaluations.
    fn eval_poly_on_coset<EF: ExtensionField<F>>(coeffs: &[F], shift: F, log_n: usize) -> Vec<EF> {
        let n = 1 << log_n;
        // Build the coset {shift * h^0, shift * h^1, ..., shift * h^{n-1}}.
        let subgroup_gen = F::two_adic_generator(log_n);
        (0..n)
            .map(|i| {
                // Coset element: shift * subgroup_gen^i.
                let coset_elem = shift * subgroup_gen.exp_u64(i as u64);
                eval_poly(coeffs, EF::from(coset_elem))
            })
            .collect()
    }

    #[test]
    fn test_interpolate_subgroup() {
        // Polynomial: f(x) = x^2 + 2x + 3, evaluated over the 8-point two-adic subgroup.
        // Known answer: f(100) = 10000 + 200 + 3 = 10203.

        // Pre-computed evaluations of f over the canonical 8-point subgroup {h^0, ..., h^7}.
        let evals = [
            6, 886605102, 1443543107, 708307799, 2, 556938009, 569722818, 1874680944,
        ]
        .map(F::from_u32);

        // Single column matrix: one polynomial, 8 evaluation rows.
        let evals_mat = RowMajorMatrix::new(evals.to_vec(), 1);

        // Interpolate at z = 100, which lies outside the subgroup.
        let point = F::from_u16(100);
        let result = evals_mat.interpolate_subgroup(point);

        // Verify the known answer: f(100) = 10203.
        assert_eq!(result, vec![F::from_u16(10203)]);
    }

    #[test]
    fn test_interpolate_coset() {
        // Polynomial: f(x) = x^2 + 2x + 3, evaluated over an 8-point coset shifted
        // by the field generator. Known answer: f(100) = 10203.

        // Coset shift: the multiplicative generator of the field.
        let shift = F::GENERATOR;

        // Pre-computed evaluations of f over the coset {shift * h^0, ..., shift * h^7}.
        let evals = [
            1026, 129027310, 457985035, 994890337, 902, 1988942953, 1555278970, 913671254,
        ]
        .map(F::from_u32);

        // Single column matrix: one polynomial, 8 rows.
        let evals_mat = RowMajorMatrix::new(evals.to_vec(), 1);

        // Part 1: test the standard coset interpolation path.
        let point = F::from_u16(100);
        let result = evals_mat.interpolate_coset(shift, point);
        assert_eq!(result, vec![F::from_u16(10203)]);

        // Part 2: test the precomputation path, which should give the same result.
        // Manually build the coset elements and inverse denominators.
        let n = evals.len();
        let k = log2_strict_usize(n);

        // Coset elements: {shift * h^0, shift * h^1, ..., shift * h^{N-1}}.
        let coset = F::two_adic_generator(k).shifted_powers(shift).collect_n(n);

        // Inverse denominators: 1/(z - coset_i) for each coset element.
        let denom: Vec<_> = coset.iter().map(|&w| point - w).collect();
        let denom = batch_multiplicative_inverse(&denom);

        // The precomputation variant must produce the same result.
        let result = evals_mat.interpolate_coset_with_precomputation(shift, point, &coset, &denom);
        assert_eq!(result, vec![F::from_u16(10203)]);
    }

    #[test]
    fn test_interpolate_coset_single_point_identity() {
        // Invariant: a constant polynomial f(x) = c evaluates to c everywhere,
        // so interpolation at any external point must recover exactly c.

        // Constant polynomial: all 8 evaluations are the same value.
        let c = F::from_u32(42);
        let evals = vec![c; 8];
        let evals_mat = RowMajorMatrix::new(evals, 1);

        // Shifted coset and an arbitrary external evaluation point.
        let shift = F::GENERATOR;
        let point = F::from_u16(1337);

        // Interpolation must recover the constant exactly.
        let result = evals_mat.interpolate_coset(shift, point);
        assert_eq!(result, vec![c]);
    }

    #[test]
    fn test_interpolate_subgroup_degree_3_correctness() {
        // Verify that a degree-3 polynomial over a degree-4 extension field
        // is correctly interpolated from exactly 2^2 = 4 subgroup points.
        // A degree-3 polynomial requires exactly 4 evaluation points.

        // f(x) = x^3 + 2*x^2 + 3*x + 4, defined over the quartic extension.
        let poly = |x: EF4| x * x * x + x * x * F::TWO + x * F::from_u32(3) + F::from_u32(4);

        // Evaluate f at the 4 elements of the canonical 2^2-subgroup.
        let subgroup = EF4::two_adic_generator(2).powers().collect_n(4);
        let evals: Vec<_> = subgroup.iter().map(|&x| poly(x)).collect();

        // Single column matrix: one polynomial, 4 evaluation rows.
        let evals_mat = RowMajorMatrix::new(evals, 1);

        // Interpolate at z = 5 and compare against direct evaluation.
        let point = EF4::from_u16(5);
        let result = evals_mat.interpolate_subgroup(point);
        let expected = poly(point);

        assert_eq!(result[0], expected);
    }

    #[test]
    fn test_interpolate_coset_multiple_polynomials() {
        // Verify batch interpolation: two polynomials evaluated over the same coset
        // are interpolated simultaneously using a 2-column matrix.
        //
        //     f_1(x) = x^2 + 2x + 3
        //     f_2(x) = 4x^2 + 5x + 6
        //
        //     Matrix layout (8 rows x 2 columns):
        //         row i = [ f_1(coset[i]), f_2(coset[i]) ]

        // Build the 8-point coset shifted by the extension field generator.
        let shift = EF4::GENERATOR;
        let coset = EF4::two_adic_generator(3)
            .shifted_powers(shift)
            .collect_n(8);

        let f1 = |x: EF4| x * x + x * F::TWO + F::from_u32(3);
        let f2 = |x: EF4| x * x * F::from_u32(4) + x * F::from_u32(5) + F::from_u32(6);

        // Interleave evaluations: [f1(c0), f2(c0), f1(c1), f2(c1), ...].
        let evals: Vec<_> = coset.iter().flat_map(|&x| vec![f1(x), f2(x)]).collect();

        // Two-column matrix: each column is one polynomial's evaluations.
        let evals_mat = RowMajorMatrix::new(evals, 2);

        // Interpolate both polynomials at z = 77.
        let point = EF4::from_u32(77);
        let result = evals_mat.interpolate_coset(shift, point);

        // Compare against direct evaluation of each polynomial at the same point.
        let expected_f1 = f1(point);
        let expected_f2 = f2(point);

        assert_eq!(result[0], expected_f1);
        assert_eq!(result[1], expected_f2);
    }

    #[test]
    fn test_interpolate_subgroup_multiple_columns() {
        // Same as the coset multi-polynomial test, but over the canonical subgroup
        // (shift = 1). Verifies that the subgroup path correctly delegates to
        // the coset path and produces identical results.
        //
        //     f_1(x) = x^2 + 2x + 3
        //     f_2(x) = 4x^2 + 5x + 6

        let f1 = |x: EF4| x * x + x * F::TWO + F::from_u32(3);
        let f2 = |x: EF4| x * x * F::from_u32(4) + x * F::from_u32(5) + F::from_u32(6);

        // Evaluation domain: the canonical 2^3 = 8-point subgroup {h^0, ..., h^7}.
        let subgroup_iter = EF4::two_adic_generator(3).powers().take(8);

        // Evaluate both polynomials on the subgroup, interleaved.
        let evals: Vec<_> = subgroup_iter.flat_map(|x| vec![f1(x), f2(x)]).collect();

        // 8 rows x 2 columns: column 0 holds f_1 evaluations, column 1 holds f_2.
        let evals_mat = RowMajorMatrix::new(evals, 2);

        // Interpolate at z = 77, which lies outside the subgroup.
        let point = EF4::from_u32(77);
        let result = evals_mat.interpolate_subgroup(point);

        // Compare against direct evaluation of each polynomial.
        let expected_f1 = f1(point);
        let expected_f2 = f2(point);

        assert_eq!(result, vec![expected_f1, expected_f2]);
    }

    proptest! {
        // Correctness: subgroup round-trip
        #[test]
        fn prop_roundtrip_subgroup(
            log_n in 1usize..=4,
            coeffs_raw in prop::collection::vec(0u32..2013265921, 1..=16),
            point_raw in 1u32..2013265921u32,
        ) {
            // Invariant: evaluate f on 2^log_n-subgroup, interpolate at z → must equal f(z).
            //
            //     coeffs  →  eval on {h^0, ..., h^{N-1}}  →  interpolate at z
            //     coeffs  →  Horner at z
            //     Both must agree.

            // Truncate to degree < N so the polynomial is uniquely determined.
            let n = 1usize << log_n;
            let coeffs: Vec<F> = coeffs_raw.iter().take(n).map(|&v| F::from_u32(v)).collect();

            // Evaluate on canonical subgroup (shift = 1).
            let evals: Vec<F> = eval_poly_on_coset(&coeffs, F::ONE, log_n);
            let evals_mat = RowMajorMatrix::new(evals, 1);

            let point = EF4::from_u32(point_raw);

            // Compare interpolation against direct Horner evaluation.
            let result = evals_mat.interpolate_subgroup(point);
            let expected = eval_poly(&coeffs, point);
            prop_assert_eq!(result[0], expected);
        }

        // Correctness: coset round-trip (shift = GENERATOR)
        #[test]
        fn prop_roundtrip_coset(
            log_n in 1usize..=4,
            coeffs_raw in prop::collection::vec(0u32..2013265921, 1..=16),
            point_raw in 1u32..2013265921u32,
        ) {
            // Same round-trip as above, but over a shifted coset {g*h^i}.
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

        // Path equivalence: standard vs precomputation
        #[test]
        fn prop_precomputation_equivalence(
            log_n in 1usize..=4,
            coeffs_raw in prop::collection::vec(0u32..2013265921, 1..=16),
            point_raw in 1u32..2013265921u32,
        ) {
            // Invariant: both code paths compute the same barycentric formula.
            //
            //     interpolate_coset           (builds coset + inverses internally)
            //     interpolate_coset_with_precomputation (caller provides them)
            //     → must be bit-identical.
            let n = 1usize << log_n;
            let coeffs: Vec<F> = coeffs_raw.iter().take(n).map(|&v| F::from_u32(v)).collect();
            let shift = F::GENERATOR;

            let evals: Vec<F> = eval_poly_on_coset(&coeffs, shift, log_n);
            let evals_mat = RowMajorMatrix::new(evals, 1);
            let point = EF4::from_u32(point_raw);

            // Standard path.
            let result_standard = evals_mat.interpolate_coset(shift, point);

            // Manual precomputation path.
            let subgroup_gen = F::two_adic_generator(log_n);
            let coset: Vec<F> =
                (0..n).map(|i| shift * subgroup_gen.exp_u64(i as u64)).collect();
            let diffs: Vec<EF4> = coset.iter().map(|&c| point - c).collect();
            let diff_invs = batch_multiplicative_inverse(&diffs);
            let result_precomp = evals_mat
                .interpolate_coset_with_precomputation(shift, point, &coset, &diff_invs);

            prop_assert_eq!(result_standard, result_precomp);
        }

        // Constant polynomial: f(x) = c → interpolation at any z must return c.
        #[test]
        fn prop_constant_polynomial(
            log_n in 1usize..=4,
            c_raw in 0u32..2013265921u32,
            point_raw in 1u32..2013265921u32,
        ) {
            let n = 1usize << log_n;
            let c = F::from_u32(c_raw);

            // N identical evaluations → constant polynomial.
            let evals = vec![c; n];
            let evals_mat = RowMajorMatrix::new(evals, 1);
            let point = EF4::from_u32(point_raw);

            let result = evals_mat.interpolate_subgroup(point);
            prop_assert_eq!(result[0], EF4::from(c));
        }

        // Linearity: interp(a*f + b*g) == a*interp(f) + b*interp(g)
        #[test]
        fn prop_linearity(
            log_n in 1usize..=3,
            f_raw in prop::collection::vec(0u32..2013265921, 1..=8),
            g_raw in prop::collection::vec(0u32..2013265921, 1..=8),
            a_raw in 0u32..2013265921u32,
            b_raw in 0u32..2013265921u32,
            point_raw in 1u32..2013265921u32,
        ) {
            // Invariant: barycentric interpolation is linear over the evaluation column.
            let n = 1usize << log_n;
            let f_coeffs: Vec<F> = f_raw.iter().take(n).map(|&v| F::from_u32(v)).collect();
            let g_coeffs: Vec<F> = g_raw.iter().take(n).map(|&v| F::from_u32(v)).collect();
            let a = F::from_u32(a_raw);
            let b = F::from_u32(b_raw);

            // Evaluate f, g, and (a*f + b*g) on the canonical subgroup.
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

        // Batch equivalence: 2-column matrix vs two 1-column matrices
        #[test]
        fn prop_batch_equals_individual(
            log_n in 1usize..=3,
            f_raw in prop::collection::vec(0u32..2013265921, 1..=8),
            g_raw in prop::collection::vec(0u32..2013265921, 1..=8),
            point_raw in 1u32..2013265921u32,
        ) {
            // Invariant: batch[col_j] == individual[col_j] for all j.
            //
            //     batch_mat (N×2):       [f(c_0) g(c_0)]     → interpolate → [f(z), g(z)]
            //                            [f(c_1) g(c_1)]
            //                            ...
            //     f_mat (N×1), g_mat (N×1) → interpolate each → f(z), g(z)
            let n = 1usize << log_n;
            let f_coeffs: Vec<F> = f_raw.iter().take(n).map(|&v| F::from_u32(v)).collect();
            let g_coeffs: Vec<F> = g_raw.iter().take(n).map(|&v| F::from_u32(v)).collect();
            let shift = F::GENERATOR;

            let f_evals: Vec<F> = eval_poly_on_coset(&f_coeffs, shift, log_n);
            let g_evals: Vec<F> = eval_poly_on_coset(&g_coeffs, shift, log_n);

            // Interleave into 2-column batch matrix.
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
