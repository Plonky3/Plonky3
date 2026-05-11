//! Lagrange interpolation over structured (two-adic coset) and arbitrary evaluation domains.
//!
//! Evaluates polynomials at out-of-domain points given their evaluations on the chosen domain.
//!
//! # Mathematical background (two-adic coset path)
//!
//! Slight variation of this approach: <https://hackmd.io/@vbuterin/barycentric_evaluation>.
//!
//! We start with the evaluations of a polynomial `f` over a coset `gH` of size `N`
//! and want to compute `f(z)`.
//!
//! Observe that `z^N - g^N` is equal to `0` at all points in the coset.
//! Thus `(z^N - g^N)/(z - gh^i)` is equal to `0` at all points except for `gh^i`
//! where it is equal to:
//! ```text
//!   N * (gh^i)^{N - 1} = N * g^N * (gh^i)^{-1}.
//! ```
//!
//! Hence `L_i(z) = h^i * (z^N - g^N)/(N * g^{N - 1} * (z - gh^i))` will be equal
//! to `1` at `gh^i` and `0` at all other points in the coset. This means that we
//! can compute `f(z)` as:
//! ```text
//!   sum_i L_i(z) f(gh^i) = (z^N - g^N)/(N * g^N) * sum_i gh^i/(z - gh^i) * f(gh^i)
//!                        = z * (z^N - g^N)/(N * g^N) * sum_i (1/(z - gh^i) - 1/z) * f(gh^i).
//! ```
//!
//! This second equality lets us trade off N extension-by-base multiplications for
//! a single extension-by-extension multiplication, an extension inversion and N
//! extension-by-extension subtractions. For large N this is worth it.
//!
//! Thus we define the **adjusted weights** to be `(1/(z - g*h^i) - 1/z)` and work with
//! these instead.
//!
//! # Arbitrary-domain path
//!
//! For evaluation domains that are not a two-adic coset we fall back to the standard
//! second-form barycentric formula with precomputed weights `w_i = 1/prod_{j != i}(x_i - x_j)`.
//! See [`InterpolateArbitrary`] for the matrix-level entry points and
//! [`interpolate_lagrange`] for a single-polynomial convenience helper.

use alloc::vec::Vec;

use p3_field::coset::TwoAdicMultiplicativeCoset;
use p3_field::{
    ExtensionField, Field, TwoAdicField, batch_multiplicative_inverse,
    scale_slice_in_place_single_core,
};
use p3_maybe_rayon::prelude::*;
use p3_util::log2_strict_usize;

use crate::Matrix;
use crate::dense::RowMajorMatrix;

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
        self.interpolate_coset_with_precomputation(shift, point, &adjusted)
    }

    /// Fastest interpolation path — zero allocation beyond the result vector.
    ///
    /// Given evaluations of a batch of polynomials over the given coset of the canonical
    /// power-of-two subgroup, evaluate the polynomials at `point`.
    ///
    /// This method takes the precomputed `adjusted_weights` and should
    /// be preferred over [`interpolate_coset`](Interpolate::interpolate_coset) when repeatedly
    /// called with the same subgroup and/or point.
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
    fn interpolate_coset_with_precomputation<EF: ExtensionField<F>>(
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

/// Computes barycentric weights w_i = 1 / prod_{j != i} (x_i - x_j).
///
/// These weights depend only on the domain points, not on polynomial values.
/// Precompute them once and reuse across many evaluation targets.
///
/// # Performance
///
/// - n(n-1)/2 field subtractions (upper-triangle symmetry trick).
/// - One batch inversion via Montgomery's trick.
///
/// # Returns
///
/// `None` if any two domain points coincide.
pub fn barycentric_weights<F: Field>(x_coords: &[F]) -> Option<Vec<F>> {
    let n = x_coords.len();
    if n == 0 {
        return Some(Vec::new());
    }

    // Accumulate denom_i = prod_{j != i} (x_i - x_j) for every point.
    let mut denoms = alloc::vec![F::ONE; n];
    for i in 0..n {
        // Only iterate j < i (strict upper triangle).
        //
        // Antisymmetry: (x_i - x_j) = -(x_j - x_i);
        // So one subtraction updates both denom_i and denom_j.
        for j in 0..i {
            let diff = x_coords[i] - x_coords[j];
            // Zero difference means a duplicate domain point.
            if diff.is_zero() {
                return None;
            }
            denoms[i] *= diff;
            denoms[j] *= -diff;
        }
    }

    // Invert all n denominators in one shot: O(n) muls + 1 inversion.
    Some(batch_multiplicative_inverse(&denoms))
}

/// Lagrange interpolation over arbitrary evaluation domains.
///
/// General-domain counterpart of the structured-domain trait.
///
/// Blanket-implemented for every matrix over a field — just import and call.
pub trait InterpolateArbitrary<F: Field>: Matrix<F> {
    /// Evaluates every column polynomial at `point` via barycentric interpolation.
    ///
    /// Each row holds evaluations at the corresponding domain point.
    ///
    /// # Performance
    ///
    /// O(n^2) weight computation + O(n * width) evaluation.
    ///
    /// # Returns
    ///
    /// - `None` if any domain points coincide.
    /// - The matching row directly when the target equals a domain point.
    fn interpolate_arbitrary_point<EF: ExtensionField<F>>(
        &self,
        x_coords: &[F],
        point: EF,
    ) -> Option<Vec<EF>> {
        debug_assert_eq!(x_coords.len(), self.height());

        // Order matters: reject duplicates BEFORE the on-domain shortcut.
        //
        // Otherwise the shortcut fires on ill-posed input whenever the
        // target equals ANY domain point — duplicate value or not.
        let weights = barycentric_weights(x_coords)?;

        // If the target matches a domain point, return that row directly.
        // This also avoids a zero in the difference vector below.
        for (i, &x) in x_coords.iter().enumerate() {
            if point == EF::from(x) {
                return Some(self.row(i).unwrap().into_iter().map(EF::from).collect());
            }
        }

        // Batch-invert all (point - x_i). Safe: coincidence was ruled out above.
        let diffs: Vec<EF> = x_coords.iter().map(|&x| point - x).collect();
        let diff_invs = batch_multiplicative_inverse(&diffs);

        Some(self.interpolate_arbitrary_with_precomputation(&weights, &diff_invs))
    }

    /// Evaluates every column polynomial at a target point with precomputed data.
    ///
    /// Hot path: O(n * width) per call when weights are reused across targets.
    ///
    /// # Safety
    ///
    /// - The evaluation point `z` must not equal any domain point `x_i`.
    /// - `weights[i]` must be the barycentric weight for `x_i`,
    ///   i.e. `1 / prod_{j != i} (x_i - x_j)`.
    /// - `diff_invs[i]` must be `1 / (z - x_i)`.
    ///
    /// # Panics
    ///
    /// Debug-panics if the slices differ in length from the matrix height.
    fn interpolate_arbitrary_with_precomputation<EF: ExtensionField<F>>(
        &self,
        weights: &[F],
        diff_invs: &[EF],
    ) -> Vec<EF> {
        debug_assert_eq!(weights.len(), self.height());
        debug_assert_eq!(diff_invs.len(), self.height());

        // Barycentric second form:
        //
        //     s_i    = w_i / (z - x_i)
        //     f_j(z) = [sum_i  s_i * M[i][j]]  /  [sum_i  s_i]
        //
        // The numerator vector is M^T * col_scale (one dot product per column).
        // The denominator is a single scalar shared across all columns.

        // Per-row scale factor: s_i = w_i * diff_inv_i.
        let col_scale: Vec<EF> = weights
            .iter()
            .zip(diff_invs)
            .map(|(&w, &d)| d * w)
            .collect();

        // Denominator: sum of all scale factors.
        let denominator = col_scale.iter().copied().fold(EF::ZERO, |a, b| a + b);
        let denom_inv = denominator.inverse();

        // Numerator per column via SIMD-packed M^T * col_scale.
        let mut evals = self.columnwise_dot_product(&col_scale);

        // Divide every column result by the shared denominator.
        scale_slice_in_place_single_core(&mut evals, denom_inv);
        evals
    }

    /// Recovers coefficient vectors for every column via batched Newton interpolation.
    ///
    /// Each row of `self` holds evaluations at the corresponding domain point.
    /// Returns an n * width matrix where row i holds degree-i coefficients.
    ///
    /// # Performance
    ///
    /// - O(n^2 * width) field operations.
    /// - O(n + width) auxiliary memory, zero allocations inside the main loop.
    ///
    /// # Returns
    ///
    /// `None` if any domain points coincide.
    fn recover_coefficients(&self, x_coords: &[F]) -> Option<RowMajorMatrix<F>> {
        let n = self.height();
        let w = self.width();
        debug_assert_eq!(x_coords.len(), n);

        if n == 0 {
            return Some(RowMajorMatrix::new(Vec::new(), w.max(1)));
        }

        // Row i of result will hold the degree-i coefficients for all w polynomials.
        let mut result = RowMajorMatrix::new(F::zero_vec(n * w), w);

        // Shared Newton basis polynomial B_k(x) = prod_{i<k} (x - x_i).
        // Stored in expanded coefficient form; starts as the constant 1.
        let mut basis = F::zero_vec(n);
        basis[0] = F::ONE;

        // Per-column scratch buffer, reused every iteration to avoid allocations.
        let mut scratch = F::zero_vec(w);

        for k in 0..n {
            let x_k = x_coords[k];

            // Evaluate B_k(x_k) directly from the roots: prod_{i<k} (x_k - x_i).
            // Cheaper than Horner on the expanded coefficients because it
            // touches only the domain array (sequential access, no dependency
            // chain on the basis coefficient array).
            let mut b_xk = F::ONE;
            for &x_i in &x_coords[..k] {
                b_xk *= x_k - x_i;
            }
            // Zero means x_k duplicates an earlier domain point.
            let b_xk_inv = b_xk.try_inverse()?;

            // Horner-evaluate all w result polynomials at x_k.
            // Process whole rows (= ascending degree) for row-major cache locality.
            //
            //     scratch_j = result[k-1][j] * x_k + result[k-2][j] * x_k + ...
            //               = P_j(x_k)
            scratch.fill(F::ZERO);
            for i in (0..k).rev() {
                let row = result.row_slice(i).unwrap();
                for j in 0..w {
                    scratch[j] = scratch[j] * x_k + row[j];
                }
            }

            // Newton correction: c_j = (y_{k,j} - P_j(x_k)) / B_k(x_k).
            // Stream the evaluation row via its iterator — no heap allocation.
            for (j, y_kj) in self.row(k).unwrap().into_iter().enumerate() {
                scratch[j] = (y_kj - scratch[j]) * b_xk_inv;
            }

            // Accumulate: result[i][j] += c_j * basis[i].
            for (i, &b_i) in basis.iter().enumerate().take(k + 1) {
                let row = result.row_mut(i);
                for j in 0..w {
                    row[j] += scratch[j] * b_i;
                }
            }

            // Extend basis: B_{k+1}(x) = B_k(x) * (x - x_k).
            // Process high-to-low so each coefficient is read before overwritten.
            //
            //     new[k+1] = b_k
            //     new[i]   = b_{i-1} - x_k * b_i    for i = k, ..., 1
            //     new[0]   = -x_k * b_0
            if k + 1 < n {
                basis[k + 1] = basis[k];
            }
            for i in (1..=k).rev() {
                basis[i] = basis[i - 1] - x_k * basis[i];
            }
            basis[0] = -x_k * basis[0];
        }

        Some(result)
    }
}

impl<F: Field, M: Matrix<F>> InterpolateArbitrary<F> for M {}

/// Interpolates a single polynomial from (x, y) pairs.
///
/// Returns coefficients in ascending degree order (index i = coefficient of x^i).
/// Convenience wrapper that builds a one-column matrix and delegates to the
/// batched Newton implementation.
///
/// # Performance
///
/// O(n^2) field operations, O(n) auxiliary memory.
///
/// # Returns
///
/// `None` if any two x-coordinates coincide.
pub fn interpolate_lagrange<F: Field>(points: &[(F, F)]) -> Option<Vec<F>> {
    if points.is_empty() {
        return Some(Vec::new());
    }
    // Split into separate domain and evaluation vectors.
    let (xs, ys): (Vec<F>, Vec<F>) = points.iter().copied().unzip();
    // Build a single-column matrix and recover coefficients via Newton.
    let evals = RowMajorMatrix::new_col(ys);
    Some(evals.recover_coefficients(&xs)?.values)
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use p3_baby_bear::BabyBear;
    use p3_field::extension::BinomialExtensionField;
    use p3_field::{
        BasedVectorSpace, ExtensionField, Field, PrimeCharacteristicRing, TwoAdicField,
        batch_multiplicative_inverse,
    };
    use p3_util::log2_strict_usize;
    use proptest::prelude::*;

    use super::*;
    use crate::dense::RowMajorMatrix;

    type F = BabyBear;
    type EF4 = BinomialExtensionField<BabyBear, 4>;

    /// Evaluate a polynomial (given by coefficients) at a point using Horner's method.
    ///
    /// Horner's method: `f(z) = c_0 + z*(c_1 + z*(c_2 + ...))`.
    /// Processes coefficients from highest degree down to constant term.
    fn eval_poly<EF: ExtensionField<F>>(coeffs: &[F], point: EF) -> EF {
        coeffs
            .iter()
            .rev()
            .fold(EF::ZERO, |acc, &c| acc * point + c)
    }

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
        // Manually build the coset elements and adjusted weights.
        let n = evals.len();
        let k = log2_strict_usize(n);

        // Coset elements: {shift * h^0, shift * h^1, ..., shift * h^{N-1}}.
        let coset = F::two_adic_generator(k).shifted_powers(shift).collect_n(n);

        // Inverse denominators: 1/(z - coset_i) for each coset element.
        let denom: Vec<_> = coset.iter().map(|&w| point - w).collect();
        let denom = batch_multiplicative_inverse(&denom);

        // Adjusted weights: 1/(z - coset_i) - 1/z.
        let adjusted = compute_adjusted_weights(point, &denom);

        // The precomputation variant must produce the same result.
        let result = evals_mat.interpolate_coset_with_precomputation(shift, point, &adjusted);
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
            //     interpolate_coset                     (builds coset + adjusted weights internally)
            //     interpolate_coset_with_precomputation (caller provides adjusted weights)
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
            let adjusted = compute_adjusted_weights(point, &diff_invs);
            let result_precomp = evals_mat
                .interpolate_coset_with_precomputation(shift, point, &adjusted);

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

    #[test]
    fn test_barycentric_weights_empty() {
        assert_eq!(barycentric_weights::<F>(&[]), Some(vec![]));
    }

    #[test]
    fn test_barycentric_weights_duplicates() {
        let xs = [F::from_u32(1), F::from_u32(2), F::from_u32(1)];
        assert_eq!(barycentric_weights(&xs), None);
    }

    #[test]
    fn test_barycentric_weights_known() {
        // For x = {0, 1, 2}:
        //   w_0 = 1/((0-1)(0-2)) = 1/2
        //   w_1 = 1/((1-0)(1-2)) = -1
        //   w_2 = 1/((2-0)(2-1)) = 1/2
        let xs = [F::from_u32(0), F::from_u32(1), F::from_u32(2)];
        let ws = barycentric_weights(&xs).unwrap();
        let half = F::TWO.inverse();
        assert_eq!(ws, vec![half, -F::ONE, half]);
    }

    #[test]
    fn test_interpolate_arbitrary_known_quadratic() {
        // f(x) = x^2 + 2x + 3.  Evaluate at x = 0, 1, 2 → y = 3, 6, 11.
        // Then interpolate at x = 100 → f(100) = 10203.
        let xs = [F::from_u32(0), F::from_u32(1), F::from_u32(2)];
        let evals = RowMajorMatrix::new(vec![F::from_u32(3), F::from_u32(6), F::from_u32(11)], 1);
        let result = evals.interpolate_arbitrary_point(&xs, F::from_u32(100));
        assert_eq!(result, Some(vec![F::from_u32(10203)]));
    }

    #[test]
    fn test_interpolate_arbitrary_point_on_domain() {
        // If we evaluate at a domain point, should return that row directly.
        let xs = [F::from_u32(0), F::from_u32(1), F::from_u32(2)];
        let evals = RowMajorMatrix::new(vec![F::from_u32(3), F::from_u32(6), F::from_u32(11)], 1);
        let result = evals.interpolate_arbitrary_point(&xs, F::from_u32(1));
        assert_eq!(result, Some(vec![F::from_u32(6)]));
    }

    #[test]
    fn test_interpolate_arbitrary_duplicates() {
        let xs = [F::from_u32(1), F::from_u32(1)];
        let evals = RowMajorMatrix::new(vec![F::from_u32(5), F::from_u32(7)], 1);
        assert_eq!(
            evals.interpolate_arbitrary_point(&xs, F::from_u32(42)),
            None
        );
    }

    #[test]
    fn test_interpolate_arbitrary_duplicates_target_on_duplicate() {
        // Invariant:
        // Barycentric Lagrange interpolation requires pairwise-distinct domain points.
        // A duplicate makes the problem ill-posed → contract returns `None`.
        //
        // Fixture state: 3 evaluations, collision at indices 0 and 2.
        //
        //     i:    0     1     2
        //     x:    1     2     1     ← duplicate at indices 0 and 2
        //     y:    10    20    30    ← rows disagree at the duplicate
        //
        // Mutation: target = 1, hitting the duplicate value.
        //
        // The first-match-on-domain shortcut would return row 0 = [10];
        // duplicate detection must beat the shortcut and yield `None`.
        let xs = [F::ONE, F::TWO, F::ONE];
        let evals = RowMajorMatrix::new(vec![F::from_u32(10), F::from_u32(20), F::from_u32(30)], 1);
        assert_eq!(evals.interpolate_arbitrary_point(&xs, F::ONE), None);
    }

    #[test]
    fn test_interpolate_arbitrary_duplicates_target_on_unique() {
        // Invariant:
        // Barycentric Lagrange interpolation requires pairwise-distinct domain points.
        // A duplicate makes the problem ill-posed → contract returns `None`.
        //
        // Fixture state: 3 evaluations, collision at indices 0 and 2.
        //
        //     i:    0     1     2
        //     x:    1     2     1     ← duplicate at indices 0 and 2
        //     y:    10    20    30
        //
        // Mutation: target = 2, hitting the unique value at i=1.
        //
        // The first-match-on-domain shortcut would return row 1 = [20];
        // duplicate detection must beat the shortcut and yield `None`.
        let xs = [F::ONE, F::TWO, F::ONE];
        let evals = RowMajorMatrix::new(vec![F::from_u32(10), F::from_u32(20), F::from_u32(30)], 1);
        assert_eq!(evals.interpolate_arbitrary_point(&xs, F::TWO), None);
    }

    #[test]
    fn test_interpolate_arbitrary_multi_column() {
        // f1(x) = x^2 + 2x + 3,  f2(x) = 4x^2 + 5x + 6.
        // Evaluate both at x = 0, 1, 2.
        let xs = [F::from_u32(0), F::from_u32(1), F::from_u32(2)];
        let evals = RowMajorMatrix::new(
            vec![
                F::from_u32(3),
                F::from_u32(6), // row 0: f1(0)=3, f2(0)=6
                F::from_u32(6),
                F::from_u32(15), // row 1: f1(1)=6, f2(1)=15
                F::from_u32(11),
                F::from_u32(32), // row 2: f1(2)=11, f2(2)=32
            ],
            2,
        );
        let result = evals
            .interpolate_arbitrary_point(&xs, F::from_u32(100))
            .unwrap();
        // f1(100) = 10203, f2(100) = 40506
        assert_eq!(result, vec![F::from_u32(10203), F::from_u32(40506)]);
    }

    #[test]
    fn test_interpolate_arbitrary_with_precomputation_equivalence() {
        let xs = [F::from_u32(0), F::from_u32(1), F::from_u32(2)];
        let evals = RowMajorMatrix::new(vec![F::from_u32(3), F::from_u32(6), F::from_u32(11)], 1);

        let point = F::from_u32(100);
        let standard = evals.interpolate_arbitrary_point(&xs, point).unwrap();

        let weights = barycentric_weights(&xs).unwrap();
        let diffs: Vec<F> = xs.iter().map(|&x| point - x).collect();
        let diff_invs = batch_multiplicative_inverse(&diffs);
        let precomp = evals.interpolate_arbitrary_with_precomputation(&weights, &diff_invs);

        assert_eq!(standard, precomp);
    }

    #[test]
    fn test_interpolate_arbitrary_extension_point() {
        // f(x) = x^2 + 2x + 3, evaluated at x = 0, 1, 2.
        let xs = [F::from_u32(0), F::from_u32(1), F::from_u32(2)];
        let evals = RowMajorMatrix::new(vec![F::from_u32(3), F::from_u32(6), F::from_u32(11)], 1);

        // Evaluate at a non-trivial extension point and compare against direct Horner.
        let point = EF4::GENERATOR;
        let result = evals.interpolate_arbitrary_point(&xs, point).unwrap();

        let expected = point * point + point * F::TWO + EF4::from(F::from_u32(3));
        assert_eq!(result, vec![expected]);
    }

    #[test]
    fn test_interpolate_arbitrary_extension_point_on_domain() {
        // EF4 target lies in the base field domain. Must return the matching row directly.
        let xs = [F::from_u32(0), F::from_u32(1), F::from_u32(2)];
        let evals = RowMajorMatrix::new(vec![F::from_u32(3), F::from_u32(6), F::from_u32(11)], 1);

        let point = EF4::from(F::from_u32(1));
        let result = evals.interpolate_arbitrary_point(&xs, point).unwrap();
        assert_eq!(result, vec![EF4::from(F::from_u32(6))]);
    }

    #[test]
    fn test_recover_coefficients_known_quadratic() {
        // f(x) = x^2 + 2x + 3 → coefficients [3, 2, 1].
        let xs = [F::from_u32(0), F::from_u32(1), F::from_u32(2)];
        let evals = RowMajorMatrix::new(vec![F::from_u32(3), F::from_u32(6), F::from_u32(11)], 1);
        let coeffs = evals.recover_coefficients(&xs).unwrap();
        assert_eq!(
            coeffs.values,
            vec![F::from_u32(3), F::from_u32(2), F::from_u32(1)]
        );
    }

    #[test]
    fn test_recover_coefficients_multi_column() {
        // f1(x) = x^2 + 2x + 3,  f2(x) = 4x^2 + 5x + 6.
        let xs = [F::from_u32(0), F::from_u32(1), F::from_u32(2)];
        let evals = RowMajorMatrix::new(
            vec![
                F::from_u32(3),
                F::from_u32(6), // x=0
                F::from_u32(6),
                F::from_u32(15), // x=1
                F::from_u32(11),
                F::from_u32(32), // x=2
            ],
            2,
        );
        let coeffs = evals.recover_coefficients(&xs).unwrap();
        // Row 0 (constant): [3, 6], Row 1 (linear): [2, 5], Row 2 (quadratic): [1, 4]
        assert_eq!(
            coeffs.values,
            vec![
                F::from_u32(3),
                F::from_u32(6),
                F::from_u32(2),
                F::from_u32(5),
                F::from_u32(1),
                F::from_u32(4),
            ]
        );
    }

    #[test]
    fn test_lagrange_empty() {
        assert_eq!(interpolate_lagrange::<F>(&[]), Some(vec![]));
    }

    #[test]
    fn test_lagrange_single_point() {
        let points = [(F::from_u32(7), F::from_u32(42))];
        assert_eq!(interpolate_lagrange(&points), Some(vec![F::from_u32(42)]));
    }

    #[test]
    fn test_lagrange_known_quadratic() {
        let points = [
            (F::from_u32(0), F::from_u32(3)),
            (F::from_u32(1), F::from_u32(6)),
            (F::from_u32(2), F::from_u32(11)),
        ];
        let coeffs = interpolate_lagrange(&points).unwrap();
        assert_eq!(coeffs, vec![F::from_u32(3), F::from_u32(2), F::from_u32(1)]);
    }

    #[test]
    fn test_lagrange_duplicate_x_returns_none() {
        let points = [
            (F::from_u32(1), F::from_u32(5)),
            (F::from_u32(1), F::from_u32(7)),
        ];
        assert_eq!(interpolate_lagrange(&points), None);
    }

    proptest! {
        #[test]
        fn prop_lagrange_roundtrip(
            n in 1usize..=8,
            coeffs_raw in prop::collection::vec(0u32..2013265921, 1..=8),
        ) {
            let mut coeffs: Vec<F> = coeffs_raw.iter().take(n).map(|&v| F::from_u32(v)).collect();
            coeffs.resize(n, F::ZERO);

            let points: Vec<(F, F)> = (0..n)
                .map(|i| {
                    let x = F::from_u32(i as u32);
                    let y = eval_poly(&coeffs, x);
                    (x, y)
                })
                .collect();

            let recovered = interpolate_lagrange(&points).unwrap();
            prop_assert_eq!(recovered, coeffs);
        }

        #[test]
        fn prop_arbitrary_roundtrip(
            n in 1usize..=8,
            coeffs_raw in prop::collection::vec(0u32..2013265921, 1..=8),
            point_raw in 1u32..2013265921u32,
        ) {
            // Evaluate polynomial at n distinct domain points.
            //
            // Then use the trait method to evaluate at a separate target point.
            let mut coeffs: Vec<F> = coeffs_raw.iter().take(n).map(|&v| F::from_u32(v)).collect();
            coeffs.resize(n, F::ZERO);

            let xs: Vec<F> = (0..n).map(|i| F::from_u32(i as u32)).collect();
            let ys: Vec<F> = xs.iter().map(|&x| eval_poly(&coeffs, x)).collect();
            let evals = RowMajorMatrix::new(ys, 1);

            let point = F::from_u32(point_raw);
            let result = evals.interpolate_arbitrary_point(&xs, point).unwrap();
            let expected = eval_poly(&coeffs, point);
            prop_assert_eq!(result[0], expected);
        }

        #[test]
        fn prop_recover_coefficients_roundtrip(
            n in 1usize..=8,
            coeffs_raw in prop::collection::vec(0u32..2013265921, 1..=8),
        ) {
            let mut coeffs: Vec<F> = coeffs_raw.iter().take(n).map(|&v| F::from_u32(v)).collect();
            coeffs.resize(n, F::ZERO);

            let xs: Vec<F> = (0..n).map(|i| F::from_u32(i as u32)).collect();
            let ys: Vec<F> = xs.iter().map(|&x| eval_poly(&coeffs, x)).collect();
            let evals = RowMajorMatrix::new(ys, 1);

            let recovered = evals.recover_coefficients(&xs).unwrap();
            prop_assert_eq!(recovered.values, coeffs);
        }

        #[test]
        fn prop_arbitrary_batch_equals_individual(
            n in 1usize..=6,
            f_raw in prop::collection::vec(0u32..2013265921, 1..=6),
            g_raw in prop::collection::vec(0u32..2013265921, 1..=6),
            point_raw in 1u32..2013265921u32,
        ) {
            // A 2-column batch must agree with two 1-column evaluations.
            let mut f_coeffs: Vec<F> = f_raw.iter().take(n).map(|&v| F::from_u32(v)).collect();
            let mut g_coeffs: Vec<F> = g_raw.iter().take(n).map(|&v| F::from_u32(v)).collect();
            f_coeffs.resize(n, F::ZERO);
            g_coeffs.resize(n, F::ZERO);

            let xs: Vec<F> = (0..n).map(|i| F::from_u32(i as u32)).collect();
            let f_ys: Vec<F> = xs.iter().map(|&x| eval_poly(&f_coeffs, x)).collect();
            let g_ys: Vec<F> = xs.iter().map(|&x| eval_poly(&g_coeffs, x)).collect();

            // Build 2-column batch matrix.
            let batch_vals: Vec<F> = f_ys.iter().zip(&g_ys)
                .flat_map(|(&f, &g)| vec![f, g])
                .collect();
            let batch_mat = RowMajorMatrix::new(batch_vals, 2);
            let f_mat = RowMajorMatrix::new(f_ys, 1);
            let g_mat = RowMajorMatrix::new(g_ys, 1);

            let point = F::from_u32(point_raw);
            let batch_result = batch_mat.interpolate_arbitrary_point(&xs, point).unwrap();
            let f_result = f_mat.interpolate_arbitrary_point(&xs, point).unwrap()[0];
            let g_result = g_mat.interpolate_arbitrary_point(&xs, point).unwrap()[0];

            prop_assert_eq!(batch_result[0], f_result);
            prop_assert_eq!(batch_result[1], g_result);
        }

        #[test]
        fn prop_precomputation_equivalence_arbitrary(
            n in 1usize..=8,
            coeffs_raw in prop::collection::vec(0u32..2013265921, 1..=8),
            point_raw in 1u32..2013265921u32,
        ) {
            // Standard path and precomputation path must agree.
            let mut coeffs: Vec<F> = coeffs_raw.iter().take(n).map(|&v| F::from_u32(v)).collect();
            coeffs.resize(n, F::ZERO);

            let xs: Vec<F> = (0..n).map(|i| F::from_u32(i as u32)).collect();
            let ys: Vec<F> = xs.iter().map(|&x| eval_poly(&coeffs, x)).collect();
            let evals = RowMajorMatrix::new(ys, 1);

            let point = F::from_u32(point_raw);
            let standard = evals.interpolate_arbitrary_point(&xs, point).unwrap();

            let weights = barycentric_weights(&xs).unwrap();
            let diffs: Vec<F> = xs.iter().map(|&x| point - x).collect();

            // Skip if point coincides with a domain point (diff_invs would panic).
            if diffs.iter().any(|d| d.is_zero()) {
                return Ok(());
            }

            let diff_invs = batch_multiplicative_inverse(&diffs);
            let precomp = evals.interpolate_arbitrary_with_precomputation(&weights, &diff_invs);
            prop_assert_eq!(standard, precomp);
        }

        #[test]
        fn prop_arbitrary_roundtrip_extension_point(
            n in 1usize..=8,
            coeffs_raw in prop::collection::vec(0u32..2013265921, 1..=8),
            point_raw in prop::collection::vec(0u32..2013265921, 4..=4),
        ) {
            // Round-trip with the target point taken from EF4: evaluate over a base-field
            // domain, interpolate at an extension-field point, and compare against direct
            // Horner evaluation in the extension.
            let mut coeffs: Vec<F> = coeffs_raw.iter().take(n).map(|&v| F::from_u32(v)).collect();
            coeffs.resize(n, F::ZERO);

            let xs: Vec<F> = (0..n).map(|i| F::from_u32(i as u32)).collect();
            let ys: Vec<F> = xs.iter().map(|&x| eval_poly(&coeffs, x)).collect();
            let evals = RowMajorMatrix::new(ys, 1);

            let point = EF4::from_basis_coefficients_iter(
                point_raw.iter().map(|&v| F::from_u32(v)),
            ).unwrap();
            let result = evals.interpolate_arbitrary_point(&xs, point).unwrap();
            let expected: EF4 = eval_poly(&coeffs, point);
            prop_assert_eq!(result[0], expected);
        }
    }
}
