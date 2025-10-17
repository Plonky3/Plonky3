use alloc::vec::Vec;
use core::ops::{Index, Range};

use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_interpolation::interpolate_subgroup;
use rand::Rng;
use rand::distr::{Distribution, StandardUniform};

use crate::evals::EvaluationsList;

/// A point `(x_1, ..., x_n)` in `F^n` for some field `F`.
#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub struct MultilinearPoint<F>(pub(crate) Vec<F>);

impl<F> MultilinearPoint<F>
where
    F: Field,
{
    /// Construct a new `MultilinearPoint` from a vector of field elements.
    pub const fn new(coords: Vec<F>) -> Self {
        Self(coords)
    }

    /// Returns the number of variables (dimension `n`).
    #[inline]
    #[must_use]
    pub const fn num_variables(&self) -> usize {
        self.0.len()
    }

    /// Return a reference to the slice of field elements
    /// defining the point.
    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &[F] {
        &self.0
    }

    /// Return an iterator over the field elements making up the point.
    #[inline]
    pub fn iter(&self) -> core::slice::Iter<'_, F> {
        self.0.iter()
    }

    /// Return a sub-point over the specified range of variables.
    #[inline]
    #[must_use]
    pub fn get_subpoint_over_range(&self, idx: Range<usize>) -> Self {
        Self(self.0[idx].to_vec())
    }

    /// Return a reference to the last variable in the point, if it exists.
    ///
    /// Returns None if the point is empty.
    #[inline]
    #[must_use]
    pub fn last_variable(&self) -> Option<&F> {
        self.0.last()
    }

    /// Converts a univariate evaluation point into a multilinear one.
    ///
    /// Uses the bijection:
    /// ```ignore
    /// f(x_1, ..., x_n) <-> g(y) := f(y^(2^(n-1)), ..., y^4, y^2, y)
    /// ```
    /// Meaning:
    /// ```ignore
    /// x_1^i_1 * ... * x_n^i_n <-> y^i
    /// ```
    /// where `(i_1, ..., i_n)` is the **big-endian** binary decomposition of `i`.
    ///
    /// Reversing the order ensures the **big-endian** convention.
    pub fn expand_from_univariate(point: F, num_variables: usize) -> Self {
        let mut res: Vec<F> = F::zero_vec(num_variables);
        let mut cur = point;

        // Fill big-endian: [y^(2^(n-1)), ..., y^2, y]
        // Loop from the last index down to the first.
        for i in (0..num_variables).rev() {
            res[i] = cur;
            cur = cur.square();
        }

        Self(res)
    }

    /// Computes `eq(c, p)`, where `p` is another `MultilinearPoint`.
    ///
    /// The **equality polynomial** for two vectors is:
    /// ```ignore
    /// eq(s1, s2) = ∏ (s1_i * s2_i + (1 - s1_i) * (1 - s2_i))
    /// ```
    #[must_use]
    pub fn eq_poly(&self, point: &Self) -> F {
        assert_eq!(self.num_variables(), point.num_variables());

        let mut acc = F::ONE;

        for (&l, &r) in self.into_iter().zip(point) {
            // This uses the algebraic identity:
            // l * r + (1 - l) * (1 - r) = 1 + 2 * l * r - l - r
            // to avoid unnecessary multiplications.
            acc *= F::ONE + l * r.double() - l - r;
        }

        acc
    }

    /// Evaluates the equality polynomial `eq(self, X)` at a folded challenge point.
    ///
    /// This method is used in protocols that "skip" folding rounds by providing a single challenge
    /// for multiple variables.
    #[must_use]
    pub fn eq_poly_with_skip<EF>(&self, r_all: &MultilinearPoint<EF>, k_skip: usize) -> EF
    where
        F: TwoAdicField,
        EF: TwoAdicField + ExtensionField<F>,
    {
        // The total number of variables `n` is inferred from the challenge point `r_all`
        // and the number of skipped variables `k_skip`.
        let n = r_all.num_variables() + k_skip - 1;

        // The point `self` (z) must be defined over the full n-variable domain.
        assert_eq!(
            self.num_variables(),
            n,
            "Constraint point must have the same number of variables as the full domain"
        );

        // Construct the evaluation table for the polynomial eq_z(X).
        // This creates a list of 2^n values, where only the entry at index `z` is ONE.
        let evals = EvaluationsList::new_from_point(self, F::ONE);

        // Reshape the flat list of 2^n evaluations into a `2^k_skip x 2^(n-k_skip)` matrix.
        // Rows correspond to the skipped variables (X0, ..., X_{k_skip-1}).
        // Columns correspond to the remaining variables.
        let num_remaining_vars = n - k_skip;
        let width = 1 << num_remaining_vars;
        let mat = evals.into_mat(width);

        // Deconstruct the challenge object `r_all`.
        // The last element is the challenge for the `k_skip` variables being folded.
        let r_skip = *r_all
            .last_variable()
            .expect("skip challenge must be present");
        // The first `n - k_skip` elements are for the remaining variables.
        let r_rest = MultilinearPoint::new(r_all.as_slice()[..num_remaining_vars].to_vec());

        // Perform the two-stage evaluation.
        // Fold the skipped variables by interpolating each column at `r_skip`.
        let folded_row = interpolate_subgroup(&mat, r_skip);

        // Evaluate the new, smaller polynomial at the remaining challenges `r_rest`.
        EvaluationsList::new(folded_row).evaluate(&r_rest)
    }
}

impl<F> MultilinearPoint<F>
where
    StandardUniform: Distribution<F>,
{
    pub fn rand<R: Rng>(rng: &mut R, num_variables: usize) -> Self {
        Self((0..num_variables).map(|_| rng.random()).collect())
    }
}

impl<'a, F> IntoIterator for &'a MultilinearPoint<F> {
    type Item = &'a F;
    type IntoIter = alloc::slice::Iter<'a, F>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl<F> IntoIterator for MultilinearPoint<F> {
    type Item = F;
    type IntoIter = alloc::vec::IntoIter<F>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<F> Index<usize> for MultilinearPoint<F> {
    type Output = F;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

#[cfg(test)]
#[allow(clippy::identity_op, clippy::erasing_op)]
mod tests {
    use alloc::vec;

    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;
    use p3_matrix::dense::RowMajorMatrix;
    use proptest::prelude::*;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use super::*;

    type F = BabyBear;
    type EF4 = BinomialExtensionField<F, 4>;

    #[test]
    fn test_num_variables() {
        let point = MultilinearPoint::<F>(vec![F::from_u64(1), F::from_u64(0), F::from_u64(1)]);
        assert_eq!(point.num_variables(), 3);
    }

    #[test]
    fn test_expand_from_univariate_single_variable() {
        let point = F::from_u64(3);
        let expanded = MultilinearPoint::expand_from_univariate(point, 1);

        // For n = 1, we expect [y]
        assert_eq!(expanded.0, vec![point]);
    }

    #[test]
    fn test_expand_from_univariate_two_variables() {
        let point = F::from_u64(2);
        let expanded = MultilinearPoint::expand_from_univariate(point, 2);

        // For n = 2, we expect [y^2, y]
        let expected = vec![point * point, point];
        assert_eq!(expanded.0, expected);
    }

    #[test]
    fn test_expand_from_univariate_three_variables() {
        let point = F::from_u64(5);
        let expanded = MultilinearPoint::expand_from_univariate(point, 3);

        // For n = 3, we expect [y^4, y^2, y]
        let expected = vec![point.exp_u64(4), point.exp_u64(2), point];
        assert_eq!(expanded.0, expected);
    }

    #[test]
    fn test_expand_from_univariate_large_variables() {
        let point = F::from_u64(7);
        let expanded = MultilinearPoint::expand_from_univariate(point, 5);

        // For n = 5, we expect [y^16, y^8, y^4, y^2, y]
        let expected = vec![
            point.exp_u64(16),
            point.exp_u64(8),
            point.exp_u64(4),
            point.exp_u64(2),
            point,
        ];
        assert_eq!(expanded.0, expected);
    }

    #[test]
    fn test_expand_from_univariate_identity() {
        let point = F::ONE;
        let expanded = MultilinearPoint::expand_from_univariate(point, 4);

        // Since 1^k = 1 for all k, the result should be [1, 1, 1, 1]
        let expected = vec![F::ONE; 4];
        assert_eq!(expanded.0, expected);
    }

    #[test]
    fn test_expand_from_univariate_zero() {
        let point = F::ZERO;
        let expanded = MultilinearPoint::expand_from_univariate(point, 4);

        // Since 0^k = 0 for all k, the result should be [0, 0, 0, 0]
        let expected = vec![F::ZERO; 4];
        assert_eq!(expanded.0, expected);
    }

    #[test]
    fn test_expand_from_univariate_empty() {
        let point = F::from_u64(9);
        let expanded = MultilinearPoint::expand_from_univariate(point, 0);

        // No variables should return an empty vector
        assert_eq!(expanded.0, vec![]);
    }

    #[test]
    fn test_expand_from_univariate_powers_correctness() {
        let point = F::from_u64(3);
        let expanded = MultilinearPoint::expand_from_univariate(point, 6);

        // For n = 6, we expect [y^32, y^16, y^8, y^4, y^2, y]
        let expected = vec![
            point.exp_u64(32),
            point.exp_u64(16),
            point.exp_u64(8),
            point.exp_u64(4),
            point.exp_u64(2),
            point,
        ];
        assert_eq!(expanded.0, expected);
    }

    #[test]
    fn test_eq_poly_outside_all_zeros() {
        let ml_point1 = MultilinearPoint(vec![F::ZERO; 4]);
        let ml_point2 = MultilinearPoint(vec![F::ZERO; 4]);

        assert_eq!(ml_point1.eq_poly(&ml_point2), F::ONE);
    }

    #[test]
    fn test_eq_poly_outside_all_ones() {
        let ml_point1 = MultilinearPoint(vec![F::ONE; 4]);
        let ml_point2 = MultilinearPoint(vec![F::ONE; 4]);

        assert_eq!(ml_point1.eq_poly(&ml_point2), F::ONE);
    }

    #[test]
    fn test_eq_poly_outside_mixed_match() {
        let ml_point1 = MultilinearPoint(vec![F::ONE, F::ZERO, F::ONE, F::ZERO]);
        let ml_point2 = MultilinearPoint(vec![F::ONE, F::ZERO, F::ONE, F::ZERO]);

        assert_eq!(ml_point1.eq_poly(&ml_point2), F::ONE);
    }

    #[test]
    fn test_eq_poly_outside_mixed_mismatch() {
        let ml_point1 = MultilinearPoint(vec![F::ONE, F::ZERO, F::ONE, F::ZERO]);
        let ml_point2 = MultilinearPoint(vec![F::ZERO, F::ONE, F::ZERO, F::ONE]);

        assert_eq!(ml_point1.eq_poly(&ml_point2), F::ZERO);
    }

    #[test]
    fn test_eq_poly_outside_single_variable_match() {
        let ml_point1 = MultilinearPoint(vec![F::ONE]);
        let ml_point2 = MultilinearPoint(vec![F::ONE]);

        assert_eq!(ml_point1.eq_poly(&ml_point2), F::ONE);
    }

    #[test]
    fn test_eq_poly_outside_single_variable_mismatch() {
        let ml_point1 = MultilinearPoint(vec![F::ONE]);
        let ml_point2 = MultilinearPoint(vec![F::ZERO]);

        assert_eq!(ml_point1.eq_poly(&ml_point2), F::ZERO);
    }

    #[test]
    fn test_eq_poly_outside_manual_comparison() {
        // Construct the first multilinear point with arbitrary non-binary field values
        let x00 = F::from_u8(17);
        let x01 = F::from_u8(56);
        let x02 = F::from_u8(5);
        let x03 = F::from_u8(12);
        let ml_point1 = MultilinearPoint(vec![x00, x01, x02, x03]);

        // Construct the second multilinear point with different non-binary field values
        let x10 = F::from_u8(43);
        let x11 = F::from_u8(5);
        let x12 = F::from_u8(54);
        let x13 = F::from_u8(242);
        let ml_point2 = MultilinearPoint(vec![x10, x11, x12, x13]);

        // Compute the equality polynomial between ml_point1 and ml_point2
        let result = ml_point1.eq_poly(&ml_point2);

        // Manually compute the expected result of the equality polynomial:
        // eq(c, p) = ∏ (c_i * p_i + (1 - c_i) * (1 - p_i))
        // This formula evaluates to 1 iff c_i == p_i for all i, and < 1 otherwise
        let expected = (x00 * x10 + (F::ONE - x00) * (F::ONE - x10))
            * (x01 * x11 + (F::ONE - x01) * (F::ONE - x11))
            * (x02 * x12 + (F::ONE - x02) * (F::ONE - x12))
            * (x03 * x13 + (F::ONE - x03) * (F::ONE - x13));

        // Assert that the method and manual computation yield the same result
        assert_eq!(result, expected);
    }

    #[test]
    fn test_eq_poly_outside_large_match() {
        let ml_point1 = MultilinearPoint(vec![
            F::ONE,
            F::ONE,
            F::ZERO,
            F::ONE,
            F::ZERO,
            F::ONE,
            F::ONE,
            F::ZERO,
        ]);
        let ml_point2 = MultilinearPoint(vec![
            F::ONE,
            F::ONE,
            F::ZERO,
            F::ONE,
            F::ZERO,
            F::ONE,
            F::ONE,
            F::ZERO,
        ]);

        assert_eq!(ml_point1.eq_poly(&ml_point2), F::ONE);
    }

    #[test]
    fn test_eq_poly_outside_large_mismatch() {
        let ml_point1 = MultilinearPoint(vec![
            F::ONE,
            F::ONE,
            F::ZERO,
            F::ONE,
            F::ZERO,
            F::ONE,
            F::ONE,
            F::ZERO,
        ]);
        let ml_point2 = MultilinearPoint(vec![
            F::ONE,
            F::ONE,
            F::ZERO,
            F::ONE,
            F::ZERO,
            F::ONE,
            F::ONE,
            F::ONE, // Last bit differs
        ]);

        assert_eq!(ml_point1.eq_poly(&ml_point2), F::ZERO);
    }

    #[test]
    fn test_eq_poly_outside_empty_vector() {
        let ml_point1 = MultilinearPoint::<F>(vec![]);
        let ml_point2 = MultilinearPoint::<F>(vec![]);

        assert_eq!(ml_point1.eq_poly(&ml_point2), F::ONE);
    }

    #[test]
    #[should_panic]
    fn test_eq_poly_outside_different_lengths() {
        let ml_point1 = MultilinearPoint(vec![F::ONE, F::ZERO]);
        let ml_point2 = MultilinearPoint(vec![F::ONE, F::ZERO, F::ONE]);

        // Should panic because lengths do not match
        let _ = ml_point1.eq_poly(&ml_point2);
    }

    #[test]
    fn test_multilinear_point_rand_not_all_same() {
        const K: usize = 20; // Number of trials
        const N: usize = 10; // Number of variables

        let mut rng = SmallRng::seed_from_u64(1);

        let mut all_same_count = 0;

        for _ in 0..K {
            let point = MultilinearPoint::<F>::rand(&mut rng, N);
            let first = point.0[0];

            // Check if all coordinates are the same as the first one
            if point.into_iter().all(|x| x == first) {
                all_same_count += 1;
            }
        }

        // If all K trials are completely uniform, the RNG is suspicious
        assert!(
            all_same_count < K,
            "rand generated uniform points in all {K} trials"
        );
    }

    proptest! {
        #[test]
        fn proptest_eq_poly_outside_matches_manual(
            (coords1, coords2) in prop::collection::vec(0u8..=250, 1..=8).prop_flat_map(|v1| {
                let len = v1.len();
                prop::collection::vec(0u8..=250, len).prop_map(move |v2| (v1.clone(), v2))
            })
        ) {
            // Convert both u8 vectors to field elements
            let p1 = MultilinearPoint(coords1.iter().copied().map(F::from_u8).collect());
            let p2 = MultilinearPoint(coords2.iter().copied().map(F::from_u8).collect());

            // Evaluate eq_poly
            let result = p1.eq_poly(&p2);

            // Compute expected value using manual formula:
            // eq(c, p) = ∏ (c_i * p_i + (1 - c_i)(1 - p_i))
            let expected = p1.into_iter().zip(p2).fold(F::ONE, |acc, (a, b)| {
                acc * (a * b + (F::ONE - a) * (F::ONE - b))
            });

            prop_assert_eq!(result, expected);
        }
    }

    #[test]
    fn test_eq_poly_with_skip() {
        // SETUP:
        // - n = 3 total variables: (X0, X1, X2).
        // - The constraint point `z` is defined over the full n=3 variables.
        // - k_skip = 2 variables to skip: X0, X1.
        let n = 3;
        let k_skip = 2;

        // The weight polynomial is W(X) = eq_z(X0, X1, X2), where z=(2,3,4).
        // The constraint point MUST be full-dimensional.
        let point = MultilinearPoint::new(vec![F::from_u32(2), F::from_u32(3), F::from_u32(4)]);

        // The verifier's full challenge object `r_all`.
        // It has (n - k_skip) + 1 = (3 - 2) + 1 = 2 elements.
        // - r_rest for remaining variable (X2).
        // - r_skip for the combined (X0, X1) domain.
        let r_rest = MultilinearPoint::new(vec![EF4::from_u32(5)]);
        let r_skip = EF4::from_u32(7);
        let r_all = MultilinearPoint::new([r_rest.as_slice(), &[r_skip]].concat());

        // ACTION: Compute W(r) using the function under test.
        let result = point.eq_poly_with_skip(&r_all, k_skip);

        // MANUAL VERIFICATION:
        // 1. Manually construct the full 8-element table for W(X) = eq_z(X0, X1, X2).
        let z0 = EF4::from(point.as_slice()[0]);
        let z1 = EF4::from(point.as_slice()[1]);
        let z2 = EF4::from(point.as_slice()[2]);
        let mut full_evals_vec = Vec::with_capacity(1 << n);
        for i in 0..(1 << n) {
            // Index `i` corresponds to the point (x0, x1, x2)
            let x0 = EF4::from_u32((i >> 2) & 1);
            let x1 = EF4::from_u32((i >> 1) & 1);
            let x2 = EF4::from_u32(i & 1);
            let term0 = z0 * x0 + (EF4::ONE - z0) * (EF4::ONE - x0);
            let term1 = z1 * x1 + (EF4::ONE - z1) * (EF4::ONE - x1);
            let term2 = z2 * x2 + (EF4::ONE - z2) * (EF4::ONE - x2);
            full_evals_vec.push(term0 * term1 * term2);
        }

        // Reshape into a 4x2 matrix (Rows: (X0,X1), Cols: X2).
        let num_remaining = n - k_skip;
        let mat = RowMajorMatrix::new(full_evals_vec, 1 << num_remaining);

        // Interpolate each column at r_skip to fold the (X0, X1) variables.
        let folded_row = interpolate_subgroup(&mat, r_skip);

        // The `folded_row` is a new 1-variable polynomial, W'(X2).
        let final_poly = EvaluationsList::new(folded_row);

        // Evaluate this final polynomial at the remaining challenge, r_rest.
        let expected = final_poly.evaluate(&r_rest);

        assert_eq!(
            result, expected,
            "Manual skip evaluation for Evaluation weight should match"
        );
    }

    #[test]
    fn test_eq_poly_with_skip_evaluation_all_vars() {
        // SETUP:
        // - n = 5 total variables: (X0, X1, X2, X3, X4).
        // - The constraint point `z` is defined over the full n=5 variables.
        // - k_skip = 5 variables to skip (all of them).
        // - This leaves 0 remaining variables.
        let n = 5;
        let k_skip = 5;

        // The weight polynomial is W(X) = eq_z(X0..X4), where z is a random 5-element point.
        let point = MultilinearPoint::new(vec![
            F::from_u32(2),
            F::from_u32(3),
            F::from_u32(5),
            F::from_u32(7),
            F::from_u32(11),
        ]);

        // The verifier's challenge object `r_all`.
        // It has (n - k_skip) + 1 = (5 - 5) + 1 = 1 element.
        // - r_rest is an empty vector for the 0 remaining variables.
        // - r_skip is the single challenge for the combined (X0..X4) domain.
        let r_rest = MultilinearPoint::new(vec![]);
        let r_skip = EF4::from_u32(13);
        let r_all = MultilinearPoint::new(vec![r_skip]);

        // Compute W(r) using the function under test.
        let result = point.eq_poly_with_skip(&r_all, k_skip);

        // MANUAL VERIFICATION:
        // Manually construct the full 2^5=32 element table for W(X) = eq_z(X0..X4).
        let mut full_evals_vec = Vec::with_capacity(1 << n);
        for i in 0..(1 << n) {
            // The evaluation of eq_z(x) is the product of n individual terms.
            let eq_val: EF4 = (0..n)
                .map(|j| {
                    // Get the j-th coordinate of the constraint point z.
                    let z_j = EF4::from(point.as_slice()[j]);
                    // Get the j-th coordinate of the hypercube point x by checking the j-th bit of i.
                    let x_j = EF4::from_u32((i >> (n - 1 - j)) & 1);
                    // Calculate the j-th term of the product.
                    z_j * x_j + (EF4::ONE - z_j) * (EF4::ONE - x_j)
                })
                .product();
            full_evals_vec.push(eq_val);
        }

        // Reshape into a 32x1 matrix (Rows: (X0..X4), Cols: empty).
        let num_remaining = n - k_skip;
        let mat = RowMajorMatrix::new(full_evals_vec, 1 << num_remaining);

        // Interpolate the single column at r_skip to fold all 5 variables.
        let folded_row = interpolate_subgroup(&mat, r_skip);

        // The `folded_row` is a new 0-variable polynomial (a constant).
        let final_poly = EvaluationsList::new(folded_row);

        // Evaluate this constant polynomial. The point `r_rest` is empty.
        let expected = final_poly.evaluate(&r_rest);

        // The result of interpolation should be a single scalar.
        assert_eq!(
            final_poly.num_evals(),
            1,
            "Folding all variables should result in a single value"
        );

        assert_eq!(
            result, expected,
            "Manual skip evaluation for n=k_skip should match"
        );
    }
}
