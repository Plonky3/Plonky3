use core::ops::{Index, Range};

use p3_field::Field;
use rand::Rng;
use rand::distr::{Distribution, StandardUniform};

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
    type IntoIter = std::slice::Iter<'a, F>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl<F> IntoIterator for MultilinearPoint<F> {
    type Item = F;
    type IntoIter = std::vec::IntoIter<F>;
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
    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;
    use proptest::prelude::*;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use super::*;

    type F = BabyBear;

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
}
