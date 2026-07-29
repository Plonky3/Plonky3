use alloc::vec::Vec;
use core::ops::{Index, RangeBounds};
use core::slice::SliceIndex;

use itertools::Itertools;
use p3_field::{ExtensionField, Field};
use p3_matrix::dense::RowMajorMatrix;
use rand::RngExt;
use rand::distr::{Distribution, StandardUniform};

/// A point `(x_1, ..., x_n)` in `F^n` for some field `F`.
#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub struct Point<F>(pub(crate) Vec<F>);

impl<F> Point<F>
where
    F: Field,
{
    /// Construct a new `Point` from a vector of field elements.
    #[must_use]
    pub const fn new(coords: Vec<F>) -> Self {
        Self(coords)
    }

    /// Construct a `Point` corresponding to a vertex of the hypercube.
    ///
    /// Returns `value` encoded big-endian: bit `num_variables - 1 - i` lands at coordinate `i`.
    #[must_use]
    pub fn hypercube(value: usize, num_variables: usize) -> Self {
        assert!(value < (1 << num_variables));
        Self(
            (0..num_variables)
                .map(|i| F::from_bool((value >> (num_variables - 1 - i)) & 1 == 1))
                .collect(),
        )
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
    pub fn get_subpoint_over_range<R: RangeBounds<usize> + SliceIndex<[F], Output = [F]>>(
        &self,
        range: R,
    ) -> Self {
        Self(self.0[range].to_vec())
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

    /// Computes the equality polynomial `eq(p, q)` for two points given as slices.
    ///
    /// The **equality polynomial** for two vectors is:
    /// ```ignore
    /// eq(p, q) = ∏ (p_i * q_i + (1 - p_i) * (1 - q_i))
    /// ```
    ///
    /// This is a static method that avoids allocating `Point` wrappers
    /// when you already have slices.
    ///
    /// # Panics
    /// Panics if `p` and `q` have different lengths.
    #[must_use]
    #[inline]
    pub fn eval_eq<EF: ExtensionField<F>>(p: &[F], q: &[EF]) -> EF {
        assert_eq!(
            p.len(),
            q.len(),
            "Points must have the same number of variables"
        );

        // This uses the algebraic identity:
        // l * r + (1 - l) * (1 - r) = 1 + 2 * l * r - l - r
        // to avoid unnecessary multiplications.
        p.iter()
            .zip(q)
            .map(|(&l, &r)| r.double() * l - l - r + F::ONE)
            .product()
    }

    /// Evaluates the selection polynomial of a point against a univariate domain element.
    ///
    /// The univariate element is expanded into the square-power vector
    /// `(var^(2^(n-1)), ..., var^2, var)`.
    ///
    /// The selection polynomial multiplies one factor per coordinate:
    /// ```text
    /// prod_i (point_i * var_i - point_i + 1)
    /// ```
    #[must_use]
    #[inline]
    pub fn eval_select<EF: ExtensionField<F>>(mut var: F, point: &[EF]) -> EF {
        // Read coordinates from the highest variable down to the lowest.
        // `var` starts at the lowest power and is squared after each factor,
        // so reversed iteration yields the descending powers var^(2^(n-1)), ..., var.
        point
            .iter()
            .rev()
            .map(|&r| {
                // Per-coordinate factor: r * (var - 1) + 1.
                // Equals 1 when var == 1, and equals 1 - r when var == 0.
                let term = r * (F::NEG_ONE + var) + EF::ONE;
                // Advance to the next-higher power for the next coordinate.
                var = var.square();
                term
            })
            // Multiply all per-coordinate factors into the selection value.
            .product()
    }

    /// Evaluates the closed-form carry state of the repeat-last successor map.
    ///
    /// The successor map sends a hypercube row `x` to `x + 1`, with the maximal
    /// row mapping to itself.
    ///
    /// Coordinates are folded from the lowest variable up to the highest while
    /// tracking three accumulators returned as a triple:
    /// - The carry weight that still wants to add 1 into the not-yet-folded prefix.
    /// - The completed contribution from rows whose increment has already settled.
    /// - The boundary weight of the all-ones row, which repeats instead of wrapping.
    ///
    /// Passing the whole point and row folds every coordinate, so the full
    /// successor value is the sum of the completed and boundary accumulators.
    ///
    /// Passing only a low-bit suffix stops early, leaving a live carry weight
    /// that an outer caller threads into the remaining prefix.
    ///
    /// # Panics
    ///
    /// Panics if the two slices have different lengths.
    #[must_use]
    #[inline]
    pub fn eval_next(point: &[F], row: &[F]) -> (F, F, F) {
        // Add 1 bit by bit; both inputs must index the same variables.
        assert_eq!(point.len(), row.len());

        // carry: weight where the +1 has not yet been absorbed (seeded at the low bit).
        let mut carry = F::ONE;
        // done: weight where the +1 has already settled.
        let mut done = F::ZERO;
        // omega: weight of the repeating all-ones row.
        let mut omega = F::ONE;
        // Fold the low bit first so the carry ripples toward higher bits.
        for (&point_bit, &row_bit) in point.iter().zip(row).rev() {
            // Equality factor: 1 when the bits agree, 0 when they differ.
            let eq = row_bit.double() * point_bit - point_bit - row_bit + F::ONE;
            let prev_carry = carry;
            // Carry lives where point_bit = 1, row_bit = 0: a 0 -> 1 still to ripple up.
            carry = prev_carry * point_bit * (F::ONE - row_bit);
            // Carry settles where point_bit = 0, row_bit = 1; settled mass rides the equality factor.
            done = done * eq + prev_carry * (F::ONE - point_bit) * row_bit;
            // All-ones weight: nonzero only where both bits are 1.
            omega *= point_bit * row_bit;
        }
        (carry, done, omega)
    }

    /// Returns a new `Point` with the variables in reversed order.
    #[must_use]
    pub fn reversed(&self) -> Self {
        Self(self.0.iter().rev().copied().collect())
    }

    /// Extends this `Point` in-place by appending the coordinates of `other`.
    pub fn extend(&mut self, other: &Self) {
        self.0.extend_from_slice(&other.0);
    }

    /// Transposes points so same-index variables are aligned in rows.
    pub fn transpose(points: &[Self], rev_order: bool) -> RowMajorMatrix<F> {
        let k = points
            .iter()
            .map(Self::num_variables)
            .all_equal_value()
            .unwrap();
        let n = points.len();
        let mut flat = F::zero_vec(k * n);
        points.iter().enumerate().for_each(|(i, point)| {
            point.iter().enumerate().for_each(|(j, &cur)| {
                if rev_order {
                    flat[(k - 1 - j) * n + i] = cur;
                } else {
                    flat[j * n + i] = cur;
                }
            });
        });
        RowMajorMatrix::new(flat, n)
    }

    /// Given a position splits the point into two sub-points.
    pub fn split_at(&self, pos: usize) -> (Self, Self) {
        let (left, right) = self.0.split_at(pos);
        (Self::new(left.to_vec()), Self::new(right.to_vec()))
    }
}

impl<F> Point<F>
where
    StandardUniform: Distribution<F>,
{
    pub fn rand<R: RngExt>(rng: &mut R, num_variables: usize) -> Self {
        Self((0..num_variables).map(|_| rng.random()).collect())
    }
}

impl<'a, F> IntoIterator for &'a Point<F> {
    type Item = &'a F;
    type IntoIter = core::slice::Iter<'a, F>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl<F> IntoIterator for Point<F> {
    type Item = F;
    type IntoIter = alloc::vec::IntoIter<F>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<F> Index<usize> for Point<F> {
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
    use proptest::prelude::*;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use super::*;

    type F = BabyBear;

    #[test]
    fn eval_next_empty_and_single_variable() {
        // Empty point: no coordinate to fold, so the carry and all-ones weights
        // stay at one and the settled weight at zero.
        assert_eq!(Point::<F>::eval_next(&[], &[]), (F::ONE, F::ZERO, F::ONE));

        // One variable: fold a single coordinate of the point against the row.
        // Hand-derived decomposition for point [p], row [r]:
        //   carry = p * (1 - r)   (a 0 -> 1 still waiting to ripple up)
        //   done  = (1 - p) * r   (the +1 has settled here)
        //   omega = p * r         (the all-ones row weight)
        let p = F::from_u64(2);
        let r = F::from_u64(3);
        let (carry, done, omega) = Point::eval_next(&[p], &[r]);
        assert_eq!(carry, p * (F::ONE - r));
        assert_eq!(done, (F::ONE - p) * r);
        assert_eq!(omega, p * r);

        // Dropping the live carry leaves the full successor value, which for a
        // single variable is just the evaluation-row coordinate.
        assert_eq!(done + omega, r);
    }

    #[test]
    fn test_num_variables() {
        let point = Point::<F>(vec![F::from_u64(1), F::from_u64(0), F::from_u64(1)]);
        assert_eq!(point.num_variables(), 3);
    }

    #[test]
    fn test_expand_from_univariate_single_variable() {
        let point = F::from_u64(3);
        let expanded = Point::expand_from_univariate(point, 1);

        // For n = 1, we expect [y]
        assert_eq!(expanded.0, vec![point]);
    }

    #[test]
    fn test_expand_from_univariate_two_variables() {
        let point = F::from_u64(2);
        let expanded = Point::expand_from_univariate(point, 2);

        // For n = 2, we expect [y^2, y]
        let expected = vec![point * point, point];
        assert_eq!(expanded.0, expected);
    }

    #[test]
    fn test_expand_from_univariate_three_variables() {
        let point = F::from_u64(5);
        let expanded = Point::expand_from_univariate(point, 3);

        // For n = 3, we expect [y^4, y^2, y]
        let expected = vec![point.exp_u64(4), point.exp_u64(2), point];
        assert_eq!(expanded.0, expected);
    }

    #[test]
    fn test_expand_from_univariate_large_variables() {
        let point = F::from_u64(7);
        let expanded = Point::expand_from_univariate(point, 5);

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
        let expanded = Point::expand_from_univariate(point, 4);

        // Since 1^k = 1 for all k, the result should be [1, 1, 1, 1]
        let expected = vec![F::ONE; 4];
        assert_eq!(expanded.0, expected);
    }

    #[test]
    fn test_expand_from_univariate_zero() {
        let point = F::ZERO;
        let expanded = Point::expand_from_univariate(point, 4);

        // Since 0^k = 0 for all k, the result should be [0, 0, 0, 0]
        let expected = F::zero_vec(4);
        assert_eq!(expanded.0, expected);
    }

    #[test]
    fn test_expand_from_univariate_empty() {
        let point = F::from_u64(9);
        let expanded = Point::expand_from_univariate(point, 0);

        // No variables should return an empty vector
        assert_eq!(expanded.0, vec![]);
    }

    #[test]
    fn test_expand_from_univariate_powers_correctness() {
        let point = F::from_u64(3);
        let expanded = Point::expand_from_univariate(point, 6);

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
    fn test_eval_eq_outside_all_zeros() {
        let ml_point1 = Point(F::zero_vec(4));
        let ml_point2 = Point(F::zero_vec(4));

        assert_eq!(
            Point::eval_eq(ml_point1.as_slice(), ml_point2.as_slice()),
            F::ONE
        );
    }

    #[test]
    fn test_eval_eq_outside_all_ones() {
        let ml_point1 = Point(vec![F::ONE; 4]);
        let ml_point2 = Point(vec![F::ONE; 4]);

        assert_eq!(
            Point::eval_eq(ml_point1.as_slice(), ml_point2.as_slice()),
            F::ONE
        );
    }

    #[test]
    fn test_eval_eq_outside_mixed_match() {
        let ml_point1 = Point(vec![F::ONE, F::ZERO, F::ONE, F::ZERO]);
        let ml_point2 = Point(vec![F::ONE, F::ZERO, F::ONE, F::ZERO]);

        assert_eq!(
            Point::eval_eq(ml_point1.as_slice(), ml_point2.as_slice()),
            F::ONE
        );
    }

    #[test]
    fn test_eval_eq_outside_mixed_mismatch() {
        let ml_point1 = Point(vec![F::ONE, F::ZERO, F::ONE, F::ZERO]);
        let ml_point2 = Point(vec![F::ZERO, F::ONE, F::ZERO, F::ONE]);

        assert_eq!(
            Point::eval_eq(ml_point1.as_slice(), ml_point2.as_slice()),
            F::ZERO
        );
    }

    #[test]
    fn test_eval_eq_static_method() {
        // Test that eval_eq works correctly with slices
        let p = [F::from_u64(2), F::from_u64(3), F::from_u64(5)];
        let q = [F::from_u64(7), F::from_u64(11), F::from_u64(13)];

        // Compute using static method
        let result = Point::eval_eq(&p, &q);

        // Compute manually: eq(p,q) = ∏ (p_i * q_i + (1 - p_i) * (1 - q_i))
        let expected = (F::ONE + p[0] * q[0].double() - p[0] - q[0])
            * (F::ONE + p[1] * q[1].double() - p[1] - q[1])
            * (F::ONE + p[2] * q[2].double() - p[2] - q[2]);

        assert_eq!(result, expected);

        // Test that it matches the Point slice API.
        let ml_p = Point::new(p.to_vec());
        let ml_q = Point::new(q.to_vec());
        assert_eq!(result, Point::eval_eq(ml_p.as_slice(), ml_q.as_slice()));
    }

    #[test]
    fn test_eval_eq_outside_single_variable_match() {
        let ml_point1 = Point(vec![F::ONE]);
        let ml_point2 = Point(vec![F::ONE]);

        assert_eq!(
            Point::eval_eq(ml_point1.as_slice(), ml_point2.as_slice()),
            F::ONE
        );
    }

    #[test]
    fn test_eval_eq_outside_single_variable_mismatch() {
        let ml_point1 = Point(vec![F::ONE]);
        let ml_point2 = Point(vec![F::ZERO]);

        assert_eq!(
            Point::eval_eq(ml_point1.as_slice(), ml_point2.as_slice()),
            F::ZERO
        );
    }

    #[test]
    fn test_eval_eq_outside_manual_comparison() {
        // Construct the first multilinear point with arbitrary non-binary field values
        let x00 = F::from_u8(17);
        let x01 = F::from_u8(56);
        let x02 = F::from_u8(5);
        let x03 = F::from_u8(12);
        let ml_point1 = Point(vec![x00, x01, x02, x03]);

        // Construct the second multilinear point with different non-binary field values
        let x10 = F::from_u8(43);
        let x11 = F::from_u8(5);
        let x12 = F::from_u8(54);
        let x13 = F::from_u8(242);
        let ml_point2 = Point(vec![x10, x11, x12, x13]);

        // Compute the equality polynomial between ml_point1 and ml_point2
        let result = Point::eval_eq(ml_point1.as_slice(), ml_point2.as_slice());

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
    fn test_eval_eq_outside_large_match() {
        let ml_point1 = Point(vec![
            F::ONE,
            F::ONE,
            F::ZERO,
            F::ONE,
            F::ZERO,
            F::ONE,
            F::ONE,
            F::ZERO,
        ]);
        let ml_point2 = Point(vec![
            F::ONE,
            F::ONE,
            F::ZERO,
            F::ONE,
            F::ZERO,
            F::ONE,
            F::ONE,
            F::ZERO,
        ]);

        assert_eq!(
            Point::eval_eq(ml_point1.as_slice(), ml_point2.as_slice()),
            F::ONE
        );
    }

    #[test]
    fn test_eval_eq_outside_large_mismatch() {
        let ml_point1 = Point(vec![
            F::ONE,
            F::ONE,
            F::ZERO,
            F::ONE,
            F::ZERO,
            F::ONE,
            F::ONE,
            F::ZERO,
        ]);
        let ml_point2 = Point(vec![
            F::ONE,
            F::ONE,
            F::ZERO,
            F::ONE,
            F::ZERO,
            F::ONE,
            F::ONE,
            F::ONE, // Last bit differs
        ]);

        assert_eq!(
            Point::eval_eq(ml_point1.as_slice(), ml_point2.as_slice()),
            F::ZERO
        );
    }

    #[test]
    fn test_eval_eq_outside_empty_vector() {
        let ml_point1 = Point::<F>(vec![]);
        let ml_point2 = Point::<F>(vec![]);

        assert_eq!(
            Point::eval_eq(ml_point1.as_slice(), ml_point2.as_slice()),
            F::ONE
        );
    }

    #[test]
    #[should_panic]
    fn test_eval_eq_outside_different_lengths() {
        let ml_point1 = Point(vec![F::ONE, F::ZERO]);
        let ml_point2 = Point(vec![F::ONE, F::ZERO, F::ONE]);

        // Should panic because lengths do not match
        let _ = Point::eval_eq(ml_point1.as_slice(), ml_point2.as_slice());
    }

    #[test]
    fn test_multilinear_point_rand_not_all_same() {
        const K: usize = 20; // Number of trials
        const N: usize = 10; // Number of variables

        let mut rng = SmallRng::seed_from_u64(1);

        let mut all_same_count = 0;

        for _ in 0..K {
            let point = Point::<F>::rand(&mut rng, N);
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
        fn proptest_eval_eq_outside_matches_manual(
            (coords1, coords2) in prop::collection::vec(0u8..=250, 1..=8).prop_flat_map(|v1| {
                let len = v1.len();
                prop::collection::vec(0u8..=250, len).prop_map(move |v2| (v1.clone(), v2))
            })
        ) {
            // Convert both u8 vectors to field elements
            let p1 = Point(coords1.iter().copied().map(F::from_u8).collect());
            let p2 = Point(coords2.iter().copied().map(F::from_u8).collect());

            // Evaluate the equality polynomial.
            let result = Point::eval_eq(p1.as_slice(), p2.as_slice());

            // Compute expected value using manual formula:
            // eq(c, p) = ∏ (c_i * p_i + (1 - c_i)(1 - p_i))
            let expected = p1.into_iter().zip(p2).fold(F::ONE, |acc, (a, b)| {
                acc * (a * b + (F::ONE - a) * (F::ONE - b))
            });

            prop_assert_eq!(result, expected);
        }
    }

    #[test]
    fn test_hypercube_zero_vars_returns_empty_point() {
        // {0,1}^0 has one point: the empty tuple.
        let point = Point::<F>::hypercube(0, 0);
        assert_eq!(point.num_variables(), 0);
        assert_eq!(point.as_slice(), &[] as &[F]);
    }

    #[test]
    fn test_hypercube_single_bit_covers_both_values() {
        // num_variables = 1: 0 → [ZERO], 1 → [ONE].
        assert_eq!(Point::<F>::hypercube(0, 1).as_slice(), &[F::ZERO]);
        assert_eq!(Point::<F>::hypercube(1, 1).as_slice(), &[F::ONE]);
    }

    #[test]
    fn test_hypercube_big_endian_layout_hand_computed() {
        // Fixture: value = 5 = 0b101, n = 3.
        //
        //     coord 0 ← bit 2 = 1 → ONE
        //     coord 1 ← bit 1 = 0 → ZERO
        //     coord 2 ← bit 0 = 1 → ONE
        let point = Point::<F>::hypercube(5, 3);
        assert_eq!(point.as_slice(), &[F::ONE, F::ZERO, F::ONE]);
    }

    #[test]
    fn test_hypercube_max_value_is_all_ones() {
        // value = (1 << n) - 1 → every bit is 1 → every coord is ONE.
        for num_variables in 1..=6 {
            let max_value = (1 << num_variables) - 1;
            let point = Point::<F>::hypercube(max_value, num_variables);
            assert_eq!(point.as_slice(), vec![F::ONE; num_variables].as_slice());
        }
    }

    #[test]
    fn test_hypercube_indexes_lexicographic_poly() {
        // Invariant: evaluating a lex-stored poly at the hypercube point
        // for index i returns the i-th stored element.
        use crate::poly::Poly;
        let num_variables = 3;

        // i-th eval = i, so lookups are identifiable by inspection.
        let evals: Vec<F> = (0..(1 << num_variables))
            .map(|i| F::from_u64(i as u64))
            .collect();
        let poly = Poly::new(evals.clone());

        for (i, &expected) in evals.iter().enumerate() {
            let point = Point::<F>::hypercube(i, num_variables);
            assert_eq!(poly.eval_base(&point), expected);
        }
    }

    #[test]
    #[should_panic]
    fn test_hypercube_panics_on_out_of_range_value() {
        // Precondition: value < 1 << num_variables.
        // Mutation: first out-of-range value.
        let _ = Point::<F>::hypercube(1 << 3, 3);
    }

    proptest! {
        #[test]
        fn proptest_hypercube_shape_and_values(num_variables in 0usize..=8, seed in any::<u64>()) {
            // Invariants:
            //   1. length == num_variables.
            //   2. every coord ∈ {ZERO, ONE}.
            let value = if num_variables == 0 { 0 } else { (seed as usize) % (1 << num_variables) };
            let point = Point::<F>::hypercube(value, num_variables);

            prop_assert_eq!(point.num_variables(), num_variables);
            for &c in point.as_slice() {
                prop_assert!(c == F::ZERO || c == F::ONE);
            }
        }

        #[test]
        fn proptest_hypercube_matches_bit_decomposition(
            num_variables in 1usize..=10,
            seed in any::<u64>(),
        ) {
            // Invariant: coord i == bit (n - 1 - i) of value.
            let value = (seed as usize) % (1 << num_variables);
            let point = Point::<F>::hypercube(value, num_variables);

            for (i, &coord) in point.as_slice().iter().enumerate() {
                let bit = (value >> (num_variables - 1 - i)) & 1;
                let expected = if bit == 1 { F::ONE } else { F::ZERO };
                prop_assert_eq!(coord, expected);
            }
        }
    }
}
