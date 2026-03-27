//! Lagrange interpolation utilities for sumcheck protocols.
//!
//! This module provides functions for evaluating quadratic polynomials using
//! Lagrange basis interpolation at the points `{0, 1, 2}`.
//!
//! # Mathematical Background
//!
//! Given evaluations of a quadratic polynomial `h(x)` at three points `{0, 1, 2}`,
//! we can reconstruct `h(r)` for any field element `r` using Lagrange interpolation:
//!
//! ```text
//! h(r) = h(0) * L_0(r) + h(1) * L_1(r) + h(2) * L_2(r)
//! ```
//!
//! where `L_i` are the Lagrange basis polynomials satisfying `L_i(j) = delta_{i,j}`.

use alloc::vec::Vec;

use p3_field::Field;

/// Computes the Lagrange basis weights for interpolation at points `{0, 1, 2}`.
///
/// Given a field element `r`, returns the array `[L_0(r), L_1(r), L_2(r)]` where
/// `L_i` are the Lagrange basis polynomials for the interpolation set `{0, 1, 2}`.
///
/// # Lagrange Basis Formulas
///
/// The Lagrange basis polynomials for interpolation at `{0, 1, 2}` are:
///
/// ```text
/// L_0(x) = (x - 1)(x - 2) / ((0 - 1)(0 - 2)) = (x - 1)(x - 2) / 2
/// L_1(x) = (x - 0)(x - 2) / ((1 - 0)(1 - 2)) = -x(x - 2) = x(2 - x)
/// L_2(x) = (x - 0)(x - 1) / ((2 - 0)(2 - 1)) = x(x - 1) / 2
/// ```
///
/// # Properties
///
/// The basis polynomials satisfy:
/// - `L_i(j) = 1` if `i == j`, else `0` (Kronecker delta)
/// - `L_0(r) + L_1(r) + L_2(r) = 1` (partition of unity)
///
/// # Arguments
///
/// * `r` - The evaluation point
///
/// # Returns
///
/// Array of three field elements `[L_0(r), L_1(r), L_2(r)]`.
fn lagrange_weights_012<F: Field>(r: F) -> [F; 3] {
    let r_minus_one = r - F::ONE;

    // L_0(r) = (r - 1)(r - 2) / 2
    let l0 = (r_minus_one * (r - F::TWO)).halve();

    // L_1(r) = r(2 - r)
    //
    // Derived from: -r(r - 2) = r(2 - r)
    let l1 = r * (F::TWO - r);

    // L_2(r) = r(r - 1) / 2
    let l2 = (r * r_minus_one).halve();

    [l0, l1, l2]
}

/// Computes tensor product Lagrange weights for multivariate interpolation.
///
/// Given a vector of evaluation points `rs = [r_0, r_1, ..., r_{k-1}]`, computes
/// the `3^k` Lagrange weights for interpolating over the grid `{0, 1, 2}^k`.
///
/// # Mathematical Background
///
/// For multivariate interpolation, the basis functions are tensor products:
///
/// ```text
/// L_{(i_0, i_1, ..., i_{k-1})}(r_0, r_1, ..., r_{k-1}) = \prod_{j=0}^{k-1} L_{i_j}(r_j)
/// ```
///
/// where each `i_j \in {0, 1, 2}` and `L_{i_j}` is the univariate Lagrange basis.
///
/// # Output Ordering
///
/// The weights are ordered lexicographically by the multi-index `(i_0, i_1, ..., i_{k-1})`:
/// - Index 0: weight for point `(0, 0, ..., 0)`
/// - Index 1: weight for point `(0, 0, ..., 1)`
/// - Index 2: weight for point `(0, 0, ..., 2)`
/// - ...
/// - Index `3^k - 1`: weight for point `(2, 2, ..., 2)`
///
/// # Arguments
///
/// * `rs` - Slice of evaluation points `[r_0, r_1, ..., r_{k-1}]`
///
/// # Returns
///
/// Vector of `3^k` field elements representing the tensor product Lagrange weights.
pub fn lagrange_weights_012_multi<F: Field>(rs: &[F]) -> Vec<F> {
    let total = 3usize.pow(rs.len() as u32);
    let mut current = Vec::with_capacity(total);
    let mut next = Vec::with_capacity(total);
    current.push(F::ONE);

    // Iteratively compute tensor products.
    //
    // After processing r_j, we have 3^(j+1) weights for the grid {0,1,2}^(j+1).
    for &r in rs {
        // Compute univariate Lagrange weights for this coordinate.
        let uni = lagrange_weights_012(r);

        // Tensor product: new_weights[3*i + j] = old_weights[i] * uni[j]
        next.clear();
        for &li in &uni {
            for &w in &current {
                next.push(w * li);
            }
        }
        core::mem::swap(&mut current, &mut next);
    }

    current
}

/// Evaluates a quadratic polynomial at point `r` given its evaluations at `{0, 1, 2}`.
///
/// Uses Lagrange interpolation to compute `h(r)` from the values `h(0)`, `h(1)`, `h(2)`.
///
/// # Mathematical Formula
///
/// ```text
/// h(r) = h(0) * L_0(r) + h(1) * L_1(r) + h(2) * L_2(r)
/// ```
///
/// where `L_i` are the Lagrange basis polynomials computed by [`lagrange_weights_012`].
///
/// # Arguments
///
/// * `e0` - The value `h(0)` (evaluation at 0)
/// * `e1` - The value `h(1)` (evaluation at 1)
/// * `e2` - The value `h(2)` (evaluation at 2)
/// * `r` - The point at which to evaluate the polynomial
///
/// # Returns
///
/// The value `h(r)`.
pub fn extrapolate_012<F: Field>(e0: F, e1: F, e2: F, r: F) -> F {
    // Compute Lagrange basis weights at point r.
    let [w0, w1, w2] = lagrange_weights_012(r);

    // Evaluate via linear combination.
    e0 * w0 + e1 * w1 + e2 * w2
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;
    use proptest::prelude::*;

    use super::*;

    type F = BabyBear;

    #[test]
    fn test_lagrange_weights_at_interpolation_points() {
        // At x = 0: L_0(0) = 1, L_1(0) = 0, L_2(0) = 0
        let [l0, l1, l2] = lagrange_weights_012(F::ZERO);
        assert_eq!(l0, F::ONE);
        assert_eq!(l1, F::ZERO);
        assert_eq!(l2, F::ZERO);

        // At x = 1: L_0(1) = 0, L_1(1) = 1, L_2(1) = 0
        let [l0, l1, l2] = lagrange_weights_012(F::ONE);
        assert_eq!(l0, F::ZERO);
        assert_eq!(l1, F::ONE);
        assert_eq!(l2, F::ZERO);

        // At x = 2: L_0(2) = 0, L_1(2) = 0, L_2(2) = 1
        let [l0, l1, l2] = lagrange_weights_012(F::TWO);
        assert_eq!(l0, F::ZERO);
        assert_eq!(l1, F::ZERO);
        assert_eq!(l2, F::ONE);
    }

    #[test]
    fn test_lagrange_weights_partition_of_unity() {
        // The Lagrange basis polynomials should sum to 1 at any point.
        for i in 0..10 {
            let r = F::from_u64(i);
            let [l0, l1, l2] = lagrange_weights_012(r);
            assert_eq!(l0 + l1 + l2, F::ONE, "Partition of unity failed at r = {i}");
        }
    }

    #[test]
    fn test_lagrange_weights_multi_k1() {
        // For k=1, multi should match the single-variable version.
        for i in 0..5 {
            let r = F::from_u64(i);
            let single = lagrange_weights_012(r);
            let multi = lagrange_weights_012_multi(&[r]);

            assert_eq!(multi.len(), 3);
            assert_eq!(multi[0], single[0]);
            assert_eq!(multi[1], single[1]);
            assert_eq!(multi[2], single[2]);
        }
    }

    #[test]
    fn test_lagrange_weights_multi_k2() {
        // For k=2, we get 9 weights.
        let r0 = F::from_u64(5);
        let r1 = F::from_u64(7);

        let weights = lagrange_weights_012_multi(&[r0, r1]);
        assert_eq!(weights.len(), 9);

        // Verify tensor product structure.
        //
        // The implementation iterates: for each new coordinate's weight L_j(r1),
        // multiply by all existing weights. This gives ordering:
        // weights[3*j + i] = L_i(r0) * L_j(r1)
        let w0 = lagrange_weights_012(r0);
        let w1 = lagrange_weights_012(r1);

        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(weights[3 * j + i], w0[i] * w1[j]);
            }
        }
    }

    #[test]
    fn test_lagrange_weights_multi_partition_of_unity() {
        // Tensor product weights should sum to 1.
        let r0 = F::from_u64(3);
        let r1 = F::from_u64(7);
        let r2 = F::from_u64(11);

        // k=1
        let w1: F = lagrange_weights_012_multi(&[r0]).into_iter().sum();
        assert_eq!(w1, F::ONE);

        // k=2
        let w2: F = lagrange_weights_012_multi(&[r0, r1]).into_iter().sum();
        assert_eq!(w2, F::ONE);

        // k=3
        let w3: F = lagrange_weights_012_multi(&[r0, r1, r2]).into_iter().sum();
        assert_eq!(w3, F::ONE);
    }

    #[test]
    fn test_extrapolate_at_interpolation_points() {
        // Extrapolating at the interpolation points should return the original values.
        let e0 = F::from_u64(7);
        let e1 = F::from_u64(13);
        let e2 = F::from_u64(23);

        assert_eq!(extrapolate_012(e0, e1, e2, F::ZERO), e0);
        assert_eq!(extrapolate_012(e0, e1, e2, F::ONE), e1);
        assert_eq!(extrapolate_012(e0, e1, e2, F::TWO), e2);
    }

    #[test]
    fn test_extrapolate_known_quadratic() {
        // Test with h(x) = x^2 + 1
        //
        // h(0) = 1, h(1) = 2, h(2) = 5
        let e0 = F::from_u64(1);
        let e1 = F::from_u64(2);
        let e2 = F::from_u64(5);

        // h(3) = 9 + 1 = 10
        assert_eq!(extrapolate_012(e0, e1, e2, F::from_u64(3)), F::from_u64(10));

        // h(4) = 16 + 1 = 17
        assert_eq!(extrapolate_012(e0, e1, e2, F::from_u64(4)), F::from_u64(17));
    }

    proptest! {
        /// Property: Extrapolating at interpolation points returns original values.
        #[test]
        fn prop_extrapolate_identity(
            e0 in 0u32..1_000_000,
            e1 in 0u32..1_000_000,
            e2 in 0u32..1_000_000,
        ) {
            let e0 = F::from_u32(e0);
            let e1 = F::from_u32(e1);
            let e2 = F::from_u32(e2);

            prop_assert_eq!(extrapolate_012(e0, e1, e2, F::ZERO), e0);
            prop_assert_eq!(extrapolate_012(e0, e1, e2, F::ONE), e1);
            prop_assert_eq!(extrapolate_012(e0, e1, e2, F::TWO), e2);
        }

        /// Property: Lagrange weights sum to 1 (partition of unity).
        #[test]
        fn prop_partition_of_unity(r in 0u32..1_000_000) {
            let r = F::from_u32(r);
            let [l0, l1, l2] = lagrange_weights_012(r);
            prop_assert_eq!(l0 + l1 + l2, F::ONE);
        }

        /// Property: Tensor product weights sum to 1.
        #[test]
        fn prop_multi_partition_of_unity(
            r0 in 0u32..1_000_000,
            r1 in 0u32..1_000_000,
        ) {
            let r0 = F::from_u32(r0);
            let r1 = F::from_u32(r1);

            let sum: F = lagrange_weights_012_multi(&[r0, r1]).into_iter().sum();
            prop_assert_eq!(sum, F::ONE);
        }

        /// Property: Extrapolation matches direct polynomial evaluation.
        ///
        /// Given evaluations at {0, 1, 2}, we can recover the unique quadratic h(x)
        /// in monomial form: h(x) = c0 + c1*x + c2*x^2.
        ///
        /// The coefficients can be computed from evaluations:
        /// - c0 = h(0)
        /// - c2 = (h(0) - 2*h(1) + h(2)) / 2
        /// - c1 = h(1) - h(0) - c2
        #[test]
        fn prop_extrapolate_matches_monomial(
            e0 in 0u32..1_000_000,
            e1 in 0u32..1_000_000,
            e2 in 0u32..1_000_000,
            r in 0u32..1_000_000,
        ) {
            let e0 = F::from_u32(e0);
            let e1 = F::from_u32(e1);
            let e2 = F::from_u32(e2);
            let r = F::from_u32(r);

            // Compute monomial coefficients.
            let c0 = e0;
            let c2 = (e0 - e1.double() + e2) * F::TWO.inverse();
            let c1 = e1 - e0 - c2;

            // Evaluate using Horner's method: c0 + r*(c1 + r*c2)
            let monomial_eval = c0 + r * (c1 + r * c2);

            // Evaluate using Lagrange extrapolation.
            let lagrange_eval = extrapolate_012(e0, e1, e2, r);

            prop_assert_eq!(lagrange_eval, monomial_eval);
        }
    }
}
