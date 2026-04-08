//! Interpolation utilities for sumcheck using evaluation at infinity.
//!
//! The round polynomial `h(X)` is degree 2. Instead of evaluating at
//! `{0, 1, 2}`, we use the evaluation set `{0, 1, inf}` where the
//! "evaluation at infinity" is the leading coefficient of `h`.
//!
//! For `h(X) = a*X^2 + b*X + c`:
//!
//! ```text
//!     h(0)   = c
//!     h(1)   = a + b + c
//!     h(inf) = a            (leading coefficient)
//! ```
//!
//! Reconstruction:
//!
//! ```text
//!     h(r) = h(0)*(1-r) + h(1)*r + h(inf)*r*(r-1)
//! ```

use alloc::vec::Vec;

use p3_field::Field;

/// Computes the interpolation weights for the evaluation set `{0, 1, inf}`.
///
/// ```text
///     L_0(r)   = 1 - r
///     L_1(r)   = r
///     L_inf(r) = r * (r - 1)
/// ```
///
/// These satisfy `L_0(0) = 1`, `L_1(1) = 1`, and the reconstruction formula
/// `h(r) = h(0)*L_0(r) + h(1)*L_1(r) + h(inf)*L_inf(r)`.
///
/// Note: these do NOT form a partition of unity (they do not sum to 1).
pub(crate) fn lagrange_weights_01inf<F: Field>(r: F) -> [F; 3] {
    let r_minus_one = r - F::ONE;

    // L_0(r) = 1 - r
    let l0 = F::ONE - r;

    // L_1(r) = r
    let l1 = r;

    // L_inf(r) = r * (r - 1)
    let l_inf = r * r_minus_one;

    [l0, l1, l_inf]
}

/// Computes tensor product weights for multivariate interpolation on `{0, 1, inf}^k`.
///
/// Given points `rs = [r_0, ..., r_{k-1}]`, produces `3^k` weights where each
/// weight is a product of univariate weights across all coordinates.
///
/// # Output Ordering
///
/// Lexicographic by multi-index `(i_0, i_1, ..., i_{k-1})` where each `i_j`
/// ranges over the three basis functions (0, 1, inf).
pub fn lagrange_weights_01inf_multi<F: Field>(rs: &[F]) -> Vec<F> {
    let total = 3usize.pow(rs.len() as u32);
    let mut current = Vec::with_capacity(total);
    let mut next = Vec::with_capacity(total);
    current.push(F::ONE);

    // Iteratively build the tensor product, one coordinate at a time.
    for &r in rs {
        let uni = lagrange_weights_01inf(r);

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

/// Evaluates a degree-2 polynomial at `r` from `(h(0), h(1), h(inf))`.
///
/// ```text
///     h(r) = h(0)*(1-r) + h(1)*r + h(inf)*r*(r-1)
/// ```
pub fn extrapolate_01inf<F: Field>(e0: F, e1: F, e_inf: F, r: F) -> F {
    let [w0, w1, w_inf] = lagrange_weights_01inf(r);
    e0 * w0 + e1 * w1 + e_inf * w_inf
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;
    use proptest::prelude::*;

    use super::*;

    type F = BabyBear;

    #[test]
    fn test_lagrange_weights_at_finite_points() {
        // At r = 0: L_0 = 1, L_1 = 0, L_inf = 0.
        let [l0, l1, l_inf] = lagrange_weights_01inf(F::ZERO);
        assert_eq!(l0, F::ONE);
        assert_eq!(l1, F::ZERO);
        assert_eq!(l_inf, F::ZERO);

        // At r = 1: L_0 = 0, L_1 = 1, L_inf = 0.
        let [l0, l1, l_inf] = lagrange_weights_01inf(F::ONE);
        assert_eq!(l0, F::ZERO);
        assert_eq!(l1, F::ONE);
        assert_eq!(l_inf, F::ZERO);
    }

    #[test]
    fn test_lagrange_weights_multi_k1() {
        // For k=1, the tensor product should match the univariate version.
        for i in 0..5 {
            let r = F::from_u64(i);
            let single = lagrange_weights_01inf(r);
            let multi = lagrange_weights_01inf_multi(&[r]);

            assert_eq!(multi.len(), 3);
            assert_eq!(multi[0], single[0]);
            assert_eq!(multi[1], single[1]);
            assert_eq!(multi[2], single[2]);
        }
    }

    #[test]
    fn test_lagrange_weights_multi_k2() {
        // For k=2, verify the tensor product structure: 9 weights.
        let r0 = F::from_u64(5);
        let r1 = F::from_u64(7);

        let weights = lagrange_weights_01inf_multi(&[r0, r1]);
        assert_eq!(weights.len(), 9);

        // weights[3*j + i] = L_i(r0) * L_j(r1)
        let w0 = lagrange_weights_01inf(r0);
        let w1 = lagrange_weights_01inf(r1);

        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(weights[3 * j + i], w0[i] * w1[j]);
            }
        }
    }

    #[test]
    fn test_extrapolate_at_finite_points() {
        // h(0) and h(1) should be recovered exactly.
        let e0 = F::from_u64(7);
        let e1 = F::from_u64(13);
        let e_inf = F::from_u64(3);

        assert_eq!(extrapolate_01inf(e0, e1, e_inf, F::ZERO), e0);
        assert_eq!(extrapolate_01inf(e0, e1, e_inf, F::ONE), e1);
    }

    #[test]
    fn test_extrapolate_known_quadratic() {
        // h(x) = x^2 + 1
        //
        // h(0) = 1, h(1) = 2, h(inf) = 1 (leading coefficient).
        //
        // h(3) = 9 + 1 = 10
        // h(4) = 16 + 1 = 17
        let e0 = F::from_u64(1);
        let e1 = F::from_u64(2);
        let e_inf = F::from_u64(1);

        assert_eq!(
            extrapolate_01inf(e0, e1, e_inf, F::from_u64(3)),
            F::from_u64(10)
        );
        assert_eq!(
            extrapolate_01inf(e0, e1, e_inf, F::from_u64(4)),
            F::from_u64(17)
        );
    }

    proptest! {
        /// Extrapolating at 0 and 1 recovers the input values.
        #[test]
        fn prop_extrapolate_identity(
            e0 in 0u32..1_000_000,
            e1 in 0u32..1_000_000,
            e_inf in 0u32..1_000_000,
        ) {
            let e0 = F::from_u32(e0);
            let e1 = F::from_u32(e1);
            let e_inf = F::from_u32(e_inf);

            prop_assert_eq!(extrapolate_01inf(e0, e1, e_inf, F::ZERO), e0);
            prop_assert_eq!(extrapolate_01inf(e0, e1, e_inf, F::ONE), e1);
        }

        /// Extrapolation matches direct monomial evaluation.
        ///
        /// Given (h(0), h(1), h(inf)) = (c, a+b+c, a), reconstruct
        /// h(r) = a*r^2 + b*r + c and compare.
        #[test]
        fn prop_extrapolate_matches_monomial(
            a_val in 0u32..1_000_000,
            b_val in 0u32..1_000_000,
            c_val in 0u32..1_000_000,
            r_val in 0u32..1_000_000,
        ) {
            let a = F::from_u32(a_val);
            let b = F::from_u32(b_val);
            let c = F::from_u32(c_val);
            let r = F::from_u32(r_val);

            // Build the three evaluation-set values.
            let e0 = c;
            let e1 = a + b + c;
            let e_inf = a;

            // Direct monomial evaluation: a*r^2 + b*r + c
            let monomial = c + r * (b + r * a);

            prop_assert_eq!(extrapolate_01inf(e0, e1, e_inf, r), monomial);
        }
    }
}
