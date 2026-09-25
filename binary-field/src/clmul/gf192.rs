//! Arithmetic in the cubic extension `GF(2^64)[y] / (y^3 + y + 1)`.
//!
//! Every routine defers the reduction modulo the base polynomial to its three outputs.
//!
//! Reduction is `F_2`-linear, so folding `y` first costs nothing and leaves three to reduce.

use super::{clmul_64x64, reduce_64};

/// Multiplication, coordinates in the basis `1, y, y^2`.
#[allow(clippy::missing_const_for_fn)]
#[inline]
pub(crate) fn poly_mul_192(a: &[u64; 3], b: &[u64; 3]) -> [u64; 3] {
    // Two coordinates per vector register, where the carryless multiply is an instruction.
    #[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
    {
        super::x86_64::poly_mul_192(a, b)
    }
    // Otherwise six 64-bit products from the target's own carryless multiply.
    #[cfg(not(all(target_arch = "x86_64", target_feature = "pclmulqdq")))]
    {
        composed_mul_192(a, b)
    }
}

/// Squaring.
#[allow(clippy::missing_const_for_fn)]
#[inline]
pub(crate) fn poly_square_192(a: &[u64; 3]) -> [u64; 3] {
    // Same split between the vector backend and the composed route as the product.
    #[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
    {
        super::x86_64::poly_square_192(a)
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "pclmulqdq")))]
    {
        composed_square_192(a)
    }
}

/// Multiplication by an element of the coefficient field.
#[allow(clippy::missing_const_for_fn)]
#[inline]
pub(crate) fn poly_mul_192_by_64(a: &[u64; 3], k: &u64) -> [u64; 3] {
    // Same split between the vector backend and the composed route as the product.
    #[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
    {
        super::x86_64::poly_mul_192_by_64(a, k)
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "pclmulqdq")))]
    {
        composed_mul_192_by_64(a, k)
    }
}

/// Sum unreduced products before paying for one reduction.
#[inline]
pub(crate) fn poly_dot_192<'a>(
    pairs: impl Iterator<Item = (&'a [u64; 3], &'a [u64; 3])>,
) -> [u64; 3] {
    // Same split between the vector backend and the composed route as the product.
    #[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
    {
        super::x86_64::poly_dot_192(pairs)
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "pclmulqdq")))]
    {
        composed_dot_192(pairs)
    }
}

/// Sum of coefficient-field multiples, one reduction for the whole sum.
///
/// Three carryless products per term, and nothing to fold in `y`.
#[inline]
pub(crate) fn poly_dot_192_by_64<'a>(
    pairs: impl Iterator<Item = (&'a [u64; 3], &'a u64)>,
) -> [u64; 3] {
    // Same split between the vector backend and the composed route as the product.
    #[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
    {
        super::x86_64::poly_dot_192_by_64(pairs)
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "pclmulqdq")))]
    {
        composed_dot_192_by_64(pairs)
    }
}

/// The unreduced product, with the top two powers of `y` already folded.
///
/// Karatsuba over three limbs, writing `c_i = a_i b_i` and `d_ij = (a_i + a_j)(b_i + b_j)`:
///
/// ```text
///     r_0  =  c_0 + c_1 + c_2 + d_12
///     r_1  =  c_0 + d_01 + d_12
///     r_2  =  c_0 + c_1 + d_02
/// ```
// Only one carryless-product backend is `const`, so the signature stays uniform.
#[allow(clippy::missing_const_for_fn)]
#[inline]
fn mul_unreduced(&[a0, a1, a2]: &[u64; 3], &[b0, b1, b2]: &[u64; 3]) -> [u128; 3] {
    // The three diagonal products, 128 bits each.
    let c0 = clmul_64x64(a0, b0);
    let c1 = clmul_64x64(a1, b1);
    let c2 = clmul_64x64(a2, b2);

    // One product per pair of limbs, carrying both of that pair's cross terms.
    let d01 = clmul_64x64(a0 ^ a1, b0 ^ b1);
    let d02 = clmul_64x64(a0 ^ a2, b0 ^ b2);
    let d12 = clmul_64x64(a1 ^ a2, b1 ^ b2);

    // The one sum two coordinates share.
    let shared = c0 ^ d12;

    // The folded coordinates r_0, r_1, r_2.
    [shared ^ c1 ^ c2, shared ^ d01, c0 ^ c1 ^ d02]
}

/// Multiplication assembled from the shared carryless product and the shared fold.
///
/// Compiled everywhere, so its tests run even where a backend supersedes it.
#[cfg_attr(
    all(target_arch = "x86_64", target_feature = "pclmulqdq"),
    allow(dead_code)
)]
#[inline]
fn composed_mul_192(a: &[u64; 3], b: &[u64; 3]) -> [u64; 3] {
    // Three reductions in the base field, one per folded coordinate.
    mul_unreduced(a, b).map(reduce_64)
}

/// Squaring assembled the same way.
#[cfg_attr(
    all(target_arch = "x86_64", target_feature = "pclmulqdq"),
    allow(dead_code)
)]
#[inline]
fn composed_square_192(&[a0, a1, a2]: &[u64; 3]) -> [u64; 3] {
    // Invariant: the cross terms double to zero, and y^4 = y^2 + y folds the top one.
    //
    //     (a_0 + a_1 y + a_2 y^2)^2  =  a_0^2 + a_2^2 y + (a_1^2 + a_2^2) y^2
    let (s0, s1, s2) = (
        clmul_64x64(a0, a0),
        clmul_64x64(a1, a1),
        clmul_64x64(a2, a2),
    );

    // Fold y^4 onto y^2 and y, then reduce each coordinate once.
    [s0, s2, s1 ^ s2].map(reduce_64)
}

/// Scaling by a coefficient assembled the same way.
#[cfg_attr(
    all(target_arch = "x86_64", target_feature = "pclmulqdq"),
    allow(dead_code)
)]
#[inline]
fn composed_mul_192_by_64(a: &[u64; 3], &k: &u64) -> [u64; 3] {
    // The scalar scales each coordinate on its own, so nothing folds in y.
    a.map(|x| reduce_64(clmul_64x64(x, k)))
}

/// A dot product assembled the same way.
#[cfg_attr(
    all(target_arch = "x86_64", target_feature = "pclmulqdq"),
    allow(dead_code)
)]
#[inline]
fn composed_dot_192<'a>(pairs: impl Iterator<Item = (&'a [u64; 3], &'a [u64; 3])>) -> [u64; 3] {
    // Accumulate the folded, unreduced coordinates of every term.
    let sum = pairs.fold([0u128; 3], |sum, (a, b)| {
        let product = mul_unreduced(a, b);
        core::array::from_fn(|i| sum[i] ^ product[i])
    });

    // Reduction is linear, so the whole sum reduces once per coordinate.
    sum.map(reduce_64)
}

/// A mixed dot product assembled the same way.
#[cfg_attr(
    all(target_arch = "x86_64", target_feature = "pclmulqdq"),
    allow(dead_code)
)]
#[inline]
fn composed_dot_192_by_64<'a>(pairs: impl Iterator<Item = (&'a [u64; 3], &'a u64)>) -> [u64; 3] {
    // Three unreduced products per term, each landing on its own coordinate.
    let sum = pairs.fold([0u128; 3], |sum, (a, &k)| {
        core::array::from_fn(|i| sum[i] ^ clmul_64x64(a[i], k))
    });

    // One reduction per coordinate for the whole sum.
    sum.map(reduce_64)
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use proptest::prelude::*;

    use super::{
        composed_dot_192, composed_dot_192_by_64, composed_mul_192, composed_mul_192_by_64,
        composed_square_192, poly_dot_192, poly_dot_192_by_64, poly_mul_192, poly_mul_192_by_64,
        poly_square_192,
    };
    use crate::clmul::poly_mul_64;

    /// Nine coefficient products and the modulus applied term by term.
    ///
    /// It shares neither the Karatsuba schedule nor the deferred reduction under test.
    fn schoolbook(a: [u64; 3], b: [u64; 3]) -> [u64; 3] {
        // Nine reduced products, placed by total degree in y.
        let mut raw = [0u64; 5];
        for i in 0..3 {
            for j in 0..3 {
                raw[i + j] ^= poly_mul_64(a[i], b[j]);
            }
        }
        // y^4 = y^2 + y, then y^3 = y + 1.
        [raw[0] ^ raw[3], raw[1] ^ raw[3] ^ raw[4], raw[2] ^ raw[4]]
    }

    /// Coordinates a random search is unlikely to reach.
    ///
    /// ```text
    ///     0, 1          the identities
    ///     all ones      every coefficient of every product is live
    ///     x^63          the highest degree, so the base fold spills furthest
    ///     0xf << 60     the whole top nibble the fold tabulates
    /// ```
    const CORNERS: [u64; 5] = [0, 1, u64::MAX, 1 << 63, 0xf << 60];

    #[test]
    fn both_routes_are_exact_on_the_extremes() {
        // Invariant: both routes reduce exactly, however far the product overflows.
        //
        // Fixture state: 5 corners in two coordinates, their sum in the third.
        //
        //     5 x 5 = 25 elements  ->  625 products and 25 squares per route
        let elements: Vec<[u64; 3]> = CORNERS
            .iter()
            .flat_map(|&x| CORNERS.iter().map(move |&y| [x, y, x ^ y]))
            .collect();
        for &a in &elements {
            assert_eq!(poly_square_192(&a), schoolbook(a, a), "{a:x?} squared");
            assert_eq!(composed_square_192(&a), schoolbook(a, a), "{a:x?} squared");
            for &b in &elements {
                assert_eq!(poly_mul_192(&a, &b), schoolbook(a, b), "{a:x?} * {b:x?}");
                assert_eq!(
                    composed_mul_192(&a, &b),
                    schoolbook(a, b),
                    "{a:x?} * {b:x?}"
                );
            }
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(4000))]

        #[test]
        fn both_routes_match_the_schoolbook_product(a: [u64; 3], b: [u64; 3], k: u64) {
            // Invariant: the selected backend and the composed route agree with nine products.
            let expected = schoolbook(a, b);
            prop_assert_eq!(poly_mul_192(&a, &b), expected);
            prop_assert_eq!(composed_mul_192(&a, &b), expected);

            // Squaring shares the fold of y^4 with the product.
            prop_assert_eq!(poly_square_192(&a), schoolbook(a, a));
            prop_assert_eq!(composed_square_192(&a), schoolbook(a, a));

            // A coefficient-field scalar is the extension element with only a constant term.
            let scaled = schoolbook(a, [k, 0, 0]);
            prop_assert_eq!(poly_mul_192_by_64(&a, &k), scaled);
            prop_assert_eq!(composed_mul_192_by_64(&a, &k), scaled);
        }

        #[test]
        fn dot_products_sum_their_terms(
            a: [u64; 3], b: [u64; 3], c: [u64; 3], d: [u64; 3], k: u64, l: u64,
        ) {
            // Invariant: one deferred reduction equals reducing every term.
            //
            //     reduce(ab + cd)  =  reduce(ab) + reduce(cd)
            let expected: [u64; 3] = {
                let (x, y) = (schoolbook(a, b), schoolbook(c, d));
                core::array::from_fn(|i| x[i] ^ y[i])
            };
            let pairs = [(&a, &b), (&c, &d)];
            prop_assert_eq!(poly_dot_192(pairs.into_iter()), expected);
            prop_assert_eq!(composed_dot_192(pairs.into_iter()), expected);

            // The same with coefficient-field weights: a k + c l.
            let mixed: [u64; 3] = {
                let (x, y) = (schoolbook(a, [k, 0, 0]), schoolbook(c, [l, 0, 0]));
                core::array::from_fn(|i| x[i] ^ y[i])
            };
            let terms = [(&a, &k), (&c, &l)];
            prop_assert_eq!(poly_dot_192_by_64(terms.into_iter()), mixed);
            prop_assert_eq!(composed_dot_192_by_64(terms.into_iter()), mixed);
        }
    }

    #[test]
    fn an_empty_dot_product_is_zero() {
        // Fixture state: no terms, so the accumulators stay zero and reduce to zero.
        assert_eq!(poly_dot_192(core::iter::empty()), [0; 3]);
        assert_eq!(poly_dot_192_by_64(core::iter::empty()), [0; 3]);
    }
}
