//! Polynomial-basis arithmetic in `GF(2)[x] / (x^64 + x^4 + x^3 + x + 1)`.
//!
//! One carryless product covers a whole multiplication, and no routine reads a table.

use super::sqrt::compact_even;
use super::{clmul_64x64, reduce_64};

/// The square root of the polynomial variable modulo this field's polynomial.
const ROOT_X: u64 = 0xffff_ffff_0000_000a;

/// Multiplication, taking and returning the polynomial representation.
//
// Only the composed route is `const`, so the signature stays uniform across targets.
#[allow(clippy::missing_const_for_fn)]
#[inline]
pub(crate) fn poly_mul_64(a: u64, b: u64) -> u64 {
    #[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
    {
        super::x86_64::poly_mul_64(a, b)
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "pclmulqdq")))]
    {
        composed_mul_64(a, b)
    }
}

/// Squaring, taking and returning the polynomial representation.
#[allow(clippy::missing_const_for_fn)]
#[inline]
pub(crate) fn poly_square_64(a: u64) -> u64 {
    #[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
    {
        super::x86_64::poly_square_64(a)
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "pclmulqdq")))]
    {
        composed_square_64(a)
    }
}

/// Sum unreduced products before paying for one reduction.
#[inline]
pub(crate) fn poly_dot_64(pairs: impl Iterator<Item = (u64, u64)>) -> u64 {
    #[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
    {
        super::x86_64::poly_dot_64(pairs)
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "pclmulqdq")))]
    {
        composed_dot_64(pairs)
    }
}

/// Multiplication assembled from the shared carryless product and the shared fold.
///
/// Compiled everywhere, so its tests run even where a backend supersedes it.
//
// Only one carryless-product backend is `const`, so the signature stays uniform.
#[allow(clippy::missing_const_for_fn)]
#[cfg_attr(
    all(target_arch = "x86_64", target_feature = "pclmulqdq"),
    allow(dead_code)
)]
#[inline]
fn composed_mul_64(a: u64, b: u64) -> u64 {
    reduce_64(clmul_64x64(a, b))
}

/// Squaring assembled the same way.
#[allow(clippy::missing_const_for_fn)]
#[cfg_attr(
    all(target_arch = "x86_64", target_feature = "pclmulqdq"),
    allow(dead_code)
)]
#[inline]
fn composed_square_64(a: u64) -> u64 {
    reduce_64(clmul_64x64(a, a))
}

/// A dot product assembled the same way.
#[cfg_attr(
    all(target_arch = "x86_64", target_feature = "pclmulqdq"),
    allow(dead_code)
)]
#[inline]
fn composed_dot_64(pairs: impl Iterator<Item = (u64, u64)>) -> u64 {
    // Reduction is `F_2`-linear, so the whole sum folds the modulus once.
    reduce_64(pairs.fold(0, |sum, (a, b)| sum ^ clmul_64x64(a, b)))
}

/// Squaring repeated a fixed number of times.
///
/// With `GFNI` the whole power is one bit-matrix product, whatever the count.
///
/// Otherwise it is `K` dependent squarings.
#[inline]
fn square_times<const K: usize>(x: u64) -> u64 {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "pclmulqdq",
        target_feature = "gfni",
        target_feature = "avx512f",
        target_feature = "avx512bw",
        target_feature = "avx512vbmi"
    ))]
    {
        // One broadcast, eight affine products and a byte permute, for any K.
        super::x86_64::square_times::<K>(x)
    }
    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "pclmulqdq",
        target_feature = "gfni",
        target_feature = "avx512f",
        target_feature = "avx512bw",
        target_feature = "avx512vbmi"
    )))]
    {
        // K dependent squarings, each one carryless product and one fold.
        (0..K).fold(x, |y, _| poly_square_64(y))
    }
}

/// The square root, which every element of a binary field has exactly one of.
///
/// Splitting by the parity of the exponents gives it in one field product:
///
/// ```text
///     a        = even(x)^2 + x * odd(x)^2
///     sqrt(a)  = even(x) + sqrt(x) * odd(x)
/// ```
#[inline]
pub(crate) fn poly_sqrt_64(a: u64) -> u64 {
    let even = compact_even(a);
    let odd = compact_even(a >> 1);
    even ^ poly_mul_64(ROOT_X, odd)
}

/// The inverse of a nonzero element, sending zero to zero.
///
/// Itoh-Tsujii on the exponent `2^64 - 2`, writing `b_k` for `x^(2^k - 1)`:
///
/// ```text
///     b_(a+b)  = b_a^(2^b) * b_b
///     chain      1, 2, 3, 6, 12, 24, 48, 60, 63
/// ```
///
/// Nine exponents is eight steps, so eight products and sixty-three squarings.
///
/// With `GFNI`, each run of squarings is one bit-matrix product instead.
///
/// None of them is indexed by the operand.
#[inline]
pub(crate) fn poly_inverse_64(x: u64) -> u64 {
    let b2 = poly_mul_64(poly_square_64(x), x);
    let b3 = poly_mul_64(poly_square_64(b2), x);
    let b6 = poly_mul_64(square_times::<3>(b3), b3);

    // Doubling the exponent index reuses the same intermediate on both sides.
    let b12 = poly_mul_64(square_times::<6>(b6), b6);
    let b24 = poly_mul_64(square_times::<12>(b12), b12);
    let b48 = poly_mul_64(square_times::<24>(b24), b24);

    // Finish 48 + 12 = 60, then 60 + 3 = 63.
    let b60 = poly_mul_64(square_times::<12>(b48), b12);
    let b63 = poly_mul_64(square_times::<3>(b60), b3);

    // One more squaring turns `2^63 - 1` into `2^64 - 2`.
    poly_square_64(b63)
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::{
        ROOT_X, composed_dot_64, composed_mul_64, composed_square_64, poly_dot_64, poly_inverse_64,
        poly_mul_64, poly_sqrt_64, poly_square_64,
    };

    /// Multiplication from the modulus alone, with no carryless product and no fold.
    fn schoolbook(a: u64, b: u64) -> u64 {
        let mut acc = 0u64;
        for i in (0..64).rev() {
            let overflowed = acc & (1 << 63) != 0;
            acc <<= 1;
            if overflowed {
                acc ^= 0b1_1011;
            }
            if (b >> i) & 1 == 1 {
                acc ^= a;
            }
        }
        acc
    }

    #[test]
    fn the_modulus_reduces_the_way_the_polynomial_says() {
        // x^63 * x = x^64 = x^4 + x^3 + x + 1, the tail spelled 0b1_1011.
        assert_eq!(poly_mul_64(1 << 63, 2), 0b1_1011);

        // x * x = x^2, nowhere near the modulus.
        assert_eq!(poly_mul_64(2, 2), 4);

        // (x + 1)^2 = x^2 + 1, since the cross term doubles to zero.
        assert_eq!(poly_square_64(3), 5);
    }

    #[test]
    fn the_square_is_linear_on_every_basis_vector() {
        // Squaring is `F_2`-linear in characteristic 2, so 64 vectors settle all 2^64 inputs.
        for i in 0..64 {
            let a = 1u64 << i;
            assert_eq!(poly_square_64(a), schoolbook(a, a), "vector {i}");
        }
    }

    #[test]
    fn the_square_root_of_the_variable_squares_back() {
        // This constant is the one thing in the root that is not derived at run time.
        assert_eq!(poly_mul_64(ROOT_X, ROOT_X), 2);
    }

    #[test]
    fn the_product_is_exact_on_the_extremes() {
        // Corners a random search is unlikely to reach.
        //
        // - the identities, and the operand whose every product coefficient is live,
        // - the highest degree, where the fold spills furthest, and the tail itself.
        const CORNERS: [u64; 6] = [0, 1, u64::MAX, 1 << 63, 0b1_1011, (1 << 32) - 1];
        for a in CORNERS {
            assert_eq!(poly_square_64(a), schoolbook(a, a), "{a:#x} squared");
            for b in CORNERS {
                assert_eq!(poly_mul_64(a, b), schoolbook(a, b), "{a:#x} * {b:#x}");
            }
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(4000))]

        /// Both the selected route and the composed one must match the modulus itself.
        #[test]
        fn the_product_matches_the_schoolbook_reduction(a: u64, b: u64) {
            let expected = schoolbook(a, b);
            prop_assert_eq!(poly_mul_64(a, b), expected);
            prop_assert_eq!(composed_mul_64(a, b), expected);
            prop_assert_eq!(poly_square_64(a), schoolbook(a, a));
            prop_assert_eq!(composed_square_64(a), schoolbook(a, a));
        }

        /// A dot product must be the sum of the individual products, on both routes.
        #[test]
        fn the_dot_product_sums_the_individual_products(a: u64, b: u64, c: u64, d: u64) {
            let expected = schoolbook(a, b) ^ schoolbook(c, d);
            let pairs = [(a, b), (c, d)];
            prop_assert_eq!(poly_dot_64(pairs.into_iter()), expected);
            prop_assert_eq!(composed_dot_64(pairs.into_iter()), expected);
        }

        #[test]
        fn the_square_root_squares_back(a: u64) {
            let root = poly_sqrt_64(a);
            prop_assert_eq!(schoolbook(root, root), a);
        }

        /// The inverse must satisfy the defining identity against the independent product.
        #[test]
        fn the_inverse_multiplies_back_to_one(a: u64) {
            prop_assert_eq!(schoolbook(a, poly_inverse_64(a)), u64::from(a != 0));
        }
    }
}
