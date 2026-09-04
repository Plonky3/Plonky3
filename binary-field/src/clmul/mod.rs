//! Carryless multiplication, and the polynomial-basis arithmetic built on it.
//!
//! The instruction works in `GF(2)[x]`, which is not the tower's representation.
//! A tower product therefore costs three steps that a polynomial-basis product does not:
//!
//! ```text
//!     map both operands into the polynomial basis
//!     multiply and reduce there
//!     map the result back
//! ```
//!
//! Each backend supplies the same routines, and each has a portable definition to test against.

mod basis;
mod sqrt;

pub(crate) use sqrt::poly_sqrt_128;

// The addition chain pays for its 320 KiB of Frobenius maps only where a product is a single
// instruction.
// Without one the tower norm wins, so the maps are not compiled at all.
#[cfg(any(
    all(target_arch = "x86_64", target_feature = "pclmulqdq"),
    all(target_arch = "aarch64", target_feature = "aes"),
))]
mod inverse;
#[cfg(any(
    all(target_arch = "x86_64", target_feature = "pclmulqdq"),
    all(target_arch = "aarch64", target_feature = "aes"),
))]
pub(crate) use inverse::poly_inverse_128;

// Compiled on every target, even where a backend supersedes it.
// Its tests then run everywhere.
// No instruction guarantees the carry argument the software multiply rests on.
#[cfg_attr(
    any(
        all(target_arch = "x86_64", target_feature = "pclmulqdq"),
        all(target_arch = "aarch64", target_feature = "aes"),
    ),
    allow(dead_code)
)]
mod portable;

// The modulus tail, which only a packed backend folds with directly.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "vpclmulqdq",
    any(target_feature = "avx2", target_feature = "avx512f")
))]
pub(crate) use basis::TAIL_128;
pub(crate) use basis::{poly_to_tower_128, tower_image_128, tower_to_poly_128};

use crate::BinaryField64;
use crate::tower::TowerLevel;

#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
mod aarch64;
#[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
mod x86_64;

/// Whether the target has a carryless-multiply instruction.
///
/// - The tower routes here only when it does, since its own recursion beats the software path.
/// - The polynomial-basis field has no such alternative and always routes here.
/// - Most targets need `+pclmulqdq` / `+aes` asked for, or `-C target-cpu=native`.
pub(crate) const HAS_HARDWARE_CLMUL: bool = cfg!(any(
    all(target_arch = "x86_64", target_feature = "pclmulqdq"),
    all(target_arch = "aarch64", target_feature = "aes"),
));

/// The carryless product of two 64-bit polynomials over `GF(2)`, one bit of `b` at a time.
///
/// The plainest possible statement of what every backend computes.
///
/// Far too slow to dispatch to.
/// It exists as the reference the tests check each backend against.
#[cfg(test)]
#[inline]
const fn scalar_clmul_64x64(a: u64, b: u64) -> u128 {
    let a = a as u128;
    let mut acc = 0u128;
    let mut i = 0;
    while i < 64 {
        // All ones when bit `i` of `b` is set, all zeros otherwise.
        let selected = 0u128.wrapping_sub(((b >> i) & 1) as u128);
        acc ^= (a << i) & selected;
        i += 1;
    }
    acc
}

#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
use aarch64::clmul_64x64;
#[cfg(not(any(
    all(target_arch = "x86_64", target_feature = "pclmulqdq"),
    all(target_arch = "aarch64", target_feature = "aes"),
)))]
use portable::clmul_64x64;
#[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
use x86_64::clmul_64x64;

/// The `256`-bit carryless product of two 128-bit polynomials, as `(low, high)`.
///
/// Schoolbook over the 64-bit halves: four independent products, no dependency between them.
/// Karatsuba saves one product, at the price of two exclusive ors ahead of it.
/// A hardware multiply is too cheap for that trade to pay.
// The software backend is `const` and the hardware ones wrap intrinsics, so they cannot be.
// Keeping this uniform stops constness from leaking into callers on some targets only.
#[allow(clippy::missing_const_for_fn)]
#[inline]
fn clmul_128x128(a: u128, b: u128) -> (u128, u128) {
    let (a0, a1) = (a as u64, (a >> 64) as u64);
    let (b0, b1) = (b as u64, (b >> 64) as u64);
    let low = clmul_64x64(a0, b0);
    let high = clmul_64x64(a1, b1);
    let middle = clmul_64x64(a0, b1) ^ clmul_64x64(a1, b0);
    (low ^ (middle << 64), high ^ (middle >> 64))
}

/// Reduces a 128-bit carryless product modulo `x^64 + x^4 + x^3 + x + 1`.
///
/// The modulus rewrites `x^64` as the tail `x^4 + x^3 + x + 1`.
/// The high half therefore folds down by that tail.
/// Degree 4 spills 4 bits back over the top; a second fold lands at degree `3 + 4`.
#[inline]
const fn reduce_64(product: u128) -> u64 {
    let low = product as u64;
    let high = (product >> 64) as u64;

    let folded = (high << 4) ^ (high << 3) ^ (high << 1) ^ high;
    let spill = (high >> 60) ^ (high >> 61) ^ (high >> 63);

    low ^ folded ^ ((spill << 4) ^ (spill << 3) ^ (spill << 1) ^ spill)
}

/// Reduces a 256-bit carryless product modulo `x^128 + x^7 + x^2 + x + 1`.
///
/// The same two-round fold as at 64 bits, with a tail of degree 7.
/// The first round spills 7 bits over the top; the second lands at degree `6 + 7`.
#[inline]
const fn reduce_128(low: u128, high: u128) -> u128 {
    let folded = (high << 7) ^ (high << 2) ^ (high << 1) ^ high;
    let spill = (high >> 121) ^ (high >> 126) ^ (high >> 127);

    low ^ folded ^ ((spill << 7) ^ (spill << 2) ^ (spill << 1) ^ spill)
}

/// Multiplication in `GF(2^64)`, taking and returning the tower representation.
#[inline]
pub(crate) fn mul_64(a: u64, b: u64) -> u64 {
    let product = clmul_64x64(basis::tower_to_poly_64(a), basis::tower_to_poly_64(b));
    basis::poly_to_tower_64(reduce_64(product))
}

/// Multiplication in `GF(2^128)`, in the polynomial representation.
///
/// Assembled and folded in general-purpose registers, so the dependency chain stays short.
#[inline]
fn composed_poly_mul_128(a: u128, b: u128) -> u128 {
    let (low, high) = clmul_128x128(a, b);
    reduce_128(low, high)
}

// The polynomial-basis arithmetic, chosen at compile time.
//
// A backend keeps every intermediate in a vector register and folds the modulus with a
// carryless product, where the integer file would need several instructions per shift.
#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
pub(crate) use aarch64::{poly_dot_128, poly_mul_128, poly_mul_128_by_64, poly_square_128};
#[cfg(not(any(
    all(target_arch = "x86_64", target_feature = "pclmulqdq"),
    all(target_arch = "aarch64", target_feature = "aes"),
)))]
pub(crate) use portable::{poly_dot_128, poly_mul_128, poly_mul_128_by_64, poly_square_128};
#[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
pub(crate) use x86_64::{poly_dot_128, poly_mul_128, poly_mul_128_by_64, poly_square_128};

/// Multiplication in `GF(2^128)`, taking and returning the tower representation.
///
/// The three changes of basis are sixteen dependent lookups each.
/// Behind a chain of loads that long there is no throughput for a vector kernel to win.
#[inline]
pub(crate) fn mul_128(a: u128, b: u128) -> u128 {
    let product = composed_poly_mul_128(basis::tower_to_poly_128(a), basis::tower_to_poly_128(b));
    basis::poly_to_tower_128(product)
}

/// Squaring in `GF(2^64)`, taking and returning the tower representation.
///
/// Squaring is `GF(2)`-linear, so in the tower basis it is a fixed matrix.
/// Tabulated a byte at a time: eight lookups, against two conversions and a product.
#[inline]
pub(crate) fn square_64(a: u64) -> u64 {
    basis::tower_square_64(a)
}

/// Squaring in `GF(2^128)`, taking and returning the tower representation.
///
/// This level's basis is the level below's twice over.
/// Both halves therefore square through the narrower table.
/// Tabulating this level directly would cost four times the space for the same lookups.
#[inline]
pub(crate) fn square_128(a: u128) -> u128 {
    // Invariant: with a = a0 + a1*X and X^2 = alpha*X + 1, the cross term doubles to zero.
    //
    //     a^2 = (a0^2 + a1^2) + alpha*a1^2*X
    let low = square_64(a as u64);
    let high = square_64((a >> 64) as u64);
    let scaled = BinaryField64::from_repr(high).mul_alpha().to_repr();
    u128::from(low ^ high) | (u128::from(scaled) << 64)
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::basis::{TAIL_64, TAIL_128, poly_mul};
    use crate::tower::TowerLevel;
    use crate::{BinaryField64, BinaryField128};

    /// Squaring in `GF(2^64)`, through the polynomial basis.
    ///
    /// The independent definition of the tower-basis square, leaning on the change of basis and
    /// the carryless product rather than on a matrix of its own.
    fn square_64_through_the_polynomial_basis(a: u64) -> u64 {
        let poly = super::basis::tower_to_poly_64(a);
        super::basis::poly_to_tower_64(super::reduce_64(super::clmul_64x64(poly, poly)))
    }

    /// Squaring in `GF(2^128)`, through the polynomial basis.
    fn square_128_through_the_polynomial_basis(a: u128) -> u128 {
        super::basis::poly_to_tower_128(super::poly_square_128(super::basis::tower_to_poly_128(a)))
    }

    /// The carryless product of two 128-bit polynomials, one bit of `b` at a time.
    fn scalar_clmul_128x128(a: u128, b: u128) -> (u128, u128) {
        let mut low = 0u128;
        let mut high = 0u128;
        for i in 0..128 {
            if (b >> i) & 1 == 1 {
                low ^= a << i;
                // A shift by the full width is undefined, so the `i = 0` case is spelled out.
                high ^= if i == 0 { 0 } else { a >> (128 - i) };
            }
        }
        (low, high)
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(2000))]

        #[test]
        fn clmul_matches_reference(a: u128, b: u128) {
            let x = BinaryField128::from_repr(a);
            let y = BinaryField128::from_repr(b);
            prop_assert_eq!(super::mul_128(a, b), x.reference_mul(y).to_repr());
        }

        #[test]
        fn clmul_64_matches_reference(a: u64, b: u64) {
            let x = BinaryField64::from_repr(a);
            let y = BinaryField64::from_repr(b);
            prop_assert_eq!(super::mul_64(a, b), x.reference_mul(y).to_repr());
        }

        #[test]
        fn clmul_square_matches_reference(a: u128) {
            let x = BinaryField128::from_repr(a);
            prop_assert_eq!(super::square_128(a), x.reference_mul(x).to_repr());
        }

        #[test]
        fn clmul_64_square_matches_reference(a: u64) {
            let x = BinaryField64::from_repr(a);
            prop_assert_eq!(super::square_64(a), x.reference_mul(x).to_repr());
        }

        /// The tower-basis squaring matrix must agree with squaring through the polynomial
        /// basis, at both levels that take it.
        #[test]
        fn the_squaring_tables_agree_with_the_polynomial_basis_route(a: u64, b: u128) {
            prop_assert_eq!(super::square_64(a), square_64_through_the_polynomial_basis(a));
            prop_assert_eq!(
                super::square_128(b),
                square_128_through_the_polynomial_basis(b),
            );
        }

        /// Whichever backend the target selected must agree with the bit-serial definition.
        #[test]
        fn the_dispatched_carryless_product_matches_the_bit_serial_one(a: u64, b: u64) {
            prop_assert_eq!(super::clmul_64x64(a, b), super::scalar_clmul_64x64(a, b));
        }

        /// The half-product decomposition must reproduce the bit-serial 256-bit product.
        #[test]
        fn the_wide_product_matches_the_bit_serial_one(a: u128, b: u128) {
            prop_assert_eq!(super::clmul_128x128(a, b), scalar_clmul_128x128(a, b));
        }

        /// The shift-and-fold reduction must agree with the bit-serial modular multiplication.
        #[test]
        fn the_reduction_agrees_with_bit_serial_modular_multiplication(a: u128, b: u128) {
            let (low, high) = scalar_clmul_128x128(a, b);
            prop_assert_eq!(super::reduce_128(low, high), poly_mul(a, b, 128, TAIL_128));

            let product = super::scalar_clmul_64x64(a as u64, b as u64);
            prop_assert_eq!(
                u128::from(super::reduce_64(product)),
                poly_mul(u128::from(a as u64), u128::from(b as u64), 64, TAIL_64),
            );
        }
    }

    proptest! {
        // The selected backend is the one routine here written per target rather than once, so
        // it is checked over many more pairs than the rest of the module.
        #![proptest_config(ProptestConfig::with_cases(20_000))]

        #[test]
        fn the_selected_backend_matches_multiplication_from_the_modulus(a: u128, b: u128) {
            // Bit-serial reduction straight from the modulus, independent of every backend.
            prop_assert_eq!(super::poly_mul_128(a, b), poly_mul(a, b, 128, TAIL_128));
            prop_assert_eq!(super::poly_square_128(a), poly_mul(a, a, 128, TAIL_128));
            prop_assert_eq!(super::poly_mul_128_by_64(a, b as u64), poly_mul(a, b as u64 as u128, 128, TAIL_128));
        }
    }

    #[test]
    fn the_selected_backend_is_exact_on_the_extremes() {
        // Invariant: the fold of the modulus is exact however far the product overflows.
        //
        // The corners a random search is unlikely to reach:
        //   - the additive and multiplicative identities,
        //   - the operands whose half products are maximal,
        //   - the operands whose reduction spill is maximal.
        const CORNERS: [u128; 8] = [
            0,
            1,
            u128::MAX,
            1 << 127,
            1 << 64,
            (1u128 << 64) - 1,
            u128::MAX << 64,
            0x8000_0000_0000_0001_8000_0000_0000_0001,
        ];

        for a in CORNERS {
            assert_eq!(
                super::poly_square_128(a),
                poly_mul(a, a, 128, TAIL_128),
                "{a:#x} squared"
            );
            for b in CORNERS {
                assert_eq!(
                    super::poly_mul_128(a, b),
                    poly_mul(a, b, 128, TAIL_128),
                    "{a:#x} * {b:#x}"
                );
            }
        }
    }

    /// A worked example small enough to check by hand.
    /// In the tower `X_0^2 = X_0 + 1`, so `0b10` squares to `0b11` at both levels.
    #[test]
    fn hand_checkable_products() {
        assert_eq!(super::mul_128(0b10, 0b10), 0b11);
        assert_eq!(super::mul_64(0b10, 0b10), 0b11);

        assert_eq!(super::mul_128(0, 0xdead_beef), 0);
        assert_eq!(super::mul_64(0, 0xdead_beef), 0);

        let a = 0x0123_4567_89ab_cdef_fedc_ba98_7654_3210u128;
        assert_eq!(super::mul_128(a, 1), a);
        assert_eq!(super::mul_128(1, a), a);
        assert_eq!(super::mul_64(a as u64, 1), a as u64);
        assert_eq!(super::mul_64(1, a as u64), a as u64);
    }
}
