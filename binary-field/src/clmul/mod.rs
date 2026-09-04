//! Carryless multiplication for the 64- and 128-bit levels of the tower.
//!
//! Hardware carryless multiply computes products in `GF(2)[x]`, which is not the tower's own
//! representation. [`basis`] supplies the change of basis between the two, leaving three steps:
//! map both operands into the polynomial basis, multiply and reduce there, and map back.
//!
//! Architecture-specific code is confined to the backends: the `64 × 64 → 128` carryless
//! product, and on some targets a whole `GF(2^128)` product that never leaves the vector unit.
//! Composing the wider product from the half products and folding the modulus are plain shifts
//! and `XOR`s, so the definition of every routine here is portable code that the tests exercise
//! on every target, and each backend is checked against it.

mod basis;

pub(crate) use basis::{poly_to_tower_128, tower_to_poly_128};

use crate::BinaryField64;
use crate::tower::TowerLevel;

#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
mod aarch64;
#[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
mod x86_64;

/// Whether the target has a carryless-multiply instruction.
///
/// The bit-serial fallback below keeps every routine in this module correct everywhere, but it
/// is far slower than the recursive tower arithmetic, so `Mul` and `square` only route through
/// this module when the instruction is really there.
///
/// This is a compile-time decision, and neither feature is in the baseline of most targets:
/// `aarch64-apple-darwin` has `aes`, but generic AArch64 Linux and every `x86_64` target need
/// `-C target-feature=+aes` / `+pclmulqdq` (or `-C target-cpu=native`) for the fast path.
pub(crate) const HAS_HARDWARE_CLMUL: bool = cfg!(any(
    all(target_arch = "x86_64", target_feature = "pclmulqdq"),
    all(target_arch = "aarch64", target_feature = "aes"),
));

/// The carryless product of two 64-bit polynomials over `GF(2)`, one bit of `b` at a time.
///
/// The portable definition of the operation the hardware instruction performs. It is compiled
/// on every target: where there is no instruction it is the fallback, and where there is one it
/// is what the tests check that instruction against.
#[cfg_attr(
    any(
        all(target_arch = "x86_64", target_feature = "pclmulqdq"),
        all(target_arch = "aarch64", target_feature = "aes"),
    ),
    // Superseded by a backend, but still the reference the tests compare against.
    allow(dead_code)
)]
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
#[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
use x86_64::clmul_64x64;

#[cfg(not(any(
    all(target_arch = "x86_64", target_feature = "pclmulqdq"),
    all(target_arch = "aarch64", target_feature = "aes"),
)))]
// The backends this stands in for wrap an intrinsic and so cannot be `const`; keeping the
// signatures uniform stops constness from leaking into everything built on top.
#[allow(clippy::missing_const_for_fn)]
#[inline]
fn clmul_64x64(a: u64, b: u64) -> u128 {
    scalar_clmul_64x64(a, b)
}

/// The `256`-bit carryless product of two 128-bit polynomials, as `(low, high)`.
///
/// Schoolbook over the 64-bit halves: with `a = a0 + a1·x^64` and `b = b0 + b1·x^64`, the four
/// half products give the low, middle and high coefficients directly.
///
/// Karatsuba writes the middle coefficient as `(a0 + a1)·(b0 + b1) + a0·b0 + a1·b1` and so
/// needs only three products, but it pays four extra `XOR`s for them and puts two of those on
/// the critical path ahead of the product they feed. A hardware carryless multiply is cheap
/// enough that the trade goes the other way.
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
/// `x^64 ≡ x^4 + x^3 + x + 1`, so the high half is folded down by multiplying it by that tail.
/// The tail has degree 4, so the fold spills 4 bits back above `x^64`; folding those in turn
/// reaches degree at most `3 + 4`, well inside the low half, and two rounds are enough.
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
/// The same two-round fold as [`reduce_64`], with a tail of degree 7: the first round spills
/// 7 bits above `x^128` and the second lands at degree at most `6 + 7`.
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

/// Multiplication in `GF(2^128)`, taking and returning the polynomial representation.
///
/// The 256-bit product is assembled and folded in general-purpose registers, so the whole
/// operation is a short dependency chain and nothing waits on a wider one.
#[inline]
fn composed_poly_mul_128(a: u128, b: u128) -> u128 {
    let (low, high) = clmul_128x128(a, b);
    reduce_128(low, high)
}

/// Multiplication in `GF(2^128)`, taking and returning the polynomial representation.
///
/// Where the target has one, this is a kernel that keeps both operands in vector registers from
/// end to end. It issues more instructions than [`composed_poly_mul_128`] but none that leave
/// the vector unit, which is the better trade only for a caller with several products in
/// flight at once; [`composed_poly_mul_128`] is the one to reach for on a dependency chain.
#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[inline]
pub(crate) fn poly_mul_128(a: u128, b: u128) -> u128 {
    aarch64::poly_mul_128(a, b)
}

/// Multiplication in `GF(2^128)`, taking and returning the polynomial representation.
#[cfg(not(all(target_arch = "aarch64", target_feature = "aes")))]
#[inline]
pub(crate) fn poly_mul_128(a: u128, b: u128) -> u128 {
    composed_poly_mul_128(a, b)
}

/// Multiplication in `GF(2^128)`, taking and returning the tower representation.
///
/// The three changes of basis around the product are sixteen dependent lookups each, so this
/// takes the composition rather than the vector kernel: there is no throughput to win behind a
/// chain of loads that long.
#[inline]
pub(crate) fn mul_128(a: u128, b: u128) -> u128 {
    let product = composed_poly_mul_128(basis::tower_to_poly_128(a), basis::tower_to_poly_128(b));
    basis::poly_to_tower_128(product)
}

/// Squaring in `GF(2^64)`, taking and returning the tower representation.
///
/// Squaring is `GF(2)`-linear in characteristic 2, so in the tower basis it is a fixed matrix,
/// tabulated a byte of the input at a time exactly as the changes of basis are. That is eight
/// lookups, against the two conversions and a carryless product a route through the polynomial
/// basis would cost.
#[inline]
pub(crate) fn square_64(a: u64) -> u64 {
    basis::tower_square_64(a)
}

/// Squaring in `GF(2^128)`, taking and returning the polynomial representation.
///
/// With `p = p0 + p1·x^64`, the middle coefficient of `p²` is `2·p0·p1`, which vanishes in
/// characteristic 2. So `p² = p0² + p1²·x^128` exactly: two carryless products with nothing to
/// fold between them, where [`clmul_128x128`] needs four and a middle term.
#[inline]
pub(crate) fn poly_square_128(a: u128) -> u128 {
    let (p0, p1) = (a as u64, (a >> 64) as u64);
    reduce_128(clmul_64x64(p0, p0), clmul_64x64(p1, p1))
}

/// Squaring in `GF(2^128)`, taking and returning the tower representation.
///
/// The tower basis of this level is the tower basis of the level below, twice over, so the two
/// halves square through [`square_64`]: with `a = a0 + a1·X` and `X² = αX + 1`, the cross term
/// vanishes in characteristic 2 and `a² = (a0² + a1²) + α·a1²·X`.
///
/// Tabulating this level directly would take a table four times the size for the same sixteen
/// lookups, so the level below's is used twice instead.
#[inline]
pub(crate) fn square_128(a: u128) -> u128 {
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

    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    proptest! {
        // The kernel below is the one routine here with no portable definition standing behind
        // it, so it is checked over many more pairs than the rest of the module.
        #![proptest_config(ProptestConfig::with_cases(20_000))]

        /// The vector-native kernel must agree bit for bit with the composition it stands in for.
        #[test]
        fn the_vector_kernel_matches_the_composition(a: u128, b: u128) {
            prop_assert_eq!(
                super::aarch64::poly_mul_128(a, b),
                super::composed_poly_mul_128(a, b),
            );
        }
    }

    /// The edge cases a random search is unlikely to reach: the identities, and the operands
    /// whose half products and reduction spill are maximal.
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    #[test]
    fn the_vector_kernel_matches_the_composition_on_extremes() {
        let corners = [
            0,
            1,
            u128::MAX,
            1 << 127,
            1 << 64,
            (1u128 << 64) - 1,
            u128::MAX << 64,
            0x8000_0000_0000_0001_8000_0000_0000_0001,
        ];
        for a in corners {
            for b in corners {
                assert_eq!(
                    super::aarch64::poly_mul_128(a, b),
                    super::composed_poly_mul_128(a, b),
                    "{a:#x} * {b:#x}"
                );
            }
        }
    }

    /// A worked example small enough to check by hand: in the tower `X_0² = X_0 + 1`, so bit
    /// pattern `0b10` squares to `0b11` at both levels, and `ONE` is a two-sided identity.
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
