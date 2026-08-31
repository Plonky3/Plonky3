//! `GF(2^128)` in the polynomial basis of `x^128 + x^7 + x^2 + x + 1`.
//!
//! [`BinaryField128`] stores its elements in the tower basis, and a product there is a change of
//! basis on each operand, a carryless multiply, and a change of basis back — three table-driven
//! conversions of sixteen dependent lookups each, which dominate the multiply. Code that
//! performs many products on the same data can pay the conversion once at each end instead, and
//! multiply through [`mul`] in between.
//!
//! Addition is `XOR` in either basis, so a sum needs no conversion at all: only products and
//! squares do, which is what this module supplies. An element is carried as the `u128` of its
//! coordinates, deliberately not as a field type — the two bases are indistinguishable once
//! wrapped, and a value that reaches ordinary [`BinaryField128`] arithmetic in the wrong one is
//! silently wrong rather than ill-typed.

use crate::tower::TowerLevel;
use crate::{BinaryField128, clmul};

/// Whether [`mul`] and [`square`] compile down to a hardware carryless multiply.
///
/// Where they do not, the carryless product is the bit-serial fallback, which is far slower than
/// the recursive tower arithmetic `BinaryField128` uses on such a target. Working in this basis
/// is a loss there, so a caller choosing between the two representations should branch on this.
pub const HAS_HARDWARE_CLMUL: bool = clmul::HAS_HARDWARE_CLMUL;

/// The polynomial-basis coordinates of a tower element.
#[must_use]
#[inline]
pub fn from_tower(x: BinaryField128) -> u128 {
    clmul::tower_to_poly_128(x.to_repr())
}

/// The tower element with the given polynomial-basis coordinates.
#[inline]
pub fn to_tower(v: u128) -> BinaryField128 {
    BinaryField128::from_repr(clmul::poly_to_tower_128(v))
}

/// The product of two elements, both given and returned in the polynomial basis.
#[must_use]
#[inline]
pub fn mul(a: u128, b: u128) -> u128 {
    clmul::poly_mul_128(a, b)
}

/// The square of an element, given and returned in the polynomial basis.
#[must_use]
#[inline]
pub fn square(a: u128) -> u128 {
    clmul::poly_square_128(a)
}

#[cfg(test)]
mod tests {
    use p3_field::PrimeCharacteristicRing;
    use proptest::prelude::*;

    use super::{from_tower, mul, square, to_tower};
    use crate::BinaryField128;

    /// Building an element from a 128-bit pattern, which every pattern is a valid one of.
    fn element(bits: u128) -> BinaryField128 {
        BinaryField128::from_le_bytes(bits.to_le_bytes())
    }

    proptest! {
        #[test]
        fn the_change_of_basis_round_trips(bits in any::<u128>()) {
            let x = element(bits);
            prop_assert_eq!(to_tower(from_tower(x)), x);
        }

        /// Addition is `XOR` in both bases, so the conversion commutes with it.
        #[test]
        fn the_change_of_basis_is_additive(a in any::<u128>(), b in any::<u128>()) {
            let (x, y) = (element(a), element(b));
            prop_assert_eq!(from_tower(x + y), from_tower(x) ^ from_tower(y));
        }

        /// The polynomial-basis product is the tower product, seen in the other basis.
        #[test]
        fn the_product_agrees_with_the_tower_product(a in any::<u128>(), b in any::<u128>()) {
            let (x, y) = (element(a), element(b));
            prop_assert_eq!(to_tower(mul(from_tower(x), from_tower(y))), x * y);
        }

        #[test]
        fn the_square_agrees_with_the_tower_square(a in any::<u128>()) {
            let x = element(a);
            prop_assert_eq!(to_tower(square(from_tower(x))), x.square());
        }
    }

    /// The basis change fixes zero and one, as any field isomorphism must.
    #[test]
    fn the_change_of_basis_fixes_the_constants() {
        assert_eq!(from_tower(BinaryField128::ZERO), 0);
        assert_eq!(from_tower(BinaryField128::ONE), 1);
        assert_eq!(to_tower(0), BinaryField128::ZERO);
        assert_eq!(to_tower(1), BinaryField128::ONE);
    }
}
