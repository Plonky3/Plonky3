//! Raw polynomial coordinates for `GF(2^128)` modulo `x^128 + x^7 + x^2 + x + 1`.
//!
//! Prefer the typed field for ordinary arithmetic.
//! These helpers are for buffers whose representation the caller manages.
//!
//! The coordinates stay a bare integer on purpose.
//! Wrapped, the two bases look alike, and a value in the wrong one is silently wrong.

use crate::tower::TowerLevel;
use crate::{BinaryField128, clmul};

/// Whether multiplication and squaring use hardware carryless multiplication.
///
/// Otherwise multiplication uses masked integer products and squaring uses bit spreading.
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

/// Multiply two elements expressed in polynomial coordinates.
// Only the software backend is `const`, so leaving this one out keeps the signature the same
// on every target.
#[allow(clippy::missing_const_for_fn)]
#[must_use]
#[inline]
pub fn mul(a: u128, b: u128) -> u128 {
    clmul::poly_mul_128(a, b)
}

/// The square of an element, given and returned in the polynomial basis.
#[allow(clippy::missing_const_for_fn)]
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
