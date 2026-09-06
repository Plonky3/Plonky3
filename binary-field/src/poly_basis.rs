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

/// Multiply every polynomial-basis element by the same scalar.
#[inline]
pub fn mul_slice(values: &mut [u128], scalar: u128) {
    if let [value] = values {
        *value = mul(*value, scalar);
        return;
    }
    for value in values {
        *value = clmul::poly_mul_128_batch(*value, scalar);
    }
}

/// Apply `(lo, hi) -> (lo + scalar*hi, lo + (scalar + 1)*hi)` in place.
///
/// # Panics
/// Panics if the slice lengths differ.
#[inline]
pub fn butterfly_forward(lo: &mut [u128], hi: &mut [u128], scalar: u128) {
    assert_eq!(lo.len(), hi.len(), "butterfly lengths differ");
    if scalar == 0 {
        for (lo, hi) in lo.iter().zip(hi) {
            *hi ^= *lo;
        }
        return;
    }
    if lo.len() == 1 {
        lo[0] ^= mul(scalar, hi[0]);
        hi[0] ^= lo[0];
        return;
    }
    for (lo, hi) in lo.iter_mut().zip(hi) {
        *lo ^= clmul::poly_mul_128_batch(scalar, *hi);
        *hi ^= *lo;
    }
}

/// Undo [`butterfly_forward`] with the same scalar.
///
/// # Panics
/// Panics if the slice lengths differ.
#[inline]
pub fn butterfly_inverse(lo: &mut [u128], hi: &mut [u128], scalar: u128) {
    assert_eq!(lo.len(), hi.len(), "butterfly lengths differ");
    if scalar == 0 {
        for (lo, hi) in lo.iter().zip(hi) {
            *hi ^= *lo;
        }
        return;
    }
    if lo.len() == 1 {
        hi[0] ^= lo[0];
        lo[0] ^= mul(scalar, hi[0]);
        return;
    }
    for (lo, hi) in lo.iter_mut().zip(hi) {
        *hi ^= *lo;
        *lo ^= clmul::poly_mul_128_batch(scalar, *hi);
    }
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

    #[test]
    fn slice_operations_match_reference_tower_arithmetic() {
        use alloc::vec::Vec;
        for len in [0, 1, 2, 3, 8, 17, 64, 257] {
            for scalar in [0, 1, 0xfeed_9876_0123_4567_89ab_cdef_9876_5432] {
                let t = element(scalar);
                let xs: Vec<_> = (0..len)
                    .map(|i| {
                        element(
                            (i as u128 + 1).wrapping_mul(0xfeed_8765_dead_beef_cafe_0123_4567_89ab),
                        )
                    })
                    .collect();
                let ys: Vec<_> = xs.iter().rev().copied().collect();
                let mut lo: Vec<_> = xs.iter().copied().map(from_tower).collect();
                let mut hi: Vec<_> = ys.iter().copied().map(from_tower).collect();
                let original = (lo.clone(), hi.clone());
                super::butterfly_forward(&mut lo, &mut hi, from_tower(t));
                for i in 0..len {
                    let expected = xs[i] + ys[i].reference_mul(t);
                    assert_eq!(to_tower(lo[i]), expected);
                    assert_eq!(to_tower(hi[i]), expected + ys[i]);
                }
                super::butterfly_inverse(&mut lo, &mut hi, from_tower(t));
                assert_eq!((&lo, &hi), (&original.0, &original.1));
                super::mul_slice(&mut lo, from_tower(t));
                for i in 0..len {
                    assert_eq!(to_tower(lo[i]), xs[i].reference_mul(t));
                }
            }
        }
    }

    #[test]
    #[should_panic = "butterfly lengths differ"]
    fn forward_rejects_mismatched_lengths() {
        super::butterfly_forward(&mut [0], &mut [], 0);
    }

    #[test]
    #[should_panic = "butterfly lengths differ"]
    fn inverse_rejects_mismatched_lengths() {
        super::butterfly_inverse(&mut [], &mut [0], 1);
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
