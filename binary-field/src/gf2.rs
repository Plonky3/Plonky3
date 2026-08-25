use core::fmt::{self, Debug, Display, Formatter};
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use num_bigint::BigUint;
use p3_field::integers::QuotientMap;
use p3_field::op_assign_macros::{
    impl_add_assign, impl_div_methods, impl_mul_methods, impl_sub_assign, ring_sum,
};
use p3_field::{
    Field, Packable, PrimeCharacteristicRing, PrimeField, PrimeField32, PrimeField64,
    RawDataSerializable, quotient_map_large_iint, quotient_map_large_uint,
};
use rand::Rng;
use rand::distr::{Distribution, StandardUniform};
use serde::{Deserialize, Serialize};

/// The prime field `GF(2) = {0, 1}`, with addition given by `XOR` and multiplication by `AND`.
///
/// This is the base case of the characteristic-2 tower `GF(2) ⊂ GF(4) ⊂ … ⊂ GF(2^128)`.
#[derive(Copy, Clone, Default, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[repr(transparent)]
#[must_use]
pub struct Gf2(u8);

impl Gf2 {
    /// Create a new field element from a bit.
    ///
    /// Only the least significant bit of `bit` is used.
    #[inline]
    const fn new(bit: u8) -> Self {
        Self(bit & 1)
    }
}

impl Packable for Gf2 {}

impl Display for Gf2 {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(&self.0, f)
    }
}

impl Debug for Gf2 {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Debug::fmt(&self.0, f)
    }
}

impl Distribution<Gf2> for StandardUniform {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Gf2 {
        Gf2::new((rng.next_u32() & 1) as u8)
    }
}

impl PrimeCharacteristicRing for Gf2 {
    type PrimeSubfield = Self;

    const ZERO: Self = Self(0);
    const ONE: Self = Self(1);
    // The characteristic is 2, so `TWO = ONE + ONE = ZERO`.
    const TWO: Self = Self(0);
    // The characteristic is 2, so `NEG_ONE = ONE`.
    const NEG_ONE: Self = Self(1);

    #[inline]
    fn from_prime_subfield(f: Self::PrimeSubfield) -> Self {
        f
    }

    #[inline]
    fn from_bool(b: bool) -> Self {
        Self(b as u8)
    }

    #[inline]
    fn double(&self) -> Self {
        // `a + a = 0` in characteristic 2.
        Self::ZERO
    }

    /// # Panics
    /// Always panics: `2` is not invertible in characteristic 2.
    #[inline]
    fn halve(&self) -> Self {
        panic!("halve is undefined in characteristic 2")
    }

    #[inline]
    fn square(&self) -> Self {
        // `0^2 = 0` and `1^2 = 1`.
        *self
    }

    #[inline]
    fn xor(&self, y: &Self) -> Self {
        *self + *y
    }

    #[inline]
    fn mul_2exp_u64(&self, exp: u64) -> Self {
        if exp == 0 { *self } else { Self::ZERO }
    }

    /// # Panics
    /// Always panics: `2` is not invertible in characteristic 2.
    #[inline]
    fn div_2exp_u64(&self, _exp: u64) -> Self {
        panic!("div_2exp_u64 is undefined in characteristic 2")
    }
}

impl Field for Gf2 {
    type Packing = Self;

    // The only nonzero element, `1`, trivially generates the (trivial) multiplicative group.
    const GENERATOR: Self = Self::ONE;

    #[inline]
    fn try_inverse(&self) -> Option<Self> {
        (self.0 == 1).then_some(Self::ONE)
    }

    #[inline]
    fn try_sqrt(&self) -> Option<Self> {
        // `0^2 = 0` and `1^2 = 1`, so every element is its own (unique) square root.
        Some(*self)
    }

    #[inline]
    fn order() -> BigUint {
        BigUint::from(2u8)
    }

    /// The enumeration `interpolation_node(0) = 0, interpolation_node(1) = 1` is the only
    /// injective enumeration of the two elements of `GF(2)` satisfying `Field::interpolation_node`'s
    /// contract. It is not injective beyond index `1`.
    #[inline]
    fn interpolation_node(i: usize) -> Self {
        Self::new(i as u8)
    }
}

impl QuotientMap<u8> for Gf2 {
    /// Convert a given `u8` integer into an element of the `Gf2` field.
    #[inline]
    fn from_int(int: u8) -> Self {
        Self::new(int)
    }

    /// Convert a given `u8` integer into an element of the `Gf2` field.
    ///
    /// Returns `None` if the input does not lie in the range `[0, 1]`.
    #[inline]
    fn from_canonical_checked(int: u8) -> Option<Self> {
        (int < 2).then(|| Self::new(int))
    }

    /// Convert a given `u8` integer into an element of the `Gf2` field.
    ///
    /// # Safety
    /// The caller must guarantee that `int < 2`.
    #[inline]
    unsafe fn from_canonical_unchecked(int: u8) -> Self {
        // SAFETY: the caller guarantees `int < 2`, so `int` is already the canonical
        // representative and masking to its least significant bit is a no-op.
        debug_assert!(int < 2);
        Self::new(int)
    }
}

impl QuotientMap<i8> for Gf2 {
    /// Convert a given `i8` integer into an element of the `Gf2` field.
    #[inline]
    fn from_int(int: i8) -> Self {
        // Two's-complement `& 1` extracts the parity bit regardless of sign.
        Self::new((int & 1) as u8)
    }

    /// Convert a given `i8` integer into an element of the `Gf2` field.
    ///
    /// Returns `None` if the input does not lie in the range `[0, 1]`.
    #[inline]
    fn from_canonical_checked(int: i8) -> Option<Self> {
        (0..2).contains(&int).then(|| Self::new(int as u8))
    }

    /// Convert a given `i8` integer into an element of the `Gf2` field.
    ///
    /// # Safety
    /// The caller must guarantee that `int` lies in `[0, 1]`.
    #[inline]
    unsafe fn from_canonical_unchecked(int: i8) -> Self {
        // SAFETY: the caller guarantees `0 <= int < 2`, so `int` is already the
        // canonical representative and casting to `u8` is exact.
        debug_assert!((0..2).contains(&int));
        Self::new(int as u8)
    }
}

// `u8` is the smallest integer type, so `QuotientMap<u8>` is implemented by hand above.
quotient_map_large_uint!(Gf2, u8, 2u8, "`[0, 1]`", "`[0, 1]`", [u16, u32, u64, u128]);

// `i8` is the smallest signed integer type, so `QuotientMap<i8>` is implemented by hand above.
quotient_map_large_iint!(
    Gf2,
    i8,
    "`[0, 1]`",
    "`[0, 1]`",
    [(i16, u16), (i32, u32), (i64, u64), (i128, u128)]
);

impl PrimeField for Gf2 {
    #[inline]
    fn as_canonical_biguint(&self) -> BigUint {
        BigUint::from(self.0)
    }
}

impl PrimeField64 for Gf2 {
    const ORDER_U64: u64 = 2;

    #[inline]
    fn as_canonical_u64(&self) -> u64 {
        self.0 as u64
    }
}

impl PrimeField32 for Gf2 {
    const ORDER_U32: u32 = 2;

    #[inline]
    fn as_canonical_u32(&self) -> u32 {
        self.0 as u32
    }
}

impl RawDataSerializable for Gf2 {
    const NUM_BYTES: usize = 1;

    #[inline]
    fn into_bytes(self) -> impl IntoIterator<Item = u8> {
        [self.0]
    }
}

impl Add for Gf2 {
    type Output = Self;

    #[inline]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn add(self, rhs: Self) -> Self {
        // Addition in `GF(2)` is `XOR`.
        Self(self.0 ^ rhs.0)
    }
}

impl Sub for Gf2 {
    type Output = Self;

    #[inline]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn sub(self, rhs: Self) -> Self {
        // Subtraction coincides with addition in characteristic 2.
        self + rhs
    }
}

impl Neg for Gf2 {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        // `-x = x` in characteristic 2.
        self
    }
}

impl Mul for Gf2 {
    type Output = Self;

    #[inline]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn mul(self, rhs: Self) -> Self {
        // Multiplication in `GF(2)` is `AND`.
        Self(self.0 & rhs.0)
    }
}

impl_add_assign!(Gf2);
impl_sub_assign!(Gf2);
impl_mul_methods!(Gf2);
impl_div_methods!(Gf2, Gf2);
ring_sum!(Gf2);

#[cfg(test)]
mod tests {
    use p3_field::{Field, PrimeCharacteristicRing};

    use super::Gf2;

    #[test]
    fn arithmetic_is_boolean() {
        assert_eq!(Gf2::ONE + Gf2::ONE, Gf2::ZERO);
        assert_eq!(Gf2::TWO, Gf2::ZERO);
        assert_eq!(Gf2::NEG_ONE, Gf2::ONE);
        assert_eq!(Gf2::ONE * Gf2::ONE, Gf2::ONE);
        assert_eq!(Gf2::ONE.inverse(), Gf2::ONE);
        assert_eq!(Gf2::ZERO.try_inverse(), None);
        assert_eq!(Gf2::from_u64(7), Gf2::ONE);
        assert_eq!(Gf2::from_u64(6), Gf2::ZERO);
    }
}
