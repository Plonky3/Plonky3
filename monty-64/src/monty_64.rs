//! An implementation of 64-bit prime fields using Montgomery representation.

use alloc::vec;
use alloc::vec::Vec;
use core::array;
use core::fmt::{self, Debug, Display, Formatter};
use core::hash::Hash;
use core::iter::{Product, Sum};
use core::marker::PhantomData;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use num_bigint::BigUint;
use p3_field::integers::QuotientMap;
use p3_field::{
    Field, Packable, PrimeCharacteristicRing, PrimeField, PrimeField64, RawDataSerializable,
    impl_raw_serializable_primefield64,
};
use p3_util::flatten_to_base;
use rand::Rng;
use rand::distr::{Distribution, StandardUniform};
use serde::{Deserialize, Deserializer, Serialize};

use crate::utils::{MontyParameters64, add, from_monty, mul, sub, to_monty};

/// A 64-bit prime field element in Montgomery form.
#[derive(Clone, Copy, Default, Eq, Hash, PartialEq)]
#[repr(transparent)]
#[must_use]
pub struct MontyField64<MP: MontyParameters64> {
    /// The Montgomery form of the field element, saved as a positive integer less than P.
    pub(crate) value: u64,
    _phantom: PhantomData<MP>,
}

impl<MP: MontyParameters64> MontyField64<MP> {
    /// Create a new field element from a u64 value.
    /// The value is converted to Montgomery form.
    #[inline(always)]
    pub const fn new(value: u64) -> Self {
        Self {
            value: to_monty::<MP>(value),
            _phantom: PhantomData,
        }
    }

    /// Create a new field element from a value already in Montgomery form.
    #[inline(always)]
    pub const fn new_monty(value: u64) -> Self {
        Self {
            value,
            _phantom: PhantomData,
        }
    }

    /// Convert the field element to its canonical u64 representation.
    #[inline(always)]
    pub(crate) const fn to_u64(elem: &Self) -> u64 {
        from_monty::<MP>(elem.value)
    }

    /// Convert a constant u64 array into a constant array of field elements.
    #[inline]
    pub const fn new_array<const N: usize>(input: [u64; N]) -> [Self; N] {
        let mut output = [const { Self::new_monty(0) }; N];
        let mut i = 0;
        while i < N {
            output[i] = Self::new(input[i]);
            i += 1;
        }
        output
    }
}

impl<MP: MontyParameters64> Ord for MontyField64<MP> {
    #[inline]
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        Self::to_u64(self).cmp(&Self::to_u64(other))
    }
}

impl<MP: MontyParameters64> PartialOrd for MontyField64<MP> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<MP: MontyParameters64> Display for MontyField64<MP> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(&Self::to_u64(self), f)
    }
}

impl<MP: MontyParameters64> Debug for MontyField64<MP> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Debug::fmt(&Self::to_u64(self), f)
    }
}

impl<MP: MontyParameters64> Distribution<MontyField64<MP>> for StandardUniform {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> MontyField64<MP> {
        loop {
            let next_u64 = rng.next_u64();
            if next_u64 < MP::PRIME {
                return MontyField64::new_monty(next_u64);
            }
        }
    }
}

impl<MP: MontyParameters64> Serialize for MontyField64<MP> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        // Serialize in Montgomery form for efficiency
        serializer.serialize_u64(self.value)
    }
}

impl<'de, MP: MontyParameters64> Deserialize<'de> for MontyField64<MP> {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        // Deserialize from Montgomery form
        let val = u64::deserialize(d)?;
        Ok(Self::new_monty(val))
    }
}

impl<MP: MontyParameters64> Packable for MontyField64<MP> {}

impl<MP: MontyParameters64> RawDataSerializable for MontyField64<MP> {
    impl_raw_serializable_primefield64!();
}

impl<MP: MontyParameters64> PrimeCharacteristicRing for MontyField64<MP> {
    type PrimeSubfield = Self;

    const ZERO: Self = MP::MONTY_ZERO;
    const ONE: Self = MP::MONTY_ONE;
    const TWO: Self = MP::MONTY_TWO;
    const NEG_ONE: Self = MP::MONTY_NEG_ONE;

    #[inline(always)]
    fn from_prime_subfield(f: Self) -> Self {
        f
    }

    #[inline]
    fn halve(&self) -> Self {
        // Compute (a + P) / 2 if a is odd, otherwise a / 2
        let a = self.value;
        if a & 1 == 1 {
            Self::new_monty(((a as u128 + MP::PRIME as u128) >> 1) as u64)
        } else {
            Self::new_monty(a >> 1)
        }
    }

    #[inline]
    fn zero_vec(len: usize) -> Vec<Self> {
        // SAFETY: Due to #[repr(transparent)], MontyField64 and u64 have the same layout
        unsafe { flatten_to_base(vec![0u64; len]) }
    }
}

impl<MP: MontyParameters64> MontyField64<MP> {
    /// Fast exponentiation using binary exponentiation
    pub fn exp_u64(&self, mut exp: u64) -> Self {
        if exp == 0 {
            return Self::ONE;
        }

        let mut base = *self;
        let mut result = Self::ONE;

        while exp > 0 {
            if exp & 1 == 1 {
                result = result * base;
            }
            base = base * base;
            exp >>= 1;
        }

        result
    }
}

impl<MP: MontyParameters64> Field for MontyField64<MP> {
    type Packing = Self;

    const GENERATOR: Self = MP::MONTY_ONE; // Placeholder - should be set by concrete implementations

    fn try_inverse(&self) -> Option<Self> {
        if self.is_zero() {
            return None;
        }

        // Convert from Montgomery to normal form, compute inverse, convert back
        let normal_val = Self::to_u64(self);

        // Use Fermat's little theorem to compute inverse in normal form
        let inv_normal = {
            let mut base = normal_val;
            let mut exp = MP::PRIME - 2;
            let mut result = 1u64;

            while exp > 0 {
                if exp & 1 == 1 {
                    result = ((result as u128) * (base as u128) % MP::PRIME as u128) as u64;
                }
                base = ((base as u128) * (base as u128) % MP::PRIME as u128) as u64;
                exp >>= 1;
            }
            result
        };

        // Convert back to Montgomery form
        Some(Self::new(inv_normal))
    }

    #[inline]
    fn order() -> BigUint {
        MP::PRIME.into()
    }
}

// Implement QuotientMap for various integer types manually

impl<MP: MontyParameters64> QuotientMap<u8> for MontyField64<MP> {
    #[inline]
    fn from_int(int: u8) -> Self {
        Self::new(int as u64)
    }

    #[inline]
    fn from_canonical_checked(int: u8) -> Option<Self> {
        Some(Self::new(int as u64))
    }

    #[inline(always)]
    unsafe fn from_canonical_unchecked(int: u8) -> Self {
        Self::new(int as u64)
    }
}

impl<MP: MontyParameters64> QuotientMap<u16> for MontyField64<MP> {
    #[inline]
    fn from_int(int: u16) -> Self {
        Self::new(int as u64)
    }

    #[inline]
    fn from_canonical_checked(int: u16) -> Option<Self> {
        Some(Self::new(int as u64))
    }

    #[inline(always)]
    unsafe fn from_canonical_unchecked(int: u16) -> Self {
        Self::new(int as u64)
    }
}

impl<MP: MontyParameters64> QuotientMap<u32> for MontyField64<MP> {
    #[inline]
    fn from_int(int: u32) -> Self {
        Self::new(int as u64)
    }

    #[inline]
    fn from_canonical_checked(int: u32) -> Option<Self> {
        Some(Self::new(int as u64))
    }

    #[inline(always)]
    unsafe fn from_canonical_unchecked(int: u32) -> Self {
        Self::new(int as u64)
    }
}

impl<MP: MontyParameters64> QuotientMap<i8> for MontyField64<MP> {
    #[inline]
    fn from_int(int: i8) -> Self {
        if int >= 0 {
            Self::new(int as u64)
        } else {
            Self::new(MP::PRIME.wrapping_add_signed(int as i64))
        }
    }

    #[inline]
    fn from_canonical_checked(int: i8) -> Option<Self> {
        Some(Self::from_int(int))
    }

    #[inline(always)]
    unsafe fn from_canonical_unchecked(int: i8) -> Self {
        Self::from_int(int)
    }
}

impl<MP: MontyParameters64> QuotientMap<i16> for MontyField64<MP> {
    #[inline]
    fn from_int(int: i16) -> Self {
        if int >= 0 {
            Self::new(int as u64)
        } else {
            Self::new(MP::PRIME.wrapping_add_signed(int as i64))
        }
    }

    #[inline]
    fn from_canonical_checked(int: i16) -> Option<Self> {
        Some(Self::from_int(int))
    }

    #[inline(always)]
    unsafe fn from_canonical_unchecked(int: i16) -> Self {
        Self::from_int(int)
    }
}

impl<MP: MontyParameters64> QuotientMap<i32> for MontyField64<MP> {
    #[inline]
    fn from_int(int: i32) -> Self {
        if int >= 0 {
            Self::new(int as u64)
        } else {
            Self::new(MP::PRIME.wrapping_add_signed(int as i64))
        }
    }

    #[inline]
    fn from_canonical_checked(int: i32) -> Option<Self> {
        Some(Self::from_int(int))
    }

    #[inline(always)]
    unsafe fn from_canonical_unchecked(int: i32) -> Self {
        Self::from_int(int)
    }
}

impl<MP: MontyParameters64> QuotientMap<u64> for MontyField64<MP> {
    #[inline]
    fn from_int(int: u64) -> Self {
        Self::new(int % MP::PRIME)
    }

    #[inline]
    fn from_canonical_checked(int: u64) -> Option<Self> {
        (int < MP::PRIME).then(|| Self::new(int))
    }

    #[inline(always)]
    unsafe fn from_canonical_unchecked(int: u64) -> Self {
        Self::new(int)
    }
}

impl<MP: MontyParameters64> QuotientMap<i64> for MontyField64<MP> {
    #[inline]
    fn from_int(int: i64) -> Self {
        if int >= 0 {
            Self::new(int as u64)
        } else {
            Self::new(MP::PRIME.wrapping_add_signed(int))
        }
    }

    #[inline]
    fn from_canonical_checked(int: i64) -> Option<Self> {
        const POS_BOUND: i64 = (u64::MAX >> 1) as i64; // Placeholder - will be overridden by concrete types
        const NEG_BOUND: i64 = -POS_BOUND;
        match int {
            0..=POS_BOUND if (int as u64) < MP::PRIME => Some(Self::new(int as u64)),
            NEG_BOUND..0 => Some(Self::new(MP::PRIME.wrapping_add_signed(int))),
            _ => None,
        }
    }

    #[inline(always)]
    unsafe fn from_canonical_unchecked(int: i64) -> Self {
        Self::from_int(int)
    }
}

impl<MP: MontyParameters64> QuotientMap<u128> for MontyField64<MP> {
    #[inline]
    fn from_int(int: u128) -> Self {
        Self::new((int % MP::PRIME as u128) as u64)
    }

    #[inline]
    fn from_canonical_checked(int: u128) -> Option<Self> {
        (int < MP::PRIME as u128).then(|| Self::new(int as u64))
    }

    #[inline(always)]
    unsafe fn from_canonical_unchecked(int: u128) -> Self {
        Self::new(int as u64)
    }
}

impl<MP: MontyParameters64> QuotientMap<i128> for MontyField64<MP> {
    #[inline]
    fn from_int(int: i128) -> Self {
        let reduced = int % (MP::PRIME as i128);
        if reduced >= 0 {
            Self::new(reduced as u64)
        } else {
            Self::new((MP::PRIME as i128 + reduced) as u64)
        }
    }

    #[inline]
    fn from_canonical_checked(int: i128) -> Option<Self> {
        let bound = (MP::PRIME >> 1) as i128;
        if int <= bound && int >= -bound {
            Some(Self::from_int(int))
        } else {
            None
        }
    }

    #[inline(always)]
    unsafe fn from_canonical_unchecked(int: i128) -> Self {
        Self::from_int(int)
    }
}

impl<MP: MontyParameters64> PrimeField for MontyField64<MP> {
    fn as_canonical_biguint(&self) -> BigUint {
        self.as_canonical_u64().into()
    }
}

impl<MP: MontyParameters64> PrimeField64 for MontyField64<MP> {
    const ORDER_U64: u64 = MP::PRIME;

    #[inline]
    fn as_canonical_u64(&self) -> u64 {
        Self::to_u64(self)
    }

    #[inline]
    fn to_unique_u64(&self) -> u64 {
        // The internal representation is already unique
        self.value
    }
}

// Arithmetic operations

impl<MP: MontyParameters64> Add for MontyField64<MP> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::new_monty(add::<MP>(self.value, rhs.value))
    }
}

impl<MP: MontyParameters64> Sub for MontyField64<MP> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::new_monty(sub::<MP>(self.value, rhs.value))
    }
}

impl<MP: MontyParameters64> Mul for MontyField64<MP> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self::new_monty(mul::<MP>(self.value, rhs.value))
    }
}

impl<MP: MontyParameters64> Neg for MontyField64<MP> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        Self::ZERO - self
    }
}

impl<MP: MontyParameters64> Div for MontyField64<MP> {
    type Output = Self;

    #[inline]
    fn div(self, rhs: Self) -> Self {
        self * rhs.inverse()
    }
}

// Assignment operators

impl<MP: MontyParameters64> AddAssign for MontyField64<MP> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<MP: MontyParameters64> SubAssign for MontyField64<MP> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<MP: MontyParameters64> MulAssign for MontyField64<MP> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<MP: MontyParameters64> DivAssign for MontyField64<MP> {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl<MP: MontyParameters64> Sum for MontyField64<MP> {
    #[inline]
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        // Use u128 accumulator to avoid overflow
        let sum = iter.map(|x| x.value as u128).sum::<u128>();
        Self::new_monty((sum % MP::PRIME as u128) as u64)
    }
}

impl<MP: MontyParameters64> Product for MontyField64<MP> {
    #[inline]
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ONE, |acc, x| acc * x)
    }
}
