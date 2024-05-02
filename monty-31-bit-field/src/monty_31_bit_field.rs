//! An abstraction of 31-bit fields which use a MONTY approach for addition and multiplication with a MONTY constant = 2^32.

use alloc::vec::Vec;
use core::fmt::{self, Debug, Display, Formatter};
use core::hash::Hash;
use core::iter::{Product, Sum};
use core::marker::PhantomData;
use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

use num_bigint::BigUint;
use p3_field::{
    AbstractField, Field, Packable, PrimeField, PrimeField32, PrimeField64, TwoAdicField,
};
use rand::distributions::{Distribution, Standard};
use rand::Rng;
use serde::{Deserialize, Deserializer, Serialize};

#[derive(Copy, Clone, Default, Eq, Hash, PartialEq)]
#[repr(transparent)] // Packed field implementations rely on this!
pub struct MontyField31<MP: MontyParameters> {
    pub(crate) value: u32,
    _phantom: PhantomData<MP>,
}

impl<MP: MontyParameters> MontyField31<MP> {
    pub const fn new(value: u32) -> Self {
        Self {
            value: to_monty::<MP>(value),
            _phantom: PhantomData,
        }
    }

    pub(crate) const fn new_monty(value: u32) -> Self {
        Self {
            value: value,
            _phantom: PhantomData,
        }
    }
}

impl<MP: MontyParameters> Ord for MontyField31<MP> {
    #[inline]
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.as_canonical_u32().cmp(&other.as_canonical_u32())
    }
}

impl<MP: MontyParameters> PartialOrd for MontyField31<MP> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<MP: MontyParameters> Display for MontyField31<MP> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(&self.as_canonical_u32(), f)
    }
}

impl<MP: MontyParameters> Debug for MontyField31<MP> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Debug::fmt(&self.as_canonical_u32(), f)
    }
}

impl<MP: MontyParameters> Distribution<MontyField31<MP>> for Standard {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> MontyField31<MP> {
        loop {
            let next_u31 = rng.next_u32() >> 1;
            let is_canonical = next_u31 < MP::PRIME;
            if is_canonical {
                return MontyField31::new_monty(next_u31);
            }
        }
    }
}

impl<MP: MontyParameters> Serialize for MontyField31<MP> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_u32(self.as_canonical_u32())
    }
}

impl<'de, MP: MontyParameters> Deserialize<'de> for MontyField31<MP> {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let val = u32::deserialize(d)?;
        Ok(MontyField31::from_canonical_u32(val))
    }
}

pub trait MontyParameters:
    Copy + Clone + Default + Eq + PartialEq + Sync + Send + Hash + 'static + TwoAdicData
{
    const PRIME: u32;

    // Constants used for multiplication and similar
    const MONTY_BITS: u32;
    const MONTY_MU: u32;
    const MONTY_MASK: u32 = ((1u64 << Self::MONTY_BITS) - 1) as u32;

    // Simple Field Values.
    const MONTY_ZERO: u32 = 0; // The monty form of 0 is always 0.
    const MONTY_ONE: u32 = to_monty::<Self>(1);
    const MONTY_TWO: u32 = to_monty::<Self>(2);
    const MONTY_NEG_ONE: u32 = Self::PRIME - Self::MONTY_ONE; // As MONTY_ONE =/= 0, MONTY_NEG_ONE = P - MONTY_ONE.

    const GEN: u32; // A generator of the fields multiplicative group.
    const MONTY_GEN: u32 = to_monty::<Self>(Self::GEN); // Generator saved in MONTY form

    fn exp_u64_generic<AF: AbstractField>(val: AF, power: u64) -> AF;
    fn try_inverse<AF: AbstractField>(p1: AF) -> Option<AF>;
}

pub trait TwoAdicData {
    const TWO_ADICITY: usize;
    const GENERATORS: Vec<u32>;
}

impl<MP: MontyParameters> Packable for MontyField31<MP> {}

impl<MP: MontyParameters> AbstractField for MontyField31<MP> {
    type F = Self;

    fn zero() -> Self {
        Self::new_monty(MP::MONTY_ZERO)
    }
    fn one() -> Self {
        Self::new_monty(MP::MONTY_ONE)
    }
    fn two() -> Self {
        Self::new_monty(MP::MONTY_TWO)
    }
    fn neg_one() -> Self {
        Self::new_monty(MP::MONTY_NEG_ONE)
    }

    #[inline]
    fn from_f(f: Self::F) -> Self {
        f
    }

    #[inline]
    fn from_bool(b: bool) -> Self {
        Self::from_canonical_u32(b as u32)
    }

    #[inline]
    fn from_canonical_u8(n: u8) -> Self {
        Self::from_canonical_u32(n as u32)
    }

    #[inline]
    fn from_canonical_u16(n: u16) -> Self {
        Self::from_canonical_u32(n as u32)
    }

    #[inline]
    fn from_canonical_u32(n: u32) -> Self {
        debug_assert!(n < MP::PRIME);
        Self::from_wrapped_u32(n)
    }

    #[inline]
    fn from_canonical_u64(n: u64) -> Self {
        debug_assert!(n < MP::PRIME as u64);
        Self::from_canonical_u32(n as u32)
    }

    #[inline]
    fn from_canonical_usize(n: usize) -> Self {
        debug_assert!(n < MP::PRIME as usize);
        Self::from_canonical_u32(n as u32)
    }

    #[inline]
    fn from_wrapped_u32(n: u32) -> Self {
        Self::new(n)
    }

    #[inline]
    fn from_wrapped_u64(n: u64) -> Self {
        Self::new_monty(to_monty_64::<MP>(n))
    }

    #[inline]
    fn generator() -> Self {
        Self::new_monty(MP::MONTY_GEN)
    }
}

impl<MP: MontyParameters> Field for MontyField31<MP> {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    type Packing = todo!();
    // type Packing = crate::PackedMontyField31Neon;
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        not(all(feature = "nightly-features", target_feature = "avx512f"))
    ))]
    type Packing = todo!();
    // type Packing = crate::PackedMontyField31AVX2;
    #[cfg(all(
        feature = "nightly-features",
        target_arch = "x86_64",
        target_feature = "avx512f"
    ))]
    type Packing = todo!();
    // type Packing = crate::PackedMontyField31AVX512;
    #[cfg(not(any(
        all(target_arch = "aarch64", target_feature = "neon"),
        all(
            target_arch = "x86_64",
            target_feature = "avx2",
            not(all(feature = "nightly-features", target_feature = "avx512f"))
        ),
        all(
            feature = "nightly-features",
            target_arch = "x86_64",
            target_feature = "avx512f"
        ),
    )))]
    type Packing = Self;

    #[inline]
    fn mul_2exp_u64(&self, exp: u64) -> Self {
        let product = (self.value as u64) << exp;
        let value = (product % (MP::PRIME as u64)) as u32;
        Self::new_monty(value)
    }

    #[inline]
    fn exp_u64_generic<AF: AbstractField<F = Self>>(val: AF, power: u64) -> AF {
        MP::exp_u64_generic(val, power)
    }

    fn try_inverse(&self) -> Option<Self> {
        MP::try_inverse(*self)
    }

    #[inline]
    fn halve(&self) -> Self {
        Self::new_monty(halve_u32::<MP>(self.value))
    }

    #[inline]
    fn order() -> BigUint {
        MP::PRIME.into()
    }
}

impl<MP: MontyParameters> PrimeField for MontyField31<MP> {
    fn as_canonical_biguint(&self) -> BigUint {
        <Self as PrimeField32>::as_canonical_u32(self).into()
    }
}

impl<MP: MontyParameters> PrimeField64 for MontyField31<MP> {
    const ORDER_U64: u64 = MP::PRIME as u64;

    #[inline]
    fn as_canonical_u64(&self) -> u64 {
        u64::from(self.as_canonical_u32())
    }
}

impl<MP: MontyParameters> PrimeField32 for MontyField31<MP> {
    const ORDER_U32: u32 = MP::PRIME;

    #[inline]
    fn as_canonical_u32(&self) -> u32 {
        from_monty::<MP>(self.value)
    }
}

impl<MP: MontyParameters> TwoAdicField for MontyField31<MP> {
    const TWO_ADICITY: usize = MP::TWO_ADICITY;
    fn two_adic_generator(bits: usize) -> Self {
        assert!(bits <= Self::TWO_ADICITY + 1);

        // MP::GENERATORS is a list of elements with MP::GENERATORS[i] having order 2^i and MP::GENERATORS[i - 1] = MP::GENERATORS[i]^2
        Self::from_canonical_u32(MP::GENERATORS[bits])
    }
}

impl<MP: MontyParameters> Add for MontyField31<MP> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        let mut sum = self.value + rhs.value;
        let (corr_sum, over) = sum.overflowing_sub(MP::PRIME);
        if !over {
            sum = corr_sum;
        }
        Self::new_monty(sum)
    }
}

impl<MP: MontyParameters> AddAssign for MontyField31<MP> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<MP: MontyParameters> Sum for MontyField31<MP> {
    #[inline]
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        // This is faster than iter.reduce(|x, y| x + y).unwrap_or(Self::zero()) for iterators of length > 2.
        // There might be a faster reduction method possible for lengths <= 16 which avoids %.

        // This sum will not overflow so long as iter.len() < 2^33.
        let sum = iter.map(|x| x.value as u64).sum::<u64>();
        Self::new_monty((sum % MP::PRIME as u64) as u32)
    }
}

impl<MP: MontyParameters> Sub for MontyField31<MP> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        let (mut diff, over) = self.value.overflowing_sub(rhs.value);
        let corr = if over { MP::PRIME } else { 0 };
        diff = diff.wrapping_add(corr);
        Self::new_monty(diff)
    }
}

impl<MP: MontyParameters> SubAssign for MontyField31<MP> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<MP: MontyParameters> Neg for MontyField31<MP> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        Self::zero() - self
    }
}

impl<MP: MontyParameters> Mul for MontyField31<MP> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let long_prod = self.value as u64 * rhs.value as u64;
        Self::new_monty(monty_reduce::<MP>(long_prod))
    }
}

impl<MP: MontyParameters> MulAssign for MontyField31<MP> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<MP: MontyParameters> Product for MontyField31<MP> {
    #[inline]
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|x, y| x * y).unwrap_or(Self::one())
    }
}

impl<MP: MontyParameters> Div for MontyField31<MP> {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    #[inline]
    fn div(self, rhs: Self) -> Self {
        self * rhs.inverse()
    }
}

/// Given an element x from a 31 bit field F_P compute x/2.
#[inline]
const fn halve_u32<MP: MontyParameters>(input: u32) -> u32 {
    let shift = (MP::PRIME + 1) >> 1;
    let shr = input >> 1;
    let lo_bit = input & 1;
    let shr_corr = shr + shift;
    if lo_bit == 0 {
        shr
    } else {
        shr_corr
    }
}

#[inline]
pub const fn to_monty<MP: MontyParameters>(x: u32) -> u32 {
    (((x as u64) << MP::MONTY_BITS) % MP::PRIME as u64) as u32
}

#[inline]
const fn to_monty_64<MP: MontyParameters>(x: u64) -> u32 {
    (((x as u128) << MP::MONTY_BITS) % MP::PRIME as u128) as u32
}

#[inline]
#[must_use]
const fn from_monty<MP: MontyParameters>(x: u32) -> u32 {
    monty_reduce::<MP>(x as u64)
}

/// Montgomery reduction of a value in `0..P << MONTY_BITS`.
#[inline]
#[must_use]
pub const fn monty_reduce<MP: MontyParameters>(x: u64) -> u32 {
    let t = x.wrapping_mul(MP::MONTY_MU as u64) & (MP::MONTY_MASK as u64);
    let u = t * (MP::PRIME as u64);

    let (x_sub_u, over) = x.overflowing_sub(u);
    let x_sub_u_hi = (x_sub_u >> MP::MONTY_BITS) as u32;
    let corr = if over { MP::PRIME } else { 0 };
    x_sub_u_hi.wrapping_add(corr)
}

/// Convert a constant u32 array into a constant field array saved in monty form.
#[inline]
#[must_use]
pub const fn to_monty_array<const N: usize, MP: MontyParameters>(
    input: [u32; N],
) -> [MontyField31<MP>; N] {
    let mut output = [MontyField31::new_monty(0); N];
    let mut i = 0;
    loop {
        if i == N {
            break;
        }
        output[i].value = to_monty::<MP>(input[i]);
        i += 1;
    }
    output
}
