//! An abstraction of 31-bit fields which use a MONTY approach for addition and multiplication with a MONTY constant = 2^32.

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

use crate::{from_monty, halve_u32, monty_reduce, to_monty, to_monty_64, FieldParameters};

#[derive(Copy, Clone, Default, Eq, Hash, PartialEq)]
#[repr(transparent)] // Packed field implementations rely on this!
pub struct MontyField31<FP: FieldParameters> {
    // TODO:
    // This will eventually be pub(crate)
    // Setting to pub for now so BabyBear and KoalaBear crates can build when this is only partially implemented.

    // This is `pub(crate)` for tests and delayed reduction strategies. If you're accessing `value` outside of those, you're
    // likely doing something fishy.
    pub value: u32,
    _phantom: PhantomData<FP>,
}

impl<FP: FieldParameters> MontyField31<FP> {
    // The standard way to crate a new element.
    pub const fn new(value: u32) -> Self {
        Self {
            value: to_monty::<FP>(value),
            _phantom: PhantomData,
        }
    }

    // Create a new field element from something already in MONTY form.
    // TODO:
    // This will eventually be pub(crate)
    // Setting to pub for now so BabyBear and KoalaBear crates can build when this is only partially implemented.

    // This is `pub(crate)` for tests and delayed reduction strategies. If you're accessing `value` outside of those, you're
    // likely doing something fishy.
    pub const fn new_monty(value: u32) -> Self {
        Self {
            value,
            _phantom: PhantomData,
        }
    }
}

impl<FP: FieldParameters> Ord for MontyField31<FP> {
    #[inline]
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.as_canonical_u32().cmp(&other.as_canonical_u32())
    }
}

impl<FP: FieldParameters> PartialOrd for MontyField31<FP> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<FP: FieldParameters> Display for MontyField31<FP> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(&self.as_canonical_u32(), f)
    }
}

impl<FP: FieldParameters> Debug for MontyField31<FP> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Debug::fmt(&self.as_canonical_u32(), f)
    }
}

impl<FP: FieldParameters> Distribution<MontyField31<FP>> for Standard {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> MontyField31<FP> {
        loop {
            let next_u31 = rng.next_u32() >> 1;
            let is_canonical = next_u31 < FP::PRIME;
            if is_canonical {
                return MontyField31::new_monty(next_u31);
            }
        }
    }
}

impl<FP: FieldParameters> Serialize for MontyField31<FP> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_u32(self.as_canonical_u32())
    }
}

impl<'de, FP: FieldParameters> Deserialize<'de> for MontyField31<FP> {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let val = u32::deserialize(d)?;
        Ok(MontyField31::from_canonical_u32(val))
    }
}

impl<FP: FieldParameters> Packable for MontyField31<FP> {}

impl<FP: FieldParameters> AbstractField for MontyField31<FP> {
    type F = Self;

    fn zero() -> Self {
        Self::new_monty(FP::MONTY_ZERO)
    }
    fn one() -> Self {
        Self::new_monty(FP::MONTY_ONE)
    }
    fn two() -> Self {
        Self::new_monty(FP::MONTY_TWO)
    }
    fn neg_one() -> Self {
        Self::new_monty(FP::MONTY_NEG_ONE)
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
        debug_assert!(n < FP::PRIME);
        Self::from_wrapped_u32(n)
    }

    #[inline]
    fn from_canonical_u64(n: u64) -> Self {
        debug_assert!(n < FP::PRIME as u64);
        Self::from_canonical_u32(n as u32)
    }

    #[inline]
    fn from_canonical_usize(n: usize) -> Self {
        debug_assert!(n < FP::PRIME as usize);
        Self::from_canonical_u32(n as u32)
    }

    #[inline]
    fn from_wrapped_u32(n: u32) -> Self {
        Self::new(n)
    }

    #[inline]
    fn from_wrapped_u64(n: u64) -> Self {
        Self::new_monty(to_monty_64::<FP>(n))
    }

    #[inline]
    fn generator() -> Self {
        Self::new_monty(FP::MONTY_GEN)
    }
}

impl<FP: FieldParameters> Field for MontyField31<FP> {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    type Packing = crate::PackedMontyField31Neon<FP>;
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        not(all(feature = "nightly-features", target_feature = "avx512f"))
    ))]
    type Packing = crate::PackedMontyField31AVX2<FP>;
    #[cfg(all(
        feature = "nightly-features",
        target_arch = "x86_64",
        target_feature = "avx512f"
    ))]
    type Packing = crate::PackedMontyField31AVX512<FP>;
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
        let value = (product % (FP::PRIME as u64)) as u32;
        Self::new_monty(value)
    }

    #[inline]
    fn exp_u64_generic<AF: AbstractField<F = Self>>(val: AF, power: u64) -> AF {
        FP::exp_u64_generic(val, power)
    }

    fn try_inverse(&self) -> Option<Self> {
        FP::try_inverse(*self)
    }

    #[inline]
    fn halve(&self) -> Self {
        Self::new_monty(halve_u32::<FP>(self.value))
    }

    #[inline]
    fn order() -> BigUint {
        FP::PRIME.into()
    }
}

impl<FP: FieldParameters> PrimeField for MontyField31<FP> {
    fn as_canonical_biguint(&self) -> BigUint {
        <Self as PrimeField32>::as_canonical_u32(self).into()
    }
}

impl<FP: FieldParameters> PrimeField64 for MontyField31<FP> {
    const ORDER_U64: u64 = FP::PRIME as u64;

    #[inline]
    fn as_canonical_u64(&self) -> u64 {
        u64::from(self.as_canonical_u32())
    }
}

impl<FP: FieldParameters> PrimeField32 for MontyField31<FP> {
    const ORDER_U32: u32 = FP::PRIME;

    #[inline]
    fn as_canonical_u32(&self) -> u32 {
        from_monty::<FP>(self.value)
    }
}

impl<FP: FieldParameters> TwoAdicField for MontyField31<FP> {
    const TWO_ADICITY: usize = FP::TWO_ADICITY;
    fn two_adic_generator(bits: usize) -> Self {
        assert!(bits <= Self::TWO_ADICITY);

        Self::from_canonical_u32(FP::u32_two_adic_generator(bits))
    }
}

impl<FP: FieldParameters> Add for MontyField31<FP> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        let mut sum = self.value + rhs.value;
        let (corr_sum, over) = sum.overflowing_sub(FP::PRIME);
        if !over {
            sum = corr_sum;
        }
        Self::new_monty(sum)
    }
}

impl<FP: FieldParameters> AddAssign for MontyField31<FP> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<FP: FieldParameters> Sum for MontyField31<FP> {
    #[inline]
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        // This is faster than iter.reduce(|x, y| x + y).unwrap_or(Self::zero()) for iterators of length > 2.
        // There might be a faster reduction method possible for lengths <= 16 which avoids %.

        // This sum will not overflow so long as iter.len() < 2^33.
        let sum = iter.map(|x| x.value as u64).sum::<u64>();
        Self::new_monty((sum % FP::PRIME as u64) as u32)
    }
}

impl<FP: FieldParameters> Sub for MontyField31<FP> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        let (mut diff, over) = self.value.overflowing_sub(rhs.value);
        let corr = if over { FP::PRIME } else { 0 };
        diff = diff.wrapping_add(corr);
        Self::new_monty(diff)
    }
}

impl<FP: FieldParameters> SubAssign for MontyField31<FP> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<FP: FieldParameters> Neg for MontyField31<FP> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        Self::zero() - self
    }
}

impl<FP: FieldParameters> Mul for MontyField31<FP> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let long_prod = self.value as u64 * rhs.value as u64;
        Self::new_monty(monty_reduce::<FP>(long_prod))
    }
}

impl<FP: FieldParameters> MulAssign for MontyField31<FP> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<FP: FieldParameters> Product for MontyField31<FP> {
    #[inline]
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|x, y| x * y).unwrap_or(Self::one())
    }
}

impl<FP: FieldParameters> Div for MontyField31<FP> {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    #[inline]
    fn div(self, rhs: Self) -> Self {
        self * rhs.inverse()
    }
}

/// Convert a constant u32 array into a constant field array saved in monty form.
/// Long term this will be removed.
#[inline]
#[must_use]
pub const fn to_monty_array<const N: usize, FP: FieldParameters>(
    input: [u32; N],
) -> [MontyField31<FP>; N] {
    let mut output = [MontyField31::new_monty(0); N];
    let mut i = 0;
    loop {
        if i == N {
            break;
        }
        output[i].value = to_monty::<FP>(input[i]);
        i += 1;
    }
    output
}
