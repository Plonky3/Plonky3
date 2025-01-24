//! An abstraction of 31-bit fields which use a MONTY approach for faster multiplication.

use alloc::vec;
use alloc::vec::Vec;
use core::fmt::{self, Debug, Display, Formatter};
use core::hash::Hash;
use core::intrinsics::transmute;
use core::iter::{Product, Sum};
use core::marker::PhantomData;
use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

use num_bigint::BigUint;
use p3_field::{
    Field, FieldAlgebra, Packable, PrimeField, PrimeField32, PrimeField64, TwoAdicField,
};
use rand::distributions::{Distribution, Standard};
use rand::Rng;
use serde::de::Error;
use serde::{Deserialize, Deserializer, Serialize};

use crate::utils::{from_monty, halve_u32, monty_reduce, to_monty, to_monty_64};
use crate::{FieldParameters, MontyParameters, TwoAdicData};

#[derive(Clone, Copy, Default, Eq, Hash, PartialEq)]
#[repr(transparent)] // Packed field implementations rely on this!
pub struct MontyField31<MP: MontyParameters> {
    /// The MONTY form of the field element, saved as a positive integer less than `P`.
    ///
    /// This is `pub(crate)` for tests and delayed reduction strategies. If you're accessing `value` outside of those, you're
    /// likely doing something fishy.
    pub(crate) value: u32,
    _phantom: PhantomData<MP>,
}

impl<MP: MontyParameters> MontyField31<MP> {
    // The standard way to crate a new element.
    // Note that new converts the input into MONTY form so should be avoided in performance critical implementations.
    #[inline(always)]
    pub const fn new(value: u32) -> Self {
        Self {
            value: to_monty::<MP>(value),
            _phantom: PhantomData,
        }
    }

    // Create a new field element from something already in MONTY form.
    // This is `pub(crate)` for tests and delayed reduction strategies. If you're using it outside of those, you're
    // likely doing something fishy.
    #[inline(always)]
    pub(crate) const fn new_monty(value: u32) -> Self {
        Self {
            value,
            _phantom: PhantomData,
        }
    }

    /// Produce a u32 in range [0, P) from a field element corresponding to the true value.
    #[inline(always)]
    pub(crate) fn to_u32(elem: &Self) -> u32 {
        from_monty::<MP>(elem.value)
    }

    /// Convert a constant u32 array into a constant array of field elements.
    /// Constant version of array.map(MontyField31::new).
    #[inline]
    pub const fn new_array<const N: usize>(input: [u32; N]) -> [Self; N] {
        let mut output = [MontyField31::new_monty(0); N];
        let mut i = 0;
        loop {
            if i == N {
                break;
            }
            output[i] = MontyField31::new(input[i]);
            i += 1;
        }
        output
    }

    /// Convert a constant 2d u32 array into a constant 2d array of field elements.
    /// Constant version of array.map(MontyField31::new_array).
    #[inline]
    pub const fn new_2d_array<const N: usize, const M: usize>(
        input: [[u32; N]; M],
    ) -> [[Self; N]; M] {
        let mut output = [[MontyField31::new_monty(0); N]; M];
        let mut i = 0;
        loop {
            if i == M {
                break;
            }
            output[i] = MontyField31::new_array(input[i]);
            i += 1;
        }
        output
    }

    /// Multiply the given MontyField31 element by `2^{-n}`.
    ///
    /// This makes use of the fact that, as the monty constant is `2^32`,
    /// the monty form of `2^{-n}` is `2^{32 - n}`. Monty reduction works
    /// provided the input is `< 2^32P` so this works for `0 <= n <= 32`.
    #[inline]
    #[must_use]
    pub const fn mul_2exp_neg_n(&self, n: u32) -> Self {
        assert!(n < 33);
        let value_mul_2exp_neg_n = (self.value as u64) << (32 - n);
        MontyField31::new_monty(monty_reduce::<MP>(value_mul_2exp_neg_n))
    }
}

impl<FP: MontyParameters> Ord for MontyField31<FP> {
    #[inline]
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        MontyField31::to_u32(self).cmp(&MontyField31::to_u32(other))
    }
}

impl<FP: MontyParameters> PartialOrd for MontyField31<FP> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<FP: MontyParameters> Display for MontyField31<FP> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(&MontyField31::to_u32(self), f)
    }
}

impl<FP: MontyParameters> Debug for MontyField31<FP> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Debug::fmt(&MontyField31::to_u32(self), f)
    }
}

impl<FP: MontyParameters> Distribution<MontyField31<FP>> for Standard {
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
        // It's faster to Serialize and Deserialize in monty form.
        serializer.serialize_u32(self.value)
    }
}

impl<'de, FP: FieldParameters> Deserialize<'de> for MontyField31<FP> {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let val = u32::deserialize(d)?;
        // Ensure that `val` satisfies our invariant, namely is `< P`.
        if val < FP::PRIME {
            // It's faster to Serialize and Deserialize in monty form.
            Ok(MontyField31::new_monty(val))
        } else {
            Err(D::Error::custom("Value is out of range"))
        }
    }
}

impl<FP: FieldParameters> Packable for MontyField31<FP> {}

impl<FP: FieldParameters> FieldAlgebra for MontyField31<FP> {
    type F = Self;

    const ZERO: Self = FP::MONTY_ZERO;
    const ONE: Self = FP::MONTY_ONE;
    const TWO: Self = FP::MONTY_TWO;
    const NEG_ONE: Self = FP::MONTY_NEG_ONE;

    #[inline(always)]
    fn from_f(f: Self::F) -> Self {
        f
    }

    #[inline(always)]
    fn from_canonical_u8(n: u8) -> Self {
        Self::from_canonical_u32(n as u32)
    }

    #[inline(always)]
    fn from_canonical_u16(n: u16) -> Self {
        Self::from_canonical_u32(n as u32)
    }

    #[inline(always)]
    fn from_canonical_u32(n: u32) -> Self {
        debug_assert!(n < FP::PRIME);
        Self::from_wrapped_u32(n)
    }

    #[inline(always)]
    fn from_canonical_u64(n: u64) -> Self {
        debug_assert!(n < FP::PRIME as u64);
        Self::from_canonical_u32(n as u32)
    }

    #[inline(always)]
    fn from_canonical_usize(n: usize) -> Self {
        debug_assert!(n < FP::PRIME as usize);
        Self::from_canonical_u32(n as u32)
    }

    #[inline(always)]
    fn from_wrapped_u32(n: u32) -> Self {
        Self::new(n)
    }

    #[inline(always)]
    fn from_wrapped_u64(n: u64) -> Self {
        Self::new_monty(to_monty_64::<FP>(n))
    }

    #[inline]
    fn mul_2exp_u64(&self, exp: u64) -> Self {
        let product = (self.value as u64) << exp;
        let value = (product % (FP::PRIME as u64)) as u32;
        Self::new_monty(value)
    }

    #[inline]
    fn zero_vec(len: usize) -> Vec<Self> {
        // SAFETY: repr(transparent) ensures transmutation safety.
        unsafe { transmute(vec![0u32; len]) }
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

    const GENERATOR: Self = FP::MONTY_GEN;

    #[inline]
    fn exp_u64_generic<FA: FieldAlgebra<F = Self>>(val: FA, power: u64) -> FA {
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
        self.as_canonical_u32().into()
    }

    #[inline]
    fn to_unique_u64(&self) -> u64 {
        // The internal representation is already a unique u32 for each field element.
        // It's fine to hash things in monty form.
        self.value as u64
    }
}

impl<FP: FieldParameters> PrimeField32 for MontyField31<FP> {
    const ORDER_U32: u32 = FP::PRIME;

    #[inline]
    fn as_canonical_u32(&self) -> u32 {
        MontyField31::to_u32(self)
    }

    #[inline]
    fn to_unique_u32(&self) -> u32 {
        // The internal representation is already a unique u32 for each field element.
        // It's fine to hash things in monty form.
        self.value
    }
}

impl<FP: FieldParameters + TwoAdicData> TwoAdicField for MontyField31<FP> {
    const TWO_ADICITY: usize = FP::TWO_ADICITY;
    fn two_adic_generator(bits: usize) -> Self {
        assert!(bits <= Self::TWO_ADICITY);
        FP::TWO_ADIC_GENERATORS.as_ref()[bits]
    }
}

impl<FP: MontyParameters> Add for MontyField31<FP> {
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

impl<FP: MontyParameters> AddAssign for MontyField31<FP> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<FP: MontyParameters> Sum for MontyField31<FP> {
    #[inline]
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        // This is faster than iter.reduce(|x, y| x + y).unwrap_or(Self::ZERO) for iterators of length > 2.
        // There might be a faster reduction method possible for lengths <= 16 which avoids %.

        // This sum will not overflow so long as iter.len() < 2^33.
        let sum = iter.map(|x| x.value as u64).sum::<u64>();
        Self::new_monty((sum % FP::PRIME as u64) as u32)
    }
}

impl<FP: MontyParameters> Sub for MontyField31<FP> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        let (mut diff, over) = self.value.overflowing_sub(rhs.value);
        let corr = if over { FP::PRIME } else { 0 };
        diff = diff.wrapping_add(corr);
        Self::new_monty(diff)
    }
}

impl<FP: MontyParameters> SubAssign for MontyField31<FP> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<FP: FieldParameters> Neg for MontyField31<FP> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        Self::ZERO - self
    }
}

impl<FP: MontyParameters> Mul for MontyField31<FP> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let long_prod = self.value as u64 * rhs.value as u64;
        Self::new_monty(monty_reduce::<FP>(long_prod))
    }
}

impl<FP: MontyParameters> MulAssign for MontyField31<FP> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<FP: FieldParameters> Product for MontyField31<FP> {
    #[inline]
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|x, y| x * y).unwrap_or(Self::ONE)
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
