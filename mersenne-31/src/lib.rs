//! The prime field `F_p` where `p = 2^31 - 1`.

#![no_std]

mod complex;

use core::fmt;
use core::fmt::{Debug, Display, Formatter};
use core::hash::{Hash, Hasher};
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, BitXorAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

pub use complex::*;
use p3_field::{AbstractField, Field, PrimeField, PrimeField32};
use rand::distributions::{Distribution, Standard};
use rand::Rng;

/// The prime field `F_p` where `p = 2^31 - 1`.
#[derive(Copy, Clone, Default)]
pub struct Mersenne31 {
    /// Not necessarily canonical, but must fit in 31 bits.
    value: u32,
}

impl Mersenne31 {
    const fn new(value: u32) -> Self {
        debug_assert!((value >> 31) == 0);
        Self { value }
    }
}

impl PartialEq for Mersenne31 {
    fn eq(&self, other: &Self) -> bool {
        self.as_canonical_u32() == other.as_canonical_u32()
    }
}

impl Eq for Mersenne31 {}

impl Hash for Mersenne31 {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u32(self.as_canonical_u32());
    }
}

impl Ord for Mersenne31 {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.as_canonical_u32().cmp(&other.as_canonical_u32())
    }
}

impl PartialOrd for Mersenne31 {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Display for Mersenne31 {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(&self.value, f)
    }
}

impl Debug for Mersenne31 {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Debug::fmt(&self.value, f)
    }
}

impl Distribution<Mersenne31> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Mersenne31 {
        loop {
            let next_u31 = rng.next_u32() >> 1;
            let is_canonical = next_u31 != Mersenne31::ORDER_U32;
            if is_canonical {
                return Mersenne31::new(next_u31);
            }
        }
    }
}

impl AbstractField for Mersenne31 {
    const ZERO: Self = Self::new(0);
    const ONE: Self = Self::new(1);
    const TWO: Self = Self::new(2);
    const NEG_ONE: Self = Self::new(Self::ORDER_U32 - 1);

    fn from_bool(b: bool) -> Self {
        Self::new(b as u32)
    }

    fn from_canonical_u8(n: u8) -> Self {
        Self::new(u32::from(n))
    }

    fn from_canonical_u16(n: u16) -> Self {
        Self::new(u32::from(n))
    }

    fn from_canonical_u32(n: u32) -> Self {
        debug_assert!(n < Self::ORDER_U32);
        Self::new(n)
    }

    /// Convert from `u64`. Undefined behavior if the input is outside the canonical range.
    fn from_canonical_u64(n: u64) -> Self {
        Self::from_canonical_u32(
            n.try_into()
                .expect("Too large to be a canonical Mersenne31 encoding"),
        )
    }

    /// Convert from `usize`. Undefined behavior if the input is outside the canonical range.
    fn from_canonical_usize(n: usize) -> Self {
        Self::from_canonical_u32(
            n.try_into()
                .expect("Too large to be a canonical Mersenne31 encoding"),
        )
    }

    fn from_wrapped_u32(n: u32) -> Self {
        // To reduce `n` to 31 bits, we clear its MSB, then add it back in its reduced form.
        let msb = n & (1 << 31);
        let msb_reduced = msb >> 31;
        Self::new(n ^ msb) + Self::new(msb_reduced)
    }

    fn from_wrapped_u64(_n: u64) -> Self {
        todo!()
    }

    // Sage: GF(2^31 - 1).multiplicative_generator()
    fn multiplicative_group_generator() -> Self {
        Self::new(7)
    }
}

impl Field for Mersenne31 {
    // TODO: Add cfg-guarded Packing for AVX2, NEON, etc.
    type Packing = Self;

    fn is_zero(&self) -> bool {
        self.value == 0 || self.value == Self::ORDER_U32
    }

    fn mul_2exp_u64(&self, exp: u64) -> Self {
        // In a Mersenne field, multiplication by 2^k is just a left rotation by k bits.
        let exp = (exp % 31) as u8;
        let left = (self.value << exp) & ((1 << 31) - 1);
        let right = self.value >> (31 - exp);
        let rotated = left | right;
        Self::new(rotated)
    }

    fn div_2exp_u64(&self, exp: u64) -> Self {
        // In a Mersenne field, division by 2^k is just a right rotation by k bits.
        let exp = (exp % 31) as u8;
        let left = self.value >> exp;
        let right = (self.value << (31 - exp)) & ((1 << 31) - 1);
        let rotated = left | right;
        Self::new(rotated)
    }

    fn try_inverse(&self) -> Option<Self> {
        // Uses algorithm 9.4.5 in Crandall and Pomerance book
        // "Prime Numbers: A Computational Perspective" to compute the inverse.

        if self.is_zero() {
            return None;
        }

        let mut a = Self::ONE;
        let mut b = Self::ZERO;
        let mut u = self.value;
        let mut v = Self::ORDER_U32;

        loop {
            // Shift off trailing zeros
            let e = u.trailing_zeros() as u64;
            u >>= e;

            // Circular shift
            a = a.mul_2exp_u64(31 - e);

            if u == 1 {
                return Some(a);
            }

            (a, b, u, v) = (a + b, a, u + v, u);
        }
    }
}

impl PrimeField for Mersenne31 {}

impl PrimeField32 for Mersenne31 {
    const ORDER_U32: u32 = (1 << 31) - 1;

    fn as_canonical_u32(&self) -> u32 {
        // Since our invariant guarantees that `value` fits in 31 bits, there is only one possible
        // `value` that is not canonical, namely 2^31 - 1 = p = 0.
        if self.value == Self::ORDER_U32 {
            0
        } else {
            self.value
        }
    }
}

impl Add for Mersenne31 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        let mut sum = self.value + rhs.value;
        // If sum's most significant bit is set, we clear it and add 1, since 2^31 = 1 mod p.
        // This addition of 1 cannot overflow 2^31, since sum has a max of
        // 2 * (2^31 - 1) = 2^32 - 2.
        let msb = sum & (1 << 31);
        sum.bitxor_assign(msb);
        sum += u32::from(msb != 0);
        Self::new(sum)
    }
}

impl AddAssign for Mersenne31 {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Sum for Mersenne31 {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|x, y| x + y).unwrap_or(Self::ZERO)
    }
}

impl Sub for Mersenne31 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        // TODO: Very naive for now.
        self + (-rhs)
    }
}

impl SubAssign for Mersenne31 {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Neg for Mersenne31 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        // Can't underflow, since self.value is 31-bits and thus can't exceed ORDER.
        Self::new(Self::ORDER_U32 - self.value)
    }
}

impl Mul for Mersenne31 {
    type Output = Self;

    #[allow(clippy::cast_possible_truncation)]
    fn mul(self, rhs: Self) -> Self {
        let prod = u64::from(self.value) * u64::from(rhs.value);
        let prod_lo = (prod as u32) & ((1 << 31) - 1);
        let prod_hi = (prod >> 31) as u32;
        Self::new(prod_lo) + Self::new(prod_hi)
    }
}

impl MulAssign for Mersenne31 {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Product for Mersenne31 {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|x, y| x * y).unwrap_or(Self::ONE)
    }
}

impl Div for Mersenne31 {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: Self) -> Self {
        self * rhs.inverse()
    }
}

#[cfg(test)]
mod tests {
    use p3_field::{AbstractField, Field, PrimeField32};
    use p3_field_testing::test_inverse;

    use crate::Mersenne31;

    type F = Mersenne31;

    #[test]
    fn add() {
        assert_eq!(F::ONE + F::ONE, F::TWO);
        assert_eq!(F::NEG_ONE + F::ONE, F::ZERO);
        assert_eq!(F::NEG_ONE + F::TWO, F::ONE);
        assert_eq!(F::NEG_ONE + F::NEG_ONE, F::new(F::ORDER_U32 - 2));
    }

    #[test]
    fn sub() {
        assert_eq!(F::ONE - F::ONE, F::ZERO);
        assert_eq!(F::TWO - F::TWO, F::ZERO);
        assert_eq!(F::NEG_ONE - F::NEG_ONE, F::ZERO);
        assert_eq!(F::TWO - F::ONE, F::ONE);
        assert_eq!(F::NEG_ONE - F::ZERO, F::NEG_ONE);
    }

    #[test]
    fn mul_2exp_u64() {
        // 1 * 2^0 = 1.
        assert_eq!(F::ONE.mul_2exp_u64(0), F::ONE);
        // 2 * 2^30 = 2^31 = 1.
        assert_eq!(F::TWO.mul_2exp_u64(30), F::ONE);
        // 5 * 2^2 = 20.
        assert_eq!(F::new(5).mul_2exp_u64(2), F::new(20));
    }

    #[test]
    fn div_2exp_u64() {
        // 1 / 2^0 = 1.
        assert_eq!(F::ONE.div_2exp_u64(0), F::ONE);
        // 2 / 2^0 = 2.
        assert_eq!(F::TWO.div_2exp_u64(0), F::TWO);
        // 32 / 2^5 = 1.
        assert_eq!(F::new(32).div_2exp_u64(5), F::new(1));
    }

    #[test]
    fn inverse() {
        test_inverse::<Mersenne31>();
    }
}
