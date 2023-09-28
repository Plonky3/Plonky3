//! The prime field `F_p` where `p = 2^31 - 1`.

#![no_std]

extern crate alloc;

mod complex;
mod dft;
mod extension;
mod radix_2_dit;

use core::fmt;
use core::fmt::{Debug, Display, Formatter};
use core::hash::{Hash, Hasher};
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, BitXorAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

pub use complex::*;
pub use dft::Mersenne31Dft;
pub use extension::*;
use p3_field::{
    exp_1717986917, exp_u64_by_squaring, AbstractField, Field, PrimeField, PrimeField32,
    PrimeField64,
};
pub use radix_2_dit::Mersenne31ComplexRadix2Dit;
use rand::distributions::{Distribution, Standard};
use rand::Rng;

/// The prime field `F_p` where `p = 2^31 - 1`.
#[derive(Copy, Clone, Default)]
pub struct Mersenne31 {
    /// Not necessarily canonical, but must fit in 31 bits.
    pub(crate) value: u32,
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
    type F = Self;

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

    fn from_wrapped_u64(n: u64) -> Self {
        // NB: Experiments suggest that it's faster to just use the
        // builtin remainder operator rather than split the input into
        // 32-bit chunks and reduce using 2^32 = 2 (mod Mersenne31).
        Self::from_canonical_u32((n % Self::ORDER_U64) as u32)
    }

    // Sage: GF(2^31 - 1).multiplicative_generator()
    fn generator() -> Self {
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
        let exp = exp % 31;
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

    #[inline]
    fn exp_u64_generic<AF: AbstractField<F = Self>>(val: AF, power: u64) -> AF {
        match power {
            1717986917 => exp_1717986917(val), // used in x^{1/5}
            _ => exp_u64_by_squaring(val, power),
        }
    }

    fn try_inverse(&self) -> Option<Self> {
        if self.is_zero() {
            return None;
        }

        // From Fermat's little theorem, in a prime field `F_p`, the inverse of `a` is `a^(p-2)`.
        // Here p-2 = 2147483646 = 1111111111111111111111111111101_2.
        // Uses 30 Squares + 7 Multiplications => 37 Operations total.

        let p1 = *self;
        let p101 = p1.exp_power_of_2(2) * p1;
        let p1111 = p101.square() * p101;
        let p11111111 = p1111.exp_power_of_2(4) * p1111;
        let p111111110000 = p11111111.exp_power_of_2(4);
        let p111111111111 = p111111110000 * p1111;
        let p1111111111111111 = p111111110000.exp_power_of_2(4) * p11111111;
        let p1111111111111111111111111111 = p1111111111111111.exp_power_of_2(12) * p111111111111;
        let p1111111111111111111111111111101 =
            p1111111111111111111111111111.exp_power_of_2(3) * p101;
        Some(p1111111111111111111111111111101)
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

impl PrimeField64 for Mersenne31 {
    const ORDER_U64: u64 = <Self as PrimeField32>::ORDER_U32 as u64;

    fn as_canonical_u64(&self) -> u64 {
        u64::from(self.as_canonical_u32())
    }

    fn linear_combination_u64<const N: usize>(u: [u64; N], v: &[Self; N]) -> Self {
        // In order not to overflow a u64, we must have sum(u) <= 2^32.
        debug_assert!(u.iter().sum::<u64>() <= (1u64 << 32));

        let mut dot = u[0] * v[0].value as u64;
        for i in 1..N {
            dot += u[i] * v[i].value as u64;
        }
        Self::from_wrapped_u64(dot)
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
    use p3_field_testing::test_field;

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
    fn exp_root() {
        // Confirm that (x^{1/5})^5 = x

        let m1 = F::from_canonical_u32(0x34167c58);
        let m2 = F::from_canonical_u32(0x61f3207b);

        assert_eq!(m1.exp_u64(1717986917).exp_const_u64::<5>(), m1);
        assert_eq!(m2.exp_u64(1717986917).exp_const_u64::<5>(), m2);
        assert_eq!(F::TWO.exp_u64(1717986917).exp_const_u64::<5>(), F::TWO);
    }

    test_field!(crate::Mersenne31);
}
