//! The prime field `F_p` where `p = 2^31 - 1`.

#![no_std]

use core::fmt;
use core::fmt::{Debug, Display, Formatter};
use core::hash::{Hash, Hasher};
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, BitXorAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};
use p3_field::field::Field;
use rand::distributions::{Distribution, Standard};
use rand::Rng;

/// The prime field `F_p` where `p = 2^31 - 1`.
#[derive(Copy, Clone, Default)]
pub struct Mersenne31 {
    /// Not necessarily canonical, but must fit in 31 bits.
    value: u32,
}

impl Mersenne31 {
    pub const ORDER: u32 = (1 << 31) - 1;

    fn new(value: u32) -> Self {
        Self {
            value: value % Self::ORDER,
        }
    }

    fn as_canonical_u32(&self) -> u32 {
        // Since our invariant guarantees that `value` fits in 31 bits, there is only one possible
        // `value` that is not canonical, namely 2^31 - 1 = p = 0.
        if self.value == Self::ORDER {
            0
        } else {
            self.value
        }
    }
}

impl PartialEq for Mersenne31 {
    fn eq(&self, other: &Self) -> bool {
        self.as_canonical_u32() == other.as_canonical_u32()
    }
}

impl Ord for Mersenne31 {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.as_canonical_u32().cmp(&other.as_canonical_u32())
    }
}

impl PartialOrd for Mersenne31 {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.as_canonical_u32().cmp(&other.as_canonical_u32()))
    }
}

impl Eq for Mersenne31 {}

impl From<u32> for Mersenne31 {
    fn from(value: u32) -> Self {
        Self::new(value)
    }
}

impl Hash for Mersenne31 {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u32(self.as_canonical_u32())
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
            let is_canonical = next_u31 != Mersenne31::ORDER;
            if is_canonical {
                return Mersenne31 { value: next_u31 };
            }
        }
    }
}

impl Field for Mersenne31 {
    // TODO: Add cfg-guarded Packing for AVX2, NEON, etc.
    type Packing = Self;

    type DistinguishedSubfield = Self;
    const EXT_DEGREE: usize = 1;

    fn lift(x: Self::DistinguishedSubfield) -> Self {
        x
    }

    fn try_lower(&self) -> Option<Self::DistinguishedSubfield> {
        Some(*self)
    }

    const ZERO: Self = Self { value: 0 };
    const ONE: Self = Self { value: 1 };
    const TWO: Self = Self { value: 2 };
    const NEG_ONE: Self = Self {
        value: Self::ORDER - 1,
    };

    const TWO_ADICITY: usize = 1;

    fn is_zero(&self) -> bool {
        self.value == 0 || self.value == Self::ORDER
    }

    fn mul_2exp_u64(&self, exp: u64) -> Self {
        // In a Mersenne field, multiplication by 2^k is just a left rotation by k bits.
        let exp = (exp % 31) as u8;
        let left = (self.value << exp) & ((1 << 31) - 1);
        let right = self.value >> (31 - exp);
        let rotated = left | right;
        Self { value: rotated }
    }

    fn div_2exp_u64(&self, exp: u64) -> Self {
        // In a Mersenne field, division by 2^k is just a right rotation by k bits.
        let exp = (exp % 31) as u8;
        let left = self.value >> exp;
        let right = (self.value << (31 - exp)) & ((1 << 31) - 1);
        let rotated = left | right;
        Self { value: rotated }
    }

    fn try_inverse(&self) -> Option<Self> {
        todo!()
    }
}

impl Add<Self> for Mersenne31 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        let mut sum = self.value + rhs.value;
        // If sum's most significant bit is set, we clear it and add 1, since 2^31 = 1 mod p.
        // This addition of 1 cannot overflow 2^31, since sum has a max of
        // 2 * (2^31 - 1) = 2^32 - 2.
        let msb = sum & (1 << 31);
        sum.bitxor_assign(msb);
        sum += (msb != 0) as u32;
        Self { value: sum }
    }
}

impl AddAssign<Self> for Mersenne31 {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Sum for Mersenne31 {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|x, y| x + y).unwrap_or(Self::ZERO)
    }
}

impl Sub<Self> for Mersenne31 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        // TODO: Very naive for now.
        self + (-rhs)
    }
}

impl SubAssign<Self> for Mersenne31 {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Neg for Mersenne31 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self {
            // Can't underflow, since self.value is 31-bits and thus can't exceed ORDER.
            value: Self::ORDER - self.value,
        }
    }
}

impl Mul<Self> for Mersenne31 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        let prod = (self.value as u64) * (rhs.value as u64);
        let prod_lo = (prod as u32) & ((1 << 31) - 1);
        let prod_hi = (prod >> 31) as u32;
        Self { value: prod_lo } + Self { value: prod_hi }
    }
}

impl MulAssign<Self> for Mersenne31 {
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
    use crate::Mersenne31;
    use p3_field::field::Field;

    type F = Mersenne31;

    #[test]
    fn add() {
        assert_eq!(F::ONE + F::ONE, F::TWO);
        assert_eq!(F::NEG_ONE + F::ONE, F::ZERO);
        assert_eq!(F::NEG_ONE + F::TWO, F::ONE);
        assert_eq!(
            F::NEG_ONE + F::NEG_ONE,
            F {
                value: F::ORDER - 2
            }
        );
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
        assert_eq!(F { value: 5 }.mul_2exp_u64(2), F { value: 20 });
    }

    #[test]
    fn div_2exp_u64() {
        // 1 / 2^0 = 1.
        assert_eq!(F::ONE.div_2exp_u64(0), F::ONE);
        // 2 / 2^0 = 2.
        assert_eq!(F::TWO.div_2exp_u64(0), F::TWO);
        // 32 / 2^5 = 1.
        assert_eq!(F { value: 32 }.div_2exp_u64(5), F { value: 1 });
    }
}
