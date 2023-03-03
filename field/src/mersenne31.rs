use crate::field::{Field, PrimeField, SemiSmoothField};
use alloc::vec;
use alloc::vec::Vec;
use core::fmt;
use core::fmt::{Display, Formatter};
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, BitXorAssign, Mul, MulAssign};
use rand::distributions::{Distribution, Standard};
use rand::Rng;

/// The prime field `F_p` where `p = 2^31 - 1`.
#[derive(Copy, Clone, Eq, PartialEq, Debug, Default)]
pub struct Mersenne31 {
    /// Not necessarily canonical, but must fit in 31 bits.
    value: u32,
}

impl Mersenne31 {
    pub const ORDER: u32 = (1 << 31) - 1;
    /// Two's complement of `ORDER`, i.e. `2^32 - ORDER`.
    pub const NEG_ORDER: u32 = Self::ORDER.wrapping_neg();
}

impl Field for Mersenne31 {
    // TODO: Add cfg-guarded Packing for AVX2, NEON, etc.
    type Packing = Self;

    const ZERO: Self = Self { value: 0 };
    const ONE: Self = Self { value: 1 };
    const TWO: Self = Self { value: 2 };
}

impl PrimeField for Mersenne31 {
    const NEG_ONE: Self = Self {
        value: Self::ORDER - 1,
    };
}

impl SemiSmoothField for Mersenne31 {
    fn semi_smooth_factors() -> Vec<u32> {
        vec![2, 3, 3, 7, 11, 31, 151, 331]
    }
}

impl Display for Mersenne31 {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(&self.value, f)
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
        sum.bitxor_assign((msb != 0) as u32);
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
        iter.fold(Self::ZERO, |acc, x| acc + x)
    }
}

impl Mul<Self> for Mersenne31 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        let prod = (self.value as u64) * (rhs.value as u64);
        let prod_lo = prod as u32;
        let prod_hi = (prod >> 32) as u32;

        // Because each value was 31 bits, prod_hi is at most 30 bits.
        // So we can apply its weight of 2 (since 2^32 = 2 mod p) without overflow.
        let prod_hi_weighted = prod_hi << 1;

        let (sum, over) = prod_lo.overflowing_add(prod_hi_weighted);
        let (sum, _) = sum.overflowing_add((over as u32) * Self::NEG_ORDER);
        // TODO: Clear most significant bit.
        Self { value: sum }
    }
}

impl MulAssign<Self> for Mersenne31 {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Distribution<Mersenne31> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Mersenne31 {
        let value = (rng.next_u64() % Mersenne31::ORDER as u64) as u32;
        Mersenne31 { value }
    }
}

impl Product for Mersenne31 {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ONE, |acc, x| acc * x)
    }
}
