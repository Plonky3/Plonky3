use core::fmt::{self, Debug, Display, Formatter};
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

use p3_field::{AbstractField, Field, PrimeField, PrimeField32, PrimeField64, TwoAdicField};
use rand::distributions::{Distribution, Standard};
use rand::Rng;

const P: u32 = 0x78000001;
const MONTY_BITS: u32 = 31;
const MONTY_MASK: u32 = (1 << MONTY_BITS) - 1;
const MONTY_MU: u32 = 0x8000001;

/// The prime field `2^31 - 2^27 + 1`, a.k.a. the Baby Bear field.
#[derive(Copy, Clone, Default, Eq, Hash, PartialEq)]
#[repr(transparent)] // `PackedBabyBearNeon` relies on this!
pub struct BabyBear {
    value: u32,
}

impl BabyBear {
    /// create a new `BabyBear` from a canonical `u32`.
    pub(crate) const fn new(n: u32) -> Self {
        Self { value: to_monty(n) }
    }
}

impl Ord for BabyBear {
    #[inline]
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.as_canonical_u32().cmp(&other.as_canonical_u32())
    }
}

impl PartialOrd for BabyBear {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Display for BabyBear {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(&self.as_canonical_u32(), f)
    }
}

impl Debug for BabyBear {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Debug::fmt(&self.as_canonical_u32(), f)
    }
}

impl Distribution<BabyBear> for Standard {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> BabyBear {
        loop {
            let next_u31 = rng.next_u32() & 0x7ffffff;
            let is_canonical = next_u31 < P;
            if is_canonical {
                return BabyBear { value: next_u31 };
            }
        }
    }
}

impl AbstractField for BabyBear {
    const ZERO: Self = Self { value: 0 };
    const ONE: Self = Self { value: 0x7ffffff };
    const TWO: Self = Self { value: 0xffffffe };
    const NEG_ONE: Self = Self { value: 0x70000002 };

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
        debug_assert!(n < P);
        Self::from_wrapped_u32(n)
    }

    #[inline]
    fn from_canonical_u64(n: u64) -> Self {
        debug_assert!(n < P as u64);
        Self::from_canonical_u32(n as u32)
    }

    #[inline]
    fn from_canonical_usize(n: usize) -> Self {
        debug_assert!(n < P as usize);
        Self::from_canonical_u32(n as u32)
    }

    #[inline]
    fn from_wrapped_u32(n: u32) -> Self {
        Self { value: to_monty(n) }
    }

    #[inline]
    fn from_wrapped_u64(n: u64) -> Self {
        Self {
            value: to_monty_64(n),
        }
    }

    #[inline]
    fn multiplicative_group_generator() -> Self {
        Self::from_canonical_u32(0x1f)
    }
}

impl Field for BabyBear {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    type Packing = crate::PackedBabyBearNeon;
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    type Packing = Self;

    #[inline]
    fn mul_2exp_u64(&self, exp: u64) -> Self {
        let product = (self.value as u64) << exp;
        let value = (product % (P as u64)) as u32;
        Self { value }
    }

    fn try_inverse(&self) -> Option<Self> {
        if self.is_zero() {
            return None;
        }

        // From Fermat's little theorem, in a prime field `F_p`, the inverse of `a` is `a^(p-2)`.
        // Here p-2 = 2013265919.

        let x1 = *self;
        let x256 = x1.exp_power_of_2(8);
        let x257 = x256 * x1;
        let x65792 = x257.exp_power_of_2(8);
        let x65793 = x65792 * x1;
        let x526344 = x65793.exp_power_of_2(3);
        let x16843008 = x526344.exp_power_of_2(5);
        let x16843009 = x16843008 * x1;
        let x33686018 = x16843009.square();
        let x50529027 = x33686018 * x16843009;
        let x101058054 = x50529027.square();
        let x117901063 = x101058054 * x16843009;
        let x118427407 = x117901063 * x526344;
        let x1894838512 = x118427407.exp_power_of_2(4);
        let x2013265919 = x1894838512 * x118427407;

        Some(x2013265919)
    }
}

impl PrimeField for BabyBear {}

impl PrimeField64 for BabyBear {
    const ORDER_U64: u64 = <Self as PrimeField32>::ORDER_U32 as u64;

    #[inline]
    fn as_canonical_u64(&self) -> u64 {
        u64::from(self.as_canonical_u32())
    }

    #[inline]
    fn linear_combination_u64<const N: usize>(u: [u64; N], v: &[Self; N]) -> Self {
        // In order not to overflow a u64, we must have sum(u) <= 2^32.
        debug_assert!(u.iter().sum::<u64>() <= (1u64 << 32));

        let mut dot = u[0] * v[0].value as u64;
        for i in 1..N {
            dot += u[i] * v[i].value as u64;
        }
        Self {
            value: (dot % (P as u64)) as u32,
        }
    }
}

impl PrimeField32 for BabyBear {
    const ORDER_U32: u32 = P;

    #[inline]
    fn as_canonical_u32(&self) -> u32 {
        from_monty(self.value)
    }
}

impl TwoAdicField for BabyBear {
    const TWO_ADICITY: usize = 27;

    #[inline]
    fn power_of_two_generator() -> Self {
        Self::from_canonical_u32(0x1a427a41)
    }
}

impl Add for BabyBear {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        let mut sum = self.value + rhs.value;
        let (corr_sum, over) = sum.overflowing_sub(P);
        if !over {
            sum = corr_sum;
        }
        Self { value: sum }
    }
}

impl AddAssign for BabyBear {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Sum for BabyBear {
    #[inline]
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|x, y| x + y).unwrap_or(Self::ZERO)
    }
}

impl Sub for BabyBear {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        let (mut diff, over) = self.value.overflowing_sub(rhs.value);
        let corr = if over { P } else { 0 };
        diff = diff.wrapping_add(corr);
        BabyBear { value: diff }
    }
}

impl SubAssign for BabyBear {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Neg for BabyBear {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        Self::ZERO - self
    }
}

impl Mul for BabyBear {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let long_prod = self.value as u64 * rhs.value as u64;
        Self {
            value: monty_reduce(long_prod),
        }
    }
}

impl MulAssign for BabyBear {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Product for BabyBear {
    #[inline]
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|x, y| x * y).unwrap_or(Self::ONE)
    }
}

impl Div for BabyBear {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    #[inline]
    fn div(self, rhs: Self) -> Self {
        self * rhs.inverse()
    }
}

#[inline]
#[must_use]
const fn to_monty(x: u32) -> u32 {
    (((x as u64) << 31) % P as u64) as u32
}

#[inline]
#[must_use]
fn to_monty_64(x: u64) -> u32 {
    (((x as u128) << 31) % P as u128) as u32
}

#[inline]
#[must_use]
fn from_monty(x: u32) -> u32 {
    monty_reduce(x as u64)
}

/// Montgomery reduction of a value in `0..P << MONTY_BITS`.
#[inline]
#[must_use]
fn monty_reduce(x: u64) -> u32 {
    let t = x.wrapping_mul(MONTY_MU as u64) & (MONTY_MASK as u64);
    let u = t * (P as u64);

    let (x_sub_u, over) = x.overflowing_sub(u);
    let x_sub_u_hi = (x_sub_u >> 31) as u32;
    let corr = if over { P } else { 0 };
    x_sub_u_hi.wrapping_add(corr)
}

#[cfg(test)]
mod tests {
    use p3_field::PrimeField64;
    use p3_field_testing::{test_field, test_two_adic_field};

    use super::*;

    type F = BabyBear;

    #[test]
    fn test_baby_bear() {
        let f = F::from_canonical_u32(100);
        assert_eq!(f.as_canonical_u64(), 100);

        let f = F::from_canonical_u32(0);
        assert!(f.is_zero());

        let f = F::from_wrapped_u32(F::ORDER_U32);
        assert!(f.is_zero());

        let f_1 = F::ONE;
        let f_1_copy = F::from_canonical_u32(1);

        let expected_result = F::ZERO;
        assert_eq!(f_1 - f_1_copy, expected_result);

        let expected_result = F::TWO;
        assert_eq!(f_1 + f_1_copy, expected_result);

        let f_2 = F::from_canonical_u32(2);
        let expected_result = F::from_canonical_u32(3);
        assert_eq!(f_1 + f_1_copy * f_2, expected_result);

        let expected_result = F::from_canonical_u32(5);
        assert_eq!(f_1 + f_2 * f_2, expected_result);

        let f_p_minus_1 = F::from_canonical_u32(F::ORDER_U32 - 1);
        let expected_result = F::ZERO;
        assert_eq!(f_1 + f_p_minus_1, expected_result);

        let f_p_minus_2 = F::from_canonical_u32(F::ORDER_U32 - 2);
        let expected_result = F::from_canonical_u32(F::ORDER_U32 - 3);
        assert_eq!(f_p_minus_1 + f_p_minus_2, expected_result);

        let expected_result = F::from_canonical_u32(1);
        assert_eq!(f_p_minus_1 - f_p_minus_2, expected_result);

        let expected_result = f_p_minus_1;
        assert_eq!(f_p_minus_2 - f_p_minus_1, expected_result);

        let expected_result = f_p_minus_2;
        assert_eq!(f_p_minus_1 - f_1, expected_result);

        let m1 = F::from_canonical_u32(0x34167c58);
        let m2 = F::from_canonical_u32(0x61f3207b);
        let expected_prod = F::from_canonical_u32(0x1b5c8046);
        assert_eq!(m1 * m2, expected_prod);
    }

    test_field!(crate::BabyBear);
    test_two_adic_field!(crate::BabyBear);
}
