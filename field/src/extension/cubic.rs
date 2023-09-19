use core::fmt::{self, Debug, Display, Formatter};
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use rand::distributions::Standard;
use rand::prelude::Distribution;

use crate::extension::BinomiallyExtendable;
use crate::field::Field;
use crate::{AbstractExtensionField, AbstractField};

#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct CubicBef<F: BinomiallyExtendable<3>>(pub [F; 3]);

impl<F: BinomiallyExtendable<3>> Default for CubicBef<F> {
    fn default() -> Self {
        Self::ZERO
    }
}

impl<F: BinomiallyExtendable<3>> From<F> for CubicBef<F> {
    fn from(x: F) -> Self {
        Self([x, F::ZERO, F::ZERO])
    }
}

impl<F: BinomiallyExtendable<3>> AbstractField for CubicBef<F> {
    const ZERO: Self = Self([F::ZERO; 3]);
    const ONE: Self = Self([F::ONE, F::ZERO, F::ZERO]);
    const TWO: Self = Self([F::TWO, F::ZERO, F::ZERO]);
    const NEG_ONE: Self = Self([F::NEG_ONE, F::ZERO, F::ZERO]);

    fn from_bool(b: bool) -> Self {
        F::from_bool(b).into()
    }

    fn from_canonical_u8(n: u8) -> Self {
        F::from_canonical_u8(n).into()
    }

    fn from_canonical_u16(n: u16) -> Self {
        F::from_canonical_u16(n).into()
    }

    fn from_canonical_u32(n: u32) -> Self {
        F::from_canonical_u32(n).into()
    }

    /// Convert from `u64`. Undefined behavior if the input is outside the canonical range.
    fn from_canonical_u64(n: u64) -> Self {
        F::from_canonical_u64(n).into()
    }

    /// Convert from `usize`. Undefined behavior if the input is outside the canonical range.
    fn from_canonical_usize(n: usize) -> Self {
        F::from_canonical_usize(n).into()
    }

    fn from_wrapped_u32(n: u32) -> Self {
        F::from_wrapped_u32(n).into()
    }

    fn from_wrapped_u64(n: u64) -> Self {
        F::from_wrapped_u64(n).into()
    }

    fn multiplicative_group_generator() -> Self {
        Self(F::ext_multiplicative_group_generator())
    }

    #[inline(always)]
    fn square(&self) -> Self {
        let Self([a0, a1, a2]) = *self;
        let w = F::W;

        let w_a2 = w * a2;

        let c0 = a0.square() + (a1 * w_a2).double();
        let c1 = w_a2 * a2 + (a0 * a1).double();
        let c2 = a1.square() + (a0 * a2).double();

        Self([c0, c1, c2])
    }
}

impl<F: BinomiallyExtendable<3>> Field for CubicBef<F> {
    type Packing = Self;
    // Algorithm 11.3.6.b in Handbook of Elliptic and Hyperelliptic Curve Cryptography.
    fn try_inverse(&self) -> Option<Self> {
        if self.is_zero() {
            return None;
        }
        let Self([a0, a1, a2]) = *self;
        let w = F::W;

        let a0_square = a0.square();
        let a1_square = a1.square();
        let a2_w = w * a2;
        let a0_a1 = a0 * a1;

        // scalar = (a0^3+wa1^3+w^2a2^3-3wa0a1a2)^-1
        let scalar = (a0_square * a0 + w * a1 * a1_square + a2_w.square() * a2
            - (F::ONE + F::TWO) * a2_w * a0_a1)
            .inverse();

        //scalar*[a0^2-wa1a2, wa2^2-a0a1, a1^2-a0a2]
        Some(Self([
            scalar * (a0_square - a1 * a2_w),
            scalar * (a2_w * a2 - a0_a1),
            scalar * (a1_square - a0 * a2),
        ]))
    }
}

impl<F: BinomiallyExtendable<3>> Display for CubicBef<F> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{} + {}*a + {}*a^2", self.0[0], self.0[1], self.0[2])
    }
}

impl<F: BinomiallyExtendable<3>> Debug for CubicBef<F> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(self, f)
    }
}

impl<F: BinomiallyExtendable<3>> Neg for CubicBef<F> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        Self([-self.0[0], -self.0[1], -self.0[2]])
    }
}

impl<F: BinomiallyExtendable<3>> Add for CubicBef<F> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self([
            self.0[0] + rhs.0[0],
            self.0[1] + rhs.0[1],
            self.0[2] + rhs.0[2],
        ])
    }
}

impl<F: BinomiallyExtendable<3>> Add<F> for CubicBef<F> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: F) -> Self {
        Self([self.0[0] + rhs, self.0[1], self.0[2]])
    }
}

impl<F: BinomiallyExtendable<3>> AddAssign for CubicBef<F> {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<F: BinomiallyExtendable<3>> AddAssign<F> for CubicBef<F> {
    fn add_assign(&mut self, rhs: F) {
        *self = *self + rhs;
    }
}

impl<F: BinomiallyExtendable<3>> Sum for CubicBef<F> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |acc, x| acc + x)
    }
}

impl<F: BinomiallyExtendable<3>> Sub for CubicBef<F> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self([
            self.0[0] - rhs.0[0],
            self.0[1] - rhs.0[1],
            self.0[2] - rhs.0[2],
        ])
    }
}

impl<F: BinomiallyExtendable<3>> Sub<F> for CubicBef<F> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: F) -> Self {
        Self([self.0[0] - rhs, self.0[1], self.0[2]])
    }
}

impl<F: BinomiallyExtendable<3>> SubAssign for CubicBef<F> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<F: BinomiallyExtendable<3>> SubAssign<F> for CubicBef<F> {
    #[inline]
    fn sub_assign(&mut self, rhs: F) {
        *self = *self - rhs;
    }
}

impl<F: BinomiallyExtendable<3>> Mul for CubicBef<F> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let Self([a0, a1, a2]) = self;
        let Self([b0, b1, b2]) = rhs;

        let a0_b0 = a0 * b0;
        let a1_b1 = a1 * b1;
        let a2_b2 = a2 * b2;

        let w = F::W;

        let c0 = a0_b0 + ((a1 + a2) * (b1 + b2) - a1_b1 - a2_b2) * w;
        let c1 = (a0 + a1) * (b0 + b1) - a0_b0 - a1_b1 + a2_b2 * w;
        let c2 = (a0 + a2) * (b0 + b2) - a0_b0 - a2_b2 + a1_b1;

        Self([c0, c1, c2])
    }
}

impl<F: BinomiallyExtendable<3>> Mul<F> for CubicBef<F> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: F) -> Self {
        Self([self.0[0] * rhs, self.0[1] * rhs, self.0[2] * rhs])
    }
}

impl<F: BinomiallyExtendable<3>> Product for CubicBef<F> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ONE, |acc, x| acc * x)
    }
}

impl<F: BinomiallyExtendable<3>> Div for CubicBef<F> {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: Self) -> Self::Output {
        self * rhs.inverse()
    }
}

impl<F: BinomiallyExtendable<3>> DivAssign for CubicBef<F> {
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl<F: BinomiallyExtendable<3>> MulAssign for CubicBef<F> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<F: BinomiallyExtendable<3>> MulAssign<F> for CubicBef<F> {
    fn mul_assign(&mut self, rhs: F) {
        *self = *self * rhs;
    }
}
impl<F: BinomiallyExtendable<3>> AbstractExtensionField<F> for CubicBef<F> {
    const D: usize = F::D;

    fn from_base(b: F) -> Self {
        Self([b, F::ZERO, F::ZERO])
    }

    fn from_base_slice(bs: &[F]) -> Self {
        assert_eq!(bs.len(), 3);
        Self([bs[0], bs[1], bs[2]])
    }

    fn as_base_slice(&self) -> &[F] {
        self.0.as_ref()
    }
}

impl<F: BinomiallyExtendable<3>> Distribution<CubicBef<F>> for Standard
where
    Standard: Distribution<F>,
{
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> CubicBef<F> {
        CubicBef::<F>::from_base_slice(&[
            Standard.sample(rng),
            Standard.sample(rng),
            Standard.sample(rng),
        ])
    }
}
