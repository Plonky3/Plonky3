use core::fmt::{self, Debug, Display, Formatter};
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use rand::distributions::Standard;
use rand::prelude::Distribution;

use crate::extension::BinomiallyExtendable;
use crate::field::Field;
use crate::{AbstractExtensionField, AbstractField};

#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct QuadraticBEF<F: BinomiallyExtendable<2>>(pub [F; 2]);

impl<F: BinomiallyExtendable<2>> Default for QuadraticBEF<F> {
    fn default() -> Self {
        Self::ZERO
    }
}

impl<F: BinomiallyExtendable<2>> From<F> for QuadraticBEF<F> {
    fn from(x: F) -> Self {
        Self([x, F::ZERO])
    }
}

impl<F: BinomiallyExtendable<2>> AbstractField for QuadraticBEF<F> {
    const ZERO: Self = Self([F::ZERO; 2]);
    const ONE: Self = Self([F::ONE, F::ZERO]);
    const TWO: Self = Self([F::TWO, F::ZERO]);
    const NEG_ONE: Self = Self([F::NEG_ONE, F::ZERO]);

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
        // Specialising mul reduces the computation of c1 from 2 muls
        // and one add to one mul and a shift

        let Self([a0, a1]) = *self;

        let c0 = a0.square() + F::W * a1.square();
        let c1 = a0 * a1.double();

        Self([c0, c1])
    }
}

impl<F: BinomiallyExtendable<2>> Field for QuadraticBEF<F> {
    type Packing = Self;
    // Algorithm 11.3.4 in Handbook of Elliptic and Hyperelliptic Curve Cryptography.
    fn try_inverse(&self) -> Option<Self> {
        if self.is_zero() {
            return None;
        }
        let Self([a0, a1]) = *self;
        let base = Self([a0, -a1]);
        let scalar = (a0.square() - F::W * a1.square()).inverse();
        Some(base * scalar)
    }
}

impl<F: BinomiallyExtendable<2>> Display for QuadraticBEF<F> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{} + {}*a", self.0[0], self.0[1])
    }
}

impl<F: BinomiallyExtendable<2>> Debug for QuadraticBEF<F> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(self, f)
    }
}

impl<F: BinomiallyExtendable<2>> Neg for QuadraticBEF<F> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        Self([-self.0[0], -self.0[1]])
    }
}

impl<F: BinomiallyExtendable<2>> Add for QuadraticBEF<F> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self([self.0[0] + rhs.0[0], self.0[1] + rhs.0[1]])
    }
}

impl<F: BinomiallyExtendable<2>> Add<F> for QuadraticBEF<F> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: F) -> Self {
        Self([self.0[0] + rhs, self.0[1]])
    }
}

impl<F: BinomiallyExtendable<2>> AddAssign for QuadraticBEF<F> {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<F: BinomiallyExtendable<2>> AddAssign<F> for QuadraticBEF<F> {
    fn add_assign(&mut self, rhs: F) {
        *self = *self + rhs;
    }
}

impl<F: BinomiallyExtendable<2>> Sum for QuadraticBEF<F> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |acc, x| acc + x)
    }
}

impl<F: BinomiallyExtendable<2>> Sub for QuadraticBEF<F> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self([self.0[0] - rhs.0[0], self.0[1] - rhs.0[1]])
    }
}

impl<F: BinomiallyExtendable<2>> Sub<F> for QuadraticBEF<F> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: F) -> Self {
        Self([self.0[0] - rhs, self.0[1]])
    }
}

impl<F: BinomiallyExtendable<2>> SubAssign for QuadraticBEF<F> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<F: BinomiallyExtendable<2>> SubAssign<F> for QuadraticBEF<F> {
    #[inline]
    fn sub_assign(&mut self, rhs: F) {
        *self = *self - rhs;
    }
}

impl<F: BinomiallyExtendable<2>> Mul for QuadraticBEF<F> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let Self([a0, a1]) = self;
        let Self([b0, b1]) = rhs;

        let c0 = a0 * b0 + F::W * a1 * b1;
        let c1 = a0 * b1 + a1 * b0;

        Self([c0, c1])
    }
}

impl<F: BinomiallyExtendable<2>> Mul<F> for QuadraticBEF<F> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: F) -> Self {
        Self([self.0[0] * rhs, self.0[1] * rhs])
    }
}

impl<F: BinomiallyExtendable<2>> Product for QuadraticBEF<F> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ONE, |acc, x| acc * x)
    }
}

impl<F: BinomiallyExtendable<2>> Div for QuadraticBEF<F> {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: Self) -> Self::Output {
        self * rhs.inverse()
    }
}

impl<F: BinomiallyExtendable<2>> DivAssign for QuadraticBEF<F> {
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl<F: BinomiallyExtendable<2>> MulAssign for QuadraticBEF<F> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<F: BinomiallyExtendable<2>> MulAssign<F> for QuadraticBEF<F> {
    fn mul_assign(&mut self, rhs: F) {
        *self = *self * rhs;
    }
}

impl<F: BinomiallyExtendable<2>> AbstractExtensionField<F> for QuadraticBEF<F> {
    const D: usize = F::D;

    fn from_base(b: F) -> Self {
        Self([b, F::ZERO])
    }

    fn from_base_slice(bs: &[F]) -> Self {
        assert_eq!(bs.len(), 2);
        Self([bs[0], bs[1]])
    }

    fn as_base_slice(&self) -> &[F] {
        self.0.as_ref()
    }
}

impl<F: BinomiallyExtendable<2>> Distribution<QuadraticBEF<F>> for Standard
where
    Standard: Distribution<F>,
{
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> QuadraticBEF<F> {
        QuadraticBEF::<F>::from_base_slice(&[Standard.sample(rng), Standard.sample(rng)])
    }
}
