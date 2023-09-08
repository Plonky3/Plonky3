use core::fmt::{self, Debug, Display, Formatter};
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use rand::distributions::Standard;
use rand::prelude::Distribution;

use crate::extension::OptimallyExtendable;
use crate::field::Field;
use crate::{AbstractExtensionField, AbstractField};

#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct QuadraticOef<F: OptimallyExtendable<2>>(pub [F; 2]);

impl<F: OptimallyExtendable<2>> Default for QuadraticOef<F> {
    fn default() -> Self {
        Self::ZERO
    }
}

impl<F: OptimallyExtendable<2>> From<F> for QuadraticOef<F> {
    fn from(x: F) -> Self {
        Self([x, F::ZERO])
    }
}

impl<F: OptimallyExtendable<2>> AbstractField for QuadraticOef<F> {
    const ZERO: Self = Self([F::ZERO; 2]);
    const ONE: Self = Self([F::ONE, F::ZERO]);
    const TWO: Self = Self([F::TWO, F::ZERO]);
    const NEG_ONE: Self = Self([F::NEG_ONE, F::ZERO]);

    #[inline(always)]
    fn square(&self) -> Self {
        // Specialising mul reduces the computation of c1 from 2 muls
        // and one add to one mul and a shift

        let Self([a0, a1]) = *self;

        let c0 = a0.square() + F::W * a1.square();
        let c1 = a0 * a1.double();

        Self([c0, c1])
    }

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
}

impl<F: OptimallyExtendable<2>> Field for QuadraticOef<F> {
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

impl<F: OptimallyExtendable<2>> Display for QuadraticOef<F> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{} + {}*a", self.0[0], self.0[1])
    }
}

impl<F: OptimallyExtendable<2>> Debug for QuadraticOef<F> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(self, f)
    }
}

impl<F: OptimallyExtendable<2>> Neg for QuadraticOef<F> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        Self([-self.0[0], -self.0[1]])
    }
}

impl<F: OptimallyExtendable<2>> Add for QuadraticOef<F> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self([self.0[0] + rhs.0[0], self.0[1] + rhs.0[1]])
    }
}

impl<F: OptimallyExtendable<2>> Add<F> for QuadraticOef<F> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: F) -> Self {
        Self([self.0[0] + rhs, self.0[1]])
    }
}

impl<F: OptimallyExtendable<2>> AddAssign for QuadraticOef<F> {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<F: OptimallyExtendable<2>> AddAssign<F> for QuadraticOef<F> {
    fn add_assign(&mut self, rhs: F) {
        *self = *self + rhs;
    }
}

impl<F: OptimallyExtendable<2>> Sum for QuadraticOef<F> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |acc, x| acc + x)
    }
}

impl<F: OptimallyExtendable<2>> Sub for QuadraticOef<F> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self([self.0[0] - rhs.0[0], self.0[1] - rhs.0[1]])
    }
}

impl<F: OptimallyExtendable<2>> Sub<F> for QuadraticOef<F> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: F) -> Self {
        Self([self.0[0] - rhs, self.0[1]])
    }
}

impl<F: OptimallyExtendable<2>> SubAssign for QuadraticOef<F> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<F: OptimallyExtendable<2>> SubAssign<F> for QuadraticOef<F> {
    #[inline]
    fn sub_assign(&mut self, rhs: F) {
        *self = *self - rhs;
    }
}

impl<F: OptimallyExtendable<2>> Mul for QuadraticOef<F> {
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

impl<F: OptimallyExtendable<2>> Mul<F> for QuadraticOef<F> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: F) -> Self {
        Self([self.0[0] * rhs, self.0[1] * rhs])
    }
}

impl<F: OptimallyExtendable<2>> Product for QuadraticOef<F> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ONE, |acc, x| acc * x)
    }
}

impl<F: OptimallyExtendable<2>> Div for QuadraticOef<F> {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: Self) -> Self::Output {
        self * rhs.inverse()
    }
}

impl<F: OptimallyExtendable<2>> DivAssign for QuadraticOef<F> {
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl<F: OptimallyExtendable<2>> MulAssign for QuadraticOef<F> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<F: OptimallyExtendable<2>> MulAssign<F> for QuadraticOef<F> {
    fn mul_assign(&mut self, rhs: F) {
        *self = *self * rhs;
    }
}

impl<F: OptimallyExtendable<2>> AbstractExtensionField<F> for QuadraticOef<F> {
    const D: usize = F::D;

    fn from_base(b: F) -> Self {
        Self([b, F::ZERO])
    }

    fn from_base_slice(bs: &[F]) -> Self {
        Self([bs[0], bs[1]])
    }

    fn as_base_slice(&self) -> &[F] {
        self.0.as_ref()
    }
}

impl<F: OptimallyExtendable<2>> Distribution<QuadraticOef<F>> for Standard
where
    Standard: Distribution<F>,
{
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> QuadraticOef<F> {
        QuadraticOef::<F>::from_base_slice(&[Standard.sample(rng), Standard.sample(rng)])
    }
}
