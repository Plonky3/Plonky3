use core::fmt::{self, Debug, Display, Formatter};
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use rand::distributions::Standard;
use rand::prelude::Distribution;

use super::{Frobenius, OptimallyExtendable};
use crate::field::Field;
use crate::{AbstractExtensionField, AbstractField};

#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct QuinticOef<F: OptimallyExtendable<5>>(pub [F; 5]);

impl<F: OptimallyExtendable<5>> Frobenius<F, 5> for QuinticOef<F> {}

impl<F: OptimallyExtendable<5>> Default for QuinticOef<F> {
    fn default() -> Self {
        Self::ZERO
    }
}

impl<F: OptimallyExtendable<5>> From<F> for QuinticOef<F> {
    fn from(x: F) -> Self {
        Self([x, F::ZERO, F::ZERO, F::ZERO, F::ZERO])
    }
}

impl<F: OptimallyExtendable<5>> AbstractField for QuinticOef<F> {
    const ZERO: Self = Self([F::ZERO; 5]);
    const ONE: Self = Self([F::ONE, F::ZERO, F::ZERO, F::ZERO, F::ZERO]);
    const TWO: Self = Self([F::TWO, F::ZERO, F::ZERO, F::ZERO, F::ZERO]);
    const NEG_ONE: Self = Self([F::NEG_ONE, F::ZERO, F::ZERO, F::ZERO, F::ZERO]);

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
        let Self([a0, a1, a2, a3, a4]) = *self;
        let w = F::W;
        let double_w = w.double();

        let c0 = a0.square() + double_w * (a1 * a4 + a2 * a3);
        let double_a0 = a0.double();
        let c1 = double_a0 * a1 + double_w * a2 * a4 + w * a3 * a3;
        let c2 = double_a0 * a2 + a1 * a1 + double_w * a4 * a3;
        let double_a1 = a1.double();
        let c3 = double_a0 * a3 + double_a1 * a2 + w * a4 * a4;
        let c4 = double_a0 * a4 + double_a1 * a3 + a2 * a2;

        Self([c0, c1, c2, c3, c4])
    }
}

impl<F: OptimallyExtendable<5>> Field for QuinticOef<F> {
    type Packing = Self;
    // Algorithm 11.3.4 in Handbook of Elliptic and Hyperelliptic Curve Cryptography.
    fn try_inverse(&self) -> Option<Self> {
        if self.is_zero() {
            return None;
        }

        // Writing 'a' for self:
        let d = self.frobenius(); // d = a^p
        let e = d * d.frobenius(); // e = a^(p + p^2)
        let f = e * e.repeated_frobenius(2); // f = a^(p + p^2 + p^3 + p^4)

        // g = a^r is in the base field, so only compute that
        // coefficient rather than the full product.
        let Self([a0, a1, a2, a3, a4]) = *self;
        let Self([b0, b1, b2, b3, b4]) = f;
        let g = a0 * b0 + F::W * (a1 * b4 + a2 * b3 + a3 * b2 + a4 * b1);

        debug_assert_eq!(Self::from(g), *self * f);

        Some(f * g.inverse())
    }
}

impl<F: OptimallyExtendable<5>> Display for QuinticOef<F> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{} + {}*a", self.0[0], self.0[1])
    }
}

impl<F: OptimallyExtendable<5>> Debug for QuinticOef<F> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(self, f)
    }
}

impl<F: OptimallyExtendable<5>> Neg for QuinticOef<F> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        Self([-self.0[0], -self.0[1], -self.0[2], -self.0[3], -self.0[4]])
    }
}

impl<F: OptimallyExtendable<5>> Add for QuinticOef<F> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self([
            self.0[0] + rhs.0[0],
            self.0[1] + rhs.0[1],
            self.0[2] + rhs.0[2],
            self.0[3] + rhs.0[3],
            self.0[4] + rhs.0[4],
        ])
    }
}

impl<F: OptimallyExtendable<5>> Add<F> for QuinticOef<F> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: F) -> Self {
        Self([self.0[0] + rhs, self.0[1], self.0[2], self.0[3], self.0[4]])
    }
}

impl<F: OptimallyExtendable<5>> AddAssign for QuinticOef<F> {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<F: OptimallyExtendable<5>> AddAssign<F> for QuinticOef<F> {
    fn add_assign(&mut self, rhs: F) {
        *self = *self + rhs;
    }
}

impl<F: OptimallyExtendable<5>> Sum for QuinticOef<F> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |acc, x| acc + x)
    }
}

impl<F: OptimallyExtendable<5>> Sub for QuinticOef<F> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self([
            self.0[0] - rhs.0[0],
            self.0[1] - rhs.0[1],
            self.0[2] - rhs.0[2],
            self.0[3] - rhs.0[3],
            self.0[4] - rhs.0[4],
        ])
    }
}

impl<F: OptimallyExtendable<5>> Sub<F> for QuinticOef<F> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: F) -> Self {
        Self([self.0[0] - rhs, self.0[1], self.0[2], self.0[3], self.0[4]])
    }
}

impl<F: OptimallyExtendable<5>> SubAssign for QuinticOef<F> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<F: OptimallyExtendable<5>> SubAssign<F> for QuinticOef<F> {
    #[inline]
    fn sub_assign(&mut self, rhs: F) {
        *self = *self - rhs;
    }
}

impl<F: OptimallyExtendable<5>> Mul for QuinticOef<F> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let Self([a0, a1, a2, a3, a4]) = self;
        let Self([b0, b1, b2, b3, b4]) = rhs;
        let w = F::W;

        let c0 = a0 * b0 + w * (a1 * b4 + a2 * b3 + a3 * b2 + a4 * b1);
        let c1 = a0 * b1 + a1 * b0 + w * (a2 * b4 + a3 * b3 + a4 * b2);
        let c2 = a0 * b2 + a1 * b1 + a2 * b0 + w * (a3 * b4 + a4 * b3);
        let c3 = a0 * b3 + a1 * b2 + a2 * b1 + a3 * b0 + w * a4 * b4;
        let c4 = a0 * b4 + a1 * b3 + a2 * b2 + a3 * b1 + a4 * b0;

        Self([c0, c1, c2, c3, c4])
    }
}

impl<F: OptimallyExtendable<5>> Mul<F> for QuinticOef<F> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: F) -> Self {
        Self([
            self.0[0] * rhs,
            self.0[1] * rhs,
            self.0[2] * rhs,
            self.0[3] * rhs,
            self.0[4] * rhs,
        ])
    }
}

impl<F: OptimallyExtendable<5>> Product for QuinticOef<F> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ONE, |acc, x| acc * x)
    }
}

impl<F: OptimallyExtendable<5>> Div for QuinticOef<F> {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: Self) -> Self::Output {
        self * rhs.inverse()
    }
}

impl<F: OptimallyExtendable<5>> DivAssign for QuinticOef<F> {
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl<F: OptimallyExtendable<5>> MulAssign for QuinticOef<F> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<F: OptimallyExtendable<5>> MulAssign<F> for QuinticOef<F> {
    fn mul_assign(&mut self, rhs: F) {
        *self = *self * rhs;
    }
}

impl<F: OptimallyExtendable<5>> AbstractExtensionField<F> for QuinticOef<F> {
    const D: usize = F::D;

    fn from_base(b: F) -> Self {
        Self([b, F::ZERO, F::ZERO, F::ZERO, F::ZERO])
    }

    fn from_base_slice(bs: &[F]) -> Self {
        assert_eq!(bs.len(), 5);
        Self([bs[0], bs[1], bs[2], bs[3], bs[4]])
    }

    fn as_base_slice(&self) -> &[F] {
        self.0.as_ref()
    }
}

impl<F: OptimallyExtendable<5>> Distribution<QuinticOef<F>> for Standard
where
    Standard: Distribution<F>,
{
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> QuinticOef<F> {
        QuinticOef::<F>::from_base_slice(&[
            Standard.sample(rng),
            Standard.sample(rng),
            Standard.sample(rng),
            Standard.sample(rng),
            Standard.sample(rng),
        ])
    }
}
