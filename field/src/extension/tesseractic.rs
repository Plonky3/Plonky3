use core::fmt::{self, Debug, Display, Formatter};
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use rand::distributions::Standard;
use rand::prelude::Distribution;

use super::{Frobenius, OptimallyExtendable};
use crate::field::Field;
use crate::{AbstractExtensionField, AbstractField};

#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct TesseracticOef<F: OptimallyExtendable<4>>(pub [F; 4]);

impl<F: OptimallyExtendable<4>> Frobenius<F, 4> for TesseracticOef<F> {}

impl<F: OptimallyExtendable<4>> Default for TesseracticOef<F> {
    fn default() -> Self {
        Self::ZERO
    }
}

impl<F: OptimallyExtendable<4>> From<F> for TesseracticOef<F> {
    fn from(x: F) -> Self {
        Self([x, F::ZERO, F::ZERO, F::ZERO])
    }
}

impl<F: OptimallyExtendable<4>> AbstractField for TesseracticOef<F> {
    const ZERO: Self = Self([F::ZERO; 4]);
    const ONE: Self = Self([F::ONE, F::ZERO, F::ZERO, F::ZERO]);
    const TWO: Self = Self([F::TWO, F::ZERO, F::ZERO, F::ZERO]);
    const NEG_ONE: Self = Self([F::NEG_ONE, F::ZERO, F::ZERO, F::ZERO]);

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
        let Self([a0, a1, a2, a3]) = *self;
        let w = F::W;
        let double_a3 = a3.double();
        let double_a1 = a1.double();

        let c0 = a0.square() + w * (double_a3 * a1 + a2.square());
        let c1 = double_a1 * a0 + double_a3 * a2 * w;
        let c2 = a1.square() + a3.square() * w + (a0 * a2).double();
        let c3 = a2 * double_a1 + a0 * double_a3;
        Self([c0, c1, c2, c3])
    }
}

impl<F: OptimallyExtendable<4>> Field for TesseracticOef<F> {
    type Packing = Self;
    // Algorithm 11.3.4 in Handbook of Elliptic and Hyperelliptic Curve Cryptography.
    fn try_inverse(&self) -> Option<Self> {
        if self.is_zero() {
            return None;
        }

        // Writing 'a' for self:
        let d = self.frobenius(); // d = a^p
        let e = d * d.frobenius(); // e = a^(p + p^2)
        let f = d * e.frobenius(); // f = a^(p + p^2 + p^3)

        // g = a^r is in the base field, so only compute that
        // coefficient rather than the full product.

        let Self([a0, a1, a2, a3]) = *self;
        let Self([b0, b1, b2, b3]) = f;
        let g = a0 * b0 + F::W * (a1 * b3 + a2 * b2 + a3 * b1);
        debug_assert_eq!(Self::from(g), *self * f);

        Some(f * g.inverse())
    }
}

impl<F: OptimallyExtendable<4>> Display for TesseracticOef<F> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{} + {}*a", self.0[0], self.0[1])
    }
}

impl<F: OptimallyExtendable<4>> Debug for TesseracticOef<F> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(self, f)
    }
}

impl<F: OptimallyExtendable<4>> Neg for TesseracticOef<F> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        Self([-self.0[0], -self.0[1], -self.0[2], -self.0[3]])
    }
}

impl<F: OptimallyExtendable<4>> Add for TesseracticOef<F> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self([
            self.0[0] + rhs.0[0],
            self.0[1] + rhs.0[1],
            self.0[2] + rhs.0[2],
            self.0[3] + rhs.0[3],
        ])
    }
}

impl<F: OptimallyExtendable<4>> Add<F> for TesseracticOef<F> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: F) -> Self {
        Self([self.0[0] + rhs, self.0[1], self.0[2], self.0[3]])
    }
}

impl<F: OptimallyExtendable<4>> AddAssign for TesseracticOef<F> {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<F: OptimallyExtendable<4>> AddAssign<F> for TesseracticOef<F> {
    fn add_assign(&mut self, rhs: F) {
        *self = *self + rhs;
    }
}

impl<F: OptimallyExtendable<4>> Sum for TesseracticOef<F> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |acc, x| acc + x)
    }
}

impl<F: OptimallyExtendable<4>> Sub for TesseracticOef<F> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self([
            self.0[0] - rhs.0[0],
            self.0[1] - rhs.0[1],
            self.0[2] - rhs.0[2],
            self.0[3] - rhs.0[3],
        ])
    }
}

impl<F: OptimallyExtendable<4>> Sub<F> for TesseracticOef<F> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: F) -> Self {
        Self([self.0[0] - rhs, self.0[1], self.0[2], self.0[3]])
    }
}

impl<F: OptimallyExtendable<4>> SubAssign for TesseracticOef<F> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<F: OptimallyExtendable<4>> SubAssign<F> for TesseracticOef<F> {
    #[inline]
    fn sub_assign(&mut self, rhs: F) {
        *self = *self - rhs;
    }
}

impl<F: OptimallyExtendable<4>> Mul for TesseracticOef<F> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let Self([a0, a1, a2, a3]) = self;
        let Self([b0, b1, b2, b3]) = rhs;
        let w = F::W;

        // use karatsuba's method to reduce the multiplications
        // let self = A0+A1X^2; rhs = B0+B1X^2
        // low = A0B0, mid = (A0+A1)(B0+B1)-A0B0-A1B1, high = A1B1
        // result = low + (mid-low-high)*X^2+ high*X^4

        // compute low degree terms
        let low0 = a0 * b0;
        let low2 = a1 * b1;
        let low1 = (a0 + a1) * (b0 + b1) - low0 - low2;

        // compute high degree terms
        // high0 = a2_b2, high2 = a3_b3
        let high0 = a2 * b2;
        let high2 = a3 * b3;
        let high1 = (a2 + a3) * (b2 + b3) - high0 - high2;

        // compute mid degree terms
        let c0 = a0 + a2;
        let d0 = b0 + b2;
        let c1 = a1 + a3;
        let d1 = b1 + b3;

        let mid0 = c0 * d0;
        let mid2 = c1 * d1;
        let mid1 = (c0 + c1) * (d0 + d1) - mid0 - mid2;

        Self([
            low0 + (mid2 - low2 - high2 + high0) * w,
            low1 + high1 * w,
            low2 + mid0 - low0 - high0 + high2 * w,
            mid1 - low1 - high1,
        ])
    }
}

impl<F: OptimallyExtendable<4>> Mul<F> for TesseracticOef<F> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: F) -> Self {
        Self([
            self.0[0] * rhs,
            self.0[1] * rhs,
            self.0[2] * rhs,
            self.0[3] * rhs,
        ])
    }
}

impl<F: OptimallyExtendable<4>> Product for TesseracticOef<F> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ONE, |acc, x| acc * x)
    }
}

impl<F: OptimallyExtendable<4>> Div for TesseracticOef<F> {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: Self) -> Self::Output {
        self * rhs.inverse()
    }
}

impl<F: OptimallyExtendable<4>> DivAssign for TesseracticOef<F> {
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl<F: OptimallyExtendable<4>> MulAssign for TesseracticOef<F> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<F: OptimallyExtendable<4>> MulAssign<F> for TesseracticOef<F> {
    fn mul_assign(&mut self, rhs: F) {
        *self = *self * rhs;
    }
}

impl<F: OptimallyExtendable<4>> AbstractExtensionField<F> for TesseracticOef<F> {
    const D: usize = F::D;

    fn from_base(b: F) -> Self {
        Self([b, F::ZERO, F::ZERO, F::ZERO])
    }

    fn from_base_slice(bs: &[F]) -> Self {
        assert_eq!(bs.len(), 4);
        Self([bs[0], bs[1], bs[2], bs[3]])
    }

    fn as_base_slice(&self) -> &[F] {
        self.0.as_ref()
    }
}

impl<F: OptimallyExtendable<4>> Distribution<TesseracticOef<F>> for Standard
where
    Standard: Distribution<F>,
{
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> TesseracticOef<F> {
        TesseracticOef::<F>::from_base_slice(&[
            Standard.sample(rng),
            Standard.sample(rng),
            Standard.sample(rng),
            Standard.sample(rng),
        ])
    }
}
