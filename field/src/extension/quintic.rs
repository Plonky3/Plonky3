use core::fmt::{self, Debug, Display, Formatter};
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use rand::distributions::Standard;
use rand::prelude::Distribution;

use super::{Frobenius, HasFrobenuis};
use crate::field::Field;
use crate::{AbstractExtensionField, AbstractField};

#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct QuinticBef<F: HasFrobenuis<5>>(pub [F; 5]);

impl<F: HasFrobenuis<5>> Frobenius<F, 5> for QuinticBef<F> {}

impl<F: HasFrobenuis<5>> Default for QuinticBef<F> {
    fn default() -> Self {
        Self::ZERO
    }
}

impl<F: HasFrobenuis<5>> From<F> for QuinticBef<F> {
    fn from(x: F) -> Self {
        Self([x, F::ZERO, F::ZERO, F::ZERO, F::ZERO])
    }
}

impl<F: HasFrobenuis<5>> AbstractField for QuinticBef<F> {
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

impl<F: HasFrobenuis<5>> Field for QuinticBef<F> {
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

impl<F: HasFrobenuis<5>> Display for QuinticBef<F> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{} + {}*a", self.0[0], self.0[1])
    }
}

impl<F: HasFrobenuis<5>> Debug for QuinticBef<F> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(self, f)
    }
}

impl<F: HasFrobenuis<5>> Neg for QuinticBef<F> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        Self([-self.0[0], -self.0[1], -self.0[2], -self.0[3], -self.0[4]])
    }
}

impl<F: HasFrobenuis<5>> Add for QuinticBef<F> {
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

impl<F: HasFrobenuis<5>> Add<F> for QuinticBef<F> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: F) -> Self {
        Self([self.0[0] + rhs, self.0[1], self.0[2], self.0[3], self.0[4]])
    }
}

impl<F: HasFrobenuis<5>> AddAssign for QuinticBef<F> {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<F: HasFrobenuis<5>> AddAssign<F> for QuinticBef<F> {
    fn add_assign(&mut self, rhs: F) {
        *self = *self + rhs;
    }
}

impl<F: HasFrobenuis<5>> Sum for QuinticBef<F> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |acc, x| acc + x)
    }
}

impl<F: HasFrobenuis<5>> Sub for QuinticBef<F> {
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

impl<F: HasFrobenuis<5>> Sub<F> for QuinticBef<F> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: F) -> Self {
        Self([self.0[0] - rhs, self.0[1], self.0[2], self.0[3], self.0[4]])
    }
}

impl<F: HasFrobenuis<5>> SubAssign for QuinticBef<F> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<F: HasFrobenuis<5>> SubAssign<F> for QuinticBef<F> {
    #[inline]
    fn sub_assign(&mut self, rhs: F) {
        *self = *self - rhs;
    }
}

impl<F: HasFrobenuis<5>> Mul for QuinticBef<F> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let Self([a0, a1, a2, a3, a4]) = self;
        let Self([b0, b1, b2, b3, b4]) = rhs;
        let w = F::W;

        // use karatsuba's method to reduce the multiplications
        // let self = A0+A1X^3; rhs = B0+B1X^4
        // low = A0B0, mid = (A0+A1)(B0+B1)-A0B0-A1B1, high = A1B1
        // result = low + (mid-low-high)*X^3+ high*X^6

        // compute low degree terms
        // low0 = a0_b0 , low4= a2b2
        let a0_b0 = a0 * b0;
        let a1_b1 = a1 * b1;
        let a2_b2 = a2 * b2;

        let low1 = (a0 + a1) * (b0 + b1) - a0_b0 - a1_b1;
        let low2 = (a0 + a2) * (b0 + b2) - a0_b0 - a2_b2 + a1_b1;
        let low3 = (a1 + a2) * (b1 + b2) - a1_b1 - a2_b2;

        // compute high degree terms

        let high0 = a3 * b3;
        let high2 = a4 * b4;
        let high1 = (a3 + a4) * (b3 + b4) - high0 - high2;

        // compute mid degree term, c2 = a2, d2= b2
        let c0 = a0 + a3;
        let c1 = a1 + a4;
        let d0 = b0 + b3;
        let d1 = b1 + b4;

        let c0_d0 = c0 * d0;
        let c1_d1 = c1 * d1;

        // mid4= 0
        let mid0 = c0_d0 - a0_b0 - high0;
        let mid1 = (c0 + c1) * (d0 + d1) - c0_d0 - c1_d1 - low1 - high1;
        let mid2 = (c0 + a2) * (d0 + b2) - c0_d0 - a2_b2 + c1_d1 - low2 - high2;
        let mid3 = (c1 + a2) * (d1 + b2) - c1_d1 - a2_b2 - low3;

        Self([
            a0_b0 + mid2 * w,
            low1 + (mid3 + high0) * w,
            low2 + high1 * w,
            low3 + mid0 + high2 * w,
            a2_b2 + mid1,
        ])
    }
}

impl<F: HasFrobenuis<5>> Mul<F> for QuinticBef<F> {
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

impl<F: HasFrobenuis<5>> Product for QuinticBef<F> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ONE, |acc, x| acc * x)
    }
}

impl<F: HasFrobenuis<5>> Div for QuinticBef<F> {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: Self) -> Self::Output {
        self * rhs.inverse()
    }
}

impl<F: HasFrobenuis<5>> DivAssign for QuinticBef<F> {
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl<F: HasFrobenuis<5>> MulAssign for QuinticBef<F> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<F: HasFrobenuis<5>> MulAssign<F> for QuinticBef<F> {
    fn mul_assign(&mut self, rhs: F) {
        *self = *self * rhs;
    }
}

impl<F: HasFrobenuis<5>> AbstractExtensionField<F> for QuinticBef<F> {
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

impl<F: HasFrobenuis<5>> Distribution<QuinticBef<F>> for Standard
where
    Standard: Distribution<F>,
{
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> QuinticBef<F> {
        QuinticBef::<F>::from_base_slice(&[
            Standard.sample(rng),
            Standard.sample(rng),
            Standard.sample(rng),
            Standard.sample(rng),
            Standard.sample(rng),
        ])
    }
}
