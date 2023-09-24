use core::array;
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::{AbstractField, Field};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FieldArray<F: Field, const N: usize>(pub [F; N]);

impl<F: Field, const N: usize> Default for FieldArray<F, N> {
    fn default() -> Self {
        Self::ZERO
    }
}

impl<F: Field, const N: usize> From<F> for FieldArray<F, N> {
    fn from(val: F) -> Self {
        [val; N].into()
    }
}

impl<F: Field, const N: usize> From<[F; N]> for FieldArray<F, N> {
    fn from(arr: [F; N]) -> Self {
        Self(arr)
    }
}

impl<F: Field, const N: usize> AbstractField for FieldArray<F, N> {
    type F = F;

    const ZERO: Self = FieldArray([F::ZERO; N]);
    const ONE: Self = FieldArray([F::ONE; N]);
    const TWO: Self = FieldArray([F::TWO; N]);
    const NEG_ONE: Self = FieldArray([F::NEG_ONE; N]);

    fn from_bool(b: bool) -> Self {
        [F::from_bool(b); N].into()
    }

    fn from_canonical_u8(n: u8) -> Self {
        [F::from_canonical_u8(n); N].into()
    }

    fn from_canonical_u16(n: u16) -> Self {
        [F::from_canonical_u16(n); N].into()
    }

    fn from_canonical_u32(n: u32) -> Self {
        [F::from_canonical_u32(n); N].into()
    }

    fn from_canonical_u64(n: u64) -> Self {
        [F::from_canonical_u64(n); N].into()
    }

    fn from_canonical_usize(n: usize) -> Self {
        [F::from_canonical_usize(n); N].into()
    }

    fn from_wrapped_u32(n: u32) -> Self {
        [F::from_wrapped_u32(n); N].into()
    }

    fn from_wrapped_u64(n: u64) -> Self {
        [F::from_wrapped_u64(n); N].into()
    }

    fn multiplicative_group_generator() -> Self {
        [F::multiplicative_group_generator(); N].into()
    }
}

impl<F: Field, const N: usize> Add for FieldArray<F, N> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        array::from_fn(|i| self.0[i] + rhs.0[i]).into()
    }
}

impl<F: Field, const N: usize> AddAssign for FieldArray<F, N> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.0.iter_mut().zip(rhs.0).for_each(|(x, y)| *x += y);
    }
}

impl<F: Field, const N: usize> Sub for FieldArray<F, N> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        array::from_fn(|i| self.0[i] - rhs.0[i]).into()
    }
}

impl<F: Field, const N: usize> SubAssign for FieldArray<F, N> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.0.iter_mut().zip(rhs.0).for_each(|(x, y)| *x -= y);
    }
}

impl<F: Field, const N: usize> Neg for FieldArray<F, N> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        self.0.map(|x| -x).into()
    }
}

impl<F: Field, const N: usize> Mul for FieldArray<F, N> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        array::from_fn(|i| self.0[i] * rhs.0[i]).into()
    }
}

impl<F: Field, const N: usize> MulAssign for FieldArray<F, N> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        self.0.iter_mut().zip(rhs.0).for_each(|(x, y)| *x *= y);
    }
}

impl<F: Field, const N: usize> Sum for FieldArray<F, N> {
    #[inline]
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|lhs, rhs| lhs + rhs).unwrap_or(Self::ZERO)
    }
}

impl<F: Field, const N: usize> Product for FieldArray<F, N> {
    #[inline]
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|lhs, rhs| lhs * rhs).unwrap_or(Self::ONE)
    }
}
