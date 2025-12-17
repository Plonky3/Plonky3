use core::array;
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Div, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign};

use p3_util::{as_base_slice, as_base_slice_mut};

use crate::batch_inverse::batch_multiplicative_inverse_general;
use crate::{Algebra, Field, PackedValue, PrimeCharacteristicRing};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(transparent)] // Needed to make `transmute`s safe.
#[must_use]
pub struct FieldArray<F: Field, const N: usize>(pub [F; N]);

impl<F: Field, const N: usize> FieldArray<F, N> {
    /// Compute the element-wise multiplicative inverse using batched inversion.
    ///
    /// Uses Montgomery's batch inversion trick to compute all inverses with a
    /// single field inversion, improving performance when N > 1.
    #[inline]
    pub(crate) fn inverse(&self) -> Self {
        let mut result = Self::default();
        batch_multiplicative_inverse_general(&self.0, &mut result.0, |x| x.inverse());
        result
    }

    /// Apply a function to each element, returning a new `FieldArray<U, N>`.
    #[inline]
    pub fn map<Func, U: Field>(self, f: Func) -> FieldArray<U, N>
    where
        Func: FnMut(F) -> U,
    {
        FieldArray(self.map_into_array(f))
    }

    /// Apply a function to each element, returning a raw array `[U; N]`.
    ///
    /// Unlike [`map`](Self::map), this does not require `U: Field`.
    #[inline]
    pub fn map_into_array<Func, U>(self, f: Func) -> [U; N]
    where
        Func: FnMut(F) -> U,
    {
        self.0.map(f)
    }

    /// View as a slice of raw `[F; N]` arrays.
    ///
    /// This is a zero-cost transmute enabled by the `#[repr(transparent)]` layout.
    #[inline]
    pub const fn as_raw_slice(s: &[Self]) -> &[[F; N]] {
        // SAFETY: `FieldArray<F, N>` is `#[repr(transparent)]` over `[F; N]`,
        // so `&[FieldArray<F, N>]` and `&[[F; N]]` have identical layouts.
        unsafe { as_base_slice(s) }
    }

    /// View as a mutable slice of raw `[F; N]` arrays.
    ///
    /// This is a zero-cost transmute enabled by the `#[repr(transparent)]` layout.
    #[inline]
    pub const fn as_raw_slice_mut(s: &mut [Self]) -> &mut [[F; N]] {
        // SAFETY: `FieldArray<F, N>` is `#[repr(transparent)]` over `[F; N]`,
        // so `&mut [FieldArray<F, N>]` and `&mut [[F; N]]` have identical layouts.
        unsafe { as_base_slice_mut(s) }
    }

    /// Reinterpret a slice of `[F; N]` as a slice of `FieldArray<F, N>`.
    ///
    /// This is a zero-cost transmute enabled by the `#[repr(transparent)]` layout.
    #[inline]
    pub const fn from_raw_slice(s: &[[F; N]]) -> &[Self] {
        // SAFETY: `FieldArray<F, N>` is `#[repr(transparent)]` over `[F; N]`,
        // so `&[[F; N]]` and `&[FieldArray<F, N>]` have identical layouts.
        unsafe { as_base_slice(s) }
    }

    /// Reinterpret a mutable slice of `[F; N]` as a mutable slice of `FieldArray<F, N>`.
    ///
    /// This is a zero-cost transmute enabled by the `#[repr(transparent)]` layout.
    #[inline]
    pub const fn from_raw_slice_mut(s: &mut [[F; N]]) -> &mut [Self] {
        // SAFETY: `FieldArray<F, N>` is `#[repr(transparent)]` over `[F; N]`,
        // so `&mut [[F; N]]` and `&mut [FieldArray<F, N>]` have identical layouts.
        unsafe { as_base_slice_mut(s) }
    }
}

impl<F: Field, const N: usize> IndexMut<usize> for FieldArray<F, N> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.0.index_mut(index)
    }
}

impl<F: Field, const N: usize> Index<usize> for FieldArray<F, N> {
    type Output = F;

    fn index(&self, index: usize) -> &Self::Output {
        self.0.index(index)
    }
}

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

impl<F: Field, const N: usize> PrimeCharacteristicRing for FieldArray<F, N> {
    type PrimeSubfield = F::PrimeSubfield;

    const ZERO: Self = Self([F::ZERO; N]);
    const ONE: Self = Self([F::ONE; N]);
    const TWO: Self = Self([F::TWO; N]);
    const NEG_ONE: Self = Self([F::NEG_ONE; N]);

    #[inline]
    fn from_prime_subfield(f: Self::PrimeSubfield) -> Self {
        F::from_prime_subfield(f).into()
    }

    #[inline]
    fn halve(&self) -> Self {
        Self(self.0.map(|x| x.halve()))
    }
}

impl<F: Field, const N: usize> Algebra<F> for FieldArray<F, N> {}

unsafe impl<F: Field, const N: usize> PackedValue for FieldArray<F, N> {
    type Value = F;

    const WIDTH: usize = N;

    fn from_slice(slice: &[Self::Value]) -> &Self {
        assert_eq!(slice.len(), Self::WIDTH);
        unsafe { &*slice.as_ptr().cast() }
    }

    fn from_slice_mut(slice: &mut [Self::Value]) -> &mut Self {
        assert_eq!(slice.len(), Self::WIDTH);
        unsafe { &mut *slice.as_mut_ptr().cast() }
    }

    fn from_fn<Fn>(f: Fn) -> Self
    where
        Fn: FnMut(usize) -> Self::Value,
    {
        Self(array::from_fn(f))
    }

    fn as_slice(&self) -> &[Self::Value] {
        &self.0
    }

    fn as_slice_mut(&mut self) -> &mut [Self::Value] {
        &mut self.0
    }
}

impl<F: Field, const N: usize> Add for FieldArray<F, N> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        array::from_fn(|i| self.0[i] + rhs.0[i]).into()
    }
}

impl<F: Field, const N: usize> Add<F> for FieldArray<F, N> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: F) -> Self::Output {
        self.0.map(|x| x + rhs).into()
    }
}

impl<F: Field, const N: usize> AddAssign for FieldArray<F, N> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.0.iter_mut().zip(rhs.0).for_each(|(x, y)| *x += y);
    }
}

impl<F: Field, const N: usize> AddAssign<F> for FieldArray<F, N> {
    #[inline]
    fn add_assign(&mut self, rhs: F) {
        self.0.iter_mut().for_each(|x| *x += rhs);
    }
}

impl<F: Field, const N: usize> Sub for FieldArray<F, N> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        array::from_fn(|i| self.0[i] - rhs.0[i]).into()
    }
}

impl<F: Field, const N: usize> Sub<F> for FieldArray<F, N> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: F) -> Self::Output {
        self.0.map(|x| x - rhs).into()
    }
}

impl<F: Field, const N: usize> SubAssign for FieldArray<F, N> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.0.iter_mut().zip(rhs.0).for_each(|(x, y)| *x -= y);
    }
}

impl<F: Field, const N: usize> SubAssign<F> for FieldArray<F, N> {
    #[inline]
    fn sub_assign(&mut self, rhs: F) {
        self.0.iter_mut().for_each(|x| *x -= rhs);
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

impl<F: Field, const N: usize> Mul<F> for FieldArray<F, N> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: F) -> Self::Output {
        self.0.map(|x| x * rhs).into()
    }
}

impl<F: Field, const N: usize> MulAssign for FieldArray<F, N> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        self.0.iter_mut().zip(rhs.0).for_each(|(x, y)| *x *= y);
    }
}

impl<F: Field, const N: usize> MulAssign<F> for FieldArray<F, N> {
    #[inline]
    fn mul_assign(&mut self, rhs: F) {
        self.0.iter_mut().for_each(|x| *x *= rhs);
    }
}

impl<F: Field, const N: usize> Div<F> for FieldArray<F, N> {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    #[inline]
    fn div(self, rhs: F) -> Self::Output {
        let rhs_inv = rhs.inverse();
        self * rhs_inv
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
