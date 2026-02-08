//! Packed quintic extension field.
//!
//! This module provides a packed version of the quintic extension field for SIMD operations.

use alloc::vec::Vec;
use core::array;
use core::fmt::Debug;
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use p3_util::{flatten_to_base, reconstitute_from_base};
use rand::distr::{Distribution, StandardUniform};
use serde::{Deserialize, Serialize};

use super::quintic_extension::{quintic_mul, quintic_square};
use super::{QuinticTrinomialExtensionField, vector_add, vector_sub};
use crate::extension::QuinticTrinomialExtendable;
use crate::{
    Algebra, BasedVectorSpace, Field, PackedField, PackedFieldExtension, PackedValue, Powers,
    PrimeCharacteristicRing, field_to_array,
};

/// Packed quintic extension field.
///
/// This is a wrapper around `[PF; 5]` where `PF` is a packed field type.
/// It enables SIMD-style operations on multiple quintic extension field elements.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, Serialize, Deserialize, PartialOrd, Ord)]
#[repr(transparent)]
#[must_use]
pub struct PackedQuinticTrinomialExtensionField<F: Field, PF: PackedField<Scalar = F>> {
    #[serde(
        with = "p3_util::array_serialization",
        bound(serialize = "PF: Serialize", deserialize = "PF: Deserialize<'de>")
    )]
    pub(crate) value: [PF; 5],
}

impl<F: Field, PF: PackedField<Scalar = F>> PackedQuinticTrinomialExtensionField<F, PF> {
    const fn new(value: [PF; 5]) -> Self {
        Self { value }
    }
}

impl<F: Field, PF: PackedField<Scalar = F>> Default
    for PackedQuinticTrinomialExtensionField<F, PF>
{
    #[inline]
    fn default() -> Self {
        Self {
            value: array::from_fn(|_| PF::ZERO),
        }
    }
}

impl<F: Field, PF: PackedField<Scalar = F>> From<QuinticTrinomialExtensionField<F>>
    for PackedQuinticTrinomialExtensionField<F, PF>
{
    #[inline]
    fn from(x: QuinticTrinomialExtensionField<F>) -> Self {
        Self {
            value: x.value.map(Into::into),
        }
    }
}

impl<F: Field, PF: PackedField<Scalar = F>> From<PF>
    for PackedQuinticTrinomialExtensionField<F, PF>
{
    #[inline]
    fn from(x: PF) -> Self {
        Self {
            value: field_to_array(x),
        }
    }
}

impl<F: Field, PF: PackedField<Scalar = F>>
    Distribution<PackedQuinticTrinomialExtensionField<F, PF>> for StandardUniform
where
    Self: Distribution<PF>,
{
    #[inline]
    fn sample<R: rand::Rng + ?Sized>(
        &self,
        rng: &mut R,
    ) -> PackedQuinticTrinomialExtensionField<F, PF> {
        PackedQuinticTrinomialExtensionField::new(array::from_fn(|_| self.sample(rng)))
    }
}

impl<F: QuinticTrinomialExtendable, PF: PackedField<Scalar = F>>
    Algebra<QuinticTrinomialExtensionField<F>> for PackedQuinticTrinomialExtensionField<F, PF>
{
}

impl<F: QuinticTrinomialExtendable, PF: PackedField<Scalar = F>> Algebra<PF>
    for PackedQuinticTrinomialExtensionField<F, PF>
{
}

impl<F, PF> PrimeCharacteristicRing for PackedQuinticTrinomialExtensionField<F, PF>
where
    F: QuinticTrinomialExtendable,
    PF: PackedField<Scalar = F>,
{
    type PrimeSubfield = PF::PrimeSubfield;

    const ZERO: Self = Self {
        value: [PF::ZERO; 5],
    };

    const ONE: Self = Self {
        value: field_to_array(PF::ONE),
    };

    const TWO: Self = Self {
        value: field_to_array(PF::TWO),
    };

    const NEG_ONE: Self = Self {
        value: field_to_array(PF::NEG_ONE),
    };

    #[inline]
    fn from_prime_subfield(val: Self::PrimeSubfield) -> Self {
        PF::from_prime_subfield(val).into()
    }

    #[inline]
    fn from_bool(b: bool) -> Self {
        PF::from_bool(b).into()
    }

    #[inline]
    fn halve(&self) -> Self {
        Self::new(self.value.map(|x| x.halve()))
    }

    #[inline(always)]
    fn square(&self) -> Self {
        let mut res = Self::default();
        quintic_square(&self.value, &mut res.value);
        res
    }

    #[inline]
    fn mul_2exp_u64(&self, exp: u64) -> Self {
        Self::new(self.value.map(|x| x.mul_2exp_u64(exp)))
    }

    #[inline]
    fn div_2exp_u64(&self, exp: u64) -> Self {
        Self::new(self.value.map(|x| x.div_2exp_u64(exp)))
    }

    #[inline]
    fn zero_vec(len: usize) -> Vec<Self> {
        // SAFETY: `Self` is `repr(transparent)` over `[PF; 5]`.
        unsafe { reconstitute_from_base(PF::zero_vec(len * 5)) }
    }
}

impl<F, PF> BasedVectorSpace<PF> for PackedQuinticTrinomialExtensionField<F, PF>
where
    F: QuinticTrinomialExtendable,
    PF: PackedField<Scalar = F>,
{
    const DIMENSION: usize = 5;

    #[inline]
    fn as_basis_coefficients_slice(&self) -> &[PF] {
        &self.value
    }

    #[inline]
    fn from_basis_coefficients_fn<Fn: FnMut(usize) -> PF>(f: Fn) -> Self {
        Self::new(array::from_fn(f))
    }

    #[inline]
    fn from_basis_coefficients_iter<I: ExactSizeIterator<Item = PF>>(mut iter: I) -> Option<Self> {
        (iter.len() == 5).then(|| Self::new(array::from_fn(|_| iter.next().unwrap())))
    }

    #[inline]
    fn flatten_to_base(vec: Vec<Self>) -> Vec<PF> {
        // SAFETY: `Self` is `repr(transparent)` over `[PF; 5]`.
        unsafe { flatten_to_base(vec) }
    }

    #[inline]
    fn reconstitute_from_base(vec: Vec<PF>) -> Vec<Self> {
        // SAFETY: `Self` is `repr(transparent)` over `[PF; 5]`.
        unsafe { reconstitute_from_base(vec) }
    }
}

// SAFETY: Memory layout is compatible with `[QuinticTrinomialExtensionField<F>; WIDTH]`.
unsafe impl<F, PF> PackedValue for PackedQuinticTrinomialExtensionField<F, PF>
where
    F: QuinticTrinomialExtendable,
    PF: PackedField<Scalar = F>,
{
    type Value = QuinticTrinomialExtensionField<F>;

    const WIDTH: usize = PF::WIDTH;

    #[inline]
    fn from_slice(slice: &[Self::Value]) -> &Self {
        assert_eq!(slice.len(), Self::WIDTH);
        unsafe { &*slice.as_ptr().cast() }
    }

    #[inline]
    fn from_slice_mut(slice: &mut [Self::Value]) -> &mut Self {
        assert_eq!(slice.len(), Self::WIDTH);
        unsafe { &mut *slice.as_mut_ptr().cast() }
    }

    #[inline]
    fn from_fn<Fn: FnMut(usize) -> Self::Value>(mut f: Fn) -> Self {
        let mut result = Self::default();
        for i in 0..Self::WIDTH {
            let val = f(i);
            for j in 0..5 {
                result.value[j].as_slice_mut()[i] = val.value[j];
            }
        }
        result
    }

    #[inline]
    fn as_slice(&self) -> &[Self::Value] {
        unsafe {
            core::slice::from_raw_parts(self as *const Self as *const Self::Value, Self::WIDTH)
        }
    }

    #[inline]
    fn as_slice_mut(&mut self) -> &mut [Self::Value] {
        unsafe {
            core::slice::from_raw_parts_mut(self as *mut Self as *mut Self::Value, Self::WIDTH)
        }
    }
}

// SAFETY: Implements all required trait bounds.
unsafe impl<F, PF> PackedField for PackedQuinticTrinomialExtensionField<F, PF>
where
    F: QuinticTrinomialExtendable,
    PF: PackedField<Scalar = F>,
{
    type Scalar = QuinticTrinomialExtensionField<F>;
}

impl<F: QuinticTrinomialExtendable> PackedFieldExtension<F, QuinticTrinomialExtensionField<F>>
    for PackedQuinticTrinomialExtensionField<F, F::Packing>
{
    #[inline]
    fn from_ext_slice(ext_slice: &[QuinticTrinomialExtensionField<F>]) -> Self {
        let width = F::Packing::WIDTH;
        assert_eq!(ext_slice.len(), width);

        let res = array::from_fn(|i| F::Packing::from_fn(|j| ext_slice[j].value[i]));
        Self::new(res)
    }

    #[inline]
    fn packed_ext_powers(base: QuinticTrinomialExtensionField<F>) -> Powers<Self> {
        use itertools::Itertools;
        let width = F::Packing::WIDTH;
        let powers = base.powers().take(width + 1).collect_vec();
        // Transpose first WIDTH powers
        let current = Self::from_ext_slice(&powers[..width]);

        // Broadcast self^WIDTH
        let multiplier = powers[width].into();

        Powers {
            base: multiplier,
            current,
        }
    }
}

impl<F, PF> Neg for PackedQuinticTrinomialExtensionField<F, PF>
where
    F: QuinticTrinomialExtendable,
    PF: PackedField<Scalar = F>,
{
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        Self::new(self.value.map(PF::neg))
    }
}

impl<F, PF> Add for PackedQuinticTrinomialExtensionField<F, PF>
where
    F: QuinticTrinomialExtendable,
    PF: PackedField<Scalar = F>,
{
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::new(vector_add(&self.value, &rhs.value))
    }
}

impl<F, PF> Add<QuinticTrinomialExtensionField<F>> for PackedQuinticTrinomialExtensionField<F, PF>
where
    F: QuinticTrinomialExtendable,
    PF: PackedField<Scalar = F>,
{
    type Output = Self;

    #[inline]
    fn add(self, rhs: QuinticTrinomialExtensionField<F>) -> Self {
        let value = vector_add(&self.value, &rhs.value);
        Self { value }
    }
}

impl<F, PF> Add<PF> for PackedQuinticTrinomialExtensionField<F, PF>
where
    F: QuinticTrinomialExtendable,
    PF: PackedField<Scalar = F>,
{
    type Output = Self;

    #[inline]
    fn add(mut self, rhs: PF) -> Self {
        self.value[0] += rhs;
        self
    }
}

impl<F, PF> AddAssign for PackedQuinticTrinomialExtensionField<F, PF>
where
    F: QuinticTrinomialExtendable,
    PF: PackedField<Scalar = F>,
{
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..5 {
            self.value[i] += rhs.value[i];
        }
    }
}

impl<F, PF> AddAssign<QuinticTrinomialExtensionField<F>>
    for PackedQuinticTrinomialExtensionField<F, PF>
where
    F: QuinticTrinomialExtendable,
    PF: PackedField<Scalar = F>,
{
    #[inline]
    fn add_assign(&mut self, rhs: QuinticTrinomialExtensionField<F>) {
        for i in 0..5 {
            self.value[i] += rhs.value[i];
        }
    }
}

impl<F, PF> AddAssign<PF> for PackedQuinticTrinomialExtensionField<F, PF>
where
    F: QuinticTrinomialExtendable,
    PF: PackedField<Scalar = F>,
{
    #[inline]
    fn add_assign(&mut self, rhs: PF) {
        self.value[0] += rhs;
    }
}

impl<F, PF> Sum for PackedQuinticTrinomialExtensionField<F, PF>
where
    F: QuinticTrinomialExtendable,
    PF: PackedField<Scalar = F>,
{
    #[inline]
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|acc, x| acc + x).unwrap_or(Self::ZERO)
    }
}

impl<F, PF> Sub for PackedQuinticTrinomialExtensionField<F, PF>
where
    F: QuinticTrinomialExtendable,
    PF: PackedField<Scalar = F>,
{
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::new(vector_sub(&self.value, &rhs.value))
    }
}

impl<F, PF> Sub<QuinticTrinomialExtensionField<F>> for PackedQuinticTrinomialExtensionField<F, PF>
where
    F: QuinticTrinomialExtendable,
    PF: PackedField<Scalar = F>,
{
    type Output = Self;

    #[inline]
    fn sub(self, rhs: QuinticTrinomialExtensionField<F>) -> Self {
        let value = vector_sub(&self.value, &rhs.value);
        Self { value }
    }
}

impl<F, PF> Sub<PF> for PackedQuinticTrinomialExtensionField<F, PF>
where
    F: QuinticTrinomialExtendable,
    PF: PackedField<Scalar = F>,
{
    type Output = Self;

    #[inline]
    fn sub(self, rhs: PF) -> Self {
        let mut res = self.value;
        res[0] -= rhs;
        Self { value: res }
    }
}

impl<F, PF> SubAssign for PackedQuinticTrinomialExtensionField<F, PF>
where
    F: QuinticTrinomialExtendable,
    PF: PackedField<Scalar = F>,
{
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<F, PF> SubAssign<QuinticTrinomialExtensionField<F>>
    for PackedQuinticTrinomialExtensionField<F, PF>
where
    F: QuinticTrinomialExtendable,
    PF: PackedField<Scalar = F>,
{
    #[inline]
    fn sub_assign(&mut self, rhs: QuinticTrinomialExtensionField<F>) {
        *self = *self - rhs;
    }
}

impl<F, PF> SubAssign<PF> for PackedQuinticTrinomialExtensionField<F, PF>
where
    F: QuinticTrinomialExtendable,
    PF: PackedField<Scalar = F>,
{
    #[inline]
    fn sub_assign(&mut self, rhs: PF) {
        *self = *self - rhs;
    }
}

impl<F, PF> Mul for PackedQuinticTrinomialExtensionField<F, PF>
where
    F: QuinticTrinomialExtendable,
    PF: PackedField<Scalar = F>,
{
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let mut res = Self::default();
        quintic_mul(&self.value, &rhs.value, &mut res.value);
        res
    }
}

impl<F, PF> Mul<QuinticTrinomialExtensionField<F>> for PackedQuinticTrinomialExtensionField<F, PF>
where
    F: QuinticTrinomialExtendable,
    PF: PackedField<Scalar = F>,
{
    type Output = Self;

    #[inline]
    fn mul(self, rhs: QuinticTrinomialExtensionField<F>) -> Self {
        self * Self::from(rhs)
    }
}

impl<F, PF> Mul<PF> for PackedQuinticTrinomialExtensionField<F, PF>
where
    F: QuinticTrinomialExtendable,
    PF: PackedField<Scalar = F>,
{
    type Output = Self;

    #[inline]
    fn mul(self, rhs: PF) -> Self {
        Self {
            value: self.value.map(|x| x * rhs),
        }
    }
}

impl<F, PF> Product for PackedQuinticTrinomialExtensionField<F, PF>
where
    F: QuinticTrinomialExtendable,
    PF: PackedField<Scalar = F>,
{
    #[inline]
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|acc, x| acc * x).unwrap_or(Self::ONE)
    }
}

impl<F, PF> MulAssign for PackedQuinticTrinomialExtensionField<F, PF>
where
    F: QuinticTrinomialExtendable,
    PF: PackedField<Scalar = F>,
{
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<F, PF> MulAssign<QuinticTrinomialExtensionField<F>>
    for PackedQuinticTrinomialExtensionField<F, PF>
where
    F: QuinticTrinomialExtendable,
    PF: PackedField<Scalar = F>,
{
    #[inline]
    fn mul_assign(&mut self, rhs: QuinticTrinomialExtensionField<F>) {
        *self = *self * rhs;
    }
}

impl<F, PF> MulAssign<PF> for PackedQuinticTrinomialExtensionField<F, PF>
where
    F: QuinticTrinomialExtendable,
    PF: PackedField<Scalar = F>,
{
    #[inline]
    fn mul_assign(&mut self, rhs: PF) {
        *self = *self * rhs;
    }
}

impl<F, PF> Sum<QuinticTrinomialExtensionField<F>> for PackedQuinticTrinomialExtensionField<F, PF>
where
    F: QuinticTrinomialExtendable,
    PF: PackedField<Scalar = F>,
{
    #[inline]
    fn sum<I: Iterator<Item = QuinticTrinomialExtensionField<F>>>(iter: I) -> Self {
        iter.map(Self::from).sum()
    }
}

impl<F, PF> Product<QuinticTrinomialExtensionField<F>>
    for PackedQuinticTrinomialExtensionField<F, PF>
where
    F: QuinticTrinomialExtendable,
    PF: PackedField<Scalar = F>,
{
    #[inline]
    fn product<I: Iterator<Item = QuinticTrinomialExtensionField<F>>>(iter: I) -> Self {
        iter.map(Self::from).product()
    }
}

impl<F, PF> Div<QuinticTrinomialExtensionField<F>> for PackedQuinticTrinomialExtensionField<F, PF>
where
    F: QuinticTrinomialExtendable,
    PF: PackedField<Scalar = F>,
{
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    #[inline]
    fn div(self, rhs: QuinticTrinomialExtensionField<F>) -> Self {
        self * Self::from(rhs.inverse())
    }
}

impl<F, PF> DivAssign<QuinticTrinomialExtensionField<F>>
    for PackedQuinticTrinomialExtensionField<F, PF>
where
    F: QuinticTrinomialExtendable,
    PF: PackedField<Scalar = F>,
{
    #[inline]
    fn div_assign(&mut self, rhs: QuinticTrinomialExtensionField<F>) {
        *self = *self / rhs;
    }
}
