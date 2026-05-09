//! Packed cubic extension field.
//!
//! This module provides a packed version of the cubic extension field for SIMD operations.

use alloc::vec::Vec;
use core::array;
use core::fmt::Debug;
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use p3_util::{flatten_to_base, reconstitute_from_base};
use rand::distr::{Distribution, StandardUniform};
use serde::{Deserialize, Serialize};

use super::cubic_extension::{cubic_square, trinomial_cubic_mul};
use super::{CubicTrinomialExtensionField, vector_add, vector_sub};
use crate::extension::CubicTrinomialExtendable;
use crate::{
    Algebra, BasedVectorSpace, Field, PackedField, PackedFieldExtension, PackedValue, Powers,
    PrimeCharacteristicRing, field_to_array,
};

/// Packed cubic extension field.
///
/// This is a wrapper around `[PF; 3]` where `PF` is a packed field type.
/// It enables SIMD-style operations on multiple cubic extension field elements.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, Serialize, Deserialize, PartialOrd, Ord)]
#[repr(transparent)]
#[must_use]
pub struct PackedCubicTrinomialExtensionField<F: Field, PF: PackedField<Scalar = F>> {
    #[serde(
        with = "p3_util::array_serialization",
        bound(serialize = "PF: Serialize", deserialize = "PF: Deserialize<'de>")
    )]
    pub(crate) value: [PF; 3],
}

impl<F: Field, PF: PackedField<Scalar = F>> PackedCubicTrinomialExtensionField<F, PF> {
    const fn new(value: [PF; 3]) -> Self {
        Self { value }
    }
}

impl<F: Field, PF: PackedField<Scalar = F>> Default for PackedCubicTrinomialExtensionField<F, PF> {
    #[inline]
    fn default() -> Self {
        Self {
            value: [PF::ZERO, PF::ZERO, PF::ZERO],
        }
    }
}

impl<F: Field, PF: PackedField<Scalar = F>> From<CubicTrinomialExtensionField<F>>
    for PackedCubicTrinomialExtensionField<F, PF>
{
    #[inline]
    fn from(x: CubicTrinomialExtensionField<F>) -> Self {
        Self {
            value: x.value.map(Into::into),
        }
    }
}

impl<F: Field, PF: PackedField<Scalar = F>> From<PF> for PackedCubicTrinomialExtensionField<F, PF> {
    #[inline]
    fn from(x: PF) -> Self {
        Self {
            value: [x, PF::ZERO, PF::ZERO],
        }
    }
}

impl<F: Field, PF: PackedField<Scalar = F>> Distribution<PackedCubicTrinomialExtensionField<F, PF>>
    for StandardUniform
where
    Self: Distribution<PF>,
{
    #[inline]
    fn sample<R: rand::Rng + ?Sized>(
        &self,
        rng: &mut R,
    ) -> PackedCubicTrinomialExtensionField<F, PF> {
        PackedCubicTrinomialExtensionField::new(array::from_fn(|_| self.sample(rng)))
    }
}

impl<F: CubicTrinomialExtendable, PF: PackedField<Scalar = F>>
    Algebra<CubicTrinomialExtensionField<F>> for PackedCubicTrinomialExtensionField<F, PF>
{
}

impl<F: CubicTrinomialExtendable, PF: PackedField<Scalar = F>> Algebra<PF>
    for PackedCubicTrinomialExtensionField<F, PF>
{
}

impl<F, PF> PrimeCharacteristicRing for PackedCubicTrinomialExtensionField<F, PF>
where
    F: CubicTrinomialExtendable,
    PF: PackedField<Scalar = F>,
{
    type PrimeSubfield = PF::PrimeSubfield;

    const ZERO: Self = Self {
        value: [PF::ZERO; 3],
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
        cubic_square(&self.value, &mut res.value);
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
        // SAFETY: `Self` is `repr(transparent)` over `[PF; 3]`.
        unsafe { reconstitute_from_base(PF::zero_vec(len * 3)) }
    }
}

impl<F, PF> BasedVectorSpace<PF> for PackedCubicTrinomialExtensionField<F, PF>
where
    F: CubicTrinomialExtendable,
    PF: PackedField<Scalar = F>,
{
    const DIMENSION: usize = 3;

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
        (iter.len() == 3).then(|| Self::new(array::from_fn(|_| iter.next().unwrap())))
    }

    #[inline]
    fn flatten_to_base(vec: Vec<Self>) -> Vec<PF> {
        // SAFETY: `Self` is `repr(transparent)` over `[PF; 3]`.
        unsafe { flatten_to_base(vec) }
    }

    #[inline]
    fn reconstitute_from_base(vec: Vec<PF>) -> Vec<Self> {
        // SAFETY: `Self` is `repr(transparent)` over `[PF; 3]`.
        unsafe { reconstitute_from_base(vec) }
    }
}

impl<F: CubicTrinomialExtendable> PackedFieldExtension<F, CubicTrinomialExtensionField<F>>
    for PackedCubicTrinomialExtensionField<F, F::Packing>
{
    #[inline]
    fn from_ext_fn(f: impl Fn(usize) -> CubicTrinomialExtensionField<F>) -> Self {
        Self::new(F::Packing::pack_columns_fn(|lane| f(lane).value))
    }

    #[inline]
    fn packed_ext_powers(base: CubicTrinomialExtensionField<F>) -> Powers<Self> {
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

impl<F, PF> Neg for PackedCubicTrinomialExtensionField<F, PF>
where
    F: CubicTrinomialExtendable,
    PF: PackedField<Scalar = F>,
{
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        Self::new(self.value.map(PF::neg))
    }
}

impl<F, PF> Add for PackedCubicTrinomialExtensionField<F, PF>
where
    F: CubicTrinomialExtendable,
    PF: PackedField<Scalar = F>,
{
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::new(vector_add(&self.value, &rhs.value))
    }
}

impl<F, PF> Add<CubicTrinomialExtensionField<F>> for PackedCubicTrinomialExtensionField<F, PF>
where
    F: CubicTrinomialExtendable,
    PF: PackedField<Scalar = F>,
{
    type Output = Self;

    #[inline]
    fn add(self, rhs: CubicTrinomialExtensionField<F>) -> Self {
        let value = vector_add(&self.value, &rhs.value);
        Self { value }
    }
}

impl<F, PF> Add<PF> for PackedCubicTrinomialExtensionField<F, PF>
where
    F: CubicTrinomialExtendable,
    PF: PackedField<Scalar = F>,
{
    type Output = Self;

    #[inline]
    fn add(mut self, rhs: PF) -> Self {
        self.value[0] += rhs;
        self
    }
}

impl<F, PF> AddAssign for PackedCubicTrinomialExtensionField<F, PF>
where
    F: CubicTrinomialExtendable,
    PF: PackedField<Scalar = F>,
{
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..3 {
            self.value[i] += rhs.value[i];
        }
    }
}

impl<F, PF> AddAssign<CubicTrinomialExtensionField<F>> for PackedCubicTrinomialExtensionField<F, PF>
where
    F: CubicTrinomialExtendable,
    PF: PackedField<Scalar = F>,
{
    #[inline]
    fn add_assign(&mut self, rhs: CubicTrinomialExtensionField<F>) {
        for i in 0..3 {
            self.value[i] += rhs.value[i];
        }
    }
}

impl<F, PF> AddAssign<PF> for PackedCubicTrinomialExtensionField<F, PF>
where
    F: CubicTrinomialExtendable,
    PF: PackedField<Scalar = F>,
{
    #[inline]
    fn add_assign(&mut self, rhs: PF) {
        self.value[0] += rhs;
    }
}

impl<F, PF> Sum for PackedCubicTrinomialExtensionField<F, PF>
where
    F: CubicTrinomialExtendable,
    PF: PackedField<Scalar = F>,
{
    #[inline]
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|acc, x| acc + x).unwrap_or(Self::ZERO)
    }
}

impl<F, PF> Sub for PackedCubicTrinomialExtensionField<F, PF>
where
    F: CubicTrinomialExtendable,
    PF: PackedField<Scalar = F>,
{
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::new(vector_sub(&self.value, &rhs.value))
    }
}

impl<F, PF> Sub<CubicTrinomialExtensionField<F>> for PackedCubicTrinomialExtensionField<F, PF>
where
    F: CubicTrinomialExtendable,
    PF: PackedField<Scalar = F>,
{
    type Output = Self;

    #[inline]
    fn sub(self, rhs: CubicTrinomialExtensionField<F>) -> Self {
        let value = vector_sub(&self.value, &rhs.value);
        Self { value }
    }
}

impl<F, PF> Sub<PF> for PackedCubicTrinomialExtensionField<F, PF>
where
    F: CubicTrinomialExtendable,
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

impl<F, PF> SubAssign for PackedCubicTrinomialExtensionField<F, PF>
where
    F: CubicTrinomialExtendable,
    PF: PackedField<Scalar = F>,
{
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<F, PF> SubAssign<CubicTrinomialExtensionField<F>> for PackedCubicTrinomialExtensionField<F, PF>
where
    F: CubicTrinomialExtendable,
    PF: PackedField<Scalar = F>,
{
    #[inline]
    fn sub_assign(&mut self, rhs: CubicTrinomialExtensionField<F>) {
        *self = *self - rhs;
    }
}

impl<F, PF> SubAssign<PF> for PackedCubicTrinomialExtensionField<F, PF>
where
    F: CubicTrinomialExtendable,
    PF: PackedField<Scalar = F>,
{
    #[inline]
    fn sub_assign(&mut self, rhs: PF) {
        *self = *self - rhs;
    }
}

impl<F, PF> Mul for PackedCubicTrinomialExtensionField<F, PF>
where
    F: CubicTrinomialExtendable,
    PF: PackedField<Scalar = F>,
{
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let mut res = Self::default();
        trinomial_cubic_mul(&self.value, &rhs.value, &mut res.value);
        res
    }
}

impl<F, PF> Mul<CubicTrinomialExtensionField<F>> for PackedCubicTrinomialExtensionField<F, PF>
where
    F: CubicTrinomialExtendable,
    PF: PackedField<Scalar = F>,
{
    type Output = Self;

    #[inline]
    fn mul(self, rhs: CubicTrinomialExtensionField<F>) -> Self {
        self * Self::from(rhs)
    }
}

impl<F, PF> Mul<PF> for PackedCubicTrinomialExtensionField<F, PF>
where
    F: CubicTrinomialExtendable,
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

impl<F, PF> Product for PackedCubicTrinomialExtensionField<F, PF>
where
    F: CubicTrinomialExtendable,
    PF: PackedField<Scalar = F>,
{
    #[inline]
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|acc, x| acc * x).unwrap_or(Self::ONE)
    }
}

impl<F, PF> MulAssign for PackedCubicTrinomialExtensionField<F, PF>
where
    F: CubicTrinomialExtendable,
    PF: PackedField<Scalar = F>,
{
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<F, PF> MulAssign<CubicTrinomialExtensionField<F>> for PackedCubicTrinomialExtensionField<F, PF>
where
    F: CubicTrinomialExtendable,
    PF: PackedField<Scalar = F>,
{
    #[inline]
    fn mul_assign(&mut self, rhs: CubicTrinomialExtensionField<F>) {
        *self = *self * rhs;
    }
}

impl<F, PF> MulAssign<PF> for PackedCubicTrinomialExtensionField<F, PF>
where
    F: CubicTrinomialExtendable,
    PF: PackedField<Scalar = F>,
{
    #[inline]
    fn mul_assign(&mut self, rhs: PF) {
        *self = *self * rhs;
    }
}

impl<F, PF> Sum<CubicTrinomialExtensionField<F>> for PackedCubicTrinomialExtensionField<F, PF>
where
    F: CubicTrinomialExtendable,
    PF: PackedField<Scalar = F>,
{
    #[inline]
    fn sum<I: Iterator<Item = CubicTrinomialExtensionField<F>>>(iter: I) -> Self {
        iter.map(Self::from).sum()
    }
}

impl<F, PF> Product<CubicTrinomialExtensionField<F>> for PackedCubicTrinomialExtensionField<F, PF>
where
    F: CubicTrinomialExtendable,
    PF: PackedField<Scalar = F>,
{
    #[inline]
    fn product<I: Iterator<Item = CubicTrinomialExtensionField<F>>>(iter: I) -> Self {
        iter.map(Self::from).product()
    }
}

impl<F, PF> Div<CubicTrinomialExtensionField<F>> for PackedCubicTrinomialExtensionField<F, PF>
where
    F: CubicTrinomialExtendable,
    PF: PackedField<Scalar = F>,
{
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    #[inline]
    fn div(self, rhs: CubicTrinomialExtensionField<F>) -> Self {
        self * Self::from(rhs.inverse())
    }
}

impl<F, PF> DivAssign<CubicTrinomialExtensionField<F>> for PackedCubicTrinomialExtensionField<F, PF>
where
    F: CubicTrinomialExtendable,
    PF: PackedField<Scalar = F>,
{
    #[inline]
    fn div_assign(&mut self, rhs: CubicTrinomialExtensionField<F>) {
        *self = *self / rhs;
    }
}

impl<F: CubicTrinomialExtendable> Div for PackedCubicTrinomialExtensionField<F, F::Packing> {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    #[inline]
    fn div(self, rhs: Self) -> Self {
        self * crate::invert_packed_extension::<F, CubicTrinomialExtensionField<F>>(rhs)
    }
}

impl<F: CubicTrinomialExtendable> DivAssign for PackedCubicTrinomialExtensionField<F, F::Packing> {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}
