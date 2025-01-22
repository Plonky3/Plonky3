use alloc::vec::Vec;
use core::array;
use core::fmt::Debug;
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use p3_util::convert_vec;
use serde::{Deserialize, Serialize};

use super::{binomial_mul, cubic_square, vector_add, vector_sub, BinomialExtensionField};
use crate::extension::BinomiallyExtendable;
use crate::field::Field;
use crate::{field_to_array, FieldAlgebra, FieldExtensionAlgebra, PackedField, Serializable};

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, Serialize, Deserialize, PartialOrd, Ord)]
#[repr(transparent)] // to make the zero_vec implementation safe
pub struct PackedBinomialExtensionField<F: Field, PF: PackedField<Scalar = F>, const D: usize> {
    #[serde(
        with = "p3_util::array_serialization",
        bound(serialize = "PF: Serialize", deserialize = "PF: Deserialize<'de>")
    )]
    pub(crate) value: [PF; D],
}

impl<F: Field, PF: PackedField<Scalar = F>, const D: usize> Default
    for PackedBinomialExtensionField<F, PF, D>
{
    fn default() -> Self {
        Self {
            value: array::from_fn(|_| PF::ZERO),
        }
    }
}

impl<F: Field, PF: PackedField<Scalar = F>, const D: usize> From<BinomialExtensionField<F, D>>
    for PackedBinomialExtensionField<F, PF, D>
{
    fn from(x: BinomialExtensionField<F, D>) -> Self {
        Self {
            value: x.value.map(Into::<PF>::into),
        }
    }
}

impl<F: Field, PF: PackedField<Scalar = F>, const D: usize> From<PF>
    for PackedBinomialExtensionField<F, PF, D>
{
    fn from(x: PF) -> Self {
        Self {
            value: field_to_array::<PF, D>(x),
        }
    }
}

impl<F, PF, const D: usize> FieldAlgebra for PackedBinomialExtensionField<F, PF, D>
where
    F: BinomiallyExtendable<D>,
    PF: PackedField<Scalar = F>,
{
    type F = BinomialExtensionField<F, D>;

    type PrimeSubfield = PF::PrimeSubfield;

    const ZERO: Self = Self {
        value: [PF::ZERO; D],
    };

    const ONE: Self = Self {
        value: field_to_array::<PF, D>(PF::ONE),
    };

    const TWO: Self = Self {
        value: field_to_array::<PF, D>(PF::TWO),
    };

    const NEG_ONE: Self = Self {
        value: field_to_array::<PF, D>(PF::NEG_ONE),
    };

    #[inline]
    fn from_prime_subfield(val: Self::PrimeSubfield) -> Self {
        PF::from_prime_subfield(val).into()
    }

    #[inline]
    fn from_bool(b: bool) -> Self {
        PF::from_bool(b).into()
    }

    #[inline(always)]
    fn square(&self) -> Self {
        match D {
            2 => {
                let a = self.value;
                let mut res = Self::default();
                res.value[0] = a[0].square() + a[1].square() * F::W;
                res.value[1] = a[0] * a[1].double();
                res
            }
            3 => {
                let mut res = Self::default();
                cubic_square(&self.value, &mut res.value, F::W);
                res
            }
            _ => <Self as Mul<Self>>::mul(*self, *self),
        }
    }

    #[inline]
    fn zero_vec(len: usize) -> Vec<Self> {
        // SAFETY: this is a repr(transparent) wrapper around an array.
        unsafe { convert_vec(PF::zero_vec(len * D)) }
    }
}

impl<F, PF, const D: usize> Serializable<PF> for PackedBinomialExtensionField<F, PF, D>
where
    F: BinomiallyExtendable<D>,
    PF: PackedField<Scalar = F>,
{
    const DIMENSION: usize = D;

    fn serialize_as_slice(&self) -> &[PF] {
        &self.value
    }

    fn deserialize_fn<Fn: FnMut(usize) -> PF>(f: Fn) -> Self {
        Self {
            value: array::from_fn(f),
        }
    }

    fn deserialize_iter<I: Iterator<Item = PF>>(iter: I) -> Self {
        let mut res = Self::default();
        for (i, b) in iter.enumerate() {
            res.value[i] = b;
        }
        res
    }
}

impl<F, PF, const D: usize> FieldExtensionAlgebra<PF> for PackedBinomialExtensionField<F, PF, D>
where
    F: BinomiallyExtendable<D>,
    PF: PackedField<Scalar = F>,
{
    const D: usize = D;
}

impl<F, PF, const D: usize> Neg for PackedBinomialExtensionField<F, PF, D>
where
    F: BinomiallyExtendable<D>,
    PF: PackedField<Scalar = F>,
{
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        Self {
            value: self.value.map(PF::neg),
        }
    }
}

impl<F, PF, const D: usize> Add for PackedBinomialExtensionField<F, PF, D>
where
    F: BinomiallyExtendable<D>,
    PF: PackedField<Scalar = F>,
{
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        let value = vector_add(&self.value, &rhs.value);
        Self { value }
    }
}

impl<F, PF, const D: usize> Add<BinomialExtensionField<F, D>>
    for PackedBinomialExtensionField<F, PF, D>
where
    F: BinomiallyExtendable<D>,
    PF: PackedField<Scalar = F>,
{
    type Output = Self;

    #[inline]
    fn add(self, rhs: BinomialExtensionField<F, D>) -> Self {
        let value = vector_add(&self.value, &rhs.value);
        Self { value }
    }
}

impl<F, PF, const D: usize> Add<PF> for PackedBinomialExtensionField<F, PF, D>
where
    F: BinomiallyExtendable<D>,
    PF: PackedField<Scalar = F>,
{
    type Output = Self;

    #[inline]
    fn add(mut self, rhs: PF) -> Self {
        self.value[0] += rhs;
        self
    }
}

impl<F, PF, const D: usize> AddAssign for PackedBinomialExtensionField<F, PF, D>
where
    F: BinomiallyExtendable<D>,
    PF: PackedField<Scalar = F>,
{
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..D {
            self.value[i] += rhs.value[i];
        }
    }
}

impl<F, PF, const D: usize> AddAssign<BinomialExtensionField<F, D>>
    for PackedBinomialExtensionField<F, PF, D>
where
    F: BinomiallyExtendable<D>,
    PF: PackedField<Scalar = F>,
{
    #[inline]
    fn add_assign(&mut self, rhs: BinomialExtensionField<F, D>) {
        for i in 0..D {
            self.value[i] += rhs.value[i];
        }
    }
}

impl<F, PF, const D: usize> AddAssign<PF> for PackedBinomialExtensionField<F, PF, D>
where
    F: BinomiallyExtendable<D>,
    PF: PackedField<Scalar = F>,
{
    #[inline]
    fn add_assign(&mut self, rhs: PF) {
        self.value[0] += rhs;
    }
}

impl<F, PF, const D: usize> Sum for PackedBinomialExtensionField<F, PF, D>
where
    F: BinomiallyExtendable<D>,
    PF: PackedField<Scalar = F>,
{
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |acc, x| acc + x)
    }
}

impl<F, PF, const D: usize> Sub for PackedBinomialExtensionField<F, PF, D>
where
    F: BinomiallyExtendable<D>,
    PF: PackedField<Scalar = F>,
{
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        let value = vector_sub(&self.value, &rhs.value);
        Self { value }
    }
}

impl<F, PF, const D: usize> Sub<BinomialExtensionField<F, D>>
    for PackedBinomialExtensionField<F, PF, D>
where
    F: BinomiallyExtendable<D>,
    PF: PackedField<Scalar = F>,
{
    type Output = Self;

    #[inline]
    fn sub(self, rhs: BinomialExtensionField<F, D>) -> Self {
        let value = vector_sub(&self.value, &rhs.value);
        Self { value }
    }
}

impl<F, PF, const D: usize> Sub<PF> for PackedBinomialExtensionField<F, PF, D>
where
    F: BinomiallyExtendable<D>,
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

impl<F, PF, const D: usize> SubAssign for PackedBinomialExtensionField<F, PF, D>
where
    F: BinomiallyExtendable<D>,
    PF: PackedField<Scalar = F>,
{
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<F, PF, const D: usize> SubAssign<BinomialExtensionField<F, D>>
    for PackedBinomialExtensionField<F, PF, D>
where
    F: BinomiallyExtendable<D>,
    PF: PackedField<Scalar = F>,
{
    #[inline]
    fn sub_assign(&mut self, rhs: BinomialExtensionField<F, D>) {
        *self = *self - rhs;
    }
}

impl<F, PF, const D: usize> SubAssign<PF> for PackedBinomialExtensionField<F, PF, D>
where
    F: BinomiallyExtendable<D>,
    PF: PackedField<Scalar = F>,
{
    #[inline]
    fn sub_assign(&mut self, rhs: PF) {
        *self = *self - rhs;
    }
}

impl<F, PF, const D: usize> Mul for PackedBinomialExtensionField<F, PF, D>
where
    F: BinomiallyExtendable<D>,
    PF: PackedField<Scalar = F>,
{
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let a = self.value;
        let b = rhs.value;
        let mut res = Self::default();
        let w: PF = F::W.into();

        binomial_mul(&a, &b, &mut res.value, w);

        res
    }
}

impl<F, PF, const D: usize> Mul<BinomialExtensionField<F, D>>
    for PackedBinomialExtensionField<F, PF, D>
where
    F: BinomiallyExtendable<D>,
    PF: PackedField<Scalar = F>,
{
    type Output = Self;

    #[inline]
    fn mul(self, rhs: BinomialExtensionField<F, D>) -> Self {
        let a = self.value;
        let b = rhs.value;
        let mut res = Self::default();
        let w: PF = F::W.into();

        binomial_mul(&a, &b, &mut res.value, w);

        res
    }
}

impl<F, PF, const D: usize> Mul<PF> for PackedBinomialExtensionField<F, PF, D>
where
    F: BinomiallyExtendable<D>,
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

impl<F, PF, const D: usize> Product for PackedBinomialExtensionField<F, PF, D>
where
    F: BinomiallyExtendable<D>,
    PF: PackedField<Scalar = F>,
{
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ONE, |acc, x| acc * x)
    }
}

impl<F, PF, const D: usize> MulAssign for PackedBinomialExtensionField<F, PF, D>
where
    F: BinomiallyExtendable<D>,
    PF: PackedField<Scalar = F>,
{
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<F, PF, const D: usize> MulAssign<BinomialExtensionField<F, D>>
    for PackedBinomialExtensionField<F, PF, D>
where
    F: BinomiallyExtendable<D>,
    PF: PackedField<Scalar = F>,
{
    #[inline]
    fn mul_assign(&mut self, rhs: BinomialExtensionField<F, D>) {
        *self = *self * rhs;
    }
}

impl<F, PF, const D: usize> MulAssign<PF> for PackedBinomialExtensionField<F, PF, D>
where
    F: BinomiallyExtendable<D>,
    PF: PackedField<Scalar = F>,
{
    #[inline]
    fn mul_assign(&mut self, rhs: PF) {
        *self = *self * rhs;
    }
}
