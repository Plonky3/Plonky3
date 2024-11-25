use alloc::vec::Vec;
use core::array;
use core::fmt::Debug;
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use itertools::Itertools;
use p3_util::convert_vec;
use serde::{Deserialize, Serialize};

use super::BinomialExtensionField;
use crate::extension::BinomiallyExtendable;
use crate::field::Field;
use crate::{
    field_to_array, AbelianGroup, CommutativeRing, FieldAlgebra, InjectiveRingHomomorphism,
    PackedField, PackedFieldExtension, Powers, PrimeCharacteristicRing, PrimeField,
};

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, Serialize, Deserialize, PartialOrd, Ord)]
#[repr(transparent)] // to make the zero_vec implementation safe
pub struct PackedBinomialExtensionField<F: Field, PF: PackedField<Scalar = F>, const D: usize> {
    #[serde(
        with = "p3_util::array_serialization",
        bound(serialize = "PF: Serialize", deserialize = "PF: Deserialize<'de>")
    )]
    pub(crate) value: [PF; D],
}

impl<F: Field, PF: PackedField<Scalar = F>, const D: usize> PackedBinomialExtensionField<F, PF, D> {
    pub(crate) const fn new(value: [PF; D]) -> Self {
        Self { value }
    }
}

impl<F: Field, PF: PackedField<Scalar = F>, const D: usize> Default
    for PackedBinomialExtensionField<F, PF, D>
{
    fn default() -> Self {
        Self::new(array::from_fn(|_| PF::ZERO))
    }
}

impl<F: Field, PF: PackedField<Scalar = F>, const D: usize> From<PF>
    for PackedBinomialExtensionField<F, PF, D>
{
    fn from(x: PF) -> Self {
        Self::new(field_to_array::<PF, D>(x))
    }
}

impl<F: Field, PF: PackedField<Scalar = F>, const D: usize> From<BinomialExtensionField<F, D>>
    for PackedBinomialExtensionField<F, PF, D>
{
    fn from(value: BinomialExtensionField<F, D>) -> Self {
        Self::new(value.value.map(|x| x.into()))
    }
}

impl<F, PF, const D: usize> PackedFieldExtension for PackedBinomialExtensionField<F, PF, D>
where
    F: BinomiallyExtendable<D> + Field<Packing = PF>,
    PF: PackedField<Scalar = F>,
{
    type BaseField = F;
    type ExtField = BinomialExtensionField<F, D>;

    fn from_ext_element(ext_elem: Self::ExtField) -> Self {
        Self::new(ext_elem.value.map(|x| x.into()))
    }

    fn from_ext_slice(ext_slice: &[Self::ExtField]) -> Self {
        let width = F::Packing::WIDTH;
        assert_eq!(ext_slice.len(), width);

        let mut res = [F::Packing::ZERO; D];

        res.iter_mut().enumerate().for_each(|(i, row_i)| {
            let row_i = row_i.as_slice_mut();
            ext_slice
                .iter()
                .enumerate()
                .for_each(|(j, vec_j)| row_i[j] = vec_j.value[i])
        });

        Self::new(res)
    }

    fn to_ext_vec(packed_ext_elem: &Self) -> Vec<Self::ExtField> {
        let width = F::Packing::WIDTH;
        let mut out_vec = Vec::new();

        for i in 0..width {
            let arr = array::from_fn(|j| packed_ext_elem.value[j].as_slice()[i]);
            let ext_elem = Self::ExtField::new(arr);
            out_vec.push(ext_elem);
        }

        out_vec
    }

    fn ext_powers_packed(base: Self::ExtField) -> crate::Powers<Self> {
        let width = F::Packing::WIDTH;
        let powers = base.powers().take(width + 1).collect_vec();
        // Transpose first WIDTH powers
        let current = Self::from_ext_slice(&powers[..width]);

        // Broadcast self^WIDTH
        let multiplier = Self::from_ext_element(powers[width]);

        Powers {
            base: multiplier,
            current,
        }
    }
}

impl<F, PF, const D: usize> AbelianGroup for PackedBinomialExtensionField<F, PF, D>
where
    F: BinomiallyExtendable<D>,
    PF: PackedField<Scalar = F>,
{
    const ZERO: Self = Self::new([PF::ZERO; D]);

    fn mul_u64(&self, r: u64) -> Self {
        *self * Self::from_u64(r)
    }

    fn mul_2exp_u64(&self, exp: u64) -> Self {
        let pow = PF::Char::TWO.exp_u64(exp);
        *self * Self::from_char(pow)
    }
}

impl<F, PF, const D: usize> CommutativeRing for PackedBinomialExtensionField<F, PF, D>
where
    F: BinomiallyExtendable<D>,
    PF: PackedField<Scalar = F>,
{
    const ONE: Self = Self::new(field_to_array::<PF, D>(PF::ONE));

    const NEG_ONE: Self = Self::new(field_to_array::<PF, D>(PF::NEG_ONE));

    #[inline(always)]
    fn square(&self) -> Self {
        match D {
            2 => {
                let a = self.value;
                let mut res = Self::default();
                res.value[0] = a[0].square() + a[1].square() * Into::<PF>::into(F::W);
                res.value[1] = a[0] * a[1].double();
                res
            }
            3 => {
                let mut res = Self::default();
                cubic_square(&self.value, &mut res.value);
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

impl<F, PF, const D: usize> PrimeCharacteristicRing for PackedBinomialExtensionField<F, PF, D>
where
    F: BinomiallyExtendable<D>,
    PF: PackedField<Scalar = F>,
{
    type Char = PF::Char;

    fn from_char(f: Self::Char) -> Self {
        PF::from_char(f).into()
    }

    fn halve(&self) -> Self {
        Self::new(self.value.map(|x| x.halve()))
    }

    fn div_2exp_u64(&self, _exp: u64) -> Self {
        todo!()
    }
}

impl<F, PF, const D: usize> InjectiveRingHomomorphism<PF> for PackedBinomialExtensionField<F, PF, D>
where
    F: BinomiallyExtendable<D>,
    PF: PackedField<Scalar = F>,
{
}

impl<F, PF, const D: usize> InjectiveRingHomomorphism<BinomialExtensionField<F, D>>
    for PackedBinomialExtensionField<F, PF, D>
where
    F: BinomiallyExtendable<D>,
    PF: PackedField<Scalar = F>,
{
}

impl<F, PF, const D: usize> InjectiveRingHomomorphism<PackedBinomialExtensionField<F, PF, D>>
    for PackedBinomialExtensionField<F, PF, D>
where
    F: BinomiallyExtendable<D>,
    PF: PackedField<Scalar = F>,
{
}

impl<F, PF, const D: usize> Neg for PackedBinomialExtensionField<F, PF, D>
where
    F: BinomiallyExtendable<D>,
    PF: PackedField<Scalar = F>,
{
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        Self::new(self.value.map(PF::neg))
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
        let mut res = self.value;
        for (r, rhs_val) in res.iter_mut().zip(rhs.value) {
            *r += rhs_val;
        }
        Self::new(res)
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
        let mut res = self.value;
        for (r, rhs_val) in res.iter_mut().zip(rhs.value) {
            *r -= rhs_val;
        }
        Self::new(res)
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
        Self::new(res)
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
        let w = F::W;
        let w_af = Into::<PF>::into(w);

        match D {
            2 => {
                res.value[0] = a[0] * b[0] + a[1] * w_af * b[1];
                res.value[1] = a[0] * b[1] + a[1] * b[0];
            }
            3 => cubic_mul(&a, &b, &mut res.value),
            _ =>
            {
                #[allow(clippy::needless_range_loop)]
                for i in 0..D {
                    for j in 0..D {
                        if i + j >= D {
                            res.value[i + j - D] += a[i] * w_af * b[j];
                        } else {
                            res.value[i + j] += a[i] * b[j];
                        }
                    }
                }
            }
        }
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
        Self::new(self.value.map(|x| x * rhs))
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

/// karatsuba multiplication for cubic extension field
#[inline]
fn cubic_mul<F: BinomiallyExtendable<D>, PF: FieldAlgebra<F>, const D: usize>(
    a: &[PF; D],
    b: &[PF; D],
    res: &mut [PF; D],
) {
    assert_eq!(D, 3);

    let a0_b0 = a[0].clone() * b[0].clone();
    let a1_b1 = a[1].clone() * b[1].clone();
    let a2_b2 = a[2].clone() * b[2].clone();

    res[0] = a0_b0.clone()
        + ((a[1].clone() + a[2].clone()) * (b[1].clone() + b[2].clone())
            - a1_b1.clone()
            - a2_b2.clone())
            * Into::<PF>::into(F::W);
    res[1] = (a[0].clone() + a[1].clone()) * (b[0].clone() + b[1].clone())
        - a0_b0.clone()
        - a1_b1.clone()
        + a2_b2.clone() * Into::<PF>::into(F::W);
    res[2] = (a[0].clone() + a[2].clone()) * (b[0].clone() + b[2].clone()) - a0_b0 - a2_b2 + a1_b1;
}

/// Section 11.3.6a in Handbook of Elliptic and Hyperelliptic Curve Cryptography.
#[inline]
fn cubic_square<F: BinomiallyExtendable<D>, PF: FieldAlgebra<F>, const D: usize>(
    a: &[PF; D],
    res: &mut [PF; D],
) {
    assert_eq!(D, 3);

    let w_a2 = a[2].clone() * Into::<PF>::into(F::W);

    res[0] = a[0].square() + (a[1].clone() * w_a2.clone()).double();
    res[1] = w_a2 * a[2].clone() + (a[0].clone() * a[1].clone()).double();
    res[2] = a[1].square() + (a[0].clone() * a[2].clone()).double();
}
