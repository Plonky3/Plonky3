use alloc::vec::Vec;
use core::array;
use core::fmt::Debug;
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use p3_util::convert_vec;
use serde::{Deserialize, Serialize};

use super::{cubic_mul, cubic_square, BinomialExtensionField};
use crate::extension::BinomiallyExtendable;
use crate::field::Field;
use crate::{field_to_array, FieldAlgebra, FieldExtensionAlgebra, PackedField};

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
    fn from_f(f: Self::F) -> Self {
        Self {
            value: f.value.map(PF::from_f),
        }
    }

    #[inline]
    fn from_bool(b: bool) -> Self {
        PF::from_bool(b).into()
    }

    #[inline]
    fn from_canonical_u8(n: u8) -> Self {
        PF::from_canonical_u8(n).into()
    }

    #[inline]
    fn from_canonical_u16(n: u16) -> Self {
        PF::from_canonical_u16(n).into()
    }

    #[inline]
    fn from_canonical_u32(n: u32) -> Self {
        PF::from_canonical_u32(n).into()
    }

    #[inline]
    fn from_canonical_u64(n: u64) -> Self {
        PF::from_canonical_u64(n).into()
    }

    #[inline]
    fn from_canonical_usize(n: usize) -> Self {
        PF::from_canonical_usize(n).into()
    }

    #[inline]
    fn from_wrapped_u32(n: u32) -> Self {
        PF::from_wrapped_u32(n).into()
    }

    #[inline]
    fn from_wrapped_u64(n: u64) -> Self {
        PF::from_wrapped_u64(n).into()
    }

    #[inline(always)]
    fn square(&self) -> Self {
        match D {
            2 => {
                let a = self.value;
                let mut res = Self::default();
                res.value[0] = a[0].square() + a[1].square() * PF::from_f(F::W);
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

impl<F, PF, const D: usize> FieldExtensionAlgebra<PF> for PackedBinomialExtensionField<F, PF, D>
where
    F: BinomiallyExtendable<D>,
    PF: PackedField<Scalar = F>,
{
    const D: usize = D;

    #[inline]
    fn from_base(b: PF) -> Self {
        Self {
            value: field_to_array(b),
        }
    }

    #[inline]
    fn from_base_slice(bs: &[PF]) -> Self {
        Self::from_base_fn(|i| bs[i])
    }

    #[inline]
    fn from_base_fn<Fn: FnMut(usize) -> PF>(f: Fn) -> Self {
        Self {
            value: array::from_fn(f),
        }
    }

    #[inline]
    fn from_base_iter<I: Iterator<Item = PF>>(iter: I) -> Self {
        let mut res = Self::default();
        for (i, b) in iter.enumerate() {
            res.value[i] = b;
        }
        res
    }

    #[inline(always)]
    fn as_base_slice(&self) -> &[PF] {
        &self.value
    }
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
        let mut res = self.value;
        for (r, rhs_val) in res.iter_mut().zip(rhs.value) {
            *r += rhs_val;
        }
        Self { value: res }
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
        Self { value: res }
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

        match D {
            2 => {
                res.value[0] = a[0] * b[0] + a[1] * w * b[1];
                res.value[1] = a[0] * b[1] + a[1] * b[0];
            }
            3 => cubic_mul(&a, &b, &mut res.value, w),
            _ =>
            {
                #[allow(clippy::needless_range_loop)]
                for i in 0..D {
                    for j in 0..D {
                        if i + j >= D {
                            res.value[i + j - D] += a[i] * w * b[j];
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
