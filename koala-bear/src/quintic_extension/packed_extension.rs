use alloc::vec::Vec;
use core::array;
use core::fmt::Debug;
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use itertools::Itertools;
use p3_field::extension::{vector_add, vector_sub};
use p3_field::{
    Algebra, BasedVectorSpace, Field, PackedField, PackedFieldExtension, PackedValue, Powers,
    PrimeCharacteristicRing, field_to_array,
};
use p3_util::{flatten_to_base, reconstitute_from_base};
use rand::distr::{Distribution, StandardUniform};
use serde::{Deserialize, Serialize};

use super::extension::QuinticExtensionField;
use crate::KoalaBear;

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, Serialize, Deserialize, PartialOrd, Ord)]
#[repr(transparent)] // Needed to make various casts safe.
pub struct PackedQuinticExtensionField<PF: PackedField<Scalar = KoalaBear>> {
    #[serde(
        with = "p3_util::array_serialization",
        bound(serialize = "PF: Serialize", deserialize = "PF: Deserialize<'de>")
    )]
    pub(crate) value: [PF; 5],
}

impl<PF: PackedField<Scalar = KoalaBear>> PackedQuinticExtensionField<PF> {
    const fn new(value: [PF; 5]) -> Self {
        Self { value }
    }
}

impl<PF: PackedField<Scalar = KoalaBear>> Default for PackedQuinticExtensionField<PF> {
    #[inline]
    fn default() -> Self {
        Self {
            value: array::from_fn(|_| PF::ZERO),
        }
    }
}

impl<PF: PackedField<Scalar = KoalaBear>> From<QuinticExtensionField<KoalaBear>>
    for PackedQuinticExtensionField<PF>
{
    #[inline]
    fn from(x: QuinticExtensionField<KoalaBear>) -> Self {
        Self {
            value: x.value.map(Into::into),
        }
    }
}

impl<PF: PackedField<Scalar = KoalaBear>> From<PF> for PackedQuinticExtensionField<PF> {
    #[inline]
    fn from(x: PF) -> Self {
        Self {
            value: field_to_array(x),
        }
    }
}

impl<PF: PackedField<Scalar = KoalaBear>> Distribution<PackedQuinticExtensionField<PF>>
    for StandardUniform
where
    Self: Distribution<PF>,
{
    #[inline]
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> PackedQuinticExtensionField<PF> {
        PackedQuinticExtensionField::new(array::from_fn(|_| self.sample(rng)))
    }
}

impl<PF: PackedField<Scalar = KoalaBear>> Algebra<QuinticExtensionField<KoalaBear>>
    for PackedQuinticExtensionField<PF>
{
}

impl<PF: PackedField<Scalar = KoalaBear>> Algebra<PF> for PackedQuinticExtensionField<PF> {}

impl<PF: PackedField<Scalar = KoalaBear>> PrimeCharacteristicRing
    for PackedQuinticExtensionField<PF>
where
    PF: PackedField<Scalar = KoalaBear>,
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

    #[inline(always)]
    fn square(&self) -> Self {
        let mut res = Self::default();
        super::extension::quintic_square(&self.value, &mut res.value);
        res
    }

    #[inline]
    fn zero_vec(len: usize) -> Vec<Self> {
        // SAFETY: this is a repr(transparent) wrapper around an array.
        unsafe { reconstitute_from_base(PF::zero_vec(len * 5)) }
    }
}

impl<PF> BasedVectorSpace<PF> for PackedQuinticExtensionField<PF>
where
    PF: PackedField<Scalar = KoalaBear>,
{
    const DIMENSION: usize = 5;

    #[inline]
    fn as_basis_coefficients_slice(&self) -> &[PF] {
        &self.value
    }

    #[inline]
    fn from_basis_coefficients_fn<Fn: FnMut(usize) -> PF>(f: Fn) -> Self {
        Self {
            value: array::from_fn(f),
        }
    }

    #[inline]
    fn from_basis_coefficients_iter<I: ExactSizeIterator<Item = PF>>(mut iter: I) -> Option<Self> {
        (iter.len() == 5).then(|| Self::new(array::from_fn(|_| iter.next().unwrap()))) // The unwrap is safe as we just checked the length of iter.
    }

    #[inline]
    fn flatten_to_base(vec: Vec<Self>) -> Vec<PF> {
        unsafe {
            // Safety:
            // As `Self` is a `repr(transparent)`, it is stored identically in memory to `[PF; D]`
            flatten_to_base(vec)
        }
    }

    #[inline]
    fn reconstitute_from_base(vec: Vec<PF>) -> Vec<Self> {
        unsafe {
            // Safety:
            // As `Self` is a `repr(transparent)`, it is stored identically in memory to `[PF; D]`
            reconstitute_from_base(vec)
        }
    }
}

impl PackedFieldExtension<KoalaBear, QuinticExtensionField<KoalaBear>>
    for PackedQuinticExtensionField<<KoalaBear as Field>::Packing>
{
    #[inline]
    fn from_ext_slice(ext_slice: &[QuinticExtensionField<KoalaBear>]) -> Self {
        let width = <KoalaBear as Field>::Packing::WIDTH;
        assert_eq!(ext_slice.len(), width);

        let res =
            array::from_fn(|i| <KoalaBear as Field>::Packing::from_fn(|j| ext_slice[j].value[i]));
        Self::new(res)
    }

    #[inline]
    fn to_ext_iter(
        iter: impl IntoIterator<Item = Self>,
    ) -> impl Iterator<Item = QuinticExtensionField<KoalaBear>> {
        let width = <KoalaBear as Field>::Packing::WIDTH;
        iter.into_iter().flat_map(move |x| {
            (0..width).map(move |i| {
                let values = array::from_fn(|j| x.value[j].as_slice()[i]);
                QuinticExtensionField::new(values)
            })
        })
    }

    #[inline]
    fn packed_ext_powers(base: QuinticExtensionField<KoalaBear>) -> p3_field::Powers<Self> {
        let width = <KoalaBear as Field>::Packing::WIDTH;
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

impl<PF> Neg for PackedQuinticExtensionField<PF>
where
    PF: PackedField<Scalar = KoalaBear>,
{
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        Self {
            value: self.value.map(PF::neg),
        }
    }
}

impl<PF> Add for PackedQuinticExtensionField<PF>
where
    PF: PackedField<Scalar = KoalaBear>,
{
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        let value = vector_add(&self.value, &rhs.value);
        Self { value }
    }
}

impl<PF> Add<QuinticExtensionField<KoalaBear>> for PackedQuinticExtensionField<PF>
where
    PF: PackedField<Scalar = KoalaBear>,
{
    type Output = Self;

    #[inline]
    fn add(self, rhs: QuinticExtensionField<KoalaBear>) -> Self {
        let value = vector_add(&self.value, &rhs.value);
        Self { value }
    }
}

impl<PF> Add<PF> for PackedQuinticExtensionField<PF>
where
    PF: PackedField<Scalar = KoalaBear>,
{
    type Output = Self;

    #[inline]
    fn add(mut self, rhs: PF) -> Self {
        self.value[0] += rhs;
        self
    }
}

impl<PF> AddAssign for PackedQuinticExtensionField<PF>
where
    PF: PackedField<Scalar = KoalaBear>,
{
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..5 {
            self.value[i] += rhs.value[i];
        }
    }
}

impl<PF> AddAssign<QuinticExtensionField<KoalaBear>> for PackedQuinticExtensionField<PF>
where
    PF: PackedField<Scalar = KoalaBear>,
{
    #[inline]
    fn add_assign(&mut self, rhs: QuinticExtensionField<KoalaBear>) {
        for i in 0..5 {
            self.value[i] += rhs.value[i];
        }
    }
}

impl<PF> AddAssign<PF> for PackedQuinticExtensionField<PF>
where
    PF: PackedField<Scalar = KoalaBear>,
{
    #[inline]
    fn add_assign(&mut self, rhs: PF) {
        self.value[0] += rhs;
    }
}

impl<PF> Sum for PackedQuinticExtensionField<PF>
where
    PF: PackedField<Scalar = KoalaBear>,
{
    #[inline]
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|acc, x| acc + x).unwrap_or(Self::ZERO)
    }
}

impl<PF> Sub for PackedQuinticExtensionField<PF>
where
    PF: PackedField<Scalar = KoalaBear>,
{
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        let value = vector_sub(&self.value, &rhs.value);
        Self { value }
    }
}

impl<PF> Sub<QuinticExtensionField<KoalaBear>> for PackedQuinticExtensionField<PF>
where
    PF: PackedField<Scalar = KoalaBear>,
{
    type Output = Self;

    #[inline]
    fn sub(self, rhs: QuinticExtensionField<KoalaBear>) -> Self {
        let value = vector_sub(&self.value, &rhs.value);
        Self { value }
    }
}

impl<PF> Sub<PF> for PackedQuinticExtensionField<PF>
where
    PF: PackedField<Scalar = KoalaBear>,
{
    type Output = Self;

    #[inline]
    fn sub(self, rhs: PF) -> Self {
        let mut res = self.value;
        res[0] -= rhs;
        Self { value: res }
    }
}

impl<PF> SubAssign for PackedQuinticExtensionField<PF>
where
    PF: PackedField<Scalar = KoalaBear>,
{
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<PF> SubAssign<QuinticExtensionField<KoalaBear>> for PackedQuinticExtensionField<PF>
where
    PF: PackedField<Scalar = KoalaBear>,
{
    #[inline]
    fn sub_assign(&mut self, rhs: QuinticExtensionField<KoalaBear>) {
        *self = *self - rhs;
    }
}

impl<PF> SubAssign<PF> for PackedQuinticExtensionField<PF>
where
    PF: PackedField<Scalar = KoalaBear>,
{
    #[inline]
    fn sub_assign(&mut self, rhs: PF) {
        *self = *self - rhs;
    }
}

impl<PF> Mul for PackedQuinticExtensionField<PF>
where
    PF: PackedField<Scalar = KoalaBear>,
{
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let a = self.value;
        let b = rhs.value;
        let mut res = Self::default();
        super::extension::quintic_mul::<KoalaBear, PF, PF>(&a, &b, &mut res.value);
        res
    }
}

impl<PF> Mul<QuinticExtensionField<KoalaBear>> for PackedQuinticExtensionField<PF>
where
    PF: PackedField<Scalar = KoalaBear>,
{
    type Output = Self;

    #[inline]
    fn mul(self, rhs: QuinticExtensionField<KoalaBear>) -> Self {
        let a = self.value;
        let b = rhs.value;
        let mut res = Self::default();
        super::extension::quintic_mul(&a, &b, &mut res.value);

        res
    }
}

impl<PF> Mul<PF> for PackedQuinticExtensionField<PF>
where
    PF: PackedField<Scalar = KoalaBear>,
{
    type Output = Self;

    #[inline]
    fn mul(self, rhs: PF) -> Self {
        Self {
            value: self.value.map(|x| x * rhs),
        }
    }
}

impl<PF> Product for PackedQuinticExtensionField<PF>
where
    PF: PackedField<Scalar = KoalaBear>,
{
    #[inline]
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|acc, x| acc * x).unwrap_or(Self::ZERO)
    }
}

impl<PF> MulAssign for PackedQuinticExtensionField<PF>
where
    PF: PackedField<Scalar = KoalaBear>,
{
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<PF> MulAssign<QuinticExtensionField<KoalaBear>> for PackedQuinticExtensionField<PF>
where
    PF: PackedField<Scalar = KoalaBear>,
{
    #[inline]
    fn mul_assign(&mut self, rhs: QuinticExtensionField<KoalaBear>) {
        *self = *self * rhs;
    }
}

impl<PF> MulAssign<PF> for PackedQuinticExtensionField<PF>
where
    PF: PackedField<Scalar = KoalaBear>,
{
    #[inline]
    fn mul_assign(&mut self, rhs: PF) {
        *self = *self * rhs;
    }
}
