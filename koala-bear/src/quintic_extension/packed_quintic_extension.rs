use alloc::vec::Vec;
use core::array;
use core::fmt::Debug;
use core::hash::Hash;
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use itertools::Itertools;
use p3_field::extension::{vector_add, vector_sub};

use p3_util::{flatten_to_base, reconstitute_from_base};
use rand::distr::{Distribution, StandardUniform};
use serde::{Deserialize, Serialize};

use p3_field::{
    Algebra, BasedVectorSpace, PackedFieldExtension, PackedValue, Powers, PrimeCharacteristicRing,
    field_to_array,
};

use crate::quintic_extension::{QuinticExtensionField, quintic_square};
use crate::{KoalaBear, kb_quintic_mul};

/* TODO 1:

Why does this does not work? `type KoalaBearPF = <KoalaBear as p3_field::Field>::Packing;`

Currently we need to specify the exact type, but it should be the same type as above, wtf?
*/

#[cfg(not(any(
    all(target_arch = "aarch64", target_feature = "neon"),
    all(target_arch = "x86_64", target_feature = "avx2",)
)))]
type KoalaBearPF = KoalaBear;

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    not(target_feature = "avx512f")
))]
type KoalaBearPF = p3_monty_31::PackedMontyField31AVX2<crate::KoalaBearParameters>;

/* TODO 2:

Why do some of the [derive(...)] here do not work?

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, Serialize, Deserialize, PartialOrd, Ord)]
#[repr(transparent)] // Needed to make various casts safe.
pub struct PackedQuinticExtensionField {
    #[serde(
        with = "p3_util::array_serialization",
        bound(
            serialize = "KoalaBearPF: Serialize",
            deserialize = "KoalaBearPF: Deserialize<'de>"
        )
    )]
    pub(crate) value: [KoalaBearPF; 5],
}

*/

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
#[repr(transparent)] // Needed to make various casts safe.
pub struct PackedQuinticExtensionField {
    pub(crate) value: [KoalaBearPF; 5],
}

impl Serialize for PackedQuinticExtensionField {
    fn serialize<S: serde::Serializer>(&self, _serializer: S) -> Result<S::Ok, S::Error> {
        todo!()
    }
}

impl<'de> Deserialize<'de> for PackedQuinticExtensionField {
    fn deserialize<D: serde::Deserializer<'de>>(_deserializer: D) -> Result<Self, D::Error> {
        todo!()
    }
}

impl Hash for PackedQuinticExtensionField {
    fn hash<H: core::hash::Hasher>(&self, _state: &mut H) {
        todo!()
    }
}

impl Ord for PackedQuinticExtensionField {
    fn cmp(&self, _other: &Self) -> core::cmp::Ordering {
        todo!()
    }
}

impl PartialOrd for PackedQuinticExtensionField {
    fn partial_cmp(&self, _other: &Self) -> Option<core::cmp::Ordering> {
        todo!()
    }
}

impl PackedQuinticExtensionField {
    const fn new(value: [KoalaBearPF; 5]) -> Self {
        Self { value }
    }
}

impl Default for PackedQuinticExtensionField {
    #[inline]
    fn default() -> Self {
        Self {
            value: array::from_fn(|_| KoalaBearPF::ZERO),
        }
    }
}

impl From<QuinticExtensionField> for PackedQuinticExtensionField {
    #[inline]
    fn from(x: QuinticExtensionField) -> Self {
        Self {
            value: x.value.map(Into::into),
        }
    }
}

impl From<KoalaBearPF> for PackedQuinticExtensionField {
    #[inline]
    fn from(x: KoalaBearPF) -> Self {
        Self {
            value: field_to_array(x),
        }
    }
}

impl Distribution<PackedQuinticExtensionField> for StandardUniform
where
    Self: Distribution<KoalaBearPF>,
{
    #[inline]
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> PackedQuinticExtensionField {
        PackedQuinticExtensionField::new(array::from_fn(|_| self.sample(rng)))
    }
}

impl Algebra<QuinticExtensionField> for PackedQuinticExtensionField {}

impl PrimeCharacteristicRing for PackedQuinticExtensionField {
    type PrimeSubfield = KoalaBear;

    const ZERO: Self = Self {
        value: [KoalaBearPF::ZERO; 5],
    };

    const ONE: Self = Self {
        value: field_to_array(KoalaBearPF::ONE),
    };

    const TWO: Self = Self {
        value: field_to_array(KoalaBearPF::TWO),
    };

    const NEG_ONE: Self = Self {
        value: field_to_array(KoalaBearPF::NEG_ONE),
    };

    #[inline]
    fn from_prime_subfield(val: Self::PrimeSubfield) -> Self {
        KoalaBearPF::from_prime_subfield(val).into()
    }

    #[inline]
    fn from_bool(b: bool) -> Self {
        KoalaBearPF::from_bool(b).into()
    }

    #[inline(always)]
    fn square(&self) -> Self {
        let mut res = Self::default();
        quintic_square(&self.value, &mut res.value);
        res
    }

    #[inline]
    fn zero_vec(len: usize) -> Vec<Self> {
        // SAFETY: this is a repr(transparent) wrapper around an array.
        unsafe { reconstitute_from_base(KoalaBearPF::zero_vec(len * 5)) }
    }
}

impl BasedVectorSpace<KoalaBearPF> for PackedQuinticExtensionField {
    const DIMENSION: usize = 5;

    #[inline]
    fn as_basis_coefficients_slice(&self) -> &[KoalaBearPF] {
        &self.value
    }

    #[inline]
    fn from_basis_coefficients_fn<Fn: FnMut(usize) -> KoalaBearPF>(f: Fn) -> Self {
        Self {
            value: array::from_fn(f),
        }
    }

    #[inline]
    fn from_basis_coefficients_iter<I: ExactSizeIterator<Item = KoalaBearPF>>(
        mut iter: I,
    ) -> Option<Self> {
        (iter.len() == 5).then(|| Self::new(array::from_fn(|_| iter.next().unwrap()))) // The unwrap is safe as we just checked the length of iter.
    }

    #[inline]
    fn flatten_to_base(vec: Vec<Self>) -> Vec<KoalaBearPF> {
        unsafe {
            // Safety:
            // As `Self` is a `repr(transparent)`, it is stored identically in memory to `[PF; D]`
            flatten_to_base(vec)
        }
    }

    #[inline]
    fn reconstitute_from_base(vec: Vec<KoalaBearPF>) -> Vec<Self> {
        unsafe {
            // Safety:
            // As `Self` is a `repr(transparent)`, it is stored identically in memory to `[PF; D]`
            reconstitute_from_base(vec)
        }
    }
}

impl Algebra<KoalaBearPF> for PackedQuinticExtensionField {}

impl PackedFieldExtension<KoalaBear, QuinticExtensionField> for PackedQuinticExtensionField {
    #[inline]
    fn from_ext_slice(ext_slice: &[QuinticExtensionField]) -> Self {
        assert_eq!(ext_slice.len(), KoalaBearPF::WIDTH);
        let res = array::from_fn(|i| KoalaBearPF::from_fn(|j| ext_slice[j].value[i]));
        Self::new(res)
    }

    #[inline]
    fn to_ext_iter(
        iter: impl IntoIterator<Item = Self>,
    ) -> impl Iterator<Item = QuinticExtensionField> {
        let width = KoalaBearPF::WIDTH;
        iter.into_iter().flat_map(move |x| {
            (0..width).map(move |i| {
                let values = array::from_fn(|j| x.value[j].as_slice()[i]);
                QuinticExtensionField::new(values)
            })
        })
    }

    #[inline]
    fn packed_ext_powers(base: QuinticExtensionField) -> p3_field::Powers<Self> {
        let width = KoalaBearPF::WIDTH;
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

impl Neg for PackedQuinticExtensionField {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        Self {
            value: self.value.map(KoalaBearPF::neg),
        }
    }
}

impl Add for PackedQuinticExtensionField {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        let value = vector_add(&self.value, &rhs.value);
        Self { value }
    }
}

impl Add<QuinticExtensionField> for PackedQuinticExtensionField {
    type Output = Self;

    #[inline]
    fn add(self, rhs: QuinticExtensionField) -> Self {
        let value = vector_add(&self.value, &rhs.value);
        Self { value }
    }
}

impl Add<KoalaBearPF> for PackedQuinticExtensionField {
    type Output = Self;

    #[inline]
    fn add(mut self, rhs: KoalaBearPF) -> Self {
        self.value[0] += rhs;
        self
    }
}

impl AddAssign for PackedQuinticExtensionField {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..5 {
            self.value[i] += rhs.value[i];
        }
    }
}

impl AddAssign<QuinticExtensionField> for PackedQuinticExtensionField {
    #[inline]
    fn add_assign(&mut self, rhs: QuinticExtensionField) {
        for i in 0..5 {
            self.value[i] += rhs.value[i];
        }
    }
}

impl AddAssign<KoalaBearPF> for PackedQuinticExtensionField {
    #[inline]
    fn add_assign(&mut self, rhs: KoalaBearPF) {
        self.value[0] += rhs;
    }
}

impl Sum for PackedQuinticExtensionField {
    #[inline]
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|acc, x| acc + x).unwrap_or(Self::ZERO)
    }
}

impl Sub for PackedQuinticExtensionField {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        let value = vector_sub(&self.value, &rhs.value);
        Self { value }
    }
}

impl Sub<QuinticExtensionField> for PackedQuinticExtensionField {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: QuinticExtensionField) -> Self {
        let value = vector_sub(&self.value, &rhs.value);
        Self { value }
    }
}

impl Sub<KoalaBearPF> for PackedQuinticExtensionField {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: KoalaBearPF) -> Self {
        let mut res = self.value;
        res[0] -= rhs;
        Self { value: res }
    }
}

impl SubAssign for PackedQuinticExtensionField {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl SubAssign<QuinticExtensionField> for PackedQuinticExtensionField {
    #[inline]
    fn sub_assign(&mut self, rhs: QuinticExtensionField) {
        *self = *self - rhs;
    }
}

impl SubAssign<KoalaBearPF> for PackedQuinticExtensionField {
    #[inline]
    fn sub_assign(&mut self, rhs: KoalaBearPF) {
        *self = *self - rhs;
    }
}

impl Mul for PackedQuinticExtensionField {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let a = self.value;
        let b = rhs.value;
        let mut res = Self::default();
        kb_quintic_mul(&a, &b, &mut res.value);
        res
    }
}

impl Mul<QuinticExtensionField> for PackedQuinticExtensionField {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: QuinticExtensionField) -> Self {
        let a = self.value;
        let b = rhs.value;
        let mut res = Self::default();
        kb_quintic_mul(&a, &b, &mut res.value);

        res
    }
}

impl Mul<KoalaBearPF> for PackedQuinticExtensionField {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: KoalaBearPF) -> Self {
        Self {
            value: self.value.map(|x| x * rhs),
        }
    }
}

impl Product for PackedQuinticExtensionField {
    #[inline]
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|acc, x| acc * x).unwrap_or(Self::ZERO)
    }
}

impl MulAssign for PackedQuinticExtensionField {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl MulAssign<QuinticExtensionField> for PackedQuinticExtensionField {
    #[inline]
    fn mul_assign(&mut self, rhs: QuinticExtensionField) {
        *self = *self * rhs;
    }
}

impl MulAssign<KoalaBearPF> for PackedQuinticExtensionField {
    #[inline]
    fn mul_assign(&mut self, rhs: KoalaBearPF) {
        *self = *self * rhs;
    }
}
