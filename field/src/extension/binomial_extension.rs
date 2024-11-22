use alloc::format;
use alloc::string::ToString;
use alloc::vec::Vec;
use core::array;
use core::fmt::{self, Debug, Display, Formatter};
use core::iter::{Product, Sum};
use core::marker::PhantomData;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use itertools::Itertools;
use num_bigint::BigUint;
use p3_util::convert_vec;
use rand::distributions::Standard;
use rand::prelude::Distribution;
use serde::{Deserialize, Serialize};

use super::{HasFrobenius, HasTwoAdicBionmialExtension};
use crate::extension::BinomiallyExtendable;
use crate::field::Field;
use crate::{
    field_to_array, AbelianGroup, CommutativeRing, ExtensionField, FieldAlgebra,
    InjectiveRingHomomorphism, Packable, PackedFieldExtension, PackedValue, Powers,
    PrimeCharacteristicRing, PrimeField, TwoAdicField,
};

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, Serialize, Deserialize, PartialOrd, Ord)]
#[repr(transparent)] // to make the zero_vec implementation safe
pub struct BinomialExtensionField<F, const D: usize, FA = F> {
    #[serde(
        with = "p3_util::array_serialization",
        bound(serialize = "FA: Serialize", deserialize = "FA: Deserialize<'de>")
    )]
    pub(crate) value: [FA; D],
    _phantom: PhantomData<F>,
}

impl<F, FA, const D: usize> BinomialExtensionField<F, D, FA> {
    pub(crate) const fn new(value: [FA; D]) -> Self {
        Self {
            value,
            _phantom: PhantomData,
        }
    }
}

impl<F: Field, FA: FieldAlgebra<F>, const D: usize> Default for BinomialExtensionField<F, D, FA> {
    fn default() -> Self {
        Self::new(array::from_fn(|_| FA::ZERO))
    }
}

impl<F: Field, FA: FieldAlgebra<F>, const D: usize> From<FA> for BinomialExtensionField<F, D, FA> {
    fn from(x: FA) -> Self {
        Self::new(field_to_array::<FA, D>(x))
    }
}

impl<F: BinomiallyExtendable<D>, const D: usize> Packable for BinomialExtensionField<F, D> {}

impl<F: BinomiallyExtendable<D>, const D: usize> ExtensionField<F>
    for BinomialExtensionField<F, D>
{
    type ExtensionPacking = BinomialExtensionField<F, D, F::Packing>;

    const D: usize = D;

    fn is_in_basefield(&self) -> bool {
        self.value[1..].iter().all(F::is_zero)
    }

    fn as_base(&self) -> Option<F> {
        if self.is_in_basefield() {
            Some(self.value[0])
        } else {
            None
        }
    }
}

impl<F: BinomiallyExtendable<D>, const D: usize> PackedFieldExtension
    for BinomialExtensionField<F, D, F::Packing>
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

impl<F: BinomiallyExtendable<D>, const D: usize> HasFrobenius<F> for BinomialExtensionField<F, D> {
    /// FrobeniusField automorphisms: x -> x^n, where n is the order of BaseField.
    fn frobenius(&self) -> Self {
        self.repeated_frobenius(1)
    }

    /// Repeated Frobenius automorphisms: x -> x^(n^count).
    ///
    /// Follows precomputation suggestion in Section 11.3.3 of the
    /// Handbook of Elliptic and Hyperelliptic Curve Cryptography.
    fn repeated_frobenius(&self, count: usize) -> Self {
        if count == 0 {
            return *self;
        } else if count >= D {
            // x |-> x^(n^D) is the identity, so x^(n^count) ==
            // x^(n^(count % D))
            return self.repeated_frobenius(count % D);
        }

        // z0 = DTH_ROOT^count = W^(k * count) where k = floor((n-1)/D)
        let mut z0 = F::DTH_ROOT;
        for _ in 1..count {
            z0 *= F::DTH_ROOT;
        }

        let mut res = [F::ZERO; D];
        for (i, z) in z0.powers().take(D).enumerate() {
            res[i] = self.value[i] * z;
        }

        Self::new(res)
    }

    /// Algorithm 11.3.4 in Handbook of Elliptic and Hyperelliptic Curve Cryptography.
    fn frobenius_inv(&self) -> Self {
        // Writing 'a' for self, we need to compute a^(r-1):
        // r = n^D-1/n-1 = n^(D-1)+n^(D-2)+...+n
        let mut f = Self::ONE;
        for _ in 1..D {
            f = (f * *self).frobenius();
        }

        // g = a^r is in the base field, so only compute that
        // coefficient rather than the full product.
        let a = self.value;
        let b = f.value;
        let mut g = F::ZERO;
        for i in 1..D {
            g += a[i] * b[D - i];
        }
        g *= F::W;
        g += a[0] * b[0];
        debug_assert_eq!(Self::from(g), *self * f);

        f * g.inverse()
    }
}

impl<F, FA, const D: usize> AbelianGroup for BinomialExtensionField<F, D, FA>
where
    F: BinomiallyExtendable<D>,
    FA: FieldAlgebra<F>,
{
    const ZERO: Self = Self::new([FA::ZERO; D]);

    fn mul_u64(&self, r: u64) -> Self {
        self.clone() * Self::from_u64(r)
    }

    fn mul_2exp_u64(&self, exp: u64) -> Self {
        let pow = FA::Char::TWO.exp_u64(exp);
        self.clone() * FA::from_char(pow)
    }
}

impl<F, FA, const D: usize> CommutativeRing for BinomialExtensionField<F, D, FA>
where
    F: BinomiallyExtendable<D>,
    FA: FieldAlgebra<F>,
{
    const ONE: Self = Self::new(field_to_array::<FA, D>(FA::ONE));

    const NEG_ONE: Self = Self::new(field_to_array::<FA, D>(FA::NEG_ONE));

    #[inline(always)]
    fn square(&self) -> Self {
        match D {
            2 => {
                let a = self.value.clone();
                let mut res = Self::default();
                res.value[0] = a[0].square() + a[1].square() * FA::from_f(F::W);
                res.value[1] = a[0].clone() * a[1].double();
                res
            }
            3 => {
                let mut res = Self::default();
                cubic_square(&self.value, &mut res.value);
                res
            }
            _ => <Self as Mul<Self>>::mul(self.clone(), self.clone()),
        }
    }

    #[inline]
    fn zero_vec(len: usize) -> Vec<Self> {
        // SAFETY: this is a repr(transparent) wrapper around an array.
        unsafe { convert_vec(FA::zero_vec(len * D)) }
    }
}

impl<F, FA, const D: usize> PrimeCharacteristicRing for BinomialExtensionField<F, D, FA>
where
    F: BinomiallyExtendable<D>,
    FA: FieldAlgebra<F>,
{
    type Char = FA::Char;

    fn from_char(f: Self::Char) -> Self {
        FA::from_char(f).into()
    }

    fn halve(&self) -> Self {
        Self::new(self.value.clone().map(|x| x.halve()))
    }

    fn div_2exp_u64(&self, _exp: u64) -> Self {
        todo!()
    }
}

impl<F, FA, const D: usize> InjectiveRingHomomorphism<F> for BinomialExtensionField<F, D, FA>
where
    F: BinomiallyExtendable<D>,
    FA: FieldAlgebra<F>,
{
    #[inline]
    fn from_f(f: F) -> Self {
        FA::from_f(f).into()
    }
}

impl<F, FA, const D: usize> FieldAlgebra<F> for BinomialExtensionField<F, D, FA>
where
    F: BinomiallyExtendable<D>,
    FA: FieldAlgebra<F>,
{
}

impl<F, const D: usize> InjectiveRingHomomorphism<BinomialExtensionField<F, D>>
    for BinomialExtensionField<F, D>
where
    F: BinomiallyExtendable<D>,
{
    #[inline]
    fn from_f(f: BinomialExtensionField<F, D>) -> Self {
        f
    }
}

impl<F: BinomiallyExtendable<D>, const D: usize> FieldAlgebra<BinomialExtensionField<F, D>>
    for BinomialExtensionField<F, D>
{
}

impl<F: BinomiallyExtendable<D>, const D: usize> Field for BinomialExtensionField<F, D> {
    type Packing = Self;

    const GENERATOR: Self = Self::new(F::EXT_GENERATOR);

    fn try_inverse(&self) -> Option<Self> {
        if self.is_zero() {
            return None;
        }

        let mut res = Self::default();

        match D {
            2 => qudratic_inv(&self.value, &mut res.value),
            3 => cubic_inv(&self.value, &mut res.value),
            _ => res = self.frobenius_inv(),
        }

        Some(res)
    }

    fn order() -> BigUint {
        F::order().pow(D as u32)
    }
}

impl<F, const D: usize> Display for BinomialExtensionField<F, D>
where
    F: BinomiallyExtendable<D>,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        if self.is_zero() {
            write!(f, "0")
        } else {
            let str = self
                .value
                .iter()
                .enumerate()
                .filter(|(_, x)| !x.is_zero())
                .map(|(i, x)| match (i, x.is_one()) {
                    (0, _) => format!("{x}"),
                    (1, true) => "X".to_string(),
                    (1, false) => format!("{x} X"),
                    (_, true) => format!("X^{i}"),
                    (_, false) => format!("{x} X^{i}"),
                })
                .join(" + ");
            write!(f, "{}", str)
        }
    }
}

impl<F, FA, const D: usize> Neg for BinomialExtensionField<F, D, FA>
where
    F: BinomiallyExtendable<D>,
    FA: FieldAlgebra<F>,
{
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        Self::new(self.value.map(FA::neg))
    }
}

impl<F, FA, const D: usize> Add for BinomialExtensionField<F, D, FA>
where
    F: BinomiallyExtendable<D>,
    FA: FieldAlgebra<F>,
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

impl<F, FA, const D: usize> Add<FA> for BinomialExtensionField<F, D, FA>
where
    F: BinomiallyExtendable<D>,
    FA: FieldAlgebra<F>,
{
    type Output = Self;

    #[inline]
    fn add(mut self, rhs: FA) -> Self {
        self.value[0] += rhs;
        self
    }
}

impl<F, FA, const D: usize> AddAssign for BinomialExtensionField<F, D, FA>
where
    F: BinomiallyExtendable<D>,
    FA: FieldAlgebra<F>,
{
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..D {
            self.value[i] += rhs.value[i].clone();
        }
    }
}

impl<F, FA, const D: usize> AddAssign<FA> for BinomialExtensionField<F, D, FA>
where
    F: BinomiallyExtendable<D>,
    FA: FieldAlgebra<F>,
{
    #[inline]
    fn add_assign(&mut self, rhs: FA) {
        self.value[0] += rhs;
    }
}

impl<F, FA, const D: usize> Sum for BinomialExtensionField<F, D, FA>
where
    F: BinomiallyExtendable<D>,
    FA: FieldAlgebra<F>,
{
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |acc, x| acc + x)
    }
}

impl<F, FA, const D: usize> Sub for BinomialExtensionField<F, D, FA>
where
    F: BinomiallyExtendable<D>,
    FA: FieldAlgebra<F>,
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

impl<F, FA, const D: usize> Sub<FA> for BinomialExtensionField<F, D, FA>
where
    F: BinomiallyExtendable<D>,
    FA: FieldAlgebra<F>,
{
    type Output = Self;

    #[inline]
    fn sub(self, rhs: FA) -> Self {
        let mut res = self.value;
        res[0] -= rhs;
        Self::new(res)
    }
}

impl<F, FA, const D: usize> SubAssign for BinomialExtensionField<F, D, FA>
where
    F: BinomiallyExtendable<D>,
    FA: FieldAlgebra<F>,
{
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = self.clone() - rhs;
    }
}

impl<F, FA, const D: usize> SubAssign<FA> for BinomialExtensionField<F, D, FA>
where
    F: BinomiallyExtendable<D>,
    FA: FieldAlgebra<F>,
{
    #[inline]
    fn sub_assign(&mut self, rhs: FA) {
        *self = self.clone() - rhs;
    }
}

impl<F, FA, const D: usize> Mul for BinomialExtensionField<F, D, FA>
where
    F: BinomiallyExtendable<D>,
    FA: FieldAlgebra<F>,
{
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let a = self.value;
        let b = rhs.value;
        let mut res = Self::default();
        let w = F::W;
        let w_af = FA::from_f(w);

        match D {
            2 => {
                res.value[0] = a[0].clone() * b[0].clone() + a[1].clone() * w_af * b[1].clone();
                res.value[1] = a[0].clone() * b[1].clone() + a[1].clone() * b[0].clone();
            }
            3 => cubic_mul(&a, &b, &mut res.value),
            _ =>
            {
                #[allow(clippy::needless_range_loop)]
                for i in 0..D {
                    for j in 0..D {
                        if i + j >= D {
                            res.value[i + j - D] += a[i].clone() * w_af.clone() * b[j].clone();
                        } else {
                            res.value[i + j] += a[i].clone() * b[j].clone();
                        }
                    }
                }
            }
        }
        res
    }
}

impl<F, FA, const D: usize> Mul<FA> for BinomialExtensionField<F, D, FA>
where
    F: BinomiallyExtendable<D>,
    FA: FieldAlgebra<F>,
{
    type Output = Self;

    #[inline]
    fn mul(self, rhs: FA) -> Self {
        Self::new(self.value.map(|x| x * rhs.clone()))
    }
}

impl<F, FA, const D: usize> Product for BinomialExtensionField<F, D, FA>
where
    F: BinomiallyExtendable<D>,
    FA: FieldAlgebra<F>,
{
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ONE, |acc, x| acc * x)
    }
}

impl<F, const D: usize> Div for BinomialExtensionField<F, D>
where
    F: BinomiallyExtendable<D>,
{
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        self * rhs.inverse()
    }
}

impl<F, const D: usize> DivAssign for BinomialExtensionField<F, D>
where
    F: BinomiallyExtendable<D>,
{
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl<F, FA, const D: usize> MulAssign for BinomialExtensionField<F, D, FA>
where
    F: BinomiallyExtendable<D>,
    FA: FieldAlgebra<F>,
{
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.clone() * rhs;
    }
}

impl<F, FA, const D: usize> MulAssign<FA> for BinomialExtensionField<F, D, FA>
where
    F: BinomiallyExtendable<D>,
    FA: FieldAlgebra<F>,
{
    #[inline]
    fn mul_assign(&mut self, rhs: FA) {
        *self = self.clone() * rhs;
    }
}

impl<F: BinomiallyExtendable<D>, const D: usize> Distribution<BinomialExtensionField<F, D>>
    for Standard
where
    Standard: Distribution<F>,
{
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> BinomialExtensionField<F, D> {
        let mut res = [F::ZERO; D];
        for r in res.iter_mut() {
            *r = Standard.sample(rng);
        }
        BinomialExtensionField::<F, D>::new(res)
    }
}

impl<F: Field + HasTwoAdicBionmialExtension<D>, const D: usize> TwoAdicField
    for BinomialExtensionField<F, D>
{
    const TWO_ADICITY: usize = F::EXT_TWO_ADICITY;

    #[inline]
    fn two_adic_generator(bits: usize) -> Self {
        Self::new(F::ext_two_adic_generator(bits))
    }
}

///Section 11.3.6b in Handbook of Elliptic and Hyperelliptic Curve Cryptography.
#[inline]
fn qudratic_inv<F: BinomiallyExtendable<D>, const D: usize>(a: &[F; D], res: &mut [F; D]) {
    assert_eq!(D, 2);

    let scalar = (a[0].square() - F::W * a[1].square()).inverse();
    res[0] = a[0] * scalar;
    res[1] = -a[1] * scalar;
}

/// Section 11.3.6b in Handbook of Elliptic and Hyperelliptic Curve Cryptography.
#[inline]
fn cubic_inv<F: BinomiallyExtendable<D>, const D: usize>(a: &[F; D], res: &mut [F; D]) {
    assert_eq!(D, 3);

    let a0_square = a[0].square();
    let a1_square = a[1].square();
    let a2_w = F::W * a[2];
    let a0_a1 = a[0] * a[1];

    // scalar = (a0^3+wa1^3+w^2a2^3-3wa0a1a2)^-1
    let scalar = (a0_square * a[0] + F::W * a[1] * a1_square + a2_w.square() * a[2]
        - F::from_char(F::Char::ONE + F::Char::TWO) * a2_w * a0_a1)
        .inverse();

    //scalar*[a0^2-wa1a2, wa2^2-a0a1, a1^2-a0a2]
    res[0] = scalar * (a0_square - a[1] * a2_w);
    res[1] = scalar * (a2_w * a[2] - a0_a1);
    res[2] = scalar * (a1_square - a[0] * a[2]);
}

/// karatsuba multiplication for cubic extension field
#[inline]
fn cubic_mul<F: BinomiallyExtendable<D>, FA: FieldAlgebra<F>, const D: usize>(
    a: &[FA; D],
    b: &[FA; D],
    res: &mut [FA; D],
) {
    assert_eq!(D, 3);

    let a0_b0 = a[0].clone() * b[0].clone();
    let a1_b1 = a[1].clone() * b[1].clone();
    let a2_b2 = a[2].clone() * b[2].clone();

    res[0] = a0_b0.clone()
        + ((a[1].clone() + a[2].clone()) * (b[1].clone() + b[2].clone())
            - a1_b1.clone()
            - a2_b2.clone())
            * FA::from_f(F::W);
    res[1] = (a[0].clone() + a[1].clone()) * (b[0].clone() + b[1].clone())
        - a0_b0.clone()
        - a1_b1.clone()
        + a2_b2.clone() * FA::from_f(F::W);
    res[2] = (a[0].clone() + a[2].clone()) * (b[0].clone() + b[2].clone()) - a0_b0 - a2_b2 + a1_b1;
}

/// Section 11.3.6a in Handbook of Elliptic and Hyperelliptic Curve Cryptography.
#[inline]
fn cubic_square<F: BinomiallyExtendable<D>, FA: FieldAlgebra<F>, const D: usize>(
    a: &[FA; D],
    res: &mut [FA; D],
) {
    assert_eq!(D, 3);

    let w_a2 = a[2].clone() * FA::from_f(F::W);

    res[0] = a[0].square() + (a[1].clone() * w_a2.clone()).double();
    res[1] = w_a2 * a[2].clone() + (a[0].clone() * a[1].clone()).double();
    res[2] = a[1].square() + (a[0].clone() * a[2].clone()).double();
}
