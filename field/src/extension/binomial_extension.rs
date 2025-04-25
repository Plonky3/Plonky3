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
use p3_util::{flatten_to_base, reconstitute_from_base};
use rand::distr::StandardUniform;
use rand::prelude::Distribution;
use serde::{Deserialize, Serialize};

use super::{HasFrobenius, HasTwoAdicBinomialExtension, PackedBinomialExtensionField};
use crate::extension::BinomiallyExtendable;
use crate::field::Field;
use crate::{
    Algebra, BasedVectorSpace, ExtensionField, Packable, PrimeCharacteristicRing,
    RawDataSerializable, TwoAdicField, field_to_array,
};

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, Serialize, Deserialize, PartialOrd, Ord)]
#[repr(transparent)] // Needed to make various casts safe.
pub struct BinomialExtensionField<F, const D: usize, A = F> {
    #[serde(
        with = "p3_util::array_serialization",
        bound(serialize = "A: Serialize", deserialize = "A: Deserialize<'de>")
    )]
    pub(crate) value: [A; D],
    _phantom: PhantomData<F>,
}

impl<F, A, const D: usize> BinomialExtensionField<F, D, A> {
    pub(crate) const fn new(value: [A; D]) -> Self {
        Self {
            value,
            _phantom: PhantomData,
        }
    }
}

impl<F: Field, A: Algebra<F>, const D: usize> Default for BinomialExtensionField<F, D, A> {
    fn default() -> Self {
        Self::new(array::from_fn(|_| A::ZERO))
    }
}

impl<F: Field, A: Algebra<F>, const D: usize> From<A> for BinomialExtensionField<F, D, A> {
    fn from(x: A) -> Self {
        Self::new(field_to_array(x))
    }
}

impl<F: BinomiallyExtendable<D>, const D: usize> Packable for BinomialExtensionField<F, D> {}

impl<F: BinomiallyExtendable<D>, A: Algebra<F>, const D: usize> BasedVectorSpace<A>
    for BinomialExtensionField<F, D, A>
{
    const DIMENSION: usize = D;

    #[inline]
    fn as_basis_coefficients_slice(&self) -> &[A] {
        &self.value
    }

    #[inline]
    fn from_basis_coefficients_fn<Fn: FnMut(usize) -> A>(f: Fn) -> Self {
        Self::new(array::from_fn(f))
    }

    #[inline]
    fn from_basis_coefficients_iter<I: ExactSizeIterator<Item = A>>(mut iter: I) -> Option<Self> {
        (iter.len() == D).then(|| Self::new(array::from_fn(|_| iter.next().unwrap()))) // The unwrap is safe as we just checked the length of iter.
    }

    #[inline]
    fn flatten_to_base(vec: Vec<Self>) -> Vec<A> {
        unsafe {
            // Safety:
            // As `Self` is a `repr(transparent)`, it is stored identically in memory to `[A; D]`
            flatten_to_base(vec)
        }
    }

    #[inline]
    fn reconstitute_from_base(vec: Vec<A>) -> Vec<Self> {
        unsafe {
            // Safety:
            // As `Self` is a `repr(transparent)`, it is stored identically in memory to `[A; D]`
            reconstitute_from_base(vec)
        }
    }
}

impl<F: BinomiallyExtendable<D>, const D: usize> ExtensionField<F>
    for BinomialExtensionField<F, D>
{
    type ExtensionPacking = PackedBinomialExtensionField<F, F::Packing, D>;

    #[inline]
    fn is_in_basefield(&self) -> bool {
        self.value[1..].iter().all(F::is_zero)
    }

    #[inline]
    fn as_base(&self) -> Option<F> {
        <Self as ExtensionField<F>>::is_in_basefield(self).then(|| self.value[0])
    }
}

impl<F: BinomiallyExtendable<D>, const D: usize> HasFrobenius<F> for BinomialExtensionField<F, D> {
    /// FrobeniusField automorphisms: x -> x^n, where n is the order of BaseField.
    #[inline]
    fn frobenius(&self) -> Self {
        self.repeated_frobenius(1)
    }

    /// Repeated Frobenius automorphisms: x -> x^(n^count).
    ///
    /// Follows precomputation suggestion in Section 11.3.3 of the
    /// Handbook of Elliptic and Hyperelliptic Curve Cryptography.
    #[inline]
    fn repeated_frobenius(&self, count: usize) -> Self {
        if count == 0 {
            return *self;
        } else if count >= D {
            // x |-> x^(n^D) is the identity, so x^(n^count) ==
            // x^(n^(count % D))
            return self.repeated_frobenius(count % D);
        }

        // z0 = DTH_ROOT^count = W^(k * count) where k = floor((n-1)/D)
        let z0 = F::DTH_ROOT.exp_u64(count as u64);

        let mut res = Self::ZERO;
        for (i, z) in z0.powers().take(D).enumerate() {
            res.value[i] = self.value[i] * z;
        }

        res
    }

    /// Algorithm 11.3.4 in Handbook of Elliptic and Hyperelliptic Curve Cryptography.
    #[inline]
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

impl<F, A, const D: usize> PrimeCharacteristicRing for BinomialExtensionField<F, D, A>
where
    F: BinomiallyExtendable<D>,
    A: Algebra<F>,
{
    type PrimeSubfield = <A as PrimeCharacteristicRing>::PrimeSubfield;

    const ZERO: Self = Self::new([A::ZERO; D]);

    const ONE: Self = Self::new(field_to_array(A::ONE));

    const TWO: Self = Self::new(field_to_array(A::TWO));

    const NEG_ONE: Self = Self::new(field_to_array(A::NEG_ONE));

    #[inline]
    fn from_prime_subfield(f: Self::PrimeSubfield) -> Self {
        <A as PrimeCharacteristicRing>::from_prime_subfield(f).into()
    }

    #[inline(always)]
    fn square(&self) -> Self {
        match D {
            2 => {
                let a = self.value.clone();
                let mut res = Self::default();
                res.value[0] = a[0].square() + a[1].square() * F::W;
                res.value[1] = a[0].clone() * a[1].double();
                res
            }
            3 => {
                let mut res = Self::default();
                cubic_square(&self.value, &mut res.value);
                res
            }
            _ => self.clone() * self.clone(),
        }
    }

    #[inline]
    fn mul_2exp_u64(&self, exp: u64) -> Self {
        Self::new(self.value.clone().map(|x| x.mul_2exp_u64(exp)))
    }

    #[inline]
    fn zero_vec(len: usize) -> Vec<Self> {
        // SAFETY: this is a repr(transparent) wrapper around an array.
        unsafe { reconstitute_from_base(F::zero_vec(len * D)) }
    }
}

impl<F: BinomiallyExtendable<D>, const D: usize> Algebra<F> for BinomialExtensionField<F, D> {}

impl<F: BinomiallyExtendable<D>, const D: usize> RawDataSerializable
    for BinomialExtensionField<F, D>
{
    const NUM_BYTES: usize = F::NUM_BYTES * D;

    #[inline]
    fn into_bytes(self) -> impl IntoIterator<Item = u8> {
        self.value.into_iter().flat_map(|x| x.into_bytes())
    }

    #[inline]
    fn into_byte_stream(input: impl IntoIterator<Item = Self>) -> impl IntoIterator<Item = u8> {
        F::into_byte_stream(input.into_iter().flat_map(|x| x.value))
    }

    #[inline]
    fn into_u32_stream(input: impl IntoIterator<Item = Self>) -> impl IntoIterator<Item = u32> {
        F::into_u32_stream(input.into_iter().flat_map(|x| x.value))
    }

    #[inline]
    fn into_u64_stream(input: impl IntoIterator<Item = Self>) -> impl IntoIterator<Item = u64> {
        F::into_u64_stream(input.into_iter().flat_map(|x| x.value))
    }

    #[inline]
    fn into_parallel_byte_streams<const N: usize>(
        input: impl IntoIterator<Item = [Self; N]>,
    ) -> impl IntoIterator<Item = [u8; N]> {
        F::into_parallel_byte_streams(
            input
                .into_iter()
                .flat_map(|x| (0..D).map(move |i| array::from_fn(|j| x[j].value[i]))),
        )
    }

    #[inline]
    fn into_parallel_u32_streams<const N: usize>(
        input: impl IntoIterator<Item = [Self; N]>,
    ) -> impl IntoIterator<Item = [u32; N]> {
        F::into_parallel_u32_streams(
            input
                .into_iter()
                .flat_map(|x| (0..D).map(move |i| array::from_fn(|j| x[j].value[i]))),
        )
    }

    #[inline]
    fn into_parallel_u64_streams<const N: usize>(
        input: impl IntoIterator<Item = [Self; N]>,
    ) -> impl IntoIterator<Item = [u64; N]> {
        F::into_parallel_u64_streams(
            input
                .into_iter()
                .flat_map(|x| (0..D).map(move |i| array::from_fn(|j| x[j].value[i]))),
        )
    }
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
            2 => quadratic_inv(&self.value, &mut res.value, F::W),
            3 => cubic_inv(&self.value, &mut res.value, F::W),
            _ => res = self.frobenius_inv(),
        }

        Some(res)
    }

    #[inline]
    fn halve(&self) -> Self {
        Self::new(self.value.map(|x| x.halve()))
    }

    #[inline]
    fn div_2exp_u64(&self, exp: u64) -> Self {
        Self::new(self.value.map(|x| x.div_2exp_u64(exp)))
    }

    #[inline]
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

impl<F, A, const D: usize> Neg for BinomialExtensionField<F, D, A>
where
    F: BinomiallyExtendable<D>,
    A: Algebra<F>,
{
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        Self::new(self.value.map(A::neg))
    }
}

impl<F, A, const D: usize> Add for BinomialExtensionField<F, D, A>
where
    F: BinomiallyExtendable<D>,
    A: Algebra<F>,
{
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        let value = vector_add(&self.value, &rhs.value);
        Self::new(value)
    }
}

impl<F, A, const D: usize> Add<A> for BinomialExtensionField<F, D, A>
where
    F: BinomiallyExtendable<D>,
    A: Algebra<F>,
{
    type Output = Self;

    #[inline]
    fn add(mut self, rhs: A) -> Self {
        self.value[0] += rhs;
        self
    }
}

impl<F, A, const D: usize> AddAssign for BinomialExtensionField<F, D, A>
where
    F: BinomiallyExtendable<D>,
    A: Algebra<F>,
{
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..D {
            self.value[i] += rhs.value[i].clone();
        }
    }
}

impl<F, A, const D: usize> AddAssign<A> for BinomialExtensionField<F, D, A>
where
    F: BinomiallyExtendable<D>,
    A: Algebra<F>,
{
    #[inline]
    fn add_assign(&mut self, rhs: A) {
        self.value[0] += rhs;
    }
}

impl<F, A, const D: usize> Sum for BinomialExtensionField<F, D, A>
where
    F: BinomiallyExtendable<D>,
    A: Algebra<F>,
{
    #[inline]
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|acc, x| acc + x).unwrap_or(Self::ZERO)
    }
}

impl<F, A, const D: usize> Sub for BinomialExtensionField<F, D, A>
where
    F: BinomiallyExtendable<D>,
    A: Algebra<F>,
{
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        let value = vector_sub(&self.value, &rhs.value);
        Self::new(value)
    }
}

impl<F, A, const D: usize> Sub<A> for BinomialExtensionField<F, D, A>
where
    F: BinomiallyExtendable<D>,
    A: Algebra<F>,
{
    type Output = Self;

    #[inline]
    fn sub(self, rhs: A) -> Self {
        let mut res = self.value;
        res[0] -= rhs;
        Self::new(res)
    }
}

impl<F, A, const D: usize> SubAssign for BinomialExtensionField<F, D, A>
where
    F: BinomiallyExtendable<D>,
    A: Algebra<F>,
{
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        for i in 0..D {
            self.value[i] -= rhs.value[i].clone();
        }
    }
}

impl<F, A, const D: usize> SubAssign<A> for BinomialExtensionField<F, D, A>
where
    F: BinomiallyExtendable<D>,
    A: Algebra<F>,
{
    #[inline]
    fn sub_assign(&mut self, rhs: A) {
        self.value[0] -= rhs;
    }
}

impl<F, A, const D: usize> Mul for BinomialExtensionField<F, D, A>
where
    F: BinomiallyExtendable<D>,
    A: Algebra<F>,
{
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let mut res = Self::default();
        binomial_mul(&self.value, &rhs.value, &mut res.value, F::W);
        res
    }
}

impl<F, A, const D: usize> Mul<A> for BinomialExtensionField<F, D, A>
where
    F: BinomiallyExtendable<D>,
    A: Algebra<F>,
{
    type Output = Self;

    #[inline]
    fn mul(self, rhs: A) -> Self {
        Self::new(self.value.map(|x| x * rhs.clone()))
    }
}

impl<F, A, const D: usize> MulAssign for BinomialExtensionField<F, D, A>
where
    F: BinomiallyExtendable<D>,
    A: Algebra<F>,
{
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.clone() * rhs;
    }
}

impl<F, A, const D: usize> MulAssign<A> for BinomialExtensionField<F, D, A>
where
    F: BinomiallyExtendable<D>,
    A: Algebra<F>,
{
    #[inline]
    fn mul_assign(&mut self, rhs: A) {
        *self = self.clone() * rhs;
    }
}

impl<F, A, const D: usize> Product for BinomialExtensionField<F, D, A>
where
    F: BinomiallyExtendable<D>,
    A: Algebra<F>,
{
    #[inline]
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|acc, x| acc * x).unwrap_or(Self::ONE)
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

impl<F: BinomiallyExtendable<D>, const D: usize> Distribution<BinomialExtensionField<F, D>>
    for StandardUniform
where
    Self: Distribution<F>,
{
    #[inline]
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> BinomialExtensionField<F, D> {
        BinomialExtensionField::new(array::from_fn(|_| self.sample(rng)))
    }
}

impl<F: Field + HasTwoAdicBinomialExtension<D>, const D: usize> TwoAdicField
    for BinomialExtensionField<F, D>
{
    const TWO_ADICITY: usize = F::EXT_TWO_ADICITY;

    #[inline]
    fn two_adic_generator(bits: usize) -> Self {
        Self::new(F::ext_two_adic_generator(bits))
    }
}

/// Add two vectors element wise.
#[inline]
pub(crate) fn vector_add<
    R: PrimeCharacteristicRing + Add<R2, Output = R>,
    R2: Clone,
    const D: usize,
>(
    a: &[R; D],
    b: &[R2; D],
) -> [R; D] {
    array::from_fn(|i| a[i].clone() + b[i].clone())
}

/// Subtract two vectors element wise.
#[inline]
pub(crate) fn vector_sub<
    R: PrimeCharacteristicRing + Sub<R2, Output = R>,
    R2: Clone,
    const D: usize,
>(
    a: &[R; D],
    b: &[R2; D],
) -> [R; D] {
    array::from_fn(|i| a[i].clone() - b[i].clone())
}

/// Multiply two vectors representing elements in a binomial extension.
#[inline]
pub(super) fn binomial_mul<
    F: Field,
    R: Algebra<F> + Mul<R2, Output = R>,
    R2: Algebra<F> + Add<Output = R2> + Clone,
    const D: usize,
>(
    a: &[R; D],
    b: &[R2; D],
    res: &mut [R; D],
    w: F,
) {
    match D {
        2 => {
            // Karatsuba multiplication for two-term polynomials (degree 2 extension).
            //
            // Let the elements be:
            // ```text
            //     A = a0 + a1·X
            //     B = b0 + b1·X
            // ```
            //
            // The multiplication follows:
            // ```text
            //     A·B = a0·b0 + (a0·b1 + a1·b0)·X + a1·b1·X²
            // ```
            // and since X² ≡ w (mod X² - w), the X² term folds into the constant term.

            // Perform simple Karatsuba:
            // - v0: X⁰ term = a0·b0
            // - v1: X¹ term = a0·b1 + a1·b0
            // - v2: X² term = a1·b1
            let (v0, v1, v2) = karatsuba(a[0].clone(), a[1].clone(), b[0].clone(), b[1].clone());

            // Compute result:
            // Coefficient of X⁰: v0 + w·v2
            res[0] = v0 + v2.clone() * w;

            // Coefficient of X¹: v1
            res[1] = v1;
        }
        3 => cubic_mul(a, b, res, w),
        4 => quartic_karatsuba_mul(a, b, res, w),
        _ => {
            for i in 0..D {
                for j in 0..D {
                    if i + j >= D {
                        res[i + j - D] += a[i].clone() * w * b[j].clone();
                    } else {
                        res[i + j] += a[i].clone() * b[j].clone();
                    }
                }
            }
        }
    }
}

/// Section 11.3.6b in Handbook of Elliptic and Hyperelliptic Curve Cryptography.
#[inline]
fn quadratic_inv<F: Field, const D: usize>(a: &[F; D], res: &mut [F; D], w: F) {
    assert_eq!(D, 2);
    let scalar = (a[0].square() - w * a[1].square()).inverse();
    res[0] = a[0] * scalar;
    res[1] = -a[1] * scalar;
}

/// Section 11.3.6b in Handbook of Elliptic and Hyperelliptic Curve Cryptography.
#[inline]
fn cubic_inv<F: Field, const D: usize>(a: &[F; D], res: &mut [F; D], w: F) {
    assert_eq!(D, 3);
    let a0_square = a[0].square();
    let a1_square = a[1].square();
    let a2_w = w * a[2];
    let a0_a1 = a[0] * a[1];

    // scalar = (a0^3+wa1^3+w^2a2^3-3wa0a1a2)^-1
    let scalar = (a0_square * a[0] + w * a[1] * a1_square + a2_w.square() * a[2]
        - (F::ONE + F::TWO) * a2_w * a0_a1)
        .inverse();

    //scalar*[a0^2-wa1a2, wa2^2-a0a1, a1^2-a0a2]
    res[0] = scalar * (a0_square - a[1] * a2_w);
    res[1] = scalar * (a2_w * a[2] - a0_a1);
    res[2] = scalar * (a1_square - a[0] * a[2]);
}

/// karatsuba multiplication for cubic extension field
#[inline]
pub(crate) fn cubic_mul<
    F: Field,
    R: Algebra<F> + Mul<R2, Output = R>,
    R2: Add<Output = R2> + Clone,
    const D: usize,
>(
    a: &[R; D],
    b: &[R2; D],
    res: &mut [R; D],
    w: F,
) {
    assert_eq!(D, 3);

    let a0_b0 = a[0].clone() * b[0].clone();
    let a1_b1 = a[1].clone() * b[1].clone();
    let a2_b2 = a[2].clone() * b[2].clone();

    res[0] = a0_b0.clone()
        + ((a[1].clone() + a[2].clone()) * (b[1].clone() + b[2].clone())
            - a1_b1.clone()
            - a2_b2.clone())
            * w;
    res[1] = (a[0].clone() + a[1].clone()) * (b[0].clone() + b[1].clone())
        - a0_b0.clone()
        - a1_b1.clone()
        + a2_b2.clone() * w;
    res[2] = (a[0].clone() + a[2].clone()) * (b[0].clone() + b[2].clone()) - a0_b0 - a2_b2 + a1_b1;
}

/// Section 11.3.6a in Handbook of Elliptic and Hyperelliptic Curve Cryptography.
#[inline]
pub(crate) fn cubic_square<F: BinomiallyExtendable<D>, A: Algebra<F>, const D: usize>(
    a: &[A; D],
    res: &mut [A; D],
) {
    assert_eq!(D, 3);

    let w_a2 = a[2].clone() * F::W;

    res[0] = a[0].square() + (a[1].clone() * w_a2.clone()).double();
    res[1] = w_a2 * a[2].clone() + (a[0].clone() * a[1].clone()).double();
    res[2] = a[1].square() + (a[0].clone() * a[2].clone()).double();
}

/// Perform a simple Karatsuba multiplication for two degree-1 polynomials.
///
/// Given:
/// ```text
///     A = a0 + a1·X
///     B = b0 + b1·X
/// ```
/// computes and returns:
/// ```text
///     (a0·b0, a0·b1 + a1·b0, a1·b1)
/// ```
/// representing coefficients for X⁰, X¹, and X² respectively.
#[inline]
fn karatsuba<R, R2>(a0: R, a1: R, b0: R2, b1: R2) -> (R, R, R)
where
    R: Add<Output = R> + Sub<Output = R> + Mul<R2, Output = R> + Clone,
    R2: Add<Output = R2> + Clone,
{
    // Compute p0 = a0·b0
    let p0 = a0.clone() * b0.clone();
    // Compute p2 = a1·b1
    let p2 = a1.clone() * b1.clone();
    // Compute p1 = (a0 + a1)·(b0 + b1) - p0 - p2
    let p1 = (a0 + a1) * (b0 + b1) - p0.clone() - p2.clone();
    (p0, p1, p2)
}

/// Optimized Karatsuba multiplication for extension degree 4,
/// using even-odd decomposition.
///
/// Let the input polynomials be:
/// ```text
///     A = a0 + a1·X + a2·X² + a3·X³
///     B = b0 + b1·X + b2·X² + b3·X³
/// ```
///
/// These can be grouped as:
/// ```text
///     A = E_A + X·O_A, where E_A = a0 + a2·X² (even powers),
///                            O_A = a1 + a3·X² (odd  powers)
///
///     B = E_B + X·O_B, where E_B = b0 + b2·X²,
///                            O_B = b1 + b3·X²
/// ```
///
/// The multiplication follows:
/// ```text
///     A·B = E_A·E_B
///         + (E_A·O_B + O_A·E_B)·X
///         + O_A·O_B·X²
/// ```
///
/// All three products above are degree-1 polynomials in `Y = X²`,
/// and we evaluate them using small Karatsuba. We then reduce `X⁴ ≡ w`
/// to fold back higher-degree terms into the lower ones.
///
/// Denoting:
/// ```text
///     E = (E0, E1, E2) = E_A·E_B
///     O = (O0, O1, O2) = O_A·O_B
///     S = (S0, S1, S2) = (E_A + O_A)(E_B + O_B)
///     M = S - E - O    = E_A·O_B + O_A·E_B
/// ```
///
/// The final coefficients are:
/// ```text
///     res[0] = E0 + w·(E2 + O1)
///     res[1] = M0 + w·M2
///     res[2] = E1 + O0 + w·O2
///     res[3] = M1
/// ```
#[inline]
fn quartic_karatsuba_mul<F, R, R2, const D: usize>(a: &[R; D], b: &[R2; D], res: &mut [R; D], w: F)
where
    F: Field,
    R: Algebra<F> + Add<Output = R> + Sub<Output = R> + Mul<R2, Output = R> + Clone,
    R2: Algebra<F> + Add<Output = R2> + Clone,
{
    debug_assert_eq!(D, 4);

    // 1. Compute E = E_A · E_B = (a0 + a2·X²)(b0 + b2·X²)
    //    - e0: X⁰ term
    //    - e1: X² term
    //    - e2: X⁴ term (to fold using X⁴ ≡ w)
    let (e0, e1, e2) = karatsuba(a[0].clone(), a[2].clone(), b[0].clone(), b[2].clone());

    // 2. Compute O = O_A · O_B = (a1 + a3·X²)(b1 + b3·X²)
    //    - o0: X⁰ term
    //    - o1: X² term
    //    - o2: X⁴ term (to fold using X⁴ ≡ w)
    let (o0, o1, o2) = karatsuba(a[1].clone(), a[3].clone(), b[1].clone(), b[3].clone());

    // 3. Compute S = (E_A + O_A)(E_B + O_B)
    //    - s0: X⁰ term
    //    - s1: X² term
    //    - s2: X⁴ term
    let (s0, s1, s2) = karatsuba(
        a[0].clone() + a[1].clone(),
        a[2].clone() + a[3].clone(),
        b[0].clone() + b[1].clone(),
        b[2].clone() + b[3].clone(),
    );

    // 4. Compute M = S - E - O = E_A·O_B + O_A·E_B
    //    - m0: X⁰ term
    //    - m1: X² term
    //    - m2: X⁴ term (to fold using X⁴ ≡ w)
    let m0 = s0.clone() - e0.clone() - o0.clone();
    let m1 = s1.clone() - e1.clone() - o1.clone();
    let m2 = s2.clone() - e2.clone() - o2.clone();

    // 5. Assemble result C = A·B
    //    - Fold every X⁴ term using X⁴ ≡ w

    // Coefficient for X⁰:
    // = e0 + w·(e2 + o1)
    res[0] = e0 + (e2.clone() + o1.clone()) * w;

    // Coefficient for X¹:
    // = m0 + w·m2
    res[1] = m0 + m2.clone() * w;

    // Coefficient for X²:
    // = e1 + o0 + w·o2
    res[2] = e1 + o0 + o2.clone() * w;

    // Coefficient for X³:
    // = m1
    res[3] = m1;
}
