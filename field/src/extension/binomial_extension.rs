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
use p3_util::{as_base_slice, as_base_slice_mut, flatten_to_base, reconstitute_from_base};
use rand::distr::StandardUniform;
use rand::prelude::Distribution;
use serde::{Deserialize, Serialize};

use super::{HasFrobenius, HasTwoAdicBinomialExtension, PackedBinomialExtensionField};
use crate::extension::{BinomiallyExtendable, BinomiallyExtendableAlgebra};
use crate::field::Field;
use crate::{
    Algebra, BasedVectorSpace, ExtensionField, Packable, PrimeCharacteristicRing,
    RawDataSerializable, TwoAdicField, field_to_array,
};

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, Serialize, Deserialize, PartialOrd, Ord)]
#[repr(transparent)] // Needed to make various casts safe.
#[must_use]
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
            flatten_to_base::<A, Self>(vec)
        }
    }

    #[inline]
    fn reconstitute_from_base(vec: Vec<A>) -> Vec<Self> {
        unsafe {
            // Safety:
            // As `Self` is a `repr(transparent)`, it is stored identically in memory to `[A; D]`
            reconstitute_from_base::<A, Self>(vec)
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
        // Slightly faster than self.repeated_frobenius(1)
        let mut res = Self::ZERO;
        for (i, z) in F::DTH_ROOT.powers().take(D).enumerate() {
            res.value[i] = self.value[i] * z;
        }

        res
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

    /// Compute the inverse of a given element making use of the Frobenius automorphism.
    ///
    /// Algorithm 11.3.4 in Handbook of Elliptic and Hyperelliptic Curve Cryptography.
    #[inline]
    fn frobenius_inv(&self) -> Self {
        // Writing 'a' for self and `q` for the order of the base field, our goal is to compute `a^{-1}`.
        //
        // Note that we can write `-1 = (q^{D - 1} + ... + q) - (q^{D - 1} + ... + q + 1)`.
        // This is a useful decomposition as powers of q can be efficiently computed using the frobenius
        // automorphism and `Norm(a) = a^{(q^{D - 1} + ... + q + 1)}` is guaranteed to lie in the base field.
        // This means that `Norm(a)^{-1}` can be computed using base field operations.
        //
        // Hence this implementation first computes `ProdConj(a) = a^{q^{D - 1} + ... + q}` using frobenius automorphisms.
        // From this, it computes `Norm(a) = a * ProdConj(a)` and returns `ProdConj(a) * Norm(a)^{-1} = a^{-1}`.

        // This loop requires a linear number of multiplications and Frobenius automorphisms.
        // If D is known, it is possible to do this in a logarithmic number. See quintic_inv
        // for an example of this.
        let mut prod_conj = self.frobenius();
        for _ in 2..D {
            prod_conj = (prod_conj * *self).frobenius();
        }

        // norm = a * prod_conj is in the base field, so only compute that
        // coefficient rather than the full product.
        let a = self.value;
        let b = prod_conj.value;
        let mut w_coeff = F::ZERO;
        // This should really be a dot product but
        // const generics doesn't let this happen:
        // b.reverse();
        // let mut g = F::dot_product::<{D - 1}>(a[1..].try_into().unwrap(), b[..D - 1].try_into().unwrap());
        for i in 1..D {
            w_coeff += a[i] * b[D - i];
        }
        let norm = F::dot_product(&[a[0], F::W], &[b[0], w_coeff]);
        debug_assert_eq!(Self::from(norm), *self * prod_conj);

        prod_conj * norm.inverse()
    }
}

impl<F, A, const D: usize> PrimeCharacteristicRing for BinomialExtensionField<F, D, A>
where
    F: BinomiallyExtendable<D>,
    A: BinomiallyExtendableAlgebra<F, D>,
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

    #[inline]
    fn halve(&self) -> Self {
        Self::new(self.value.clone().map(|x| x.halve()))
    }

    #[inline(always)]
    fn square(&self) -> Self {
        let mut res = Self::default();
        let w = F::W;
        binomial_square(&self.value, &mut res.value, w);
        res
    }

    #[inline]
    fn mul_2exp_u64(&self, exp: u64) -> Self {
        // Depending on the field, this might be a little slower than
        // the default implementation if the compiler doesn't realize `F::TWO.exp_u64(exp)` is a constant.
        Self::new(self.value.clone().map(|x| x.mul_2exp_u64(exp)))
    }

    #[inline]
    fn div_2exp_u64(&self, exp: u64) -> Self {
        // Depending on the field, this might be a little slower than
        // the default implementation if the compiler doesn't realize `F::ONE.halve().exp_u64(exp)` is a constant.
        Self::new(self.value.clone().map(|x| x.div_2exp_u64(exp)))
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
            4 => quartic_inv(&self.value, &mut res.value, F::W),
            5 => res = quintic_inv(self),
            8 => octic_inv(&self.value, &mut res.value, F::W),
            _ => res = self.frobenius_inv(),
        }

        Some(res)
    }

    #[inline]
    fn add_slices(slice_1: &mut [Self], slice_2: &[Self]) {
        // By construction, Self is repr(transparent) over [F; D].
        // Additionally, addition is F-linear. Hence we can cast
        // everything to F and use F's add_slices.
        unsafe {
            let base_slice_1 = as_base_slice_mut(slice_1);
            let base_slice_2 = as_base_slice(slice_2);

            F::add_slices(base_slice_1, base_slice_2);
        }
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
            write!(f, "{str}")
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
    A: BinomiallyExtendableAlgebra<F, D>,
{
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        let value = A::binomial_add(&self.value, &rhs.value);
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
    A: BinomiallyExtendableAlgebra<F, D>,
{
    #[inline]
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|acc, x| acc + x).unwrap_or(Self::ZERO)
    }
}

impl<F, A, const D: usize> Sub for BinomialExtensionField<F, D, A>
where
    F: BinomiallyExtendable<D>,
    A: BinomiallyExtendableAlgebra<F, D>,
{
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        let value = A::binomial_sub(&self.value, &rhs.value);
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
    A: BinomiallyExtendableAlgebra<F, D>,
{
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let a = self.value;
        let b = rhs.value;
        let mut res = Self::default();
        let w = F::W;

        A::binomial_mul(&a, &b, &mut res.value, w);

        res
    }
}

impl<F, A, const D: usize> Mul<A> for BinomialExtensionField<F, D, A>
where
    F: BinomiallyExtendable<D>,
    A: BinomiallyExtendableAlgebra<F, D>,
{
    type Output = Self;

    #[inline]
    fn mul(self, rhs: A) -> Self {
        Self::new(A::binomial_base_mul(self.value, rhs))
    }
}

impl<F, A, const D: usize> MulAssign for BinomialExtensionField<F, D, A>
where
    F: BinomiallyExtendable<D>,
    A: BinomiallyExtendableAlgebra<F, D>,
{
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.clone() * rhs;
    }
}

impl<F, A, const D: usize> MulAssign<A> for BinomialExtensionField<F, D, A>
where
    F: BinomiallyExtendable<D>,
    A: BinomiallyExtendableAlgebra<F, D>,
{
    #[inline]
    fn mul_assign(&mut self, rhs: A) {
        *self = self.clone() * rhs;
    }
}

impl<F, A, const D: usize> Product for BinomialExtensionField<F, D, A>
where
    F: BinomiallyExtendable<D>,
    A: BinomiallyExtendableAlgebra<F, D>,
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
pub fn vector_add<R: PrimeCharacteristicRing + Add<R2, Output = R>, R2: Clone, const D: usize>(
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
    R: Algebra<F> + Algebra<R2>,
    R2: Algebra<F>,
    const D: usize,
>(
    a: &[R; D],
    b: &[R2; D],
    res: &mut [R; D],
    w: F,
) {
    match D {
        2 => quadratic_mul(a, b, res, w),
        3 => cubic_mul(a, b, res, w),
        4 => quartic_mul(a, b, res, w),
        5 => quintic_mul(a, b, res, w),
        8 => octic_mul(a, b, res, w),
        _ =>
        {
            #[allow(clippy::needless_range_loop)]
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

/// Square a vector representing an element in a binomial extension.
///
/// This is optimized for the case that R is a prime field or its packing.
#[inline]
pub(super) fn binomial_square<F: Field, R: Algebra<F>, const D: usize>(
    a: &[R; D],
    res: &mut [R; D],
    w: F,
) {
    match D {
        2 => {
            let a1_w = a[1].clone() * w;
            res[0] = R::dot_product(a[..].try_into().unwrap(), &[a[0].clone(), a1_w]);
            res[1] = a[0].clone() * a[1].double();
        }
        3 => cubic_square(a, res, w),
        4 => quartic_square(a, res, w),
        5 => quintic_square(a, res, w),
        8 => octic_square(a, res, w),
        _ => binomial_mul::<F, R, R, D>(a, a, res, w),
    }
}

/// Optimized multiplication for quadratic extension field.
///
/// Makes use of the in built field dot product code. This is optimized for the case that
/// R is a prime field or its packing.
///
/// ```text
///     A = a0 + a1·X
///     B = b0 + b1·X
/// ```
/// Where `X` satisfies `X² = w`. Then the product is:
/// ```text
///     A·B = a0·b0 + a1·b1·w + (a0·b1 + a1·b0)·X
/// ```
#[inline]
fn quadratic_mul<F, R, R2, const D: usize>(a: &[R; D], b: &[R2; D], res: &mut [R; D], w: F)
where
    F: Field,
    R: Algebra<F> + Algebra<R2>,
    R2: Algebra<F>,
{
    let b1_w = b[1].clone() * w;

    // Compute a0·b0 + a1·b1·w
    res[0] = R::dot_product(
        a[..].try_into().unwrap(),
        &[b[0].clone().into(), b1_w.into()],
    );

    // Compute a0·b1 + a1·b0
    res[1] = R::dot_product(
        &[a[0].clone(), a[1].clone()],
        &[b[1].clone().into(), b[0].clone().into()],
    );
}

///Section 11.3.6b in Handbook of Elliptic and Hyperelliptic Curve Cryptography.
#[inline]
fn quadratic_inv<F: Field, const D: usize>(a: &[F; D], res: &mut [F; D], w: F) {
    assert_eq!(D, 2);
    let neg_a1 = -a[1];
    let scalar = F::dot_product(&[a[0], neg_a1], &[a[0], w * a[1]]).inverse();
    res[0] = a[0] * scalar;
    res[1] = neg_a1 * scalar;
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
fn cubic_mul<F: Field, R: Algebra<F> + Algebra<R2>, R2: Algebra<F>, const D: usize>(
    a: &[R; D],
    b: &[R2; D],
    res: &mut [R; D],
    w: F,
) {
    assert_eq!(D, 3);
    // TODO: Test if we should switch to a naive multiplication approach using dot products.
    // This is mainly used for a degree 3 extension of Complex<Mersenne31> so this approach might be faster.

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
fn cubic_square<F: Field, R: Algebra<F>, const D: usize>(a: &[R; D], res: &mut [R; D], w: F) {
    assert_eq!(D, 3);

    let w_a2 = a[2].clone() * w;

    res[0] = a[0].square() + (a[1].clone() * w_a2.clone()).double();
    res[1] = w_a2 * a[2].clone() + (a[0].clone() * a[1].clone()).double();
    res[2] = a[1].square() + (a[0].clone() * a[2].clone()).double();
}

/// Multiplication in a quartic binomial extension field.
///
/// Makes use of the in built field dot product code. This is optimized for the case that
/// R is a prime field or its packing.
#[inline]
pub fn quartic_mul<F, R, R2, const D: usize>(a: &[R; D], b: &[R2; D], res: &mut [R; D], w: F)
where
    F: Field,
    R: Algebra<F> + Algebra<R2>,
    R2: Algebra<F>,
{
    assert_eq!(D, 4);
    let b_r_rev: [R; 5] = [
        b[3].clone().into(),
        b[2].clone().into(),
        b[1].clone().into(),
        b[0].clone().into(),
        w.into(),
    ];

    // Constant term = a0*b0 + w(a1*b3 + a2*b2 + a3*b1)
    let w_coeff_0 =
        R::dot_product::<3>(a[1..].try_into().unwrap(), b_r_rev[..3].try_into().unwrap());
    res[0] = R::dot_product(&[a[0].clone(), w_coeff_0], b_r_rev[3..].try_into().unwrap());

    // Linear term = a0*b1 + a1*b0 + w(a2*b3 + a3*b2)
    let w_coeff_1 =
        R::dot_product::<2>(a[2..].try_into().unwrap(), b_r_rev[..2].try_into().unwrap());
    res[1] = R::dot_product(
        &[a[0].clone(), a[1].clone(), w_coeff_1],
        b_r_rev[2..].try_into().unwrap(),
    );

    // Square term = a0*b2 + a1*b1 + a2*b0 + w(a3*b3)
    let b3_w = b[3].clone() * w;
    res[2] = R::dot_product::<4>(
        a[..4].try_into().unwrap(),
        &[
            b_r_rev[1].clone(),
            b_r_rev[2].clone(),
            b_r_rev[3].clone(),
            b3_w.into(),
        ],
    );

    // Cubic term = a0*b3 + a1*b2 + a2*b1 + a3*b0
    res[3] = R::dot_product::<4>(a[..].try_into().unwrap(), b_r_rev[..4].try_into().unwrap());
}

/// Compute the inverse of a quartic binomial extension field element.
#[inline]
fn quartic_inv<F: Field, const D: usize>(a: &[F; D], res: &mut [F; D], w: F) {
    assert_eq!(D, 4);

    // We use the fact that the quartic extension is a tower of quadratic extensions.
    // We can see this by writing our element as a = a0 + a1·X + a2·X² + a3·X³ = (a0 + a2·X²) + (a1 + a3·X²)·X.
    // Explicitly our tower looks like F < F[x]/(X²-w) < F[x]/(X⁴-w).
    // Using this, we can compute the inverse of a in three steps:

    // Compute the norm of our element with respect to F[x]/(X²-w).
    // This is given by:
    //      ((a0 + a2·X²) + (a1 + a3·X²)·X) * ((a0 + a2·X²) - (a1 + a3·X²)·X)
    //          = (a0 + a2·X²)² - (a1 + a3·X²)²
    //          = (a0² + w·a2² - 2w·a1·a3) + (2·a0·a2 - a1² - w·a3²)·X²
    //          = norm_0 + norm_1·X² = norm
    let neg_a1 = -a[1];
    let a3_w = a[3] * w;
    let norm_0 = F::dot_product(&[a[0], a[2], neg_a1.double()], &[a[0], a[2] * w, a3_w]);
    let norm_1 = F::dot_product(&[a[0], a[1], -a[3]], &[a[2].double(), neg_a1, a3_w]);

    // Now we compute the inverse of norm = norm_0 + norm_1·X².
    let mut inv = [F::ZERO; 2];
    quadratic_inv(&[norm_0, norm_1], &mut inv, w);

    // Then the inverse of a is given by:
    //      a⁻¹ = ((a0 + a2·X²) - (a1 + a3·X²)·X)·norm⁻¹
    //          = (a0 + a2·X²)·norm⁻¹ - (a1 + a3·X²)·norm⁻¹·X
    // Both of these multiplications can be done in the quadratic extension field.
    let mut out_evn = [F::ZERO; 2];
    let mut out_odd = [F::ZERO; 2];
    quadratic_mul(&[a[0], a[2]], &inv, &mut out_evn, w);
    quadratic_mul(&[a[1], a[3]], &inv, &mut out_odd, w);

    res[0] = out_evn[0];
    res[1] = -out_odd[0];
    res[2] = out_evn[1];
    res[3] = -out_odd[1];
}

/// Optimized Square function for quadratic extension field.
///
/// Makes use of the in built field dot product code. This is optimized for the case that
/// R is a prime field or its packing.
#[inline]
fn quartic_square<F, R, const D: usize>(a: &[R; D], res: &mut [R; D], w: F)
where
    F: Field,
    R: Algebra<F>,
{
    assert_eq!(D, 4);

    let two_a0 = a[0].double();
    let two_a1 = a[1].double();
    let two_a2 = a[2].double();
    let a2_w = a[2].clone() * w;
    let a3_w = a[3].clone() * w;

    // Constant term = a0*a0 + w*a2*a2 + 2*w*a1*a3
    res[0] = R::dot_product(
        &[a[0].clone(), a2_w, two_a1],
        &[a[0].clone(), a[2].clone(), a3_w.clone()],
    );

    // Linear term = 2*a0*a1 + 2*w*a2*a3)
    res[1] = R::dot_product(
        &[two_a0.clone(), two_a2.clone()],
        &[a[1].clone(), a3_w.clone()],
    );

    // Square term = a1*a1 + w*a3*a3 + 2*a0*a2
    res[2] = R::dot_product(
        &[a[1].clone(), a3_w, two_a0.clone()],
        &[a[1].clone(), a[3].clone(), a[2].clone()],
    );

    // Cubic term = 2*a0*a3 + 2*a1*a2)
    res[3] = R::dot_product(&[two_a0, two_a2], &[a[3].clone(), a[1].clone()]);
}

/// Multiplication in a quintic binomial extension field.
///
/// Makes use of the in built field dot product code. This is optimized for the case that
/// R is a prime field or its packing.
pub fn quintic_mul<F, R, R2, const D: usize>(a: &[R; D], b: &[R2; D], res: &mut [R; D], w: F)
where
    F: Field,
    R: Algebra<F> + Algebra<R2>,
    R2: Algebra<F>,
{
    assert_eq!(D, 5);
    let b_r_rev: [R; 6] = [
        b[4].clone().into(),
        b[3].clone().into(),
        b[2].clone().into(),
        b[1].clone().into(),
        b[0].clone().into(),
        w.into(),
    ];

    // Constant term = a0*b0 + w(a1*b4 + a2*b3 + a3*b2 + a4*b1)
    let w_coeff_0 =
        R::dot_product::<4>(a[1..].try_into().unwrap(), b_r_rev[..4].try_into().unwrap());
    res[0] = R::dot_product(&[a[0].clone(), w_coeff_0], b_r_rev[4..].try_into().unwrap());

    // Linear term = a0*b1 + a1*b0 + w(a2*b4 + a3*b3 + a4*b2)
    let w_coeff_1 =
        R::dot_product::<3>(a[2..].try_into().unwrap(), b_r_rev[..3].try_into().unwrap());
    res[1] = R::dot_product(
        &[a[0].clone(), a[1].clone(), w_coeff_1],
        b_r_rev[3..].try_into().unwrap(),
    );

    // Square term = a0*b2 + a1*b1 + a2*b0 + w(a3*b4 + a4*b3)
    let w_coeff_2 =
        R::dot_product::<2>(a[3..].try_into().unwrap(), b_r_rev[..2].try_into().unwrap());
    res[2] = R::dot_product(
        &[a[0].clone(), a[1].clone(), a[2].clone(), w_coeff_2],
        b_r_rev[2..].try_into().unwrap(),
    );

    // Cubic term = a0*b3 + a1*b2 + a2*b1 + a3*b0 + w*a4*b4
    let b4_w = b[4].clone() * w;
    res[3] = R::dot_product::<5>(
        a[..5].try_into().unwrap(),
        &[
            b_r_rev[1].clone(),
            b_r_rev[2].clone(),
            b_r_rev[3].clone(),
            b_r_rev[4].clone(),
            b4_w.into(),
        ],
    );

    // Quartic term = a0*b4 + a1*b3 + a2*b2 + a3*b1 + a4*b0
    res[4] = R::dot_product::<5>(a[..].try_into().unwrap(), b_r_rev[..5].try_into().unwrap());
}

/// Optimized Square function for quintic extension field elements.
///
/// Makes use of the in built field dot product code. This is optimized for the case that
/// R is a prime field or its packing.
#[inline]
fn quintic_square<F, R, const D: usize>(a: &[R; D], res: &mut [R; D], w: F)
where
    F: Field,
    R: Algebra<F>,
{
    assert_eq!(D, 5);

    let two_a0 = a[0].double();
    let two_a1 = a[1].double();
    let two_a2 = a[2].double();
    let two_a3 = a[3].double();
    let w_a3 = a[3].clone() * w;
    let w_a4 = a[4].clone() * w;

    // Constant term = a0*a0 + 2*w(a1*a4 + a2*a3)
    res[0] = R::dot_product(
        &[a[0].clone(), w_a4.clone(), w_a3.clone()],
        &[a[0].clone(), two_a1.clone(), two_a2.clone()],
    );

    // Linear term = w*a3*a3 + 2*(a0*a1 + w * a2*a4)
    res[1] = R::dot_product(
        &[w_a3, two_a0.clone(), w_a4.clone()],
        &[a[3].clone(), a[1].clone(), two_a2],
    );

    // Square term = a1*a1 + 2 * (a0*a2 + w*a3*a4)
    res[2] = R::dot_product(
        &[a[1].clone(), two_a0.clone(), w_a4.clone()],
        &[a[1].clone(), a[2].clone(), two_a3],
    );

    // Cubic term = w*a4*a4 + 2*(a0*a3 + a1*a2)
    res[3] = R::dot_product(
        &[w_a4, two_a0.clone(), two_a1.clone()],
        &[a[4].clone(), a[3].clone(), a[2].clone()],
    );

    // Quartic term = a2*a2 + 2*(a0*a4 + a1*a3)
    res[4] = R::dot_product(
        &[a[2].clone(), two_a0, two_a1],
        &[a[2].clone(), a[4].clone(), a[3].clone()],
    );
}

/// Optimized Square function for octic extension field elements.
///
/// Makes use of the in built field dot product code. This is optimized for the case that
/// R is a prime field or its packing.
#[inline]
fn octic_square<F, R, const D: usize>(a: &[R; D], res: &mut [R; D], w: F)
where
    F: Field,
    R: Algebra<F>,
{
    assert_eq!(D, 8);

    let a0_2 = a[0].double();
    let a1_2 = a[1].double();
    let a2_2 = a[2].double();
    let a3_2 = a[3].double();
    let w_a4 = a[4].clone() * w;
    let w_a5 = a[5].clone() * w;
    let w_a6 = a[6].clone() * w;
    let w_a7 = a[7].clone() * w;
    let w_a5_2 = w_a5.double();
    let w_a6_2 = w_a6.double();
    let w_a7_2 = w_a7.double();

    // Constant coefficient = a0² + w (2(a1 * a7 + a2 * a6 + a3 * a5) + a4²)
    res[0] = R::dot_product(
        &[
            a[0].clone(),
            a[1].clone(),
            a[2].clone(),
            a[3].clone(),
            a[4].clone(),
        ],
        &[
            a[0].clone(),
            w_a7_2.clone(),
            w_a6_2.clone(),
            w_a5_2.clone(),
            w_a4,
        ],
    );

    // Linear coefficient = 2(a0 * a1 + w(a2 * a7 + a3 * a6 + a4 * a5))
    res[1] = R::dot_product(
        &[a0_2.clone(), a[2].clone(), a[3].clone(), a[4].clone()],
        &[a[1].clone(), w_a7_2.clone(), w_a6_2.clone(), w_a5_2.clone()],
    );

    // Square coefficient = 2a0 * a2 + a1² + w(2(a3 * a7 + a4 * a6) + a5²)
    res[2] = R::dot_product(
        &[
            a0_2.clone(),
            a[1].clone(),
            a[3].clone(),
            a[4].clone(),
            a[5].clone(),
        ],
        &[
            a[2].clone(),
            a[1].clone(),
            w_a7_2.clone(),
            w_a6_2.clone(),
            w_a5,
        ],
    );

    // Cube coefficient = 2(a0 * a3 + a1 * a2 + w(a4 * a7 + a5 * a6)
    res[3] = R::dot_product(
        &[a0_2.clone(), a1_2.clone(), a[4].clone(), a[5].clone()],
        &[a[3].clone(), a[2].clone(), w_a7_2.clone(), w_a6_2.clone()],
    );

    // Quartic coefficient = 2(a0 * a4 + a1 * a3) + a2² + w(2 * a7 * a5 + a6²)
    res[4] = R::dot_product(
        &[
            a0_2.clone(),
            a1_2.clone(),
            a[2].clone(),
            a[5].clone(),
            a[6].clone(),
        ],
        &[
            a[4].clone(),
            a[3].clone(),
            a[2].clone(),
            w_a7_2.clone(),
            w_a6,
        ],
    );

    // Quintic coefficient = 2 * (a0 * a5 + a1 * a4 + a2 * a3 + w * a6 * a7)
    res[5] = R::dot_product(
        &[a0_2.clone(), a1_2.clone(), a2_2.clone(), a[6].clone()],
        &[a[5].clone(), a[4].clone(), a[3].clone(), w_a7_2],
    );

    // Sextic coefficient = 2(a0 * a6 + a1 * a5 + a2 * a4) + a3² + w * a7²
    res[6] = R::dot_product(
        &[
            a0_2.clone(),
            a1_2.clone(),
            a2_2.clone(),
            a[3].clone(),
            a[7].clone(),
        ],
        &[a[6].clone(), a[5].clone(), a[4].clone(), a[3].clone(), w_a7],
    );

    // Final coefficient = 2(a0 * a7 + a1 * a6 + a2 * a5 + a3 * a4)
    res[7] = R::dot_product(
        &[a0_2, a1_2, a2_2, a3_2],
        &[a[7].clone(), a[6].clone(), a[5].clone(), a[4].clone()],
    );
}

/// Compute the inverse of a quintic binomial extension field element.
#[inline]
fn quintic_inv<F: BinomiallyExtendable<D>, const D: usize>(
    a: &BinomialExtensionField<F, D>,
) -> BinomialExtensionField<F, D> {
    // Writing 'a' for self, we need to compute: `prod_conj = a^{q^4 + q^3 + q^2 + q}`
    let a_exp_q = a.frobenius();
    let a_exp_q_plus_q_sq = (*a * a_exp_q).frobenius();
    let prod_conj = a_exp_q_plus_q_sq * a_exp_q_plus_q_sq.repeated_frobenius(2);

    // norm = a * prod_conj is in the base field, so only compute that
    // coefficient rather than the full product.
    let a_vals = a.value;
    let mut b = prod_conj.value;
    b.reverse();

    let w_coeff = F::dot_product::<4>(a.value[1..].try_into().unwrap(), b[..4].try_into().unwrap());
    let norm = F::dot_product::<2>(&[a_vals[0], F::W], &[b[4], w_coeff]);
    debug_assert_eq!(BinomialExtensionField::<F, D>::from(norm), *a * prod_conj);

    prod_conj * norm.inverse()
}

/// Compute the (D-N)'th coefficient in the multiplication of two elements in a degree
/// D binomial extension field.
///
/// a_0 * b_{D - N} + ... + a_{D - N} * b_0 + w * (a_{D - N + 1}b_{D - 1} + ... + a_{D - 1}b_{D - N + 1})
///
/// # Inputs
/// - a: An array of coefficients.
/// - b: An array of coefficients in reverse order with last element equal to `W`
#[inline]
fn compute_coefficient<
    F,
    R,
    const D: usize,
    const D_PLUS_1: usize,
    const N: usize,
    const D_PLUS_1_MIN_N: usize,
>(
    a: &[R; D],
    b_rev: &[R; D_PLUS_1],
) -> R
where
    F: Field,
    R: Algebra<F>,
{
    let w_coeff = R::dot_product::<N>(
        a[(D - N)..].try_into().unwrap(),
        b_rev[..N].try_into().unwrap(),
    );
    let mut scratch: [R; D_PLUS_1_MIN_N] = array::from_fn(|i| a[i].clone());
    scratch[D_PLUS_1_MIN_N - 1] = w_coeff;
    R::dot_product(&scratch, b_rev[N..].try_into().unwrap())
}

/// Multiplication in an octic binomial extension field.
///
/// Makes use of the in built field dot product code. This is optimized for the case that
/// R is a prime field or its packing.
#[inline]
pub fn octic_mul<F, R, R2, const D: usize>(a: &[R; D], b: &[R2; D], res: &mut [R; D], w: F)
where
    F: Field,
    R: Algebra<F> + Algebra<R2>,
    R2: Algebra<F>,
{
    assert_eq!(D, 8);
    let a: &[R; 8] = a[..].try_into().unwrap();
    let mut b_r_rev: [R; 9] = [
        b[7].clone().into(),
        b[6].clone().into(),
        b[5].clone().into(),
        b[4].clone().into(),
        b[3].clone().into(),
        b[2].clone().into(),
        b[1].clone().into(),
        b[0].clone().into(),
        w.into(),
    ];

    // Constant coefficient = a0*b0 + w(a1*b7 + ... + a7*b1)
    res[0] = compute_coefficient::<F, R, 8, 9, 7, 2>(a, &b_r_rev);

    // Linear coefficient = a0*b1 + a1*b0 + w(a2*b7 + ... + a7*b2)
    res[1] = compute_coefficient::<F, R, 8, 9, 6, 3>(a, &b_r_rev);

    // Square coefficient = a0*b2 + .. + a2*b0 + w(a3*b7 + ... + a7*b3)
    res[2] = compute_coefficient::<F, R, 8, 9, 5, 4>(a, &b_r_rev);

    // Cube coefficient = a0*b3 + .. + a3*b0 + w(a4*b7 + ... + a7*b4)
    res[3] = compute_coefficient::<F, R, 8, 9, 4, 5>(a, &b_r_rev);

    // Quartic coefficient = a0*b4 + ... + a4*b0 + w(a5*b7 + ... + a7*b5)
    res[4] = compute_coefficient::<F, R, 8, 9, 3, 6>(a, &b_r_rev);

    // Quintic coefficient = a0*b5 + ... + a5*b0 + w(a6*b7 + ... + a7*b6)
    res[5] = compute_coefficient::<F, R, 8, 9, 2, 7>(a, &b_r_rev);

    // Sextic coefficient = a0*b6 + ... + a6*b0 + w*a7*b7
    b_r_rev[8] *= b[7].clone();
    res[6] = R::dot_product::<8>(a, b_r_rev[1..].try_into().unwrap());

    // Final coefficient = a0*b7 + ... + a7*b0
    res[7] = R::dot_product::<8>(a, b_r_rev[..8].try_into().unwrap());
}

/// Compute the inverse of a octic binomial extension field element.
#[inline]
fn octic_inv<F: Field, const D: usize>(a: &[F; D], res: &mut [F; D], w: F) {
    assert_eq!(D, 8);

    // We use the fact that the octic extension is a tower of extensions.
    // Explicitly our tower looks like F < F[x]/(X⁴ - w) < F[x]/(X^8 - w).
    // Using this, we can compute the inverse of a in three steps:

    // Compute the norm of our element with respect to F[x]/(X⁴-w).
    // Writing a = a0 + a1·X + a2·X² + a3·X³ + a4·X⁴ + a5·X⁵ + a6·X⁶ + a7·X⁷
    //           = (a0 + a2·X² + a4·X⁴ + a6·X⁶) + (a1 + a3·X² + a5·X⁴ + a7·X⁶)·X
    //           = evens + odds·X
    //
    // The norm is given by:
    //    norm = (evens + odds·X) * (evens - odds·X)
    //          = evens² - odds²·X²
    //
    // This costs 2 multiplications in the quartic extension field.
    let evns = [a[0], a[2], a[4], a[6]];
    let odds = [a[1], a[3], a[5], a[7]];
    let mut evns_sq = [F::ZERO; 4];
    let mut odds_sq = [F::ZERO; 4];
    quartic_square(&evns, &mut evns_sq, w);
    quartic_square(&odds, &mut odds_sq, w);
    // odds_sq is multiplied by X^2 so we need to rotate it and multiply by a factor of w.
    let norm = [
        evns_sq[0] - w * odds_sq[3],
        evns_sq[1] - odds_sq[0],
        evns_sq[2] - odds_sq[1],
        evns_sq[3] - odds_sq[2],
    ];

    // Now we compute the inverse of norm inside F[x]/(X⁴ - w). We already have an efficient function for this.
    let mut norm_inv = [F::ZERO; 4];
    quartic_inv(&norm, &mut norm_inv, w);

    // Then the inverse of a is given by:
    //      a⁻¹ = (evens - odds·X)·norm⁻¹
    //          = evens·norm⁻¹ - odds·norm⁻¹·X
    //
    // Both of these multiplications can again be done in the quartic extension field.
    let mut out_evn = [F::ZERO; 4];
    let mut out_odd = [F::ZERO; 4];
    quartic_mul(&evns, &norm_inv, &mut out_evn, w);
    quartic_mul(&odds, &norm_inv, &mut out_odd, w);

    res[0] = out_evn[0];
    res[1] = -out_odd[0];
    res[2] = out_evn[1];
    res[3] = -out_odd[1];
    res[4] = out_evn[2];
    res[5] = -out_odd[2];
    res[6] = out_evn[3];
    res[7] = -out_odd[3];
}
