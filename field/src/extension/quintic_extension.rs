//! Degree-5 extension field using the trinomial `X^5 + X^2 - 1`.
//!
//! This extension requires that `X^5 + X^2 - 1` is irreducible over the base field.
//! Currently used for KoalaBear where irreducibility has been verified.
//!
//! Reduction identity: `X^5 = 1 - X^2`

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

use super::packed_quintic_extension::PackedQuinticExtensionField;
use super::{HasFrobenius, HasTwoAdicQuinticExtension};
use crate::extension::{QuinticExtendableAlgebra, QuinticTrinomialExtendable};
use crate::field::Field;
use crate::{
    Algebra, BasedVectorSpace, ExtensionField, Packable, PackedFieldExtension,
    PrimeCharacteristicRing, RawDataSerializable, TwoAdicField, field_to_array,
};

/// A degree-5 extension field using the trinomial `X^5 + X^2 - 1`.
///
/// Elements are represented as `a_0 + a_1*X + a_2*X^2 + a_3*X^3 + a_4*X^4`.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, Serialize, Deserialize, PartialOrd, Ord)]
#[repr(transparent)] // Required for safe memory layout casts.
#[must_use]
pub struct QuinticExtensionField<F, A = F> {
    #[serde(
        with = "p3_util::array_serialization",
        bound(serialize = "A: Serialize", deserialize = "A: Deserialize<'de>")
    )]
    pub(crate) value: [A; 5],
    _phantom: PhantomData<F>,
}

impl<F, A> QuinticExtensionField<F, A> {
    /// Create an extension field element from coefficient array.
    ///
    /// The coefficients represent the polynomial `value[0] + value[1]*X + ... + value[4]*X^4`.
    #[inline]
    pub const fn new(value: [A; 5]) -> Self {
        Self {
            value,
            _phantom: PhantomData,
        }
    }
}

impl<F: Copy> QuinticExtensionField<F, F> {
    /// Convert a `[[F; 5]; N]` array to an array of extension field elements.
    ///
    /// # Panics
    /// Panics if `N == 0`.
    #[inline]
    pub const fn new_array<const N: usize>(input: [[F; 5]; N]) -> [Self; N] {
        const { assert!(N > 0) }
        let mut output = [Self::new(input[0]); N];
        let mut i = 1;
        while i < N {
            output[i] = Self::new(input[i]);
            i += 1;
        }
        output
    }
}

impl<F: Field, A: Algebra<F>> Default for QuinticExtensionField<F, A> {
    fn default() -> Self {
        Self::new(array::from_fn(|_| A::ZERO))
    }
}

impl<F: Field, A: Algebra<F>> From<A> for QuinticExtensionField<F, A> {
    fn from(x: A) -> Self {
        Self::new(field_to_array(x))
    }
}

impl<F, A> From<[A; 5]> for QuinticExtensionField<F, A> {
    #[inline]
    fn from(x: [A; 5]) -> Self {
        Self {
            value: x,
            _phantom: PhantomData,
        }
    }
}

impl<F: QuinticTrinomialExtendable> Packable for QuinticExtensionField<F> {}

impl<F: QuinticTrinomialExtendable, A: Algebra<F>> BasedVectorSpace<A>
    for QuinticExtensionField<F, A>
{
    const DIMENSION: usize = 5;

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
        (iter.len() == 5).then(|| Self::new(array::from_fn(|_| iter.next().unwrap())))
    }

    #[inline]
    fn flatten_to_base(vec: Vec<Self>) -> Vec<A> {
        // SAFETY: `Self` is `repr(transparent)` over `[A; 5]`.
        unsafe { flatten_to_base::<A, Self>(vec) }
    }

    #[inline]
    fn reconstitute_from_base(vec: Vec<A>) -> Vec<Self> {
        // SAFETY: `Self` is `repr(transparent)` over `[A; 5]`.
        unsafe { reconstitute_from_base::<A, Self>(vec) }
    }
}

impl<F: QuinticTrinomialExtendable> ExtensionField<F> for QuinticExtensionField<F>
where
    PackedQuinticExtensionField<F, F::Packing>: PackedFieldExtension<F, Self>,
{
    type ExtensionPacking = PackedQuinticExtensionField<F, F::Packing>;

    #[inline]
    fn is_in_basefield(&self) -> bool {
        self.value[1..].iter().all(F::is_zero)
    }

    #[inline]
    fn as_base(&self) -> Option<F> {
        <Self as ExtensionField<F>>::is_in_basefield(self).then(|| self.value[0])
    }
}

impl<F: QuinticTrinomialExtendable> HasFrobenius<F> for QuinticExtensionField<F> {
    #[inline]
    fn frobenius(&self) -> Self {
        let a = &self.value;
        let fc = &F::FROBENIUS_COEFFS;

        let c0 = a[0] + a[1] * fc[0][0] + a[2] * fc[1][0] + a[3] * fc[2][0] + a[4] * fc[3][0];
        let c1 = a[1] * fc[0][1] + a[2] * fc[1][1] + a[3] * fc[2][1] + a[4] * fc[3][1];
        let c2 = a[1] * fc[0][2] + a[2] * fc[1][2] + a[3] * fc[2][2] + a[4] * fc[3][2];
        let c3 = a[1] * fc[0][3] + a[2] * fc[1][3] + a[3] * fc[2][3] + a[4] * fc[3][3];
        let c4 = a[1] * fc[0][4] + a[2] * fc[1][4] + a[3] * fc[2][4] + a[4] * fc[3][4];

        Self::new([c0, c1, c2, c3, c4])
    }

    /// Apply Frobenius `count` times: `x → x^{p^count}`.
    #[inline]
    fn repeated_frobenius(&self, count: usize) -> Self {
        match count % 5 {
            0 => *self,
            _ => {
                let mut result = *self;
                for _ in 0..(count % 5) {
                    result = result.frobenius();
                }
                result
            }
        }
    }

    /// Compute pseudo-inverse using Frobenius automorphism.
    ///
    /// Returns `0` if `self == 0`, and `1/self` otherwise.
    ///
    /// Uses the identity: `a^{-1} = ProdConj(a) * Norm(a)^{-1}` where
    /// - `ProdConj(a) = a^{p^4 + p^3 + p^2 + p}`,
    /// - `Norm(a) = a * ProdConj(a)` is in the base field.
    #[inline]
    fn pseudo_inv(&self) -> Self {
        if self.is_zero() {
            return Self::ZERO;
        }

        // Compute ProdConj(a) = a^{p^4 + p^3 + p^2 + p} efficiently:
        // a^{p + p^2} = (a * a^p)^p, then a^{p + p^2 + p^3 + p^4} = (a^{p+p^2}) * (a^{p+p^2})^{p^2}
        let a_exp_p = self.frobenius();
        let a_exp_p_plus_p2 = (*self * a_exp_p).frobenius();
        let prod_conj = a_exp_p_plus_p2 * a_exp_p_plus_p2.repeated_frobenius(2);

        // Norm(a) = a * ProdConj(a) lies in the base field.
        // Compute only the constant coefficient.
        let norm = self.compute_norm_with_prod_conj(&prod_conj);
        debug_assert_eq!(Self::from(norm), *self * prod_conj);

        prod_conj * norm.inverse()
    }
}

impl<F: QuinticTrinomialExtendable> QuinticExtensionField<F> {
    /// Compute the norm given pre-computed product of conjugates.
    ///
    /// The norm `Norm(a) = a * prod_conj` lies in the base field.
    /// This computes only the constant coefficient for efficiency.
    #[inline]
    fn compute_norm_with_prod_conj(&self, prod_conj: &Self) -> F {
        let a = &self.value;
        let b = &prod_conj.value;

        // For trinomial X^5 + X^2 - 1, the constant term of a*b is:
        // c_0 + c_5 - c_8 where c_k = Σ_{i+j=k} a_i*b_j
        let c0 = a[0] * b[0];
        let c5 = a[1] * b[4] + a[2] * b[3] + a[3] * b[2] + a[4] * b[1];
        let c8 = a[4] * b[4];

        c0 + c5 - c8
    }
}

impl<F, A> PrimeCharacteristicRing for QuinticExtensionField<F, A>
where
    F: QuinticTrinomialExtendable,
    A: QuinticExtendableAlgebra<F>,
{
    type PrimeSubfield = <A as PrimeCharacteristicRing>::PrimeSubfield;

    const ZERO: Self = Self::new([A::ZERO; 5]);
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
        A::quintic_square(&self.value, &mut res.value);
        res
    }

    #[inline]
    fn mul_2exp_u64(&self, exp: u64) -> Self {
        Self::new(self.value.clone().map(|x| x.mul_2exp_u64(exp)))
    }

    #[inline]
    fn div_2exp_u64(&self, exp: u64) -> Self {
        Self::new(self.value.clone().map(|x| x.div_2exp_u64(exp)))
    }

    #[inline]
    fn zero_vec(len: usize) -> Vec<Self> {
        // SAFETY: `Self` is `repr(transparent)` over `[A; 5]`.
        unsafe { reconstitute_from_base(A::zero_vec(len * 5)) }
    }
}

impl<F: QuinticTrinomialExtendable> Algebra<F> for QuinticExtensionField<F> {}

impl<F: QuinticTrinomialExtendable> RawDataSerializable for QuinticExtensionField<F> {
    const NUM_BYTES: usize = F::NUM_BYTES * 5;

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
                .flat_map(|x| (0..5).map(move |i| array::from_fn(|j| x[j].value[i]))),
        )
    }

    #[inline]
    fn into_parallel_u32_streams<const N: usize>(
        input: impl IntoIterator<Item = [Self; N]>,
    ) -> impl IntoIterator<Item = [u32; N]> {
        F::into_parallel_u32_streams(
            input
                .into_iter()
                .flat_map(|x| (0..5).map(move |i| array::from_fn(|j| x[j].value[i]))),
        )
    }

    #[inline]
    fn into_parallel_u64_streams<const N: usize>(
        input: impl IntoIterator<Item = [Self; N]>,
    ) -> impl IntoIterator<Item = [u64; N]> {
        F::into_parallel_u64_streams(
            input
                .into_iter()
                .flat_map(|x| (0..5).map(move |i| array::from_fn(|j| x[j].value[i]))),
        )
    }
}

impl<F: QuinticTrinomialExtendable> Field for QuinticExtensionField<F> {
    type Packing = Self;

    const GENERATOR: Self = Self::new(F::EXT_GENERATOR);

    fn try_inverse(&self) -> Option<Self> {
        if self.is_zero() {
            return None;
        }
        Some(self.pseudo_inv())
    }

    #[inline]
    fn add_slices(slice_1: &mut [Self], slice_2: &[Self]) {
        // SAFETY: `Self` is `repr(transparent)` over `[F; 5]`.
        // Addition is F-linear, so we can operate on base field slices.
        unsafe {
            let base_slice_1 = as_base_slice_mut(slice_1);
            let base_slice_2 = as_base_slice(slice_2);
            F::add_slices(base_slice_1, base_slice_2);
        }
    }

    #[inline]
    fn order() -> BigUint {
        F::order().pow(5)
    }
}

impl<F: QuinticTrinomialExtendable> Display for QuinticExtensionField<F> {
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

impl<F, A> Neg for QuinticExtensionField<F, A>
where
    F: QuinticTrinomialExtendable,
    A: Algebra<F>,
{
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        Self::new(self.value.map(A::neg))
    }
}

impl<F, A> Add for QuinticExtensionField<F, A>
where
    F: QuinticTrinomialExtendable,
    A: QuinticExtendableAlgebra<F>,
{
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::new(A::quintic_add(&self.value, &rhs.value))
    }
}

impl<F, A> Add<A> for QuinticExtensionField<F, A>
where
    F: QuinticTrinomialExtendable,
    A: Algebra<F>,
{
    type Output = Self;

    #[inline]
    fn add(mut self, rhs: A) -> Self {
        self.value[0] += rhs;
        self
    }
}

impl<F, A> AddAssign for QuinticExtensionField<F, A>
where
    F: QuinticTrinomialExtendable,
    A: Algebra<F>,
{
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..5 {
            self.value[i] += rhs.value[i].clone();
        }
    }
}

impl<F, A> AddAssign<A> for QuinticExtensionField<F, A>
where
    F: QuinticTrinomialExtendable,
    A: Algebra<F>,
{
    #[inline]
    fn add_assign(&mut self, rhs: A) {
        self.value[0] += rhs;
    }
}

impl<F, A> Sum for QuinticExtensionField<F, A>
where
    F: QuinticTrinomialExtendable,
    A: QuinticExtendableAlgebra<F>,
{
    #[inline]
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|acc, x| acc + x).unwrap_or(Self::ZERO)
    }
}

impl<F, A> Sub for QuinticExtensionField<F, A>
where
    F: QuinticTrinomialExtendable,
    A: QuinticExtendableAlgebra<F>,
{
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::new(A::quintic_sub(&self.value, &rhs.value))
    }
}

impl<F, A> Sub<A> for QuinticExtensionField<F, A>
where
    F: QuinticTrinomialExtendable,
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

impl<F, A> SubAssign for QuinticExtensionField<F, A>
where
    F: QuinticTrinomialExtendable,
    A: Algebra<F>,
{
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        for i in 0..5 {
            self.value[i] -= rhs.value[i].clone();
        }
    }
}

impl<F, A> SubAssign<A> for QuinticExtensionField<F, A>
where
    F: QuinticTrinomialExtendable,
    A: Algebra<F>,
{
    #[inline]
    fn sub_assign(&mut self, rhs: A) {
        self.value[0] -= rhs;
    }
}

impl<F, A> Mul for QuinticExtensionField<F, A>
where
    F: QuinticTrinomialExtendable,
    A: QuinticExtendableAlgebra<F>,
{
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let mut res = Self::default();
        A::quintic_mul(&self.value, &rhs.value, &mut res.value);
        res
    }
}

impl<F, A> Mul<A> for QuinticExtensionField<F, A>
where
    F: QuinticTrinomialExtendable,
    A: QuinticExtendableAlgebra<F>,
{
    type Output = Self;

    #[inline]
    fn mul(self, rhs: A) -> Self {
        Self::new(A::quintic_base_mul(self.value, rhs))
    }
}

impl<F, A> MulAssign for QuinticExtensionField<F, A>
where
    F: QuinticTrinomialExtendable,
    A: QuinticExtendableAlgebra<F>,
{
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.clone() * rhs;
    }
}

impl<F, A> MulAssign<A> for QuinticExtensionField<F, A>
where
    F: QuinticTrinomialExtendable,
    A: QuinticExtendableAlgebra<F>,
{
    #[inline]
    fn mul_assign(&mut self, rhs: A) {
        *self = self.clone() * rhs;
    }
}

impl<F, A> Product for QuinticExtensionField<F, A>
where
    F: QuinticTrinomialExtendable,
    A: QuinticExtendableAlgebra<F>,
{
    #[inline]
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|acc, x| acc * x).unwrap_or(Self::ONE)
    }
}

impl<F> Div for QuinticExtensionField<F>
where
    F: QuinticTrinomialExtendable,
{
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        self * rhs.inverse()
    }
}

impl<F> DivAssign for QuinticExtensionField<F>
where
    F: QuinticTrinomialExtendable,
{
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl<F: QuinticTrinomialExtendable> Distribution<QuinticExtensionField<F>> for StandardUniform
where
    Self: Distribution<F>,
{
    #[inline]
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> QuinticExtensionField<F> {
        QuinticExtensionField::new(array::from_fn(|_| self.sample(rng)))
    }
}

impl<F: QuinticTrinomialExtendable + HasTwoAdicQuinticExtension> TwoAdicField
    for QuinticExtensionField<F>
{
    const TWO_ADICITY: usize = F::EXT_TWO_ADICITY;

    #[inline]
    fn two_adic_generator(bits: usize) -> Self {
        Self::new(F::ext_two_adic_generator(bits))
    }
}

/// Multiply two elements in the quintic extension field.
#[inline]
pub(super) fn quintic_mul<R: PrimeCharacteristicRing>(a: &[R; 5], b: &[R; 5], res: &mut [R; 5]) {
    // TODO: This is unoptimized.

    // Compute raw product coefficients c_0 through c_8
    let c0 = a[0].clone() * b[0].clone();
    let c1 = a[0].clone() * b[1].clone() + a[1].clone() * b[0].clone();
    let c2 =
        a[0].clone() * b[2].clone() + a[1].clone() * b[1].clone() + a[2].clone() * b[0].clone();
    let c3 = a[0].clone() * b[3].clone()
        + a[1].clone() * b[2].clone()
        + a[2].clone() * b[1].clone()
        + a[3].clone() * b[0].clone();
    let c4 = a[0].clone() * b[4].clone()
        + a[1].clone() * b[3].clone()
        + a[2].clone() * b[2].clone()
        + a[3].clone() * b[1].clone()
        + a[4].clone() * b[0].clone();

    // High-degree coefficients
    let c5 = a[1].clone() * b[4].clone()
        + a[2].clone() * b[3].clone()
        + a[3].clone() * b[2].clone()
        + a[4].clone() * b[1].clone();
    let c6 =
        a[2].clone() * b[4].clone() + a[3].clone() * b[3].clone() + a[4].clone() * b[2].clone();
    let c7 = a[3].clone() * b[4].clone() + a[4].clone() * b[3].clone();
    let c8 = a[4].clone() * b[4].clone();

    // Apply reduction: X^5 = 1 - X^2, X^6 = X - X^3, X^7 = X^2 - X^4, X^8 = X^3 + X^2 - 1
    res[0] = c0 + c5.clone() - c8.clone();
    res[1] = c1 + c6.clone();
    res[2] = c2 - c5 + c7.clone() + c8.clone();
    res[3] = c3 - c6 + c8;
    res[4] = c4 - c7;
}

/// Square an element in the quintic extension field.
#[inline]
pub(super) fn quintic_square<R: PrimeCharacteristicRing>(a: &[R; 5], res: &mut [R; 5]) {
    // TODO: This is unoptimized. See `binomial_extension.rs` for optimization ideas.

    // Diagonal squares
    let a0_sq = a[0].square();
    let a1_sq = a[1].square();
    let a2_sq = a[2].square();
    let a3_sq = a[3].square();
    let a4_sq = a[4].square();

    // Cross products (doubled using efficient addition)
    let a0a1_2 = (a[0].clone() * a[1].clone()).double();
    let a0a2_2 = (a[0].clone() * a[2].clone()).double();
    let a0a3_2 = (a[0].clone() * a[3].clone()).double();
    let a0a4_2 = (a[0].clone() * a[4].clone()).double();
    let a1a2_2 = (a[1].clone() * a[2].clone()).double();
    let a1a3_2 = (a[1].clone() * a[3].clone()).double();
    let a1a4_2 = (a[1].clone() * a[4].clone()).double();
    let a2a3_2 = (a[2].clone() * a[3].clone()).double();
    let a2a4_2 = (a[2].clone() * a[4].clone()).double();
    let a3a4_2 = (a[3].clone() * a[4].clone()).double();

    // Raw coefficients c_k = Σ_{i+j=k} a_i * a_j
    let c0 = a0_sq;
    let c1 = a0a1_2;
    let c2 = a0a2_2 + a1_sq;
    let c3 = a0a3_2 + a1a2_2;
    let c4 = a0a4_2 + a1a3_2 + a2_sq;
    let c5 = a1a4_2 + a2a3_2;
    let c6 = a2a4_2 + a3_sq;
    let c7 = a3a4_2;
    let c8 = a4_sq;

    // Apply reduction
    res[0] = c0 + c5.clone() - c8.clone();
    res[1] = c1 + c6.clone();
    res[2] = c2 - c5 + c7.clone() + c8.clone();
    res[3] = c3 - c6 + c8;
    res[4] = c4 - c7;
}
