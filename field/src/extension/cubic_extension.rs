//! Degree-3 extension field using the trinomial `X^3 - X - 1`.
//!
//! Reduction: `X^3 = X + 1`, so `X^4 = X^2 + X`.

use alloc::format;
use alloc::string::ToString;
use alloc::vec::Vec;
use core::array;
use core::fmt::{self, Display, Formatter};
use core::iter::{Product, Sum};
use core::marker::PhantomData;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use itertools::Itertools;
use num_bigint::BigUint;
use p3_util::{as_base_slice, as_base_slice_mut, flatten_to_base, reconstitute_from_base};
use rand::distr::StandardUniform;
use rand::prelude::Distribution;
use serde::{Deserialize, Serialize};

use super::packed_cubic_extension::PackedCubicTrinomialExtensionField;
use super::{HasFrobenius, HasTwoAdicCubicExtension};
use crate::extension::{CubicExtendableAlgebra, CubicTrinomialExtendable};
use crate::field::Field;
use crate::{
    Algebra, BasedVectorSpace, ExtensionField, Packable, PackedFieldExtension,
    PrimeCharacteristicRing, RawDataSerializable, TwoAdicField, field_to_array,
};

/// A degree-3 extension field using `X^3 - X - 1`.
///
/// Elements are `a_0 + a_1 X + a_2 X^2` with coefficients in the base field.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, Serialize, Deserialize, PartialOrd, Ord)]
#[repr(transparent)] // Needed to make various casts safe.
#[must_use]
pub struct CubicTrinomialExtensionField<F, A = F> {
    #[serde(
        with = "p3_util::array_serialization",
        bound(serialize = "A: Serialize", deserialize = "A: Deserialize<'de>")
    )]
    pub(crate) value: [A; 3],
    _phantom: PhantomData<F>,
}

impl<F, A> CubicTrinomialExtensionField<F, A> {
    /// Create an extension field element from an array of base elements.
    ///
    /// Any array is accepted. No reduction is required since
    /// base elements are already valid field elements.
    #[inline]
    pub const fn new(value: [A; 3]) -> Self {
        Self {
            value,
            _phantom: PhantomData,
        }
    }
}

impl<F: Copy> CubicTrinomialExtensionField<F, F> {
    /// Convert a `[[F; D]; N]` array to an array of extension field elements.
    ///
    /// Const version of `input.map(BinomialExtensionField::new)`.
    ///
    /// # Panics
    /// Panics if `N == 0`.
    #[inline]
    pub const fn new_array<const N: usize>(input: [[F; 3]; N]) -> [Self; N] {
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

impl<F: Field, A: Algebra<F>> Default for CubicTrinomialExtensionField<F, A> {
    fn default() -> Self {
        Self::new(array::from_fn(|_| A::ZERO))
    }
}

impl<F: Field, A: Algebra<F>> From<A> for CubicTrinomialExtensionField<F, A> {
    fn from(x: A) -> Self {
        Self::new(field_to_array(x))
    }
}

impl<F, A> From<[A; 3]> for CubicTrinomialExtensionField<F, A> {
    #[inline]
    fn from(x: [A; 3]) -> Self {
        Self {
            value: x,
            _phantom: PhantomData,
        }
    }
}

impl<F: CubicTrinomialExtendable> Packable for CubicTrinomialExtensionField<F> {}

impl<F: CubicTrinomialExtendable, A: Algebra<F>> BasedVectorSpace<A>
    for CubicTrinomialExtensionField<F, A>
{
    const DIMENSION: usize = 3;

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
        (iter.len() == 3).then(|| Self::new(array::from_fn(|_| iter.next().unwrap())))
    }

    #[inline]
    fn flatten_to_base(vec: Vec<Self>) -> Vec<A> {
        // Safety:
        // As `Self` is a `repr(transparent)`, it is stored identically in memory to `[A; D]`
        unsafe { flatten_to_base::<A, Self>(vec) }
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

impl<F: CubicTrinomialExtendable> ExtensionField<F> for CubicTrinomialExtensionField<F>
where
    PackedCubicTrinomialExtensionField<F, F::Packing>: PackedFieldExtension<F, Self>,
{
    type ExtensionPacking = PackedCubicTrinomialExtensionField<F, F::Packing>;

    #[inline]
    fn is_in_basefield(&self) -> bool {
        self.value[1..].iter().all(F::is_zero)
    }

    #[inline]
    fn as_base(&self) -> Option<F> {
        <Self as ExtensionField<F>>::is_in_basefield(self).then(|| self.value[0])
    }
}

impl<F: CubicTrinomialExtendable> HasFrobenius<F> for CubicTrinomialExtensionField<F> {
    /// FrobeniusField automorphisms: x -> x^n, where n is the order of BaseField.
    #[inline]
    fn frobenius(&self) -> Self {
        let a = &self.value;
        let m = F::FROBENIUS_MATRIX;
        let c0 = m[0][0] * a[0] + m[0][1] * a[1] + m[0][2] * a[2];
        let c1 = m[1][0] * a[0] + m[1][1] * a[1] + m[1][2] * a[2];
        let c2 = m[2][0] * a[0] + m[2][1] * a[1] + m[2][2] * a[2];
        Self::new([c0, c1, c2])
    }

    /// Apply Frobenius `count` times: `x → x^{p^count}`.
    #[inline]
    fn repeated_frobenius(&self, count: usize) -> Self {
        match count % 3 {
            0 => *self,
            _ => {
                let mut result = *self;
                for _ in 0..(count % 3) {
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
    /// - `ProdConj(a) = a^{p + p^2}`,
    /// - `Norm(a) = a * ProdConj(a)` is in the base field.
    #[inline]
    fn pseudo_inv(&self) -> Self {
        if self.is_zero() {
            return Self::ZERO;
        }
        let a_exp_p = self.frobenius();
        let prod_conj = (*self * a_exp_p).frobenius();
        let norm_elt = *self * prod_conj;
        debug_assert!(
            norm_elt.value[1].is_zero() && norm_elt.value[2].is_zero(),
            "norm should lie in the base field"
        );
        let norm = norm_elt.value[0];
        prod_conj * norm.inverse()
    }
}

impl<F, A> PrimeCharacteristicRing for CubicTrinomialExtensionField<F, A>
where
    F: CubicTrinomialExtendable,
    A: CubicExtendableAlgebra<F> + Copy,
{
    type PrimeSubfield = <A as PrimeCharacteristicRing>::PrimeSubfield;

    const ZERO: Self = Self::new([A::ZERO; 3]);
    const ONE: Self = Self::new(field_to_array(A::ONE));
    const TWO: Self = Self::new(field_to_array(A::TWO));
    const NEG_ONE: Self = Self::new(field_to_array(A::NEG_ONE));

    #[inline]
    fn from_prime_subfield(f: Self::PrimeSubfield) -> Self {
        <A as PrimeCharacteristicRing>::from_prime_subfield(f).into()
    }

    #[inline]
    fn halve(&self) -> Self {
        Self::new(array::from_fn(|i| self.value[i].halve()))
    }

    #[inline(always)]
    fn square(&self) -> Self {
        let mut res = Self::default();
        A::cubic_square(&self.value, &mut res.value);
        res
    }

    #[inline]
    fn mul_2exp_u64(&self, exp: u64) -> Self {
        Self::new(array::from_fn(|i| self.value[i].mul_2exp_u64(exp)))
    }

    #[inline]
    fn div_2exp_u64(&self, exp: u64) -> Self {
        Self::new(array::from_fn(|i| self.value[i].div_2exp_u64(exp)))
    }

    #[inline]
    fn zero_vec(len: usize) -> Vec<Self> {
        unsafe { reconstitute_from_base(A::zero_vec(len * 3)) }
    }
}

impl<F: CubicTrinomialExtendable> Algebra<F> for CubicTrinomialExtensionField<F> {}

impl<F: CubicTrinomialExtendable> RawDataSerializable for CubicTrinomialExtensionField<F> {
    const NUM_BYTES: usize = F::NUM_BYTES * 3;

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
                .flat_map(|x| (0..3).map(move |i| array::from_fn(|j| x[j].value[i]))),
        )
    }

    #[inline]
    fn into_parallel_u32_streams<const N: usize>(
        input: impl IntoIterator<Item = [Self; N]>,
    ) -> impl IntoIterator<Item = [u32; N]> {
        F::into_parallel_u32_streams(
            input
                .into_iter()
                .flat_map(|x| (0..3).map(move |i| array::from_fn(|j| x[j].value[i]))),
        )
    }

    #[inline]
    fn into_parallel_u64_streams<const N: usize>(
        input: impl IntoIterator<Item = [Self; N]>,
    ) -> impl IntoIterator<Item = [u64; N]> {
        F::into_parallel_u64_streams(
            input
                .into_iter()
                .flat_map(|x| (0..3).map(move |i| array::from_fn(|j| x[j].value[i]))),
        )
    }
}

impl<F: CubicTrinomialExtendable> Field for CubicTrinomialExtensionField<F> {
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
        unsafe {
            let base_slice_1 = as_base_slice_mut(slice_1);
            let base_slice_2 = as_base_slice(slice_2);
            F::add_slices(base_slice_1, base_slice_2);
        }
    }

    #[inline]
    fn order() -> BigUint {
        F::order().pow(3)
    }
}

impl<F: CubicTrinomialExtendable> Display for CubicTrinomialExtensionField<F> {
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

impl<F, A> Neg for CubicTrinomialExtensionField<F, A>
where
    F: CubicTrinomialExtendable,
    A: Algebra<F>,
{
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        Self::new(self.value.map(A::neg))
    }
}

impl<F, A> Add for CubicTrinomialExtensionField<F, A>
where
    F: CubicTrinomialExtendable,
    A: CubicExtendableAlgebra<F>,
{
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::new(A::cubic_add(&self.value, &rhs.value))
    }
}

impl<F, A> Add<A> for CubicTrinomialExtensionField<F, A>
where
    F: CubicTrinomialExtendable,
    A: Algebra<F>,
{
    type Output = Self;

    #[inline]
    fn add(mut self, rhs: A) -> Self {
        self.value[0] += rhs;
        self
    }
}

impl<F, A> AddAssign for CubicTrinomialExtensionField<F, A>
where
    F: CubicTrinomialExtendable,
    A: CubicExtendableAlgebra<F>,
{
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.value = A::cubic_add(&self.value, &rhs.value);
    }
}

impl<F, A> AddAssign<A> for CubicTrinomialExtensionField<F, A>
where
    F: CubicTrinomialExtendable,
    A: Algebra<F>,
{
    #[inline]
    fn add_assign(&mut self, rhs: A) {
        self.value[0] += rhs;
    }
}

impl<F, A> Sum for CubicTrinomialExtensionField<F, A>
where
    F: CubicTrinomialExtendable,
    A: CubicExtendableAlgebra<F> + Copy,
{
    #[inline]
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|acc, x| acc + x).unwrap_or(Self::ZERO)
    }
}

impl<F, A> Sub for CubicTrinomialExtensionField<F, A>
where
    F: CubicTrinomialExtendable,
    A: CubicExtendableAlgebra<F>,
{
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::new(A::cubic_sub(&self.value, &rhs.value))
    }
}

impl<F, A> Sub<A> for CubicTrinomialExtensionField<F, A>
where
    F: CubicTrinomialExtendable,
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

impl<F, A> SubAssign for CubicTrinomialExtensionField<F, A>
where
    F: CubicTrinomialExtendable,
    A: CubicExtendableAlgebra<F>,
{
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.value = A::cubic_sub(&self.value, &rhs.value);
    }
}

impl<F, A> SubAssign<A> for CubicTrinomialExtensionField<F, A>
where
    F: CubicTrinomialExtendable,
    A: Algebra<F>,
{
    #[inline]
    fn sub_assign(&mut self, rhs: A) {
        self.value[0] -= rhs;
    }
}

impl<F, A> Mul for CubicTrinomialExtensionField<F, A>
where
    F: CubicTrinomialExtendable,
    A: CubicExtendableAlgebra<F>,
{
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let mut res = Self::default();
        A::cubic_mul(&self.value, &rhs.value, &mut res.value);
        res
    }
}

impl<F, A> Mul<A> for CubicTrinomialExtensionField<F, A>
where
    F: CubicTrinomialExtendable,
    A: CubicExtendableAlgebra<F>,
{
    type Output = Self;

    #[inline]
    fn mul(self, rhs: A) -> Self {
        Self::new(A::cubic_base_mul(self.value, rhs))
    }
}

impl<F, A> MulAssign for CubicTrinomialExtensionField<F, A>
where
    F: CubicTrinomialExtendable,
    A: CubicExtendableAlgebra<F>,
{
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.clone() * rhs;
    }
}

impl<F, A> MulAssign<A> for CubicTrinomialExtensionField<F, A>
where
    F: CubicTrinomialExtendable,
    A: CubicExtendableAlgebra<F>,
{
    #[inline]
    fn mul_assign(&mut self, rhs: A) {
        *self = self.clone() * rhs;
    }
}

impl<F, A> Product for CubicTrinomialExtensionField<F, A>
where
    F: CubicTrinomialExtendable,
    A: CubicExtendableAlgebra<F> + Copy,
{
    #[inline]
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|acc, x| acc * x).unwrap_or(Self::ONE)
    }
}

impl<F> Div for CubicTrinomialExtensionField<F>
where
    F: CubicTrinomialExtendable,
{
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        self * rhs.inverse()
    }
}

impl<F> DivAssign for CubicTrinomialExtensionField<F>
where
    F: CubicTrinomialExtendable,
{
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl<F: CubicTrinomialExtendable> Distribution<CubicTrinomialExtensionField<F>> for StandardUniform
where
    Self: Distribution<F>,
{
    #[inline]
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> CubicTrinomialExtensionField<F> {
        CubicTrinomialExtensionField::new(array::from_fn(|_| self.sample(rng)))
    }
}

impl<F: CubicTrinomialExtendable + HasTwoAdicCubicExtension> TwoAdicField
    for CubicTrinomialExtensionField<F>
{
    const TWO_ADICITY: usize = F::EXT_TWO_ADICITY;

    #[inline]
    fn two_adic_generator(bits: usize) -> Self {
        Self::new(F::ext_two_adic_generator(bits))
    }
}

/// Multiply in `R[X]/(X^3 - X - 1)`.
#[inline]
pub fn trinomial_cubic_mul<R: PrimeCharacteristicRing>(a: &[R; 3], b: &[R; 3], res: &mut [R; 3]) {
    let c0 = a[0].dup() * b[0].dup();
    let c1 = R::dot_product::<2>(&[a[0].dup(), a[1].dup()], &[b[1].dup(), b[0].dup()]);
    let c2 = R::dot_product::<3>(
        &[a[0].dup(), a[1].dup(), a[2].dup()],
        &[b[2].dup(), b[1].dup(), b[0].dup()],
    );
    let c3 = R::dot_product::<2>(&[a[1].dup(), a[2].dup()], &[b[2].dup(), b[1].dup()]);
    let c4 = a[2].dup() * b[2].dup();

    res[0] = c0 + c3.dup();
    res[1] = c1 + c3 + c4.dup();
    res[2] = c2 + c4;
}

#[inline]
pub(super) fn cubic_square<R: PrimeCharacteristicRing>(a: &[R; 3], res: &mut [R; 3]) {
    let a0_sq = a[0].square();
    let a1_sq = a[1].square();
    let a2_sq = a[2].square();
    let a1_a2 = a[1].dup() * a[2].dup();

    res[0] = a0_sq + a1_a2.double();
    res[1] = (a[0].dup() * a[1].dup() + a1_a2).double() + a2_sq.dup();
    res[2] = (a[0].dup() * a[2].dup()).double() + a1_sq + a2_sq;
}
