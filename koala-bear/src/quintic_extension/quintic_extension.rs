use alloc::format;
use alloc::string::ToString;
use alloc::vec::Vec;
use core::array;
use core::fmt::{self, Debug, Display, Formatter};
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use p3_field::extension::HasFrobenius;

use itertools::Itertools;
use num_bigint::BigUint;
use p3_util::{as_base_slice, as_base_slice_mut, flatten_to_base, reconstitute_from_base};
use rand::distr::StandardUniform;
use rand::prelude::Distribution;
use serde::{Deserialize, Serialize};

use super::packed_quintic_extension::PackedQuinticExtensionField;
use crate::QuinticExtendable;
use p3_field::{
    Algebra, BasedVectorSpace, ExtensionField, Field, Packable, PrimeCharacteristicRing,
    RawDataSerializable, TwoAdicField, field_to_array,
};

/// Quintic Extension Field (degree 5), specifically designed for Koala-Bear
/// Irreducible polynomial: X^5 + X^2 - 1
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, Serialize, Deserialize, PartialOrd, Ord)]
#[repr(transparent)] // Needed to make various casts safe.
#[must_use]
pub struct QuinticExtensionField<F> {
    #[serde(
        with = "p3_util::array_serialization",
        bound(serialize = "F: Serialize", deserialize = "F: Deserialize<'de>")
    )]
    pub(crate) value: [F; 5],
}

impl<F> QuinticExtensionField<F> {
    pub(crate) const fn new(value: [F; 5]) -> Self {
        Self { value }
    }
}

impl<F: Field> Default for QuinticExtensionField<F> {
    fn default() -> Self {
        Self::new(array::from_fn(|_| F::ZERO))
    }
}

impl<F: Field> From<F> for QuinticExtensionField<F> {
    fn from(x: F) -> Self {
        Self::new(field_to_array(x))
    }
}

impl<F: QuinticExtendable> Packable for QuinticExtensionField<F> {}

impl<F: QuinticExtendable> BasedVectorSpace<F> for QuinticExtensionField<F> {
    const DIMENSION: usize = 5;

    #[inline]
    fn as_basis_coefficients_slice(&self) -> &[F] {
        &self.value
    }

    #[inline]
    fn from_basis_coefficients_fn<Fn: FnMut(usize) -> F>(f: Fn) -> Self {
        Self::new(array::from_fn(f))
    }

    #[inline]
    fn from_basis_coefficients_iter<I: ExactSizeIterator<Item = F>>(mut iter: I) -> Option<Self> {
        (iter.len() == 5).then(|| Self::new(array::from_fn(|_| iter.next().unwrap()))) // The unwrap is safe as we just checked the length of iter.
    }

    #[inline]
    fn flatten_to_base(vec: Vec<Self>) -> Vec<F> {
        unsafe {
            // Safety:
            // As `Self` is a `repr(transparent)`, it is stored identically in memory to `[A; 5]`
            flatten_to_base::<F, Self>(vec)
        }
    }

    #[inline]
    fn reconstitute_from_base(vec: Vec<F>) -> Vec<Self> {
        unsafe {
            // Safety:
            // As `Self` is a `repr(transparent)`, it is stored identically in memory to `[A; 5]`
            reconstitute_from_base::<F, Self>(vec)
        }
    }
}

impl<F: QuinticExtendable> ExtensionField<F> for QuinticExtensionField<F> {
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

impl<F: QuinticExtendable> HasFrobenius<F> for QuinticExtensionField<F> {
    /// FrobeniusField automorphisms: x -> x^n, where n is the order of BaseField.
    #[inline]
    fn frobenius(&self) -> Self {
        let mut res = Self::ZERO;
        res.value[0] = self.value[0];
        for i in 0..4 {
            for j in 0..5 {
                res.value[j] += self.value[i + 1] * F::FROBENIUS_MATRIX[i][j];
            }
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
        } else if count >= 5 {
            // x |-> x^(n^D) is the identity, so x^(n^count) ==
            // x^(n^(count % D))
            return self.repeated_frobenius(count % 5);
        }

        // TODO optimize as it's done in `binomial_extension.rs`
        let mut res = self.frobenius();
        for _ in 1..count {
            res = res.frobenius();
        }
        res
    }

    #[inline]
    fn frobenius_inv(&self) -> Self {
        unimplemented!()
    }
}

impl<F> PrimeCharacteristicRing for QuinticExtensionField<F>
where
    F: QuinticExtendable,
{
    type PrimeSubfield = <F as PrimeCharacteristicRing>::PrimeSubfield;

    const ZERO: Self = Self::new([F::ZERO; 5]);

    const ONE: Self = Self::new(field_to_array(F::ONE));

    const TWO: Self = Self::new(field_to_array(F::TWO));

    const NEG_ONE: Self = Self::new(field_to_array(F::NEG_ONE));

    #[inline]
    fn from_prime_subfield(f: Self::PrimeSubfield) -> Self {
        <F as PrimeCharacteristicRing>::from_prime_subfield(f).into()
    }

    #[inline]
    fn halve(&self) -> Self {
        Self::new(self.value.clone().map(|x| x.halve()))
    }

    #[inline(always)]
    fn square(&self) -> Self {
        let mut res = Self::default();
        quintic_square(&self.value, &mut res.value);
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
        unsafe { reconstitute_from_base(F::zero_vec(len * 5)) }
    }
}

impl<F: QuinticExtendable> Algebra<F> for QuinticExtensionField<F> {}

impl<F: QuinticExtendable> RawDataSerializable for QuinticExtensionField<F> {
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

impl<F: QuinticExtendable> Field for QuinticExtensionField<F> {
    type Packing = Self;

    const GENERATOR: Self = Self::new(F::EXT_GENERATOR);

    fn try_inverse(&self) -> Option<Self> {
        if self.is_zero() {
            return None;
        }

        Some(quintic_inv(self))
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
        F::order().pow(5)
    }
}

impl<F> Display for QuinticExtensionField<F>
where
    F: QuinticExtendable,
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

impl<F> Neg for QuinticExtensionField<F>
where
    F: QuinticExtendable,
{
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        Self::new(self.value.map(F::neg))
    }
}

impl<F> Add for QuinticExtensionField<F>
where
    F: QuinticExtendable,
{
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        let value = F::kb_quintic_add(&self.value, &rhs.value);
        Self::new(value)
    }
}

impl<F> Add<F> for QuinticExtensionField<F>
where
    F: QuinticExtendable,
{
    type Output = Self;

    #[inline]
    fn add(mut self, rhs: F) -> Self {
        self.value[0] += rhs;
        self
    }
}

impl<F> AddAssign for QuinticExtensionField<F>
where
    F: QuinticExtendable,
{
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..5 {
            self.value[i] += rhs.value[i].clone();
        }
    }
}

impl<F> AddAssign<F> for QuinticExtensionField<F>
where
    F: QuinticExtendable,
{
    #[inline]
    fn add_assign(&mut self, rhs: F) {
        self.value[0] += rhs;
    }
}

impl<F> Sum for QuinticExtensionField<F>
where
    F: QuinticExtendable,
{
    #[inline]
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|acc, x| acc + x).unwrap_or(Self::ZERO)
    }
}

impl<F> Sub for QuinticExtensionField<F>
where
    F: QuinticExtendable,
{
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        let value = F::kb_quintic_sub(&self.value, &rhs.value);
        Self::new(value)
    }
}

impl<F> Sub<F> for QuinticExtensionField<F>
where
    F: QuinticExtendable,
{
    type Output = Self;

    #[inline]
    fn sub(self, rhs: F) -> Self {
        let mut res = self.value;
        res[0] -= rhs;
        Self::new(res)
    }
}

impl<F> SubAssign for QuinticExtensionField<F>
where
    F: QuinticExtendable,
{
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        for i in 0..5 {
            self.value[i] -= rhs.value[i].clone();
        }
    }
}

impl<F> SubAssign<F> for QuinticExtensionField<F>
where
    F: QuinticExtendable,
{
    #[inline]
    fn sub_assign(&mut self, rhs: F) {
        self.value[0] -= rhs;
    }
}

impl<F> Mul for QuinticExtensionField<F>
where
    F: QuinticExtendable,
{
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let a = self.value;
        let b = rhs.value;
        let mut res = Self::default();

        F::kb_quintic_mul(&a, &b, &mut res.value);

        res
    }
}

impl<F> Mul<F> for QuinticExtensionField<F>
where
    F: QuinticExtendable,
{
    type Output = Self;

    #[inline]
    fn mul(self, rhs: F) -> Self {
        Self::new(F::kb_quintic_base_mul(self.value, rhs))
    }
}

impl<F> MulAssign for QuinticExtensionField<F>
where
    F: QuinticExtendable,
{
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.clone() * rhs;
    }
}

impl<F> MulAssign<F> for QuinticExtensionField<F>
where
    F: QuinticExtendable,
{
    #[inline]
    fn mul_assign(&mut self, rhs: F) {
        *self = self.clone() * rhs;
    }
}

impl<F> Product for QuinticExtensionField<F>
where
    F: QuinticExtendable,
{
    #[inline]
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|acc, x| acc * x).unwrap_or(Self::ONE)
    }
}

impl<F> Div for QuinticExtensionField<F>
where
    F: QuinticExtendable,
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
    F: QuinticExtendable,
{
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl<F: QuinticExtendable> Distribution<QuinticExtensionField<F>> for StandardUniform
where
    Self: Distribution<F>,
{
    #[inline]
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> QuinticExtensionField<F> {
        QuinticExtensionField::new(array::from_fn(|_| self.sample(rng)))
    }
}

impl<F: TwoAdicField + QuinticExtendable> TwoAdicField for QuinticExtensionField<F> {
    const TWO_ADICITY: usize = F::TWO_ADICITY;

    #[inline]
    fn two_adic_generator(bits: usize) -> Self {
        F::two_adic_generator(bits).into()
    }
}

/// In our extension field: X^5 = 1 - X^2
///
/// (a0 + a1X + a2X^2 + a3X^3 + a4X^4) * (b0 + b1X + b2X^2 + b3X^3 + b4X^4) =
//   (a0b0 + a1b4 + a2b3 + a3b2 + a4b1 - a4b4)
// + (a0b1 + a1b0 + a2b4 + a3b3 + a4*b2).X
// + (a0b2 + a1b1 + a2b0 - a1b4 - a2b3 - a3b2 - a4b1 + a3b4 + a4b3 + a4b4).X^2
// + (a0b3 + a1b2 + a2b1 + a3b0 - a2b4 - a3b3 - a4b2 + a4b4).X^3
// + (a0b4 + a1b3 + a2b2 + a3b1 + a4b0 - a3b4 - a4*b3).X^4
pub fn kb_quintic_mul<F, R, R2>(a: &[R; 5], b: &[R2; 5], res: &mut [R; 5])
where
    F: Field,
    R: Algebra<F> + Algebra<R2>,
    R2: Algebra<F>,
{
    // Convert b to R type for computation
    let b_r: [R; 5] = [
        b[0].clone().into(),
        b[1].clone().into(),
        b[2].clone().into(),
        b[3].clone().into(),
        b[4].clone().into(),
    ];

    let b_0_minus_3 = b_r[0].clone() - b_r[3].clone();
    let b_1_minus_4 = b_r[1].clone() - b_r[4].clone();
    let b_4_minus_2 = b_r[4].clone() - b_r[2].clone();

    // Constant term = a0*b0 + a1*b4 + a2*b3 + a3*b2 + a4*b1 - a4*b4
    res[0] = R::dot_product::<5>(
        a,
        &[
            b_r[0].clone(),
            b_r[4].clone(),
            b_r[3].clone(),
            b_r[2].clone(),
            b_1_minus_4.clone(),
        ],
    );

    // Linear term = a0*b1 + a1*b0 + a2*b4 + a3*b3 + a4*b2
    res[1] = R::dot_product::<5>(
        a,
        &[
            b_r[1].clone(),
            b_r[0].clone(),
            b_r[4].clone(),
            b_r[3].clone(),
            b_r[2].clone(),
        ],
    );

    // Square term = a0*b2 + a1*b1 + a2*b0 - a1*b4 - a2*b3 - a3*b2 - a4*b1 + a3*b4 + a4*b3 + a4*b4
    res[2] = R::dot_product::<5>(
        a,
        &[
            b_r[2].clone(),
            b_1_minus_4.clone(),
            b_0_minus_3.clone(),
            b_4_minus_2.clone(),
            b_r[3].clone() - b_1_minus_4.clone(),
        ],
    );

    // Cubic term = a0*b3 + a1*b2 + a2*b1 + a3*b0 - a2*b4 - a3*b3 - a4*b2 + a4*b4
    res[3] = R::dot_product::<5>(
        a,
        &[
            b_r[3].clone(),
            b_r[2].clone(),
            b_1_minus_4.clone(),
            b_0_minus_3.clone(),
            b_4_minus_2.clone(),
        ],
    );

    // Quartic term = a0*b4 + a1*b3 + a2*b2 + a3*b1 + a4*b0 - a3*b4 - a4*b3
    res[4] = R::dot_product::<5>(
        a,
        &[
            b_r[4].clone(),
            b_r[3].clone(),
            b_r[2].clone(),
            b_1_minus_4.clone(),
            b_0_minus_3.clone(),
        ],
    );
}

/*
In our extension field: X^5 = 1 - X^2

(a0 + a1*X + a2*X^2 + a3*X^3 + a4*X^4)^2 =
  (a0^2 + 2*a1*a4 + 2*a2*a3 - a4^2)
+ (2*a0*a1 + a3^2 + 2*a2*a4).X
+ (a1^2 + 2*a0*a2 - 2*a1*a4 - 2*a2*a3 + 2*a3*a4 + a4^2).X^2
+ (2*a0*a3 + 2*a1*a2 - a3^2 - 2*a2*a4 + a4^2).X^3
+ (a2^2 + 2*a0*a4 + 2*a1*a3 - 2*a3*a4).X^4

*/
#[inline]
pub(crate) fn quintic_square<F, R>(a: &[R; 5], res: &mut [R; 5])
where
    F: Field,
    R: Algebra<F>,
{
    let two_a0 = a[0].double();
    let two_a1 = a[1].double();
    let two_a2 = a[2].double();
    let two_a3 = a[3].double();

    let two_a1_a4 = two_a1.clone() * a[4].clone();
    let two_a2_a3 = two_a2.clone() * a[3].clone();
    let two_a2_a4 = two_a2.clone() * a[4].clone();
    let two_a3_a4 = two_a3.clone() * a[4].clone();

    let a3_square = a[3].square();
    let a4_square = a[4].square();

    // Constant term = a0^2 + 2*a1*a4 + 2*a2*a3 - a4^2
    res[0] = R::dot_product(
        &[a[0].clone(), two_a1.clone()],
        &[a[0].clone(), a[4].clone()],
    ) + two_a2_a3.clone()
        - a4_square.clone();

    // Linear term = 2*a0*a1 + a3^2 + 2*a2*a4
    res[1] = two_a0.clone() * a[1].clone() + a3_square.clone() + two_a2_a4.clone();

    // Square term = a1^2 + 2*a0*a2 - 2*a1*a4 - 2*a2*a3 + 2*a3*a4 + a4^2
    res[2] = a[1].square() + two_a0.clone() * a[2].clone() - two_a1_a4.clone() - two_a2_a3.clone()
        + two_a3_a4.clone()
        + a4_square.clone();

    // Cubic term = 2*a0*a3 + 2*a1*a2 - a3^2 - 2*a2*a4 + a4^2
    res[3] = R::dot_product(
        &[two_a0.clone(), two_a1.clone()],
        &[a[3].clone(), a[2].clone()],
    ) - a3_square.clone()
        - two_a2_a4.clone()
        + a4_square.clone();

    // Quartic term = a2^2 + 2*a0*a4 + 2*a1*a3 - 2*a3*a4
    res[4] = R::dot_product(
        &[two_a0.clone(), two_a1.clone()],
        &[a[4].clone(), a[3].clone()],
    ) + a[2].square()
        - two_a3_a4.clone();
}

/// Compute the inverse of a quintic binomial extension field element.
#[inline]
fn quintic_inv<F: QuinticExtendable>(a: &QuinticExtensionField<F>) -> QuinticExtensionField<F> {
    // Writing 'a' for self, we need to compute: `prod_conj = a^{q^4 + q^3 + q^2 + q}`
    let a_exp_q = a.frobenius();
    let a_exp_q_plus_q_sq = (*a * a_exp_q).frobenius();
    let prod_conj = a_exp_q_plus_q_sq * a_exp_q_plus_q_sq.repeated_frobenius(2);

    // norm = a * prod_conj is in the base field, so only compute that
    // coefficient rather than the full product.
    let norm = F::dot_product::<5>(
        &a.value,
        &[
            prod_conj.value[0].clone(),
            prod_conj.value[4].clone(),
            prod_conj.value[3].clone(),
            prod_conj.value[2].clone(),
            prod_conj.value[1].clone() - prod_conj.value[4].clone(),
        ],
    );

    debug_assert_eq!(QuinticExtensionField::<F>::from(norm), *a * prod_conj);

    prod_conj * norm.inverse()
}

// fn compute_frobenius_matrix<F: QuinticExtendable>() {
//     for i in 1..5 {
//         let mut x = QuinticExtensionField::<F>::default();
//         x.value[i] = F::ONE;
//         let x = x.exp_u64(F::order().to_u64_digits()[0]);
//         print!("\n[");
//         for j in 0..5 {
//             print!(" MontyField31::new({}), ", x.value[j]);
//         }
//         print!("], ");
//     }
//     std::io::Write::flush(&mut std::io::stdout()).unwrap();
// }
