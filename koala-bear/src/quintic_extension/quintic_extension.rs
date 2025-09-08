use alloc::format;
use alloc::string::ToString;
use alloc::vec::Vec;
use core::array;
use core::fmt::{self, Debug, Display, Formatter};
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use itertools::Itertools;
use num_bigint::BigUint;
use p3_field::extension::HasFrobenius;
use p3_monty_31::{MontyParameters, base_mul_packed};

use p3_util::{as_base_slice, as_base_slice_mut, flatten_to_base, reconstitute_from_base};
use rand::distr::StandardUniform;
use rand::prelude::Distribution;
use serde::{Deserialize, Serialize};

use crate::{KoalaBear, KoalaBearParameters};

use super::PackedQuinticExtensionField;
use p3_field::{
    Algebra, BasedVectorSpace, ExtensionField, Field, Packable, PrimeCharacteristicRing,
    RawDataSerializable, TwoAdicField, field_to_array, packed_mod_add, packed_mod_sub,
};

/// Quintic Extension Field (degree 5), specifically designed for Koala-Bear
/// Irreducible polynomial: X^5 + X^2 - 1
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, Serialize, Deserialize, PartialOrd, Ord)]
#[repr(transparent)] // Needed to make various casts safe.
#[must_use]
pub struct QuinticExtensionField {
    #[serde(
        with = "p3_util::array_serialization",
        bound(
            serialize = "KoalaBear: Serialize",
            deserialize = "KoalaBear: Deserialize<'de>"
        )
    )]
    pub(crate) value: [KoalaBear; 5],
}

impl QuinticExtensionField {
    pub(crate) const fn new(value: [KoalaBear; 5]) -> Self {
        Self { value }
    }
}

impl Default for QuinticExtensionField {
    fn default() -> Self {
        Self::new(array::from_fn(|_| KoalaBear::ZERO))
    }
}

impl From<KoalaBear> for QuinticExtensionField {
    fn from(x: KoalaBear) -> Self {
        Self::new(field_to_array(x))
    }
}

impl Packable for QuinticExtensionField {}

impl BasedVectorSpace<KoalaBear> for QuinticExtensionField {
    const DIMENSION: usize = 5;

    #[inline]
    fn as_basis_coefficients_slice(&self) -> &[KoalaBear] {
        &self.value
    }

    #[inline]
    fn from_basis_coefficients_fn<Fn: FnMut(usize) -> KoalaBear>(f: Fn) -> Self {
        Self::new(array::from_fn(f))
    }

    #[inline]
    fn from_basis_coefficients_iter<I: ExactSizeIterator<Item = KoalaBear>>(
        mut iter: I,
    ) -> Option<Self> {
        (iter.len() == 5).then(|| Self::new(array::from_fn(|_| iter.next().unwrap()))) // The unwrap is safe as we just checked the length of iter.
    }

    #[inline]
    fn flatten_to_base(vec: Vec<Self>) -> Vec<KoalaBear> {
        unsafe {
            // Safety:
            // As `Self` is a `repr(transparent)`, it is stored identically in memory to `[A; 5]`
            flatten_to_base::<KoalaBear, Self>(vec)
        }
    }

    #[inline]
    fn reconstitute_from_base(vec: Vec<KoalaBear>) -> Vec<Self> {
        unsafe {
            // Safety:
            // As `Self` is a `repr(transparent)`, it is stored identically in memory to `[A; 5]`
            reconstitute_from_base::<KoalaBear, Self>(vec)
        }
    }
}

impl ExtensionField<KoalaBear> for QuinticExtensionField {
    type ExtensionPacking = PackedQuinticExtensionField;

    #[inline]
    fn is_in_basefield(&self) -> bool {
        self.value[1..].iter().all(KoalaBear::is_zero)
    }

    #[inline]
    fn as_base(&self) -> Option<KoalaBear> {
        <Self as ExtensionField<KoalaBear>>::is_in_basefield(self).then(|| self.value[0])
    }
}

const FROBENIUS_MATRIX: [[KoalaBear; 5]; 4] = [
    [
        KoalaBear::new(1576402667),
        KoalaBear::new(1173144480),
        KoalaBear::new(1567662457),
        KoalaBear::new(1206866823),
        KoalaBear::new(2428146),
    ],
    [
        KoalaBear::new(1680345488),
        KoalaBear::new(1381986),
        KoalaBear::new(615237464),
        KoalaBear::new(1380104858),
        KoalaBear::new(295431824),
    ],
    [
        KoalaBear::new(441230756),
        KoalaBear::new(323126830),
        KoalaBear::new(704986542),
        KoalaBear::new(1445620072),
        KoalaBear::new(503505220),
    ],
    [
        KoalaBear::new(1364444097),
        KoalaBear::new(1144738982),
        KoalaBear::new(2008416047),
        KoalaBear::new(143367062),
        KoalaBear::new(1027410849),
    ],
];

impl HasFrobenius<KoalaBear> for QuinticExtensionField {
    /// FrobeniusField automorphisms: x -> x^n, where n is the order of BaseField.
    #[inline]
    fn frobenius(&self) -> Self {
        let mut res = Self::ZERO;
        res.value[0] = self.value[0];
        for i in 0..4 {
            for j in 0..5 {
                res.value[j] += self.value[i + 1] * FROBENIUS_MATRIX[i][j];
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

impl PrimeCharacteristicRing for QuinticExtensionField {
    type PrimeSubfield = <KoalaBear as PrimeCharacteristicRing>::PrimeSubfield;

    const ZERO: Self = Self::new([KoalaBear::ZERO; 5]);

    const ONE: Self = Self::new(field_to_array(KoalaBear::ONE));

    const TWO: Self = Self::new(field_to_array(KoalaBear::TWO));

    const NEG_ONE: Self = Self::new(field_to_array(KoalaBear::NEG_ONE));

    #[inline]
    fn from_prime_subfield(f: Self::PrimeSubfield) -> Self {
        <KoalaBear as PrimeCharacteristicRing>::from_prime_subfield(f).into()
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
        unsafe { reconstitute_from_base(KoalaBear::zero_vec(len * 5)) }
    }
}

impl Algebra<KoalaBear> for QuinticExtensionField {}

impl RawDataSerializable for QuinticExtensionField {
    const NUM_BYTES: usize = KoalaBear::NUM_BYTES * 5;

    #[inline]
    fn into_bytes(self) -> impl IntoIterator<Item = u8> {
        self.value.into_iter().flat_map(|x| x.into_bytes())
    }

    #[inline]
    fn into_byte_stream(input: impl IntoIterator<Item = Self>) -> impl IntoIterator<Item = u8> {
        KoalaBear::into_byte_stream(input.into_iter().flat_map(|x| x.value))
    }

    #[inline]
    fn into_u32_stream(input: impl IntoIterator<Item = Self>) -> impl IntoIterator<Item = u32> {
        KoalaBear::into_u32_stream(input.into_iter().flat_map(|x| x.value))
    }

    #[inline]
    fn into_u64_stream(input: impl IntoIterator<Item = Self>) -> impl IntoIterator<Item = u64> {
        KoalaBear::into_u64_stream(input.into_iter().flat_map(|x| x.value))
    }

    #[inline]
    fn into_parallel_byte_streams<const N: usize>(
        input: impl IntoIterator<Item = [Self; N]>,
    ) -> impl IntoIterator<Item = [u8; N]> {
        KoalaBear::into_parallel_byte_streams(
            input
                .into_iter()
                .flat_map(|x| (0..5).map(move |i| array::from_fn(|j| x[j].value[i]))),
        )
    }

    #[inline]
    fn into_parallel_u32_streams<const N: usize>(
        input: impl IntoIterator<Item = [Self; N]>,
    ) -> impl IntoIterator<Item = [u32; N]> {
        KoalaBear::into_parallel_u32_streams(
            input
                .into_iter()
                .flat_map(|x| (0..5).map(move |i| array::from_fn(|j| x[j].value[i]))),
        )
    }

    #[inline]
    fn into_parallel_u64_streams<const N: usize>(
        input: impl IntoIterator<Item = [Self; N]>,
    ) -> impl IntoIterator<Item = [u64; N]> {
        KoalaBear::into_parallel_u64_streams(
            input
                .into_iter()
                .flat_map(|x| (0..5).map(move |i| array::from_fn(|j| x[j].value[i]))),
        )
    }
}

impl Field for QuinticExtensionField {
    type Packing = Self;

    const GENERATOR: Self = Self::new(KoalaBear::new_array([2, 1, 0, 0, 0]));

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

            KoalaBear::add_slices(base_slice_1, base_slice_2);
        }
    }

    #[inline]
    fn order() -> BigUint {
        KoalaBear::order().pow(5)
    }

    fn is_zero(&self) -> bool {
        *self == Self::ZERO
    }

    fn is_one(&self) -> bool {
        *self == Self::ONE
    }

    fn inverse(&self) -> Self {
        self.try_inverse().expect("Tried to invert zero")
    }

    fn bits() -> usize {
        Self::order().bits() as usize
    }
}

impl Display for QuinticExtensionField {
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

impl Neg for QuinticExtensionField {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        Self::new(self.value.map(KoalaBear::neg))
    }
}

impl Add for QuinticExtensionField {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        // TODO duplicated with binomial_add
        let mut res = [KoalaBear::ZERO; 5];
        unsafe {
            // Safe as Self is repr(transparent) and stores a single u32.
            let a: &[u32; 5] = &*(self.value.as_ptr() as *const [u32; 5]);
            let b: &[u32; 5] = &*(rhs.value.as_ptr() as *const [u32; 5]);
            let res: &mut [u32; 5] = &mut *(res.as_mut_ptr() as *mut [u32; 5]);

            packed_mod_add(
                a,
                b,
                res,
                KoalaBearParameters::PRIME,
                add::<KoalaBearParameters>,
            );
        }
        Self::new(res)
    }
}

/// TODO duplicated
///
/// Add two integers modulo `P = MP::PRIME`.
///
/// Assumes that `P` is less than `2^31` and `a + b <= 2P` for all array pairs `a, b`.
/// If the inputs are not in this range, the result may be incorrect.
/// The result will be in the range `[0, P]` and equal to `(a + b) mod P`.
/// It will be equal to `P` if and only if `a + b = 2P` so provided `a + b < 2P`
/// the result is guaranteed to be less than `P`.
#[inline]
#[must_use]
pub(crate) fn add<MP: MontyParameters>(lhs: u32, rhs: u32) -> u32 {
    let mut sum = lhs + rhs;
    let (corr_sum, over) = sum.overflowing_sub(MP::PRIME);
    if !over {
        sum = corr_sum;
    }
    sum
}

impl Add<KoalaBear> for QuinticExtensionField {
    type Output = Self;

    #[inline]
    fn add(mut self, rhs: KoalaBear) -> Self {
        self.value[0] += rhs;
        self
    }
}

impl AddAssign for QuinticExtensionField {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..5 {
            self.value[i] += rhs.value[i].clone();
        }
    }
}

impl AddAssign<KoalaBear> for QuinticExtensionField {
    #[inline]
    fn add_assign(&mut self, rhs: KoalaBear) {
        self.value[0] += rhs;
    }
}

impl Sum for QuinticExtensionField {
    #[inline]
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|acc, x| acc + x).unwrap_or(Self::ZERO)
    }
}

impl Sub for QuinticExtensionField {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        // TODO duplicated with binomial_sub
        let mut res: [p3_monty_31::MontyField31<KoalaBearParameters>; 5] = [KoalaBear::ZERO; 5];
        unsafe {
            // Safe as Self is repr(transparent) and stores a single u32.
            let a: &[u32; 5] = &*(self.value.as_ptr() as *const [u32; 5]);
            let b: &[u32; 5] = &*(rhs.value.as_ptr() as *const [u32; 5]);
            let res: &mut [u32; 5] = &mut *(res.as_mut_ptr() as *mut [u32; 5]);

            packed_mod_sub(
                a,
                b,
                res,
                KoalaBearParameters::PRIME,
                sub::<KoalaBearParameters>,
            );
        }
        Self::new(res)
    }
}

// TODO duplicated
#[inline]
#[must_use]
pub(crate) fn sub<MP: MontyParameters>(lhs: u32, rhs: u32) -> u32 {
    let (mut diff, over) = lhs.overflowing_sub(rhs);
    let corr = if over { MP::PRIME } else { 0 };
    diff = diff.wrapping_add(corr);
    diff
}

impl Sub<KoalaBear> for QuinticExtensionField {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: KoalaBear) -> Self {
        let mut res = self.value;
        res[0] -= rhs;
        Self::new(res)
    }
}

impl SubAssign for QuinticExtensionField {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        for i in 0..5 {
            self.value[i] -= rhs.value[i].clone();
        }
    }
}

impl SubAssign<KoalaBear> for QuinticExtensionField {
    #[inline]
    fn sub_assign(&mut self, rhs: KoalaBear) {
        self.value[0] -= rhs;
    }
}

impl Mul for QuinticExtensionField {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let a = &self.value;
        let b = &rhs.value;
        let mut res = Self::default();

        kb_quintic_mul_packed(a, b, &mut res.value);

        res
    }
}

impl Mul<KoalaBear> for QuinticExtensionField {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: KoalaBear) -> Self {
        let mut res = [KoalaBear::ZERO; 5];
        base_mul_packed(self.value, rhs, &mut res);
        Self::new(res)
    }
}

impl MulAssign for QuinticExtensionField {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.clone() * rhs;
    }
}

impl MulAssign<KoalaBear> for QuinticExtensionField {
    #[inline]
    fn mul_assign(&mut self, rhs: KoalaBear) {
        *self = self.clone() * rhs;
    }
}

impl Product for QuinticExtensionField {
    #[inline]
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|acc, x| acc * x).unwrap_or(Self::ONE)
    }
}

impl Div for QuinticExtensionField {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        self * rhs.inverse()
    }
}

impl DivAssign for QuinticExtensionField {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl Distribution<QuinticExtensionField> for StandardUniform
where
    Self: Distribution<KoalaBear>,
{
    #[inline]
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> QuinticExtensionField {
        QuinticExtensionField::new(array::from_fn(|_| self.sample(rng)))
    }
}

impl TwoAdicField for QuinticExtensionField {
    const TWO_ADICITY: usize = KoalaBear::TWO_ADICITY;

    #[inline]
    fn two_adic_generator(bits: usize) -> Self {
        KoalaBear::two_adic_generator(bits).into()
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
pub(crate) fn kb_quintic_mul<R, R2>(a: &[R; 5], b: &[R2; 5], res: &mut [R; 5])
where
    R: Algebra<KoalaBear> + Algebra<R2>,
    R2: Algebra<KoalaBear>,
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
pub(crate) fn quintic_square<R>(a: &[R; 5], res: &mut [R; 5])
where
    R: Algebra<KoalaBear>,
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
fn quintic_inv(a: &QuinticExtensionField) -> QuinticExtensionField {
    // Writing 'a' for self, we need to compute: `prod_conj = a^{q^4 + q^3 + q^2 + q}`
    let a_exp_q = a.frobenius();
    let a_exp_q_plus_q_sq = (*a * a_exp_q).frobenius();
    let prod_conj = a_exp_q_plus_q_sq * a_exp_q_plus_q_sq.repeated_frobenius(2);

    // norm = a * prod_conj is in the base field, so only compute that
    // coefficient rather than the full product.
    let norm = KoalaBear::dot_product::<5>(
        &a.value,
        &[
            prod_conj.value[0].clone(),
            prod_conj.value[4].clone(),
            prod_conj.value[3].clone(),
            prod_conj.value[2].clone(),
            prod_conj.value[1].clone() - prod_conj.value[4].clone(),
        ],
    );

    debug_assert_eq!(QuinticExtensionField::from(norm), *a * prod_conj);

    prod_conj * norm.inverse()
}

#[cfg(not(any(
    all(target_arch = "aarch64", target_feature = "neon"),
    all(target_arch = "x86_64", target_feature = "avx2",)
)))]
/// If no packings are available, we use the generic binomial extension multiplication functions.
#[inline]
pub(crate) fn kb_quintic_mul_packed(
    a: &[KoalaBear; 5],
    b: &[KoalaBear; 5],
    res: &mut [KoalaBear; 5],
) {
    kb_quintic_mul(a, b, res);
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    not(target_feature = "avx512f")
))]
/// Multiplication in a quintic binomial extension field.
#[inline]
pub(crate) fn kb_quintic_mul_packed(
    a: &[KoalaBear; 5],
    b: &[KoalaBear; 5],
    res: &mut [KoalaBear; 5],
) {
    use p3_monty_31::PackedMontyField31AVX2;
    // TODO: This could likely be optimised further with more effort.
    // in particular it would benefit from a custom AVX2 implementation.

    let zero = KoalaBear::ZERO;
    let b_0_minus_3 = b[0] - b[3];
    let b_1_minus_4 = b[1] - b[4];
    let b_4_minus_2 = b[4] - b[2];
    let b_3_minus_b_1_minus_4 = b[3] - b_1_minus_4;

    let lhs = [
        PackedMontyField31AVX2([a[0], a[0], a[0], a[0], a[0], a[4], a[4], a[4]]),
        PackedMontyField31AVX2([a[1], a[1], a[1], a[1], a[1], zero, zero, zero]),
        PackedMontyField31AVX2([a[2], a[2], a[2], a[2], a[2], zero, zero, zero]),
        PackedMontyField31AVX2([a[3], a[3], a[3], a[3], a[3], zero, zero, zero]),
    ];
    let rhs = [
        PackedMontyField31AVX2([
            b[0],
            b[1],
            b[2],
            b[3],
            b[4],
            b_1_minus_4,
            b[2],
            b_3_minus_b_1_minus_4,
        ]),
        PackedMontyField31AVX2([b[4], b[0], b_1_minus_4, b[2], b[3], zero, zero, zero]),
        PackedMontyField31AVX2([b[3], b[4], b_0_minus_3, b_1_minus_4, b[2], zero, zero, zero]),
        PackedMontyField31AVX2([
            b[2],
            b[3],
            b_4_minus_2,
            b_0_minus_3,
            b_1_minus_4,
            zero,
            zero,
            zero,
        ]),
    ];

    let dot_res =
        unsafe { PackedMontyField31AVX2::from_vector(p3_monty_31::dot_product_4(lhs, rhs)) };

    // We managed to compute 3 of the extra terms in the last 3 places of the dot product.
    // This leaves us with 2 terms remaining we need to compute manually.
    let extra1 = b_4_minus_2 * a[4];
    let extra2 = b_0_minus_3 * a[4];

    let extra_addition = PackedMontyField31AVX2([
        dot_res.0[5],
        dot_res.0[6],
        dot_res.0[7],
        extra1,
        extra2,
        zero,
        zero,
        zero,
    ]);
    let total = dot_res + extra_addition;

    res.copy_from_slice(&total.0[..5]);
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
/// Multiplication in a quintic binomial extension field.
#[inline]
pub(crate) fn kb_quintic_mul_packed<FP>(
    a: &[MontyField31<FP>; 5],
    b: &[MontyField31<FP>; 5],
    res: &mut [MontyField31<FP>; 5],
) where
    FP: FieldParameters + QuinticExtensionData,
{
    // TODO: It's plausible that this could be improved by folding the computation of packed_b into
    // the custom AVX512 implementation. Moreover, AVX512 is really a bit to large so we are wasting a lot
    // of space. A custom implementation which mixes AVX512 and AVX2 code might well be able to
    // improve one that is here.
    let zero = MontyField31::<FP>::ZERO;
    let b_0_minus_3 = b[0] - b[3];
    let b_1_minus_4 = b[1] - b[4];
    let b_4_minus_2 = b[4] - b[2];
    let b_3_minus_b_1_minus_4 = b[3] - b_1_minus_4;

    // Constant term = a0*b0 + w(a1*b4 + a2*b3 + a3*b2 + a4*b1)
    // Linear term = a0*b1 + a1*b0 + w(a2*b4 + a3*b3 + a4*b2)
    // Square term = a0*b2 + a1*b1 + a2*b0 + w(a3*b4 + a4*b3)
    // Cubic term = a0*b3 + a1*b2 + a2*b1 + a3*b0 + w*a4*b4
    // Quartic term = a0*b4 + a1*b3 + a2*b2 + a3*b1 + a4*b0

    // Each packed vector can do 8 multiplications at once. As we have
    // 25 multiplications to do we will need to use at least 3 packed vectors
    // but we might as well use 4 so we can make use of dot_product_2.
    // TODO: This can probably be improved by using a custom function.
    let lhs = [
        PackedMontyField31AVX512([
            a[0], a[2], a[0], a[2], a[0], a[2], a[0], a[2], a[2], a[2], a[4], a[4], a[4], a[4],
            a[4], zero,
        ]),
        PackedMontyField31AVX512([
            a[1], a[3], a[1], a[3], a[1], a[3], a[1], a[3], a[1], a[3], zero, zero, zero, zero,
            zero, zero,
        ]),
    ];
    let rhs = [
        PackedMontyField31AVX512([
            b[0],
            b[3],
            b[1],
            b[4],
            b[2],
            b_0_minus_3,
            b[3],
            b_1_minus_4,
            b[4],
            b[2],
            b_1_minus_4,
            b[2],
            b_3_minus_b_1_minus_4,
            b_4_minus_2,
            b_0_minus_3,
            zero,
        ]),
        PackedMontyField31AVX512([
            b[4],
            b[2],
            b[0],
            b[3],
            b_1_minus_4,
            b_4_minus_2,
            b[2],
            b_0_minus_3,
            b[3],
            b_1_minus_4,
            zero,
            zero,
            zero,
            zero,
            zero,
            zero,
        ]),
    ];

    let dot = unsafe { PackedMontyField31AVX512::from_vector(dot_product_2(lhs, rhs)).0 };

    let sumand1 =
        PackedMontyField31AVX512::from_monty_array([dot[0], dot[2], dot[4], dot[6], dot[8]]);
    let sumand2 =
        PackedMontyField31AVX512::from_monty_array([dot[1], dot[3], dot[5], dot[7], dot[9]]);
    let sumand3 =
        PackedMontyField31AVX512::from_monty_array([dot[10], dot[11], dot[12], dot[13], dot[14]]);
    let sum = sumand1 + sumand2 + sumand3;

    res.copy_from_slice(&sum.0[..5]);
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
/// Multiplication in a quintic binomial extension field.
#[inline]
pub(crate) fn kb_quintic_mul_packed<FP>(
    a: &[KoalaBear; 5],
    b: &[KoalaBear; 5],
    res: &mut [KoalaBear; 5],
) where
    FP: FieldParameters + QuinticExtensionData,
{
    // TODO: This could be optimised further with a custom NEON implementation.
    let b_0_minus_3 = b[0] - b[3];
    let b_1_minus_4 = b[1] - b[4];
    let b_4_minus_2 = b[4] - b[2];
    let b_3_minus_b_1_minus_4 = b[3] - b_1_minus_4;

    // Constant term = a0*b0 + w(a1*b4 + a2*b3 + a3*b2 + a4*b1)
    // Linear term = a0*b1 + a1*b0 + w(a2*b4 + a3*b3 + a4*b2)
    // Square term = a0*b2 + a1*b1 + a2*b0 + w(a3*b4 + a4*b3)
    // Cubic term = a0*b3 + a1*b2 + a2*b1 + a3*b0 + w*a4*b4
    // Quartic term = a0*b4 + a1*b3 + a2*b2 + a3*b1 + a4*b0
    let lhs: [PackedMontyField31Neon<FP>; 5] = [
        a[0].into(),
        a[1].into(),
        a[2].into(),
        a[3].into(),
        a[4].into(),
    ];
    let rhs = [
        PackedMontyField31Neon([b[0], b[1], b[2], b[4]]),
        PackedMontyField31Neon([b[4], b[0], b_1_minus_4, b[3]]),
        PackedMontyField31Neon([b[3], b[4], b_0_minus_3, b[2]]),
        PackedMontyField31Neon([b[2], b[3], b_4_minus_2, b_1_minus_4]),
        PackedMontyField31Neon([b_1_minus_4, b[2], b_3_minus_b_1_minus_4, b_0_minus_3]),
    ];

    let dot = PackedMontyField31Neon::dot_product(&lhs, &rhs).0;

    res[..4].copy_from_slice(&dot);
    res[4] = MontyField31::dot_product::<5>(
        &[a[0], a[1], a[2], a[3], a[4]],
        &[b[4], b[3], b[2], b_1_minus_4, b_0_minus_3],
    );
}

// fn compute_frobenius_matrix() {
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
