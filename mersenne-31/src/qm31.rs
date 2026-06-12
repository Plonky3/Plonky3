//! The degree-4 extension of Mersenne31, built as the binomial extension by
//! `X² - (2 + i)` over the complex extension `Mersenne31[i]`, flattened to a
//! degree-4 vector space over `Mersenne31`.
//!
//! The tower type `BinomialExtensionField<Complex<Mersenne31>, 2>` only knows
//! it is an extension of `Complex<Mersenne31>`. This module supplies the
//! `Algebra<Mersenne31>`, `BasedVectorSpace<Mersenne31>` and
//! `ExtensionField<Mersenne31>` impls (plus the packed counterpart
//! [`PackedQM31`]) that let it serve as the challenge field of a STARK over
//! `Mersenne31`, with 4 · 31 = 124 bits of extension size.
//!
//! The flattened basis is `[1, i, u, iu]` (`u² = 2 + i`), i.e. the in-memory
//! order of the nested `[[Mersenne31; 2]; 2]` representation.

use alloc::vec::Vec;
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use core::slice;

use p3_field::extension::{BinomialExtensionField, Complex};
use p3_field::{
    Algebra, BasedVectorSpace, ExtensionField, Field, PackedFieldExtension, PackedValue, Powers,
    PrimeCharacteristicRing,
};
use p3_util::{as_base_slice, flatten_to_base, reconstitute_from_base};

use crate::Mersenne31;

/// The degree-4 extension of Mersenne31: `Mersenne31[i][u]` with `i² = -1`
/// and `u² = 2 + i`.
pub type QM31 = BinomialExtensionField<Complex<Mersenne31>, 2>;

type CM31 = Complex<Mersenne31>;
type PackedM31 = <Mersenne31 as Field>::Packing;
type PackedCM31 = <CM31 as ExtensionField<Mersenne31>>::ExtensionPacking;

/// The two `CM31` coefficients of a `QM31` element.
#[inline(always)]
fn coeffs(x: &QM31) -> [CM31; 2] {
    let s = BasedVectorSpace::<CM31>::as_basis_coefficients_slice(x);
    [s[0], s[1]]
}

/// Multiply a packed complex coefficient by `W = 2 + i` using only additions:
/// `(a + bi)(2 + i) = (2a - b) + (a + 2b)i`.
#[inline(always)]
fn packed_mul_by_w(c: PackedCM31) -> PackedCM31 {
    let s = BasedVectorSpace::<PackedM31>::as_basis_coefficients_slice(&c);
    let (re, im) = (s[0], s[1]);
    PackedCM31::new([re + re - im, re + im + im])
}

// ---------------------------------------------------------------------------
// Scalar flattening: QM31 as an algebra / vector space / extension over M31
// ---------------------------------------------------------------------------

impl From<Mersenne31> for QM31 {
    #[inline]
    fn from(x: Mersenne31) -> Self {
        Self::new([CM31::from(x), CM31::ZERO])
    }
}

impl Add<Mersenne31> for QM31 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Mersenne31) -> Self {
        let [c0, c1] = coeffs(&self);
        Self::new([c0 + rhs, c1])
    }
}

impl AddAssign<Mersenne31> for QM31 {
    #[inline]
    fn add_assign(&mut self, rhs: Mersenne31) {
        *self = *self + rhs;
    }
}

impl Sub<Mersenne31> for QM31 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Mersenne31) -> Self {
        let [c0, c1] = coeffs(&self);
        Self::new([c0 - rhs, c1])
    }
}

impl SubAssign<Mersenne31> for QM31 {
    #[inline]
    fn sub_assign(&mut self, rhs: Mersenne31) {
        *self = *self - rhs;
    }
}

impl Mul<Mersenne31> for QM31 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Mersenne31) -> Self {
        let [c0, c1] = coeffs(&self);
        Self::new([c0 * rhs, c1 * rhs])
    }
}

impl MulAssign<Mersenne31> for QM31 {
    #[inline]
    fn mul_assign(&mut self, rhs: Mersenne31) {
        *self = *self * rhs;
    }
}

impl Algebra<Mersenne31> for QM31 {}

impl BasedVectorSpace<Mersenne31> for QM31 {
    const DIMENSION: usize = 4;

    #[inline]
    fn as_basis_coefficients_slice(&self) -> &[Mersenne31] {
        // SAFETY: `QM31` is `repr(transparent)` over `[CM31; 2]` and `CM31`
        // over `[Mersenne31; 2]`, so `QM31` is layout-identical to
        // `[Mersenne31; 4]`.
        unsafe { as_base_slice(slice::from_ref(self)) }
    }

    #[inline]
    fn from_basis_coefficients_fn<Fn: FnMut(usize) -> Mersenne31>(mut f: Fn) -> Self {
        Self::new(core::array::from_fn(|i| {
            CM31::from_basis_coefficients_fn(|j| f(2 * i + j))
        }))
    }

    #[inline]
    fn from_basis_coefficients_iter<I: ExactSizeIterator<Item = Mersenne31>>(
        mut iter: I,
    ) -> Option<Self> {
        (iter.len() == 4).then(|| Self::from_basis_coefficients_fn(|_| iter.next().unwrap()))
    }

    #[inline]
    fn flatten_to_base(vec: Vec<Self>) -> Vec<Mersenne31> {
        // SAFETY: `QM31` is layout-identical to `[Mersenne31; 4]` (see
        // `as_basis_coefficients_slice`) and has the same alignment as `Mersenne31`.
        unsafe { flatten_to_base(vec) }
    }

    #[inline]
    fn reconstitute_from_base(vec: Vec<Mersenne31>) -> Vec<Self> {
        // SAFETY: `QM31` is layout-identical to `[Mersenne31; 4]` (see
        // `as_basis_coefficients_slice`) and has the same alignment as `Mersenne31`.
        unsafe { reconstitute_from_base(vec) }
    }
}

impl ExtensionField<Mersenne31> for QM31 {
    type ExtensionPacking = PackedQM31;

    #[inline]
    fn is_in_basefield(&self) -> bool {
        BasedVectorSpace::<Mersenne31>::as_basis_coefficients_slice(self)[1..]
            .iter()
            .all(Mersenne31::is_zero)
    }

    #[inline]
    fn as_base(&self) -> Option<Mersenne31> {
        ExtensionField::<Mersenne31>::is_in_basefield(self)
            .then(|| BasedVectorSpace::<Mersenne31>::as_basis_coefficients_slice(self)[0])
    }
}

// ---------------------------------------------------------------------------
// PackedQM31: SIMD-lane-parallel QM31, two packed CM31 coefficients
// ---------------------------------------------------------------------------

/// Packed representation of [`QM31`]: two packed `Complex<Mersenne31>`
/// coefficients, each holding `PackedM31::WIDTH` lanes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(transparent)]
pub struct PackedQM31(pub [PackedCM31; 2]);

impl PrimeCharacteristicRing for PackedQM31 {
    type PrimeSubfield = Mersenne31;

    const ZERO: Self = Self([PackedCM31::ZERO; 2]);
    const ONE: Self = Self([PackedCM31::ONE, PackedCM31::ZERO]);
    const TWO: Self = Self([PackedCM31::TWO, PackedCM31::ZERO]);
    const NEG_ONE: Self = Self([PackedCM31::NEG_ONE, PackedCM31::ZERO]);

    #[inline]
    fn from_prime_subfield(val: Self::PrimeSubfield) -> Self {
        Self([PackedCM31::from_prime_subfield(val), PackedCM31::ZERO])
    }

    #[inline]
    fn halve(&self) -> Self {
        Self(self.0.map(|c| c.halve()))
    }

    #[inline]
    fn mul_2exp_u64(&self, exp: u64) -> Self {
        Self(self.0.map(|c| c.mul_2exp_u64(exp)))
    }

    #[inline]
    fn div_2exp_u64(&self, exp: u64) -> Self {
        Self(self.0.map(|c| c.div_2exp_u64(exp)))
    }

    #[inline]
    fn zero_vec(len: usize) -> Vec<Self> {
        // SAFETY: `Self` is `repr(transparent)` over `[PackedCM31; 2]`.
        unsafe { reconstitute_from_base(PackedCM31::zero_vec(len * 2)) }
    }
}

impl Neg for PackedQM31 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self(self.0.map(Neg::neg))
    }
}

impl Add for PackedQM31 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self([self.0[0] + rhs.0[0], self.0[1] + rhs.0[1]])
    }
}

impl AddAssign for PackedQM31 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Sub for PackedQM31 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self([self.0[0] - rhs.0[0], self.0[1] - rhs.0[1]])
    }
}

impl SubAssign for PackedQM31 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Mul for PackedQM31 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        // Karatsuba over CM31 with the cheap W = 2 + i correction:
        //   (a0 + a1 u)(b0 + b1 u) = a0 b0 + W a1 b1 + (a0 b1 + a1 b0) u
        // with a0 b1 + a1 b0 = (a0 + a1)(b0 + b1) - a0 b0 - a1 b1,
        // for 3 full CM31 multiplications instead of 4.
        let [a0, a1] = self.0;
        let [b0, b1] = rhs.0;
        let m0 = a0 * b0;
        let m1 = a1 * b1;
        let m2 = (a0 + a1) * (b0 + b1);
        Self([m0 + packed_mul_by_w(m1), m2 - m0 - m1])
    }
}

impl MulAssign for PackedQM31 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl core::ops::Div for PackedQM31 {
    type Output = Self;
    #[allow(clippy::suspicious_arithmetic_impl)]
    #[inline]
    fn div(self, rhs: Self) -> Self {
        self * p3_field::invert_packed_extension::<Mersenne31, QM31>(rhs)
    }
}

impl core::ops::DivAssign for PackedQM31 {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl Sum for PackedQM31 {
    #[inline]
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |acc, x| acc + x)
    }
}

impl Product for PackedQM31 {
    #[inline]
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ONE, |acc, x| acc * x)
    }
}

// --- Algebra<QM31> ---

impl From<QM31> for PackedQM31 {
    #[inline]
    fn from(x: QM31) -> Self {
        let [c0, c1] = coeffs(&x);
        Self([c0.into(), c1.into()])
    }
}

impl Add<QM31> for PackedQM31 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: QM31) -> Self {
        let [b0, b1] = coeffs(&rhs);
        Self([self.0[0] + b0, self.0[1] + b1])
    }
}

impl AddAssign<QM31> for PackedQM31 {
    #[inline]
    fn add_assign(&mut self, rhs: QM31) {
        *self = *self + rhs;
    }
}

impl Sub<QM31> for PackedQM31 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: QM31) -> Self {
        let [b0, b1] = coeffs(&rhs);
        Self([self.0[0] - b0, self.0[1] - b1])
    }
}

impl SubAssign<QM31> for PackedQM31 {
    #[inline]
    fn sub_assign(&mut self, rhs: QM31) {
        *self = *self - rhs;
    }
}

impl Mul<QM31> for PackedQM31 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: QM31) -> Self {
        let [a0, a1] = self.0;
        let [b0, b1] = coeffs(&rhs);
        let m0 = a0 * b0;
        let m1 = a1 * b1;
        let m2 = (a0 + a1) * (b0 + b1);
        Self([m0 + packed_mul_by_w(m1), m2 - m0 - m1])
    }
}

impl MulAssign<QM31> for PackedQM31 {
    #[inline]
    fn mul_assign(&mut self, rhs: QM31) {
        *self = *self * rhs;
    }
}

impl Algebra<QM31> for PackedQM31 {}

// --- Algebra<PackedM31> ---

impl From<PackedM31> for PackedQM31 {
    #[inline]
    fn from(x: PackedM31) -> Self {
        Self([x.into(), PackedCM31::ZERO])
    }
}

impl Add<PackedM31> for PackedQM31 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: PackedM31) -> Self {
        Self([self.0[0] + rhs, self.0[1]])
    }
}

impl AddAssign<PackedM31> for PackedQM31 {
    #[inline]
    fn add_assign(&mut self, rhs: PackedM31) {
        *self = *self + rhs;
    }
}

impl Sub<PackedM31> for PackedQM31 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: PackedM31) -> Self {
        Self([self.0[0] - rhs, self.0[1]])
    }
}

impl SubAssign<PackedM31> for PackedQM31 {
    #[inline]
    fn sub_assign(&mut self, rhs: PackedM31) {
        *self = *self - rhs;
    }
}

impl Mul<PackedM31> for PackedQM31 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: PackedM31) -> Self {
        Self([self.0[0] * rhs, self.0[1] * rhs])
    }
}

impl MulAssign<PackedM31> for PackedQM31 {
    #[inline]
    fn mul_assign(&mut self, rhs: PackedM31) {
        *self = *self * rhs;
    }
}

impl Algebra<PackedM31> for PackedQM31 {}

impl BasedVectorSpace<PackedM31> for PackedQM31 {
    const DIMENSION: usize = 4;

    #[inline]
    fn as_basis_coefficients_slice(&self) -> &[PackedM31] {
        // SAFETY: `PackedQM31` is `repr(transparent)` over `[PackedCM31; 2]`
        // and `PackedCM31` over `[PackedM31; 2]`, so `PackedQM31` is
        // layout-identical to `[PackedM31; 4]`.
        unsafe { as_base_slice(slice::from_ref(self)) }
    }

    #[inline]
    fn from_basis_coefficients_fn<Fn: FnMut(usize) -> PackedM31>(mut f: Fn) -> Self {
        Self(core::array::from_fn(|i| {
            PackedCM31::from_basis_coefficients_fn(|j| f(2 * i + j))
        }))
    }

    #[inline]
    fn from_basis_coefficients_iter<I: ExactSizeIterator<Item = PackedM31>>(
        mut iter: I,
    ) -> Option<Self> {
        (iter.len() == 4).then(|| Self::from_basis_coefficients_fn(|_| iter.next().unwrap()))
    }
}

impl rand::distr::Distribution<PackedQM31> for rand::distr::StandardUniform {
    #[inline]
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> PackedQM31 {
        PackedQM31(core::array::from_fn(|_| {
            <PackedCM31 as BasedVectorSpace<PackedM31>>::from_basis_coefficients_fn(|_| {
                self.sample(rng)
            })
        }))
    }
}

impl PackedFieldExtension<Mersenne31, QM31> for PackedQM31 {
    #[inline]
    fn packed_ext_powers(base: QM31) -> Powers<Self> {
        let width = PackedM31::WIDTH;
        let powers = base.powers().collect_n(width + 1);
        // Transpose the first WIDTH powers into the lanes.
        let current = Self::from_ext_slice(&powers[..width]);
        // Broadcast base^WIDTH as the per-step multiplier.
        Powers {
            base: powers[width].into(),
            current,
        }
    }
}

#[cfg(test)]
mod tests {
    use num_bigint::BigUint;
    use p3_field::PrimeCharacteristicRing;
    use p3_field_testing::{test_extension_field, test_field, test_packed_extension_field};

    use super::*;

    type F = Mersenne31;
    type EF = QM31;

    const ZEROS: [EF; 1] = [EF::ZERO];
    const ONES: [EF; 1] = [EF::ONE];

    // The prime factorization of P^4 - 1 (same multiplicative group as the
    // quadratic-over-complex view tested in `extension.rs`).
    fn multiplicative_group_prime_factorization() -> [(BigUint, u32); 11] {
        [
            (BigUint::from(2u8), 33),
            (BigUint::from(3u8), 2),
            (BigUint::from(5u8), 1),
            (BigUint::from(7u8), 1),
            (BigUint::from(11u8), 1),
            (BigUint::from(31u8), 1),
            (BigUint::from(151u8), 1),
            (BigUint::from(331u16), 1),
            (BigUint::from(733u16), 1),
            (BigUint::from(1709u16), 1),
            (BigUint::from(368140581013u64), 1),
        ]
    }

    test_field!(
        super::EF,
        &super::ZEROS,
        &super::ONES,
        &super::multiplicative_group_prime_factorization()
    );

    test_extension_field!(super::F, super::EF);

    type Pef = PackedQM31;
    const PACKED_ZEROS: [Pef; 1] = [Pef::ZERO];
    const PACKED_ONES: [Pef; 1] = [Pef::ONE];
    test_packed_extension_field!(
        super::F,
        super::EF,
        super::Pef,
        &super::PACKED_ZEROS,
        &super::PACKED_ONES
    );

    /// The flattened M31 basis order must match the nested CM31 layout.
    #[test]
    fn flattened_basis_order_matches_nested_layout() {
        use p3_field::BasedVectorSpace;

        let x = QM31::new([
            Complex::new_complex(F::new(1), F::new(2)),
            Complex::new_complex(F::new(3), F::new(4)),
        ]);
        let flat = BasedVectorSpace::<F>::as_basis_coefficients_slice(&x);
        assert_eq!(flat, &[F::new(1), F::new(2), F::new(3), F::new(4)]);

        let rebuilt = <QM31 as BasedVectorSpace<F>>::from_basis_coefficients_slice(flat).unwrap();
        assert_eq!(rebuilt, x);
    }

    /// Packed multiplication must agree with scalar multiplication lane-wise.
    #[test]
    fn packed_mul_matches_scalar() {
        use p3_field::PackedFieldExtension;
        use rand::rngs::SmallRng;
        use rand::{RngExt, SeedableRng};

        let mut rng = SmallRng::seed_from_u64(1);
        let width = <PackedM31 as p3_field::PackedValue>::WIDTH;
        let xs: alloc::vec::Vec<QM31> = (0..width).map(|_| rng.random()).collect();
        let ys: alloc::vec::Vec<QM31> = (0..width).map(|_| rng.random()).collect();

        let px = <PackedQM31 as PackedFieldExtension<F, EF>>::from_ext_slice(&xs);
        let py = <PackedQM31 as PackedFieldExtension<F, EF>>::from_ext_slice(&ys);
        let prod = px * py;

        for lane in 0..width {
            assert_eq!(
                <PackedQM31 as PackedFieldExtension<F, EF>>::extract(&prod, lane),
                xs[lane] * ys[lane]
            );
        }
    }

    /// The zero-copy `flatten_to_base`/`reconstitute_from_base` overrides must match
    /// the basis-coefficient view element by element and round-trip exactly.
    #[test]
    fn flatten_reconstitute_roundtrip() {
        use alloc::vec::Vec;

        use rand::rngs::SmallRng;
        use rand::{RngExt, SeedableRng};

        let mut rng = SmallRng::seed_from_u64(7);
        let xs: Vec<QM31> = (0..23).map(|_| rng.random()).collect();

        let flat = <QM31 as BasedVectorSpace<F>>::flatten_to_base(xs.clone());
        let expected: Vec<F> = xs
            .iter()
            .flat_map(|x| x.as_basis_coefficients_slice().to_vec())
            .collect();
        assert_eq!(flat, expected);

        let back = <QM31 as BasedVectorSpace<F>>::reconstitute_from_base(flat);
        assert_eq!(back, xs);
    }
}
