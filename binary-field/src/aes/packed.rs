//! Blocks of AES-field bytes, sixteen, thirty-two or sixty-four at a time.
//!
//! The block width is a parameter rather than three separate types.
//!
//! Every operation below is written once and specialized per width by the compiler.
//!
//! How many bytes one instruction covers is a separate question, settled by the target.
//!
//! A block wider than the register is swept in several passes.

use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use core::{array, ptr};

use p3_field::{
    Algebra, Field, PackedField, PackedFieldPow2, PackedValue, PrimeCharacteristicRing,
};
use rand::distr::{Distribution, StandardUniform};
use rand::{Rng, RngExt};

use super::engine::{invert_slice, mul_slice};
use super::{ByteMatrix, Rijndael8b};
use crate::Gf2;

/// The block width the scalar field advertises as its packing.
///
/// Sixty-four bytes is one half-kilobit register.
///
/// The widest instruction then runs once per block rather than four times.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "gfni",
    target_feature = "avx512f",
    target_feature = "avx512bw"
))]
pub(super) const PACKING_WIDTH: usize = 64;

/// The block width the scalar field advertises as its packing.
///
/// Sixteen bytes is the widest register the byte-wise instructions are guaranteed in.
///
/// It is also what a compiler vectorizes the portable fallback into.
#[cfg(not(all(
    target_arch = "x86_64",
    target_feature = "gfni",
    target_feature = "avx512f",
    target_feature = "avx512bw"
)))]
pub(super) const PACKING_WIDTH: usize = 16;

/// A block of elements of the AES field, held as plain consecutive bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
// Needed to make the byte views below sound.
#[repr(transparent)]
#[must_use]
pub struct PackedRijndael8b<const N: usize>([Rijndael8b; N]);

impl<const N: usize> Default for PackedRijndael8b<N> {
    #[inline]
    fn default() -> Self {
        Self::ZERO
    }
}

impl<const N: usize> PackedRijndael8b<N> {
    /// The same element in every position.
    #[inline]
    const fn splat(value: Rijndael8b) -> Self {
        // A width that is not a power of two would break the interleave below.
        const {
            assert!(
                N.is_power_of_two(),
                "the block width must be a power of two"
            );
        }
        Self([value; N])
    }

    /// The block seen as plain bytes.
    #[inline]
    const fn bytes(&self) -> &[u8; N] {
        // SAFETY: the scalar is `repr(transparent)` over a byte and this type over the array.
        //
        // The block is therefore exactly `N` consecutive bytes at the same alignment.
        unsafe { &*ptr::from_ref(self).cast::<[u8; N]>() }
    }

    /// The block seen as plain bytes, for writing.
    #[inline]
    const fn bytes_mut(&mut self) -> &mut [u8; N] {
        // SAFETY: as above, and every byte is a valid element, so no write can be out of range.
        unsafe { &mut *ptr::from_mut(self).cast::<[u8; N]>() }
    }

    /// Replaces every element with its image under one `F_2`-linear map.
    pub fn apply(&mut self, map: ByteMatrix) {
        map.apply_slice(self.bytes_mut());
    }

    /// Raises every element to the power `2^k`.
    ///
    /// One tabulated map covers any exponent.
    ///
    /// Repeated squaring would cost one product per step instead.
    pub fn frobenius(&mut self, power: usize) {
        self.apply(Rijndael8b::frobenius_map(power));
    }

    /// Replaces every element with its inverse, leaving zero alone.
    ///
    /// The byte-wise hardware inverse is one instruction per register.
    ///
    /// The fallback is an addition chain of eleven products per element.
    pub fn invert_or_zero(&mut self) {
        invert_slice(self.bytes_mut());
    }
}

impl<const N: usize> From<Rijndael8b> for PackedRijndael8b<N> {
    #[inline]
    fn from(value: Rijndael8b) -> Self {
        Self::splat(value)
    }
}

impl<const N: usize> From<Gf2> for PackedRijndael8b<N> {
    /// `GF(2)` is the prime subfield, embedded as `{ZERO, ONE}` in every position.
    #[inline]
    fn from(value: Gf2) -> Self {
        Self::splat(Rijndael8b::from(value))
    }
}

impl<const N: usize> Add for PackedRijndael8b<N> {
    type Output = Self;

    #[inline]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn add(self, rhs: Self) -> Self {
        // Addition in characteristic 2 is `XOR`, position by position.
        Self(array::from_fn(|i| self.0[i] + rhs.0[i]))
    }
}

impl<const N: usize> Sub for PackedRijndael8b<N> {
    type Output = Self;

    #[inline]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn sub(self, rhs: Self) -> Self {
        // Subtraction coincides with addition in characteristic 2.
        self + rhs
    }
}

impl<const N: usize> Neg for PackedRijndael8b<N> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        // `-x = x` in characteristic 2.
        self
    }
}

impl<const N: usize> Mul for PackedRijndael8b<N> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let mut out = self;
        mul_slice(out.bytes_mut(), rhs.bytes());
        out
    }
}

impl<const N: usize> Div for PackedRijndael8b<N> {
    type Output = Self;

    /// # Panics
    /// Panics if any position of the divisor is zero.
    #[inline]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: Self) -> Self {
        let mut inverse = rhs;
        assert!(
            !inverse.bytes().contains(&0),
            "tried to invert zero in a block"
        );
        inverse.invert_or_zero();
        self * inverse
    }
}

impl<const N: usize> DivAssign for PackedRijndael8b<N> {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl<const N: usize> Add<Rijndael8b> for PackedRijndael8b<N> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Rijndael8b) -> Self {
        self + Self::splat(rhs)
    }
}

impl<const N: usize> Add<PackedRijndael8b<N>> for Rijndael8b {
    type Output = PackedRijndael8b<N>;

    #[inline]
    fn add(self, rhs: PackedRijndael8b<N>) -> Self::Output {
        rhs + self
    }
}

impl<const N: usize> Sub<Rijndael8b> for PackedRijndael8b<N> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Rijndael8b) -> Self {
        self - Self::splat(rhs)
    }
}

impl<const N: usize> Sub<PackedRijndael8b<N>> for Rijndael8b {
    type Output = PackedRijndael8b<N>;

    #[inline]
    fn sub(self, rhs: PackedRijndael8b<N>) -> Self::Output {
        PackedRijndael8b::splat(self) - rhs
    }
}

impl<const N: usize> Mul<Rijndael8b> for PackedRijndael8b<N> {
    type Output = Self;

    /// Scaling is `F_2`-linear, so a fixed factor rides the map engine rather than a broadcast.
    #[inline]
    fn mul(mut self, rhs: Rijndael8b) -> Self {
        self.apply(rhs.scaling_matrix());
        self
    }
}

impl<const N: usize> Mul<PackedRijndael8b<N>> for Rijndael8b {
    type Output = PackedRijndael8b<N>;

    #[inline]
    fn mul(self, rhs: PackedRijndael8b<N>) -> Self::Output {
        rhs * self
    }
}

impl<const N: usize> Div<Rijndael8b> for PackedRijndael8b<N> {
    type Output = Self;

    #[inline]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: Rijndael8b) -> Self {
        self * rhs.inverse()
    }
}

impl<const N: usize> DivAssign<Rijndael8b> for PackedRijndael8b<N> {
    #[inline]
    #[allow(clippy::suspicious_op_assign_impl)]
    fn div_assign(&mut self, rhs: Rijndael8b) {
        *self *= rhs.inverse();
    }
}

impl<const N: usize, T: Into<Self>> AddAssign<T> for PackedRijndael8b<N> {
    #[inline]
    fn add_assign(&mut self, rhs: T) {
        *self = *self + rhs.into();
    }
}

impl<const N: usize, T: Into<Self>> SubAssign<T> for PackedRijndael8b<N> {
    #[inline]
    fn sub_assign(&mut self, rhs: T) {
        *self = *self - rhs.into();
    }
}

impl<const N: usize, T: Into<Self>> MulAssign<T> for PackedRijndael8b<N> {
    #[inline]
    fn mul_assign(&mut self, rhs: T) {
        *self = *self * rhs.into();
    }
}

impl<const N: usize> Sum for PackedRijndael8b<N> {
    #[inline]
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|x, y| x + y).unwrap_or(Self::ZERO)
    }
}

impl<const N: usize> Product for PackedRijndael8b<N> {
    #[inline]
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|x, y| x * y).unwrap_or(Self::ONE)
    }
}

impl<const N: usize> Sum<Rijndael8b> for PackedRijndael8b<N> {
    #[inline]
    fn sum<I: Iterator<Item = Rijndael8b>>(iter: I) -> Self {
        iter.sum::<Rijndael8b>().into()
    }
}

impl<const N: usize> Product<Rijndael8b> for PackedRijndael8b<N> {
    #[inline]
    fn product<I: Iterator<Item = Rijndael8b>>(iter: I) -> Self {
        iter.product::<Rijndael8b>().into()
    }
}

impl<const N: usize> Distribution<PackedRijndael8b<N>> for StandardUniform {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> PackedRijndael8b<N> {
        PackedRijndael8b(array::from_fn(|_| rng.random()))
    }
}

impl<const N: usize> PrimeCharacteristicRing for PackedRijndael8b<N> {
    type PrimeSubfield = Gf2;

    const ZERO: Self = Self::splat(Rijndael8b::ZERO);
    const ONE: Self = Self::splat(Rijndael8b::ONE);
    // The characteristic is 2, so `TWO = ONE + ONE = ZERO`.
    const TWO: Self = Self::splat(Rijndael8b::ZERO);
    // The characteristic is 2, so `NEG_ONE = ONE`.
    const NEG_ONE: Self = Self::splat(Rijndael8b::ONE);

    #[inline]
    fn from_prime_subfield(f: Self::PrimeSubfield) -> Self {
        Self::splat(Rijndael8b::from_prime_subfield(f))
    }

    #[inline]
    fn double(&self) -> Self {
        // `a + a = 0` in characteristic 2.
        Self::ZERO
    }

    /// # Panics
    /// Always panics: `2` is not invertible in characteristic 2.
    #[inline]
    fn halve(&self) -> Self {
        panic!("halve is undefined in characteristic 2")
    }

    #[inline]
    fn xor(&self, y: &Self) -> Self {
        *self + *y
    }

    #[inline]
    fn mul_2exp_u64(&self, exp: u64) -> Self {
        if exp == 0 { *self } else { Self::ZERO }
    }

    /// # Panics
    /// Always panics: `2` is not invertible in characteristic 2.
    #[inline]
    fn div_2exp_u64(&self, _exp: u64) -> Self {
        panic!("div_2exp_u64 is undefined in characteristic 2")
    }
}

impl<const N: usize> Algebra<Rijndael8b> for PackedRijndael8b<N> {}

impl<const N: usize> Algebra<Gf2> for PackedRijndael8b<N> {}

impl<const N: usize> Add<Gf2> for PackedRijndael8b<N> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Gf2) -> Self {
        self + Self::from(rhs)
    }
}

impl<const N: usize> Sub<Gf2> for PackedRijndael8b<N> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Gf2) -> Self {
        self - Self::from(rhs)
    }
}

impl<const N: usize> Mul<Gf2> for PackedRijndael8b<N> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Gf2) -> Self {
        // The prime subfield has two elements, so this keeps the block or clears it.
        if rhs.is_one() { self } else { Self::ZERO }
    }
}

// SAFETY: the type is `repr(transparent)` over the array, so the reference casts below hold.
unsafe impl<const N: usize> PackedValue for PackedRijndael8b<N> {
    type Value = Rijndael8b;

    const WIDTH: usize = N;

    #[inline]
    fn from_slice(slice: &[Self::Value]) -> &Self {
        assert_eq!(slice.len(), Self::WIDTH);
        // SAFETY: the length check above makes the source exactly one block.
        unsafe { &*slice.as_ptr().cast() }
    }

    #[inline]
    fn from_slice_mut(slice: &mut [Self::Value]) -> &mut Self {
        assert_eq!(slice.len(), Self::WIDTH);
        // SAFETY: the length check above makes the source exactly one block.
        unsafe { &mut *slice.as_mut_ptr().cast() }
    }

    #[inline]
    fn as_slice(&self) -> &[Self::Value] {
        &self.0
    }

    #[inline]
    fn as_slice_mut(&mut self) -> &mut [Self::Value] {
        &mut self.0
    }

    #[inline]
    fn from_fn<F: FnMut(usize) -> Self::Value>(f: F) -> Self {
        Self(array::from_fn(f))
    }
}

// SAFETY: the transparent array satisfies the packed layout contract.
//
// Every operation acts on each byte independently.
unsafe impl<const N: usize> PackedField for PackedRijndael8b<N> {
    type Scalar = Rijndael8b;
}

// SAFETY: the widths this type is used at are powers of two.
unsafe impl<const N: usize> PackedFieldPow2 for PackedRijndael8b<N> {
    /// # Panics
    /// Panics if the block length does not divide the width, or is not a power of two.
    #[inline]
    fn interleave(&self, other: Self, block_len: usize) -> (Self, Self) {
        assert!(
            block_len.is_power_of_two() && block_len <= N && N.is_multiple_of(block_len),
            "unsupported block_len"
        );

        // One block each leaves nothing to pair up, so the two vectors pass through.
        if block_len == N {
            return (*self, other);
        }

        // Stack the two vectors, cut them into two-by-two matrices of blocks, transpose those.
        //
        //     left  blocks: a_0, b_0, a_2, b_2, ...
        //     right blocks: a_1, b_1, a_3, b_3, ...
        let block = |from: &Self, index: usize, offset: usize| from.0[index * block_len + offset];
        let left = array::from_fn(|i| {
            let (m, offset) = (i / block_len, i % block_len);
            if m % 2 == 0 {
                block(self, m, offset)
            } else {
                block(&other, m - 1, offset)
            }
        });
        let right = array::from_fn(|i| {
            let (m, offset) = (i / block_len, i % block_len);
            if m % 2 == 0 {
                block(self, m + 1, offset)
            } else {
                block(&other, m, offset)
            }
        });
        (Self(left), Self(right))
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use p3_field::{PackedFieldPow2, PackedValue, PrimeCharacteristicRing};
    use proptest::prelude::*;

    use super::PackedRijndael8b;
    use crate::Rijndael8b;
    use crate::aes::{invert_byte, mul_bytes};

    /// The block widths the crate exports, so every sweep length is covered.
    ///
    /// Sixteen is one narrow register, sixty-four is one wide one, and thirty-two is neither.
    type Narrow = PackedRijndael8b<16>;
    type Middle = PackedRijndael8b<32>;
    type Wide = PackedRijndael8b<64>;

    /// The bytes a random search is unlikely to reach.
    ///
    /// ```text
    ///     0     the absorbing element
    ///     1     the identity
    ///     0x1b  the modulus tail, so one more fold happens
    ///     0x80  the highest basis vector, so doubling it overflows
    /// ```
    const SPECIAL: [u8; 4] = [0, 1, 0x1b, 0x80];

    /// One extreme byte per position, cycling through the corners.
    fn specials<const N: usize>() -> PackedRijndael8b<N> {
        PackedValue::from_fn(|i| Rijndael8b::from_byte(SPECIAL[i % SPECIAL.len()]))
    }

    /// The block operations against the scalar routines, one position at a time.
    fn block_agrees<const N: usize>(a: [u8; N], b: [u8; N]) -> Result<(), TestCaseError> {
        let pack =
            |bytes: [u8; N]| PackedRijndael8b::<N>::from_fn(|i| Rijndael8b::from_byte(bytes[i]));
        let (x, y) = (pack(a), pack(b));

        let expected = pack(core::array::from_fn(|i| mul_bytes(a[i], b[i])));
        prop_assert_eq!(x * y, expected);

        let expected = pack(core::array::from_fn(|i| a[i] ^ b[i]));
        prop_assert_eq!(x + y, expected);

        // Scaling by the first position's byte, which rides the map engine rather than a product.
        let scalar = Rijndael8b::from_byte(b[0]);
        let expected = pack(core::array::from_fn(|i| mul_bytes(a[i], b[0])));
        prop_assert_eq!(x * scalar, expected);

        let mut inverted = x;
        inverted.invert_or_zero();
        prop_assert_eq!(inverted, pack(core::array::from_fn(|i| invert_byte(a[i]))));

        Ok(())
    }

    #[test]
    fn every_width_agrees_with_the_scalar_routines_at_the_corners() {
        // Fixture state: each width, with every corner pair placed at position 0.
        //
        // The other positions rotate through the corners.
        //
        // A value that leaks across a register boundary then lands on a different extreme.
        for (i, &x) in SPECIAL.iter().enumerate() {
            for (j, &y) in SPECIAL.iter().enumerate() {
                let a = |lane: usize| {
                    if lane == 0 {
                        x
                    } else {
                        SPECIAL[(i + lane) % 4]
                    }
                };
                let b = |lane: usize| {
                    if lane == 0 {
                        y
                    } else {
                        SPECIAL[(j + lane) % 4]
                    }
                };
                block_agrees::<16>(core::array::from_fn(a), core::array::from_fn(b)).unwrap();
                block_agrees::<32>(core::array::from_fn(a), core::array::from_fn(b)).unwrap();
                block_agrees::<64>(core::array::from_fn(a), core::array::from_fn(b)).unwrap();
            }
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(256))]

        #[test]
        fn every_width_agrees_with_the_scalar_routines(
            a in prop::collection::vec(any::<u8>(), 64),
            b in prop::collection::vec(any::<u8>(), 64),
        ) {
            let take = |from: &Vec<u8>, n: usize| -> Vec<u8> { from[..n].to_vec() };
            let fixed = |v: Vec<u8>| -> [u8; 16] { core::array::from_fn(|i| v[i]) };
            block_agrees::<16>(fixed(take(&a, 16)), fixed(take(&b, 16)))?;

            let fixed = |v: Vec<u8>| -> [u8; 32] { core::array::from_fn(|i| v[i]) };
            block_agrees::<32>(fixed(take(&a, 32)), fixed(take(&b, 32)))?;

            let fixed = |v: Vec<u8>| -> [u8; 64] { core::array::from_fn(|i| v[i]) };
            block_agrees::<64>(fixed(a), fixed(b))?;
        }
    }

    #[test]
    fn the_interleave_convention_is_the_two_by_two_transpose() {
        // Invariant: stack the two vectors, cut into two-by-two matrices of blocks, transpose.
        //
        // Fixture state: sixteen positions carrying 0..15 against 16..31.
        //
        //     block length 1:  left takes a_0 b_0 a_2 b_2 ...
        //     naive reading:   left would take a_0 b_0 a_1 b_1 ...
        //
        // The two disagree from the third position on, which is what pins the convention.
        let pack = |base: u8| Narrow::from_fn(|i| Rijndael8b::from_byte(base + i as u8));
        let (a, b) = (pack(0), pack(16));
        let expect = |bytes: [u8; 16]| Narrow::from_fn(|i| Rijndael8b::from_byte(bytes[i]));

        let (left, right) = a.interleave(b, 1);
        let want_left = [0, 16, 2, 18, 4, 20, 6, 22, 8, 24, 10, 26, 12, 28, 14, 30];
        assert_eq!(left, expect(want_left));
        assert_eq!(
            right,
            expect([1, 17, 3, 19, 5, 21, 7, 23, 9, 25, 11, 27, 13, 29, 15, 31])
        );

        // The naive reading would put a_1 where the transpose puts a_2.
        let naive_left = [0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23];
        assert_ne!(want_left, naive_left, "the anchor must discriminate");

        let (left, right) = a.interleave(b, 2);
        assert_eq!(
            left,
            expect([0, 1, 16, 17, 4, 5, 20, 21, 8, 9, 24, 25, 12, 13, 28, 29])
        );
        assert_eq!(
            right,
            expect([2, 3, 18, 19, 6, 7, 22, 23, 10, 11, 26, 27, 14, 15, 30, 31])
        );

        let (left, right) = a.interleave(b, 8);
        assert_eq!(
            left,
            expect([0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23])
        );
        assert_eq!(
            right,
            expect([8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31])
        );

        // One block each leaves nothing to pair up.
        assert_eq!(a.interleave(b, 16), (a, b));
    }

    #[test]
    #[should_panic = "tried to invert zero in a block"]
    fn dividing_by_a_block_holding_zero_is_rejected() {
        let _quotient = Narrow::ONE / Narrow::ZERO;
    }

    // Each suite lives in its own module, so the three widths keep separate test names.
    //
    // The suites expand into a module of their own, which is why the paths reach out twice.
    mod narrow {
        use p3_field_testing::test_packed_binary_field;

        test_packed_binary_field!(
            super::super::Narrow,
            &[super::super::Narrow::ZERO],
            &[super::super::Narrow::ONE],
            super::super::specials::<16>()
        );
    }

    mod middle {
        use p3_field_testing::test_packed_binary_field;

        test_packed_binary_field!(
            super::super::Middle,
            &[super::super::Middle::ZERO],
            &[super::super::Middle::ONE],
            super::super::specials::<32>()
        );
    }

    mod wide {
        use p3_field_testing::test_packed_binary_field;

        test_packed_binary_field!(
            super::super::Wide,
            &[super::super::Wide::ZERO],
            &[super::super::Wide::ONE],
            super::super::specials::<64>()
        );
    }
}
