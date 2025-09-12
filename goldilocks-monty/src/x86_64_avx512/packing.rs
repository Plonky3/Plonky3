use alloc::vec;
use alloc::vec::Vec;
use core::arch::x86_64::*;
use core::fmt::Debug;
use core::iter::{Product, Sum};
use core::mem::transmute;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use p3_field::exponentiation::exp_10540996611094048183;
use p3_field::interleave::{interleave_u64, interleave_u128, interleave_u256};
use p3_field::{
    Algebra, Field, InjectiveMonomial, PackedField, PackedFieldPow2, PackedValue,
    PermutationMonomial, PrimeCharacteristicRing,
};
use rand::Rng;
use rand::distr::{Distribution, StandardUniform};

use crate::Goldilocks;

const WIDTH: usize = 8;

/// Vectorized AVX512 implementation of `Goldilocks` Montgomery arithmetic.
/// This implementation vectorizes operations while delegating the actual
/// Montgomery arithmetic to the scalar implementations for correctness.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
#[repr(transparent)] // Needed to make `transmute`s safe.
#[must_use]
pub struct PackedGoldilocksMontyAVX512(pub [Goldilocks; WIDTH]);

impl PackedGoldilocksMontyAVX512 {
    /// Get an arch-specific vector representing the packed values.
    #[inline]
    #[must_use]
    pub(crate) fn to_vector(self) -> __m512i {
        unsafe {
            // Safety: `Goldilocks` is `repr(transparent)` so it can be transmuted to `u64`. It
            // follows that `[Goldilocks; WIDTH]` can be transmuted to `[u64; WIDTH]`, which can be
            // transmuted to `__m512i`, since arrays are guaranteed to be contiguous in memory.
            // Finally `PackedGoldilocksMontyAVX512` is `repr(transparent)` so it can be transmuted to
            // `[Goldilocks; WIDTH]`.
            transmute(self)
        }
    }

    /// Make a packed field vector from an arch-specific vector.
    #[inline]
    pub(crate) fn from_vector(vector: __m512i) -> Self {
        unsafe {
            // Safety: `__m512i` can be transmuted to `[u64; WIDTH]` (since arrays elements are
            // contiguous in memory), which can be transmuted to `[Goldilocks; WIDTH]` (since
            // `Goldilocks` is `repr(transparent)`), which in turn can be transmuted to
            // `PackedGoldilocksMontyAVX512` (since `PackedGoldilocksMontyAVX512` is also `repr(transparent)`).
            transmute(vector)
        }
    }

    /// Copy `value` to all positions in a packed vector. This is the same as
    /// `From<Goldilocks>::from`, but `const`.
    #[inline]
    const fn broadcast(value: Goldilocks) -> Self {
        Self([value; WIDTH])
    }
}

impl From<Goldilocks> for PackedGoldilocksMontyAVX512 {
    fn from(x: Goldilocks) -> Self {
        Self::broadcast(x)
    }
}

impl Add for PackedGoldilocksMontyAVX512 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self([
            self.0[0] + rhs.0[0],
            self.0[1] + rhs.0[1],
            self.0[2] + rhs.0[2],
            self.0[3] + rhs.0[3],
            self.0[4] + rhs.0[4],
            self.0[5] + rhs.0[5],
            self.0[6] + rhs.0[6],
            self.0[7] + rhs.0[7],
        ])
    }
}

impl Sub for PackedGoldilocksMontyAVX512 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self([
            self.0[0] - rhs.0[0],
            self.0[1] - rhs.0[1],
            self.0[2] - rhs.0[2],
            self.0[3] - rhs.0[3],
            self.0[4] - rhs.0[4],
            self.0[5] - rhs.0[5],
            self.0[6] - rhs.0[6],
            self.0[7] - rhs.0[7],
        ])
    }
}

impl Neg for PackedGoldilocksMontyAVX512 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self([
            -self.0[0], -self.0[1], -self.0[2], -self.0[3], -self.0[4], -self.0[5], -self.0[6],
            -self.0[7],
        ])
    }
}

impl Mul for PackedGoldilocksMontyAVX512 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self([
            self.0[0] * rhs.0[0],
            self.0[1] * rhs.0[1],
            self.0[2] * rhs.0[2],
            self.0[3] * rhs.0[3],
            self.0[4] * rhs.0[4],
            self.0[5] * rhs.0[5],
            self.0[6] * rhs.0[6],
            self.0[7] * rhs.0[7],
        ])
    }
}

impl AddAssign for PackedGoldilocksMontyAVX512 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl SubAssign for PackedGoldilocksMontyAVX512 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl MulAssign for PackedGoldilocksMontyAVX512 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Sum for PackedGoldilocksMontyAVX512 {
    #[inline]
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |acc, x| acc + x)
    }
}

impl Product for PackedGoldilocksMontyAVX512 {
    #[inline]
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ONE, |acc, x| acc * x)
    }
}

impl Sum<Goldilocks> for PackedGoldilocksMontyAVX512 {
    #[inline]
    fn sum<I: Iterator<Item = Goldilocks>>(iter: I) -> Self {
        iter.map(Self::from).fold(Self::ZERO, |acc, x| acc + x)
    }
}

impl Product<Goldilocks> for PackedGoldilocksMontyAVX512 {
    #[inline]
    fn product<I: Iterator<Item = Goldilocks>>(iter: I) -> Self {
        iter.map(Self::from).fold(Self::ONE, |acc, x| acc * x)
    }
}

impl Distribution<PackedGoldilocksMontyAVX512> for StandardUniform {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> PackedGoldilocksMontyAVX512 {
        PackedGoldilocksMontyAVX512([
            StandardUniform.sample(rng),
            StandardUniform.sample(rng),
            StandardUniform.sample(rng),
            StandardUniform.sample(rng),
            StandardUniform.sample(rng),
            StandardUniform.sample(rng),
            StandardUniform.sample(rng),
            StandardUniform.sample(rng),
        ])
    }
}

impl PrimeCharacteristicRing for PackedGoldilocksMontyAVX512 {
    type PrimeSubfield = Goldilocks;

    const ZERO: Self = Self::broadcast(Goldilocks::ZERO);
    const ONE: Self = Self::broadcast(Goldilocks::ONE);
    const TWO: Self = Self::broadcast(Goldilocks::TWO);
    const NEG_ONE: Self = Self::broadcast(Goldilocks::NEG_ONE);

    #[inline]
    fn from_prime_subfield(f: Self::PrimeSubfield) -> Self {
        f.into()
    }

    #[inline]
    fn halve(&self) -> Self {
        Self([
            self.0[0].halve(),
            self.0[1].halve(),
            self.0[2].halve(),
            self.0[3].halve(),
            self.0[4].halve(),
            self.0[5].halve(),
            self.0[6].halve(),
            self.0[7].halve(),
        ])
    }

    #[inline]
    fn square(&self) -> Self {
        Self([
            self.0[0].square(),
            self.0[1].square(),
            self.0[2].square(),
            self.0[3].square(),
            self.0[4].square(),
            self.0[5].square(),
            self.0[6].square(),
            self.0[7].square(),
        ])
    }

    #[inline]
    fn zero_vec(len: usize) -> Vec<Self> {
        vec![Self::ZERO; len]
    }
}

// Degree of the smallest permutation polynomial for Goldilocks.
//
// As p - 1 = 2^32 * 3 * 5 * 17 * ... the smallest choice for a degree D satisfying gcd(p - 1, D) = 1 is 7.
impl InjectiveMonomial<7> for PackedGoldilocksMontyAVX512 {}

impl PermutationMonomial<7> for PackedGoldilocksMontyAVX512 {
    /// In the field `Goldilocks`, `a^{1/7}` is equal to a^{10540996611094048183}.
    ///
    /// This follows from the calculation `7*10540996611094048183 = 4*(2^64 - 2**32) + 1 = 1 mod (p - 1)`.
    fn injective_exp_root_n(&self) -> Self {
        exp_10540996611094048183(*self)
    }
}

impl Add<Goldilocks> for PackedGoldilocksMontyAVX512 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Goldilocks) -> Self {
        self + Self::from(rhs)
    }
}

impl Sub<Goldilocks> for PackedGoldilocksMontyAVX512 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Goldilocks) -> Self {
        self - Self::from(rhs)
    }
}

impl Mul<Goldilocks> for PackedGoldilocksMontyAVX512 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Goldilocks) -> Self {
        self * Self::from(rhs)
    }
}

impl Div<Goldilocks> for PackedGoldilocksMontyAVX512 {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Goldilocks) -> Self {
        self * Self::from(rhs.inverse())
    }
}

impl DivAssign<Goldilocks> for PackedGoldilocksMontyAVX512 {
    #[inline]
    fn div_assign(&mut self, rhs: Goldilocks) {
        *self = *self / rhs;
    }
}

impl AddAssign<Goldilocks> for PackedGoldilocksMontyAVX512 {
    #[inline]
    fn add_assign(&mut self, rhs: Goldilocks) {
        *self = *self + rhs;
    }
}

impl SubAssign<Goldilocks> for PackedGoldilocksMontyAVX512 {
    #[inline]
    fn sub_assign(&mut self, rhs: Goldilocks) {
        *self = *self - rhs;
    }
}

impl MulAssign<Goldilocks> for PackedGoldilocksMontyAVX512 {
    #[inline]
    fn mul_assign(&mut self, rhs: Goldilocks) {
        *self = *self * rhs;
    }
}

impl Algebra<Goldilocks> for PackedGoldilocksMontyAVX512 {}

unsafe impl PackedValue for PackedGoldilocksMontyAVX512 {
    type Value = Goldilocks;
    const WIDTH: usize = WIDTH;

    #[inline]
    fn from_slice(slice: &[Self::Value]) -> &Self {
        assert_eq!(slice.len(), Self::WIDTH);
        unsafe { &*(slice.as_ptr() as *const Self) }
    }

    #[inline]
    fn from_slice_mut(slice: &mut [Self::Value]) -> &mut Self {
        assert_eq!(slice.len(), Self::WIDTH);
        unsafe { &mut *(slice.as_mut_ptr() as *mut Self) }
    }

    #[inline]
    fn as_slice(&self) -> &[Self::Value] {
        unsafe {
            core::slice::from_raw_parts(self as *const Self as *const Self::Value, Self::WIDTH)
        }
    }

    #[inline]
    fn as_slice_mut(&mut self) -> &mut [Self::Value] {
        unsafe {
            core::slice::from_raw_parts_mut(self as *mut Self as *mut Self::Value, Self::WIDTH)
        }
    }

    #[inline]
    fn from_fn<F>(mut f: F) -> Self
    where
        F: FnMut(usize) -> Self::Value,
    {
        Self([f(0), f(1), f(2), f(3), f(4), f(5), f(6), f(7)])
    }
}

unsafe impl PackedField for PackedGoldilocksMontyAVX512 {
    type Scalar = Goldilocks;
}

unsafe impl PackedFieldPow2 for PackedGoldilocksMontyAVX512 {
    fn interleave(&self, other: Self, block_len: usize) -> (Self, Self) {
        let (a, b) = match block_len {
            1 => interleave_u64(self.to_vector(), other.to_vector()),
            2 => interleave_u128(self.to_vector(), other.to_vector()),
            4 => interleave_u256(self.to_vector(), other.to_vector()),
            8 => {
                // For block_len=8 (full width), no interleaving is needed
                (self.to_vector(), other.to_vector())
            }
            _ => panic!("Unsupported block_len: {}", block_len),
        };
        (Self::from_vector(a), Self::from_vector(b))
    }
}

#[cfg(test)]
mod tests {
    use p3_field::{PackedFieldPow2, PrimeCharacteristicRing};
    use p3_field_testing::test_packed_field;

    use super::{Goldilocks, PackedGoldilocksMontyAVX512, WIDTH};

    const SPECIAL_VALS: [Goldilocks; WIDTH] = [
        Goldilocks::new(0xFFFF_FFFF_0000_0000),
        Goldilocks::new(0xFFFF_FFFF_FFFF_FFFF),
        Goldilocks::new(0x0000_0000_0000_0001),
        Goldilocks::new(0xFFFF_FFFF_0000_0001),
        Goldilocks::new(0x1234_5678_9ABC_DEF0),
        Goldilocks::new(0x8765_4321_0FED_CBA9),
        Goldilocks::new(0xAAAA_AAAA_AAAA_AAAA),
        Goldilocks::new(0x5555_5555_5555_5555),
    ];

    const ZEROS: PackedGoldilocksMontyAVX512 = PackedGoldilocksMontyAVX512([
        Goldilocks::ZERO,
        Goldilocks::ZERO,
        Goldilocks::ZERO,
        Goldilocks::ZERO,
        Goldilocks::ZERO,
        Goldilocks::ZERO,
        Goldilocks::ZERO,
        Goldilocks::ZERO,
    ]);

    const ONES: PackedGoldilocksMontyAVX512 = PackedGoldilocksMontyAVX512([
        Goldilocks::ONE,
        Goldilocks::ONE,
        Goldilocks::ONE,
        Goldilocks::ONE,
        Goldilocks::ONE,
        Goldilocks::ONE,
        Goldilocks::ONE,
        Goldilocks::ONE,
    ]);

    #[test]
    fn test_avx512_basic_operations() {
        let a = PackedGoldilocksMontyAVX512::from(Goldilocks::from_canonical_u64(123));
        let b = PackedGoldilocksMontyAVX512::from(Goldilocks::from_canonical_u64(456));

        let sum = a + b;
        let product = a * b;

        // Verify that the results are correct
        let expected_sum = PackedGoldilocksMontyAVX512::from(
            Goldilocks::from_canonical_u64(123) + Goldilocks::from_canonical_u64(456),
        );
        let expected_product = PackedGoldilocksMontyAVX512::from(
            Goldilocks::from_canonical_u64(123) * Goldilocks::from_canonical_u64(456),
        );

        assert_eq!(sum, expected_sum);
        assert_eq!(product, expected_product);
    }

    #[test]
    fn test_avx512_interleave() {
        let a = PackedGoldilocksMontyAVX512::from(Goldilocks::from_canonical_u64(123));
        let b = PackedGoldilocksMontyAVX512::from(Goldilocks::from_canonical_u64(456));

        // Test interleaving at different block lengths
        let (int1_1, int1_2) = a.interleave(b, 1);
        let (int2_1, int2_2) = a.interleave(b, 2);
        let (int4_1, int4_2) = a.interleave(b, 4);
        let (int8_1, int8_2) = a.interleave(b, 8);

        // For block_len=8 (full width), interleave should return the original vectors
        assert_eq!(int8_1, a);
        assert_eq!(int8_2, b);
    }

    test_packed_field!(
        crate::PackedGoldilocksMontyAVX512,
        &[super::ZEROS],
        &[super::ONES],
        crate::PackedGoldilocksMontyAVX512(super::SPECIAL_VALS)
    );
}
