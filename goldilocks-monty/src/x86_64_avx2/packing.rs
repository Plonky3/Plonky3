use alloc::vec;
use alloc::vec::Vec;
use core::arch::x86_64::*;
use core::fmt::Debug;
use core::iter::{Product, Sum};
use core::mem::transmute;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use p3_field::exponentiation::exp_10540996611094048183;
use p3_field::interleave::{interleave_u64, interleave_u128};
use p3_field::{
    Algebra, Field, InjectiveMonomial, PackedField, PackedFieldPow2, PackedValue,
    PermutationMonomial, PrimeCharacteristicRing,
};
use rand::Rng;
use rand::distr::{Distribution, StandardUniform};

use crate::Goldilocks;

const WIDTH: usize = 4;

/// Vectorized AVX2 implementation of `Goldilocks` Montgomery arithmetic.
/// This implementation vectorizes operations while delegating the actual
/// Montgomery arithmetic to the scalar implementations for correctness.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
#[repr(transparent)] // Needed to make `transmute`s safe.
#[must_use]
pub struct PackedGoldilocksMontyAVX2(pub [Goldilocks; WIDTH]);

impl PackedGoldilocksMontyAVX2 {
    /// Get an arch-specific vector representing the packed values.
    #[inline]
    #[must_use]
    pub(crate) fn to_vector(self) -> __m256i {
        unsafe {
            // Safety: `Goldilocks` is `repr(transparent)` so it can be transmuted to `u64`. It
            // follows that `[Goldilocks; WIDTH]` can be transmuted to `[u64; WIDTH]`, which can be
            // transmuted to `__m256i`, since arrays are guaranteed to be contiguous in memory.
            // Finally `PackedGoldilocksMontyAVX2` is `repr(transparent)` so it can be transmuted to
            // `[Goldilocks; WIDTH]`.
            transmute(self)
        }
    }

    /// Make a packed field vector from an arch-specific vector.
    #[inline]
    pub(crate) fn from_vector(vector: __m256i) -> Self {
        unsafe {
            // Safety: `__m256i` can be transmuted to `[u64; WIDTH]` (since arrays elements are
            // contiguous in memory), which can be transmuted to `[Goldilocks; WIDTH]` (since
            // `Goldilocks` is `repr(transparent)`), which in turn can be transmuted to
            // `PackedGoldilocksMontyAVX2` (since `PackedGoldilocksMontyAVX2` is also `repr(transparent)`).
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

impl From<Goldilocks> for PackedGoldilocksMontyAVX2 {
    fn from(x: Goldilocks) -> Self {
        Self::broadcast(x)
    }
}

impl Add for PackedGoldilocksMontyAVX2 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self([
            self.0[0] + rhs.0[0],
            self.0[1] + rhs.0[1],
            self.0[2] + rhs.0[2],
            self.0[3] + rhs.0[3],
        ])
    }
}

impl Sub for PackedGoldilocksMontyAVX2 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self([
            self.0[0] - rhs.0[0],
            self.0[1] - rhs.0[1],
            self.0[2] - rhs.0[2],
            self.0[3] - rhs.0[3],
        ])
    }
}

impl Neg for PackedGoldilocksMontyAVX2 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self([-self.0[0], -self.0[1], -self.0[2], -self.0[3]])
    }
}

impl Mul for PackedGoldilocksMontyAVX2 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self([
            self.0[0] * rhs.0[0],
            self.0[1] * rhs.0[1],
            self.0[2] * rhs.0[2],
            self.0[3] * rhs.0[3],
        ])
    }
}

impl AddAssign for PackedGoldilocksMontyAVX2 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl SubAssign for PackedGoldilocksMontyAVX2 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl MulAssign for PackedGoldilocksMontyAVX2 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Sum for PackedGoldilocksMontyAVX2 {
    #[inline]
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |acc, x| acc + x)
    }
}

impl Product for PackedGoldilocksMontyAVX2 {
    #[inline]
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ONE, |acc, x| acc * x)
    }
}

impl Sum<Goldilocks> for PackedGoldilocksMontyAVX2 {
    #[inline]
    fn sum<I: Iterator<Item = Goldilocks>>(iter: I) -> Self {
        iter.map(Self::from).fold(Self::ZERO, |acc, x| acc + x)
    }
}

impl Product<Goldilocks> for PackedGoldilocksMontyAVX2 {
    #[inline]
    fn product<I: Iterator<Item = Goldilocks>>(iter: I) -> Self {
        iter.map(Self::from).fold(Self::ONE, |acc, x| acc * x)
    }
}

impl Distribution<PackedGoldilocksMontyAVX2> for StandardUniform {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> PackedGoldilocksMontyAVX2 {
        PackedGoldilocksMontyAVX2([
            StandardUniform.sample(rng),
            StandardUniform.sample(rng),
            StandardUniform.sample(rng),
            StandardUniform.sample(rng),
        ])
    }
}

impl PrimeCharacteristicRing for PackedGoldilocksMontyAVX2 {
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
        ])
    }

    #[inline]
    fn square(&self) -> Self {
        Self([
            self.0[0].square(),
            self.0[1].square(),
            self.0[2].square(),
            self.0[3].square(),
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
impl InjectiveMonomial<7> for PackedGoldilocksMontyAVX2 {}

impl PermutationMonomial<7> for PackedGoldilocksMontyAVX2 {
    /// In the field `Goldilocks`, `a^{1/7}` is equal to a^{10540996611094048183}.
    ///
    /// This follows from the calculation `7*10540996611094048183 = 4*(2^64 - 2**32) + 1 = 1 mod (p - 1)`.
    fn injective_exp_root_n(&self) -> Self {
        exp_10540996611094048183(*self)
    }
}

impl Add<Goldilocks> for PackedGoldilocksMontyAVX2 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Goldilocks) -> Self {
        self + Self::from(rhs)
    }
}

impl Sub<Goldilocks> for PackedGoldilocksMontyAVX2 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Goldilocks) -> Self {
        self - Self::from(rhs)
    }
}

impl Mul<Goldilocks> for PackedGoldilocksMontyAVX2 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Goldilocks) -> Self {
        self * Self::from(rhs)
    }
}

impl Div<Goldilocks> for PackedGoldilocksMontyAVX2 {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Goldilocks) -> Self {
        self * Self::from(rhs.inverse())
    }
}

impl DivAssign<Goldilocks> for PackedGoldilocksMontyAVX2 {
    #[inline]
    fn div_assign(&mut self, rhs: Goldilocks) {
        *self = *self / rhs;
    }
}

impl AddAssign<Goldilocks> for PackedGoldilocksMontyAVX2 {
    #[inline]
    fn add_assign(&mut self, rhs: Goldilocks) {
        *self = *self + rhs;
    }
}

impl SubAssign<Goldilocks> for PackedGoldilocksMontyAVX2 {
    #[inline]
    fn sub_assign(&mut self, rhs: Goldilocks) {
        *self = *self - rhs;
    }
}

impl MulAssign<Goldilocks> for PackedGoldilocksMontyAVX2 {
    #[inline]
    fn mul_assign(&mut self, rhs: Goldilocks) {
        *self = *self * rhs;
    }
}

impl Algebra<Goldilocks> for PackedGoldilocksMontyAVX2 {}

unsafe impl PackedValue for PackedGoldilocksMontyAVX2 {
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
        Self([f(0), f(1), f(2), f(3)])
    }
}

unsafe impl PackedField for PackedGoldilocksMontyAVX2 {
    type Scalar = Goldilocks;
}

unsafe impl PackedFieldPow2 for PackedGoldilocksMontyAVX2 {
    fn interleave(&self, other: Self, block_len: usize) -> (Self, Self) {
        let (a, b) = match block_len {
            1 => interleave_u64(self.to_vector(), other.to_vector()),
            2 => interleave_u128(self.to_vector(), other.to_vector()),
            4 => {
                // For block_len=4 (full width), no interleaving is needed
                (self.to_vector(), other.to_vector())
            }
            _ => panic!("Unsupported block_len: {}", block_len),
        };
        (Self::from_vector(a), Self::from_vector(b))
    }
}

#[cfg(test)]
mod tests {
    use p3_field::PrimeCharacteristicRing;
    use p3_field_testing::test_packed_field;

    use super::{Goldilocks, PackedGoldilocksMontyAVX2, WIDTH};

    const SPECIAL_VALS: [Goldilocks; WIDTH] = [
        Goldilocks::new(0xFFFF_FFFF_0000_0000),
        Goldilocks::new(0xFFFF_FFFF_FFFF_FFFF),
        Goldilocks::new(0x0000_0000_0000_0001),
        Goldilocks::new(0xFFFF_FFFF_0000_0001),
    ];

    const ZEROS: PackedGoldilocksMontyAVX2 = PackedGoldilocksMontyAVX2([
        Goldilocks::ZERO,
        Goldilocks::ZERO,
        Goldilocks::ZERO,
        Goldilocks::ZERO,
    ]);

    const ONES: PackedGoldilocksMontyAVX2 = PackedGoldilocksMontyAVX2([
        Goldilocks::ONE,
        Goldilocks::ONE,
        Goldilocks::ONE,
        Goldilocks::ONE,
    ]);

    test_packed_field!(
        crate::PackedGoldilocksMontyAVX2,
        &[super::ZEROS],
        &[super::ONES],
        crate::PackedGoldilocksMontyAVX2(super::SPECIAL_VALS)
    );
}
