use alloc::vec::Vec;
use core::arch::aarch64::{self, uint32x4_t};
use core::iter::{Product, Sum};
use core::mem::transmute;
use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

use p3_field::{Field, FieldAlgebra, PackedField, PackedFieldPow2, PackedValue};
use p3_util::convert_vec;
use rand::distributions::{Distribution, Standard};
use rand::Rng;

use crate::Mersenne31;

const WIDTH: usize = 4;
const P: uint32x4_t = unsafe { transmute::<[u32; WIDTH], _>([0x7fffffff; WIDTH]) };

/// Vectorized NEON implementation of `Mersenne31` arithmetic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)] // This needed to make `transmute`s safe.
pub struct PackedMersenne31Neon(pub [Mersenne31; WIDTH]);

impl PackedMersenne31Neon {
    #[inline]
    #[must_use]
    /// Get an arch-specific vector representing the packed values.
    fn to_vector(self) -> uint32x4_t {
        unsafe {
            // Safety: `Mersenne31` is `repr(transparent)` so it can be transmuted to `u32`. It
            // follows that `[Mersenne31; WIDTH]` can be transmuted to `[u32; WIDTH]`, which can be
            // transmuted to `uint32x4_t`, since arrays are guaranteed to be contiguous in memory.
            // Finally `PackedMersenne31Neon` is `repr(transparent)` so it can be transmuted to
            // `[Mersenne31; WIDTH]`.
            transmute(self)
        }
    }

    #[inline]
    #[must_use]
    /// Make a packed field vector from an arch-specific vector.
    ///
    /// SAFETY: The caller must ensure that each element of `vector` represents a valid
    /// `Mersenne31`.  In particular, each element of vector must be in `0..=P` (i.e. it fits in 31
    /// bits).
    unsafe fn from_vector(vector: uint32x4_t) -> Self {
        // Safety: It is up to the user to ensure that elements of `vector` represent valid
        // `Mersenne31` values. We must only reason about memory representations. `uint32x4_t` can
        // be transmuted to `[u32; WIDTH]` (since arrays elements are contiguous in memory), which
        // can be transmuted to `[Mersenne31; WIDTH]` (since `Mersenne31` is `repr(transparent)`),
        // which in turn can be transmuted to `PackedMersenne31Neon` (since `PackedMersenne31Neon`
        // is also `repr(transparent)`).
        transmute(vector)
    }

    /// Copy `value` to all positions in a packed vector. This is the same as
    /// `From<Mersenne31>::from`, but `const`.
    #[inline]
    #[must_use]
    const fn broadcast(value: Mersenne31) -> Self {
        Self([value; WIDTH])
    }
}

impl Add for PackedMersenne31Neon {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        let lhs = self.to_vector();
        let rhs = rhs.to_vector();
        let res = add(lhs, rhs);
        unsafe {
            // Safety: `add` returns valid values when given valid values.
            Self::from_vector(res)
        }
    }
}

impl Mul for PackedMersenne31Neon {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let lhs = self.to_vector();
        let rhs = rhs.to_vector();
        let res = mul(lhs, rhs);
        unsafe {
            // Safety: `mul` returns valid values when given valid values.
            Self::from_vector(res)
        }
    }
}

impl Neg for PackedMersenne31Neon {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        let val = self.to_vector();
        let res = neg(val);
        unsafe {
            // Safety: `neg` returns valid values when given valid values.
            Self::from_vector(res)
        }
    }
}

impl Sub for PackedMersenne31Neon {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        let lhs = self.to_vector();
        let rhs = rhs.to_vector();
        let res = sub(lhs, rhs);
        unsafe {
            // Safety: `sub` returns valid values when given valid values.
            Self::from_vector(res)
        }
    }
}

/// Given a `val` in `0, ..., 2 P`, return a `res` in `0, ..., P` such that `res = val (mod P)`
#[inline]
#[must_use]
fn reduce_sum(val: uint32x4_t) -> uint32x4_t {
    // val is in 0, ..., 2 P. If val is in 0, ..., P - 1 then it is valid and
    // u := (val - P) mod 2^32 is in P <u 2^32 - P, ..., 2^32 - 1 and unsigned_min(val, u) = val as
    // desired. If val is in P + 1, ..., 2 P, then u is in 1, ..., P < P + 1 so u is valid, and
    // unsigned_min(val, u) = u as desired. The remaining case of val = P, u = 0 is trivial.

    unsafe {
        // Safety: If this code got compiled then NEON intrinsics are available.
        let u = aarch64::vsubq_u32(val, P);
        aarch64::vminq_u32(val, u)
    }
}

/// Add two vectors of Mersenne-31 field elements that fit in 31 bits.
/// If the inputs do not fit in 31 bits, the result is undefined.
#[inline]
#[must_use]
fn add(lhs: uint32x4_t, rhs: uint32x4_t) -> uint32x4_t {
    // We want this to compile to:
    //      add   t.4s, lhs.4s, rhs.4s
    //      sub   u.4s, t.4s, P.4s
    //      umin  res.4s, t.4s, u.4s
    // throughput: .75 cyc/vec (5.33 els/cyc)
    // latency: 6 cyc

    // lhs and rhs are in 0, ..., P, and we want the result to also be in that range.
    // t := lhs + rhs is in 0, ..., 2 P, so we apply reduce_sum.

    unsafe {
        // Safety: If this code got compiled then NEON intrinsics are available.
        let t = aarch64::vaddq_u32(lhs, rhs);
        reduce_sum(t)
    }
}

/// Multiply two 31-bit numbers to obtain a 62-bit immediate result, and return the high 31 bits of
/// that result. Results are arbitrary if the inputs do not fit in 31 bits.
#[inline]
#[must_use]
fn mul_31x31_to_hi_31(lhs: uint32x4_t, rhs: uint32x4_t) -> uint32x4_t {
    // This is just a wrapper around `aarch64::vqdmulhq_s32`, so we don't have to worry about the
    // casting elsewhere.
    unsafe {
        // Safety: If this code got compiled then NEON intrinsics are available.
        aarch64::vreinterpretq_u32_s32(aarch64::vqdmulhq_s32(
            aarch64::vreinterpretq_s32_u32(lhs),
            aarch64::vreinterpretq_s32_u32(rhs),
        ))
    }
}

/// Multiply vectors of Mersenne-31 field elements that fit in 31 bits.
/// If the inputs do not fit in 31 bits, the result is undefined.
#[inline]
#[must_use]
fn mul(lhs: uint32x4_t, rhs: uint32x4_t) -> uint32x4_t {
    // We want this to compile to:
    //      sqdmulh  prod_hi31.4s, lhs.4s, rhs.4s
    //      mul      t.4s, lhs.4s, rhs.4s
    //      mla      t.4s, prod_hi31.4s, P.4s
    //      sub      u.4s, t.4s, P.4s
    //      umin     res.4s, t.4s, u.4s
    // throughput: 1.25 cyc/vec (3.2 els/cyc)
    // latency: 10 cyc

    // We want to return res in 0, ..., P such that res = lhs * rhs (mod P).
    // Let prod := lhs * rhs. Break it up into prod = 2^31 prod_hi31 + prod_lo31, where both limbs
    // are in 0, ..., 2^31 - 1. Then prod = prod_hi31 + prod_lo31 (mod P), so let
    // t := prod_hi31 + prod_lo31.
    // Define prod_lo32 = prod mod 2^32 and observe that
    //   prod_lo32 = prod_lo31 + 2^31 (prod_hi31 mod 2)
    //             = prod_lo31 + 2^31 prod_hi31                                          (mod 2^32)
    // Then
    //   t = prod_lo32 - 2^31 prod_hi31 + prod_hi31                                      (mod 2^32)
    //     = prod_lo32 - (2^31 - 1) prod_hi31                                            (mod 2^32)
    //     = prod_lo32 - prod_hi31 * P                                                   (mod 2^32)
    //
    // t is in 0, ..., 2 P, so we apply reduce_sum to get the result.

    unsafe {
        // Safety: If this code got compiled then NEON intrinsics are available.
        let prod_hi31 = mul_31x31_to_hi_31(lhs, rhs);
        let prod_lo32 = aarch64::vmulq_u32(lhs, rhs);
        let t = aarch64::vmlsq_u32(prod_lo32, prod_hi31, P);
        reduce_sum(t)
    }
}

/// Negate a vector of Mersenne-31 field elements that fit in 31 bits.
/// If the inputs do not fit in 31 bits, the result is undefined.
#[inline]
#[must_use]
fn neg(val: uint32x4_t) -> uint32x4_t {
    // We want this to compile to:
    //      eor  res.16b, val.16b, P.16b
    // throughput: .25 cyc/vec (16 els/cyc)
    // latency: 2 cyc

    // val is in 0, ..., P, so res := P - val is also in 0, ..., P.

    unsafe {
        // Safety: If this code got compiled then NEON intrinsics are available.
        aarch64::vsubq_u32(P, val)
    }
}

/// Subtract vectors of Mersenne-31 field elements that fit in 31 bits.
/// If the inputs do not fit in 31 bits, the result is undefined.
#[inline]
#[must_use]
fn sub(lhs: uint32x4_t, rhs: uint32x4_t) -> uint32x4_t {
    // We want this to compile to:
    //      sub   res.4s, lhs.4s, rhs.4s
    //      cmhi  underflow.4s, rhs.4s, lhs.4s
    //      mls   res.4s, underflow.4s, P.4s
    // throughput: .75 cyc/vec (5.33 els/cyc)
    // latency: 5 cyc

    // lhs and rhs are in 0, ..., P, and we want the result to also be in that range.
    // Define: diff := (lhs - rhs) mod 2^32
    //         underflow := 2^32 - 1 if lhs <u rhs else 0
    //         res := (diff - underflow * P) mod 2^32
    // By cases:
    // 1. If lhs >=u rhs, then diff is in 0, ..., P and underflow is 0. res = diff is valid.
    // 2. Otherwise, lhs <u rhs, so diff is in 2^32 - P, ..., 2^32 - 1 and underflow is 2^32 - 1.
    //    res = (diff + P) mod 2^32 is in 0, ..., P - 1, so it is valid.

    unsafe {
        // Safety: If this code got compiled then NEON intrinsics are available.
        let diff = aarch64::vsubq_u32(lhs, rhs);
        let underflow = aarch64::vcltq_u32(lhs, rhs);
        aarch64::vmlsq_u32(diff, underflow, P)
    }
}

impl From<Mersenne31> for PackedMersenne31Neon {
    #[inline]
    fn from(value: Mersenne31) -> Self {
        Self::broadcast(value)
    }
}

impl Default for PackedMersenne31Neon {
    #[inline]
    fn default() -> Self {
        Mersenne31::default().into()
    }
}

impl AddAssign for PackedMersenne31Neon {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl MulAssign for PackedMersenne31Neon {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl SubAssign for PackedMersenne31Neon {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Sum for PackedMersenne31Neon {
    #[inline]
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.reduce(|lhs, rhs| lhs + rhs).unwrap_or(Self::ZERO)
    }
}

impl Product for PackedMersenne31Neon {
    #[inline]
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.reduce(|lhs, rhs| lhs * rhs).unwrap_or(Self::ONE)
    }
}

impl FieldAlgebra for PackedMersenne31Neon {
    type F = Mersenne31;

    const ZERO: Self = Self::broadcast(Mersenne31::ZERO);
    const ONE: Self = Self::broadcast(Mersenne31::ONE);
    const TWO: Self = Self::broadcast(Mersenne31::TWO);
    const NEG_ONE: Self = Self::broadcast(Mersenne31::NEG_ONE);

    #[inline]
    fn from_f(f: Self::F) -> Self {
        f.into()
    }
    #[inline]
    fn from_canonical_u8(n: u8) -> Self {
        Mersenne31::from_canonical_u8(n).into()
    }
    #[inline]
    fn from_canonical_u16(n: u16) -> Self {
        Mersenne31::from_canonical_u16(n).into()
    }
    #[inline]
    fn from_canonical_u32(n: u32) -> Self {
        Mersenne31::from_canonical_u32(n).into()
    }
    #[inline]
    fn from_canonical_u64(n: u64) -> Self {
        Mersenne31::from_canonical_u64(n).into()
    }
    #[inline]
    fn from_canonical_usize(n: usize) -> Self {
        Mersenne31::from_canonical_usize(n).into()
    }

    #[inline]
    fn from_wrapped_u32(n: u32) -> Self {
        Mersenne31::from_wrapped_u32(n).into()
    }
    #[inline]
    fn from_wrapped_u64(n: u64) -> Self {
        Mersenne31::from_wrapped_u64(n).into()
    }

    #[inline(always)]
    fn zero_vec(len: usize) -> Vec<Self> {
        // SAFETY: this is a repr(transparent) wrapper around an array.
        unsafe { convert_vec(Self::F::zero_vec(len * WIDTH)) }
    }
}

impl Add<Mersenne31> for PackedMersenne31Neon {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Mersenne31) -> Self {
        self + Self::from(rhs)
    }
}

impl Mul<Mersenne31> for PackedMersenne31Neon {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Mersenne31) -> Self {
        self * Self::from(rhs)
    }
}

impl Sub<Mersenne31> for PackedMersenne31Neon {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Mersenne31) -> Self {
        self - Self::from(rhs)
    }
}

impl AddAssign<Mersenne31> for PackedMersenne31Neon {
    #[inline]
    fn add_assign(&mut self, rhs: Mersenne31) {
        *self += Self::from(rhs)
    }
}

impl MulAssign<Mersenne31> for PackedMersenne31Neon {
    #[inline]
    fn mul_assign(&mut self, rhs: Mersenne31) {
        *self *= Self::from(rhs)
    }
}

impl SubAssign<Mersenne31> for PackedMersenne31Neon {
    #[inline]
    fn sub_assign(&mut self, rhs: Mersenne31) {
        *self -= Self::from(rhs)
    }
}

impl Sum<Mersenne31> for PackedMersenne31Neon {
    #[inline]
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = Mersenne31>,
    {
        iter.sum::<Mersenne31>().into()
    }
}

impl Product<Mersenne31> for PackedMersenne31Neon {
    #[inline]
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = Mersenne31>,
    {
        iter.product::<Mersenne31>().into()
    }
}

impl Div<Mersenne31> for PackedMersenne31Neon {
    type Output = Self;
    #[allow(clippy::suspicious_arithmetic_impl)]
    #[inline]
    fn div(self, rhs: Mersenne31) -> Self {
        self * rhs.inverse()
    }
}

impl Add<PackedMersenne31Neon> for Mersenne31 {
    type Output = PackedMersenne31Neon;
    #[inline]
    fn add(self, rhs: PackedMersenne31Neon) -> PackedMersenne31Neon {
        PackedMersenne31Neon::from(self) + rhs
    }
}

impl Mul<PackedMersenne31Neon> for Mersenne31 {
    type Output = PackedMersenne31Neon;
    #[inline]
    fn mul(self, rhs: PackedMersenne31Neon) -> PackedMersenne31Neon {
        PackedMersenne31Neon::from(self) * rhs
    }
}

impl Sub<PackedMersenne31Neon> for Mersenne31 {
    type Output = PackedMersenne31Neon;
    #[inline]
    fn sub(self, rhs: PackedMersenne31Neon) -> PackedMersenne31Neon {
        PackedMersenne31Neon::from(self) - rhs
    }
}

impl Distribution<PackedMersenne31Neon> for Standard {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> PackedMersenne31Neon {
        PackedMersenne31Neon(rng.gen())
    }
}

#[inline]
#[must_use]
fn interleave1(v0: uint32x4_t, v1: uint32x4_t) -> (uint32x4_t, uint32x4_t) {
    // We want this to compile to:
    //      trn1  res0.4s, v0.4s, v1.4s
    //      trn2  res1.4s, v0.4s, v1.4s
    // throughput: .5 cyc/2 vec (16 els/cyc)
    // latency: 2 cyc
    unsafe {
        // Safety: If this code got compiled then NEON intrinsics are available.
        (aarch64::vtrn1q_u32(v0, v1), aarch64::vtrn2q_u32(v0, v1))
    }
}

#[inline]
#[must_use]
fn interleave2(v0: uint32x4_t, v1: uint32x4_t) -> (uint32x4_t, uint32x4_t) {
    // We want this to compile to:
    //      trn1  res0.2d, v0.2d, v1.2d
    //      trn2  res1.2d, v0.2d, v1.2d
    // throughput: .5 cyc/2 vec (16 els/cyc)
    // latency: 2 cyc

    // To transpose 64-bit blocks, cast the [u32; 4] vectors to [u64; 2], transpose, and cast back.
    unsafe {
        // Safety: If this code got compiled then NEON intrinsics are available.
        let v0 = aarch64::vreinterpretq_u64_u32(v0);
        let v1 = aarch64::vreinterpretq_u64_u32(v1);
        (
            aarch64::vreinterpretq_u32_u64(aarch64::vtrn1q_u64(v0, v1)),
            aarch64::vreinterpretq_u32_u64(aarch64::vtrn2q_u64(v0, v1)),
        )
    }
}

unsafe impl PackedValue for PackedMersenne31Neon {
    type Value = Mersenne31;

    const WIDTH: usize = WIDTH;

    #[inline]
    fn from_slice(slice: &[Mersenne31]) -> &Self {
        assert_eq!(slice.len(), Self::WIDTH);
        unsafe {
            // Safety: `[Mersenne31; WIDTH]` can be transmuted to `PackedMersenne31Neon` since the
            // latter is `repr(transparent)`. They have the same alignment, so the reference cast
            // is safe too.
            &*slice.as_ptr().cast()
        }
    }
    #[inline]
    fn from_slice_mut(slice: &mut [Mersenne31]) -> &mut Self {
        assert_eq!(slice.len(), Self::WIDTH);
        unsafe {
            // Safety: `[Mersenne31; WIDTH]` can be transmuted to `PackedMersenne31Neon` since the
            // latter is `repr(transparent)`. They have the same alignment, so the reference cast
            // is safe too.
            &mut *slice.as_mut_ptr().cast()
        }
    }

    /// Similar to `core:array::from_fn`.
    #[inline]
    fn from_fn<F: FnMut(usize) -> Mersenne31>(f: F) -> Self {
        let vals_arr: [_; WIDTH] = core::array::from_fn(f);
        Self(vals_arr)
    }

    #[inline]
    fn as_slice(&self) -> &[Mersenne31] {
        &self.0[..]
    }
    #[inline]
    fn as_slice_mut(&mut self) -> &mut [Mersenne31] {
        &mut self.0[..]
    }
}

unsafe impl PackedField for PackedMersenne31Neon {
    type Scalar = Mersenne31;
}

unsafe impl PackedFieldPow2 for PackedMersenne31Neon {
    #[inline]
    fn interleave(&self, other: Self, block_len: usize) -> (Self, Self) {
        let (v0, v1) = (self.to_vector(), other.to_vector());
        let (res0, res1) = match block_len {
            1 => interleave1(v0, v1),
            2 => interleave2(v0, v1),
            4 => (v0, v1),
            _ => panic!("unsupported block_len"),
        };
        unsafe {
            // Safety: all values are in canonical form (we haven't changed them).
            (Self::from_vector(res0), Self::from_vector(res1))
        }
    }
}

#[cfg(test)]
mod tests {
    use p3_field_testing::test_packed_field;

    use super::{Mersenne31, WIDTH};

    /// Zero has a redundant representation, so let's test both.
    const ZEROS: [Mersenne31; WIDTH] =
        Mersenne31::new_array([0x00000000, 0x7fffffff, 0x00000000, 0x7fffffff]);

    const SPECIAL_VALS: [Mersenne31; WIDTH] =
        Mersenne31::new_array([0x00000000, 0x00000001, 0x00000002, 0x7ffffffe]);

    test_packed_field!(
        crate::PackedMersenne31Neon,
        crate::PackedMersenne31Neon(super::ZEROS),
        crate::PackedMersenne31Neon(super::SPECIAL_VALS)
    );
}
