use core::arch::aarch64::{self, uint32x4_t};
use core::iter::{Product, Sum};
use core::mem::transmute;
use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

use p3_field::{AbstractField, AbstractionOf, Field, PackedField};

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
    #[must_use]
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
    #[must_use]
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
    #[must_use]
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
    #[must_use]
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
    #[must_use]
    fn from(value: Mersenne31) -> Self {
        Self::broadcast(value)
    }
}

impl Default for PackedMersenne31Neon {
    #[inline]
    #[must_use]
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
    #[must_use]
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.reduce(|lhs, rhs| lhs + rhs).unwrap_or(Self::ZERO)
    }
}

impl Product for PackedMersenne31Neon {
    #[inline]
    #[must_use]
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.reduce(|lhs, rhs| lhs * rhs).unwrap_or(Self::ONE)
    }
}

impl AbstractField for PackedMersenne31Neon {
    const ZERO: Self = Self::broadcast(Mersenne31::ZERO);
    const ONE: Self = Self::broadcast(Mersenne31::ONE);
    const TWO: Self = Self::broadcast(Mersenne31::TWO);
    const NEG_ONE: Self = Self::broadcast(Mersenne31::NEG_ONE);

    #[inline]
    #[must_use]
    fn from_bool(b: bool) -> Self {
        Mersenne31::from_bool(b).into()
    }
    #[inline]
    #[must_use]
    fn from_canonical_u8(n: u8) -> Self {
        Mersenne31::from_canonical_u8(n).into()
    }
    #[inline]
    #[must_use]
    fn from_canonical_u16(n: u16) -> Self {
        Mersenne31::from_canonical_u16(n).into()
    }
    #[inline]
    #[must_use]
    fn from_canonical_u32(n: u32) -> Self {
        Mersenne31::from_canonical_u32(n).into()
    }
    #[inline]
    #[must_use]
    fn from_canonical_u64(n: u64) -> Self {
        Mersenne31::from_canonical_u64(n).into()
    }
    #[inline]
    #[must_use]
    fn from_canonical_usize(n: usize) -> Self {
        Mersenne31::from_canonical_usize(n).into()
    }

    #[inline]
    #[must_use]
    fn from_wrapped_u32(n: u32) -> Self {
        Mersenne31::from_wrapped_u32(n).into()
    }
    #[inline]
    #[must_use]
    fn from_wrapped_u64(n: u64) -> Self {
        Mersenne31::from_wrapped_u64(n).into()
    }

    #[inline]
    #[must_use]
    fn multiplicative_group_generator() -> Self {
        Mersenne31::multiplicative_group_generator().into()
    }
}

impl Add<Mersenne31> for PackedMersenne31Neon {
    type Output = Self;
    #[inline]
    #[must_use]
    fn add(self, rhs: Mersenne31) -> Self {
        self + Self::from(rhs)
    }
}

impl Mul<Mersenne31> for PackedMersenne31Neon {
    type Output = Self;
    #[inline]
    #[must_use]
    fn mul(self, rhs: Mersenne31) -> Self {
        self * Self::from(rhs)
    }
}

impl Sub<Mersenne31> for PackedMersenne31Neon {
    type Output = Self;
    #[inline]
    #[must_use]
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
    #[must_use]
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = Mersenne31>,
    {
        iter.sum::<Mersenne31>().into()
    }
}

impl Product<Mersenne31> for PackedMersenne31Neon {
    #[inline]
    #[must_use]
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = Mersenne31>,
    {
        iter.product::<Mersenne31>().into()
    }
}

impl AbstractionOf<Mersenne31> for PackedMersenne31Neon {}

impl Div<Mersenne31> for PackedMersenne31Neon {
    type Output = Self;
    #[allow(clippy::suspicious_arithmetic_impl)]
    #[inline]
    #[must_use]
    fn div(self, rhs: Mersenne31) -> Self {
        self * rhs.inverse()
    }
}

impl Add<PackedMersenne31Neon> for Mersenne31 {
    type Output = PackedMersenne31Neon;
    #[inline]
    #[must_use]
    fn add(self, rhs: PackedMersenne31Neon) -> PackedMersenne31Neon {
        PackedMersenne31Neon::from(self) + rhs
    }
}

impl Mul<PackedMersenne31Neon> for Mersenne31 {
    type Output = PackedMersenne31Neon;
    #[inline]
    #[must_use]
    fn mul(self, rhs: PackedMersenne31Neon) -> PackedMersenne31Neon {
        PackedMersenne31Neon::from(self) * rhs
    }
}

impl Sub<PackedMersenne31Neon> for Mersenne31 {
    type Output = PackedMersenne31Neon;
    #[inline]
    #[must_use]
    fn sub(self, rhs: PackedMersenne31Neon) -> PackedMersenne31Neon {
        PackedMersenne31Neon::from(self) - rhs
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

unsafe impl PackedField for PackedMersenne31Neon {
    type Scalar = Mersenne31;

    const WIDTH: usize = WIDTH;

    #[inline]
    #[must_use]
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
    #[must_use]
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
    #[must_use]
    fn as_slice(&self) -> &[Mersenne31] {
        &self.0[..]
    }
    #[inline]
    #[must_use]
    fn as_slice_mut(&mut self) -> &mut [Mersenne31] {
        &mut self.0[..]
    }

    #[inline]
    #[must_use]
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
    use super::*;

    type F = Mersenne31;
    type P = PackedMersenne31Neon;

    fn array_from_canonical(vals: [u32; WIDTH]) -> [F; WIDTH] {
        vals.map(|v| F::from_canonical_u32(v))
    }

    fn packed_from_canonical(vals: [u32; WIDTH]) -> P {
        PackedMersenne31Neon(array_from_canonical(vals))
    }

    #[test]
    fn test_interleave_1() {
        let vec0 = packed_from_canonical([1, 2, 3, 4]);
        let vec1 = packed_from_canonical([5, 6, 7, 8]);

        let expected0 = packed_from_canonical([1, 5, 3, 7]);
        let expected1 = packed_from_canonical([2, 6, 4, 8]);

        let (res0, res1) = vec0.interleave(vec1, 1);
        assert_eq!(res0, expected0);
        assert_eq!(res1, expected1);
    }

    #[test]
    fn test_interleave_2() {
        let vec0 = packed_from_canonical([1, 2, 3, 4]);
        let vec1 = packed_from_canonical([5, 6, 7, 8]);

        let expected0 = packed_from_canonical([1, 2, 5, 6]);
        let expected1 = packed_from_canonical([3, 4, 7, 8]);

        let (res0, res1) = vec0.interleave(vec1, 2);
        assert_eq!(res0, expected0);
        assert_eq!(res1, expected1);
    }

    #[test]
    fn test_interleave_4() {
        let vec0 = packed_from_canonical([1, 2, 3, 4]);
        let vec1 = packed_from_canonical([5, 6, 7, 8]);

        let (res0, res1) = vec0.interleave(vec1, 4);
        assert_eq!(res0, vec0);
        assert_eq!(res1, vec1);
    }

    #[test]
    fn test_add_associative() {
        let vec0 = packed_from_canonical([0x5379f3d7, 0x702b9db2, 0x6f54190a, 0x0fd40697]);
        let vec1 = packed_from_canonical([0x4e1ce6a6, 0x07100ca0, 0x0f27d0e8, 0x6ab0f017]);
        let vec2 = packed_from_canonical([0x3767261e, 0x46966e27, 0x25690f5a, 0x2ba2b5fa]);

        let res0 = (vec0 + vec1) + vec2;
        let res1 = vec0 + (vec1 + vec2);

        assert_eq!(res0, res1);
    }

    #[test]
    fn test_add_commutative() {
        let vec0 = packed_from_canonical([0x4431e0aa, 0x3f7cac53, 0x6c65b84f, 0x393370c6]);
        let vec1 = packed_from_canonical([0x13f3646a, 0x17bab2b2, 0x154d5424, 0x58a5a24c]);

        let res0 = vec0 + vec1;
        let res1 = vec1 + vec0;

        assert_eq!(res0, res1);
    }

    #[test]
    fn test_additive_identity_right() {
        let vec = packed_from_canonical([0x37585a7d, 0x6f1de589, 0x41e1be7e, 0x712071b8]);
        let res = vec + P::ZERO;
        assert_eq!(res, vec);
    }

    #[test]
    fn test_additive_identity_left() {
        let vec = packed_from_canonical([0x2456f91e, 0x0783a205, 0x58826627, 0x1a5e3f16]);
        let res = P::ZERO + vec;
        assert_eq!(res, vec);
    }

    #[test]
    fn test_additive_inverse_add_neg() {
        let vec = packed_from_canonical([0x28267ebf, 0x0b83d23e, 0x67a59e3d, 0x0ba2fb25]);
        let neg_vec = -vec;
        let res = vec + neg_vec;
        assert_eq!(res, P::ZERO);
    }

    #[test]
    fn test_additive_inverse_sub() {
        let vec = packed_from_canonical([0x2f0a7c0e, 0x50163480, 0x12eac826, 0x2e52b121]);
        let res = vec - vec;
        assert_eq!(res, P::ZERO);
    }

    #[test]
    fn test_sub_anticommutative() {
        let vec0 = packed_from_canonical([0x0a715ea4, 0x17877e5e, 0x1a67e27c, 0x29e13b42]);
        let vec1 = packed_from_canonical([0x4168263c, 0x3c9fc759, 0x435424e9, 0x5cac2afd]);

        let res0 = vec0 - vec1;
        let res1 = -(vec1 - vec0);

        assert_eq!(res0, res1);
    }

    #[test]
    fn test_sub_zero() {
        let vec = packed_from_canonical([0x10df1248, 0x65050015, 0x73151d8d, 0x443341a8]);
        let res = vec - P::ZERO;
        assert_eq!(res, vec);
    }

    #[test]
    fn test_zero_sub() {
        let vec = packed_from_canonical([0x1af0d41c, 0x3c1795f4, 0x54da13f5, 0x43cd3f94]);
        let res0 = P::ZERO - vec;
        let res1 = -vec;
        assert_eq!(res0, res1);
    }

    #[test]
    fn test_neg_own_inverse() {
        let vec = packed_from_canonical([0x25335335, 0x32d48910, 0x74468a5f, 0x61906a18]);
        let res = --vec;
        assert_eq!(res, vec);
    }

    #[test]
    fn test_sub_is_add_neg() {
        let vec0 = packed_from_canonical([0x2ab6719a, 0x0991137e, 0x0e5c6bea, 0x1dbbb162]);
        let vec1 = packed_from_canonical([0x26c7239d, 0x56a2318b, 0x1a839b59, 0x1ec6f977]);
        let res0 = vec0 - vec1;
        let res1 = vec0 + (-vec1);
        assert_eq!(res0, res1);
    }

    #[test]
    fn test_mul_associative() {
        let vec0 = packed_from_canonical([0x3b442fc7, 0x15b736fc, 0x5daa6c48, 0x4995dea0]);
        let vec1 = packed_from_canonical([0x582918b6, 0x55b89326, 0x3b579856, 0x10769872]);
        let vec2 = packed_from_canonical([0x6a7bbe26, 0x7139a20b, 0x280f42d5, 0x0efde6a8]);

        let res0 = (vec0 * vec1) * vec2;
        let res1 = vec0 * (vec1 * vec2);

        assert_eq!(res0, res1);
    }

    #[test]
    fn test_mul_commutative() {
        let vec0 = packed_from_canonical([0x18e2fe1a, 0x54cb2eed, 0x35662447, 0x5be20656]);
        let vec1 = packed_from_canonical([0x7715ab49, 0x1937ec0d, 0x561c3def, 0x14f502f9]);

        let res0 = vec0 * vec1;
        let res1 = vec1 * vec0;

        assert_eq!(res0, res1);
    }

    #[test]
    fn test_multiplicative_identity_right() {
        let vec = packed_from_canonical([0x64628378, 0x345e3dc8, 0x766770eb, 0x21e5ad7c]);
        let res = vec * P::ONE;
        assert_eq!(res, vec);
    }

    #[test]
    fn test_multiplicative_identity_left() {
        let vec = packed_from_canonical([0x48910ae4, 0x4dd95ad3, 0x334eaf5e, 0x44e5d03b]);
        let res = P::ONE * vec;
        assert_eq!(res, vec);
    }

    #[test]
    fn test_multiplicative_inverse() {
        let vec = packed_from_canonical([0x1b288c21, 0x600c50af, 0x3ea44d7a, 0x62209fc9]);
        let inverses = packed_from_canonical([0x3a133939, 0x4736cf9a, 0x1e94daf7, 0x40eb93f3]);
        let res = vec * inverses;
        assert_eq!(res, P::ONE);
    }

    #[test]
    fn test_mul_zero() {
        let vec = packed_from_canonical([0x675f87cd, 0x2bb57f1b, 0x1b636b90, 0x25fd5dbc]);
        let res = vec * P::ZERO;
        assert_eq!(res, P::ZERO);
    }

    #[test]
    fn test_zero_mul() {
        let vec = packed_from_canonical([0x76d898cd, 0x12fed26d, 0x385dd0ea, 0x0a6cfb68]);
        let res = P::ZERO * vec;
        assert_eq!(res, P::ZERO);
    }

    #[test]
    fn test_mul_negone() {
        let vec = packed_from_canonical([0x3ac44c8d, 0x2690778c, 0x64c25465, 0x60c62b6d]);
        let res0 = vec * P::NEG_ONE;
        let res1 = -vec;
        assert_eq!(res0, res1);
    }

    #[test]
    fn test_negone_mul() {
        let vec = packed_from_canonical([0x45fdb5d9, 0x3e2571d7, 0x1438d182, 0x6addc720]);
        let res0 = P::NEG_ONE * vec;
        let res1 = -vec;
        assert_eq!(res0, res1);
    }

    #[test]
    fn test_neg_distributivity_left() {
        let vec0 = packed_from_canonical([0x347079a0, 0x09f865aa, 0x3f469975, 0x48436fa4]);
        let vec1 = packed_from_canonical([0x354839ad, 0x6f464895, 0x2afb410c, 0x2918c070]);

        let res0 = vec0 * -vec1;
        let res1 = -(vec0 * vec1);

        assert_eq!(res0, res1);
    }

    #[test]
    fn test_neg_distributivity_right() {
        let vec0 = packed_from_canonical([0x62fda8dc, 0x15a702d3, 0x4ee8e5a4, 0x2e8ea106]);
        let vec1 = packed_from_canonical([0x606f79ae, 0x3cc952a6, 0x43e31901, 0x34721ad8]);

        let res0 = -vec0 * vec1;
        let res1 = -(vec0 * vec1);

        assert_eq!(res0, res1);
    }

    #[test]
    fn test_add_distributivity_left() {
        let vec0 = packed_from_canonical([0x46b0c8a7, 0x1f3058ee, 0x44451138, 0x3c97af99]);
        let vec1 = packed_from_canonical([0x6247b46a, 0x0614b336, 0x76730d3c, 0x15b1ab60]);
        let vec2 = packed_from_canonical([0x20619eaf, 0x628800a8, 0x672c9d96, 0x44de32c3]);

        let res0 = vec0 * (vec1 + vec2);
        let res1 = vec0 * vec1 + vec0 * vec2;

        assert_eq!(res0, res1);
    }

    #[test]
    fn test_add_distributivity_right() {
        let vec0 = packed_from_canonical([0x0829c9c5, 0x6b66bdcb, 0x4e906be1, 0x16f11cfa]);
        let vec1 = packed_from_canonical([0x482922d7, 0x72816043, 0x5d63df54, 0x58ca0b7d]);
        let vec2 = packed_from_canonical([0x2127f6c0, 0x0814236c, 0x339d4b6f, 0x24d2b44d]);

        let res0 = (vec0 + vec1) * vec2;
        let res1 = vec0 * vec2 + vec1 * vec2;

        assert_eq!(res0, res1);
    }

    #[test]
    fn test_sub_distributivity_left() {
        let vec0 = packed_from_canonical([0x1c123d16, 0x62d3de88, 0x64ff0336, 0x474de37c]);
        let vec1 = packed_from_canonical([0x06758404, 0x295c96ca, 0x6ffbc647, 0x3b111808]);
        let vec2 = packed_from_canonical([0x591a66de, 0x6b69fbb6, 0x2d206c14, 0x6e5f7d0d]);

        let res0 = vec0 * (vec1 - vec2);
        let res1 = vec0 * vec1 - vec0 * vec2;

        assert_eq!(res0, res1);
    }

    #[test]
    fn test_sub_distributivity_right() {
        let vec0 = packed_from_canonical([0x00252ae1, 0x3e07c401, 0x6fd67c67, 0x767af10f]);
        let vec1 = packed_from_canonical([0x5c44c949, 0x180dc429, 0x0ccd2a7b, 0x51258be1]);
        let vec2 = packed_from_canonical([0x5126fb21, 0x58ed3919, 0x6a2f735d, 0x05ab2a69]);

        let res0 = (vec0 - vec1) * vec2;
        let res1 = vec0 * vec2 - vec1 * vec2;

        assert_eq!(res0, res1);
    }

    #[test]
    fn test_one_plus_one() {
        assert_eq!(P::ONE + P::ONE, P::TWO);
    }

    #[test]
    fn test_negone_plus_two() {
        assert_eq!(P::NEG_ONE + P::TWO, P::ONE);
    }

    #[test]
    fn test_double() {
        let vec = packed_from_canonical([0x6fc7aefd, 0x5166e726, 0x21e648d2, 0x1dd0790f]);
        let res0 = P::TWO * vec;
        let res1 = vec + vec;
        assert_eq!(res0, res1);
    }

    #[test]
    fn test_add_vs_scalar() {
        let arr0 = array_from_canonical([0x496d8163, 0x68125590, 0x191cd03b, 0x65b9abef]);
        let arr1 = array_from_canonical([0x6db594e1, 0x5b1f6289, 0x74f15e13, 0x546936a8]);

        let vec0 = PackedMersenne31Neon(arr0);
        let vec1 = PackedMersenne31Neon(arr1);
        let vec_res = vec0 + vec1;

        for i in 0..WIDTH {
            assert_eq!(vec_res.0[i], arr0[i] + arr1[i]);
        }
    }

    #[test]
    fn test_add_vs_scalar_special_vals_left() {
        let arr0 = [F::ZERO, F::ONE, F::TWO, F::NEG_ONE];
        let arr1 = array_from_canonical([0x4205a2f6, 0x6f4715f1, 0x29ed7f70, 0x70915992]);

        let vec0 = PackedMersenne31Neon(arr0);
        let vec1 = PackedMersenne31Neon(arr1);
        let vec_res = vec0 + vec1;

        for i in 0..WIDTH {
            assert_eq!(vec_res.0[i], arr0[i] + arr1[i]);
        }
    }

    #[test]
    fn test_add_vs_scalar_special_vals_right() {
        let arr0 = array_from_canonical([0x5f8329a7, 0x0f1166bb, 0x657bcb14, 0x0185c34a]);
        let arr1 = [F::ZERO, F::ONE, F::TWO, F::NEG_ONE];

        let vec0 = PackedMersenne31Neon(arr0);
        let vec1 = PackedMersenne31Neon(arr1);
        let vec_res = vec0 + vec1;

        for i in 0..WIDTH {
            assert_eq!(vec_res.0[i], arr0[i] + arr1[i]);
        }
    }

    #[test]
    fn test_sub_vs_scalar() {
        let arr0 = array_from_canonical([0x6daef778, 0x0e868440, 0x54e7ca64, 0x01a9acab]);
        let arr1 = array_from_canonical([0x45609584, 0x67b63536, 0x0f72a573, 0x234a312e]);

        let vec0 = PackedMersenne31Neon(arr0);
        let vec1 = PackedMersenne31Neon(arr1);
        let vec_res = vec0 - vec1;

        for i in 0..WIDTH {
            assert_eq!(vec_res.0[i], arr0[i] - arr1[i]);
        }
    }

    #[test]
    fn test_sub_vs_scalar_special_vals_left() {
        let arr0 = [F::ZERO, F::ONE, F::TWO, F::NEG_ONE];
        let arr1 = array_from_canonical([0x4205a2f6, 0x6f4715f1, 0x29ed7f70, 0x70915992]);

        let vec0 = PackedMersenne31Neon(arr0);
        let vec1 = PackedMersenne31Neon(arr1);
        let vec_res = vec0 - vec1;

        for i in 0..WIDTH {
            assert_eq!(vec_res.0[i], arr0[i] - arr1[i]);
        }
    }

    #[test]
    fn test_sub_vs_scalar_special_vals_right() {
        let arr0 = array_from_canonical([0x5f8329a7, 0x0f1166bb, 0x657bcb14, 0x0185c34a]);
        let arr1 = [F::ZERO, F::ONE, F::TWO, F::NEG_ONE];

        let vec0 = PackedMersenne31Neon(arr0);
        let vec1 = PackedMersenne31Neon(arr1);
        let vec_res = vec0 - vec1;

        for i in 0..WIDTH {
            assert_eq!(vec_res.0[i], arr0[i] - arr1[i]);
        }
    }

    #[test]
    fn test_mul_vs_scalar() {
        let arr0 = array_from_canonical([0x13655880, 0x5223ea02, 0x5d7f4f90, 0x1494b624]);
        let arr1 = array_from_canonical([0x0ad5743c, 0x44956741, 0x533bc885, 0x7723a25b]);

        let vec0 = PackedMersenne31Neon(arr0);
        let vec1 = PackedMersenne31Neon(arr1);
        let vec_res = vec0 * vec1;

        for i in 0..WIDTH {
            assert_eq!(vec_res.0[i], arr0[i] * arr1[i]);
        }
    }

    #[test]
    fn test_mul_vs_scalar_special_vals_left() {
        let arr0 = [F::ZERO, F::ONE, F::TWO, F::NEG_ONE];
        let arr1 = array_from_canonical([0x4205a2f6, 0x6f4715f1, 0x29ed7f70, 0x70915992]);

        let vec0 = PackedMersenne31Neon(arr0);
        let vec1 = PackedMersenne31Neon(arr1);
        let vec_res = vec0 * vec1;

        for i in 0..WIDTH {
            assert_eq!(vec_res.0[i], arr0[i] * arr1[i]);
        }
    }

    #[test]
    fn test_mul_vs_scalar_special_vals_right() {
        let arr0 = array_from_canonical([0x5f8329a7, 0x0f1166bb, 0x657bcb14, 0x0185c34a]);
        let arr1 = [F::ZERO, F::ONE, F::TWO, F::NEG_ONE];

        let vec0 = PackedMersenne31Neon(arr0);
        let vec1 = PackedMersenne31Neon(arr1);
        let vec_res = vec0 * vec1;

        for i in 0..WIDTH {
            assert_eq!(vec_res.0[i], arr0[i] * arr1[i]);
        }
    }

    #[test]
    fn test_neg_vs_scalar() {
        let arr = array_from_canonical([0x1971a7b5, 0x00305be1, 0x52c08410, 0x39cb2586]);

        let vec = PackedMersenne31Neon(arr);
        let vec_res = -vec;

        for i in 0..WIDTH {
            assert_eq!(vec_res.0[i], -arr[i]);
        }
    }

    #[test]
    fn test_neg_vs_scalar_special_vals() {
        let arr = [F::ZERO, F::ONE, F::TWO, F::NEG_ONE];

        let vec = PackedMersenne31Neon(arr);
        let vec_res = -vec;

        for i in 0..WIDTH {
            assert_eq!(vec_res.0[i], -arr[i]);
        }
    }
}
