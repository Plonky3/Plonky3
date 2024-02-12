use core::arch::x86_64::{self, __m256i};
use core::iter::{Product, Sum};
use core::mem::transmute;
use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};
use rand::distributions::{Distribution, Standard};
use rand::Rng;

use p3_field::{AbstractField, Field, PackedField};

use crate::BabyBear;

const WIDTH: usize = 8;
const P: __m256i = unsafe { transmute::<[u32; WIDTH], _>([0x78000001; WIDTH]) };
// On x86 MONTY_BITS is always 32, so MU = P^-1 (mod 2^32) = 0x88000001.
const MU: __m256i = unsafe { transmute::<[u32; WIDTH], _>([0x88000001; WIDTH]) };

/// Vectorized AVX2 implementation of `BabyBear` arithmetic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)] // This needed to make `transmute`s safe.
pub struct PackedBabyBearAVX2(pub [BabyBear; WIDTH]);

impl PackedBabyBearAVX2 {
    #[inline]
    #[must_use]
    /// Get an arch-specific vector representing the packed values.
    fn to_vector(self) -> __m256i {
        unsafe {
            // Safety: `BabyBear` is `repr(transparent)` so it can be transmuted to `u32`. It
            // follows that `[BabyBear; WIDTH]` can be transmuted to `[u32; WIDTH]`, which can be
            // transmuted to `__m256i`, since arrays are guaranteed to be contiguous in memory.
            // Finally `PackedBabyBearAVX2` is `repr(transparent)` so it can be transmuted to
            // `[BabyBear; WIDTH]`.
            transmute(self)
        }
    }

    #[inline]
    #[must_use]
    /// Make a packed field vector from an arch-specific vector.
    ///
    /// SAFETY: The caller must ensure that each element of `vector` represents a valid `BabyBear`.
    /// In particular, each element of vector must be in `0..P` (canonical form).
    unsafe fn from_vector(vector: __m256i) -> Self {
        // Safety: It is up to the user to ensure that elements of `vector` represent valid
        // `BabyBear` values. We must only reason about memory representations. `__m256i` can be
        // transmuted to `[u32; WIDTH]` (since arrays elements are contiguous in memory), which can
        // be transmuted to `[BabyBear; WIDTH]` (since `BabyBear` is `repr(transparent)`), which in
        // turn can be transmuted to `PackedBabyBearAVX2` (since `PackedBabyBearAVX2` is also
        // `repr(transparent)`).
        transmute(vector)
    }

    /// Copy `value` to all positions in a packed vector. This is the same as
    /// `From<BabyBear>::from`, but `const`.
    #[inline]
    #[must_use]
    const fn broadcast(value: BabyBear) -> Self {
        Self([value; WIDTH])
    }
}

impl Add for PackedBabyBearAVX2 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        let lhs = self.to_vector();
        let rhs = rhs.to_vector();
        let res = add(lhs, rhs);
        unsafe {
            // Safety: `add` returns values in canonical form when given values in canonical form.
            Self::from_vector(res)
        }
    }
}

impl Mul for PackedBabyBearAVX2 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let lhs = self.to_vector();
        let rhs = rhs.to_vector();
        let res = mul(lhs, rhs);
        unsafe {
            // Safety: `mul` returns values in canonical form when given values in canonical form.
            Self::from_vector(res)
        }
    }
}

impl Neg for PackedBabyBearAVX2 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        let val = self.to_vector();
        let res = neg(val);
        unsafe {
            // Safety: `neg` returns values in canonical form when given values in canonical form.
            Self::from_vector(res)
        }
    }
}

impl Sub for PackedBabyBearAVX2 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        let lhs = self.to_vector();
        let rhs = rhs.to_vector();
        let res = sub(lhs, rhs);
        unsafe {
            // Safety: `sub` returns values in canonical form when given values in canonical form.
            Self::from_vector(res)
        }
    }
}

/// Add two vectors of Baby Bear field elements in canonical form.
/// If the inputs are not in canonical form, the result is undefined.
#[inline]
#[must_use]
fn add(lhs: __m256i, rhs: __m256i) -> __m256i {
    // We want this to compile to:
    //      vpaddd   t, lhs, rhs
    //      vpsubd   u, t, P
    //      vpminud  res, t, u
    // throughput: 1 cyc/vec (8 els/cyc)
    // latency: 3 cyc

    //   Let t := lhs + rhs. We want to return t mod P. Recall that lhs and rhs are in
    // 0, ..., P - 1, so t is in 0, ..., 2 P - 2 (< 2^32). It suffices to return t if t < P and
    // t - P otherwise.
    //   Let u := (t - P) mod 2^32 and r := unsigned_min(t, u).
    //   If t is in 0, ..., P - 1, then u is in (P - 1 <) 2^32 - P, ..., 2^32 - 1 and r = t.
    // Otherwise, t is in P, ..., 2 P - 2, u is in 0, ..., P - 2 (< P) and r = u. Hence, r is t if
    // t < P and t - P otherwise, as desired.

    unsafe {
        // Safety: If this code got compiled then AVX2 intrinsics are available.
        let t = x86_64::_mm256_add_epi32(lhs, rhs);
        let u = x86_64::_mm256_sub_epi32(t, P);
        x86_64::_mm256_min_epu32(t, u)
    }
}

// MONTGOMERY MULTIPLICATION
//   This implementation is based on [1] but with minor changes. The reduction is as follows:
//
// Constants: P = 2^31 - 2^27 + 1
//            B = 2^31
//            mu = P^-1 mod B
// Input: 0 <= C < P B
// Output: 0 <= R < P such that R = C B^-1 (mod P)
//   1. Q := mu C mod B
//   2. D := (C - Q P) / B
//   3. R := if D < 0 then D + P else D
//
// We first show that the division in step 2. is exact. It suffices to show that C = Q P (mod B). By
// definition of Q and mu, we have Q P = mu C P = P^-1 C P = C (mod B). We also have
// C - Q P = C (mod P), so thus D = C B^-1 (mod P).
//
// It remains to show that R is in the correct range. It suffices to show that -P <= D < P. We know
// that 0 <= C < P B and 0 <= Q P < P B. Then -P B < C - QP < P B and -P < D < P, as desired.
//
// [1] Modern Computer Arithmetic, Richard Brent and Paul Zimmermann, Cambridge University Press,
//     2010, algorithm 2.7.

#[inline]
#[must_use]
#[allow(non_snake_case)]
fn monty_d(lhs: __m256i, rhs: __m256i) -> __m256i {
    unsafe {
        let prod = x86_64::_mm256_mul_epu32(lhs, rhs);
        let q = x86_64::_mm256_mul_epu32(prod, MU);
        let q_P = x86_64::_mm256_mul_epu32(q, P);
        x86_64::_mm256_sub_epi64(prod, q_P)
    }
}

#[inline]
#[must_use]
fn movehdup_epi32(x: __m256i) -> __m256i {
    // This instruction is only available in the floating-point flavor; this distinction is only for
    // historical reasons and no longer matters. We cast to floats, duplicate, and cast back.
    unsafe {
        x86_64::_mm256_castps_si256(x86_64::_mm256_movehdup_ps(x86_64::_mm256_castsi256_ps(x)))
    }
}

/// Multiply vectors of Baby Bear field elements in canonical form.
/// If the inputs are not in canonical form, the result is undefined.
#[inline]
#[must_use]
fn mul(lhs: __m256i, rhs: __m256i) -> __m256i {
    // We want this to compile to:
    //      vmovshdup  lhs_odd, lhs
    //      vmovshdup  rhs_odd, rhs
    //      vpmuludq   prod_evn, lhs, rhs
    //      vpmuludq   prod_odd, lhs_odd, rhs_odd
    //      vpmuludq   q_evn, prod_evn, MU
    //      vpmuludq   q_odd, prod_odd, MU
    //      vpmuludq   q_P_evn, q_evn, P
    //      vpmuludq   q_P_odd, q_odd, P
    //      vpsubq     d_evn, prod_evn, q_P_evn
    //      vpsubq     d_odd, prod_odd, q_P_odd
    //      vmovshdup  d_evn_hi, d_evn
    //      vpblendd   t, d_evn_hi, d_odd, aah
    //      vpaddd     u, t, P
    //      vpminud    res, t, u
    // throughput: 4.67 cyc/vec (1.71 els/cyc)
    // latency: 21 cyc
    unsafe {
        let lhs_evn = lhs;
        let rhs_evn = rhs;
        let lhs_odd = movehdup_epi32(lhs);
        let rhs_odd = movehdup_epi32(rhs);

        let d_evn = monty_d(lhs_evn, rhs_evn);
        let d_odd = monty_d(lhs_odd, rhs_odd);

        let d_evn_hi = movehdup_epi32(d_evn);
        let t = x86_64::_mm256_blend_epi32::<0b10101010>(d_evn_hi, d_odd);

        let u = x86_64::_mm256_add_epi32(t, P);
        x86_64::_mm256_min_epu32(t, u)
    }
}

/// Negate a vector of Baby Bear field elements in canonical form.
/// If the inputs are not in canonical form, the result is undefined.
#[inline]
#[must_use]
fn neg(val: __m256i) -> __m256i {
    // We want this to compile to:
    //      vpsubd   t, P, val
    //      vpsignd  res, t, val
    // throughput: .67 cyc/vec (12 els/cyc)
    // latency: 2 cyc

    //   The vpsignd instruction is poorly named, because it doesn't _return_ or _copy_ the sign of
    // anything, but _multiplies_ x by the sign of y (treating both as signed integers). In other
    // words,
    //                       { x            if y >s 0,
    //      vpsignd(x, y) := { 0            if y = 0,
    //                       { -x mod 2^32  if y <s 0.
    //   We define t := P - val and note that t = -val (mod P). When val is in {1, ..., P - 1}, t is
    // similarly in {1, ..., P - 1}, so it's in canonical form. Otherwise, val = 0 and t = P.
    //   This is where we define res := vpsignd(t, val). The sign bit of val is never set so either
    // val = 0 or val >s 0. If val = 0, then res = vpsignd(t, 0) = 0, as desired. Otherwise,
    // res = vpsignd(t, val) = t passes t through.
    unsafe {
        // Safety: If this code got compiled then AVX2 intrinsics are available.
        let t = x86_64::_mm256_sub_epi32(P, val);
        x86_64::_mm256_sign_epi32(t, val)
    }
}

/// Subtract vectors of Baby Bear field elements in canonical form.
/// If the inputs are not in canonical form, the result is undefined.
#[inline]
#[must_use]
fn sub(lhs: __m256i, rhs: __m256i) -> __m256i {
    // We want this to compile to:
    //      vpsubd   t, lhs, rhs
    //      vpaddd   u, t, P
    //      vpminud  res, t, u
    // throughput: 1 cyc/vec (8 els/cyc)
    // latency: 3 cyc

    //   Let t := lhs - rhs. We want to return t mod P. Recall that lhs and rhs are in
    // 0, ..., P - 1, so t is in (-2^31 <) -P + 1, ..., P - 1 (< 2^31). It suffices to return t if
    // t >= 0 and t + P otherwise.
    //   Let u := (t + P) mod 2^32 and r := unsigned_min(t, u).
    //   If t is in 0, ..., P - 1, then u is in P, ..., 2 P - 1 and r = t.
    // Otherwise, t is in -P + 1, ..., -1; u is in 1, ..., P - 1 (< P) and r = u. Hence, r is t if
    // t < P and t - P otherwise, as desired.
    unsafe {
        // Safety: If this code got compiled then AVX2 intrinsics are available.
        let t = x86_64::_mm256_sub_epi32(lhs, rhs);
        let u = x86_64::_mm256_add_epi32(t, P);
        x86_64::_mm256_min_epu32(t, u)
    }
}

impl From<BabyBear> for PackedBabyBearAVX2 {
    #[inline]
    fn from(value: BabyBear) -> Self {
        Self::broadcast(value)
    }
}

impl Default for PackedBabyBearAVX2 {
    #[inline]
    fn default() -> Self {
        BabyBear::default().into()
    }
}

impl AddAssign for PackedBabyBearAVX2 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl MulAssign for PackedBabyBearAVX2 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl SubAssign for PackedBabyBearAVX2 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Sum for PackedBabyBearAVX2 {
    #[inline]
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.reduce(|lhs, rhs| lhs + rhs).unwrap_or(Self::zero())
    }
}

impl Product for PackedBabyBearAVX2 {
    #[inline]
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.reduce(|lhs, rhs| lhs * rhs).unwrap_or(Self::one())
    }
}

impl AbstractField for PackedBabyBearAVX2 {
    type F = BabyBear;

    #[inline]
    fn zero() -> Self {
        BabyBear::zero().into()
    }

    #[inline]
    fn one() -> Self {
        BabyBear::one().into()
    }

    #[inline]
    fn two() -> Self {
        BabyBear::two().into()
    }

    #[inline]
    fn neg_one() -> Self {
        BabyBear::neg_one().into()
    }

    #[inline]
    fn from_f(f: Self::F) -> Self {
        f.into()
    }

    #[inline]
    fn from_bool(b: bool) -> Self {
        BabyBear::from_bool(b).into()
    }
    #[inline]
    fn from_canonical_u8(n: u8) -> Self {
        BabyBear::from_canonical_u8(n).into()
    }
    #[inline]
    fn from_canonical_u16(n: u16) -> Self {
        BabyBear::from_canonical_u16(n).into()
    }
    #[inline]
    fn from_canonical_u32(n: u32) -> Self {
        BabyBear::from_canonical_u32(n).into()
    }
    #[inline]
    fn from_canonical_u64(n: u64) -> Self {
        BabyBear::from_canonical_u64(n).into()
    }
    #[inline]
    fn from_canonical_usize(n: usize) -> Self {
        BabyBear::from_canonical_usize(n).into()
    }

    #[inline]
    fn from_wrapped_u32(n: u32) -> Self {
        BabyBear::from_wrapped_u32(n).into()
    }
    #[inline]
    fn from_wrapped_u64(n: u64) -> Self {
        BabyBear::from_wrapped_u64(n).into()
    }

    #[inline]
    fn generator() -> Self {
        BabyBear::generator().into()
    }
}

impl Add<BabyBear> for PackedBabyBearAVX2 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: BabyBear) -> Self {
        self + Self::from(rhs)
    }
}

impl Mul<BabyBear> for PackedBabyBearAVX2 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: BabyBear) -> Self {
        self * Self::from(rhs)
    }
}

impl Sub<BabyBear> for PackedBabyBearAVX2 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: BabyBear) -> Self {
        self - Self::from(rhs)
    }
}

impl AddAssign<BabyBear> for PackedBabyBearAVX2 {
    #[inline]
    fn add_assign(&mut self, rhs: BabyBear) {
        *self += Self::from(rhs)
    }
}

impl MulAssign<BabyBear> for PackedBabyBearAVX2 {
    #[inline]
    fn mul_assign(&mut self, rhs: BabyBear) {
        *self *= Self::from(rhs)
    }
}

impl SubAssign<BabyBear> for PackedBabyBearAVX2 {
    #[inline]
    fn sub_assign(&mut self, rhs: BabyBear) {
        *self -= Self::from(rhs)
    }
}

impl Sum<BabyBear> for PackedBabyBearAVX2 {
    #[inline]
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = BabyBear>,
    {
        iter.sum::<BabyBear>().into()
    }
}

impl Product<BabyBear> for PackedBabyBearAVX2 {
    #[inline]
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = BabyBear>,
    {
        iter.product::<BabyBear>().into()
    }
}

impl Div<BabyBear> for PackedBabyBearAVX2 {
    type Output = Self;
    #[allow(clippy::suspicious_arithmetic_impl)]
    #[inline]
    fn div(self, rhs: BabyBear) -> Self {
        self * rhs.inverse()
    }
}

impl Add<PackedBabyBearAVX2> for BabyBear {
    type Output = PackedBabyBearAVX2;
    #[inline]
    fn add(self, rhs: PackedBabyBearAVX2) -> PackedBabyBearAVX2 {
        PackedBabyBearAVX2::from(self) + rhs
    }
}

impl Mul<PackedBabyBearAVX2> for BabyBear {
    type Output = PackedBabyBearAVX2;
    #[inline]
    fn mul(self, rhs: PackedBabyBearAVX2) -> PackedBabyBearAVX2 {
        PackedBabyBearAVX2::from(self) * rhs
    }
}

impl Sub<PackedBabyBearAVX2> for BabyBear {
    type Output = PackedBabyBearAVX2;
    #[inline]
    fn sub(self, rhs: PackedBabyBearAVX2) -> PackedBabyBearAVX2 {
        PackedBabyBearAVX2::from(self) - rhs
    }
}

impl Distribution<PackedBabyBearAVX2> for Standard {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> PackedBabyBearAVX2 {
        PackedBabyBearAVX2(rng.gen())
    }
}

#[inline]
#[must_use]
fn interleave1(a: __m256i, b: __m256i) -> (__m256i, __m256i) {
    // We want this to compile to:
    //      vpsllq    t, a, 32
    //      vpsrlq    u, b, 32
    //      vpblendd  res0, a, u, aah
    //      vpblendd  res1, t, b, aah
    // throughput: 1.33 cyc/2 vec (12 els/cyc)
    // latency: (1 -> 1)  1 cyc
    //          (1 -> 2)  2 cyc
    //          (2 -> 1)  2 cyc
    //          (2 -> 2)  1 cyc
    unsafe {
        // Safety: If this code got compiled then AVX2 intrinsics are available.

        // We currently have:
        //   a = [ a0  a1  a2  a3  a4  a5  a6  a7 ],
        //   b = [ b0  b1  b2  b3  b4  b5  b6  b7 ].
        // First form
        //   t = [ a1   0  a3   0  a5   0  a7   0 ].
        //   u = [  0  b0   0  b2   0  b4   0  b6 ].
        let t = x86_64::_mm256_srli_epi64::<32>(a);
        let u = x86_64::_mm256_slli_epi64::<32>(b);

        // Then
        //   res0 = [ a0  b0  a2  b2  a4  b4  a6  b6 ],
        //   res1 = [ a1  b1  a3  b3  a5  b5  a7  b7 ].
        (
            x86_64::_mm256_blend_epi32::<0b10101010>(a, u),
            x86_64::_mm256_blend_epi32::<0b10101010>(t, b),
        )
    }
}

#[inline]
#[must_use]
fn interleave2(a: __m256i, b: __m256i) -> (__m256i, __m256i) {
    // We want this to compile to:
    //      vpalignr  t, b, a, 8
    //      vpblendd  res0, a, t, cch
    //      vpblendd  res1, t, b, cch
    // throughput: 1 cyc/2 vec (16 els/cyc)
    // latency: 2 cyc

    unsafe {
        // Safety: If this code got compiled then AVX2 intrinsics are available.

        // We currently have:
        //   a = [ a0  a1  a2  a3  a4  a5  a6  a7 ],
        //   b = [ b0  b1  b2  b3  b4  b5  b6  b7 ].
        // First form
        //   t = [ a2  a3  b0  b1  a6  a7  b4  b5 ].
        let t = x86_64::_mm256_alignr_epi8::<8>(b, a);

        // Then
        //   res0 = [ a0  a1  b0  b1  a4  a5  b4  b5 ],
        //   res1 = [ a2  a3  b2  b3  a6  a7  b6  b7 ].
        (
            x86_64::_mm256_blend_epi32::<0b11001100>(a, t),
            x86_64::_mm256_blend_epi32::<0b11001100>(t, b),
        )
    }
}

#[inline]
#[must_use]
fn interleave4(a: __m256i, b: __m256i) -> (__m256i, __m256i) {
    // We want this to compile to:
    //      vperm2i128  t, a, b, 21h
    //      vpblendd    res0, a, t, f0h
    //      vpblendd    res1, t, b, f0h
    // throughput: 1 cyc/2 vec (16 els/cyc)
    // latency: 4 cyc

    unsafe {
        // Safety: If this code got compiled then AVX2 intrinsics are available.

        // We currently have:
        //   a = [ a0  a1  a2  a3  a4  a5  a6  a7 ],
        //   b = [ b0  b1  b2  b3  b4  b5  b6  b7 ].
        // First form
        //   t = [ a4  a5  a6  a7  b0  b1  b2  b3 ].
        let t = x86_64::_mm256_permute2x128_si256::<0x21>(a, b);

        // Then
        //   res0 = [ a0  a1  a2  a3  b0  b1  b2  b3 ],
        //   res1 = [ a4  a5  a6  a7  b4  b5  b6  b7 ].
        (
            x86_64::_mm256_blend_epi32::<0b11110000>(a, t),
            x86_64::_mm256_blend_epi32::<0b11110000>(t, b),
        )
    }
}

unsafe impl PackedField for PackedBabyBearAVX2 {
    type Scalar = BabyBear;

    const WIDTH: usize = WIDTH;

    #[inline]
    fn from_slice(slice: &[BabyBear]) -> &Self {
        assert_eq!(slice.len(), Self::WIDTH);
        unsafe {
            // Safety: `[BabyBear; WIDTH]` can be transmuted to `PackedBabyBearAVX2` since the
            // latter is `repr(transparent)`. They have the same alignment, so the reference cast is
            // safe too.
            &*slice.as_ptr().cast()
        }
    }
    #[inline]
    fn from_slice_mut(slice: &mut [BabyBear]) -> &mut Self {
        assert_eq!(slice.len(), Self::WIDTH);
        unsafe {
            // Safety: `[BabyBear; WIDTH]` can be transmuted to `PackedBabyBearAVX2` since the
            // latter is `repr(transparent)`. They have the same alignment, so the reference cast is
            // safe too.
            &mut *slice.as_mut_ptr().cast()
        }
    }

    /// Similar to `core:array::from_fn`.
    #[inline]
    fn from_fn<F: FnMut(usize) -> BabyBear>(f: F) -> Self {
        let vals_arr: [_; WIDTH] = core::array::from_fn(f);
        Self(vals_arr)
    }

    #[inline]
    fn as_slice(&self) -> &[BabyBear] {
        &self.0[..]
    }
    #[inline]
    fn as_slice_mut(&mut self) -> &mut [BabyBear] {
        &mut self.0[..]
    }

    #[inline]
    fn interleave(&self, other: Self, block_len: usize) -> (Self, Self) {
        let (v0, v1) = (self.to_vector(), other.to_vector());
        let (res0, res1) = match block_len {
            1 => interleave1(v0, v1),
            2 => interleave2(v0, v1),
            4 => interleave4(v0, v1),
            8 => (v0, v1),
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
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha20Rng;

    use super::*;

    type F = BabyBear;
    type P = PackedBabyBearAVX2;

    const fn array_from_valid_reps(vals: [u32; WIDTH]) -> [F; WIDTH] {
        let mut res = [BabyBear { value: 0 }; WIDTH];
        let mut i = 0;
        while i < WIDTH {
            res[i] = BabyBear { value: vals[i] };
            i += 1;
        }
        res
    }

    const fn packed_from_valid_reps(vals: [u32; WIDTH]) -> P {
        PackedBabyBearAVX2(array_from_valid_reps(vals))
    }

    fn array_from_random(seed: u64) -> [F; WIDTH] {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        [(); WIDTH].map(|_| rng.gen())
    }

    fn packed_from_random(seed: u64) -> P {
        PackedBabyBearAVX2(array_from_random(seed))
    }

    const SPECIAL_VALS: [F; WIDTH] = array_from_valid_reps([
        0x00000000, 0x00000001, 0x78000000, 0x77ffffff, 0x3c000000, 0x0ffffffe, 0x68000003,
        0x70000002,
    ]);

    #[test]
    fn test_interleave_1() {
        let vec0 = packed_from_valid_reps([0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7]);
        let vec1 = packed_from_valid_reps([0x8, 0x9, 0xa, 0xb, 0xc, 0xd, 0xe, 0xf]);

        let expected0 = packed_from_valid_reps([0x0, 0x8, 0x2, 0xa, 0x4, 0xc, 0x6, 0xe]);
        let expected1 = packed_from_valid_reps([0x1, 0x9, 0x3, 0xb, 0x5, 0xd, 0x7, 0xf]);

        let (res0, res1) = vec0.interleave(vec1, 1);
        assert_eq!(res0, expected0);
        assert_eq!(res1, expected1);
    }

    #[test]
    fn test_interleave_2() {
        let vec0 = packed_from_valid_reps([0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7]);
        let vec1 = packed_from_valid_reps([0x8, 0x9, 0xa, 0xb, 0xc, 0xd, 0xe, 0xf]);

        let expected0 = packed_from_valid_reps([0x0, 0x1, 0x8, 0x9, 0x4, 0x5, 0xc, 0xd]);
        let expected1 = packed_from_valid_reps([0x2, 0x3, 0xa, 0xb, 0x6, 0x7, 0xe, 0xf]);

        let (res0, res1) = vec0.interleave(vec1, 2);
        assert_eq!(res0, expected0);
        assert_eq!(res1, expected1);
    }

    #[test]
    fn test_interleave_4() {
        let vec0 = packed_from_valid_reps([0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7]);
        let vec1 = packed_from_valid_reps([0x8, 0x9, 0xa, 0xb, 0xc, 0xd, 0xe, 0xf]);

        let expected0 = packed_from_valid_reps([0x0, 0x1, 0x2, 0x3, 0x8, 0x9, 0xa, 0xb]);
        let expected1 = packed_from_valid_reps([0x4, 0x5, 0x6, 0x7, 0xc, 0xd, 0xe, 0xf]);

        let (res0, res1) = vec0.interleave(vec1, 4);
        assert_eq!(res0, expected0);
        assert_eq!(res1, expected1);
    }

    #[test]
    fn test_interleave_8() {
        let vec0 = packed_from_valid_reps([0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7]);
        let vec1 = packed_from_valid_reps([0x8, 0x9, 0xa, 0xb, 0xc, 0xd, 0xe, 0xf]);

        let (res0, res1) = vec0.interleave(vec1, 8);
        assert_eq!(res0, vec0);
        assert_eq!(res1, vec1);
    }

    #[test]
    fn test_add_associative() {
        let vec0 = packed_from_random(0x8b078c2b693c893f);
        let vec1 = packed_from_random(0x4ff5dec04791e481);
        let vec2 = packed_from_random(0x5806c495e9451f8e);

        let res0 = (vec0 + vec1) + vec2;
        let res1 = vec0 + (vec1 + vec2);

        assert_eq!(res0, res1);
    }

    #[test]
    fn test_add_commutative() {
        let vec0 = packed_from_random(0xe1bf9cac02e9072a);
        let vec1 = packed_from_random(0xb5061e7de6a6c677);

        let res0 = vec0 + vec1;
        let res1 = vec1 + vec0;

        assert_eq!(res0, res1);
    }

    #[test]
    fn test_additive_identity_right() {
        let vec = packed_from_random(0xbcd56facf6a714b5);
        let res = vec + P::zero();
        assert_eq!(res, vec);
    }

    #[test]
    fn test_additive_identity_left() {
        let vec = packed_from_random(0xb614285cd641233c);
        let res = P::zero() + vec;
        assert_eq!(res, vec);
    }

    #[test]
    fn test_additive_inverse_add_neg() {
        let vec = packed_from_random(0x4b89c8d023c9c62e);
        let neg_vec = -vec;
        let res = vec + neg_vec;
        assert_eq!(res, P::zero());
    }

    #[test]
    fn test_additive_inverse_sub() {
        let vec = packed_from_random(0x2c94652ee5561341);
        let res = vec - vec;
        assert_eq!(res, P::zero());
    }

    #[test]
    fn test_sub_anticommutative() {
        let vec0 = packed_from_random(0xf3783730a14b460e);
        let vec1 = packed_from_random(0x5b6f827a023525ee);

        let res0 = vec0 - vec1;
        let res1 = -(vec1 - vec0);

        assert_eq!(res0, res1);
    }

    #[test]
    fn test_sub_zero() {
        let vec = packed_from_random(0xc1a526f8226ec1e5);
        let res = vec - P::zero();
        assert_eq!(res, vec);
    }

    #[test]
    fn test_zero_sub() {
        let vec = packed_from_random(0x4444b9c090519333);
        let res0 = P::zero() - vec;
        let res1 = -vec;
        assert_eq!(res0, res1);
    }

    #[test]
    fn test_neg_own_inverse() {
        let vec = packed_from_random(0xee4df174b850a35f);
        let res = --vec;
        assert_eq!(res, vec);
    }

    #[test]
    fn test_sub_is_add_neg() {
        let vec0 = packed_from_random(0x18f4b5c3a08e49fe);
        let vec1 = packed_from_random(0x39bd37a1dc24d492);
        let res0 = vec0 - vec1;
        let res1 = vec0 + (-vec1);
        assert_eq!(res0, res1);
    }

    #[test]
    fn test_mul_associative() {
        let vec0 = packed_from_random(0x0b1ee4d7c979d50c);
        let vec1 = packed_from_random(0x39faa0844a36e45a);
        let vec2 = packed_from_random(0x08fac4ee76260e44);

        let res0 = (vec0 * vec1) * vec2;
        let res1 = vec0 * (vec1 * vec2);

        assert_eq!(res0, res1);
    }

    #[test]
    fn test_mul_commutative() {
        let vec0 = packed_from_random(0x10debdcbd409270c);
        let vec1 = packed_from_random(0x927bc073c1c92b2f);

        let res0 = vec0 * vec1;
        let res1 = vec1 * vec0;

        assert_eq!(res0, res1);
    }

    #[test]
    fn test_multiplicative_identity_right() {
        let vec = packed_from_random(0xdf0a646b6b0c2c36);
        let res = vec * P::one();
        assert_eq!(res, vec);
    }

    #[test]
    fn test_multiplicative_identity_left() {
        let vec = packed_from_random(0x7b4d890bf7a38bd2);
        let res = P::one() * vec;
        assert_eq!(res, vec);
    }

    #[test]
    fn test_multiplicative_inverse() {
        let arr = array_from_random(0xb0c7a5153103c5a8);
        let arr_inv = arr.map(|x| x.inverse());

        let vec = PackedBabyBearAVX2(arr);
        let vec_inv = PackedBabyBearAVX2(arr_inv);

        let res = vec * vec_inv;
        assert_eq!(res, P::one());
    }

    #[test]
    fn test_mul_zero() {
        let vec = packed_from_random(0x7f998daa72489bd7);
        let res = vec * P::zero();
        assert_eq!(res, P::zero());
    }

    #[test]
    fn test_zero_mul() {
        let vec = packed_from_random(0x683bc2dd355b06e5);
        let res = P::zero() * vec;
        assert_eq!(res, P::zero());
    }

    #[test]
    fn test_mul_negone() {
        let vec = packed_from_random(0x97cb9670a8251202);
        let res0 = vec * P::neg_one();
        let res1 = -vec;
        assert_eq!(res0, res1);
    }

    #[test]
    fn test_negone_mul() {
        let vec = packed_from_random(0xadae69873b5d3baf);
        let res0 = P::neg_one() * vec;
        let res1 = -vec;
        assert_eq!(res0, res1);
    }

    #[test]
    fn test_neg_distributivity_left() {
        let vec0 = packed_from_random(0xd0efd6f272c7de93);
        let vec1 = packed_from_random(0xd5dd2cf5e76dd694);

        let res0 = vec0 * -vec1;
        let res1 = -(vec0 * vec1);

        assert_eq!(res0, res1);
    }

    #[test]
    fn test_neg_distributivity_right() {
        let vec0 = packed_from_random(0x0da9b03cd4b79b09);
        let vec1 = packed_from_random(0x9964d3f4beaf1857);

        let res0 = -vec0 * vec1;
        let res1 = -(vec0 * vec1);

        assert_eq!(res0, res1);
    }

    #[test]
    fn test_add_distributivity_left() {
        let vec0 = packed_from_random(0x278d9e202925a1d1);
        let vec1 = packed_from_random(0xf04cbac0cbad419f);
        let vec2 = packed_from_random(0x76976e2abdc5a056);

        let res0 = vec0 * (vec1 + vec2);
        let res1 = vec0 * vec1 + vec0 * vec2;

        assert_eq!(res0, res1);
    }

    #[test]
    fn test_add_distributivity_right() {
        let vec0 = packed_from_random(0xbe1b606eafe2a2b8);
        let vec1 = packed_from_random(0x552686a0978ab571);
        let vec2 = packed_from_random(0x36f6eec4fd31a460);

        let res0 = (vec0 + vec1) * vec2;
        let res1 = vec0 * vec2 + vec1 * vec2;

        assert_eq!(res0, res1);
    }

    #[test]
    fn test_sub_distributivity_left() {
        let vec0 = packed_from_random(0x817d4a27febb0349);
        let vec1 = packed_from_random(0x1eaf62a921d6519b);
        let vec2 = packed_from_random(0xfec0fb8d3849465a);

        let res0 = vec0 * (vec1 - vec2);
        let res1 = vec0 * vec1 - vec0 * vec2;

        assert_eq!(res0, res1);
    }

    #[test]
    fn test_sub_distributivity_right() {
        let vec0 = packed_from_random(0x5a4a82e8e2394585);
        let vec1 = packed_from_random(0x6006b1443a22b102);
        let vec2 = packed_from_random(0x5a22deac65fcd454);

        let res0 = (vec0 - vec1) * vec2;
        let res1 = vec0 * vec2 - vec1 * vec2;

        assert_eq!(res0, res1);
    }

    #[test]
    fn test_one_plus_one() {
        assert_eq!(P::one() + P::one(), P::two());
    }

    #[test]
    fn test_negone_plus_two() {
        assert_eq!(P::neg_one() + P::two(), P::one());
    }

    #[test]
    fn test_double() {
        let vec = packed_from_random(0x2e61a907650881e9);
        let res0 = P::two() * vec;
        let res1 = vec + vec;
        assert_eq!(res0, res1);
    }

    #[test]
    fn test_add_vs_scalar() {
        let arr0 = array_from_random(0xac23b5a694dabf70);
        let arr1 = array_from_random(0xd249ec90e8a6e733);

        let vec0 = PackedBabyBearAVX2(arr0);
        let vec1 = PackedBabyBearAVX2(arr1);
        let vec_res = vec0 + vec1;

        for i in 0..WIDTH {
            assert_eq!(vec_res.0[i], arr0[i] + arr1[i]);
        }
    }

    #[test]
    fn test_add_vs_scalar_special_vals_left() {
        let arr0 = SPECIAL_VALS;
        let arr1 = array_from_random(0x1e2b153f07b64cf3);

        let vec0 = PackedBabyBearAVX2(arr0);
        let vec1 = PackedBabyBearAVX2(arr1);
        let vec_res = vec0 + vec1;

        for i in 0..WIDTH {
            assert_eq!(vec_res.0[i], arr0[i] + arr1[i]);
        }
    }

    #[test]
    fn test_add_vs_scalar_special_vals_right() {
        let arr0 = array_from_random(0xfcf974ac7625a260);
        let arr1 = SPECIAL_VALS;

        let vec0 = PackedBabyBearAVX2(arr0);
        let vec1 = PackedBabyBearAVX2(arr1);
        let vec_res = vec0 + vec1;

        for i in 0..WIDTH {
            assert_eq!(vec_res.0[i], arr0[i] + arr1[i]);
        }
    }

    #[test]
    fn test_sub_vs_scalar() {
        let arr0 = array_from_random(0x167ce9d8e920876e);
        let arr1 = array_from_random(0x52ddcdd3461e046f);

        let vec0 = PackedBabyBearAVX2(arr0);
        let vec1 = PackedBabyBearAVX2(arr1);
        let vec_res = vec0 - vec1;

        for i in 0..WIDTH {
            assert_eq!(vec_res.0[i], arr0[i] - arr1[i]);
        }
    }

    #[test]
    fn test_sub_vs_scalar_special_vals_left() {
        let arr0 = SPECIAL_VALS;
        let arr1 = array_from_random(0x358498640bfe1375);

        let vec0 = PackedBabyBearAVX2(arr0);
        let vec1 = PackedBabyBearAVX2(arr1);
        let vec_res = vec0 - vec1;

        for i in 0..WIDTH {
            assert_eq!(vec_res.0[i], arr0[i] - arr1[i]);
        }
    }

    #[test]
    fn test_sub_vs_scalar_special_vals_right() {
        let arr0 = array_from_random(0x05d81ebfb8f0005c);
        let arr1 = SPECIAL_VALS;

        let vec0 = PackedBabyBearAVX2(arr0);
        let vec1 = PackedBabyBearAVX2(arr1);
        let vec_res = vec0 - vec1;

        for i in 0..WIDTH {
            assert_eq!(vec_res.0[i], arr0[i] - arr1[i]);
        }
    }

    #[test]
    fn test_mul_vs_scalar() {
        let arr0 = array_from_random(0x4242ebdc09b74d77);
        let arr1 = array_from_random(0x9937b275b3c056cd);

        let vec0 = PackedBabyBearAVX2(arr0);
        let vec1 = PackedBabyBearAVX2(arr1);
        let vec_res = vec0 * vec1;

        for i in 0..WIDTH {
            assert_eq!(vec_res.0[i], arr0[i] * arr1[i]);
        }
    }

    #[test]
    fn test_mul_vs_scalar_special_vals_left() {
        let arr0 = SPECIAL_VALS;
        let arr1 = array_from_random(0x5285448b835458a3);

        let vec0 = PackedBabyBearAVX2(arr0);
        let vec1 = PackedBabyBearAVX2(arr1);
        let vec_res = vec0 * vec1;

        for i in 0..WIDTH {
            assert_eq!(vec_res.0[i], arr0[i] * arr1[i]);
        }
    }

    #[test]
    fn test_mul_vs_scalar_special_vals_right() {
        let arr0 = array_from_random(0x22508dc80001d865);
        let arr1 = SPECIAL_VALS;

        let vec0 = PackedBabyBearAVX2(arr0);
        let vec1 = PackedBabyBearAVX2(arr1);
        let vec_res = vec0 * vec1;

        for i in 0..WIDTH {
            assert_eq!(vec_res.0[i], arr0[i] * arr1[i]);
        }
    }

    #[test]
    fn test_neg_vs_scalar() {
        let arr = array_from_random(0xc3c273a9b334372f);

        let vec = PackedBabyBearAVX2(arr);
        let vec_res = -vec;

        for i in 0..WIDTH {
            assert_eq!(vec_res.0[i], -arr[i]);
        }
    }

    #[test]
    fn test_neg_vs_scalar_special_vals() {
        let arr = SPECIAL_VALS;

        let vec = PackedBabyBearAVX2(arr);
        let vec_res = -vec;

        for i in 0..WIDTH {
            assert_eq!(vec_res.0[i], -arr[i]);
        }
    }
}
