use core::arch::x86_64::{self, __m256i};
use core::iter::{Product, Sum};
use core::mem::transmute;
use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

use p3_field::{AbstractField, Field, PackedField, PackedValue};
use rand::distributions::{Distribution, Standard};
use rand::Rng;

use crate::KoalaBear;

const WIDTH: usize = 8;
const P: __m256i = unsafe { transmute::<[u32; WIDTH], _>([0x7f000001; WIDTH]) };
const MU: __m256i = unsafe { transmute::<[u32; WIDTH], _>([0x81000001; WIDTH]) };

/// Vectorized AVX2 implementation of `KoalaBear` arithmetic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)] // This needed to make `transmute`s safe.
pub struct PackedKoalaBearAVX2(pub [KoalaBear; WIDTH]);

impl PackedKoalaBearAVX2 {
    #[inline]
    #[must_use]
    /// Get an arch-specific vector representing the packed values.
    fn to_vector(self) -> __m256i {
        unsafe {
            // Safety: `KoalaBear` is `repr(transparent)` so it can be transmuted to `u32`. It
            // follows that `[KoalaBear; WIDTH]` can be transmuted to `[u32; WIDTH]`, which can be
            // transmuted to `__m256i`, since arrays are guaranteed to be contiguous in memory.
            // Finally `PackedKoalaBearAVX2` is `repr(transparent)` so it can be transmuted to
            // `[KoalaBear; WIDTH]`.
            transmute(self)
        }
    }

    #[inline]
    #[must_use]
    /// Make a packed field vector from an arch-specific vector.
    ///
    /// SAFETY: The caller must ensure that each element of `vector` represents a valid `KoalaBear`.
    /// In particular, each element of vector must be in `0..P` (canonical form).
    unsafe fn from_vector(vector: __m256i) -> Self {
        // Safety: It is up to the user to ensure that elements of `vector` represent valid
        // `KoalaBear` values. We must only reason about memory representations. `__m256i` can be
        // transmuted to `[u32; WIDTH]` (since arrays elements are contiguous in memory), which can
        // be transmuted to `[KoalaBear; WIDTH]` (since `KoalaBear` is `repr(transparent)`), which in
        // turn can be transmuted to `PackedKoalaBearAVX2` (since `PackedKoalaBearAVX2` is also
        // `repr(transparent)`).
        transmute(vector)
    }

    /// Copy `value` to all positions in a packed vector. This is the same as
    /// `From<KoalaBear>::from`, but `const`.
    #[inline]
    #[must_use]
    const fn broadcast(value: KoalaBear) -> Self {
        Self([value; WIDTH])
    }
}

impl Add for PackedKoalaBearAVX2 {
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

impl Mul for PackedKoalaBearAVX2 {
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

impl Neg for PackedKoalaBearAVX2 {
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

impl Sub for PackedKoalaBearAVX2 {
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
//            B = 2^32
//            μ = P^-1 mod B
// Input: 0 <= C < P B
// Output: 0 <= R < P such that R = C B^-1 (mod P)
//   1. Q := μ C mod B
//   2. D := (C - Q P) / B
//   3. R := if D < 0 then D + P else D
//
// We first show that the division in step 2. is exact. It suffices to show that C = Q P (mod B). By
// definition of Q and μ, we have Q P = μ C P = P^-1 C P = C (mod B). We also have
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

impl From<KoalaBear> for PackedKoalaBearAVX2 {
    #[inline]
    fn from(value: KoalaBear) -> Self {
        Self::broadcast(value)
    }
}

impl Default for PackedKoalaBearAVX2 {
    #[inline]
    fn default() -> Self {
        KoalaBear::default().into()
    }
}

impl AddAssign for PackedKoalaBearAVX2 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl MulAssign for PackedKoalaBearAVX2 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl SubAssign for PackedKoalaBearAVX2 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Sum for PackedKoalaBearAVX2 {
    #[inline]
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.reduce(|lhs, rhs| lhs + rhs).unwrap_or(Self::zero())
    }
}

impl Product for PackedKoalaBearAVX2 {
    #[inline]
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.reduce(|lhs, rhs| lhs * rhs).unwrap_or(Self::one())
    }
}

impl AbstractField for PackedKoalaBearAVX2 {
    type F = KoalaBear;

    #[inline]
    fn zero() -> Self {
        KoalaBear::zero().into()
    }

    #[inline]
    fn one() -> Self {
        KoalaBear::one().into()
    }

    #[inline]
    fn two() -> Self {
        KoalaBear::two().into()
    }

    #[inline]
    fn neg_one() -> Self {
        KoalaBear::neg_one().into()
    }

    #[inline]
    fn from_f(f: Self::F) -> Self {
        f.into()
    }

    #[inline]
    fn from_bool(b: bool) -> Self {
        KoalaBear::from_bool(b).into()
    }
    #[inline]
    fn from_canonical_u8(n: u8) -> Self {
        KoalaBear::from_canonical_u8(n).into()
    }
    #[inline]
    fn from_canonical_u16(n: u16) -> Self {
        KoalaBear::from_canonical_u16(n).into()
    }
    #[inline]
    fn from_canonical_u32(n: u32) -> Self {
        KoalaBear::from_canonical_u32(n).into()
    }
    #[inline]
    fn from_canonical_u64(n: u64) -> Self {
        KoalaBear::from_canonical_u64(n).into()
    }
    #[inline]
    fn from_canonical_usize(n: usize) -> Self {
        KoalaBear::from_canonical_usize(n).into()
    }

    #[inline]
    fn from_wrapped_u32(n: u32) -> Self {
        KoalaBear::from_wrapped_u32(n).into()
    }
    #[inline]
    fn from_wrapped_u64(n: u64) -> Self {
        KoalaBear::from_wrapped_u64(n).into()
    }

    #[inline]
    fn generator() -> Self {
        KoalaBear::generator().into()
    }
}

impl Add<KoalaBear> for PackedKoalaBearAVX2 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: KoalaBear) -> Self {
        self + Self::from(rhs)
    }
}

impl Mul<KoalaBear> for PackedKoalaBearAVX2 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: KoalaBear) -> Self {
        self * Self::from(rhs)
    }
}

impl Sub<KoalaBear> for PackedKoalaBearAVX2 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: KoalaBear) -> Self {
        self - Self::from(rhs)
    }
}

impl AddAssign<KoalaBear> for PackedKoalaBearAVX2 {
    #[inline]
    fn add_assign(&mut self, rhs: KoalaBear) {
        *self += Self::from(rhs)
    }
}

impl MulAssign<KoalaBear> for PackedKoalaBearAVX2 {
    #[inline]
    fn mul_assign(&mut self, rhs: KoalaBear) {
        *self *= Self::from(rhs)
    }
}

impl SubAssign<KoalaBear> for PackedKoalaBearAVX2 {
    #[inline]
    fn sub_assign(&mut self, rhs: KoalaBear) {
        *self -= Self::from(rhs)
    }
}

impl Sum<KoalaBear> for PackedKoalaBearAVX2 {
    #[inline]
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = KoalaBear>,
    {
        iter.sum::<KoalaBear>().into()
    }
}

impl Product<KoalaBear> for PackedKoalaBearAVX2 {
    #[inline]
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = KoalaBear>,
    {
        iter.product::<KoalaBear>().into()
    }
}

impl Div<KoalaBear> for PackedKoalaBearAVX2 {
    type Output = Self;
    #[allow(clippy::suspicious_arithmetic_impl)]
    #[inline]
    fn div(self, rhs: KoalaBear) -> Self {
        self * rhs.inverse()
    }
}

impl Add<PackedKoalaBearAVX2> for KoalaBear {
    type Output = PackedKoalaBearAVX2;
    #[inline]
    fn add(self, rhs: PackedKoalaBearAVX2) -> PackedKoalaBearAVX2 {
        PackedKoalaBearAVX2::from(self) + rhs
    }
}

impl Mul<PackedKoalaBearAVX2> for KoalaBear {
    type Output = PackedKoalaBearAVX2;
    #[inline]
    fn mul(self, rhs: PackedKoalaBearAVX2) -> PackedKoalaBearAVX2 {
        PackedKoalaBearAVX2::from(self) * rhs
    }
}

impl Sub<PackedKoalaBearAVX2> for KoalaBear {
    type Output = PackedKoalaBearAVX2;
    #[inline]
    fn sub(self, rhs: PackedKoalaBearAVX2) -> PackedKoalaBearAVX2 {
        PackedKoalaBearAVX2::from(self) - rhs
    }
}

impl Distribution<PackedKoalaBearAVX2> for Standard {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> PackedKoalaBearAVX2 {
        PackedKoalaBearAVX2(rng.gen())
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

unsafe impl PackedValue for PackedKoalaBearAVX2 {
    type Value = KoalaBear;

    const WIDTH: usize = WIDTH;

    #[inline]
    fn from_slice(slice: &[KoalaBear]) -> &Self {
        assert_eq!(slice.len(), Self::WIDTH);
        unsafe {
            // Safety: `[KoalaBear; WIDTH]` can be transmuted to `PackedKoalaBearAVX2` since the
            // latter is `repr(transparent)`. They have the same alignment, so the reference cast is
            // safe too.
            &*slice.as_ptr().cast()
        }
    }
    #[inline]
    fn from_slice_mut(slice: &mut [KoalaBear]) -> &mut Self {
        assert_eq!(slice.len(), Self::WIDTH);
        unsafe {
            // Safety: `[KoalaBear; WIDTH]` can be transmuted to `PackedKoalaBearAVX2` since the
            // latter is `repr(transparent)`. They have the same alignment, so the reference cast is
            // safe too.
            &mut *slice.as_mut_ptr().cast()
        }
    }

    /// Similar to `core:array::from_fn`.
    #[inline]
    fn from_fn<F: FnMut(usize) -> KoalaBear>(f: F) -> Self {
        let vals_arr: [_; WIDTH] = core::array::from_fn(f);
        Self(vals_arr)
    }

    #[inline]
    fn as_slice(&self) -> &[KoalaBear] {
        &self.0[..]
    }
    #[inline]
    fn as_slice_mut(&mut self) -> &mut [KoalaBear] {
        &mut self.0[..]
    }
}

unsafe impl PackedField for PackedKoalaBearAVX2 {
    type Scalar = KoalaBear;

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
    use p3_field_testing::test_packed_field;

    use super::{KoalaBear, WIDTH};
    use crate::to_koalabear_array;

    const ZEROS: [KoalaBear; WIDTH] = to_koalabear_array([0x00000000; WIDTH]);

    const SPECIAL_VALS: [KoalaBear; WIDTH] = to_koalabear_array([
        0x00000000, 0x00000001, 0x7f000000, 0x7effffff, 0x3f800000, 0x0ffffffe, 0x68000003,
        0x70000002,
    ]);

    test_packed_field!(
        { super::WIDTH },
        crate::KoalaBear,
        crate::PackedKoalaBearAVX2,
        super::ZEROS,
        super::SPECIAL_VALS
    );
}
