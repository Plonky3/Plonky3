use core::arch::x86_64::{self, __m512i, __mmask16, __mmask8};
use core::iter::{Product, Sum};
use core::mem::transmute;
use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

use p3_field::{AbstractField, Field, PackedField, PackedValue};
use rand::distributions::{Distribution, Standard};
use rand::Rng;

use crate::BabyBear;

const WIDTH: usize = 16;
const P: __m512i = unsafe { transmute::<[u32; WIDTH], _>([0x78000001; WIDTH]) };
// On x86 MONTY_BITS is always 32, so MU = P^-1 (mod 2^32) = 0x88000001.
const MU: __m512i = unsafe { transmute::<[u32; WIDTH], _>([0x88000001; WIDTH]) };
const EVENS: __mmask16 = 0b0101010101010101;
const EVENS4: __mmask16 = 0x0f0f;

/// Vectorized AVX-512F implementation of `BabyBear` arithmetic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)] // This needed to make `transmute`s safe.
pub struct PackedBabyBearAVX512(pub [BabyBear; WIDTH]);

impl PackedBabyBearAVX512 {
    #[inline]
    #[must_use]
    /// Get an arch-specific vector representing the packed values.
    fn to_vector(self) -> __m512i {
        unsafe {
            // Safety: `BabyBear` is `repr(transparent)` so it can be transmuted to `u32`. It
            // follows that `[BabyBear; WIDTH]` can be transmuted to `[u32; WIDTH]`, which can be
            // transmuted to `__m512i`, since arrays are guaranteed to be contiguous in memory.
            // Finally `PackedBabyBearAVX512` is `repr(transparent)` so it can be transmuted to
            // `[BabyBear; WIDTH]`.
            transmute(self)
        }
    }

    #[inline]
    #[must_use]
    /// Make a packed field vector from an arch-specific vector.
    ///
    /// SAFETY: The caller must ensure that each element of `vector` represents a valid
    /// `BabyBear`. In particular, each element of vector must be in `0..=P`.
    unsafe fn from_vector(vector: __m512i) -> Self {
        // Safety: It is up to the user to ensure that elements of `vector` represent valid
        // `BabyBear` values. We must only reason about memory representations. `__m512i` can be
        // transmuted to `[u32; WIDTH]` (since arrays elements are contiguous in memory), which can
        // be transmuted to `[BabyBear; WIDTH]` (since `BabyBear` is `repr(transparent)`), which
        // in turn can be transmuted to `PackedBabyBearAVX512` (since `PackedBabyBearAVX512` is also
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

impl Add for PackedBabyBearAVX512 {
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

impl Mul for PackedBabyBearAVX512 {
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

impl Neg for PackedBabyBearAVX512 {
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

impl Sub for PackedBabyBearAVX512 {
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
fn add(lhs: __m512i, rhs: __m512i) -> __m512i {
    // We want this to compile to:
    //      vpaddd   t, lhs, rhs
    //      vpsubd   u, t, P
    //      vpminud  res, t, u
    // throughput: 1.5 cyc/vec (10.67 els/cyc)
    // latency: 3 cyc

    //   Let t := lhs + rhs. We want to return t mod P. Recall that lhs and rhs are in
    // 0, ..., P - 1, so t is in 0, ..., 2 P - 2 (< 2^32). It suffices to return t if t < P and
    // t - P otherwise.
    //   Let u := (t - P) mod 2^32 and r := unsigned_min(t, u).
    //   If t is in 0, ..., P - 1, then u is in (P - 1 <) 2^32 - P, ..., 2^32 - 1 and r = t.
    // Otherwise, t is in P, ..., 2 P - 2, u is in 0, ..., P - 2 (< P) and r = u. Hence, r is t if
    // t < P and t - P otherwise, as desired.

    unsafe {
        // Safety: If this code got compiled then AVX-512F intrinsics are available.
        let t = x86_64::_mm512_add_epi32(lhs, rhs);
        let u = x86_64::_mm512_sub_epi32(t, P);
        x86_64::_mm512_min_epu32(t, u)
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

/// Viewing the input as a vector of 16 `u32`s, copy the odd elements into the even elements below
/// them. In other words, for all `0 <= i < 8`, set the even elements according to
/// `res[2 * i] := a[2 * i + 1]`, and the odd elements according to
/// `res[2 * i + 1] := a[2 * i + 1]`.
#[inline]
#[must_use]
fn movehdup_epi32(a: __m512i) -> __m512i {
    // The instruction is only available in the floating-point flavor; this distinction is only for
    // historical reasons and no longer matters. We cast to floats, do the thing, and cast back.
    unsafe {
        x86_64::_mm512_castps_si512(x86_64::_mm512_movehdup_ps(x86_64::_mm512_castsi512_ps(a)))
    }
}

/// Viewing `a` as a vector of 16 `u32`s, copy the odd elements into the even elements below them,
/// then merge with `src` according to the mask provided. In other words, for all `0 <= i < 8`, set
/// the even elements according to `res[2 * i] := if k[2 * i] { a[2 * i + 1] } else { src[2 * i] }`,
/// and the odd elements according to
/// `res[2 * i + 1] := if k[2 * i + 1] { a[2 * i + 1] } else { src[2 * i + 1] }`.
#[inline]
#[must_use]
fn mask_movehdup_epi32(src: __m512i, k: __mmask16, a: __m512i) -> __m512i {
    // The instruction is only available in the floating-point flavor; this distinction is only for
    // historical reasons and no longer matters. We cast to floats, do the thing, and cast back.
    unsafe {
        let src = x86_64::_mm512_castsi512_ps(src);
        let a = x86_64::_mm512_castsi512_ps(a);
        x86_64::_mm512_castps_si512(x86_64::_mm512_mask_movehdup_ps(src, k, a))
    }
}

/// Multiply vectors of Baby Bear field elements in canonical form.
/// If the inputs are not in canonical form, the result is undefined.
#[inline]
#[must_use]
#[allow(non_snake_case)]
fn mul(lhs: __m512i, rhs: __m512i) -> __m512i {
    // We want this to compile to:
    //      vmovshdup  lhs_odd, lhs
    //      vmovshdup  rhs_odd, rhs
    //      vpmuludq   prod_evn, lhs, rhs
    //      vpmuludq   prod_hi, lhs_odd, rhs_odd
    //      vpmuludq   q_evn, prod_evn, MU
    //      vpmuludq   q_odd, prod_hi, MU
    //      vmovshdup  prod_hi{EVENS}, prod_evn
    //      vpmuludq   q_P_evn, q_evn, P
    //      vpmuludq   q_P_hi, q_odd, P
    //      vmovshdup  q_P_hi{EVENS}, q_P_evn
    //      vpcmpltud  underflow, prod_hi, q_P_hi
    //      vpsubd     res, prod_hi, q_P_hi
    //      vpaddd     res{underflow}, res, P
    // throughput: 6.5 cyc/vec (2.46 els/cyc)
    // latency: 21 cyc
    unsafe {
        // `vpmuludq` only reads the even doublewords, so when we pass `lhs` and `rhs` directly we
        // get the eight products at even positions.
        let lhs_evn = lhs;
        let rhs_evn = rhs;

        // Copy the odd doublewords into even positions to compute the eight products at odd
        // positions.
        // NB: The odd doublewords are ignored by `vpmuludq`, so we have a lot of choices for how to
        // do this; `vmovshdup` is nice because it runs on a memory port if the operand is in
        // memory, thus improving our throughput.
        let lhs_odd = movehdup_epi32(lhs);
        let rhs_odd = movehdup_epi32(rhs);

        let prod_evn = x86_64::_mm512_mul_epu32(lhs_evn, rhs_evn);
        let prod_odd = x86_64::_mm512_mul_epu32(lhs_odd, rhs_odd);

        let q_evn = x86_64::_mm512_mul_epu32(prod_evn, MU);
        let q_odd = x86_64::_mm512_mul_epu32(prod_odd, MU);

        // Get all the high halves as one vector: this is `(lhs * rhs) >> 32`.
        // NB: `vpermt2d` may feel like a more intuitive choice here, but it has much higher
        // latency.
        let prod_hi = mask_movehdup_epi32(prod_odd, EVENS, prod_evn);

        // Normally we'd want to mask to perform % 2**32, but the instruction below only reads the
        // low 32 bits anyway.
        let q_P_evn = x86_64::_mm512_mul_epu32(q_evn, P);
        let q_P_odd = x86_64::_mm512_mul_epu32(q_odd, P);

        // We can ignore all the low halves of `q_P` as they cancel out. Get all the high halves as
        // one vector.
        let q_P_hi = mask_movehdup_epi32(q_P_odd, EVENS, q_P_evn);

        // Subtraction `prod_hi - q_P_hi` modulo `P`.
        // NB: Normally we'd `vpaddd P` and take the `vpminud`, but `vpminud` runs on port 0, which
        // is already under a lot of pressure performing multiplications. To relieve this pressure,
        // we check for underflow to generate a mask, and then conditionally add `P`. The underflow
        // check runs on port 5, increasing our throughput, although it does cost us an additional
        // cycle of latency.
        let underflow = x86_64::_mm512_cmplt_epu32_mask(prod_hi, q_P_hi);
        let t = x86_64::_mm512_sub_epi32(prod_hi, q_P_hi);
        x86_64::_mm512_mask_add_epi32(t, underflow, t, P)
    }
}

/// Negate a vector of Baby Bear field elements in canonical form.
/// If the inputs are not in canonical form, the result is undefined.
#[inline]
#[must_use]
fn neg(val: __m512i) -> __m512i {
    // We want this to compile to:
    //      vptestmd  nonzero, val, val
    //      vpsubd    res{nonzero}{z}, P, val
    // throughput: 1 cyc/vec (16 els/cyc)
    // latency: 4 cyc

    // NB: This routine prioritizes throughput over latency. An alternative method would be to do
    // sub(0, val), which would result in shorter latency, but also lower throughput.

    //   If val is nonzero, then val is in {1, ..., P - 1} and P - val is in the same range. If val
    // is zero, then the result is zeroed by masking.
    unsafe {
        // Safety: If this code got compiled then AVX-512F intrinsics are available.
        let nonzero = x86_64::_mm512_test_epi32_mask(val, val);
        x86_64::_mm512_maskz_sub_epi32(nonzero, P, val)
    }
}

/// Subtract vectors of Baby Bear field elements in canonical form.
/// If the inputs are not in canonical form, the result is undefined.
#[inline]
#[must_use]
fn sub(lhs: __m512i, rhs: __m512i) -> __m512i {
    // We want this to compile to:
    //      vpsubd   t, lhs, rhs
    //      vpaddd   u, t, P
    //      vpminud  res, t, u
    // throughput: 1.5 cyc/vec (10.67 els/cyc)
    // latency: 3 cyc

    //   Let t := lhs - rhs. We want to return t mod P. Recall that lhs and rhs are in
    // 0, ..., P - 1, so t is in (-2^31 <) -P + 1, ..., P - 1 (< 2^31). It suffices to return t if
    // t >= 0 and t + P otherwise.
    //   Let u := (t + P) mod 2^32 and r := unsigned_min(t, u).
    //   If t is in 0, ..., P - 1, then u is in P, ..., 2 P - 1 and r = t.
    // Otherwise, t is in -P + 1, ..., -1; u is in 1, ..., P - 1 (< P) and r = u. Hence, r is t if
    // t < P and t - P otherwise, as desired.
    unsafe {
        // Safety: If this code got compiled then AVX-512F intrinsics are available.
        let t = x86_64::_mm512_sub_epi32(lhs, rhs);
        let u = x86_64::_mm512_add_epi32(t, P);
        x86_64::_mm512_min_epu32(t, u)
    }
}

impl From<BabyBear> for PackedBabyBearAVX512 {
    #[inline]
    fn from(value: BabyBear) -> Self {
        Self::broadcast(value)
    }
}

impl Default for PackedBabyBearAVX512 {
    #[inline]
    fn default() -> Self {
        BabyBear::default().into()
    }
}

impl AddAssign for PackedBabyBearAVX512 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl MulAssign for PackedBabyBearAVX512 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl SubAssign for PackedBabyBearAVX512 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Sum for PackedBabyBearAVX512 {
    #[inline]
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.reduce(|lhs, rhs| lhs + rhs).unwrap_or(Self::zero())
    }
}

impl Product for PackedBabyBearAVX512 {
    #[inline]
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.reduce(|lhs, rhs| lhs * rhs).unwrap_or(Self::one())
    }
}

impl AbstractField for PackedBabyBearAVX512 {
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

impl Add<BabyBear> for PackedBabyBearAVX512 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: BabyBear) -> Self {
        self + Self::from(rhs)
    }
}

impl Mul<BabyBear> for PackedBabyBearAVX512 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: BabyBear) -> Self {
        self * Self::from(rhs)
    }
}

impl Sub<BabyBear> for PackedBabyBearAVX512 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: BabyBear) -> Self {
        self - Self::from(rhs)
    }
}

impl AddAssign<BabyBear> for PackedBabyBearAVX512 {
    #[inline]
    fn add_assign(&mut self, rhs: BabyBear) {
        *self += Self::from(rhs)
    }
}

impl MulAssign<BabyBear> for PackedBabyBearAVX512 {
    #[inline]
    fn mul_assign(&mut self, rhs: BabyBear) {
        *self *= Self::from(rhs)
    }
}

impl SubAssign<BabyBear> for PackedBabyBearAVX512 {
    #[inline]
    fn sub_assign(&mut self, rhs: BabyBear) {
        *self -= Self::from(rhs)
    }
}

impl Sum<BabyBear> for PackedBabyBearAVX512 {
    #[inline]
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = BabyBear>,
    {
        iter.sum::<BabyBear>().into()
    }
}

impl Product<BabyBear> for PackedBabyBearAVX512 {
    #[inline]
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = BabyBear>,
    {
        iter.product::<BabyBear>().into()
    }
}

impl Div<BabyBear> for PackedBabyBearAVX512 {
    type Output = Self;
    #[allow(clippy::suspicious_arithmetic_impl)]
    #[inline]
    fn div(self, rhs: BabyBear) -> Self {
        self * rhs.inverse()
    }
}

impl Add<PackedBabyBearAVX512> for BabyBear {
    type Output = PackedBabyBearAVX512;
    #[inline]
    fn add(self, rhs: PackedBabyBearAVX512) -> PackedBabyBearAVX512 {
        PackedBabyBearAVX512::from(self) + rhs
    }
}

impl Mul<PackedBabyBearAVX512> for BabyBear {
    type Output = PackedBabyBearAVX512;
    #[inline]
    fn mul(self, rhs: PackedBabyBearAVX512) -> PackedBabyBearAVX512 {
        PackedBabyBearAVX512::from(self) * rhs
    }
}

impl Sub<PackedBabyBearAVX512> for BabyBear {
    type Output = PackedBabyBearAVX512;
    #[inline]
    fn sub(self, rhs: PackedBabyBearAVX512) -> PackedBabyBearAVX512 {
        PackedBabyBearAVX512::from(self) - rhs
    }
}

impl Distribution<PackedBabyBearAVX512> for Standard {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> PackedBabyBearAVX512 {
        PackedBabyBearAVX512(rng.gen())
    }
}

// vpshrdq requires AVX-512VBMI2.
#[cfg(target_feature = "avx512vbmi2")]
#[inline]
#[must_use]
fn interleave1_antidiagonal(x: __m512i, y: __m512i) -> __m512i {
    unsafe {
        // Safety: If this code got compiled then AVX-512VBMI2 intrinsics are available.
        x86_64::_mm512_shrdi_epi64::<32>(y, x)
    }
}

// If we can't use vpshrdq, then do a vpermi2d, but we waste a register and double the latency.
#[cfg(not(target_feature = "avx512vbmi2"))]
#[inline]
#[must_use]
fn interleave1_antidiagonal(x: __m512i, y: __m512i) -> __m512i {
    const INTERLEAVE1_INDICES: __m512i = unsafe {
        // Safety: `[u32; 16]` is trivially transmutable to `__m512i`.
        transmute::<[u32; WIDTH], _>([
            0x01, 0x10, 0x03, 0x12, 0x05, 0x14, 0x07, 0x16, 0x09, 0x18, 0x0b, 0x1a, 0x0d, 0x1c,
            0x0f, 0x1e,
        ])
    };
    unsafe {
        // Safety: If this code got compiled then AVX-512F intrinsics are available.
        x86_64::_mm512_permutex2var_epi32(x, INTERLEAVE1_INDICES, y)
    }
}

#[inline]
#[must_use]
fn interleave1(x: __m512i, y: __m512i) -> (__m512i, __m512i) {
    // If we have AVX-512VBMI2, we want this to compile to:
    //      vpshrdq    t, x, y, 32
    //      vpblendmd  res0 {EVENS}, t, x
    //      vpblendmd  res1 {EVENS}, y, t
    // throughput: 1.5 cyc/2 vec (21.33 els/cyc)
    // latency: 2 cyc
    //
    // Otherwise, we want it to compile to:
    //      vmovdqa32  t, INTERLEAVE1_INDICES
    //      vpermi2d   t, x, y
    //      vpblendmd  res0 {EVENS}, t, x
    //      vpblendmd  res1 {EVENS}, y, t
    // throughput: 1.5 cyc/2 vec (21.33 els/cyc)
    // latency: 4 cyc

    // We currently have:
    //   x = [ x0  x1  x2  x3  x4  x5  x6  x7  x8  x9  xa  xb  xc  xd  xe  xf ],
    //   y = [ y0  y1  y2  y3  y4  y5  y6  y7  y8  y9  ya  yb  yc  yd  ye  yf ].
    // First form
    //   t = [ x1  y0  x3  y2  x5  y4  x7  y6  x9  y8  xb  ya  xd  yc  xf  ye ].
    let t = interleave1_antidiagonal(x, y);

    unsafe {
        // Safety: If this code got compiled then AVX-512F intrinsics are available.

        // Then
        //   res0 = [ x0  y0  x2  y2  x4  y4  x6  y6  x8  y8  xa  ya  xc  yc  xe  ye ],
        //   res1 = [ x1  y1  x3  y3  x5  y5  x7  y7  x9  y9  xb  yb  xd  yd  xf  yf ].
        (
            x86_64::_mm512_mask_blend_epi32(EVENS, t, x),
            x86_64::_mm512_mask_blend_epi32(EVENS, y, t),
        )
    }
}

#[inline]
#[must_use]
fn shuffle_epi64<const MASK: i32>(a: __m512i, b: __m512i) -> __m512i {
    // The instruction is only available in the floating-point flavor; this distinction is only for
    // historical reasons and no longer matters. We cast to floats, do the thing, and cast back.
    unsafe {
        let a = x86_64::_mm512_castsi512_pd(a);
        let b = x86_64::_mm512_castsi512_pd(b);
        x86_64::_mm512_castpd_si512(x86_64::_mm512_shuffle_pd::<MASK>(a, b))
    }
}

#[inline]
#[must_use]
fn interleave2(x: __m512i, y: __m512i) -> (__m512i, __m512i) {
    // We want this to compile to:
    //      vshufpd    t, x, y, 55h
    //      vpblendmq  res0 {EVENS}, t, x
    //      vpblendmq  res1 {EVENS}, y, t
    // throughput: 1.5 cyc/2 vec (21.33 els/cyc)
    // latency: 2 cyc

    unsafe {
        // Safety: If this code got compiled then AVX-512F intrinsics are available.

        // We currently have:
        //   x = [ x0  x1  x2  x3  x4  x5  x6  x7  x8  x9  xa  xb  xc  xd  xe  xf ],
        //   y = [ y0  y1  y2  y3  y4  y5  y6  y7  y8  y9  ya  yb  yc  yd  ye  yf ].
        // First form
        //   t = [ x2  x3  y0  y1  x6  x7  y4  y5  xa  xb  y8  y9  xe  xf  yc  yd ].
        let t = shuffle_epi64::<0b01010101>(x, y);

        // Then
        //   res0 = [ x0  x1  y0  y1  x4  x5  y4  y5  x8  x9  y8  y9  xc  xd  yc  yd ],
        //   res1 = [ x2  x3  y2  y3  x6  x7  y6  y7  xa  xb  ya  yb  xe  xf  ye  yf ].
        (
            x86_64::_mm512_mask_blend_epi64(EVENS as __mmask8, t, x),
            x86_64::_mm512_mask_blend_epi64(EVENS as __mmask8, y, t),
        )
    }
}

#[inline]
#[must_use]
fn interleave4(x: __m512i, y: __m512i) -> (__m512i, __m512i) {
    // We want this to compile to:
    //      vmovdqa64   t, INTERLEAVE4_INDICES
    //      vpermi2q    t, x, y
    //      vpblendmd   res0 {EVENS4}, t, x
    //      vpblendmd   res1 {EVENS4}, y, t
    // throughput: 1.5 cyc/2 vec (21.33 els/cyc)
    // latency: 4 cyc

    const INTERLEAVE4_INDICES: __m512i = unsafe {
        // Safety: `[u64; 8]` is trivially transmutable to `__m512i`.
        transmute::<[u64; WIDTH / 2], _>([0o02, 0o03, 0o10, 0o11, 0o06, 0o07, 0o14, 0o15])
    };

    unsafe {
        // Safety: If this code got compiled then AVX-512F intrinsics are available.

        // We currently have:
        //   x = [ x0  x1  x2  x3  x4  x5  x6  x7  x8  x9  xa  xb  xc  xd  xe  xf ],
        //   y = [ y0  y1  y2  y3  y4  y5  y6  y7  y8  y9  ya  yb  yc  yd  ye  yf ].
        // First form
        //   t = [ x4  x5  x6  x7  y0  y1  y2  y3  xc  xd  xe  xf  y8  y9  ya  yb ].
        let t = x86_64::_mm512_permutex2var_epi64(x, INTERLEAVE4_INDICES, y);

        // Then
        //   res0 = [ x0  x1  x2  x3  y0  y1  y2  y3  x8  x9  xa  xb  y8  y9  ya  yb ],
        //   res1 = [ x4  x5  x6  x7  y4  y5  y6  y7  xc  xd  xe  xf  yc  yd  ye  yf ].
        (
            x86_64::_mm512_mask_blend_epi32(EVENS4, t, x),
            x86_64::_mm512_mask_blend_epi32(EVENS4, y, t),
        )
    }
}

#[inline]
#[must_use]
fn interleave8(x: __m512i, y: __m512i) -> (__m512i, __m512i) {
    // We want this to compile to:
    //      vshufi64x2  t, x, b, 4eh
    //      vpblendmq   res0 {EVENS4}, t, x
    //      vpblendmq   res1 {EVENS4}, y, t
    // throughput: 1.5 cyc/2 vec (21.33 els/cyc)
    // latency: 4 cyc

    unsafe {
        // Safety: If this code got compiled then AVX-512F intrinsics are available.

        // We currently have:
        //   x = [ x0  x1  x2  x3  x4  x5  x6  x7  x8  x9  xa  xb  xc  xd  xe  xf ],
        //   y = [ y0  y1  y2  y3  y4  y5  y6  y7  y8  y9  ya  yb  yc  yd  ye  yf ].
        // First form
        //   t = [ x8  x9  xa  xb  xc  xd  xe  xf  y0  y1  y2  y3  y4  y5  y6  y7 ].
        let t = x86_64::_mm512_shuffle_i64x2::<0b01_00_11_10>(x, y);

        // Then
        //   res0 = [ x0  x1  x2  x3  x4  x5  x6  x7  y0  y1  y2  y3  y4  y5  y6  y7 ],
        //   res1 = [ x8  x9  xa  xb  xc  xd  xe  xf  y8  y9  ya  yb  yc  yd  ye  yf ].
        (
            x86_64::_mm512_mask_blend_epi64(EVENS4 as __mmask8, t, x),
            x86_64::_mm512_mask_blend_epi64(EVENS4 as __mmask8, y, t),
        )
    }
}

unsafe impl PackedValue for PackedBabyBearAVX512 {
    type Value = BabyBear;

    const WIDTH: usize = WIDTH;

    #[inline]
    fn from_slice(slice: &[BabyBear]) -> &Self {
        assert_eq!(slice.len(), Self::WIDTH);
        unsafe {
            // Safety: `[BabyBear; WIDTH]` can be transmuted to `PackedBabyBearAVX512` since the
            // latter is `repr(transparent)`. They have the same alignment, so the reference cast is
            // safe too.
            &*slice.as_ptr().cast()
        }
    }
    #[inline]
    fn from_slice_mut(slice: &mut [BabyBear]) -> &mut Self {
        assert_eq!(slice.len(), Self::WIDTH);
        unsafe {
            // Safety: `[BabyBear; WIDTH]` can be transmuted to `PackedBabyBearAVX512` since the
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
}

unsafe impl PackedField for PackedBabyBearAVX512 {
    type Scalar = BabyBear;

    #[inline]
    fn interleave(&self, other: Self, block_len: usize) -> (Self, Self) {
        let (v0, v1) = (self.to_vector(), other.to_vector());
        let (res0, res1) = match block_len {
            1 => interleave1(v0, v1),
            2 => interleave2(v0, v1),
            4 => interleave4(v0, v1),
            8 => interleave8(v0, v1),
            16 => (v0, v1),
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
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    use super::*;

    type F = BabyBear;
    type P = PackedBabyBearAVX512;

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
        PackedBabyBearAVX512(array_from_valid_reps(vals))
    }

    fn array_from_random(seed: u64) -> [F; WIDTH] {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        [(); WIDTH].map(|_| rng.gen())
    }

    fn packed_from_random(seed: u64) -> P {
        PackedBabyBearAVX512(array_from_random(seed))
    }

    const SPECIAL_VALS: [F; WIDTH] = array_from_valid_reps([
        0x00000000, 0x00000001, 0x78000000, 0x77ffffff, 0x3c000000, 0x0ffffffe, 0x68000003,
        0x70000002, 0x00000000, 0x00000001, 0x78000000, 0x77ffffff, 0x3c000000, 0x0ffffffe,
        0x68000003, 0x70000002,
    ]);

    #[test]
    fn test_interleave_1() {
        let vec0 = packed_from_valid_reps([
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d,
            0x0e, 0x0f,
        ]);
        let vec1 = packed_from_valid_reps([
            0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d,
            0x1e, 0x1f,
        ]);

        let expected0 = packed_from_valid_reps([
            0x00, 0x10, 0x02, 0x12, 0x04, 0x14, 0x06, 0x16, 0x08, 0x18, 0x0a, 0x1a, 0x0c, 0x1c,
            0x0e, 0x1e,
        ]);
        let expected1 = packed_from_valid_reps([
            0x01, 0x11, 0x03, 0x13, 0x05, 0x15, 0x07, 0x17, 0x09, 0x19, 0x0b, 0x1b, 0x0d, 0x1d,
            0x0f, 0x1f,
        ]);

        let (res0, res1) = vec0.interleave(vec1, 1);
        assert_eq!(res0, expected0);
        assert_eq!(res1, expected1);
    }

    #[test]
    fn test_interleave_2() {
        let vec0 = packed_from_valid_reps([
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d,
            0x0e, 0x0f,
        ]);
        let vec1 = packed_from_valid_reps([
            0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d,
            0x1e, 0x1f,
        ]);

        let expected0 = packed_from_valid_reps([
            0x00, 0x01, 0x10, 0x11, 0x04, 0x05, 0x14, 0x15, 0x08, 0x09, 0x18, 0x19, 0x0c, 0x0d,
            0x1c, 0x1d,
        ]);
        let expected1 = packed_from_valid_reps([
            0x02, 0x03, 0x12, 0x13, 0x06, 0x07, 0x16, 0x17, 0x0a, 0x0b, 0x1a, 0x1b, 0x0e, 0x0f,
            0x1e, 0x1f,
        ]);

        let (res0, res1) = vec0.interleave(vec1, 2);
        assert_eq!(res0, expected0);
        assert_eq!(res1, expected1);
    }

    #[test]
    fn test_interleave_4() {
        let vec0 = packed_from_valid_reps([
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d,
            0x0e, 0x0f,
        ]);
        let vec1 = packed_from_valid_reps([
            0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d,
            0x1e, 0x1f,
        ]);

        let expected0 = packed_from_valid_reps([
            0x00, 0x01, 0x02, 0x03, 0x10, 0x11, 0x12, 0x13, 0x08, 0x09, 0x0a, 0x0b, 0x18, 0x19,
            0x1a, 0x1b,
        ]);
        let expected1 = packed_from_valid_reps([
            0x04, 0x05, 0x06, 0x07, 0x14, 0x15, 0x16, 0x17, 0x0c, 0x0d, 0x0e, 0x0f, 0x1c, 0x1d,
            0x1e, 0x1f,
        ]);

        let (res0, res1) = vec0.interleave(vec1, 4);
        assert_eq!(res0, expected0);
        assert_eq!(res1, expected1);
    }

    #[test]
    fn test_interleave_8() {
        let vec0 = packed_from_valid_reps([
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d,
            0x0e, 0x0f,
        ]);
        let vec1 = packed_from_valid_reps([
            0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d,
            0x1e, 0x1f,
        ]);

        let expected0 = packed_from_valid_reps([
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15,
            0x16, 0x17,
        ]);
        let expected1 = packed_from_valid_reps([
            0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d,
            0x1e, 0x1f,
        ]);

        let (res0, res1) = vec0.interleave(vec1, 8);
        assert_eq!(res0, expected0);
        assert_eq!(res1, expected1);
    }

    #[test]
    fn test_interleave_16() {
        let vec0 = packed_from_valid_reps([
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d,
            0x0e, 0x0f,
        ]);
        let vec1 = packed_from_valid_reps([
            0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d,
            0x1e, 0x1f,
        ]);

        let (res0, res1) = vec0.interleave(vec1, 16);
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

        let vec = PackedBabyBearAVX512(arr);
        let vec_inv = PackedBabyBearAVX512(arr_inv);

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

        let vec0 = PackedBabyBearAVX512(arr0);
        let vec1 = PackedBabyBearAVX512(arr1);
        let vec_res = vec0 + vec1;

        for i in 0..WIDTH {
            assert_eq!(vec_res.0[i], arr0[i] + arr1[i]);
        }
    }

    #[test]
    fn test_add_vs_scalar_special_vals_left() {
        let arr0 = SPECIAL_VALS;
        let arr1 = array_from_random(0x1e2b153f07b64cf3);

        let vec0 = PackedBabyBearAVX512(arr0);
        let vec1 = PackedBabyBearAVX512(arr1);
        let vec_res = vec0 + vec1;

        for i in 0..WIDTH {
            assert_eq!(vec_res.0[i], arr0[i] + arr1[i]);
        }
    }

    #[test]
    fn test_add_vs_scalar_special_vals_right() {
        let arr0 = array_from_random(0xfcf974ac7625a260);
        let arr1 = SPECIAL_VALS;

        let vec0 = PackedBabyBearAVX512(arr0);
        let vec1 = PackedBabyBearAVX512(arr1);
        let vec_res = vec0 + vec1;

        for i in 0..WIDTH {
            assert_eq!(vec_res.0[i], arr0[i] + arr1[i]);
        }
    }

    #[test]
    fn test_sub_vs_scalar() {
        let arr0 = array_from_random(0x167ce9d8e920876e);
        let arr1 = array_from_random(0x52ddcdd3461e046f);

        let vec0 = PackedBabyBearAVX512(arr0);
        let vec1 = PackedBabyBearAVX512(arr1);
        let vec_res = vec0 - vec1;

        for i in 0..WIDTH {
            assert_eq!(vec_res.0[i], arr0[i] - arr1[i]);
        }
    }

    #[test]
    fn test_sub_vs_scalar_special_vals_left() {
        let arr0 = SPECIAL_VALS;
        let arr1 = array_from_random(0x358498640bfe1375);

        let vec0 = PackedBabyBearAVX512(arr0);
        let vec1 = PackedBabyBearAVX512(arr1);
        let vec_res = vec0 - vec1;

        for i in 0..WIDTH {
            assert_eq!(vec_res.0[i], arr0[i] - arr1[i]);
        }
    }

    #[test]
    fn test_sub_vs_scalar_special_vals_right() {
        let arr0 = array_from_random(0x05d81ebfb8f0005c);
        let arr1 = SPECIAL_VALS;

        let vec0 = PackedBabyBearAVX512(arr0);
        let vec1 = PackedBabyBearAVX512(arr1);
        let vec_res = vec0 - vec1;

        for i in 0..WIDTH {
            assert_eq!(vec_res.0[i], arr0[i] - arr1[i]);
        }
    }

    #[test]
    fn test_mul_vs_scalar() {
        let arr0 = array_from_random(0x4242ebdc09b74d77);
        let arr1 = array_from_random(0x9937b275b3c056cd);

        let vec0 = PackedBabyBearAVX512(arr0);
        let vec1 = PackedBabyBearAVX512(arr1);
        let vec_res = vec0 * vec1;

        for i in 0..WIDTH {
            assert_eq!(vec_res.0[i], arr0[i] * arr1[i]);
        }
    }

    #[test]
    fn test_mul_vs_scalar_special_vals_left() {
        let arr0 = SPECIAL_VALS;
        let arr1 = array_from_random(0x5285448b835458a3);

        let vec0 = PackedBabyBearAVX512(arr0);
        let vec1 = PackedBabyBearAVX512(arr1);
        let vec_res = vec0 * vec1;

        for i in 0..WIDTH {
            assert_eq!(vec_res.0[i], arr0[i] * arr1[i]);
        }
    }

    #[test]
    fn test_mul_vs_scalar_special_vals_right() {
        let arr0 = array_from_random(0x22508dc80001d865);
        let arr1 = SPECIAL_VALS;

        let vec0 = PackedBabyBearAVX512(arr0);
        let vec1 = PackedBabyBearAVX512(arr1);
        let vec_res = vec0 * vec1;

        for i in 0..WIDTH {
            assert_eq!(vec_res.0[i], arr0[i] * arr1[i]);
        }
    }

    #[test]
    fn test_neg_vs_scalar() {
        let arr = array_from_random(0xc3c273a9b334372f);

        let vec = PackedBabyBearAVX512(arr);
        let vec_res = -vec;

        for i in 0..WIDTH {
            assert_eq!(vec_res.0[i], -arr[i]);
        }
    }

    #[test]
    fn test_neg_vs_scalar_special_vals() {
        let arr = SPECIAL_VALS;

        let vec = PackedBabyBearAVX512(arr);
        let vec_res = -vec;

        for i in 0..WIDTH {
            assert_eq!(vec_res.0[i], -arr[i]);
        }
    }
}
