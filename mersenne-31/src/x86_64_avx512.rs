use core::arch::x86_64::{self, __m512i, __mmask16, __mmask8};
use core::iter::{Product, Sum};
use core::mem::transmute;
use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

use p3_field::{AbstractField, Field, PackedField, PackedValue};

use crate::Mersenne31;

const WIDTH: usize = 16;
const P: __m512i = unsafe { transmute::<[u32; WIDTH], _>([0x7fffffff; WIDTH]) };
const EVENS: __mmask16 = 0b0101010101010101;
const ODDS: __mmask16 = 0b1010101010101010;
const EVENS4: __mmask16 = 0x0f0f;

/// Vectorized AVX-512F implementation of `Mersenne31` arithmetic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)] // This needed to make `transmute`s safe.
pub struct PackedMersenne31AVX512(pub [Mersenne31; WIDTH]);

impl PackedMersenne31AVX512 {
    #[inline]
    #[must_use]
    /// Get an arch-specific vector representing the packed values.
    fn to_vector(self) -> __m512i {
        unsafe {
            // Safety: `Mersenne31` is `repr(transparent)` so it can be transmuted to `u32`. It
            // follows that `[Mersenne31; WIDTH]` can be transmuted to `[u32; WIDTH]`, which can be
            // transmuted to `__m512i`, since arrays are guaranteed to be contiguous in memory.
            // Finally `PackedMersenne31AVX512` is `repr(transparent)` so it can be transmuted to
            // `[Mersenne31; WIDTH]`.
            transmute(self)
        }
    }

    #[inline]
    #[must_use]
    /// Make a packed field vector from an arch-specific vector.
    ///
    /// SAFETY: The caller must ensure that each element of `vector` represents a valid
    /// `Mersenne31`. In particular, each element of vector must be in `0..=P`.
    unsafe fn from_vector(vector: __m512i) -> Self {
        // Safety: It is up to the user to ensure that elements of `vector` represent valid
        // `Mersenne31` values. We must only reason about memory representations. `__m512i` can be
        // transmuted to `[u32; WIDTH]` (since arrays elements are contiguous in memory), which can
        // be transmuted to `[Mersenne31; WIDTH]` (since `Mersenne31` is `repr(transparent)`), which
        // in turn can be transmuted to `PackedMersenne31AVX512` (since `PackedMersenne31AVX512` is also
        // `repr(transparent)`).
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

impl Add for PackedMersenne31AVX512 {
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

impl Mul for PackedMersenne31AVX512 {
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

impl Neg for PackedMersenne31AVX512 {
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

impl Sub for PackedMersenne31AVX512 {
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

/// Add two vectors of Mersenne-31 field elements represented as values in {0, ..., P}.
/// If the inputs do not conform to this representation, the result is undefined.
#[inline]
#[must_use]
fn add(lhs: __m512i, rhs: __m512i) -> __m512i {
    // We want this to compile to:
    //      vpaddd   t, lhs, rhs
    //      vpsubd   u, t, P
    //      vpminud  res, t, u
    // throughput: 1.5 cyc/vec (10.67 els/cyc)
    // latency: 3 cyc

    //   Let t := lhs + rhs. We want to return a value r in {0, ..., P} such that r = t (mod P).
    //   Define u := (t - P) mod 2^32 and r := min(t, u). t is in {0, ..., 2 P}. We argue by cases.
    //   If t is in {0, ..., P - 1}, then u is in {(P - 1 <) 2^32 - P, ..., 2^32 - 1}, so r = t is
    // in the correct range.
    //   If t is in {P, ..., 2 P}, then u is in {0, ..., P} and r = u is in the correct range.
    unsafe {
        // Safety: If this code got compiled then AVX-512F intrinsics are available.
        let t = x86_64::_mm512_add_epi32(lhs, rhs);
        let u = x86_64::_mm512_sub_epi32(t, P);
        x86_64::_mm512_min_epu32(t, u)
    }
}

#[inline]
#[must_use]
fn movehdup_epi32(a: __m512i) -> __m512i {
    // The instruction is only available in the floating-point flavor; this distinction is only for
    // historical reasons and no longer matters. We cast to floats, do the thing, and cast back.
    unsafe {
        x86_64::_mm512_castps_si512(x86_64::_mm512_movehdup_ps(x86_64::_mm512_castsi512_ps(a)))
    }
}

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

#[inline]
#[must_use]
fn mask_moveldup_epi32(src: __m512i, k: __mmask16, a: __m512i) -> __m512i {
    // The instruction is only available in the floating-point flavor; this distinction is only for
    // historical reasons and no longer matters. We cast to floats, do the thing, and cast back.
    unsafe {
        let src = x86_64::_mm512_castsi512_ps(src);
        let a = x86_64::_mm512_castsi512_ps(a);
        x86_64::_mm512_castps_si512(x86_64::_mm512_mask_moveldup_ps(src, k, a))
    }
}

/// Multiply vectors of Mersenne-31 field elements represented as values in {0, ..., P}.
/// If the inputs do not conform to this representation, the result is undefined.
#[inline]
#[must_use]
fn mul(lhs: __m512i, rhs: __m512i) -> __m512i {
    // We want this to compile to:
    // vpaddd     lhs_evn_dbl, lhs, lhs
    // vmovshdup  rhs_odd, rhs
    // vpsrlq     lhs_odd_dbl, lhs, 31
    // vpmuludq   prod_lo_dbl, lhs_evn_dbl, rhs
    // vpmuludq   prod_odd_dbl, lhs_odd_dbl, rhs_odd
    // vmovdqa32  prod_hi, prod_odd_dbl
    // vmovshdup  prod_hi{EVENS}, prod_lo_dbl
    // vmovsldup  prod_lo_dbl{ODDS}, prod_odd_dbl
    // vpsrld     prod_lo, prod_lo_dbl, 1
    // vpaddd     t, prod_lo, prod_hi
    // vpsubd     u, t, P
    // vpminud    res, t, u
    // throughput: 5.5 cyc/vec (2.91 els/cyc)
    // latency: (lhs->res) 15 cyc, (rhs->res) 14 cyc
    unsafe {
        // vpmuludq only reads the bottom 32 bits of every 64-bit quadword.
        // The even indices are already in the bottom 32 bits of a quadword, so we can leave them.
        let rhs_evn = rhs;
        // Again, vpmuludq only reads the bottom 32 bits so we don't need to clear the top. But we
        // do want to double the lhs.
        let lhs_evn_dbl = x86_64::_mm512_add_epi32(lhs, lhs);
        // Copy the high 32 bits in each quadword of rhs down to the low 32.
        let rhs_odd = movehdup_epi32(rhs);
        // Right shift by 31 is equivalent to moving the high 32 bits down to the low 32, and then
        // doubling it. So these are the odd indices in lhs, but doubled.
        let lhs_odd_dbl = x86_64::_mm512_srli_epi64::<31>(lhs);

        // Multiply odd indices; since lhs_odd_dbl is doubled, these products are also doubled.
        // prod_odd_dbl.quadword[i] = 2 * lhs.doubleword[2 * i + 1] * rhs.doubleword[2 * i + 1]
        let prod_odd_dbl = x86_64::_mm512_mul_epu32(lhs_odd_dbl, rhs_odd);
        // Multiply even indices; these are also doubled.
        // prod_evn_dbl.quadword[i] = 2 * lhs.doubleword[2 * i] * rhs.doubleword[2 * i]
        let prod_evn_dbl = x86_64::_mm512_mul_epu32(lhs_evn_dbl, rhs_evn);

        // Move the low halves of odd products into odd positions; keep the low halves of even
        // products in even positions (where they already are). Note that the products are doubled,
        // so the result is a vector of all the low halves, but doubled.
        let prod_lo_dbl = mask_moveldup_epi32(prod_evn_dbl, ODDS, prod_odd_dbl);
        // Move the high halves of even products into even positions, keeping the high halves of odd
        // products where they are. The products are doubled, but we are looking at (prod >> 32),
        // which cancels out the doubling, so this result is _not_ doubled.
        let prod_hi = mask_movehdup_epi32(prod_odd_dbl, EVENS, prod_evn_dbl);
        // Right shift to undo the doubling.
        let prod_lo = x86_64::_mm512_srli_epi32::<1>(prod_lo_dbl);

        // Standard addition of two 31-bit values.
        add(prod_lo, prod_hi)
    }
}

/// Negate a vector of Mersenne-31 field elements represented as values in {0, ..., P}.
/// If the input does not conform to this representation, the result is undefined.
#[inline]
#[must_use]
fn neg(val: __m512i) -> __m512i {
    // We want this to compile to:
    //      vpxord  res, val, P
    // throughput: .5 cyc/vec (32 els/cyc)
    // latency: 1 cyc

    //   Since val is in {0, ..., P (= 2^31 - 1)}, res = val XOR P = P - val. Then res is in {0,
    // ..., P}.
    unsafe {
        // Safety: If this code got compiled then AVX-512F intrinsics are available.
        x86_64::_mm512_xor_epi32(val, P)
    }
}

/// Subtract vectors of Mersenne-31 field elements represented as values in {0, ..., P}.
/// If the inputs do not conform to this representation, the result is undefined.
#[inline]
#[must_use]
fn sub(lhs: __m512i, rhs: __m512i) -> __m512i {
    // We want this to compile to:
    //      vpsubd   t, lhs, rhs
    //      vpaddd   u, t, P
    //      vpminud  res, t, u
    // throughput: 1.5 cyc/vec (10.67 els/cyc)
    // latency: 3 cyc

    //   Let d := lhs - rhs and t := d mod 2^32. We want to return a value r in {0, ..., P} such
    // that r = d (mod P).
    //   Define u := (t + P) mod 2^32 and r := min(t, u). d is in {-P, ..., P}. We argue by cases.
    //   If d is in {0, ..., P}, then t = d and u is in {P, ..., 2 P}. r = t is in the correct
    // range.
    //   If d is in {-P, ..., -1}, then t is in {2^32 - P, ..., 2^32 - 1} and u is in
    // {0, ..., P - 1}. r = u is in the correct range.
    unsafe {
        // Safety: If this code got compiled then AVX-512F intrinsics are available.
        let t = x86_64::_mm512_sub_epi32(lhs, rhs);
        let u = x86_64::_mm512_add_epi32(t, P);
        x86_64::_mm512_min_epu32(t, u)
    }
}

impl From<Mersenne31> for PackedMersenne31AVX512 {
    #[inline]
    fn from(value: Mersenne31) -> Self {
        Self::broadcast(value)
    }
}

impl Default for PackedMersenne31AVX512 {
    #[inline]
    fn default() -> Self {
        Mersenne31::default().into()
    }
}

impl AddAssign for PackedMersenne31AVX512 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl MulAssign for PackedMersenne31AVX512 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl SubAssign for PackedMersenne31AVX512 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Sum for PackedMersenne31AVX512 {
    #[inline]
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.reduce(|lhs, rhs| lhs + rhs).unwrap_or(Self::zero())
    }
}

impl Product for PackedMersenne31AVX512 {
    #[inline]
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.reduce(|lhs, rhs| lhs * rhs).unwrap_or(Self::one())
    }
}

impl AbstractField for PackedMersenne31AVX512 {
    type F = Mersenne31;

    #[inline]
    fn zero() -> Self {
        Mersenne31::zero().into()
    }

    #[inline]
    fn one() -> Self {
        Mersenne31::one().into()
    }

    #[inline]
    fn two() -> Self {
        Mersenne31::two().into()
    }

    #[inline]
    fn neg_one() -> Self {
        Mersenne31::neg_one().into()
    }

    #[inline]
    fn from_f(f: Self::F) -> Self {
        f.into()
    }

    #[inline]
    fn from_bool(b: bool) -> Self {
        Mersenne31::from_bool(b).into()
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

    #[inline]
    fn generator() -> Self {
        Mersenne31::generator().into()
    }
}

impl Add<Mersenne31> for PackedMersenne31AVX512 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Mersenne31) -> Self {
        self + Self::from(rhs)
    }
}

impl Mul<Mersenne31> for PackedMersenne31AVX512 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Mersenne31) -> Self {
        self * Self::from(rhs)
    }
}

impl Sub<Mersenne31> for PackedMersenne31AVX512 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Mersenne31) -> Self {
        self - Self::from(rhs)
    }
}

impl AddAssign<Mersenne31> for PackedMersenne31AVX512 {
    #[inline]
    fn add_assign(&mut self, rhs: Mersenne31) {
        *self += Self::from(rhs)
    }
}

impl MulAssign<Mersenne31> for PackedMersenne31AVX512 {
    #[inline]
    fn mul_assign(&mut self, rhs: Mersenne31) {
        *self *= Self::from(rhs)
    }
}

impl SubAssign<Mersenne31> for PackedMersenne31AVX512 {
    #[inline]
    fn sub_assign(&mut self, rhs: Mersenne31) {
        *self -= Self::from(rhs)
    }
}

impl Sum<Mersenne31> for PackedMersenne31AVX512 {
    #[inline]
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = Mersenne31>,
    {
        iter.sum::<Mersenne31>().into()
    }
}

impl Product<Mersenne31> for PackedMersenne31AVX512 {
    #[inline]
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = Mersenne31>,
    {
        iter.product::<Mersenne31>().into()
    }
}

impl Div<Mersenne31> for PackedMersenne31AVX512 {
    type Output = Self;
    #[allow(clippy::suspicious_arithmetic_impl)]
    #[inline]
    fn div(self, rhs: Mersenne31) -> Self {
        self * rhs.inverse()
    }
}

impl Add<PackedMersenne31AVX512> for Mersenne31 {
    type Output = PackedMersenne31AVX512;
    #[inline]
    fn add(self, rhs: PackedMersenne31AVX512) -> PackedMersenne31AVX512 {
        PackedMersenne31AVX512::from(self) + rhs
    }
}

impl Mul<PackedMersenne31AVX512> for Mersenne31 {
    type Output = PackedMersenne31AVX512;
    #[inline]
    fn mul(self, rhs: PackedMersenne31AVX512) -> PackedMersenne31AVX512 {
        PackedMersenne31AVX512::from(self) * rhs
    }
}

impl Sub<PackedMersenne31AVX512> for Mersenne31 {
    type Output = PackedMersenne31AVX512;
    #[inline]
    fn sub(self, rhs: PackedMersenne31AVX512) -> PackedMersenne31AVX512 {
        PackedMersenne31AVX512::from(self) - rhs
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

unsafe impl PackedValue for PackedMersenne31AVX512 {
    type Value = Mersenne31;

    const WIDTH: usize = WIDTH;

    #[inline]
    fn from_slice(slice: &[Mersenne31]) -> &Self {
        assert_eq!(slice.len(), Self::WIDTH);
        unsafe {
            // Safety: `[Mersenne31; WIDTH]` can be transmuted to `PackedMersenne31AVX512` since the
            // latter is `repr(transparent)`. They have the same alignment, so the reference cast is
            // safe too.
            &*slice.as_ptr().cast()
        }
    }
    #[inline]
    fn from_slice_mut(slice: &mut [Mersenne31]) -> &mut Self {
        assert_eq!(slice.len(), Self::WIDTH);
        unsafe {
            // Safety: `[Mersenne31; WIDTH]` can be transmuted to `PackedMersenne31AVX512` since the
            // latter is `repr(transparent)`. They have the same alignment, so the reference cast is
            // safe too.
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

unsafe impl PackedField for PackedMersenne31AVX512 {
    type Scalar = Mersenne31;

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
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha20Rng;

    use super::*;

    type F = Mersenne31;
    type P = PackedMersenne31AVX512;

    const fn array_from_valid_reps(vals: [u32; WIDTH]) -> [F; WIDTH] {
        let mut res = [Mersenne31 { value: 0 }; WIDTH];
        let mut i = 0;
        while i < WIDTH {
            res[i] = Mersenne31 { value: vals[i] };
            i += 1;
        }
        res
    }

    const fn packed_from_valid_reps(vals: [u32; WIDTH]) -> P {
        PackedMersenne31AVX512(array_from_valid_reps(vals))
    }

    fn array_from_random(seed: u64) -> [F; WIDTH] {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        [(); WIDTH].map(|_| rng.gen())
    }

    fn packed_from_random(seed: u64) -> P {
        PackedMersenne31AVX512(array_from_random(seed))
    }

    /// Zero has a redundant representation, so let's test both.
    const BOTH_ZEROS: P = packed_from_valid_reps([
        0x00000000, 0x7fffffff, 0x00000000, 0x7fffffff, 0x00000000, 0x7fffffff, 0x00000000,
        0x7fffffff, 0x00000000, 0x7fffffff, 0x00000000, 0x7fffffff, 0x00000000, 0x7fffffff,
        0x00000000, 0x7fffffff,
    ]);

    const SPECIAL_VALS: [F; WIDTH] = array_from_valid_reps([
        0x00000000, 0x7fffffff, 0x00000001, 0x7ffffffe, 0x00000002, 0x7ffffffd, 0x40000000,
        0x3fffffff, 0x00000000, 0x7fffffff, 0x00000001, 0x7ffffffe, 0x00000002, 0x7ffffffd,
        0x40000000, 0x3fffffff,
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
        let res = vec + BOTH_ZEROS;
        assert_eq!(res, vec);
    }

    #[test]
    fn test_additive_identity_left() {
        let vec = packed_from_random(0xb614285cd641233c);
        let res = BOTH_ZEROS + vec;
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
        let res = vec - BOTH_ZEROS;
        assert_eq!(res, vec);
    }

    #[test]
    fn test_zero_sub() {
        let vec = packed_from_random(0x4444b9c090519333);
        let res0 = BOTH_ZEROS - vec;
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

        let vec = PackedMersenne31AVX512(arr);
        let vec_inv = PackedMersenne31AVX512(arr_inv);

        let res = vec * vec_inv;
        assert_eq!(res, P::one());
    }

    #[test]
    fn test_mul_zero() {
        let vec = packed_from_random(0x7f998daa72489bd7);
        let res = vec * BOTH_ZEROS;
        assert_eq!(res, P::zero());
    }

    #[test]
    fn test_zero_mul() {
        let vec = packed_from_random(0x683bc2dd355b06e5);
        let res = BOTH_ZEROS * vec;
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

        let vec0 = PackedMersenne31AVX512(arr0);
        let vec1 = PackedMersenne31AVX512(arr1);
        let vec_res = vec0 + vec1;

        for i in 0..WIDTH {
            assert_eq!(vec_res.0[i], arr0[i] + arr1[i]);
        }
    }

    #[test]
    fn test_add_vs_scalar_special_vals_left() {
        let arr0 = SPECIAL_VALS;
        let arr1 = array_from_random(0x1e2b153f07b64cf3);

        let vec0 = PackedMersenne31AVX512(arr0);
        let vec1 = PackedMersenne31AVX512(arr1);
        let vec_res = vec0 + vec1;

        for i in 0..WIDTH {
            assert_eq!(vec_res.0[i], arr0[i] + arr1[i]);
        }
    }

    #[test]
    fn test_add_vs_scalar_special_vals_right() {
        let arr0 = array_from_random(0xfcf974ac7625a260);
        let arr1 = SPECIAL_VALS;

        let vec0 = PackedMersenne31AVX512(arr0);
        let vec1 = PackedMersenne31AVX512(arr1);
        let vec_res = vec0 + vec1;

        for i in 0..WIDTH {
            assert_eq!(vec_res.0[i], arr0[i] + arr1[i]);
        }
    }

    #[test]
    fn test_sub_vs_scalar() {
        let arr0 = array_from_random(0x167ce9d8e920876e);
        let arr1 = array_from_random(0x52ddcdd3461e046f);

        let vec0 = PackedMersenne31AVX512(arr0);
        let vec1 = PackedMersenne31AVX512(arr1);
        let vec_res = vec0 - vec1;

        for i in 0..WIDTH {
            assert_eq!(vec_res.0[i], arr0[i] - arr1[i]);
        }
    }

    #[test]
    fn test_sub_vs_scalar_special_vals_left() {
        let arr0 = SPECIAL_VALS;
        let arr1 = array_from_random(0x358498640bfe1375);

        let vec0 = PackedMersenne31AVX512(arr0);
        let vec1 = PackedMersenne31AVX512(arr1);
        let vec_res = vec0 - vec1;

        for i in 0..WIDTH {
            assert_eq!(vec_res.0[i], arr0[i] - arr1[i]);
        }
    }

    #[test]
    fn test_sub_vs_scalar_special_vals_right() {
        let arr0 = array_from_random(0x05d81ebfb8f0005c);
        let arr1 = SPECIAL_VALS;

        let vec0 = PackedMersenne31AVX512(arr0);
        let vec1 = PackedMersenne31AVX512(arr1);
        let vec_res = vec0 - vec1;

        for i in 0..WIDTH {
            assert_eq!(vec_res.0[i], arr0[i] - arr1[i]);
        }
    }

    #[test]
    fn test_mul_vs_scalar() {
        let arr0 = array_from_random(0x4242ebdc09b74d77);
        let arr1 = array_from_random(0x9937b275b3c056cd);

        let vec0 = PackedMersenne31AVX512(arr0);
        let vec1 = PackedMersenne31AVX512(arr1);
        let vec_res = vec0 * vec1;

        for i in 0..WIDTH {
            assert_eq!(vec_res.0[i], arr0[i] * arr1[i]);
        }
    }

    #[test]
    fn test_mul_vs_scalar_special_vals_left() {
        let arr0 = SPECIAL_VALS;
        let arr1 = array_from_random(0x5285448b835458a3);

        let vec0 = PackedMersenne31AVX512(arr0);
        let vec1 = PackedMersenne31AVX512(arr1);
        let vec_res = vec0 * vec1;

        for i in 0..WIDTH {
            assert_eq!(vec_res.0[i], arr0[i] * arr1[i]);
        }
    }

    #[test]
    fn test_mul_vs_scalar_special_vals_right() {
        let arr0 = array_from_random(0x22508dc80001d865);
        let arr1 = SPECIAL_VALS;

        let vec0 = PackedMersenne31AVX512(arr0);
        let vec1 = PackedMersenne31AVX512(arr1);
        let vec_res = vec0 * vec1;

        for i in 0..WIDTH {
            assert_eq!(vec_res.0[i], arr0[i] * arr1[i]);
        }
    }

    #[test]
    fn test_neg_vs_scalar() {
        let arr = array_from_random(0xc3c273a9b334372f);

        let vec = PackedMersenne31AVX512(arr);
        let vec_res = -vec;

        for i in 0..WIDTH {
            assert_eq!(vec_res.0[i], -arr[i]);
        }
    }

    #[test]
    fn test_neg_vs_scalar_special_vals() {
        let arr = SPECIAL_VALS;

        let vec = PackedMersenne31AVX512(arr);
        let vec_res = -vec;

        for i in 0..WIDTH {
            assert_eq!(vec_res.0[i], -arr[i]);
        }
    }
}
