use core::arch::x86_64::{self, __m256i};
use core::iter::{Product, Sum};
use core::mem::transmute;
use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

use p3_field::{AbstractField, Field, PackedField, PackedValue};
use rand::distributions::{Distribution, Standard};
use rand::Rng;

use crate::Mersenne31;

const WIDTH: usize = 8;
const P: __m256i = unsafe { transmute::<[u32; WIDTH], _>([0x7fffffff; WIDTH]) };

/// Vectorized AVX2 implementation of `Mersenne31` arithmetic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)] // This needed to make `transmute`s safe.
pub struct PackedMersenne31AVX2(pub [Mersenne31; WIDTH]);

impl PackedMersenne31AVX2 {
    #[inline]
    #[must_use]
    /// Get an arch-specific vector representing the packed values.
    fn to_vector(self) -> __m256i {
        unsafe {
            // Safety: `Mersenne31` is `repr(transparent)` so it can be transmuted to `u32`. It
            // follows that `[Mersenne31; WIDTH]` can be transmuted to `[u32; WIDTH]`, which can be
            // transmuted to `__m256i`, since arrays are guaranteed to be contiguous in memory.
            // Finally `PackedMersenne31AVX2` is `repr(transparent)` so it can be transmuted to
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
    unsafe fn from_vector(vector: __m256i) -> Self {
        // Safety: It is up to the user to ensure that elements of `vector` represent valid
        // `Mersenne31` values. We must only reason about memory representations. `__m256i` can be
        // transmuted to `[u32; WIDTH]` (since arrays elements are contiguous in memory), which can
        // be transmuted to `[Mersenne31; WIDTH]` (since `Mersenne31` is `repr(transparent)`), which
        // in turn can be transmuted to `PackedMersenne31AVX2` (since `PackedMersenne31AVX2` is also
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

impl Add for PackedMersenne31AVX2 {
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

impl Mul for PackedMersenne31AVX2 {
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

impl Neg for PackedMersenne31AVX2 {
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

impl Sub for PackedMersenne31AVX2 {
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
fn add(lhs: __m256i, rhs: __m256i) -> __m256i {
    // We want this to compile to:
    //      vpaddd   t, lhs, rhs
    //      vpsubd   u, t, P
    //      vpminud  res, t, u
    // throughput: 1 cyc/vec (8 els/cyc)
    // latency: 3 cyc

    //   Let t := lhs + rhs. We want to return a value r in {0, ..., P} such that r = t (mod P).
    //   Define u := (t - P) mod 2^32 and r := min(t, u). t is in {0, ..., 2 P}. We argue by cases.
    //   If t is in {0, ..., P - 1}, then u is in {(P - 1 <) 2^32 - P, ..., 2^32 - 1}, so r = t is
    // in the correct range.
    //   If t is in {P, ..., 2 P}, then u is in {0, ..., P} and r = u is in the correct range.
    unsafe {
        // Safety: If this code got compiled then AVX2 intrinsics are available.
        let t = x86_64::_mm256_add_epi32(lhs, rhs);
        let u = x86_64::_mm256_sub_epi32(t, P);
        x86_64::_mm256_min_epu32(t, u)
    }
}

#[inline]
#[must_use]
fn movehdup_epi32(x: __m256i) -> __m256i {
    // The instruction is only available in the floating-point flavor; this distinction is only for
    // historical reasons and no longer matters. We cast to floats, duplicate, and cast back.
    unsafe {
        x86_64::_mm256_castps_si256(x86_64::_mm256_movehdup_ps(x86_64::_mm256_castsi256_ps(x)))
    }
}

/// Multiply vectors of Mersenne-31 field elements represented as values in {0, ..., P}.
/// If the inputs do not conform to this representation, the result is undefined.
#[inline]
#[must_use]
fn mul(lhs: __m256i, rhs: __m256i) -> __m256i {
    // We want this to compile to:
    // vpsrlq     lhs_odd_dbl, lhs, 31
    // vmovshdup  rhs_odd, rhs
    // vpmuludq   prod_odd_dbl, lhs_odd_dbl, rhs_odd
    // vpmuludq   prod_evn, lhs, rhs
    // vpsllq     prod_odd_lo_dirty, prod_odd_dbl, 31
    // vpsrlq     prod_evn_hi, prod_evn, 31
    // vpblendd   prod_lo_dirty, prod_evn, prod_odd_lo_dirty, aah
    // vpblendd   prod_hi, prod_evn_hi, prod_odd_dbl, aah
    // vpand      prod_lo, prod_lo_dirty, P
    // vpaddd     t, prod_lo, prod_hi
    // vpsubd     u, t, P
    // vpminud    res, t, u
    // throughput: 4 cyc/vec (2 els/cyc)
    // latency: 13 cyc
    unsafe {
        // vpmuludq only reads the bottom 32 bits of every 64-bit quadword.
        // The even indices are already in the bottom 32 bits of a quadword, so we can leave them.
        let lhs_evn = lhs;
        let rhs_evn = rhs;
        // Right shift by 31 is equivalent to moving the high 32 bits down to the low 32, and then
        // doubling it. So these are the odd indices in lhs, but doubled.
        let lhs_odd_dbl = x86_64::_mm256_srli_epi64::<31>(lhs);
        // Copy the high 32 bits in each quadword of rhs down to the low 32.
        let rhs_odd = movehdup_epi32(rhs);

        // Multiply odd indices; since lhs_odd_dbl is doubled, these products are also doubled.
        // prod_odd_dbl.quadword[i] = 2 * lsh.doubleword[2 * i + 1] * rhs.doubleword[2 * i + 1]
        let prod_odd_dbl = x86_64::_mm256_mul_epu32(rhs_odd, lhs_odd_dbl);
        // Multiply even indices.
        // prod_evn.quadword[i] = lsh.doubleword[2 * i] * rhs.doubleword[2 * i]
        let prod_evn = x86_64::_mm256_mul_epu32(rhs_evn, lhs_evn);

        // We now need to extract the low 31 bits and the high 31 bits of each 62 bit product and
        // prepare to add them.
        // Put the low 31 bits of the product (recall that it is shifted left by 1) in an odd
        // doubleword. (Notice that the high 31 bits are already in an odd doubleword in
        // prod_odd_dbl.) We will still need to clear the sign bit, hence we mark it _dirty.
        let prod_odd_lo_dirty = x86_64::_mm256_slli_epi64::<31>(prod_odd_dbl);
        // Put the high 31 bits in an even doubleword, again noting that in prod_evn the even
        // doublewords contain the low 31 bits (with a dirty sign bit).
        let prod_evn_hi = x86_64::_mm256_srli_epi64::<31>(prod_evn);

        // Put all the low halves of all the products into one vector. Take the even values from
        // prod_evn and odd values from prod_odd_lo_dirty. Note that the sign bits still need
        // clearing.
        let prod_lo_dirty = x86_64::_mm256_blend_epi32::<0b10101010>(prod_evn, prod_odd_lo_dirty);
        // Now put all the high halves into one vector. The even values come from prod_evn_hi and
        // the odd values come from prod_odd_dbl.
        let prod_hi = x86_64::_mm256_blend_epi32::<0b10101010>(prod_evn_hi, prod_odd_dbl);
        // Clear the most significant bit.
        let prod_lo = x86_64::_mm256_and_si256(prod_lo_dirty, P);

        // Standard addition of two 31-bit values.
        add(prod_lo, prod_hi)
    }
}

/// Negate a vector of Mersenne-31 field elements represented as values in {0, ..., P}.
/// If the input does not conform to this representation, the result is undefined.
#[inline]
#[must_use]
fn neg(val: __m256i) -> __m256i {
    // We want this to compile to:
    //      vpxor  res, val, P
    // throughput: .33 cyc/vec (24 els/cyc)
    // latency: 1 cyc

    //   Since val is in {0, ..., P (= 2^31 - 1)}, res = val XOR P = P - val. Then res is in {0,
    // ..., P}.
    unsafe {
        // Safety: If this code got compiled then AVX2 intrinsics are available.
        x86_64::_mm256_xor_si256(val, P)
    }
}

/// Subtract vectors of Mersenne-31 field elements represented as values in {0, ..., P}.
/// If the inputs do not conform to this representation, the result is undefined.
#[inline]
#[must_use]
fn sub(lhs: __m256i, rhs: __m256i) -> __m256i {
    // We want this to compile to:
    //      vpsubd   t, lhs, rhs
    //      vpaddd   u, t, P
    //      vpminud  res, t, u
    // throughput: 1 cyc/vec (8 els/cyc)
    // latency: 3 cyc

    //   Let d := lhs - rhs and t := d mod 2^32. We want to return a value r in {0, ..., P} such
    // that r = d (mod P).
    //   Define u := (t + P) mod 2^32 and r := min(t, u). d is in {-P, ..., P}. We argue by cases.
    //   If d is in {0, ..., P}, then t = d and u is in {P, ..., 2 P}. r = t is in the correct
    // range.
    //   If d is in {-P, ..., -1}, then t is in {2^32 - P, ..., 2^32 - 1} and u is in
    // {0, ..., P - 1}. r = u is in the correct range.
    unsafe {
        // Safety: If this code got compiled then AVX2 intrinsics are available.
        let t = x86_64::_mm256_sub_epi32(lhs, rhs);
        let u = x86_64::_mm256_add_epi32(t, P);
        x86_64::_mm256_min_epu32(t, u)
    }
}

impl From<Mersenne31> for PackedMersenne31AVX2 {
    #[inline]
    fn from(value: Mersenne31) -> Self {
        Self::broadcast(value)
    }
}

impl Default for PackedMersenne31AVX2 {
    #[inline]
    fn default() -> Self {
        Mersenne31::default().into()
    }
}

impl AddAssign for PackedMersenne31AVX2 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl MulAssign for PackedMersenne31AVX2 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl SubAssign for PackedMersenne31AVX2 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Sum for PackedMersenne31AVX2 {
    #[inline]
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.reduce(|lhs, rhs| lhs + rhs).unwrap_or(Self::zero())
    }
}

impl Product for PackedMersenne31AVX2 {
    #[inline]
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.reduce(|lhs, rhs| lhs * rhs).unwrap_or(Self::one())
    }
}

impl AbstractField for PackedMersenne31AVX2 {
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

impl Add<Mersenne31> for PackedMersenne31AVX2 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Mersenne31) -> Self {
        self + Self::from(rhs)
    }
}

impl Mul<Mersenne31> for PackedMersenne31AVX2 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Mersenne31) -> Self {
        self * Self::from(rhs)
    }
}

impl Sub<Mersenne31> for PackedMersenne31AVX2 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Mersenne31) -> Self {
        self - Self::from(rhs)
    }
}

impl AddAssign<Mersenne31> for PackedMersenne31AVX2 {
    #[inline]
    fn add_assign(&mut self, rhs: Mersenne31) {
        *self += Self::from(rhs)
    }
}

impl MulAssign<Mersenne31> for PackedMersenne31AVX2 {
    #[inline]
    fn mul_assign(&mut self, rhs: Mersenne31) {
        *self *= Self::from(rhs)
    }
}

impl SubAssign<Mersenne31> for PackedMersenne31AVX2 {
    #[inline]
    fn sub_assign(&mut self, rhs: Mersenne31) {
        *self -= Self::from(rhs)
    }
}

impl Sum<Mersenne31> for PackedMersenne31AVX2 {
    #[inline]
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = Mersenne31>,
    {
        iter.sum::<Mersenne31>().into()
    }
}

impl Product<Mersenne31> for PackedMersenne31AVX2 {
    #[inline]
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = Mersenne31>,
    {
        iter.product::<Mersenne31>().into()
    }
}

impl Div<Mersenne31> for PackedMersenne31AVX2 {
    type Output = Self;
    #[allow(clippy::suspicious_arithmetic_impl)]
    #[inline]
    fn div(self, rhs: Mersenne31) -> Self {
        self * rhs.inverse()
    }
}

impl Add<PackedMersenne31AVX2> for Mersenne31 {
    type Output = PackedMersenne31AVX2;
    #[inline]
    fn add(self, rhs: PackedMersenne31AVX2) -> PackedMersenne31AVX2 {
        PackedMersenne31AVX2::from(self) + rhs
    }
}

impl Mul<PackedMersenne31AVX2> for Mersenne31 {
    type Output = PackedMersenne31AVX2;
    #[inline]
    fn mul(self, rhs: PackedMersenne31AVX2) -> PackedMersenne31AVX2 {
        PackedMersenne31AVX2::from(self) * rhs
    }
}

impl Sub<PackedMersenne31AVX2> for Mersenne31 {
    type Output = PackedMersenne31AVX2;
    #[inline]
    fn sub(self, rhs: PackedMersenne31AVX2) -> PackedMersenne31AVX2 {
        PackedMersenne31AVX2::from(self) - rhs
    }
}

impl Distribution<PackedMersenne31AVX2> for Standard {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> PackedMersenne31AVX2 {
        PackedMersenne31AVX2(rng.gen())
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

unsafe impl PackedValue for PackedMersenne31AVX2 {
    type Value = Mersenne31;

    const WIDTH: usize = WIDTH;

    #[inline]
    fn from_slice(slice: &[Mersenne31]) -> &Self {
        assert_eq!(slice.len(), Self::WIDTH);
        unsafe {
            // Safety: `[Mersenne31; WIDTH]` can be transmuted to `PackedMersenne31AVX2` since the
            // latter is `repr(transparent)`. They have the same alignment, so the reference cast is
            // safe too.
            &*slice.as_ptr().cast()
        }
    }
    #[inline]
    fn from_slice_mut(slice: &mut [Mersenne31]) -> &mut Self {
        assert_eq!(slice.len(), Self::WIDTH);
        unsafe {
            // Safety: `[Mersenne31; WIDTH]` can be transmuted to `PackedMersenne31AVX2` since the
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

unsafe impl PackedField for PackedMersenne31AVX2 {
    type Scalar = Mersenne31;

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

    use super::{Mersenne31, WIDTH};
    use crate::to_mersenne31_array;

    /// Zero has a redundant representation, so let's test both.
    const ZEROS: [Mersenne31; WIDTH] = to_mersenne31_array([
        0x00000000, 0x7fffffff, 0x00000000, 0x7fffffff, 0x00000000, 0x7fffffff, 0x00000000,
        0x7fffffff,
    ]);

    const SPECIAL_VALS: [Mersenne31; WIDTH] = to_mersenne31_array([
        0x00000000, 0x7fffffff, 0x00000001, 0x7ffffffe, 0x00000002, 0x7ffffffd, 0x40000000,
        0x3fffffff,
    ]);

    test_packed_field!(
        crate::PackedMersenne31AVX2,
        crate::PackedMersenne31AVX2(super::ZEROS),
        crate::PackedMersenne31AVX2(super::SPECIAL_VALS)
    );
}
