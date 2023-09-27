use core::arch::aarch64::{self, uint32x4_t};
use core::arch::asm;
use core::hint::unreachable_unchecked;
use core::iter::{Product, Sum};
use core::mem::transmute;
use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

use p3_field::{AbstractField, Field, PackedField};
use rand::distributions::{Distribution, Standard};
use rand::Rng;

use crate::BabyBear;

const WIDTH: usize = 4;
const P: uint32x4_t = unsafe { transmute::<[u32; WIDTH], _>([0x78000001; WIDTH]) };
const MU: uint32x4_t = unsafe { transmute::<[u32; WIDTH], _>([0x08000001; WIDTH]) };
const TOP_BIT: uint32x4_t = unsafe { transmute::<[u32; WIDTH], _>([0x80000000; WIDTH]) };

/// Vectorized NEON implementation of `BabyBear` arithmetic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)] // This needed to make `transmute`s safe.
pub struct PackedBabyBearNeon(pub [BabyBear; WIDTH]);

impl PackedBabyBearNeon {
    #[inline]
    #[must_use]
    /// Get an arch-specific vector representing the packed values.
    fn to_vector(self) -> uint32x4_t {
        unsafe {
            // Safety: `BabyBear` is `repr(transparent)` so it can be transmuted to `u32`. It
            // follows that `[BabyBear; WIDTH]` can be transmuted to `[u32; WIDTH]`, which can be
            // transmuted to `uint32x4_t`, since arrays are guaranteed to be contiguous in memory.
            // Finally `PackedBabyBearNeon` is `repr(transparent)` so it can be transmuted to
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
    unsafe fn from_vector(vector: uint32x4_t) -> Self {
        // Safety: It is up to the user to ensure that elements of `vector` represent valid
        // `BabyBear` values. We must only reason about memory representations. `uint32x4_t` can be
        // transmuted to `[u32; WIDTH]` (since arrays elements are contiguous in memory), which can
        // be transmuted to `[BabyBear; WIDTH]` (since `BabyBear` is `repr(transparent)`), which in
        // turn can be transmuted to `PackedBabyBearNeon` (since `PackedBabyBearNeon` is also
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

impl Add for PackedBabyBearNeon {
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

impl Mul for PackedBabyBearNeon {
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

impl Neg for PackedBabyBearNeon {
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

impl Sub for PackedBabyBearNeon {
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

/// No-op. Prevents the compiler from deducing the value of the vector.
///
/// Similar to `std::hint::black_box`, it can be used to stop the compiler applying undesirable
/// "optimizations". Unlike the built-in `black_box`, it does not force the value to be written to
/// and then read from the stack.
#[inline]
#[must_use]
fn confuse_compiler(x: uint32x4_t) -> uint32x4_t {
    let y;
    unsafe {
        asm!(
            "/*{0:v}*/",
            inlateout(vreg) x => y,
            options(nomem, nostack, preserves_flags, pure),
        );
        // Below tells the compiler the semantics of this so it can still do constant folding, etc.
        // You may ask, doesn't it defeat the point of the inline asm block to tell the compiler
        // what it does? The answer is that we still inhibit the transform we want to avoid, so
        // apparently not. Idk, LLVM works in mysterious ways.
        if transmute::<_, [u32; 4]>(x) != transmute::<_, [u32; 4]>(y) {
            unreachable_unchecked();
        }
    }
    y
}

/// Add two vectors of Baby Bear field elements in canonical form.
/// If the inputs are not in canonical form, the result is undefined.
#[inline]
#[must_use]
fn add(lhs: uint32x4_t, rhs: uint32x4_t) -> uint32x4_t {
    // We want this to compile to:
    //      add   t.4s, lhs.4s, rhs.4s
    //      sub   u.4s, t.4s, P.4s
    //      umin  res.4s, t.4s, u.4s
    // throughput: .75 cyc/vec (5.33 els/cyc)
    // latency: 6 cyc

    //   Let `t := lhs + rhs`. We want to return `t mod P`. Recall that `lhs` and `rhs` are in
    // `0, ..., P - 1`, so `t` is in `0, ..., 2 P - 2 (< 2^32)`. It suffices to return `t` if
    // `t < P` and `t - P` otherwise.
    //   Let `u := (t - P) mod 2^32` and `r := unsigned_min(t, u)`.
    //   If `t` is in `0, ..., P - 1`, then `u` is in `(P - 1 <) 2^32 - P, ..., 2^32 - 1` and
    // `r = t`. Otherwise `t` is in `P, ..., 2 P - 2`, `u` is in `0, ..., P - 2 (< P)` and `r = u`.
    // Hence, `r` is `t` if `t < P` and `t - P` otherwise, as desired.

    unsafe {
        // Safety: If this code got compiled then NEON intrinsics are available.
        let t = aarch64::vaddq_u32(lhs, rhs);
        let u = aarch64::vsubq_u32(t, P);
        aarch64::vminq_u32(t, u)
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

// MONTGOMERY MULTIPLICATION
//   This implementation is based on [1] but with minor changes. The reduction is as follows:
//
// Constants: P = 2^31 - 2^27 + 1
//            B = 2^31
//            mu = P^-1 mod B
// Input: 0 <= C < P B
// Output: 0 <= R < P such that R = C B^-1 (mod P)
//   1. Q := mu C mod B
//   2. T := (C - Q P) / B
//   3. R := if T < 0 then T + P else T
//
// We first show that the division in step 2. is exact. It suffices to show that C = Q P (mod B). By
// definition of Q and mu, we have Q P = mu C P = P^-1 C P = C (mod B). We also have
// C - Q P = C (mod P), so thus T = C B^-1 (mod P).
//
// It remains to show that R is in the correct range. It suffices to show that -P <= T < P. We know
// that 0 <= C < P B and 0 <= Q P < P B. Then -P B < C - QP < P B and -P < T < P, as desired.
//
// In practice, we take advantage of the fact that C = Q P (mod B) to avoid a long multiplication
// when computing Q P: we only need the top half of the product. A more practical implementation is
// as follows:
//   1. Q := mu C mod B
//   2. T := C // B - Q P // B
//   3. R := if T < 0 then T + P else T
// "//" denotes truncated division.
//
// [1] Modern Computer Arithmetic, Richard Brent and Paul Zimmermann, Cambridge University Press,
//     2010, algorithm 2.7.

/// Compute the high 31 bits of the long product. This is `C // B` in the description above.
#[inline]
#[must_use]
fn monty_mul_hi(lhs: uint32x4_t, rhs: uint32x4_t) -> uint32x4_t {
    // We want this to compile to:
    //      sqdmulh  res.4s, lhs.4s, rhs.4s
    // throughput: .25 cyc/vec (16 els/cyc)
    // latency: 3 cyc
    mul_31x31_to_hi_31(lhs, rhs)
}

/// Compute `Q P // B` in the description above.
#[allow(non_snake_case)]
#[inline]
#[must_use]
fn monty_mul_lo(lhs: uint32x4_t, rhs: uint32x4_t) -> uint32x4_t {
    // We want this to compile to:
    //      mul      rhs_mu_mod_2pow32.rs, rhs.4s, MU.4s
    //      mul      mu_C_mod_2pow32.rs, lhs.4s, rhs_mu_mod_2pow32.4s
    //      bic      mu_C_mod_2pow31.rs, mu_C_mod_2pow32.rs, 0x80, lsl #24
    //      sqdmulh  res.4s, mu_C_mod_2pow31.4s, P.4s
    // throughput: 1 cyc/vec (4 els/cyc)
    // latency: (1->1) 8 cyc
    //          (2->1) 11 cyc
    unsafe {
        // Safety: If this code got compiled then NEON intrinsics are available.
        let rhs_mu_mod_2pow32 = aarch64::vmulq_u32(rhs, MU);
        let mu_C_mod_2pow32 = aarch64::vmulq_u32(lhs, rhs_mu_mod_2pow32);
        let mu_C_mod_2pow31 = aarch64::vbicq_u32(mu_C_mod_2pow32, TOP_BIT);
        mul_31x31_to_hi_31(mu_C_mod_2pow31, P)
    }
}

/// Multiply vectors of Baby Bear field elements in canonical form.
/// If the inputs are not in canonical form, the result is undefined.
#[inline]
#[must_use]
fn mul(lhs: uint32x4_t, rhs: uint32x4_t) -> uint32x4_t {
    // throughput: 2 cyc/vec (2 els/cyc)
    // latency: (1->1) 13 cyc
    //          (2->1) 16 cyc
    let hi = monty_mul_hi(lhs, rhs);
    let lo = monty_mul_lo(lhs, rhs);
    sub(hi, lo)
}

/// Negate a vector of Baby Bear field elements in canonical form.
/// If the inputs are not in canonical form, the result is undefined.
#[inline]
#[must_use]
fn neg(val: uint32x4_t) -> uint32x4_t {
    // We want this to compile to:
    //      sub   t.4s, P.4s, val.4s
    //      cmeq  is_zero.4s, val.4s, #0
    //      bic   res.4s, t.4s, is_zero.4s
    // throughput: .75 cyc/vec (5.33 els/cyc)
    // latency: 4 cyc

    // This has the same throughput as `sub(0, val)` but slightly lower latency.

    //   We want to return (-val) mod P. This is equivalent to returning `0` if `val = 0` and
    // `P - val` otherwise, since `val` is in `0, ..., P - 1`.
    //   Let `t := P - val` and let `is_zero := (-1) mod 2^32` if `val = 0` and `0` otherwise.
    //   We return `r := t & ~is_zero`, which is `t` if `val > 0` and `0` otherwise, as desired.
    unsafe {
        // Safety: If this code got compiled then NEON intrinsics are available.
        let t = aarch64::vsubq_u32(P, val);
        let is_zero = aarch64::vceqzq_u32(val);
        aarch64::vbicq_u32(t, is_zero)
    }
}

/// Subtract vectors of Baby Bear field elements in canonical form.
/// If the inputs are not in canonical form, the result is undefined.
#[inline]
#[must_use]
fn sub(lhs: uint32x4_t, rhs: uint32x4_t) -> uint32x4_t {
    // We want this to compile to:
    //      sub   res.4s, lhs.4s, rhs.4s
    //      cmhi  underflow.4s, rhs.4s, lhs.4s
    //      mls   res.4s, underflow.4s, P.4s
    // throughput: .75 cyc/vec (5.33 els/cyc)
    // latency: 5 cyc

    //   Let `d := lhs - rhs`. We want to return `d mod P`.
    //   Since `lhs` and `rhs` are both in `0, ..., P - 1`, `d` is in `-P + 1, ..., P - 1`. It
    // suffices to return `d + P` if `d < 0` and `d` otherwise.
    //   Equivalently, we return `d + P` if `rhs > lhs` and `d` otherwise.  Observe that this
    // permits us to perform all calculations `mod 2^32`, so define `diff := d mod 2^32`.
    //   Let `underflow` be `-1 mod 2^32` if `rhs > lhs` and `0` otherwise.
    //   Finally, let `r := (diff - underflow * P) mod 2^32` and observe that
    // `r = (diff + P) mod 2^32` if `rhs > lhs` and `diff` otherwise, as desired.
    unsafe {
        // Safety: If this code got compiled then NEON intrinsics are available.
        let diff = aarch64::vsubq_u32(lhs, rhs);
        let underflow = aarch64::vcltq_u32(lhs, rhs);
        // We really want to emit a `mls` instruction here. The compiler knows that `underflow` is
        // either 0 or -1 and will try to do an `and` and `add` instead, which is slower on the M1.
        // The `confuse_compiler` prevents this "optimization".
        aarch64::vmlsq_u32(diff, confuse_compiler(underflow), P)
    }
}

impl From<BabyBear> for PackedBabyBearNeon {
    #[inline]
    fn from(value: BabyBear) -> Self {
        Self::broadcast(value)
    }
}

impl Default for PackedBabyBearNeon {
    #[inline]
    fn default() -> Self {
        BabyBear::default().into()
    }
}

impl AddAssign for PackedBabyBearNeon {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl MulAssign for PackedBabyBearNeon {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl SubAssign for PackedBabyBearNeon {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Sum for PackedBabyBearNeon {
    #[inline]
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.reduce(|lhs, rhs| lhs + rhs).unwrap_or(Self::ZERO)
    }
}

impl Product for PackedBabyBearNeon {
    #[inline]
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.reduce(|lhs, rhs| lhs * rhs).unwrap_or(Self::ONE)
    }
}

impl AbstractField for PackedBabyBearNeon {
    type F = BabyBear;

    const ZERO: Self = Self::broadcast(BabyBear::ZERO);
    const ONE: Self = Self::broadcast(BabyBear::ONE);
    const TWO: Self = Self::broadcast(BabyBear::TWO);
    const NEG_ONE: Self = Self::broadcast(BabyBear::NEG_ONE);

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

impl Add<BabyBear> for PackedBabyBearNeon {
    type Output = Self;
    #[inline]
    fn add(self, rhs: BabyBear) -> Self {
        self + Self::from(rhs)
    }
}

impl Mul<BabyBear> for PackedBabyBearNeon {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: BabyBear) -> Self {
        self * Self::from(rhs)
    }
}

impl Sub<BabyBear> for PackedBabyBearNeon {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: BabyBear) -> Self {
        self - Self::from(rhs)
    }
}

impl AddAssign<BabyBear> for PackedBabyBearNeon {
    #[inline]
    fn add_assign(&mut self, rhs: BabyBear) {
        *self += Self::from(rhs)
    }
}

impl MulAssign<BabyBear> for PackedBabyBearNeon {
    #[inline]
    fn mul_assign(&mut self, rhs: BabyBear) {
        *self *= Self::from(rhs)
    }
}

impl SubAssign<BabyBear> for PackedBabyBearNeon {
    #[inline]
    fn sub_assign(&mut self, rhs: BabyBear) {
        *self -= Self::from(rhs)
    }
}

impl Sum<BabyBear> for PackedBabyBearNeon {
    #[inline]
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = BabyBear>,
    {
        iter.sum::<BabyBear>().into()
    }
}

impl Product<BabyBear> for PackedBabyBearNeon {
    #[inline]
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = BabyBear>,
    {
        iter.product::<BabyBear>().into()
    }
}

impl Div<BabyBear> for PackedBabyBearNeon {
    type Output = Self;
    #[allow(clippy::suspicious_arithmetic_impl)]
    #[inline]
    fn div(self, rhs: BabyBear) -> Self {
        self * rhs.inverse()
    }
}

impl Add<PackedBabyBearNeon> for BabyBear {
    type Output = PackedBabyBearNeon;
    #[inline]
    fn add(self, rhs: PackedBabyBearNeon) -> PackedBabyBearNeon {
        PackedBabyBearNeon::from(self) + rhs
    }
}

impl Mul<PackedBabyBearNeon> for BabyBear {
    type Output = PackedBabyBearNeon;
    #[inline]
    fn mul(self, rhs: PackedBabyBearNeon) -> PackedBabyBearNeon {
        PackedBabyBearNeon::from(self) * rhs
    }
}

impl Sub<PackedBabyBearNeon> for BabyBear {
    type Output = PackedBabyBearNeon;
    #[inline]
    fn sub(self, rhs: PackedBabyBearNeon) -> PackedBabyBearNeon {
        PackedBabyBearNeon::from(self) - rhs
    }
}

impl Distribution<PackedBabyBearNeon> for Standard {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> PackedBabyBearNeon {
        PackedBabyBearNeon(rng.gen())
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

unsafe impl PackedField for PackedBabyBearNeon {
    type Scalar = BabyBear;

    const WIDTH: usize = WIDTH;

    #[inline]
    fn from_slice(slice: &[BabyBear]) -> &Self {
        assert_eq!(slice.len(), Self::WIDTH);
        unsafe {
            // Safety: `[BabyBear; WIDTH]` can be transmuted to `PackedBabyBearNeon` since the
            // latter is `repr(transparent)`. They have the same alignment, so the reference cast is
            // safe too.
            &*slice.as_ptr().cast()
        }
    }
    #[inline]
    fn from_slice_mut(slice: &mut [BabyBear]) -> &mut Self {
        assert_eq!(slice.len(), Self::WIDTH);
        unsafe {
            // Safety: `[BabyBear; WIDTH]` can be transmuted to `PackedBabyBearNeon` since the
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

    type F = BabyBear;
    type P = PackedBabyBearNeon;

    fn array_from_canonical(vals: [u32; WIDTH]) -> [F; WIDTH] {
        vals.map(F::from_canonical_u32)
    }

    fn packed_from_canonical(vals: [u32; WIDTH]) -> P {
        PackedBabyBearNeon(array_from_canonical(vals))
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
        let res = -(-vec);
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
        let inverses = packed_from_canonical([0x654400cb, 0x060e1058, 0x2b9a681f, 0x4fea4617]);
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

        let vec0 = PackedBabyBearNeon(arr0);
        let vec1 = PackedBabyBearNeon(arr1);
        let vec_res = vec0 + vec1;

        for i in 0..WIDTH {
            assert_eq!(vec_res.0[i], arr0[i] + arr1[i]);
        }
    }

    #[test]
    fn test_add_vs_scalar_special_vals_left() {
        let arr0 = [F::ZERO, F::ONE, F::TWO, F::NEG_ONE];
        let arr1 = array_from_canonical([0x4205a2f6, 0x6f4715f1, 0x29ed7f70, 0x70915992]);

        let vec0 = PackedBabyBearNeon(arr0);
        let vec1 = PackedBabyBearNeon(arr1);
        let vec_res = vec0 + vec1;

        for i in 0..WIDTH {
            assert_eq!(vec_res.0[i], arr0[i] + arr1[i]);
        }
    }

    #[test]
    fn test_add_vs_scalar_special_vals_right() {
        let arr0 = array_from_canonical([0x5f8329a7, 0x0f1166bb, 0x657bcb14, 0x0185c34a]);
        let arr1 = [F::ZERO, F::ONE, F::TWO, F::NEG_ONE];

        let vec0 = PackedBabyBearNeon(arr0);
        let vec1 = PackedBabyBearNeon(arr1);
        let vec_res = vec0 + vec1;

        for i in 0..WIDTH {
            assert_eq!(vec_res.0[i], arr0[i] + arr1[i]);
        }
    }

    #[test]
    fn test_sub_vs_scalar() {
        let arr0 = array_from_canonical([0x6daef778, 0x0e868440, 0x54e7ca64, 0x01a9acab]);
        let arr1 = array_from_canonical([0x45609584, 0x67b63536, 0x0f72a573, 0x234a312e]);

        let vec0 = PackedBabyBearNeon(arr0);
        let vec1 = PackedBabyBearNeon(arr1);
        let vec_res = vec0 - vec1;

        for i in 0..WIDTH {
            assert_eq!(vec_res.0[i], arr0[i] - arr1[i]);
        }
    }

    #[test]
    fn test_sub_vs_scalar_special_vals_left() {
        let arr0 = [F::ZERO, F::ONE, F::TWO, F::NEG_ONE];
        let arr1 = array_from_canonical([0x4205a2f6, 0x6f4715f1, 0x29ed7f70, 0x70915992]);

        let vec0 = PackedBabyBearNeon(arr0);
        let vec1 = PackedBabyBearNeon(arr1);
        let vec_res = vec0 - vec1;

        for i in 0..WIDTH {
            assert_eq!(vec_res.0[i], arr0[i] - arr1[i]);
        }
    }

    #[test]
    fn test_sub_vs_scalar_special_vals_right() {
        let arr0 = array_from_canonical([0x5f8329a7, 0x0f1166bb, 0x657bcb14, 0x0185c34a]);
        let arr1 = [F::ZERO, F::ONE, F::TWO, F::NEG_ONE];

        let vec0 = PackedBabyBearNeon(arr0);
        let vec1 = PackedBabyBearNeon(arr1);
        let vec_res = vec0 - vec1;

        for i in 0..WIDTH {
            assert_eq!(vec_res.0[i], arr0[i] - arr1[i]);
        }
    }

    #[test]
    fn test_mul_vs_scalar() {
        let arr0 = array_from_canonical([0x13655880, 0x5223ea02, 0x5d7f4f90, 0x1494b624]);
        let arr1 = array_from_canonical([0x0ad5743c, 0x44956741, 0x533bc885, 0x7723a25b]);

        let vec0 = PackedBabyBearNeon(arr0);
        let vec1 = PackedBabyBearNeon(arr1);
        let vec_res = vec0 * vec1;

        for i in 0..WIDTH {
            assert_eq!(vec_res.0[i], arr0[i] * arr1[i]);
        }
    }

    #[test]
    fn test_mul_vs_scalar_special_vals_left() {
        let arr0 = [F::ZERO, F::ONE, F::TWO, F::NEG_ONE];
        let arr1 = array_from_canonical([0x4205a2f6, 0x6f4715f1, 0x29ed7f70, 0x70915992]);

        let vec0 = PackedBabyBearNeon(arr0);
        let vec1 = PackedBabyBearNeon(arr1);
        let vec_res = vec0 * vec1;

        for i in 0..WIDTH {
            assert_eq!(vec_res.0[i], arr0[i] * arr1[i]);
        }
    }

    #[test]
    fn test_mul_vs_scalar_special_vals_right() {
        let arr0 = array_from_canonical([0x5f8329a7, 0x0f1166bb, 0x657bcb14, 0x0185c34a]);
        let arr1 = [F::ZERO, F::ONE, F::TWO, F::NEG_ONE];

        let vec0 = PackedBabyBearNeon(arr0);
        let vec1 = PackedBabyBearNeon(arr1);
        let vec_res = vec0 * vec1;

        for i in 0..WIDTH {
            assert_eq!(vec_res.0[i], arr0[i] * arr1[i]);
        }
    }

    #[test]
    fn test_neg_vs_scalar() {
        let arr = array_from_canonical([0x1971a7b5, 0x00305be1, 0x52c08410, 0x39cb2586]);

        let vec = PackedBabyBearNeon(arr);
        let vec_res = -vec;

        #[allow(clippy::needless_range_loop)]
        for i in 0..WIDTH {
            assert_eq!(vec_res.0[i], -arr[i]);
        }
    }

    #[test]
    fn test_neg_vs_scalar_special_vals() {
        let arr = [F::ZERO, F::ONE, F::TWO, F::NEG_ONE];

        let vec = PackedBabyBearNeon(arr);
        let vec_res = -vec;

        #[allow(clippy::needless_range_loop)]
        for i in 0..WIDTH {
            assert_eq!(vec_res.0[i], -arr[i]);
        }
    }
}
