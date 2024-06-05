use core::arch::aarch64::{self, int32x4_t, uint32x4_t};
use core::arch::asm;
use core::hint::unreachable_unchecked;
use core::iter::{Product, Sum};
use core::mem::transmute;
use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

use p3_field::{AbstractField, Field, PackedField, PackedValue};
use rand::distributions::{Distribution, Standard};
use rand::Rng;

use crate::KoalaBear;

const WIDTH: usize = 4;
const P: uint32x4_t = unsafe { transmute::<[u32; WIDTH], _>([0x7f000001; WIDTH]) };
// This MU is the same 0x81000001 as elsewhere, just interpreted as an `i32`.
const MU: int32x4_t = unsafe { transmute::<[i32; WIDTH], _>([-0x7effffff; WIDTH]) };

/// Vectorized NEON implementation of `KoalaBear` arithmetic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)] // This needed to make `transmute`s safe.
pub struct PackedKoalaBearNeon(pub [KoalaBear; WIDTH]);

impl PackedKoalaBearNeon {
    #[inline]
    #[must_use]
    /// Get an arch-specific vector representing the packed values.
    fn to_vector(self) -> uint32x4_t {
        unsafe {
            // Safety: `KoalaBear` is `repr(transparent)` so it can be transmuted to `u32`. It
            // follows that `[KoalaBear; WIDTH]` can be transmuted to `[u32; WIDTH]`, which can be
            // transmuted to `uint32x4_t`, since arrays are guaranteed to be contiguous in memory.
            // Finally `PackedKoalaBearNeon` is `repr(transparent)` so it can be transmuted to
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
    unsafe fn from_vector(vector: uint32x4_t) -> Self {
        // Safety: It is up to the user to ensure that elements of `vector` represent valid
        // `KoalaBear` values. We must only reason about memory representations. `uint32x4_t` can be
        // transmuted to `[u32; WIDTH]` (since arrays elements are contiguous in memory), which can
        // be transmuted to `[KoalaBear; WIDTH]` (since `KoalaBear` is `repr(transparent)`), which in
        // turn can be transmuted to `PackedKoalaBearNeon` (since `PackedKoalaBearNeon` is also
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

impl Add for PackedKoalaBearNeon {
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

impl Mul for PackedKoalaBearNeon {
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

impl Neg for PackedKoalaBearNeon {
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

impl Sub for PackedKoalaBearNeon {
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
        if transmute::<uint32x4_t, [u32; 4]>(x) != transmute::<uint32x4_t, [u32; 4]>(y) {
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

// MONTGOMERY MULTIPLICATION
//   This implementation is based on [1] but with changes. The reduction is as follows:
//
// Constants: P = 2^31 - 2^24 + 1
//            B = 2^32
//            μ = P^-1 mod B
// Input: -P^2 <= C <= P^2
// Output: -P < D < P such that D = C B^-1 (mod P)
// Define:
//   smod_B(a) = r, where -B/2 <= r <= B/2 - 1 and r = a (mod B).
// Algorithm:
//   1. Q := smod_B(μ C)
//   2. D := (C - Q P) / B
//
// We first show that the division in step 2. is exact. It suffices to show that C = Q P (mod B). By
// definition of Q, smod_B, and μ, we have Q P = smod_B(μ C) P = μ C P = P^-1 C P = C (mod B).
//
// We also have C - Q P = C (mod P), so thus D = C B^-1 (mod P).
//
// It remains to show that D is in the correct range. It suffices to show that -P B < C - Q P < P B.
// We know that -P^2 <= C <= P^2 and (-B / 2) P <= Q P <= (B/2 - 1) P. Then
// (1 - B/2) P - P^2 <= C - Q P <= (B/2) P + P^2. Now, P < B/2, so B/2 + P < B and
// (B/2) P + P^2 < P B; also B/2 - 1 + P < B, so -P B < (1 - B/2) P - P^2.
// Hence, -P B < C - Q P < P B as desired.
//
// [1] Modern Computer Arithmetic, Richard Brent and Paul Zimmermann, Cambridge University Press,
//     2010, algorithm 2.7.

#[inline]
#[must_use]
fn mulby_mu(val: int32x4_t) -> int32x4_t {
    // We want this to compile to:
    //      mul      res.4s, val.4s, MU.4s
    // throughput: .25 cyc/vec (16 els/cyc)
    // latency: 3 cyc

    unsafe { aarch64::vmulq_s32(val, MU) }
}

#[inline]
#[must_use]
fn get_c_hi(lhs: int32x4_t, rhs: int32x4_t) -> int32x4_t {
    // We want this to compile to:
    //      sqdmulh  c_hi.4s, lhs.4s, rhs.4s
    // throughput: .25 cyc/vec (16 els/cyc)
    // latency: 3 cyc

    unsafe {
        // Get bits 31, ..., 62 of C. Note that `sqdmulh` saturates when the product doesn't fit in
        // an `i63`, but this cannot happen here due to our bounds on `lhs` and `rhs`.
        aarch64::vqdmulhq_s32(lhs, rhs)
    }
}

#[inline]
#[must_use]
fn get_qp_hi(lhs: int32x4_t, mu_rhs: int32x4_t) -> int32x4_t {
    // We want this to compile to:
    //      mul      q.4s, lhs.4s, mu_rhs.4s
    //      sqdmulh  qp_hi.4s, q.4s, P.4s
    // throughput: .5 cyc/vec (8 els/cyc)
    // latency: 6 cyc

    unsafe {
        // Form `Q`.
        let q = aarch64::vmulq_s32(lhs, mu_rhs);

        // Gets bits 31, ..., 62 of Q P. Again, saturation is not an issue because `P` is not
        // -2**31.
        aarch64::vqdmulhq_s32(q, aarch64::vreinterpretq_s32_u32(P))
    }
}

#[inline]
#[must_use]
fn get_d(c_hi: int32x4_t, qp_hi: int32x4_t) -> int32x4_t {
    // We want this to compile to:
    //      shsub    res.4s, c_hi.4s, qp_hi.4s
    // throughput: .25 cyc/vec (16 els/cyc)
    // latency: 2 cyc

    unsafe {
        // Form D. Note that `c_hi` is C >> 31 and `qp_hi` is (Q P) >> 31, whereas we want
        // (C - Q P) >> 32, so we need to subtract and divide by 2. Luckily NEON has an instruction
        // for that! The lowest bit of `c_hi` and `qp_hi` is the same, so the division is exact.
        aarch64::vhsubq_s32(c_hi, qp_hi)
    }
}

#[inline]
#[must_use]
fn get_reduced_d(c_hi: int32x4_t, qp_hi: int32x4_t) -> uint32x4_t {
    // We want this to compile to:
    //      shsub    res.4s, c_hi.4s, qp_hi.4s
    //      cmgt     underflow.4s, qp_hi.4s, c_hi.4s
    //      mls      res.4s, underflow.4s, P.4s
    // throughput: .75 cyc/vec (5.33 els/cyc)
    // latency: 5 cyc

    unsafe {
        let d = aarch64::vreinterpretq_u32_s32(get_d(c_hi, qp_hi));

        // Finally, we reduce D to canonical form. D is negative iff `c_hi > qp_hi`, so if that's the
        // case then we add P. Note that if `c_hi > qp_hi` then `underflow` is -1, so we must
        // _subtract_ `underflow` * P.
        let underflow = aarch64::vcltq_s32(c_hi, qp_hi);
        aarch64::vmlsq_u32(d, confuse_compiler(underflow), P)
    }
}

#[inline]
#[must_use]
fn mul(lhs: uint32x4_t, rhs: uint32x4_t) -> uint32x4_t {
    // We want this to compile to:
    //      sqdmulh  c_hi.4s, lhs.4s, rhs.4s
    //      mul      mu_rhs.4s, rhs.4s, MU.4s
    //      mul      q.4s, lhs.4s, mu_rhs.4s
    //      sqdmulh  qp_hi.4s, q.4s, P.4s
    //      shsub    res.4s, c_hi.4s, qp_hi.4s
    //      cmgt     underflow.4s, qp_hi.4s, c_hi.4s
    //      mls      res.4s, underflow.4s, P.4s
    // throughput: 1.75 cyc/vec (2.29 els/cyc)
    // latency: (lhs->) 11 cyc, (rhs->) 14 cyc

    unsafe {
        // No-op. The inputs are non-negative so we're free to interpret them as signed numbers.
        let lhs = aarch64::vreinterpretq_s32_u32(lhs);
        let rhs = aarch64::vreinterpretq_s32_u32(rhs);

        let mu_rhs = mulby_mu(rhs);
        let c_hi = get_c_hi(lhs, rhs);
        let qp_hi = get_qp_hi(lhs, mu_rhs);
        get_reduced_d(c_hi, qp_hi)
    }
}

#[inline]
#[must_use]
fn cube(val: uint32x4_t) -> uint32x4_t {
    // throughput: 2.75 cyc/vec (1.45 els/cyc)
    // latency: 22 cyc

    unsafe {
        let val = aarch64::vreinterpretq_s32_u32(val);
        let mu_val = mulby_mu(val);

        let c_hi_2 = get_c_hi(val, val);
        let qp_hi_2 = get_qp_hi(val, mu_val);
        let val_2 = get_d(c_hi_2, qp_hi_2);

        let c_hi_3 = get_c_hi(val_2, val);
        let qp_hi_3 = get_qp_hi(val_2, mu_val);
        get_reduced_d(c_hi_3, qp_hi_3)
    }
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

impl From<KoalaBear> for PackedKoalaBearNeon {
    #[inline]
    fn from(value: KoalaBear) -> Self {
        Self::broadcast(value)
    }
}

impl Default for PackedKoalaBearNeon {
    #[inline]
    fn default() -> Self {
        KoalaBear::default().into()
    }
}

impl AddAssign for PackedKoalaBearNeon {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl MulAssign for PackedKoalaBearNeon {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl SubAssign for PackedKoalaBearNeon {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Sum for PackedKoalaBearNeon {
    #[inline]
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.reduce(|lhs, rhs| lhs + rhs).unwrap_or(Self::zero())
    }
}

impl Product for PackedKoalaBearNeon {
    #[inline]
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.reduce(|lhs, rhs| lhs * rhs).unwrap_or(Self::one())
    }
}

impl AbstractField for PackedKoalaBearNeon {
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

    #[inline]
    fn cube(&self) -> Self {
        let val = self.to_vector();
        let res = cube(val);
        unsafe {
            // Safety: `cube` returns values in canonical form when given values in canonical form.
            Self::from_vector(res)
        }
    }
}

impl Add<KoalaBear> for PackedKoalaBearNeon {
    type Output = Self;
    #[inline]
    fn add(self, rhs: KoalaBear) -> Self {
        self + Self::from(rhs)
    }
}

impl Mul<KoalaBear> for PackedKoalaBearNeon {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: KoalaBear) -> Self {
        self * Self::from(rhs)
    }
}

impl Sub<KoalaBear> for PackedKoalaBearNeon {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: KoalaBear) -> Self {
        self - Self::from(rhs)
    }
}

impl AddAssign<KoalaBear> for PackedKoalaBearNeon {
    #[inline]
    fn add_assign(&mut self, rhs: KoalaBear) {
        *self += Self::from(rhs)
    }
}

impl MulAssign<KoalaBear> for PackedKoalaBearNeon {
    #[inline]
    fn mul_assign(&mut self, rhs: KoalaBear) {
        *self *= Self::from(rhs)
    }
}

impl SubAssign<KoalaBear> for PackedKoalaBearNeon {
    #[inline]
    fn sub_assign(&mut self, rhs: KoalaBear) {
        *self -= Self::from(rhs)
    }
}

impl Sum<KoalaBear> for PackedKoalaBearNeon {
    #[inline]
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = KoalaBear>,
    {
        iter.sum::<KoalaBear>().into()
    }
}

impl Product<KoalaBear> for PackedKoalaBearNeon {
    #[inline]
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = KoalaBear>,
    {
        iter.product::<KoalaBear>().into()
    }
}

impl Div<KoalaBear> for PackedKoalaBearNeon {
    type Output = Self;
    #[allow(clippy::suspicious_arithmetic_impl)]
    #[inline]
    fn div(self, rhs: KoalaBear) -> Self {
        self * rhs.inverse()
    }
}

impl Add<PackedKoalaBearNeon> for KoalaBear {
    type Output = PackedKoalaBearNeon;
    #[inline]
    fn add(self, rhs: PackedKoalaBearNeon) -> PackedKoalaBearNeon {
        PackedKoalaBearNeon::from(self) + rhs
    }
}

impl Mul<PackedKoalaBearNeon> for KoalaBear {
    type Output = PackedKoalaBearNeon;
    #[inline]
    fn mul(self, rhs: PackedKoalaBearNeon) -> PackedKoalaBearNeon {
        PackedKoalaBearNeon::from(self) * rhs
    }
}

impl Sub<PackedKoalaBearNeon> for KoalaBear {
    type Output = PackedKoalaBearNeon;
    #[inline]
    fn sub(self, rhs: PackedKoalaBearNeon) -> PackedKoalaBearNeon {
        PackedKoalaBearNeon::from(self) - rhs
    }
}

impl Distribution<PackedKoalaBearNeon> for Standard {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> PackedKoalaBearNeon {
        PackedKoalaBearNeon(rng.gen())
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

unsafe impl PackedValue for PackedKoalaBearNeon {
    type Value = KoalaBear;
    const WIDTH: usize = WIDTH;

    #[inline]
    fn from_slice(slice: &[KoalaBear]) -> &Self {
        assert_eq!(slice.len(), Self::WIDTH);
        unsafe {
            // Safety: `[KoalaBear; WIDTH]` can be transmuted to `PackedKoalaBearNeon` since the
            // latter is `repr(transparent)`. They have the same alignment, so the reference cast is
            // safe too.
            &*slice.as_ptr().cast()
        }
    }
    #[inline]
    fn from_slice_mut(slice: &mut [KoalaBear]) -> &mut Self {
        assert_eq!(slice.len(), Self::WIDTH);
        unsafe {
            // Safety: `[KoalaBear; WIDTH]` can be transmuted to `PackedKoalaBearNeon` since the
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

unsafe impl PackedField for PackedKoalaBearNeon {
    type Scalar = KoalaBear;

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
    use p3_field::AbstractField;
    use p3_field_testing::test_packed_field;

    use super::{KoalaBear, PackedKoalaBearNeon, WIDTH};
    use crate::to_koalabear_array;

    const SPECIAL_VALS: [KoalaBear; WIDTH] =
        to_koalabear_array([0x00000000, 0x00000001, 0x00000002, 0x7f000000]);

    test_packed_field!(
        crate::PackedKoalaBearNeon,
        crate::PackedKoalaBearNeon::zero(),
        crate::PackedKoalaBearNeon(super::SPECIAL_VALS)
    );

    #[test]
    fn test_cube_vs_mul() {
        let vec = PackedKoalaBearNeon(to_koalabear_array([
            0x4efd5eaf, 0x311b8e0c, 0x74dd27c1, 0x449613f0,
        ]));
        let res0 = vec * vec.square();
        let res1 = vec.cube();
        assert_eq!(res0, res1);
    }

    #[test]
    fn test_cube_vs_scalar() {
        let arr = to_koalabear_array([0x57155037, 0x71bdcc8e, 0x301f94d, 0x435938a6]);

        let vec = PackedKoalaBearNeon(arr);
        let vec_res = vec.cube();

        #[allow(clippy::needless_range_loop)]
        for i in 0..WIDTH {
            assert_eq!(vec_res.0[i], arr[i].cube());
        }
    }

    #[test]
    fn test_cube_vs_scalar_special_vals() {
        let vec = PackedKoalaBearNeon(SPECIAL_VALS);
        let vec_res = vec.cube();

        #[allow(clippy::needless_range_loop)]
        for i in 0..WIDTH {
            assert_eq!(vec_res.0[i], SPECIAL_VALS[i].cube());
        }
    }
}
