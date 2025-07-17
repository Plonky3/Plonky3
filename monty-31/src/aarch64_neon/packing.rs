use alloc::vec::Vec;
use core::arch::aarch64::{self, int32x4_t, uint32x4_t};
use core::arch::asm;
use core::hint::unreachable_unchecked;
use core::iter::{Product, Sum};
use core::mem::transmute;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use p3_field::interleave::{interleave_u32, interleave_u64};
use p3_field::op_assign_macros::{
    impl_add_assign, impl_add_base_field, impl_div_methods, impl_mul_base_field, impl_mul_methods,
    impl_packed_value, impl_rng, impl_sub_assign, impl_sub_base_field, impl_sum_prod_base_field,
    ring_sum,
};
use p3_field::{
    Algebra, Field, InjectiveMonomial, PackedField, PackedFieldPow2, PackedValue,
    PermutationMonomial, PrimeCharacteristicRing, impl_packed_field_pow_2,
};
use p3_util::reconstitute_from_base;
use rand::Rng;
use rand::distr::{Distribution, StandardUniform};

use crate::{
    BinomialExtensionData, FieldParameters, MontyField31, PackedMontyParameters,
    RelativelyPrimePower,
};

const WIDTH: usize = 4;

pub trait MontyParametersNeon {
    const PACKED_P: uint32x4_t;
    const PACKED_MU: int32x4_t;
}

/// Vectorized NEON implementation of `MontyField31` arithmetic.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(transparent)] // Needed to make `transmute`s safe.
pub struct PackedMontyField31Neon<PMP: PackedMontyParameters>(pub [MontyField31<PMP>; WIDTH]);

impl<PMP: PackedMontyParameters> PackedMontyField31Neon<PMP> {
    #[inline]
    #[must_use]
    /// Get an arch-specific vector representing the packed values.
    fn to_vector(self) -> uint32x4_t {
        unsafe {
            // Safety: `MontyField31` is `repr(transparent)` so it can be transmuted to `u32`. It
            // follows that `[MontyField31; WIDTH]` can be transmuted to `[u32; WIDTH]`, which can be
            // transmuted to `uint32x4_t`, since arrays are guaranteed to be contiguous in memory.
            // Finally `PackedMontyField31Neon` is `repr(transparent)` so it can be transmuted to
            // `[MontyField31; WIDTH]`.
            transmute(self)
        }
    }

    #[inline]
    #[must_use]
    /// Make a packed field vector from an arch-specific vector.
    ///
    /// SAFETY: The caller must ensure that each element of `vector` represents a valid `MontyField31`.
    /// In particular, each element of vector must be in `0..P` (canonical form).
    unsafe fn from_vector(vector: uint32x4_t) -> Self {
        unsafe {
            // Safety: It is up to the user to ensure that elements of `vector` represent valid
            // `MontyField31` values. We must only reason about memory representations. `uint32x4_t` can be
            // transmuted to `[u32; WIDTH]` (since arrays elements are contiguous in memory), which can
            // be transmuted to `[MontyField31; WIDTH]` (since `MontyField31` is `repr(transparent)`), which in
            // turn can be transmuted to `PackedMontyField31Neon` (since `PackedMontyField31Neon` is also
            // `repr(transparent)`).
            transmute(vector)
        }
    }

    /// Copy `value` to all positions in a packed vector. This is the same as
    /// `From<MontyField31>::from`, but `const`.
    #[inline]
    #[must_use]
    const fn broadcast(value: MontyField31<PMP>) -> Self {
        Self([value; WIDTH])
    }
}

impl<PMP: PackedMontyParameters> From<MontyField31<PMP>> for PackedMontyField31Neon<PMP> {
    #[inline]
    fn from(value: MontyField31<PMP>) -> Self {
        Self::broadcast(value)
    }
}

impl<PMP: PackedMontyParameters> Add for PackedMontyField31Neon<PMP> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        let lhs = self.to_vector();
        let rhs = rhs.to_vector();
        let res = add::<PMP>(lhs, rhs);
        unsafe {
            // Safety: `add` returns values in canonical form when given values in canonical form.
            Self::from_vector(res)
        }
    }
}

impl<PMP: PackedMontyParameters> Sub for PackedMontyField31Neon<PMP> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        let lhs = self.to_vector();
        let rhs = rhs.to_vector();
        let res = sub::<PMP>(lhs, rhs);
        unsafe {
            // Safety: `sub` returns values in canonical form when given values in canonical form.
            Self::from_vector(res)
        }
    }
}

impl<PMP: PackedMontyParameters> Neg for PackedMontyField31Neon<PMP> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        let val = self.to_vector();
        let res = neg::<PMP>(val);
        unsafe {
            // Safety: `neg` returns values in canonical form when given values in canonical form.
            Self::from_vector(res)
        }
    }
}

impl<PMP: PackedMontyParameters> Mul for PackedMontyField31Neon<PMP> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let lhs = self.to_vector();
        let rhs = rhs.to_vector();
        let res = mul::<PMP>(lhs, rhs);
        unsafe {
            // Safety: `mul` returns values in canonical form when given values in canonical form.
            Self::from_vector(res)
        }
    }
}

impl_add_assign!(PackedMontyField31Neon, (PackedMontyParameters, PMP));
impl_sub_assign!(PackedMontyField31Neon, (PackedMontyParameters, PMP));
impl_mul_methods!(PackedMontyField31Neon, (FieldParameters, FP));
ring_sum!(PackedMontyField31Neon, (FieldParameters, FP));
impl_rng!(PackedMontyField31Neon, (PackedMontyParameters, PMP));

impl<FP: FieldParameters> PrimeCharacteristicRing for PackedMontyField31Neon<FP> {
    type PrimeSubfield = MontyField31<FP>;

    const ZERO: Self = Self::broadcast(MontyField31::ZERO);
    const ONE: Self = Self::broadcast(MontyField31::ONE);
    const TWO: Self = Self::broadcast(MontyField31::TWO);
    const NEG_ONE: Self = Self::broadcast(MontyField31::NEG_ONE);

    #[inline]
    fn from_prime_subfield(f: Self::PrimeSubfield) -> Self {
        f.into()
    }

    #[inline]
    fn cube(&self) -> Self {
        let val = self.to_vector();
        let res = cube::<FP>(val);
        unsafe {
            // Safety: `cube` returns values in canonical form when given values in canonical form.
            Self::from_vector(res)
        }
    }

    #[inline(always)]
    fn zero_vec(len: usize) -> Vec<Self> {
        // SAFETY: this is a repr(transparent) wrapper around an array.
        unsafe { reconstitute_from_base(MontyField31::<FP>::zero_vec(len * WIDTH)) }
    }
}

impl_add_base_field!(
    PackedMontyField31Neon,
    MontyField31,
    (PackedMontyParameters, PMP)
);
impl_sub_base_field!(
    PackedMontyField31Neon,
    MontyField31,
    (PackedMontyParameters, PMP)
);
impl_mul_base_field!(
    PackedMontyField31Neon,
    MontyField31,
    (PackedMontyParameters, PMP)
);
impl_div_methods!(PackedMontyField31Neon, MontyField31, (FieldParameters, FP));
impl_sum_prod_base_field!(PackedMontyField31Neon, MontyField31, (FieldParameters, FP));

impl<FP: FieldParameters> Algebra<MontyField31<FP>> for PackedMontyField31Neon<FP> {}

impl<FP: FieldParameters + RelativelyPrimePower<D>, const D: u64> InjectiveMonomial<D>
    for PackedMontyField31Neon<FP>
{
}

impl<FP: FieldParameters + RelativelyPrimePower<D>, const D: u64> PermutationMonomial<D>
    for PackedMontyField31Neon<FP>
{
    fn injective_exp_root_n(&self) -> Self {
        FP::exp_root_d(*self)
    }
}

/// No-op. Prevents the compiler from deducing the value of the vector.
///
/// Similar to `core::hint::black_box`, it can be used to stop the compiler applying undesirable
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

/// Add two vectors of Monty31 field elements in canonical form.
/// If the inputs are not in canonical form, the result is undefined.
#[inline]
#[must_use]
fn add<MPNeon: MontyParametersNeon>(lhs: uint32x4_t, rhs: uint32x4_t) -> uint32x4_t {
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
        let u = aarch64::vsubq_u32(t, MPNeon::PACKED_P);
        aarch64::vminq_u32(t, u)
    }
}

// MONTGOMERY MULTIPLICATION
//   This implementation is based on [1] but with changes. The reduction is as follows:
//
// Constants: P < 2^31
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
fn mulby_mu<MPNeon: MontyParametersNeon>(val: int32x4_t) -> int32x4_t {
    // We want this to compile to:
    //      mul      res.4s, val.4s, MU.4s
    // throughput: .25 cyc/vec (16 els/cyc)
    // latency: 3 cyc

    unsafe { aarch64::vmulq_s32(val, MPNeon::PACKED_MU) }
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
fn get_qp_hi<MPNeon: MontyParametersNeon>(lhs: int32x4_t, mu_rhs: int32x4_t) -> int32x4_t {
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
        aarch64::vqdmulhq_s32(q, aarch64::vreinterpretq_s32_u32(MPNeon::PACKED_P))
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
fn get_reduced_d<MPNeon: MontyParametersNeon>(c_hi: int32x4_t, qp_hi: int32x4_t) -> uint32x4_t {
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
        aarch64::vmlsq_u32(d, confuse_compiler(underflow), MPNeon::PACKED_P)
    }
}

#[inline]
#[must_use]
fn mul<MPNeon: MontyParametersNeon>(lhs: uint32x4_t, rhs: uint32x4_t) -> uint32x4_t {
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

        let mu_rhs = mulby_mu::<MPNeon>(rhs);
        let c_hi = get_c_hi(lhs, rhs);
        let qp_hi = get_qp_hi::<MPNeon>(lhs, mu_rhs);
        get_reduced_d::<MPNeon>(c_hi, qp_hi)
    }
}

#[inline]
#[must_use]
fn cube<MPNeon: MontyParametersNeon>(val: uint32x4_t) -> uint32x4_t {
    // throughput: 2.75 cyc/vec (1.45 els/cyc)
    // latency: 22 cyc

    unsafe {
        let val = aarch64::vreinterpretq_s32_u32(val);
        let mu_val = mulby_mu::<MPNeon>(val);

        let c_hi_2 = get_c_hi(val, val);
        let qp_hi_2 = get_qp_hi::<MPNeon>(val, mu_val);
        let val_2 = get_d(c_hi_2, qp_hi_2);

        let c_hi_3 = get_c_hi(val_2, val);
        let qp_hi_3 = get_qp_hi::<MPNeon>(val_2, mu_val);
        get_reduced_d::<MPNeon>(c_hi_3, qp_hi_3)
    }
}

/// Negate a vector of Monty31 field elements in canonical form.
/// If the inputs are not in canonical form, the result is undefined.
#[inline]
#[must_use]
fn neg<MPNeon: MontyParametersNeon>(val: uint32x4_t) -> uint32x4_t {
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
        let t = aarch64::vsubq_u32(MPNeon::PACKED_P, val);
        let is_zero = aarch64::vceqzq_u32(val);
        aarch64::vbicq_u32(t, is_zero)
    }
}

/// Subtract vectors of Monty31 field elements in canonical form.
/// If the inputs are not in canonical form, the result is undefined.
#[inline]
#[must_use]
fn sub<MPNeon: MontyParametersNeon>(lhs: uint32x4_t, rhs: uint32x4_t) -> uint32x4_t {
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
        aarch64::vmlsq_u32(diff, confuse_compiler(underflow), MPNeon::PACKED_P)
    }
}

impl_packed_value!(
    PackedMontyField31Neon,
    MontyField31,
    WIDTH,
    (PackedMontyParameters, PMP)
);

unsafe impl<FP: FieldParameters> PackedField for PackedMontyField31Neon<FP> {
    type Scalar = MontyField31<FP>;
}

impl_packed_field_pow_2!(
    PackedMontyField31Neon, (FieldParameters, FP);
    [
        (1, interleave_u32),
        (2, interleave_u64),
    ],
    WIDTH
);

/// Multiplication in a quartic binomial extension field.
///
/// TODO: This could likely be optimised further with more effort.
#[inline]
pub fn quartic_mul_packed<FP, const WIDTH: usize>(
    a: &[MontyField31<FP>; WIDTH],
    b: &[MontyField31<FP>; WIDTH],
    res: &mut [MontyField31<FP>; WIDTH],
) where
    FP: FieldParameters + BinomialExtensionData<WIDTH>,
{
    assert_eq!(WIDTH, 4);

    let b_w1 = FP::mul_w(b[1]);
    let b_w2 = FP::mul_w(b[2]);
    let b_w3 = FP::mul_w(b[3]);

    // Constant term = a0*b0 + w(a1*b3 + a2*b2 + a3*b1)
    // Linear term = a0*b1 + a1*b0 + w(a2*b3 + a3*b2)
    // Square term = a0*b2 + a1*b1 + a2*b0 + w(a3*b3)
    // Cubic term = a0*b3 + a1*b2 + a2*b1 + a3*b0
    let lhs: [PackedMontyField31Neon<FP>; 4] = [a[0].into(), a[1].into(), a[2].into(), a[3].into()];
    let rhs = [
        PackedMontyField31Neon([b[0], b[1], b[2], b[3]]),
        PackedMontyField31Neon([b_w3, b[0], b[1], b[2]]),
        PackedMontyField31Neon([b_w2, b_w3, b[0], b[1]]),
        PackedMontyField31Neon([b_w1, b_w2, b_w3, b[0]]),
    ];

    let dot = PackedMontyField31Neon::dot_product(&lhs, &rhs).0;

    res[..].copy_from_slice(&dot);
}

/// Multiplication in a quintic binomial extension field.
///
/// TODO: This could likely be optimised further with more effort.
#[inline]
pub(crate) fn quintic_mul_packed<FP, const WIDTH: usize>(
    a: &[MontyField31<FP>; WIDTH],
    b: &[MontyField31<FP>; WIDTH],
    res: &mut [MontyField31<FP>; WIDTH],
) where
    FP: FieldParameters + BinomialExtensionData<WIDTH>,
{
    assert_eq!(WIDTH, 5);
    let b_w_1 = FP::mul_w(b[1]);
    let b_w_2 = FP::mul_w(b[2]);
    let b_w_3 = FP::mul_w(b[3]);
    let b_w_4 = FP::mul_w(b[4]);

    // Constant term = a0*b0 + w(a1*b4 + a2*b3 + a3*b2 + a4*b1)
    // Linear term = a0*b1 + a1*b0 + w(a2*b4 + a3*b3 + a4*b2)
    // Square term = a0*b2 + a1*b1 + a2*b0 + w(a3*b4 + a4*b3)
    // Cubic term = a0*b3 + a1*b2 + a2*b1 + a3*b0 + w*a4*b4
    // Quartic term = a0*b4 + a1*b3 + a2*b2 + a3*b1 + a4*b0
    let lhs: [PackedMontyField31Neon<FP>; 8] = [
        a[0].into(),
        a[1].into(),
        a[2].into(),
        a[3].into(),
        a[4].into(),
    ];
    let rhs = [
        PackedMontyField31Neon([b[0], b[1], b[2], b[3]]),
        PackedMontyField31Neon([b_w_4, b[0], b[1], b[2]]),
        PackedMontyField31Neon([b_w_3, b_w_4, b[0], b[1]]),
        PackedMontyField31Neon([b_w_2, b_w_3, b_w_4, b[0]]),
        PackedMontyField31Neon([b_w_1, b_w_2, b_w_3, b_w_4]),
    ];

    let dot = PackedMontyField31Neon::dot_product(&lhs, &rhs);

    res[..4].copy_from_slice(&dot);
    res[4] = MontyField31::dot_product(&[a], &[b[4], b[3], b[2], b[1], b[0]]);
}

/// Multiplication in an octic binomial extension field.
///
/// TODO: This could likely be optimised further with more effort.
#[inline]
pub fn octic_mul_packed<FP: FieldParameters, const WIDTH: usize>(
    a: &[MontyField31<FP>; WIDTH],
    b: &[MontyField31<FP>; WIDTH],
    res: &mut [MontyField31<FP>; WIDTH],
) where
    FP: FieldParameters + BinomialExtensionData<WIDTH>,
{
    assert_eq!(WIDTH, 8);
    let packed_b = PackedMontyField31Neon([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]);
    let b_w = FP::mul_w(packed_b).0;

    // Constant coefficient = a0*b0 + w(a1*b7 + ... + a7*b1)
    // Linear coefficient = a0*b1 + a1*b0 + w(a2*b7 + ... + a7*b2)
    // Square coefficient = a0*b2 + .. + a2*b0 + w(a3*b7 + ... + a7*b3)
    // Cube coefficient = a0*b3 + .. + a3*b0 + w(a4*b7 + ... + a7*b4)
    // Quartic coefficient = a0*b4 + ... + a4*b0 + w(a5*b7 + ... + a7*b5)
    // Quintic coefficient = a0*b5 + ... + a5*b0 + w(a6*b7 + ... + a7*b6)
    // Sextic coefficient = a0*b6 + ... + a6*b0 + w*a7*b7
    // Final coefficient = a0*b7 + ... + a7*b0
    let lhs: [PackedMontyField31Neon<FP>; 8] = [
        a[0].into(),
        a[1].into(),
        a[2].into(),
        a[3].into(),
        a[4].into(),
        a[5].into(),
        a[6].into(),
        a[7].into(),
    ];
    let rhs_0 = [
        PackedMontyField31Neon([b[0], b[1], b[2], b[3]]),
        PackedMontyField31Neon([b_w[7], b[0], b[1], b[2]]),
        PackedMontyField31Neon([b_w[6], b_w[7], b[0], b[1]]),
        PackedMontyField31Neon([b_w[5], b_w[6], b_w[7], b[0]]),
        PackedMontyField31Neon([b_w[4], b_w[5], b_w[6], b_w[7]]),
        PackedMontyField31Neon([b_w[3], b_w[4], b_w[5], b_w[6]]),
        PackedMontyField31Neon([b_w[2], b_w[3], b_w[4], b_w[5]]),
        PackedMontyField31Neon([b_w[1], b_w[2], b_w[3], b_w[4]]),
    ];
    let rhs_1 = [
        PackedMontyField31Neon(b[4], b[5], b[6], b[7]),
        PackedMontyField31Neon(b[3], b[4], b[5], b[6]),
        PackedMontyField31Neon(b[2], b[3], b[4], b[5]),
        PackedMontyField31Neon(b[1], b[2], b[3], b[4]),
        PackedMontyField31Neon(b[0], b[1], b[2], b[3]),
        PackedMontyField31Neon(b_w[7], b[0], b[1], b[2]),
        PackedMontyField31Neon(b_w[6], b_w[7], b[0], b[1]),
        PackedMontyField31Neon(b_w[5], b_w[6], b_w[7], b[0]),
    ];

    let dot_0 = PackedMontyField31Neon::dot_product(&lhs, &rhs_0).0;
    let dot_1 = PackedMontyField31Neon::dot_product(&lhs, &rhs_1).0;

    res[..4].copy_from_slice(&dot_0);
    res[4..].copy_from_slice(&dot_1);
}
