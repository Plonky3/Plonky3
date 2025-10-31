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
    PermutationMonomial, PrimeCharacteristicRing, impl_packed_field_pow_2, uint32x4_mod_add,
    uint32x4_mod_sub,
};
use p3_util::reconstitute_from_base;
use rand::Rng;
use rand::distr::{Distribution, StandardUniform};

use super::utils::halve_neon;
use crate::{
    BinomialExtensionData, FieldParameters, MontyField31, PackedMontyParameters,
    RelativelyPrimePower,
};

const WIDTH: usize = 4;

pub trait MontyParametersNeon {
    const PACKED_P: uint32x4_t;
    const PACKED_MU: int32x4_t;
}

/// A trait to allow functions to be generic over scalar `MontyField31` and packed `PackedMontyField31Neon`.
trait IntoVec<P: PackedMontyParameters>: Copy {
    /// Convert the value to a NEON vector, broadcasting if it's a scalar.
    fn into_vec(self) -> uint32x4_t;
}

impl<P: PackedMontyParameters> IntoVec<P> for PackedMontyField31Neon<P> {
    #[inline(always)]
    fn into_vec(self) -> uint32x4_t {
        self.to_vector()
    }
}

impl<P: PackedMontyParameters> IntoVec<P> for MontyField31<P> {
    #[inline(always)]
    fn into_vec(self) -> uint32x4_t {
        // Broadcast the scalar value to all lanes of the vector.
        unsafe { aarch64::vdupq_n_u32(self.value) }
    }
}

/// Vectorized NEON implementation of `MontyField31` arithmetic.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(transparent)] // Needed to make `transmute`s safe.
#[must_use]
pub struct PackedMontyField31Neon<PMP: PackedMontyParameters>(pub [MontyField31<PMP>; WIDTH]);

impl<PMP: PackedMontyParameters> PackedMontyField31Neon<PMP> {
    /// Get an arch-specific vector representing the packed values.
    #[inline]
    #[must_use]
    pub(crate) fn to_vector(self) -> uint32x4_t {
        unsafe {
            // Safety: `MontyField31` is `repr(transparent)` so it can be transmuted to `u32`. It
            // follows that `[MontyField31; WIDTH]` can be transmuted to `[u32; WIDTH]`, which can be
            // transmuted to `uint32x4_t`, since arrays are guaranteed to be contiguous in memory.
            // Finally `PackedMontyField31Neon` is `repr(transparent)` so it can be transmuted to
            // `[MontyField31; WIDTH]`.
            transmute(self)
        }
    }

    /// Get an arch-specific vector representing the packed values.
    #[inline]
    #[must_use]
    pub(crate) fn to_signed_vector(self) -> int32x4_t {
        unsafe {
            // Safety: `MontyField31` is `repr(transparent)` so it can be transmuted to `u32` furthermore
            // the u32 is guaranteed to be less than `2^31` so it can be safely reinterpreted as an `i32`. It
            // follows that `[MontyField31; WIDTH]` can be transmuted to `[i32; WIDTH]`, which can be
            // transmuted to `int32x4_t`, since arrays are guaranteed to be contiguous in memory.
            // Finally `PackedMontyField31Neon` is `repr(transparent)` so it can be transmuted to
            // `[MontyField31; WIDTH]`.
            transmute(self)
        }
    }

    /// Make a packed field vector from an arch-specific vector.
    ///
    /// SAFETY: The caller must ensure that each element of `vector` represents a valid `MontyField31`.
    /// In particular, each element of vector must be in `0..P` (canonical form).
    #[inline]
    pub(crate) unsafe fn from_vector(vector: uint32x4_t) -> Self {
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
        let res = uint32x4_mod_add(lhs, rhs, PMP::PACKED_P);
        unsafe {
            // Safety: `uint32x4_mod_add` returns values in canonical form when given values in canonical form.
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
        let res = uint32x4_mod_sub(lhs, rhs, PMP::PACKED_P);
        unsafe {
            // Safety: `uint32x4_mod_sub` returns values in canonical form when given values in canonical form.
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
        let lhs = self.to_signed_vector();
        let rhs = rhs.to_signed_vector();
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
    fn halve(&self) -> Self {
        let val = self.to_vector();
        let halved = halve_neon::<FP>(val);
        unsafe {
            // Safety: `halve_neon` returns values in canonical form when given values in canonical form.
            Self::from_vector(halved)
        }
    }

    #[inline]
    fn cube(&self) -> Self {
        let val = self.to_signed_vector();
        let res = cube::<FP>(val);
        unsafe {
            // Safety: `cube` returns values in canonical form when given values in canonical form.
            Self::from_vector(res)
        }
    }

    #[inline(always)]
    fn dot_product<const N: usize>(u: &[Self; N], v: &[Self; N]) -> Self {
        general_dot_product::<_, _, _, N>(u, v)
    }

    #[inline(always)]
    fn zero_vec(len: usize) -> Vec<Self> {
        // SAFETY: this is a repr(transparent) wrapper around an array.
        unsafe { reconstitute_from_base(MontyField31::<FP>::zero_vec(len * WIDTH)) }
    }

    #[inline(always)]
    fn exp_const_u64<const POWER: u64>(&self) -> Self {
        // We provide specialised code for the powers 3, 5, 7 as these turn up regularly.
        // The other powers could be specialised similarly but we ignore this for now.
        match POWER {
            0 => Self::ONE,
            1 => *self,
            2 => self.square(),
            3 => self.cube(),
            4 => self.square().square(),
            5 => {
                let val = self.to_signed_vector();
                unsafe {
                    // Safety: `exp_5` returns values in canonical form when given values in canonical form.
                    let res = exp_5::<FP>(val);
                    Self::from_vector(res)
                }
            }
            6 => self.square().cube(),
            7 => {
                let val = self.to_signed_vector();
                unsafe {
                    // Safety: `exp_7` returns values in canonical form when given values in canonical form.
                    let res = exp_7::<FP>(val);
                    Self::from_vector(res)
                }
            }
            _ => self.exp_u64(POWER),
        }
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

/// Multiply MontyField31 field elements.
///
/// # Safety
/// Inputs must be signed 32-bit integers in the range [-P, P].
/// Outputs will be a unsigned 32-bit integers in canonical form [0, ..., P).
#[inline]
#[must_use]
fn mul<MPNeon: MontyParametersNeon>(lhs: int32x4_t, rhs: int32x4_t) -> uint32x4_t {
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
        let mu_rhs = mulby_mu::<MPNeon>(rhs);
        let d = mul_with_precomp::<MPNeon, true>(lhs, rhs, mu_rhs);

        // Safe as mul_with_precomp::<MPNeon, true> returns integers in [0, P)
        aarch64::vreinterpretq_u32_s32(d)
    }
}

/// Multiply MontyField31 field elements using precomputation.
///
/// Allows us to reuse `mu_rhs`.
///
/// # Safety
/// Both `lhs` and `rhs` must be signed 32-bit integers in the range [-P, P].
/// `mu_rhs` must be equal to `MPNeon::PACKED_MU * rhs mod 2^32`
///
/// Output will be signed 32-bit integers either in (-P, P) if CANONICAL is set to false
/// or in [0, P) if CANONICAL is set to true.
#[inline]
#[must_use]
fn mul_with_precomp<MPNeon: MontyParametersNeon, const CANONICAL: bool>(
    lhs: int32x4_t,
    rhs: int32x4_t,
    mu_rhs: int32x4_t,
) -> int32x4_t {
    // If CANONICAL:
    //  We want this to compile to:
    //      sqdmulh  c_hi.4s, lhs.4s, rhs.4s
    //      mul      q.4s, lhs.4s, mu_rhs.4s
    //      sqdmulh  qp_hi.4s, q.4s, P.4s
    //      shsub    res.4s, c_hi.4s, qp_hi.4s
    //      cmgt     underflow.4s, qp_hi.4s, c_hi.4s
    //      mls      res.4s, underflow.4s, P.4s
    //
    //      throughput: 1.5 cyc/vec (2.66 els/cyc)
    //      latency: 11 cyc
    //
    // If !CANONICAL:
    //  We want this to compile to:
    //      sqdmulh  c_hi.4s, lhs.4s, rhs.4s
    //      mul      q.4s, lhs.4s, mu_rhs.4s
    //      sqdmulh  qp_hi.4s, q.4s, P.4s
    //      shsub    res.4s, c_hi.4s, qp_hi.4s
    //
    //      throughput: 1 cyc/vec (4 els/cyc)
    //      latency: 8 cyc
    //
    unsafe {
        let c_hi = get_c_hi(lhs, rhs);
        let qp_hi = get_qp_hi::<MPNeon>(lhs, mu_rhs);
        let d = aarch64::vhsubq_s32(c_hi, qp_hi);

        // This branch will be removed by the compiler.
        if CANONICAL {
            // We reduce d to canonical form. d is negative iff `c_hi > qp_hi`, so if that's the
            // case then we add P. Note that if `c_hi > qp_hi` then `underflow` is -1, so we must
            // _subtract_ `underflow` * P.
            let underflow = aarch64::vcltq_s32(c_hi, qp_hi);

            // As underflow and MPNeon::PACKED_P are unsigned we use the unsigned version of multiply
            // and subtract. Note that on bits, the signed and unsigned versions are literally identical.
            let reduced = aarch64::vmlsq_u32(
                aarch64::vreinterpretq_u32_s32(d),
                confuse_compiler(underflow),
                MPNeon::PACKED_P,
            );

            // We convert back to int32x4_t to match the function output.
            aarch64::vreinterpretq_s32_u32(reduced)
        } else {
            d
        }
    }
}

/// Take cube of MontyField31 field elements.
///
/// # Safety
/// Inputs must be signed 32-bit integers in the range [-P, P].
/// Outputs will be a unsigned 32-bit integers in canonical form [0, ..., P).
#[inline]
#[must_use]
fn cube<MPNeon: MontyParametersNeon>(val: int32x4_t) -> uint32x4_t {
    // throughput: 2.75 cyc/vec (1.45 els/cyc)
    // latency: 22 cyc

    unsafe {
        let mu_val = mulby_mu::<MPNeon>(val);

        let val_2 = mul_with_precomp::<MPNeon, false>(val, val, mu_val);
        let val_3 = mul_with_precomp::<MPNeon, true>(val_2, val, mu_val);

        // Safe as mul_with_precomp::<MPNeon, true> returns integers in [0, P)
        aarch64::vreinterpretq_u32_s32(val_3)
    }
}

/// Take the fifth power of the MontyField31 field elements.
///
/// # Safety
/// Inputs must be signed 32-bit integers in the range [-P, P].
/// Outputs will be a unsigned 32-bit integers in canonical form [0, ..., P).
#[inline]
#[must_use]
fn exp_5<MPNeon: MontyParametersNeon>(val: int32x4_t) -> uint32x4_t {
    // throughput: 4 cyc/vec (1 els/cyc)
    // latency: 30 cyc

    unsafe {
        let mu_val = mulby_mu::<MPNeon>(val);

        let val_2 = mul_with_precomp::<MPNeon, false>(val, val, mu_val);

        // mu_val_2 and val_3 can be computed in parallel.
        let mu_val_2 = mulby_mu::<MPNeon>(val_2);
        let val_3 = mul_with_precomp::<MPNeon, false>(val_2, val, mu_val);

        let val_5 = mul_with_precomp::<MPNeon, true>(val_3, val_2, mu_val_2);

        // Safe as mul_with_precomp::<MPNeon, true> returns integers in [0, P)
        aarch64::vreinterpretq_u32_s32(val_5)
    }
}

/// Take the seventh power of the MontyField31 field elements.
///
/// # Safety
/// Inputs must be signed 32-bit integers in the range [-P, P].
/// Outputs will be a unsigned 32-bit integers in canonical form [0, ..., P).
#[inline]
#[must_use]
fn exp_7<MPNeon: MontyParametersNeon>(val: int32x4_t) -> uint32x4_t {
    // throughput: 5.25 cyc/vec (0.76 els/cyc)
    // latency: 33 cyc

    unsafe {
        let mu_val = mulby_mu::<MPNeon>(val);

        let val_2 = mul_with_precomp::<MPNeon, false>(val, val, mu_val);

        // mu_val_2, val_4 and val_3, mu_val_3 can be computed in parallel.
        let mu_val_2 = mulby_mu::<MPNeon>(val_2);
        let val_3 = mul_with_precomp::<MPNeon, false>(val_2, val, mu_val);

        let mu_val_3 = mulby_mu::<MPNeon>(val_3);
        let val_4 = mul_with_precomp::<MPNeon, false>(val_2, val_2, mu_val_2);

        let val_7 = mul_with_precomp::<MPNeon, true>(val_4, val_3, mu_val_3);

        // Safe as mul_with_precomp::<MPNeon, true> returns integers in [0, P)
        aarch64::vreinterpretq_u32_s32(val_7)
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

impl_packed_value!(
    PackedMontyField31Neon,
    MontyField31,
    WIDTH,
    (PackedMontyParameters, PMP)
);

unsafe impl<FP: FieldParameters> PackedField for PackedMontyField31Neon<FP> {
    type Scalar = MontyField31<FP>;

    #[inline]
    fn packed_linear_combination<const N: usize>(coeffs: &[Self::Scalar], vecs: &[Self]) -> Self {
        general_dot_product::<_, _, _, N>(coeffs, vecs)
    }
}

impl_packed_field_pow_2!(
    PackedMontyField31Neon, (FieldParameters, FP);
    [
        (1, interleave_u32),
        (2, interleave_u64),
    ],
    WIDTH
);

/// Compute the elementary function `l0*r0 + l1*r1` given four inputs
/// in canonical form.
///
/// If the inputs are not in canonical form, the result is undefined.
#[inline]
unsafe fn dot_product_2<P, LHS, RHS>(lhs: &[LHS; 2], rhs: &[RHS; 2]) -> PackedMontyField31Neon<P>
where
    P: FieldParameters + MontyParametersNeon,
    LHS: IntoVec<P>,
    RHS: IntoVec<P>,
{
    unsafe {
        // Accumulate the full 64-bit sum C = l0*r0 + l1*r1.

        // Low half (Lanes 0 & 1)
        let mut sum_l = aarch64::vmull_u32(
            aarch64::vget_low_u32(lhs[0].into_vec()),
            aarch64::vget_low_u32(rhs[0].into_vec()),
        );
        sum_l = aarch64::vmlal_u32(
            sum_l,
            aarch64::vget_low_u32(lhs[1].into_vec()),
            aarch64::vget_low_u32(rhs[1].into_vec()),
        );

        // High half (Lanes 2 & 3)
        let mut sum_h = aarch64::vmull_high_u32(lhs[0].into_vec(), rhs[0].into_vec());
        sum_h = aarch64::vmlal_high_u32(sum_h, lhs[1].into_vec(), rhs[1].into_vec());

        // Split C into 32-bit low halves per lane: c_lo = C mod 2^{32}
        let c_lo = aarch64::vuzp1q_u32(
            aarch64::vreinterpretq_u32_u64(sum_l),
            aarch64::vreinterpretq_u32_u64(sum_h),
        );

        // q ≡ c_lo ⋅ μ (mod 2^{32}), with μ = −P^{-1} (mod 2^{32}).
        let q = aarch64::vmulq_u32(c_lo, aarch64::vreinterpretq_u32_s32(P::PACKED_MU));

        // Compute d = (C - q⋅P) / B using multiply-subtract-long instructions.
        //
        // This combines the multiplication q⋅P and subtraction C - q⋅P in one step.
        let d_l = aarch64::vmlsl_u32(
            sum_l,
            aarch64::vget_low_u32(q),
            aarch64::vget_low_u32(P::PACKED_P),
        );
        let d_h = aarch64::vmlsl_high_u32(sum_h, q, P::PACKED_P);

        // Extract the high 32 bits (the division by B = 2^32) from d_l and d_h.
        let d = aarch64::vuzp2q_u32(
            aarch64::vreinterpretq_u32_u64(d_l),
            aarch64::vreinterpretq_u32_u64(d_h),
        );

        // Canonicalize d from (-P, P) to [0, P) branchlessly.
        //
        // The `vmlsq_u32` instruction computes `a - (b * c)`.
        // - If `d` is negative (interpreted as unsigned, it's >= 2^31), the mask is `-1` (all 1s),
        //   so we compute `d - (-1 * P) = d + P`.
        // - If `d` is non-negative, the mask is `0`, so we compute `d - (0 * P) = d`.
        //
        // Check if d >= 2^31 (i.e., negative when interpreted as signed).
        let underflow = aarch64::vcgeq_u32(d, aarch64::vdupq_n_u32(1u32 << 31));
        let canonical_res = aarch64::vmlsq_u32(d, underflow, P::PACKED_P);

        // Safety: The result is now in canonical form [0, P).
        PackedMontyField31Neon::from_vector(canonical_res)
    }
}

/// A general fast dot product implementation using NEON.
#[inline(always)]
fn general_dot_product<P, LHS, RHS, const N: usize>(
    lhs: &[LHS],
    rhs: &[RHS],
) -> PackedMontyField31Neon<P>
where
    P: FieldParameters + MontyParametersNeon,
    LHS: IntoVec<P> + Into<PackedMontyField31Neon<P>>,
    RHS: IntoVec<P> + Into<PackedMontyField31Neon<P>>,
{
    assert_eq!(lhs.len(), N);
    assert_eq!(rhs.len(), N);
    match N {
        0 => PackedMontyField31Neon::<P>::ZERO,
        1 => lhs[0].into() * rhs[0].into(),
        2 => unsafe { dot_product_2(&[lhs[0], lhs[1]], &[rhs[0], rhs[1]]) },
        3 => {
            let lhs_packed = [
                lhs[0].into(),
                lhs[1].into(),
                lhs[2].into(),
                PackedMontyField31Neon::<P>::ZERO,
            ];
            let rhs_packed = [
                rhs[0].into(),
                rhs[1].into(),
                rhs[2].into(),
                PackedMontyField31Neon::<P>::ZERO,
            ];
            unsafe { dot_product_4(&lhs_packed, &rhs_packed) }
        }
        4 => unsafe {
            dot_product_4(
                &[lhs[0], lhs[1], lhs[2], lhs[3]],
                &[rhs[0], rhs[1], rhs[2], rhs[3]],
            )
        },
        64 => {
            let sum_4s: [PackedMontyField31Neon<P>; 16] = core::array::from_fn(|i| {
                let start = i * 4;
                unsafe {
                    dot_product_4(
                        &[lhs[start], lhs[start + 1], lhs[start + 2], lhs[start + 3]],
                        &[rhs[start], rhs[start + 1], rhs[start + 2], rhs[start + 3]],
                    )
                }
            });
            PackedMontyField31Neon::<P>::sum_array::<16>(&sum_4s)
        }
        _ => {
            // Initialize accumulator with the first chunk of 4.
            let mut acc = unsafe {
                dot_product_4(
                    &[lhs[0], lhs[1], lhs[2], lhs[3]],
                    &[rhs[0], rhs[1], rhs[2], rhs[3]],
                )
            };

            // Loop over the rest of the full chunks of 4.
            for i in (4..N).step_by(4) {
                if i + 3 < N {
                    acc += unsafe {
                        dot_product_4(
                            &[lhs[i], lhs[i + 1], lhs[i + 2], lhs[i + 3]],
                            &[rhs[i], rhs[i + 1], rhs[i + 2], rhs[i + 3]],
                        )
                    };
                }
            }

            // Handle the remainder recursively by creating new arrays and calling self.
            match N % 4 {
                0 => acc,
                1 => {
                    let rem_start = N - 1;
                    let lhs_rem: [_; 1] = core::array::from_fn(|i| lhs[rem_start + i]);
                    let rhs_rem: [_; 1] = core::array::from_fn(|i| rhs[rem_start + i]);
                    acc + general_dot_product::<_, _, _, 1>(&lhs_rem, &rhs_rem)
                }
                2 => {
                    let rem_start = N - 2;
                    let lhs_rem: [_; 2] = core::array::from_fn(|i| lhs[rem_start + i]);
                    let rhs_rem: [_; 2] = core::array::from_fn(|i| rhs[rem_start + i]);
                    acc + general_dot_product::<_, _, _, 2>(&lhs_rem, &rhs_rem)
                }
                3 => {
                    let rem_start = N - 3;
                    let lhs_rem: [_; 3] = core::array::from_fn(|i| lhs[rem_start + i]);
                    let rhs_rem: [_; 3] = core::array::from_fn(|i| rhs[rem_start + i]);
                    acc + general_dot_product::<_, _, _, 3>(&lhs_rem, &rhs_rem)
                }
                _ => unreachable!(),
            }
        }
    }
}

/// Compute the elementary function `l0*r0 + l1*r1 + l2*r2 + l3*r3` given eight inputs
/// in canonical form.
///
/// If the inputs are not in canonical form, the result is undefined.
#[inline]
unsafe fn dot_product_4<P, LHS, RHS>(lhs: &[LHS; 4], rhs: &[RHS; 4]) -> PackedMontyField31Neon<P>
where
    P: FieldParameters + MontyParametersNeon,
    LHS: IntoVec<P>,
    RHS: IntoVec<P>,
{
    unsafe {
        // Accumulate the full 64-bit sum C = Σ lhs_i ⋅ rhs_i.

        // Low half (Lanes 0 & 1)
        let mut sum_l = aarch64::vmull_u32(
            aarch64::vget_low_u32(lhs[0].into_vec()),
            aarch64::vget_low_u32(rhs[0].into_vec()),
        );
        sum_l = aarch64::vmlal_u32(
            sum_l,
            aarch64::vget_low_u32(lhs[1].into_vec()),
            aarch64::vget_low_u32(rhs[1].into_vec()),
        );
        sum_l = aarch64::vmlal_u32(
            sum_l,
            aarch64::vget_low_u32(lhs[2].into_vec()),
            aarch64::vget_low_u32(rhs[2].into_vec()),
        );
        sum_l = aarch64::vmlal_u32(
            sum_l,
            aarch64::vget_low_u32(lhs[3].into_vec()),
            aarch64::vget_low_u32(rhs[3].into_vec()),
        );

        // High half (Lanes 2 & 3)
        let mut sum_h = aarch64::vmull_high_u32(lhs[0].into_vec(), rhs[0].into_vec());
        sum_h = aarch64::vmlal_high_u32(sum_h, lhs[1].into_vec(), rhs[1].into_vec());
        sum_h = aarch64::vmlal_high_u32(sum_h, lhs[2].into_vec(), rhs[2].into_vec());
        sum_h = aarch64::vmlal_high_u32(sum_h, lhs[3].into_vec(), rhs[3].into_vec());

        // Split C into 32-bit halves per lane:
        // - c_lo = C mod 2^{32},
        // - c_hi = C >> 32.
        let c_lo = aarch64::vuzp1q_u32(
            aarch64::vreinterpretq_u32_u64(sum_l),
            aarch64::vreinterpretq_u32_u64(sum_h),
        );
        let c_hi = aarch64::vuzp2q_u32(
            aarch64::vreinterpretq_u32_u64(sum_l),
            aarch64::vreinterpretq_u32_u64(sum_h),
        );

        // Since C < 4P^2 and P < 2^{31}, we have c_hi < 2P.
        // We want to compute: c_hi' ∈ [0,P) satisfying c_hi' = c_hi mod P.
        let c_hi_sub = aarch64::vsubq_u32(c_hi, P::PACKED_P);
        let c_hi_prime = aarch64::vminq_u32(c_hi, c_hi_sub);

        // q ≡ c_lo ⋅ μ (mod 2^{32}), with μ = −P^{-1} (mod 2^{32}).
        let q = aarch64::vmulq_u32(c_lo, aarch64::vreinterpretq_u32_s32(P::PACKED_MU));

        // Compute (q⋅P)_hi = high 32 bits of q⋅P per lane (exact unsigned widening multiply).
        let qp_l = aarch64::vmull_u32(aarch64::vget_low_u32(q), aarch64::vget_low_u32(P::PACKED_P));
        let qp_h = aarch64::vmull_high_u32(q, P::PACKED_P);
        let qp_hi = aarch64::vuzp2q_u32(
            aarch64::vreinterpretq_u32_u64(qp_l),
            aarch64::vreinterpretq_u32_u64(qp_h),
        );

        let d = aarch64::vsubq_u32(c_hi_prime, qp_hi);

        // Canonicalize d from (-P, P) to [0, P) branchlessly.
        //
        // The `vmlsq_u32` instruction computes `a - (b * c)`.
        // - If `d` is negative, the mask is `-1` (all 1s), so we compute `d - (-1 * P) = d + P`.
        // - If `d` is non-negative, the mask is `0`, so we compute `d - (0 * P) = d`.
        let underflow = aarch64::vcltq_u32(c_hi_prime, qp_hi);
        let canonical_res = aarch64::vmlsq_u32(d, underflow, P::PACKED_P);

        // Safety: The result is now in canonical form [0, P).
        PackedMontyField31Neon::from_vector(canonical_res)
    }
}

/// Multiplication in a quartic binomial extension field.
#[inline]
pub(crate) fn quartic_mul_packed<FP, const WIDTH: usize>(
    a: &[MontyField31<FP>; WIDTH],
    b: &[MontyField31<FP>; WIDTH],
    res: &mut [MontyField31<FP>; WIDTH],
) where
    FP: FieldParameters + BinomialExtensionData<WIDTH> + MontyParametersNeon,
{
    assert_eq!(WIDTH, 4);

    // Precompute w⋅b once (base-field multiply by the binomial constant).
    let packed_b = PackedMontyField31Neon([b[0], b[1], b[2], b[3]]);
    let w_b = FP::mul_w(packed_b).0;

    // Constant term = a0*b0 + w(a1*b3 + a2*b2 + a3*b1)
    // Linear term = a0*b1 + a1*b0 + w(a2*b3 + a3*b2)
    // Square term = a0*b2 + a1*b1 + a2*b0 + w(a3*b3)
    // Cubic term = a0*b3 + a1*b2 + a2*b1 + a3*b0
    let cols = [
        PackedMontyField31Neon([b[0], b[1], b[2], b[3]]),
        PackedMontyField31Neon([w_b[3], b[0], b[1], b[2]]),
        PackedMontyField31Neon([w_b[2], w_b[3], b[0], b[1]]),
        PackedMontyField31Neon([w_b[1], w_b[2], w_b[3], b[0]]),
    ];

    // Arrange a’s coefficients for the dot product.
    let a_coeffs = [a[0], a[1], a[2], a[3]];

    let result = unsafe { dot_product_4(&a_coeffs, &cols) };

    res.copy_from_slice(&result.0);
}

/// Multiplication in a quintic binomial extension field.
#[inline]
pub(crate) fn quintic_mul_packed<FP, const WIDTH: usize>(
    a: &[MontyField31<FP>; WIDTH],
    b: &[MontyField31<FP>; WIDTH],
    res: &mut [MontyField31<FP>; WIDTH],
) where
    FP: FieldParameters + BinomialExtensionData<WIDTH>,
{
    // TODO: This could be optimised further with a custom NEON implementation.
    assert_eq!(WIDTH, 5);
    let packed_b = PackedMontyField31Neon([b[1], b[2], b[3], b[4]]);
    let w_b = FP::mul_w(packed_b).0;
    let w_b1 = w_b[0];
    let w_b2 = w_b[1];
    let w_b3 = w_b[2];
    let w_b4 = w_b[3];

    // Constant term = a0*b0 + w(a1*b4 + a2*b3 + a3*b2 + a4*b1)
    // Linear term = a0*b1 + a1*b0 + w(a2*b4 + a3*b3 + a4*b2)
    // Square term = a0*b2 + a1*b1 + a2*b0 + w(a3*b4 + a4*b3)
    // Cubic term = a0*b3 + a1*b2 + a2*b1 + a3*b0 + w*a4*b4
    // Quartic term = a0*b4 + a1*b3 + a2*b2 + a3*b1 + a4*b0
    let lhs: [PackedMontyField31Neon<FP>; 5] = [
        a[0].into(),
        a[1].into(),
        a[2].into(),
        a[3].into(),
        a[4].into(),
    ];
    let rhs = [
        PackedMontyField31Neon([b[0], b[1], b[2], b[3]]),
        PackedMontyField31Neon([w_b4, b[0], b[1], b[2]]),
        PackedMontyField31Neon([w_b3, w_b4, b[0], b[1]]),
        PackedMontyField31Neon([w_b2, w_b3, w_b4, b[0]]),
        PackedMontyField31Neon([w_b1, w_b2, w_b3, w_b4]),
    ];

    let dot = PackedMontyField31Neon::dot_product(&lhs, &rhs).0;

    res[..4].copy_from_slice(&dot);
    res[4] =
        MontyField31::dot_product::<5>(a[..].try_into().unwrap(), &[b[4], b[3], b[2], b[1], b[0]]);
}

/// Multiplication in an octic binomial extension field.
#[inline]
pub(crate) fn octic_mul_packed<FP, const WIDTH: usize>(
    a: &[MontyField31<FP>; WIDTH],
    b: &[MontyField31<FP>; WIDTH],
    res: &mut [MontyField31<FP>; WIDTH],
) where
    FP: FieldParameters + BinomialExtensionData<WIDTH>,
{
    // TODO: This could be optimised further with a custom NEON implementation.
    assert_eq!(WIDTH, 8);
    let packed_b_lo = PackedMontyField31Neon([b[0], b[1], b[2], b[3]]);
    let packed_b_hi = PackedMontyField31Neon([b[4], b[5], b[6], b[7]]);
    let w_b_lo = FP::mul_w(packed_b_lo).0;
    let w_b_hi = FP::mul_w(packed_b_hi).0;

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
        PackedMontyField31Neon([w_b_hi[3], b[0], b[1], b[2]]),
        PackedMontyField31Neon([w_b_hi[2], w_b_hi[3], b[0], b[1]]),
        PackedMontyField31Neon([w_b_hi[1], w_b_hi[2], w_b_hi[3], b[0]]),
        PackedMontyField31Neon([w_b_hi[0], w_b_hi[1], w_b_hi[2], w_b_hi[3]]),
        PackedMontyField31Neon([w_b_lo[3], w_b_hi[0], w_b_hi[1], w_b_hi[2]]),
        PackedMontyField31Neon([w_b_lo[2], w_b_lo[3], w_b_hi[0], w_b_hi[1]]),
        PackedMontyField31Neon([w_b_lo[1], w_b_lo[2], w_b_lo[3], w_b_hi[0]]),
    ];
    let rhs_1 = [
        PackedMontyField31Neon([b[4], b[5], b[6], b[7]]),
        PackedMontyField31Neon([b[3], b[4], b[5], b[6]]),
        PackedMontyField31Neon([b[2], b[3], b[4], b[5]]),
        PackedMontyField31Neon([b[1], b[2], b[3], b[4]]),
        PackedMontyField31Neon([b[0], b[1], b[2], b[3]]),
        PackedMontyField31Neon([w_b_hi[3], b[0], b[1], b[2]]),
        PackedMontyField31Neon([w_b_hi[2], w_b_hi[3], b[0], b[1]]),
        PackedMontyField31Neon([w_b_hi[1], w_b_hi[2], w_b_hi[3], b[0]]),
    ];

    let dot_0 = PackedMontyField31Neon::dot_product(&lhs, &rhs_0).0;
    let dot_1 = PackedMontyField31Neon::dot_product(&lhs, &rhs_1).0;

    res[..4].copy_from_slice(&dot_0);
    res[4..].copy_from_slice(&dot_1);
}

/// Multiplication by a base field element in a binomial extension field.
#[inline]
pub(crate) fn base_mul_packed<FP, const WIDTH: usize>(
    a: [MontyField31<FP>; WIDTH],
    b: MontyField31<FP>,
    res: &mut [MontyField31<FP>; WIDTH],
) where
    FP: FieldParameters + BinomialExtensionData<WIDTH>,
{
    match WIDTH {
        1 => res[0] = a[0] * b,
        4 => {
            let lhs = PackedMontyField31Neon([a[0], a[1], a[2], a[3]]);

            let out = lhs * b;

            res.copy_from_slice(&out.0[..4]);
        }
        5 => {
            let lhs = PackedMontyField31Neon([a[0], a[1], a[2], a[3]]);

            let out = lhs * b;
            res[4] = a[4] * b;

            res[..4].copy_from_slice(&out.0[..4]);
        }
        8 => {
            let lhs_lo = PackedMontyField31Neon([a[0], a[1], a[2], a[3]]);
            let lhs_hi = PackedMontyField31Neon([a[4], a[5], a[6], a[7]]);

            let out_lo = lhs_lo * b;
            let out_hi = lhs_hi * b;

            res[..4].copy_from_slice(&out_lo.0);
            res[4..].copy_from_slice(&out_hi.0);
        }
        _ => panic!("Unsupported binomial extension degree: {}", WIDTH),
    }
}

/// Raise MontyField31 field elements to a small constant power `D`.
///
/// Currently, `D` must be one of 3, 5, or 7, if other powers are needed we can easily add them.
///
/// # Safety
/// Inputs must be signed 32-bit integers in the range `[-P, P]`.
/// Outputs will be unsigned 32-bit integers in canonical form `[0, P)`.
#[inline(always)]
#[must_use]
pub(crate) fn exp_small<PMP, const D: u64>(val: int32x4_t) -> uint32x4_t
where
    PMP: PackedMontyParameters + FieldParameters,
{
    match D {
        3 => cube::<PMP>(val),
        5 => exp_5::<PMP>(val),
        7 => exp_7::<PMP>(val),
        _ => panic!("No exp function for given D"),
    }
}
