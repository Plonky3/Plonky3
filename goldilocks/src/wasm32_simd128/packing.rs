//! Resources:
//! 1. WebAssembly SIMD proposal: https://github.com/WebAssembly/simd/blob/main/proposals/simd/SIMD.md
//! 2. The arithmetic recipes are the standard Goldilocks SIMD recipes, mimicking the existing
//!    `aarch64_neon` and `x86_64_avx2` backends with the following intrinsic correspondence:
//!
//!      uint64x2_t                 → v128
//!      veorq_u64(a, b)            → v128_xor(a, b)
//!      vaddq_u64(a, b)            → i64x2_add(a, b)
//!      vsubq_u64(a, b)            → i64x2_sub(a, b)
//!      vcgtq_s64(a, b)            → i64x2_gt(a, b)
//!      vbicq_u64(a, b)            → v128_andnot(a, b)  (= a & !b)
//!      vshrq_n_u64::<32>(a)       → u64x2_shr(a, 32)
//!      vdupq_n_u64(x)             → u64x2_splat(x)
//!      vreinterpretq_s64_u64(x)   → identity (v128 is type-erased)

use alloc::vec::Vec;
use core::arch::wasm32::{
    i32x4_shuffle, i64x2_add, i64x2_extmul_low_u32x4, i64x2_gt, i64x2_shl, i64x2_shuffle,
    i64x2_sub, u64x2_shr, u64x2_splat, v128, v128_and, v128_andnot, v128_or, v128_xor,
};
use core::fmt::Debug;
use core::iter::{Product, Sum};
use core::mem::transmute;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use p3_field::exponentiation::exp_10540996611094048183;
use p3_field::op_assign_macros::{
    impl_add_assign, impl_add_base_field, impl_div_methods, impl_mul_base_field, impl_mul_methods,
    impl_packed_field_div, impl_packed_value, impl_rng, impl_sub_assign, impl_sub_base_field,
    impl_sum_prod_base_field, ring_sum,
};
use p3_field::{
    Algebra, Field, InjectiveMonomial, PackedField, PackedFieldPow2, PackedValue,
    PermutationMonomial, PrimeCharacteristicRing, PrimeField64,
};
use p3_util::reconstitute_from_base;
use rand::distr::{Distribution, StandardUniform};
use rand::{Rng, RngExt};

use crate::{Goldilocks, P};

const WIDTH: usize = 2;

/// Equal to `2^32 - 1 = 2^64 mod P`.
const EPSILON: u64 = Goldilocks::ORDER_U64.wrapping_neg();

// Compile-time guard: `PackedGoldilocksWasmSimd128` is only sound to transmute to/from `v128` if
// its byte layout matches. `[Goldilocks; 2]` === `[u64; 2]` === `v128` (16 bytes total).
const _LAYOUT_INVARIANTS: () = {
    assert!(size_of::<[Goldilocks; WIDTH]>() == size_of::<v128>());
    assert!(size_of::<Goldilocks>() == size_of::<u64>());
};

/// Vectorized wasm32-simd128 implementation of `Goldilocks` arithmetic.
///
/// `repr(transparent)` over `[Goldilocks; WIDTH]` so we can `transmute` freely
/// between `[Goldilocks; 2]`, `[u64; 2]`, and `v128`.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
#[repr(transparent)]
#[must_use]
pub struct PackedGoldilocksWasmSimd128(pub [Goldilocks; WIDTH]);

impl PackedGoldilocksWasmSimd128 {
    #[inline]
    #[must_use]
    pub(crate) fn to_vector(self) -> v128 {
        // SAFETY: see `_LAYOUT_INVARIANTS` — byte layout matches.
        unsafe { transmute(self) }
    }

    /// Make a packed field vector from an arch-specific vector.
    ///
    /// Elements of `Goldilocks` are allowed to be arbitrary `u64`s so this function
    /// is safe unlike the `Mersenne31/MontyField31` variants.
    #[inline]
    pub(crate) fn from_vector(vector: v128) -> Self {
        // SAFETY: see `_LAYOUT_INVARIANTS` — byte layout matches.
        unsafe { transmute(vector) }
    }

    /// Copy `value` to all positions in a packed vector. This is the same as
    /// `From<Goldilocks>::from`, but `const`.
    #[inline]
    const fn broadcast(value: Goldilocks) -> Self {
        Self([value; WIDTH])
    }
}

impl From<Goldilocks> for PackedGoldilocksWasmSimd128 {
    fn from(x: Goldilocks) -> Self {
        Self::broadcast(x)
    }
}

impl Add for PackedGoldilocksWasmSimd128 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::from_vector(add(self.to_vector(), rhs.to_vector()))
    }
}

impl Sub for PackedGoldilocksWasmSimd128 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::from_vector(sub(self.to_vector(), rhs.to_vector()))
    }
}

impl Neg for PackedGoldilocksWasmSimd128 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self::from_vector(neg(self.to_vector()))
    }
}

impl Mul for PackedGoldilocksWasmSimd128 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self::from_vector(mul(self.to_vector(), rhs.to_vector()))
    }
}

impl_add_assign!(PackedGoldilocksWasmSimd128);
impl_sub_assign!(PackedGoldilocksWasmSimd128);
impl_mul_methods!(PackedGoldilocksWasmSimd128);
ring_sum!(PackedGoldilocksWasmSimd128);
impl_rng!(PackedGoldilocksWasmSimd128);

impl PrimeCharacteristicRing for PackedGoldilocksWasmSimd128 {
    type PrimeSubfield = Goldilocks;

    const ZERO: Self = Self::broadcast(Goldilocks::ZERO);
    const ONE: Self = Self::broadcast(Goldilocks::ONE);
    const TWO: Self = Self::broadcast(Goldilocks::TWO);
    const NEG_ONE: Self = Self::broadcast(Goldilocks::NEG_ONE);

    #[inline]
    fn from_prime_subfield(f: Self::PrimeSubfield) -> Self {
        f.into()
    }

    #[inline]
    fn halve(&self) -> Self {
        Self::from_vector(halve(self.to_vector()))
    }

    #[inline]
    fn double(&self) -> Self {
        Self::from_vector(double(self.to_vector()))
    }

    #[inline]
    fn square(&self) -> Self {
        Self::from_vector(square(self.to_vector()))
    }

    #[inline]
    fn zero_vec(len: usize) -> Vec<Self> {
        // SAFETY: this is a repr(transparent) wrapper around an array.
        unsafe { reconstitute_from_base(Goldilocks::zero_vec(len * WIDTH)) }
    }

    #[inline]
    fn dot_product<const N: usize>(lhs: &[Self; N], rhs: &[Self; N]) -> Self {
        Self::from_fn(|lane| {
            let lhs_lane: [Goldilocks; N] = core::array::from_fn(|i| lhs[i].as_slice()[lane]);
            let rhs_lane: [Goldilocks; N] = core::array::from_fn(|i| rhs[i].as_slice()[lane]);
            Goldilocks::dot_product(&lhs_lane, &rhs_lane)
        })
    }
}

impl InjectiveMonomial<7> for PackedGoldilocksWasmSimd128 {}

impl PermutationMonomial<7> for PackedGoldilocksWasmSimd128 {
    /// In the field `Goldilocks`, `a^{1/7}` is equal to a^{10540996611094048183}.
    ///
    /// This follows from the calculation `7*10540996611094048183 = 4*(2^64 - 2**32) + 1 = 1 mod (p - 1)`.
    fn injective_exp_root_n(&self) -> Self {
        exp_10540996611094048183(*self)
    }
}

impl_add_base_field!(PackedGoldilocksWasmSimd128, Goldilocks);
impl_sub_base_field!(PackedGoldilocksWasmSimd128, Goldilocks);
impl_mul_base_field!(PackedGoldilocksWasmSimd128, Goldilocks);
impl_div_methods!(PackedGoldilocksWasmSimd128, Goldilocks);
impl_packed_field_div!(PackedGoldilocksWasmSimd128);
impl_sum_prod_base_field!(PackedGoldilocksWasmSimd128, Goldilocks);

impl Algebra<Goldilocks> for PackedGoldilocksWasmSimd128 {
    // Matches the aarch64 NEON chunk for the same WIDTH=2 lane layout.
    const BATCHED_LC_CHUNK: usize = 2;

    #[inline]
    fn mixed_dot_product<const N: usize>(a: &[Self; N], f: &[Goldilocks; N]) -> Self {
        Self::from_fn(|lane| {
            let a_lane: [Goldilocks; N] = core::array::from_fn(|i| a[i].as_slice()[lane]);
            Goldilocks::dot_product(&a_lane, f)
        })
    }
}

impl_packed_value!(PackedGoldilocksWasmSimd128, Goldilocks, WIDTH);

unsafe impl PackedField for PackedGoldilocksWasmSimd128 {
    type Scalar = Goldilocks;
}

/// Interleave two `u64x2` vectors at the element level.
/// For `block_len = 1`: `[a0, a1] x [b0, b1] -> ([a0, b0], [a1, b1])`.
#[inline]
pub fn interleave_u64(v0: v128, v1: v128) -> (v128, v128) {
    // `i64x2_shuffle::<I0, I1>(a, b)` selects lanes from `concat(a; b)`, where 0,1 are
    // lanes of `a` and 2,3 are lanes of `b`.
    let r0 = i64x2_shuffle::<0, 2>(v0, v1);
    let r1 = i64x2_shuffle::<1, 3>(v0, v1);
    (r0, r1)
}

unsafe impl PackedFieldPow2 for PackedGoldilocksWasmSimd128 {
    fn interleave(&self, other: Self, block_len: usize) -> (Self, Self) {
        let (v0, v1) = (self.to_vector(), other.to_vector());
        let (res0, res1) = match block_len {
            1 => interleave_u64(v0, v1),
            2 => (v0, v1),
            _ => panic!("unsupported block length"),
        };
        (Self::from_vector(res0), Self::from_vector(res1))
    }
}

const SIGN_BIT: v128 =
    unsafe { transmute::<[u64; WIDTH], v128>([0x8000_0000_0000_0000u64; WIDTH]) };
const SHIFTED_FIELD_ORDER: v128 = unsafe {
    transmute::<[u64; WIDTH], v128>([Goldilocks::ORDER_U64 ^ 0x8000_0000_0000_0000u64; WIDTH])
};
const EPSILON_VEC: v128 = unsafe { transmute::<[u64; WIDTH], v128>([EPSILON; WIDTH]) };

/// Add `2^63` with overflow. Needed to emulate unsigned comparisons.
#[inline(always)]
fn shift(x: v128) -> v128 {
    v128_xor(x, SIGN_BIT)
}

// If `x_s < SHIFTED_FIELD_ORDER` (signed comparison), add `EPSILON` to canonicalize.
// The neon impl uses `vbicq_u64(EPSILON_VEC, mask) = EPSILON_VEC & !mask`. wasm32's
// `v128_andnot(a, b) = a & !b` matches.
#[inline(always)]
fn canonicalize_s(x_s: v128) -> v128 {
    let mask = i64x2_gt(SHIFTED_FIELD_ORDER, x_s);
    let wrapback_amt = v128_andnot(EPSILON_VEC, mask);
    i64x2_add(x_s, wrapback_amt)
}

/// Addition `u64 + u64 -> u64`. Assumes that `x + y < 2^64 + FIELD_ORDER`. The second
/// argument is pre-shifted by `1 << 63`. The result is similarly shifted.
#[inline(always)]
fn add_no_double_overflow_64_64s_s(x: v128, y_s: v128) -> v128 {
    let res_wrapped_s = i64x2_add(x, y_s);
    // Overflow detected: `y_s > res_wrapped_s` (signed). On overflow, add `EPSILON`.
    let mask = i64x2_gt(y_s, res_wrapped_s);
    let wrapback_amt = u64x2_shr(mask, 32);
    i64x2_add(res_wrapped_s, wrapback_amt)
}

/// Goldilocks modular addition. Computes `x + y mod FIELD_ORDER`.
///
/// Inputs can be arbitrary, output is not guaranteed to be less than `FIELD_ORDER`.
#[inline]
fn add(x: v128, y: v128) -> v128 {
    let y_s = shift(y);
    let res_s = add_no_double_overflow_64_64s_s(x, canonicalize_s(y_s));
    shift(res_s)
}

/// Goldilocks modular subtraction. Computes `x - y mod FIELD_ORDER`.
///
/// Inputs can be arbitrary, output is not guaranteed to be less than `FIELD_ORDER`.
#[inline]
fn sub(x: v128, y: v128) -> v128 {
    let y_s = canonicalize_s(shift(y));
    let x_s = shift(x);
    let mask = i64x2_gt(y_s, x_s);
    let wrapback_amt = u64x2_shr(mask, 32);
    let res_wrapped = i64x2_sub(x_s, y_s);
    i64x2_sub(res_wrapped, wrapback_amt)
}

/// Goldilocks modular negation. Computes `-x mod FIELD_ORDER`.
///
/// Input can be arbitrary, output is not guaranteed to be less than `FIELD_ORDER`.
#[inline]
fn neg(y: v128) -> v128 {
    let y_s = shift(y);
    i64x2_sub(SHIFTED_FIELD_ORDER, canonicalize_s(y_s))
}

/// Halve a vector of Goldilocks field elements.
#[inline(always)]
pub(crate) fn halve(input: v128) -> v128 {
    let one = u64x2_splat(1);
    let zero = u64x2_splat(0);
    let half_v = u64x2_splat(P.div_ceil(2));
    let least_bit = v128_and(input, one);
    let t = u64x2_shr(input, 1);
    // `neg_least_bit` is 0 or -1 (all bits set within each lane).
    let neg_least_bit = i64x2_sub(zero, least_bit);
    let maybe_half = v128_and(half_v, neg_least_bit);
    i64x2_add(t, maybe_half)
}

// ============================================================================
// Multiplication: schoolbook 64×64 → 128 + Goldilocks reduction.
// ============================================================================

/// Pack the low 32 bits of each `u64` lane into `u32` lanes 0 and 1.
/// Input  `u32x4` view: `[a0_lo, a0_hi, a1_lo, a1_hi]`.
/// Output `u32x4` view: `[a0_lo, a1_lo,    *,     *]`.
#[inline(always)]
fn lo32(a: v128) -> v128 {
    i32x4_shuffle::<0, 2, 0, 0>(a, a)
}

/// Pack the high 32 bits of each `u64` lane into `u32` lanes 0 and 1.
/// Input  `u32x4` view: `[a0_lo, a0_hi, a1_lo, a1_hi]`.
/// Output `u32x4` view: `[a0_hi, a1_hi,    *,     *]`.
#[inline(always)]
fn hi32(a: v128) -> v128 {
    i32x4_shuffle::<1, 3, 0, 0>(a, a)
}

/// 32×32 → 64-bit unsigned multiply, lane-aligned.
#[inline(always)]
fn mul_u32_lanes(a_packed: v128, b_packed: v128) -> v128 {
    i64x2_extmul_low_u32x4(a_packed, b_packed)
}

/// Full 64×64 → 128 multiply per lane. Returns `(hi, lo)` where the 128-bit product
/// per lane equals `lo + hi * 2^64`. Translation of the AVX2 `mul64_64`.
#[inline]
fn mul64_64(x: v128, y: v128) -> (v128, v128) {
    let x_lo = lo32(x);
    let x_hi = hi32(x);
    let y_lo = lo32(y);
    let y_hi = hi32(y);

    // Four pairwise 32×32 → 64 products.
    let ll = mul_u32_lanes(x_lo, y_lo); // x_lo * y_lo
    let lh = mul_u32_lanes(x_lo, y_hi); // x_lo * y_hi
    let hl = mul_u32_lanes(x_hi, y_lo);
    let hh = mul_u32_lanes(x_hi, y_hi);

    // Bignum addition (AVX2 algorithm verbatim):
    //   t0 = hl + (ll >> 32)              (no overflow: ≤ (2^32-1)^2 + (2^32-1) < 2^64)
    //   t1 = lh + (t0 & 0xFFFFFFFF)       (no overflow)
    //   t2 = hh + (t0 >> 32)              (no overflow)
    //   res_hi = t2 + (t1 >> 32)          (no overflow)
    //   res_lo = (ll & 0xFFFFFFFF) | ((t1 & 0xFFFFFFFF) << 32)
    let ll_hi = u64x2_shr(ll, 32);
    let t0 = i64x2_add(hl, ll_hi);
    let t0_lo = v128_and(t0, EPSILON_VEC);
    let t0_hi = u64x2_shr(t0, 32);
    let t1 = i64x2_add(lh, t0_lo);
    let t2 = i64x2_add(hh, t0_hi);
    let t1_hi = u64x2_shr(t1, 32);
    let res_hi = i64x2_add(t2, t1_hi);

    let ll_lo32 = v128_and(ll, EPSILON_VEC);
    let t1_lo32 = v128_and(t1, EPSILON_VEC);
    let t1_shifted = i64x2_shl(t1_lo32, 32);
    let res_lo = v128_or(ll_lo32, t1_shifted);

    (res_hi, res_lo)
}

/// Goldilocks addition of a "small" number. `x_s` is pre-shifted by `2^63`. `y` is
/// assumed to be `<= 2^64 - 2^32 = 0xffffffff00000000`. The result is shifted by `2^63`.
#[inline(always)]
fn add_small_64s_64_s(x_s: v128, y: v128) -> v128 {
    let res_wrapped_s = i64x2_add(x_s, y);
    let mask = i64x2_gt(x_s, res_wrapped_s); // -1 if overflow
    let wrapback_amt = u64x2_shr(mask, 32); // 0xFFFFFFFF if overflow else 0
    i64x2_add(res_wrapped_s, wrapback_amt)
}

/// Goldilocks subtraction of a "small" number. `x_s` is pre-shifted by `2^63`. `y` is
/// assumed to be `<= 0xffffffff00000000`. The result is shifted by `2^63`.
#[inline(always)]
fn sub_small_64s_64_s(x_s: v128, y: v128) -> v128 {
    let res_wrapped_s = i64x2_sub(x_s, y);
    let mask = i64x2_gt(res_wrapped_s, x_s); // -1 if underflow
    let wrapback_amt = u64x2_shr(mask, 32);
    i64x2_sub(res_wrapped_s, wrapback_amt)
}

/// Given a 128-bit value `(hi, lo)`, reduce it modulo the Goldilocks field order.
///
/// The result will be a 64-bit value but may be larger than `FIELD_ORDER`. Uses
/// `2^64 ≡ 2^32 - 1 (mod p)` and `2^96 ≡ -1 (mod p)`.
#[inline]
fn reduce128(hi: v128, lo: v128) -> v128 {
    let lo_s = shift(lo);
    // `2^96 ≡ -1`, so the contribution of `hi_hi * 2^96` is `-hi_hi`.
    let hi_hi = u64x2_shr(hi, 32);
    let lo1_s = sub_small_64s_64_s(lo_s, hi_hi);

    // `hi_lo32 * EPSILON` where `EPSILON = 2^32 - 1`.
    // Computed as `(hi_lo32 << 32) - hi_lo32`, avoiding a full multiply.
    // `hi_lo32 <= 2^32 - 1`, so `(hi_lo32 << 32) <= 2^64 - 2^32`, no overflow.
    let hi_lo32 = v128_and(hi, EPSILON_VEC);
    let hi_lo32_shifted = i64x2_shl(hi_lo32, 32);
    let t1 = i64x2_sub(hi_lo32_shifted, hi_lo32);

    // Result is at most `(2^32 - 1)^2 < 2^64`, so `add_small_64s_64_s` applies.
    let lo2_s = add_small_64s_64_s(lo1_s, t1);
    shift(lo2_s)
}

/// Goldilocks modular multiplication. Computes `x * y mod FIELD_ORDER`.
///
/// Inputs can be arbitrary, output is not guaranteed to be less than `FIELD_ORDER`.
#[inline]
fn mul(x: v128, y: v128) -> v128 {
    let (hi, lo) = mul64_64(x, y);
    reduce128(hi, lo)
}

/// Full 64×64 → 128 squaring.
/// Exploits `lh = hl` so only three 32×32 products are needed instead of four.
#[inline]
fn square64(x: v128) -> (v128, v128) {
    let x_lo = lo32(x);
    let x_hi = hi32(x);
    let ll = mul_u32_lanes(x_lo, x_lo);
    let lh = mul_u32_lanes(x_lo, x_hi);
    let hh = mul_u32_lanes(x_hi, x_hi);
    // 128-bit product = ll + lh·2^33 + hh·2^64.
    let ll_hi = u64x2_shr(ll, 33);
    let t0 = i64x2_add(lh, ll_hi);
    let t0_hi = u64x2_shr(t0, 31);
    let res_hi = i64x2_add(hh, t0_hi);
    let lh_shifted = i64x2_shl(lh, 33);
    let res_lo = i64x2_add(ll, lh_shifted);
    (res_hi, res_lo)
}

#[inline]
fn square(x: v128) -> v128 {
    let (hi, lo) = square64(x);
    reduce128(hi, lo)
}

/// Goldilocks modular doubling, falls back to `add`.
#[inline(always)]
fn double(x: v128) -> v128 {
    add(x, x)
}

#[cfg(test)]
mod tests {
    use p3_field_testing::test_packed_field;

    use super::{Goldilocks, PackedGoldilocksWasmSimd128, WIDTH};

    const SPECIAL_VALS: [Goldilocks; WIDTH] =
        Goldilocks::new_array([0xFFFF_FFFF_0000_0000, 0xFFFF_FFFF_FFFF_FFFF]);

    const ZEROS: PackedGoldilocksWasmSimd128 =
        PackedGoldilocksWasmSimd128(Goldilocks::new_array([
            0x0000_0000_0000_0000,
            0xFFFF_FFFF_0000_0001, // = P, canonicalizes to 0
        ]));

    const ONES: PackedGoldilocksWasmSimd128 = PackedGoldilocksWasmSimd128(Goldilocks::new_array([
        0x0000_0000_0000_0001,
        0xFFFF_FFFF_0000_0002, // = P + 1, canonicalizes to 1
    ]));

    test_packed_field!(
        crate::PackedGoldilocksWasmSimd128,
        &[super::ZEROS],
        &[super::ONES],
        crate::PackedGoldilocksWasmSimd128(super::SPECIAL_VALS)
    );
}
