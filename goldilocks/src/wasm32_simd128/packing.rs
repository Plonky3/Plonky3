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
    i32x4_shuffle, i64x2_add, i64x2_extmul_high_u32x4, i64x2_extmul_low_u32x4, i64x2_gt, i64x2_shl,
    i64x2_shuffle, i64x2_sub, u64x2_shr, u64x2_splat, v128, v128_and, v128_andnot, v128_or,
    v128_xor,
};
use core::fmt::Debug;
use core::iter::{Product, Sum};
use core::mem::transmute;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use p3_field::exponentiation::exp_10540996611094048183;
use p3_field::op_assign_macros::{
    impl_add_assign, impl_add_base_field, impl_div_methods, impl_mul_methods,
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

    /// Add a value whose lane representatives are known to be canonical.
    /// Callers must canonicalize constants before packing them.
    #[inline(always)]
    pub(crate) fn add_canonical(self, rhs: Self) -> Self {
        Self::from_vector(shift(add_no_double_overflow_64_64s_s(
            self.to_vector(),
            shift(rhs.to_vector()),
        )))
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
    fn mul_2exp_u64(&self, mut exp: u64) -> Self {
        exp %= 192;
        match exp {
            0 => *self,
            1 => self.double(),
            2..=32 => Self::from_vector(mul_2exp_small(self.to_vector(), exp as u32)),
            _ => *self * Self::broadcast(Goldilocks::power_of_two(exp)),
        }
    }

    #[inline]
    fn div_2exp_u64(&self, mut exp: u64) -> Self {
        exp %= 192;
        match exp {
            0 => *self,
            1 => self.halve(),
            2..=32 => {
                let x = self.to_vector();
                let lo = v128_and(x, u64x2_splat((1u64 << exp) - 1));
                let hi = u64x2_shr(x, exp as u32);
                let a = i64x2_add(hi, i64x2_shl(lo, (32 - exp) as u32));
                let b = i64x2_shl(lo, (64 - exp) as u32);
                // 2^-exp = 2^(32-exp) - 2^(64-exp) mod P. Both a and b are
                // below P; in particular b <= 2^64 - 2^32, as required by the helper.
                Self::from_vector(shift(sub_small_64s_64_s(shift(a), b)))
            }
            _ => *self * Self::broadcast(Goldilocks::power_of_two(192 - exp)),
        }
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
    fn sum_array<const N: usize>(input: &[Self]) -> Self {
        assert_eq!(N, input.len());
        match N {
            0 => Self::ZERO,
            1 => input[0],
            2 => input[0] + input[1],
            _ => Self::from_vector(sum_delayed_reduce::<N>(input)),
        }
    }

    #[inline]
    fn dot_product<const N: usize>(lhs: &[Self; N], rhs: &[Self; N]) -> Self {
        match N {
            0 => Self::ZERO,
            1 => lhs[0] * rhs[0],
            _ => Self::from_vector(dot_products::<N>(|i| {
                mul64_64(lhs[i].to_vector(), rhs[i].to_vector())
            })),
        }
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
impl Mul<Goldilocks> for PackedGoldilocksWasmSimd128 {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Goldilocks) -> Self {
        let (hi, lo) = mul64_scalar(self.to_vector(), rhs);
        Self::from_vector(reduce128(hi, lo))
    }
}

impl Mul<PackedGoldilocksWasmSimd128> for Goldilocks {
    type Output = PackedGoldilocksWasmSimd128;

    #[inline]
    fn mul(self, rhs: PackedGoldilocksWasmSimd128) -> Self::Output {
        rhs * self
    }
}
impl_div_methods!(PackedGoldilocksWasmSimd128, Goldilocks);
impl_packed_field_div!(PackedGoldilocksWasmSimd128);
impl_sum_prod_base_field!(PackedGoldilocksWasmSimd128, Goldilocks);

impl Algebra<Goldilocks> for PackedGoldilocksWasmSimd128 {
    #[inline]
    fn quadratic_extension_square(a: &[Self; 2], w: Goldilocks) -> [Self; 2] {
        let square1 = a[1].square();
        // The Goldilocks quadratic extension uses the canonical constant 7, so
        // this branch folds away at its call sites. Other coefficients remain valid.
        let weighted_square1 = if w.value == 7 {
            square1.mul_2exp_u64(3) - square1
        } else {
            square1 * w
        };
        [a[0].square() + weighted_square1, (a[0] * a[1]).double()]
    }

    // Vectorized wrappers inherit this constant but not the tail-aware override below.
    const BATCHED_LC_CHUNK: usize = 4;

    #[inline]
    fn batched_linear_combination(values: &[Self], coeffs: &[Goldilocks]) -> Self {
        assert_eq!(values.len(), coeffs.len());
        // Amortize reduction across long inputs; 64 terms are inside dot_products' bound.
        let (values, tail_values) = values.as_chunks::<64>();
        let (coeffs, tail_coeffs) = coeffs.as_chunks::<64>();
        // Seed the accumulator with the tail, avoiding an extra addition for short inputs.
        let mut acc = match tail_values {
            [] => Self::ZERO,
            [value] => *value * tail_coeffs[0],
            _ => Self::from_vector(dot_products_n(tail_values.len(), |i| {
                mul64_scalar(tail_values[i].to_vector(), tail_coeffs[i])
            })),
        };
        for (values, coeffs) in values.iter().zip(coeffs) {
            acc += Self::mixed_dot_product(values, coeffs);
        }
        acc
    }

    #[inline]
    fn mixed_dot_product<const N: usize>(a: &[Self; N], f: &[Goldilocks; N]) -> Self {
        match N {
            0 => Self::ZERO,
            1 => a[0] * f[0],
            _ => Self::from_vector(dot_products::<N>(|i| mul64_scalar(a[i].to_vector(), f[i]))),
        }
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

    mul64_limbs(x_lo, x_hi, y_lo, y_hi)
}

/// Keep the scalar's low/high u32 limbs adjacent while multiplying, then transpose
/// the four products back to packed-field lanes. Broadcasting each u32 limb separately
/// makes LLVM replace widening multiplies with `i64x2.mul`; adjacent unequal limbs
/// preserve `extmul` without optimizer barriers or scalar lane extraction.
#[inline]
fn mul64_scalar(x: v128, y: Goldilocks) -> (v128, v128) {
    let y = u64x2_splat(y.value);
    let y_rev = i32x4_shuffle::<1, 0, 3, 2>(y, y);
    let same0 = i64x2_extmul_low_u32x4(x, y);
    let same1 = i64x2_extmul_high_u32x4(x, y);
    let cross0 = i64x2_extmul_low_u32x4(x, y_rev);
    let cross1 = i64x2_extmul_high_u32x4(x, y_rev);
    let ll = i64x2_shuffle::<0, 2>(same0, same1);
    let hh = i64x2_shuffle::<1, 3>(same0, same1);
    let lh = i64x2_shuffle::<0, 2>(cross0, cross1);
    let hl = i64x2_shuffle::<1, 3>(cross0, cross1);
    combine_products(ll, lh, hl, hh)
}

#[inline]
fn mul64_limbs(x_lo: v128, x_hi: v128, y_lo: v128, y_hi: v128) -> (v128, v128) {
    // Four pairwise 32×32 → 64 products.
    let ll = mul_u32_lanes(x_lo, y_lo); // x_lo * y_lo
    let lh = mul_u32_lanes(x_lo, y_hi); // x_lo * y_hi
    let hl = mul_u32_lanes(x_hi, y_lo);
    let hh = mul_u32_lanes(x_hi, y_hi);
    combine_products(ll, lh, hl, hh)
}

#[inline]
fn combine_products(ll: v128, lh: v128, hl: v128, hh: v128) -> (v128, v128) {
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

/// Return `1` in each lane where the `a + b` addition overflowed, else `0`.
///
/// This bitwise carry formula keeps the comparison vectorized on Wasm SIMD, which has no
/// unsigned 64-bit vector comparison: `((a & b) | ((a | b) & !sum)) >> 63`.
#[inline(always)]
fn unsigned_add_carry(a: v128, b: v128, sum: v128) -> v128 {
    let carry_mask = v128_or(v128_and(a, b), v128_andnot(v128_or(a, b), sum));
    u64x2_shr(carry_mask, 63)
}

/// Delayed-reduction dot product from full `(hi, lo)` products with a single final
/// [`reduce128`] instead of one reduction per term. Mirrors the scalar
/// `Goldilocks::dot_product`'s `N > 2` algorithm (see `goldilocks.rs`), vectorized to 2 lanes.
///
/// Products are produced one at a time by `get` rather than passed as `&[v128; N]`
/// slices: materializing the `N` vectors first via `core::array::from_fn(|i| ...to_vector())`
/// doesn't get elided by the wasm backend, and it costs real time and stack for `N` above a
/// handful.
///
/// Each 128-bit product `val` is split at bit 96 (not bit 64) into `lo96 + hi32 * 2^96`:
/// `hi32 = val >> 96` is bounded by `2^32 - 1` per term, so up to `N <= 2^31` terms can be
/// summed into a single 64-bit-per-lane accumulator (`acc_hi96`) without overflow. The full
/// 128-bit `val` is separately accumulated with wrapping 128-bit-per-lane addition
/// (`acc_lo`); at the end, `acc_lo - (acc_hi96 << 96)` recovers `sum(lo96_i)` exactly modulo
/// `2^128`, because that sum is itself `< 2^127` (`N <= 2^31` terms, each `lo96_i < 2^96`).
/// Finally `2^96 ≡ -1 (mod P)` folds `acc_hi96` back in before the single [`reduce128`] call.
#[inline]
fn dot_products<const N: usize>(get: impl Fn(usize) -> (v128, v128)) -> v128 {
    const {
        assert!((N as u32) <= (1 << 31));
    }

    dot_products_n(N, get)
}

/// Shared accumulator for fixed-size dots and runtime tails of fewer than 64 terms.
/// The caller must ensure n <= 2^31, as required by dot_products.
#[inline]
fn dot_products_n(n: usize, get: impl Fn(usize) -> (v128, v128)) -> v128 {
    let mut acc_lo_hi = u64x2_splat(0);
    let mut acc_lo_lo = u64x2_splat(0);
    let mut acc_hi96 = u64x2_splat(0);

    for i in 0..n {
        let (term_hi, term_lo) = get(i);
        let term_hi96 = u64x2_shr(term_hi, 32);

        let new_lo_lo = i64x2_add(acc_lo_lo, term_lo);
        let carry = unsigned_add_carry(acc_lo_lo, term_lo, new_lo_lo);
        acc_lo_hi = i64x2_add(i64x2_add(acc_lo_hi, term_hi), carry);
        acc_lo_lo = new_lo_lo;

        acc_hi96 = i64x2_add(acc_hi96, term_hi96);
    }

    // `lo = acc_lo - (acc_hi96 << 96)`. The subtrahend's low 64 bits are always 0, so
    // subtracting it never borrows into the low word.
    let hi96_shifted = i64x2_shl(acc_hi96, 32);
    let lo_hi = i64x2_sub(acc_lo_hi, hi96_shifted);
    let lo_lo = acc_lo_lo;

    // `sum = lo + (P - acc_hi96)`, a 128-bit + 64-bit add with carry into the high word.
    let p_minus_hi = i64x2_sub(u64x2_splat(P), acc_hi96);
    let sum_lo = i64x2_add(lo_lo, p_minus_hi);
    let carry2 = unsigned_add_carry(lo_lo, p_minus_hi, sum_lo);
    let sum_hi = i64x2_add(lo_hi, carry2);

    reduce128(sum_hi, sum_lo)
}

/// Delayed-reduction sum of the original packed input, keeping only the wrapped low word and a
/// carry count. Each input is an arbitrary 64-bit value, so the exact sum is
/// `acc_lo + carry_count * 2^64`; on wasm32, `N <= 2^32 - 1` and `carry_count <= N - 1`.
#[inline]
fn sum_delayed_reduce<const N: usize>(terms: &[PackedGoldilocksWasmSimd128]) -> v128 {
    let mut acc_lo = terms[0].to_vector();
    let mut carry_count = u64x2_splat(0);

    for term in &terms[1..] {
        let term = term.to_vector();
        let new_lo = i64x2_add(acc_lo, term);
        carry_count = i64x2_add(carry_count, unsigned_add_carry(acc_lo, term, new_lo));
        acc_lo = new_lo;
    }

    // `2^64 ≡ EPSILON (mod P)`, so the high carry count contributes
    // `correction = (carry_count << 32) - carry_count`. The largest possible count is
    // `2^32 - 2`, making `correction <= (2^32 - 2)(2^32 - 1) < P`; the shift and subtraction
    // therefore cannot wrap. The stronger bound `correction <= 0xffffffff00000000` satisfies
    // `add_small_64s_64_s`'s precondition for an arbitrary pre-shifted `acc_lo`, so one overflow
    // correction is sufficient and no full `reduce128` is needed.
    let correction = i64x2_sub(i64x2_shl(carry_count, 32), carry_count);
    shift(add_small_64s_64_s(shift(acc_lo), correction))
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

/// Fold the bits shifted out of a u64 lane using `2^64 = EPSILON (mod P)`.
/// For `1 <= exp <= 32`, `hi <= EPSILON`, so `hi * EPSILON <= EPSILON^2`.
/// This fits in u64 and satisfies the small-add helper's stronger bound.
#[inline(always)]
fn mul_2exp_small(x: v128, exp: u32) -> v128 {
    let hi = u64x2_shr(x, 64 - exp);
    let lo = i64x2_shl(x, exp);
    let correction = i64x2_sub(i64x2_shl(hi, 32), hi);
    shift(add_small_64s_64_s(shift(lo), correction))
}

#[inline(always)]
fn double(x: v128) -> v128 {
    mul_2exp_small(x, 1)
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

    /// Check the algebra hook and real packed-extension dispatch against u128 modular
    /// arithmetic. Reduce squares before multiplying by w to keep the oracle within u128.
    #[test]
    fn quadratic_square_full_u64_oracle() {
        use p3_field::extension::PackedBinomialExtensionField;
        use p3_field::{Algebra, BasedVectorSpace, PrimeCharacteristicRing, PrimeField64};
        use rand::rngs::SmallRng;
        use rand::{RngExt, SeedableRng};

        type PF = PackedGoldilocksWasmSimd128;
        type EF = PackedBinomialExtensionField<Goldilocks, PF, 2>;

        fn check(raw: [[u64; 2]; 2], w: u64) {
            let a = raw.map(|lanes| PackedGoldilocksWasmSimd128(lanes.map(Goldilocks::new)));
            let actual = PF::quadratic_extension_square(&a, Goldilocks::new(w));
            let p = u128::from(super::P);
            for lane in 0..2 {
                let a0 = u128::from(raw[0][lane]);
                let a1 = u128::from(raw[1][lane]);
                let expected = [
                    ((a0 * a0 % p + (a1 * a1 % p) * u128::from(w) % p) % p) as u64,
                    (2 * (a0 * a1 % p) % p) as u64,
                ];
                let scalar = Goldilocks::quadratic_extension_square(
                    &[a[0].0[lane], a[1].0[lane]],
                    Goldilocks::new(w),
                );
                for i in 0..2 {
                    assert_eq!(actual[i].0[lane].as_canonical_u64(), expected[i]);
                    assert_eq!(scalar[i].as_canonical_u64(), expected[i]);
                }
            }
            if w == 7 {
                let packed_square = EF::new(a).square();
                assert_eq!(
                    <EF as BasedVectorSpace<PF>>::as_basis_coefficients_slice(&packed_square),
                    &actual
                );
            }
        }

        const EDGES: [u64; 10] = [
            0,
            1,
            7,
            (1 << 32) - 1,
            1 << 32,
            1 << 63,
            super::P - 1,
            super::P,
            super::P + 7,
            u64::MAX,
        ];
        for a in EDGES {
            for b in EDGES {
                for w in EDGES {
                    check([[a, b], [b, a]], w);
                }
            }
        }
        let mut rng = SmallRng::seed_from_u64(0x5A0A_2E);
        for _ in 0..512 {
            let raw = rng.random();
            check(raw, 7);
            check(raw, rng.random());
        }
    }

    /// Independent modular arithmetic oracle, including noncanonical representatives.
    #[test]
    fn arithmetic_full_u64_oracle() {
        use p3_field::{PrimeCharacteristicRing, PrimeField64};
        use rand::rngs::SmallRng;
        use rand::{RngExt, SeedableRng};

        fn check(raw: [u64; 2], coefficient: u64) {
            let x = PackedGoldilocksWasmSimd128(raw.map(Goldilocks::new));
            let y = Goldilocks::new(coefficient);
            let product = x * y;
            let mut assigned = x;
            assigned *= y;
            assert_eq!(assigned, product);
            // Preserve assignment support for arbitrary Into<Self> RHS types.
            struct IntoOnly(Goldilocks);
            impl From<IntoOnly> for PackedGoldilocksWasmSimd128 {
                fn from(rhs: IntoOnly) -> Self {
                    Self::from(rhs.0)
                }
            }
            let mut assigned_custom = x;
            assigned_custom *= IntoOnly(y);
            assert_eq!(assigned_custom, product);
            let canonical_y =
                PackedGoldilocksWasmSimd128::from(Goldilocks::new(coefficient % super::P));
            assert_eq!(x.add_canonical(canonical_y), x + canonical_y);
            assert_eq!(product, y * x);
            let (product_hi, product_lo) = super::mul64_scalar(x.to_vector(), y);
            let (square_hi, square_lo) = super::square64(x.to_vector());
            let product_hi: [u64; 2] = unsafe { core::mem::transmute(product_hi) };
            let product_lo: [u64; 2] = unsafe { core::mem::transmute(product_lo) };
            let square_hi: [u64; 2] = unsafe { core::mem::transmute(square_hi) };
            let square_lo: [u64; 2] = unsafe { core::mem::transmute(square_lo) };
            for lane in 0..2 {
                assert_eq!(
                    (u128::from(product_hi[lane]) << 64) | u128::from(product_lo[lane]),
                    u128::from(raw[lane]) * u128::from(coefficient)
                );
                assert_eq!(
                    (u128::from(square_hi[lane]) << 64) | u128::from(square_lo[lane]),
                    u128::from(raw[lane]) * u128::from(raw[lane])
                );
            }
            for lane in 0..2 {
                let expected =
                    (u128::from(raw[lane]) * u128::from(coefficient)) % u128::from(super::P);
                assert_eq!(product.0[lane].as_canonical_u64(), expected as u64);
            }
            for exp in (0..=193).chain([255, 384, u64::MAX]) {
                // Compute 2^exp mod P independently of the field power-of-two helper.
                let mut power = 1u128;
                let mut base = 2u128;
                let mut remaining = exp;
                while remaining != 0 {
                    if remaining & 1 != 0 {
                        power = power * base % u128::from(super::P);
                    }
                    base = base * base % u128::from(super::P);
                    remaining >>= 1;
                }
                let actual = x.mul_2exp_u64(exp);
                for lane in 0..2 {
                    let expected = u128::from(raw[lane]) * power % u128::from(super::P);
                    assert_eq!(
                        actual.0[lane].as_canonical_u64(),
                        expected as u64,
                        "raw={raw:?}, exp={exp}, lane={lane}"
                    );
                }
                if exp == 1 {
                    assert_eq!(actual, x.double());
                }
            }
        }

        const EDGES: [u64; 10] = [
            0,
            1,
            (1 << 32) - 1,
            1 << 32,
            (1 << 63) - 1,
            1 << 63,
            super::P - 1,
            super::P,
            super::P + 1,
            u64::MAX,
        ];
        for &a in &EDGES {
            for &b in &EDGES {
                check([a, b], b);
            }
        }
        let mut rng = SmallRng::seed_from_u64(0xA117_64);
        for _ in 0..128 {
            check(rng.random(), rng.random());
        }
    }

    /// Check the carry helper directly, including both operand orderings and values at the
    /// full `u64` boundary. The delayed sum and dot tests below exercise this helper indirectly,
    /// but their arithmetic can otherwise mask an operand-ordering error in carry detection.
    #[test]
    fn unsigned_add_carry_matches_wrapping_add() {
        use rand::rngs::SmallRng;
        use rand::{RngExt, SeedableRng};

        fn check(a: [u64; 2], b: [u64; 2]) {
            let sum = [a[0].wrapping_add(b[0]), a[1].wrapping_add(b[1])];
            let carry = super::unsigned_add_carry(
                unsafe { core::mem::transmute::<[u64; 2], core::arch::wasm32::v128>(a) },
                unsafe { core::mem::transmute::<[u64; 2], core::arch::wasm32::v128>(b) },
                unsafe { core::mem::transmute::<[u64; 2], core::arch::wasm32::v128>(sum) },
            );
            let carry: [u64; 2] =
                unsafe { core::mem::transmute::<core::arch::wasm32::v128, [u64; 2]>(carry) };
            assert_eq!(carry[0], u64::from(sum[0] < a[0]));
            assert_eq!(carry[1], u64::from(sum[1] < a[1]));
        }

        const EDGES: [u64; 8] = [
            0,
            1,
            2,
            0x7FFF_FFFF_FFFF_FFFF,
            0x8000_0000_0000_0000,
            0xFFFF_FFFF_0000_0000,
            u64::MAX - 1,
            u64::MAX,
        ];
        for &a in &EDGES {
            for &b in &EDGES {
                check([a, b], [b, a]);
                check([b, a], [a, b]);
            }
        }

        let mut rng = SmallRng::seed_from_u64(0x00CA_770F_F1CE);
        for _ in 0..4096 {
            let a = [rng.random(), rng.random()];
            let b = [rng.random(), rng.random()];
            check(a, b);
        }
    }

    /// Adversarial + random coverage for `sum_array`'s delayed-reduction path (`N > 2`),
    /// across every lane independently.
    #[test]
    fn sum_array_delayed_reduction_matches_scalar() {
        use p3_field::{PackedValue, PrimeCharacteristicRing, PrimeField64};
        use rand::rngs::SmallRng;
        use rand::{RngExt, SeedableRng};

        fn check<const N: usize>(terms0: [Goldilocks; N], terms1: [Goldilocks; N]) {
            let packed: [PackedGoldilocksWasmSimd128; N] =
                core::array::from_fn(|i| PackedGoldilocksWasmSimd128([terms0[i], terms1[i]]));

            let expected0 = Goldilocks::sum_array::<N>(&terms0);
            let expected1 = Goldilocks::sum_array::<N>(&terms1);
            let actual = PackedGoldilocksWasmSimd128::sum_array::<N>(&packed);

            assert_eq!(
                actual.as_slice()[0].as_canonical_u64(),
                expected0.as_canonical_u64(),
                "N={N} mismatch at lane 0: terms={terms0:?}"
            );
            assert_eq!(
                actual.as_slice()[1].as_canonical_u64(),
                expected1.as_canonical_u64(),
                "N={N} mismatch at lane 1: terms={terms1:?}"
            );
        }

        // Every term at the maximal non-canonical representative, in lane 0, paired against
        // zero in lane 1: the densest possible carry chain for the wrapping 128-bit
        // accumulator, at every N from 3 (first delayed-reduction arm) to 32.
        macro_rules! check_edge_n {
            ($n:literal) => {
                check::<$n>([Goldilocks::new(u64::MAX); $n], [Goldilocks::ZERO; $n]);
            };
        }
        check::<2>([Goldilocks::new(u64::MAX); 2], [Goldilocks::ZERO; 2]);
        check_edge_n!(3);
        check_edge_n!(4);
        check_edge_n!(5);
        check_edge_n!(7);
        check_edge_n!(8);
        check_edge_n!(11);
        check_edge_n!(12);
        check_edge_n!(15);
        check_edge_n!(16);
        check_edge_n!(32);

        let mut rng = SmallRng::seed_from_u64(0x005A_A0D1_CA7E);
        macro_rules! check_random_n {
            ($n:literal, $count:literal) => {
                for _ in 0..$count {
                    let terms0: [Goldilocks; $n] = core::array::from_fn(|_| rng.random());
                    let terms1: [Goldilocks; $n] = core::array::from_fn(|_| rng.random());
                    check::<$n>(terms0, terms1);
                }
            };
        }
        check_random_n!(3, 32);
        check_random_n!(7, 32);
        check_random_n!(11, 16);
        check_random_n!(15, 16);
        check_random_n!(64, 8);
    }

    /// Compare every sum lane with an independent full-u64 sum modulo the field order. The
    /// scalar Goldilocks sum is deliberately not used as the oracle here because it shares the
    /// delayed-reduction shape that this packed implementation exercises.
    #[test]
    fn sum_array_full_u64_oracle() {
        use p3_field::{PackedValue, PrimeCharacteristicRing, PrimeField64};
        use rand::rngs::SmallRng;
        use rand::{RngExt, SeedableRng};

        fn oracle(values: &[u64]) -> u64 {
            (values.iter().map(|&value| u128::from(value)).sum::<u128>()
                % u128::from(Goldilocks::ORDER_U64)) as u64
        }

        fn check<const N: usize>(terms0: [u64; N], terms1: [u64; N]) {
            let packed: [PackedGoldilocksWasmSimd128; N] = core::array::from_fn(|i| {
                PackedGoldilocksWasmSimd128([
                    Goldilocks::new(terms0[i]),
                    Goldilocks::new(terms1[i]),
                ])
            });
            let actual = PackedGoldilocksWasmSimd128::sum_array::<N>(&packed);

            assert_eq!(actual.as_slice()[0].as_canonical_u64(), oracle(&terms0));
            assert_eq!(actual.as_slice()[1].as_canonical_u64(), oracle(&terms1));
        }

        macro_rules! check_length {
            ($rng:ident, $n:literal, $count:literal) => {{
                check::<$n>([u64::MAX; $n], [0; $n]);
                check::<$n>([0xFFFF_FFFF_0000_0000; $n], [u64::MAX; $n]);
                for _ in 0..$count {
                    let terms0 = core::array::from_fn(|_| $rng.random::<u64>());
                    let terms1 = core::array::from_fn(|_| $rng.random::<u64>());
                    check::<$n>(terms0, terms1);
                }
            }};
        }

        let mut rng = SmallRng::seed_from_u64(0x5A_0B17_5EED);
        check_length!(rng, 0, 16);
        check_length!(rng, 1, 16);
        check_length!(rng, 2, 16);
        check_length!(rng, 3, 16);
        check_length!(rng, 4, 16);
        check_length!(rng, 5, 16);
        check_length!(rng, 6, 16);
        check_length!(rng, 7, 16);
        check_length!(rng, 11, 16);
        check_length!(rng, 15, 16);
        check_length!(rng, 16, 16);
        check_length!(rng, 32, 16);
        check_length!(rng, 64, 8);
        check_length!(rng, 129, 4);
    }

    /// Adversarial + random coverage for `dot_product`'s delayed-reduction path (`N > 1`),
    /// across every lane independently, for `N` both below and (via repeated calls) well
    /// above the width the scalar `match` arms special-case.
    #[test]
    fn dot_product_delayed_reduction_matches_scalar() {
        use p3_field::{PackedValue, PrimeCharacteristicRing, PrimeField64};
        use rand::rngs::SmallRng;
        use rand::{RngExt, SeedableRng};

        const EDGE_VALUES: [u64; 5] = [
            0,
            1,
            Goldilocks::ORDER_U64 - 1,
            0xFFFF_FFFF_0000_0000, // = 2^64 - 2^32, one below the field order
            u64::MAX,              // maximal non-canonical representative
        ];

        /// Checks lane 0 against `(lhs0, rhs0)` and lane 1 against `(lhs1, rhs1)`
        /// independently, so a bug that crosses lanes is caught, not just one that's
        /// uniform across both.
        fn check<const N: usize>(
            lhs0: [Goldilocks; N],
            rhs0: [Goldilocks; N],
            lhs1: [Goldilocks; N],
            rhs1: [Goldilocks; N],
        ) {
            let packed_lhs: [PackedGoldilocksWasmSimd128; N] =
                core::array::from_fn(|i| PackedGoldilocksWasmSimd128([lhs0[i], lhs1[i]]));
            let packed_rhs: [PackedGoldilocksWasmSimd128; N] =
                core::array::from_fn(|i| PackedGoldilocksWasmSimd128([rhs0[i], rhs1[i]]));

            let expected0 = Goldilocks::dot_product(&lhs0, &rhs0);
            let expected1 = Goldilocks::dot_product(&lhs1, &rhs1);
            let actual = PackedGoldilocksWasmSimd128::dot_product(&packed_lhs, &packed_rhs);

            assert_eq!(
                actual.as_slice()[0].as_canonical_u64(),
                expected0.as_canonical_u64(),
                "N={N} mismatch at lane 0: lhs={lhs0:?} rhs={rhs0:?}"
            );
            assert_eq!(
                actual.as_slice()[1].as_canonical_u64(),
                expected1.as_canonical_u64(),
                "N={N} mismatch at lane 1: lhs={lhs1:?} rhs={rhs1:?}"
            );
        }

        // All-maximal-value products in lane 0, all-zero in lane 1, every N from 2 to 32:
        // the densest possible adversarial case for the bit-96 split (every term's top-32-bit
        // contribution is maximal), paired against the opposite extreme in the other lane.
        macro_rules! check_edge_n {
            ($n:literal) => {
                check::<$n>(
                    [Goldilocks::new(u64::MAX); $n],
                    [Goldilocks::new(u64::MAX); $n],
                    [Goldilocks::ZERO; $n],
                    [Goldilocks::new(u64::MAX); $n],
                );
            };
        }
        check_edge_n!(2);
        check_edge_n!(3);
        check_edge_n!(4);
        check_edge_n!(5);
        check_edge_n!(8);
        check_edge_n!(12);
        check_edge_n!(16);
        check_edge_n!(32);

        // Edge-value permutations for small N, same pattern reversed between lanes.
        for &a in &EDGE_VALUES {
            for &b in &EDGE_VALUES {
                for &c in &EDGE_VALUES {
                    check::<3>(
                        [Goldilocks::new(a), Goldilocks::new(b), Goldilocks::new(c)],
                        [Goldilocks::new(c), Goldilocks::new(b), Goldilocks::new(a)],
                        [Goldilocks::new(c), Goldilocks::new(b), Goldilocks::new(a)],
                        [Goldilocks::new(a), Goldilocks::new(b), Goldilocks::new(c)],
                    );
                }
            }
        }

        // Random stress across a range of N, including N well above what a single loop
        // iteration bound might be expected to special-case.
        let mut rng = SmallRng::seed_from_u64(0x00D0_79A0_D7CE);
        macro_rules! check_random_n {
            ($n:literal, $count:literal) => {
                for _ in 0..$count {
                    let lhs0: [Goldilocks; $n] = core::array::from_fn(|_| rng.random());
                    let rhs0: [Goldilocks; $n] = core::array::from_fn(|_| rng.random());
                    let lhs1: [Goldilocks; $n] = core::array::from_fn(|_| rng.random());
                    let rhs1: [Goldilocks; $n] = core::array::from_fn(|_| rng.random());
                    check::<$n>(lhs0, rhs0, lhs1, rhs1);
                }
            };
        }
        check_random_n!(2, 32);
        check_random_n!(3, 32);
        check_random_n!(4, 32);
        check_random_n!(7, 32);
        check_random_n!(16, 16);
        check_random_n!(64, 8);
    }

    /// Check long chunks and their short tails with arbitrary u64 representatives.
    #[test]
    fn batched_linear_combination_full_u64_oracle() {
        use alloc::vec::Vec;

        use p3_field::{Algebra, PrimeField64};
        use rand::rngs::SmallRng;
        use rand::{RngExt, SeedableRng};

        let mut rng = SmallRng::seed_from_u64(0x1C64_7A11);
        for len in [
            0, 1, 2, 3, 4, 7, 8, 16, 31, 32, 63, 64, 65, 67, 68, 127, 128, 129, 255, 256, 257,
        ] {
            for edge in [true, false] {
                let values: Vec<_> = (0..len)
                    .map(|_| {
                        let raw = if edge { [u64::MAX, 0] } else { rng.random() };
                        PackedGoldilocksWasmSimd128(raw.map(Goldilocks::new))
                    })
                    .collect();
                let coeffs: Vec<_> = (0..len)
                    .map(|_| Goldilocks::new(if edge { u64::MAX } else { rng.random() }))
                    .collect();
                let actual =
                    PackedGoldilocksWasmSimd128::batched_linear_combination(&values, &coeffs);
                let p = u128::from(super::P);
                for lane in 0..2 {
                    let expected = values
                        .iter()
                        .zip(&coeffs)
                        .fold(0u128, |acc, (value, coeff)| {
                            (acc + u128::from(value.0[lane].value) * u128::from(coeff.value) % p)
                                % p
                        });
                    assert_eq!(
                        actual.0[lane].as_canonical_u64(),
                        expected as u64,
                        "len={len}, lane={lane}"
                    );
                }
            }
        }
    }

    /// Adversarial coverage for mixed scalar-coefficient products and their shared
    /// delayed reduction, checking each packed lane independently against the scalar dot.
    #[test]
    fn mixed_dot_product_delayed_reduction_matches_scalar() {
        use p3_field::{Algebra, PackedValue, PrimeCharacteristicRing, PrimeField64};
        use rand::rngs::SmallRng;
        use rand::{RngExt, SeedableRng};

        fn check<const N: usize>(a0: [Goldilocks; N], a1: [Goldilocks; N], f: [Goldilocks; N]) {
            let packed_a: [PackedGoldilocksWasmSimd128; N] =
                core::array::from_fn(|i| PackedGoldilocksWasmSimd128([a0[i], a1[i]]));

            let expected0 = Goldilocks::dot_product(&a0, &f);
            let expected1 = Goldilocks::dot_product(&a1, &f);
            let actual = PackedGoldilocksWasmSimd128::mixed_dot_product(&packed_a, &f);

            assert_eq!(
                actual.as_slice()[0].as_canonical_u64(),
                expected0.as_canonical_u64(),
                "N={N} mismatch at lane 0"
            );
            assert_eq!(
                actual.as_slice()[1].as_canonical_u64(),
                expected1.as_canonical_u64(),
                "N={N} mismatch at lane 1"
            );
        }

        macro_rules! check_edge_n {
            ($n:literal) => {
                check::<$n>(
                    [Goldilocks::new(u64::MAX); $n],
                    [Goldilocks::ZERO; $n],
                    [Goldilocks::new(u64::MAX); $n],
                );
            };
        }
        check_edge_n!(2);
        check_edge_n!(5);
        check_edge_n!(8);
        check_edge_n!(16);
        check_edge_n!(32);

        let mut rng = SmallRng::seed_from_u64(0x011E_DD07_9A0D);
        macro_rules! check_random_n {
            ($n:literal, $count:literal) => {
                for _ in 0..$count {
                    let a0: [Goldilocks; $n] = core::array::from_fn(|_| rng.random());
                    let a1: [Goldilocks; $n] = core::array::from_fn(|_| rng.random());
                    let f: [Goldilocks; $n] = core::array::from_fn(|_| rng.random());
                    check::<$n>(a0, a1, f);
                }
            };
        }
        check_random_n!(2, 16);
        check_random_n!(3, 16);
        check_random_n!(8, 16);
        check_random_n!(16, 8);
    }
}
