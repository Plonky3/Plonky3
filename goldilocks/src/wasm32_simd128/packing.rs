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
    fn sum_array<const N: usize>(input: &[Self]) -> Self {
        assert_eq!(N, input.len());
        const {
            assert!((N as u32) <= (1 << 31));
        }
        match N {
            0 => Self::ZERO,
            1 => input[0],
            2 => input[0] + input[1],
            _ => {
                let vectors: [v128; N] = core::array::from_fn(|i| input[i].to_vector());
                Self::from_vector(sum_delayed_reduce::<N>(&vectors))
            }
        }
    }

    #[inline]
    fn dot_product<const N: usize>(lhs: &[Self; N], rhs: &[Self; N]) -> Self {
        const {
            assert!((N as u32) <= (1 << 31));
        }
        match N {
            0 => Self::ZERO,
            1 => lhs[0] * rhs[0],
            _ => Self::from_vector(dot_product_delayed_reduce::<N>(
                &core::array::from_fn(|i| lhs[i].to_vector()),
                &core::array::from_fn(|i| rhs[i].to_vector()),
            )),
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
impl_mul_base_field!(PackedGoldilocksWasmSimd128, Goldilocks);
impl_div_methods!(PackedGoldilocksWasmSimd128, Goldilocks);
impl_packed_field_div!(PackedGoldilocksWasmSimd128);
impl_sum_prod_base_field!(PackedGoldilocksWasmSimd128, Goldilocks);

impl Algebra<Goldilocks> for PackedGoldilocksWasmSimd128 {
    // Benchmarked across slice lengths 8, 16, 33, 64, 256 under both wasmtime/Cranelift and
    // Node/V8, since the two engines disagree sharply on the best chunk: Cranelift likes
    // chunk=32/64 for aligned lengths (falling back to the unvectorized per-element
    // remainder path in `chunked_linear_combination` otherwise), while V8 prefers chunk=16
    // and regresses badly (2.2-2.3x slower than optimal) at chunk=32/64 regardless of
    // alignment. chunk=4 is the min-max choice: worst-case 1.38x slower than the
    // best-for-that-engine-and-length chunk across all 10 (engine, length) combinations
    // tested, versus 1.57-1.59x for chunk=8/16 and 2.2x+ for chunk=32/64 — always
    // reasonable, never catastrophic, on either engine.
    const BATCHED_LC_CHUNK: usize = 4;

    #[inline]
    fn mixed_dot_product<const N: usize>(a: &[Self; N], f: &[Goldilocks; N]) -> Self {
        const {
            assert!((N as u32) <= (1 << 31));
        }
        match N {
            0 => Self::ZERO,
            1 => a[0] * f[0],
            _ => Self::from_vector(dot_product_delayed_reduce::<N>(
                &core::array::from_fn(|i| a[i].to_vector()),
                &core::array::from_fn(|i| Self::from(f[i]).to_vector()),
            )),
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

/// `1` in each lane where `a < b` (unsigned), else `0`. Used to detect unsigned-add
/// overflow when accumulating 128-bit-per-lane values across `v128` pairs, via the same
/// sign-bit-shift trick as [`canonicalize_s`] and friends.
#[inline(always)]
fn unsigned_lt_as_carry(a: v128, b: v128) -> v128 {
    let mask = i64x2_gt(shift(b), shift(a));
    u64x2_shr(mask, 63)
}

/// Delayed-reduction dot product: `sum(lhs[i] * rhs[i])` with a single final [`reduce128`]
/// instead of one reduction per term. Mirrors the scalar `Goldilocks::dot_product`'s
/// `N > 2` algorithm (see `goldilocks.rs`), vectorized to 2 lanes.
///
/// Each 128-bit product `val` is split at bit 96 (not bit 64) into `lo96 + hi32 * 2^96`:
/// `hi32 = val >> 96` is bounded by `2^32 - 1` per term, so up to `N <= 2^31` terms can be
/// summed into a single 64-bit-per-lane accumulator (`acc_hi96`) without overflow. The full
/// 128-bit `val` is separately accumulated with wrapping 128-bit-per-lane addition
/// (`acc_lo`); at the end, `acc_lo - (acc_hi96 << 96)` recovers `sum(lo96_i)` exactly modulo
/// `2^128`, because that sum is itself `< 2^127` (`N <= 2^31` terms, each `lo96_i < 2^96`).
/// Finally `2^96 ≡ -1 (mod P)` folds `acc_hi96` back in before the single [`reduce128`] call.
#[inline]
fn dot_product_delayed_reduce<const N: usize>(lhs: &[v128; N], rhs: &[v128; N]) -> v128 {
    let mut acc_lo_hi = u64x2_splat(0);
    let mut acc_lo_lo = u64x2_splat(0);
    let mut acc_hi96 = u64x2_splat(0);

    for i in 0..N {
        let (term_hi, term_lo) = mul64_64(lhs[i], rhs[i]);
        let term_hi96 = u64x2_shr(term_hi, 32);

        let new_lo_lo = i64x2_add(acc_lo_lo, term_lo);
        let carry = unsigned_lt_as_carry(new_lo_lo, acc_lo_lo);
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
    let carry2 = unsigned_lt_as_carry(sum_lo, lo_lo);
    let sum_hi = i64x2_add(lo_hi, carry2);

    reduce128(sum_hi, sum_lo)
}

/// Delayed-reduction sum: `sum(terms)` with a single final [`reduce128`] instead of one
/// reduction per term (the generic `sum_array`/`+`-chain default pays a full `add`, ~9 ops
/// including a canonicalize step, for every term).
///
/// Each term is a single (arbitrary, possibly non-canonical) 64-bit value, i.e. a 128-bit
/// value with a zero high half, so — unlike [`dot_product_delayed_reduce`] — no bit-96 split
/// is needed: accumulating with plain wrapping 128-bit-per-lane addition (carry from the low
/// word into the high word on overflow) gives the *exact* sum as long as `N < 2^64`, which
/// always holds. `reduce128` finishes it.
#[inline]
fn sum_delayed_reduce<const N: usize>(terms: &[v128; N]) -> v128 {
    let mut acc_hi = u64x2_splat(0);
    let mut acc_lo = u64x2_splat(0);

    for &term in terms {
        let new_lo = i64x2_add(acc_lo, term);
        let carry = unsigned_lt_as_carry(new_lo, acc_lo);
        acc_hi = i64x2_add(acc_hi, carry);
        acc_lo = new_lo;
    }

    reduce128(acc_hi, acc_lo)
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

        let mut rng = SmallRng::seed_from_u64(0x5A_A0_D1CA_7E);
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
        let mut rng = SmallRng::seed_from_u64(0xD07_9A0D_7CE);
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

    /// Adversarial coverage for `mixed_dot_product`, which reuses the same
    /// `dot_product_delayed_reduce` machinery with the coefficients broadcast per term
    /// instead of genuinely packed — the new risk is specifically in that broadcast wiring.
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

        let mut rng = SmallRng::seed_from_u64(0x11ED_D07_9A0D);
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
