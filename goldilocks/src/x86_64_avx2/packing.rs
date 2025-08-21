use alloc::vec::Vec;
use core::arch::x86_64::*;
use core::fmt::Debug;
use core::iter::{Product, Sum};
use core::mem::transmute;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use p3_field::exponentiation::exp_10540996611094048183;
use p3_field::interleave::{interleave_u64, interleave_u128};
use p3_field::op_assign_macros::{
    impl_add_assign, impl_add_base_field, impl_div_methods, impl_mul_base_field, impl_mul_methods,
    impl_packed_value, impl_rng, impl_sub_assign, impl_sub_base_field, impl_sum_prod_base_field,
    ring_sum,
};
use p3_field::{
    Algebra, Field, InjectiveMonomial, PackedField, PackedFieldPow2, PackedValue,
    PermutationMonomial, PrimeCharacteristicRing, PrimeField64, impl_packed_field_pow_2,
};
use p3_util::reconstitute_from_base;
use rand::Rng;
use rand::distr::{Distribution, StandardUniform};

use crate::{Goldilocks, P};

const WIDTH: usize = 4;

/// Vectorized AVX2 implementation of `Goldilocks` arithmetic.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
#[repr(transparent)] // Needed to make `transmute`s safe.
#[must_use]
pub struct PackedGoldilocksAVX2(pub [Goldilocks; WIDTH]);

impl PackedGoldilocksAVX2 {
    /// Get an arch-specific vector representing the packed values.
    #[inline]
    #[must_use]
    pub(crate) fn to_vector(self) -> __m256i {
        unsafe {
            // Safety: `Goldilocks` is `repr(transparent)` so it can be transmuted to `u64`. It
            // follows that `[Goldilocks; WIDTH]` can be transmuted to `[u64; WIDTH]`, which can be
            // transmuted to `__m256i`, since arrays are guaranteed to be contiguous in memory.
            // Finally `PackedGoldilocksAVX2` is `repr(transparent)` so it can be transmuted to
            // `[Goldilocks; WIDTH]`.
            transmute(self)
        }
    }

    /// Make a packed field vector from an arch-specific vector.
    ///
    /// Elements of `Goldilocks` are allowed to be arbitrary u64s so this function
    /// is safe unlike the `Mersenne31/MontyField31` variants.
    #[inline]
    pub(crate) fn from_vector(vector: __m256i) -> Self {
        unsafe {
            // Safety: `__m256i` can be transmuted to `[u64; WIDTH]` (since arrays elements are
            // contiguous in memory), which can be transmuted to `[Goldilocks; WIDTH]` (since
            // `Goldilocks` is `repr(transparent)`), which in turn can be transmuted to
            // `PackedGoldilocksAVX2` (since `PackedGoldilocksAVX2` is also `repr(transparent)`).
            transmute(vector)
        }
    }

    /// Copy `value` to all positions in a packed vector. This is the same as
    /// `From<Goldilocks>::from`, but `const`.
    #[inline]
    const fn broadcast(value: Goldilocks) -> Self {
        Self([value; WIDTH])
    }
}

impl From<Goldilocks> for PackedGoldilocksAVX2 {
    fn from(x: Goldilocks) -> Self {
        Self::broadcast(x)
    }
}

impl Add for PackedGoldilocksAVX2 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::from_vector(add(self.to_vector(), rhs.to_vector()))
    }
}

impl Sub for PackedGoldilocksAVX2 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::from_vector(sub(self.to_vector(), rhs.to_vector()))
    }
}

impl Neg for PackedGoldilocksAVX2 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self::from_vector(neg(self.to_vector()))
    }
}

impl Mul for PackedGoldilocksAVX2 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self::from_vector(mul(self.to_vector(), rhs.to_vector()))
    }
}

impl_add_assign!(PackedGoldilocksAVX2);
impl_sub_assign!(PackedGoldilocksAVX2);
impl_mul_methods!(PackedGoldilocksAVX2);
ring_sum!(PackedGoldilocksAVX2);
impl_rng!(PackedGoldilocksAVX2);

impl PrimeCharacteristicRing for PackedGoldilocksAVX2 {
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
    fn square(&self) -> Self {
        Self::from_vector(square(self.to_vector()))
    }

    #[inline]
    fn zero_vec(len: usize) -> Vec<Self> {
        // SAFETY: this is a repr(transparent) wrapper around an array.
        unsafe { reconstitute_from_base(Goldilocks::zero_vec(len * WIDTH)) }
    }
}

// Degree of the smallest permutation polynomial for Goldilocks.
//
// As p - 1 = 2^32 * 3 * 5 * 17 * ... the smallest choice for a degree D satisfying gcd(p - 1, D) = 1 is 7.
impl InjectiveMonomial<7> for PackedGoldilocksAVX2 {}

impl PermutationMonomial<7> for PackedGoldilocksAVX2 {
    /// In the field `Goldilocks`, `a^{1/7}` is equal to a^{10540996611094048183}.
    ///
    /// This follows from the calculation `7*10540996611094048183 = 4*(2^64 - 2**32) + 1 = 1 mod (p - 1)`.
    fn injective_exp_root_n(&self) -> Self {
        exp_10540996611094048183(*self)
    }
}

impl_add_base_field!(PackedGoldilocksAVX2, Goldilocks);
impl_sub_base_field!(PackedGoldilocksAVX2, Goldilocks);
impl_mul_base_field!(PackedGoldilocksAVX2, Goldilocks);
impl_div_methods!(PackedGoldilocksAVX2, Goldilocks);
impl_sum_prod_base_field!(PackedGoldilocksAVX2, Goldilocks);

impl Algebra<Goldilocks> for PackedGoldilocksAVX2 {}

impl_packed_value!(PackedGoldilocksAVX2, Goldilocks, WIDTH);

unsafe impl PackedField for PackedGoldilocksAVX2 {
    type Scalar = Goldilocks;
}

impl_packed_field_pow_2!(
    PackedGoldilocksAVX2;
    [
        (1, interleave_u64),
        (2, interleave_u128),
    ],
    WIDTH
);

// Resources:
// 1. Intel Intrinsics Guide for explanation of each intrinsic:
//    https://software.intel.com/sites/landingpage/IntrinsicsGuide/
// 2. uops.info lists micro-ops for each instruction: https://uops.info/table.html
// 3. Intel optimization manual for introduction to x86 vector extensions and best practices:
//    https://software.intel.com/content/www/us/en/develop/download/intel-64-and-ia-32-architectures-optimization-reference-manual.html

// Preliminary knowledge:
// 1. Vector code usually avoids branching. Instead of branches, we can do input selection with
//    _mm256_blendv_epi8 or similar instruction. If all we're doing is conditionally zeroing a
//    vector element then _mm256_and_si256 or _mm256_andnot_si256 may be used and are cheaper.
//
// 2. AVX does not support addition with carry but 128-bit (2-word) addition can be easily
//    emulated. The method recognizes that for a + b overflowed iff (a + b) < a:
//        i. res_lo = a_lo + b_lo
//       ii. carry_mask = res_lo < a_lo
//      iii. res_hi = a_hi + b_hi - carry_mask
//    Notice that carry_mask is subtracted, not added. This is because AVX comparison instructions
//    return -1 (all bits 1) for true and 0 for false.
//
// 3. AVX does not have unsigned 64-bit comparisons. Those can be emulated with signed comparisons
//    by recognizing that a <u b iff a + (1 << 63) <s b + (1 << 63), where the addition wraps around
//    and the comparisons are unsigned and signed respectively. The shift function adds/subtracts
//    1 << 63 to enable this trick.
//      Example: addition with carry.
//        i. a_lo_s = shift(a_lo)
//       ii. res_lo_s = a_lo_s + b_lo
//      iii. carry_mask = res_lo_s <s a_lo_s
//       iv. res_lo = shift(res_lo_s)
//        v. res_hi = a_hi + b_hi - carry_mask
//    The suffix _s denotes a value that has been shifted by 1 << 63. The result of addition is
//    shifted if exactly one of the operands is shifted, as is the case on line ii. Line iii.
//    performs a signed comparison res_lo_s <s a_lo_s on shifted values to emulate unsigned
//    comparison res_lo <u a_lo on unshifted values. Finally, line iv. reverses the shift so the
//    result can be returned.
//      When performing a chain of calculations, we can often save instructions by letting the shift
//    propagate through and only undoing it when necessary. For example, to compute the addition of
//    three two-word (128-bit) numbers we can do:
//        i. a_lo_s = shift(a_lo)
//       ii. tmp_lo_s = a_lo_s + b_lo
//      iii. tmp_carry_mask = tmp_lo_s <s a_lo_s
//       iv. tmp_hi = a_hi + b_hi - tmp_carry_mask
//        v. res_lo_s = tmp_lo_s + c_lo
//       vi. res_carry_mask = res_lo_s <s tmp_lo_s
//      vii. res_lo = shift(res_lo_s)
//     viii. res_hi = tmp_hi + c_hi - res_carry_mask
//    Notice that the above 3-value addition still only requires two calls to shift, just like our
//    2-value addition.

const SIGN_BIT: __m256i = unsafe { transmute([i64::MIN; WIDTH]) };
const SHIFTED_FIELD_ORDER: __m256i =
    unsafe { transmute([Goldilocks::ORDER_U64 ^ (i64::MIN as u64); WIDTH]) };

/// Equal to 2^32 - 1 = 2^64 mod P.
const EPSILON: __m256i = unsafe { transmute([Goldilocks::ORDER_U64.wrapping_neg(); WIDTH]) };

/// Add 2^63 with overflow. Needed to emulate unsigned comparisons (see point 3. in
/// packed_prime_field.rs).
#[inline]
pub fn shift(x: __m256i) -> __m256i {
    unsafe { _mm256_xor_si256(x, SIGN_BIT) }
}

/// Convert to canonical representation.
/// The argument is assumed to be shifted by 1 << 63 (i.e. x_s = x + 1<<63, where x is the field
///   value). The returned value is similarly shifted by 1 << 63 (i.e. we return y_s = y + (1<<63),
///   where 0 <= y < FIELD_ORDER).
#[inline]
unsafe fn canonicalize_s(x_s: __m256i) -> __m256i {
    unsafe {
        // If x >= FIELD_ORDER then corresponding mask bits are all 0; otherwise all 1.
        let mask = _mm256_cmpgt_epi64(SHIFTED_FIELD_ORDER, x_s);
        // wrapback_amt is -FIELD_ORDER if mask is 0; otherwise 0.
        let wrapback_amt = _mm256_andnot_si256(mask, EPSILON);
        _mm256_add_epi64(x_s, wrapback_amt)
    }
}

/// Addition u64 + u64 -> u64. Assumes that x + y < 2^64 + FIELD_ORDER. The second argument is
/// pre-shifted by 1 << 63. The result is similarly shifted.
#[inline]
unsafe fn add_no_double_overflow_64_64s_s(x: __m256i, y_s: __m256i) -> __m256i {
    unsafe {
        let res_wrapped_s = _mm256_add_epi64(x, y_s);
        let mask = _mm256_cmpgt_epi64(y_s, res_wrapped_s); // -1 if overflowed else 0.
        let wrapback_amt = _mm256_srli_epi64::<32>(mask); // -FIELD_ORDER if overflowed else 0.
        _mm256_add_epi64(res_wrapped_s, wrapback_amt)
    }
}

/// Goldilocks modular addition. Computes `x + y mod FIELD_ORDER`.
///
/// Inputs can be arbitrary, output is not guaranteed to be less than `FIELD_ORDER`.
#[inline]
fn add(x: __m256i, y: __m256i) -> __m256i {
    unsafe {
        let y_s = shift(y);
        let res_s = add_no_double_overflow_64_64s_s(x, canonicalize_s(y_s));
        shift(res_s)
    }
}

/// Goldilocks modular subtraction. Computes `x - y mod FIELD_ORDER`.
///
/// Inputs can be arbitrary, output is not guaranteed to be less than `FIELD_ORDER`.
#[inline]
fn sub(x: __m256i, y: __m256i) -> __m256i {
    unsafe {
        let mut y_s = shift(y);
        y_s = canonicalize_s(y_s);
        let x_s = shift(x);
        let mask = _mm256_cmpgt_epi64(y_s, x_s); // -1 if sub will underflow (y > x) else 0.
        let wrapback_amt = _mm256_srli_epi64::<32>(mask); // -FIELD_ORDER if underflow else 0.
        let res_wrapped = _mm256_sub_epi64(x_s, y_s);
        _mm256_sub_epi64(res_wrapped, wrapback_amt)
    }
}

/// Goldilocks modular negation. Computes `-x mod FIELD_ORDER`.
///
/// Input can be arbitrary, output is not guaranteed to be less than `FIELD_ORDER`.
#[inline]
fn neg(y: __m256i) -> __m256i {
    unsafe {
        let y_s = shift(y);
        _mm256_sub_epi64(SHIFTED_FIELD_ORDER, canonicalize_s(y_s))
    }
}

/// Halve a vector of Goldilocks field elements.
#[inline(always)]
pub(crate) fn halve(input: __m256i) -> __m256i {
    /*
        We want this to compile to:
            vpand    least_bit, val, ONE
            vpsrlq   t, val, 1
            vpsubq   neg_least_bit, ZERO, least_bit
            vpand    maybe_half, HALF, neg_least_bit
            vpaddq   res, t, maybe_half
        throughput: 1.67 cyc/vec
        latency: 4 cyc

        Given an element val in [0, P), we want to compute val/2 mod P.
        If val is even: val/2 mod P = val/2 = val >> 1.
        If val is odd: val/2 mod P = (val + P)/2 = (val >> 1) + (P + 1)/2
    */
    unsafe {
        // Safety: If this code got compiled then AVX2 intrinsics are available.
        const ONE: __m256i = unsafe { transmute([1_i64; 4]) };
        const ZERO: __m256i = unsafe { transmute([0_i64; 4]) };
        let half = _mm256_set1_epi64x(P.div_ceil(2) as i64); // Compiler should realise this is constant.

        let least_bit = _mm256_and_si256(input, ONE); // Determine the parity of val.
        let t = _mm256_srli_epi64::<1>(input);

        // Negate the least bit giving us either 0 (all bits 0) or -1 (all bits 1).
        // It would be better to use vpsignq but this instruction does not exist.
        let neg_least_bit = _mm256_sub_epi64(ZERO, least_bit);

        // This does nothing when least_bit = 1 and sets the corresponding entry to 0 when least_bit = 0
        let maybe_half = _mm256_and_si256(half, neg_least_bit);
        _mm256_add_epi64(t, maybe_half)
    }
}

/// Full 64-bit by 64-bit multiplication. This emulated multiplication is 1.33x slower than the
/// scalar instruction, but may be worth it if we want our data to live in vector registers.
#[inline]
fn mul64_64(x: __m256i, y: __m256i) -> (__m256i, __m256i) {
    unsafe {
        // We want to move the high 32 bits to the low position. The multiplication instruction ignores
        // the high 32 bits, so it's ok to just duplicate it into the low position. This duplication can
        // be done on port 5; bitshifts run on ports 0 and 1, competing with multiplication.
        //   This instruction is only provided for 32-bit floats, not integers. Idk why Intel makes the
        // distinction; the casts are free and it guarantees that the exact bit pattern is preserved.
        // Using a swizzle instruction of the wrong domain (float vs int) does not increase latency
        // since Haswell.
        let x_hi = _mm256_castps_si256(_mm256_movehdup_ps(_mm256_castsi256_ps(x)));
        let y_hi = _mm256_castps_si256(_mm256_movehdup_ps(_mm256_castsi256_ps(y)));

        // All four pairwise multiplications
        let mul_ll = _mm256_mul_epu32(x, y);
        let mul_lh = _mm256_mul_epu32(x, y_hi);
        let mul_hl = _mm256_mul_epu32(x_hi, y);
        let mul_hh = _mm256_mul_epu32(x_hi, y_hi);

        // Bignum addition
        // Extract high 32 bits of mul_ll and add to mul_hl. This cannot overflow.
        let mul_ll_hi = _mm256_srli_epi64::<32>(mul_ll);
        let t0 = _mm256_add_epi64(mul_hl, mul_ll_hi);
        // Extract low 32 bits of t0 and add to mul_lh. Again, this cannot overflow.
        // Also, extract high 32 bits of t0 and add to mul_hh.
        let t0_lo = _mm256_and_si256(t0, EPSILON);
        let t0_hi = _mm256_srli_epi64::<32>(t0);
        let t1 = _mm256_add_epi64(mul_lh, t0_lo);
        let t2 = _mm256_add_epi64(mul_hh, t0_hi);
        // Lastly, extract the high 32 bits of t1 and add to t2.
        let t1_hi = _mm256_srli_epi64::<32>(t1);
        let res_hi = _mm256_add_epi64(t2, t1_hi);

        // Form res_lo by combining the low half of mul_ll with the low half of t1 (shifted into high
        // position).
        let t1_lo = _mm256_castps_si256(_mm256_moveldup_ps(_mm256_castsi256_ps(t1)));
        let res_lo = _mm256_blend_epi32::<0xaa>(mul_ll, t1_lo);

        (res_hi, res_lo)
    }
}

/// Full 64-bit squaring. This routine is 1.2x faster than the scalar instruction.
#[inline]
fn square64(x: __m256i) -> (__m256i, __m256i) {
    unsafe {
        // Get high 32 bits of x. See comment in mul64_64_s.
        let x_hi = _mm256_castps_si256(_mm256_movehdup_ps(_mm256_castsi256_ps(x)));

        // All pairwise multiplications.
        let mul_ll = _mm256_mul_epu32(x, x);
        let mul_lh = _mm256_mul_epu32(x, x_hi);
        let mul_hh = _mm256_mul_epu32(x_hi, x_hi);

        // Bignum addition, but mul_lh is shifted by 33 bits (not 32).
        let mul_ll_hi = _mm256_srli_epi64::<33>(mul_ll);
        let t0 = _mm256_add_epi64(mul_lh, mul_ll_hi);
        let t0_hi = _mm256_srli_epi64::<31>(t0);
        let res_hi = _mm256_add_epi64(mul_hh, t0_hi);

        // Form low result by adding the mul_ll and the low 31 bits of mul_lh (shifted to the high
        // position).
        let mul_lh_lo = _mm256_slli_epi64::<33>(mul_lh);
        let res_lo = _mm256_add_epi64(mul_ll, mul_lh_lo);

        (res_hi, res_lo)
    }
}

/// Goldilocks addition of a "small" number. `x_s` is pre-shifted by 2**63. `y` is assumed to be
/// `<= 2^64 - 2^32 = 0xffffffff00000000`. The result is shifted by 2**63.
#[inline]
unsafe fn add_small_64s_64_s(x_s: __m256i, y: __m256i) -> __m256i {
    unsafe {
        let res_wrapped_s = _mm256_add_epi64(x_s, y);
        // 32-bit compare is faster than 64-bit. It's safe as long as x > res_wrapped iff x >> 32 >
        // res_wrapped >> 32. The case of x >> 32 > res_wrapped >> 32 is trivial and so is <. The case
        // where x >> 32 = res_wrapped >> 32 remains. If x >> 32 = res_wrapped >> 32, then y >> 32 =
        // 0xffffffff and the addition of the low 32 bits generated a carry. This can never occur if y
        // <= 0xffffffff00000000: if y >> 32 = 0xffffffff, then no carry can occur.
        let mask = _mm256_cmpgt_epi32(x_s, res_wrapped_s); // -1 if overflowed else 0.
        // The mask contains 0xffffffff in the high 32 bits if wraparound occurred and 0 otherwise.
        let wrapback_amt = _mm256_srli_epi64::<32>(mask); // -FIELD_ORDER if overflowed else 0.
        _mm256_add_epi64(res_wrapped_s, wrapback_amt)
    }
}

/// Goldilocks subtraction of a "small" number. `x_s` is pre-shifted by 2**63. `y` is assumed to be
/// <= `0xffffffff00000000`. The result is shifted by 2**63.
#[inline]
unsafe fn sub_small_64s_64_s(x_s: __m256i, y: __m256i) -> __m256i {
    unsafe {
        let res_wrapped_s = _mm256_sub_epi64(x_s, y);
        // 32-bit compare is faster than 64-bit. It's safe as long as res_wrapped > x iff res_wrapped >>
        // 32 > x >> 32. The case of res_wrapped >> 32 > x >> 32 is trivial and so is <. The case where
        // res_wrapped >> 32 = x >> 32 remains. If res_wrapped >> 32 = x >> 32, then y >> 32 =
        // 0xffffffff and the subtraction of the low 32 bits generated a borrow. This can never occur if
        // y <= 0xffffffff00000000: if y >> 32 = 0xffffffff, then no borrow can occur.
        let mask = _mm256_cmpgt_epi32(res_wrapped_s, x_s); // -1 if underflowed else 0.
        // The mask contains 0xffffffff in the high 32 bits if wraparound occurred and 0 otherwise.
        let wrapback_amt = _mm256_srli_epi64::<32>(mask); // -FIELD_ORDER if underflowed else 0.
        _mm256_sub_epi64(res_wrapped_s, wrapback_amt)
    }
}

/// Given a 128-bit value represented as two 64-bit halves, reduce it modulo the Goldilocks field order.
///
/// The result will be a 64-bit value but may be larger than `FIELD_ORDER`.
#[inline]
fn reduce128(x: (__m256i, __m256i)) -> __m256i {
    unsafe {
        let (hi0, lo0) = x;

        // First we shift lo0 to lo0_s = lo0 + 2^{63} mod 2^64
        // This lets us emulate unsigned comparisons
        let lo0_s = shift(lo0);

        // Get the top 32 bits of hi_hi0.
        let hi_hi0 = _mm256_srli_epi64::<32>(hi0);

        // Computes lo0_s - hi_hi0 mod FIELD_ORDER.
        // Makes sense to do as 2^96 = -1 mod FIELD_ORDER.
        // sub_small_64s_64_s is safe to use as `hi_hi0 < 2^32`.
        let lo1_s = sub_small_64s_64_s(lo0_s, hi_hi0);

        // Compute the product of the bottom 32 bits of hi0 with 2^64 = 2^32 - 1 mod FIELD_ORDER
        // _mm256_mul_epu32 ignores the top 32 bits so just use that.
        let t1 = _mm256_mul_epu32(hi0, EPSILON);

        // Clearly t1 <= (2^32 - 1)^2 = 2^64 - 2^33 + 1 so we can use `add_small_64s_64_s` to get
        // `lo2_s = lo1_s + t1 mod FIELD_ORDER.`
        let lo2_s = add_small_64s_64_s(lo1_s, t1);

        // Finally just need to correct for the shift.
        shift(lo2_s)
    }
}

/// Goldilocks modular multiplication. Computes `x * y mod FIELD_ORDER`.
///
/// Inputs can be arbitrary, output is not guaranteed to be less than `FIELD_ORDER`.
#[inline]
fn mul(x: __m256i, y: __m256i) -> __m256i {
    reduce128(mul64_64(x, y))
}

/// Goldilocks modular square. Computes `x^2 mod FIELD_ORDER`.
///
/// Input can be arbitrary, output is not guaranteed to be less than `FIELD_ORDER`.
#[inline]
fn square(x: __m256i) -> __m256i {
    reduce128(square64(x))
}

#[cfg(test)]
mod tests {
    use p3_field_testing::test_packed_field;

    use super::{Goldilocks, PackedGoldilocksAVX2, WIDTH};

    const SPECIAL_VALS: [Goldilocks; WIDTH] = Goldilocks::new_array([
        0xFFFF_FFFF_0000_0000,
        0xFFFF_FFFF_FFFF_FFFF,
        0x0000_0000_0000_0001,
        0xFFFF_FFFF_0000_0001,
    ]);

    const ZEROS: PackedGoldilocksAVX2 = PackedGoldilocksAVX2(Goldilocks::new_array([
        0x0000_0000_0000_0000,
        0xFFFF_FFFF_0000_0001,
        0x0000_0000_0000_0000,
        0xFFFF_FFFF_0000_0001,
    ]));

    const ONES: PackedGoldilocksAVX2 = PackedGoldilocksAVX2(Goldilocks::new_array([
        0x0000_0000_0000_0001,
        0xFFFF_FFFF_0000_0002,
        0x0000_0000_0000_0001,
        0xFFFF_FFFF_0000_0002,
    ]));

    test_packed_field!(
        crate::PackedGoldilocksAVX2,
        &[super::ZEROS],
        &[super::ONES],
        crate::PackedGoldilocksAVX2(super::SPECIAL_VALS)
    );
}
