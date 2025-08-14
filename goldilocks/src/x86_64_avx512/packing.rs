use alloc::vec::Vec;
use core::arch::x86_64::*;
use core::fmt::Debug;
use core::iter::{Product, Sum};
use core::mem::transmute;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use p3_field::exponentiation::exp_10540996611094048183;
use p3_field::interleave::{interleave_u64, interleave_u128, interleave_u256};
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

const WIDTH: usize = 8;

/// Vectorized AVX512 implementation of `Goldilocks` arithmetic.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
#[repr(transparent)] // Needed to make `transmute`s safe.
#[must_use]
pub struct PackedGoldilocksAVX512(pub [Goldilocks; WIDTH]);

impl PackedGoldilocksAVX512 {
    /// Get an arch-specific vector representing the packed values.
    #[inline]
    #[must_use]
    pub(crate) fn to_vector(self) -> __m512i {
        unsafe {
            // Safety: `Goldilocks` is `repr(transparent)` so it can be transmuted to `u64`. It
            // follows that `[Goldilocks; WIDTH]` can be transmuted to `[u64; WIDTH]`, which can be
            // transmuted to `__m512i`, since arrays are guaranteed to be contiguous in memory.
            // Finally `PackedGoldilocksAVX512` is `repr(transparent)` so it can be transmuted to
            // `[Goldilocks; WIDTH]`.
            transmute(self)
        }
    }

    /// Make a packed field vector from an arch-specific vector.
    ///
    /// Elements of `Goldilocks` are allowed to be arbitrary u64s so this function
    /// is safe unlike the `Mersenne31/MontyField31` variants.
    #[inline]
    pub(crate) fn from_vector(vector: __m512i) -> Self {
        unsafe {
            // Safety: `__m512i` can be transmuted to `[u64; WIDTH]` (since arrays elements are
            // contiguous in memory), which can be transmuted to `[Goldilocks; WIDTH]` (since
            // `Goldilocks` is `repr(transparent)`), which in turn can be transmuted to
            // `PackedGoldilocksAVX512` (since `PackedGoldilocksAVX512` is also `repr(transparent)`).
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

impl From<Goldilocks> for PackedGoldilocksAVX512 {
    fn from(x: Goldilocks) -> Self {
        Self::broadcast(x)
    }
}

impl Add for PackedGoldilocksAVX512 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::from_vector(add(self.to_vector(), rhs.to_vector()))
    }
}

impl Sub for PackedGoldilocksAVX512 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::from_vector(sub(self.to_vector(), rhs.to_vector()))
    }
}

impl Neg for PackedGoldilocksAVX512 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self::from_vector(neg(self.to_vector()))
    }
}

impl Mul for PackedGoldilocksAVX512 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self::from_vector(mul(self.to_vector(), rhs.to_vector()))
    }
}

impl_add_assign!(PackedGoldilocksAVX512);
impl_sub_assign!(PackedGoldilocksAVX512);
impl_mul_methods!(PackedGoldilocksAVX512);
ring_sum!(PackedGoldilocksAVX512);
impl_rng!(PackedGoldilocksAVX512);

impl PrimeCharacteristicRing for PackedGoldilocksAVX512 {
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

impl_add_base_field!(PackedGoldilocksAVX512, Goldilocks);
impl_sub_base_field!(PackedGoldilocksAVX512, Goldilocks);
impl_mul_base_field!(PackedGoldilocksAVX512, Goldilocks);
impl_div_methods!(PackedGoldilocksAVX512, Goldilocks);
impl_sum_prod_base_field!(PackedGoldilocksAVX512, Goldilocks);

impl Algebra<Goldilocks> for PackedGoldilocksAVX512 {}

// Degree of the smallest permutation polynomial for Goldilocks.
//
// As p - 1 = 2^32 * 3 * 5 * 17 * ... the smallest choice for a degree D satisfying gcd(p - 1, D) = 1 is 7.
impl InjectiveMonomial<7> for PackedGoldilocksAVX512 {}

impl PermutationMonomial<7> for PackedGoldilocksAVX512 {
    /// In the field `Goldilocks`, `a^{1/7}` is equal to a^{10540996611094048183}.
    ///
    /// This follows from the calculation `7*10540996611094048183 = 4*(2^64 - 2**32) + 1 = 1 mod (p - 1)`.
    fn injective_exp_root_n(&self) -> Self {
        exp_10540996611094048183(*self)
    }
}

impl_packed_value!(PackedGoldilocksAVX512, Goldilocks, WIDTH);

unsafe impl PackedField for PackedGoldilocksAVX512 {
    type Scalar = Goldilocks;
}

impl_packed_field_pow_2!(
    PackedGoldilocksAVX512;
    [
        (1, interleave_u64),
        (2, interleave_u128),
        (4, interleave_u256),
    ],
    WIDTH
);

const FIELD_ORDER: __m512i = unsafe { transmute([Goldilocks::ORDER_U64; WIDTH]) };
const EPSILON: __m512i = unsafe { transmute([Goldilocks::ORDER_U64.wrapping_neg(); WIDTH]) };

#[inline]
unsafe fn canonicalize(x: __m512i) -> __m512i {
    unsafe {
        let mask = _mm512_cmpge_epu64_mask(x, FIELD_ORDER);
        _mm512_mask_sub_epi64(x, mask, x, FIELD_ORDER)
    }
}

/// Compute the modular addition `x + y mod FIELD_ORDER`.
///
/// This function is always safe if `y < FIELD_ORDER` but may also be used in a wider
/// set of circumstances if bounds on `x` are known.
///
/// The result will be a u64 which may be greater than FIELD_ORDER.
///
/// Safety:
///     User must ensure that x + y < 2^64 + FIELD_ORDER.
#[inline]
unsafe fn add_no_double_overflow_64_64(x: __m512i, y: __m512i) -> __m512i {
    unsafe {
        let res_wrapped = _mm512_add_epi64(x, y);
        let mask = _mm512_cmplt_epu64_mask(res_wrapped, y); // mask set if add overflowed
        _mm512_mask_sub_epi64(res_wrapped, mask, res_wrapped, FIELD_ORDER)
    }
}

/// Compute the modular subtraction x - y mod FIELD_ORDER.
///
/// This function is always safe if `y < FIELD_ORDER` but may also be used in a wider
/// set of circumstances if bounds on `x` are known.
///
/// The result will be a u64 which may be greater than FIELD_ORDER.
///
/// Safety:
///     User must ensure that x - y > -FIELD_ORDER.
#[inline]
unsafe fn sub_no_double_overflow_64_64(x: __m512i, y: __m512i) -> __m512i {
    unsafe {
        let mask = _mm512_cmplt_epu64_mask(x, y); // mask set if sub will underflow (x < y)
        let res_wrapped = _mm512_sub_epi64(x, y);
        _mm512_mask_add_epi64(res_wrapped, mask, res_wrapped, FIELD_ORDER)
    }
}

/// Goldilocks modular addition. Computes `x + y mod FIELD_ORDER`.
///
/// Inputs can be arbitrary, output is not guaranteed to be less than `FIELD_ORDER`.
#[inline]
fn add(x: __m512i, y: __m512i) -> __m512i {
    unsafe { add_no_double_overflow_64_64(x, canonicalize(y)) }
}

/// Goldilocks modular subtraction. Computes `x - y mod FIELD_ORDER`.
///
/// Inputs can be arbitrary, output is not guaranteed to be less than `FIELD_ORDER`.
#[inline]
fn sub(x: __m512i, y: __m512i) -> __m512i {
    unsafe { sub_no_double_overflow_64_64(x, canonicalize(y)) }
}

/// Goldilocks modular negation. Computes `-x mod FIELD_ORDER`.
///
/// Input can be arbitrary, output is not guaranteed to be less than `FIELD_ORDER`.
#[inline]
fn neg(y: __m512i) -> __m512i {
    unsafe { _mm512_sub_epi64(FIELD_ORDER, canonicalize(y)) }
}

/// Halve a vector of Goldilocks field elements.
#[inline(always)]
pub(crate) fn halve(input: __m512i) -> __m512i {
    /*
        We want this to compile to:
            vptestmq  least_bit, val, ONE
            vpsrlq    res, val, 1
            vpaddq    res{least_bit}, res, maybe_half
        throughput: 2 cyc/vec
        latency: 4 cyc

        Given an element val in [0, P), we want to compute val/2 mod P.
        If val is even: val/2 mod P = val/2 = val >> 1.
        If val is odd: val/2 mod P = (val + P)/2 = (val >> 1) + (P + 1)/2
    */
    unsafe {
        // Safety: If this code got compiled then AVX512 intrinsics are available.
        const ONE: __m512i = unsafe { transmute([1_i64; 8]) };
        let half = _mm512_set1_epi64(P.div_ceil(2) as i64); // Compiler realises this is constant.

        let least_bit = _mm512_test_epi64_mask(input, ONE); // Determine the parity of val.
        let t = _mm512_srli_epi64::<1>(input);
        // This does nothing when least_bit = 1 and sets the corresponding entry to 0 when least_bit = 0
        _mm512_mask_add_epi64(t, least_bit, t, half)
    }
}

#[allow(clippy::useless_transmute)]
const LO_32_BITS_MASK: __mmask16 = unsafe { transmute(0b0101010101010101u16) };

/// Full 64-bit by 64-bit multiplication.
#[inline]
fn mul64_64(x: __m512i, y: __m512i) -> (__m512i, __m512i) {
    unsafe {
        // We want to move the high 32 bits to the low position. The multiplication instruction ignores
        // the high 32 bits, so it's ok to just duplicate it into the low position. This duplication can
        // be done on port 5; bitshifts run on port 0, competing with multiplication.
        //   This instruction is only provided for 32-bit floats, not integers. Idk why Intel makes the
        // distinction; the casts are free and it guarantees that the exact bit pattern is preserved.
        // Using a swizzle instruction of the wrong domain (float vs int) does not increase latency
        // since Haswell.
        let x_hi = _mm512_castps_si512(_mm512_movehdup_ps(_mm512_castsi512_ps(x)));
        let y_hi = _mm512_castps_si512(_mm512_movehdup_ps(_mm512_castsi512_ps(y)));

        // All four pairwise multiplications
        let mul_ll = _mm512_mul_epu32(x, y);
        let mul_lh = _mm512_mul_epu32(x, y_hi);
        let mul_hl = _mm512_mul_epu32(x_hi, y);
        let mul_hh = _mm512_mul_epu32(x_hi, y_hi);

        // Bignum addition
        // Extract high 32 bits of mul_ll and add to mul_hl. This cannot overflow.
        let mul_ll_hi = _mm512_srli_epi64::<32>(mul_ll);
        let t0 = _mm512_add_epi64(mul_hl, mul_ll_hi);
        // Extract low 32 bits of t0 and add to mul_lh. Again, this cannot overflow.
        // Also, extract high 32 bits of t0 and add to mul_hh.
        let t0_lo = _mm512_and_si512(t0, EPSILON);
        let t0_hi = _mm512_srli_epi64::<32>(t0);
        let t1 = _mm512_add_epi64(mul_lh, t0_lo);
        let t2 = _mm512_add_epi64(mul_hh, t0_hi);
        // Lastly, extract the high 32 bits of t1 and add to t2.
        let t1_hi = _mm512_srli_epi64::<32>(t1);
        let res_hi = _mm512_add_epi64(t2, t1_hi);

        // Form res_lo by combining the low half of mul_ll with the low half of t1 (shifted into high
        // position).
        let t1_lo = _mm512_castps_si512(_mm512_moveldup_ps(_mm512_castsi512_ps(t1)));
        let res_lo = _mm512_mask_blend_epi32(LO_32_BITS_MASK, t1_lo, mul_ll);

        (res_hi, res_lo)
    }
}

/// Full 64-bit squaring.
#[inline]
fn square64(x: __m512i) -> (__m512i, __m512i) {
    unsafe {
        // Get high 32 bits of x. See comment in mul64_64_s.
        let x_hi = _mm512_castps_si512(_mm512_movehdup_ps(_mm512_castsi512_ps(x)));

        // All pairwise multiplications.
        let mul_ll = _mm512_mul_epu32(x, x);
        let mul_lh = _mm512_mul_epu32(x, x_hi);
        let mul_hh = _mm512_mul_epu32(x_hi, x_hi);

        // Bignum addition, but mul_lh is shifted by 33 bits (not 32).
        let mul_ll_hi = _mm512_srli_epi64::<33>(mul_ll);
        let t0 = _mm512_add_epi64(mul_lh, mul_ll_hi);
        let t0_hi = _mm512_srli_epi64::<31>(t0);
        let res_hi = _mm512_add_epi64(mul_hh, t0_hi);

        // Form low result by adding the mul_ll and the low 31 bits of mul_lh (shifted to the high
        // position).
        let mul_lh_lo = _mm512_slli_epi64::<33>(mul_lh);
        let res_lo = _mm512_add_epi64(mul_ll, mul_lh_lo);

        (res_hi, res_lo)
    }
}

/// Given a 128-bit value represented as two 64-bit halves, reduce it modulo the Goldilocks field order.
///
/// The result will be a 64-bit value but may be larger than `FIELD_ORDER`.
#[inline]
fn reduce128(x: (__m512i, __m512i)) -> __m512i {
    unsafe {
        let (hi0, lo0) = x;

        // Find the high 32 bits of hi0.
        let hi_hi0 = _mm512_srli_epi64::<32>(hi0);

        // Computes lo0_s - hi_hi0 mod FIELD_ORDER.
        // Makes sense to do as 2^96 = -1 mod FIELD_ORDER.
        // `sub_no_double_overflow_64_64` is safe to use as `hi_hi0 < 2^32`.
        let lo1 = sub_no_double_overflow_64_64(lo0, hi_hi0);

        // Compute the product of the bottom 32 bits of hi0 with 2^64 = 2^32 - 1 mod FIELD_ORDER
        // _mm256_mul_epu32 ignores the top 32 bits so just use that.
        let t1 = _mm512_mul_epu32(hi0, EPSILON);

        // Clearly t1 <= (2^32 - 1)^2 = 2^64 - 2^33 + 1 < FIELD_ORDER so we can use `add_no_double_overflow_64_64` to get
        // `lo1 + t1 mod FIELD_ORDER.`
        add_no_double_overflow_64_64(lo1, t1)
    }
}

/// Goldilocks modular multiplication. Computes `x * y mod FIELD_ORDER`.
///
/// Inputs can be arbitrary, output is not guaranteed to be less than `FIELD_ORDER`.
#[inline]
fn mul(x: __m512i, y: __m512i) -> __m512i {
    reduce128(mul64_64(x, y))
}

/// Goldilocks modular square. Computes `x^2 mod FIELD_ORDER`.
///
/// Input can be arbitrary, output is not guaranteed to be less than `FIELD_ORDER`.
#[inline]
fn square(x: __m512i) -> __m512i {
    reduce128(square64(x))
}

#[cfg(test)]
mod tests {
    use p3_field_testing::test_packed_field;

    use super::{Goldilocks, PackedGoldilocksAVX512, WIDTH};

    const SPECIAL_VALS: [Goldilocks; WIDTH] = Goldilocks::new_array([
        0xFFFF_FFFF_0000_0001,
        0xFFFF_FFFF_0000_0000,
        0xFFFF_FFFE_FFFF_FFFF,
        0xFFFF_FFFF_FFFF_FFFF,
        0x0000_0000_0000_0000,
        0x0000_0000_0000_0001,
        0x0000_0000_0000_0002,
        0x0FFF_FFFF_F000_0000,
    ]);

    const ZEROS: PackedGoldilocksAVX512 = PackedGoldilocksAVX512(Goldilocks::new_array([
        0x0000_0000_0000_0000,
        0xFFFF_FFFF_0000_0001,
        0x0000_0000_0000_0000,
        0xFFFF_FFFF_0000_0001,
        0x0000_0000_0000_0000,
        0xFFFF_FFFF_0000_0001,
        0x0000_0000_0000_0000,
        0xFFFF_FFFF_0000_0001,
    ]));

    const ONES: PackedGoldilocksAVX512 = PackedGoldilocksAVX512(Goldilocks::new_array([
        0x0000_0000_0000_0001,
        0xFFFF_FFFF_0000_0002,
        0x0000_0000_0000_0001,
        0xFFFF_FFFF_0000_0002,
        0x0000_0000_0000_0001,
        0xFFFF_FFFF_0000_0002,
        0x0000_0000_0000_0001,
        0xFFFF_FFFF_0000_0002,
    ]));

    test_packed_field!(
        crate::PackedGoldilocksAVX512,
        &[super::ZEROS],
        &[super::ONES],
        crate::PackedGoldilocksAVX512(super::SPECIAL_VALS)
    );
}
