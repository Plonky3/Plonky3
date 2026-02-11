use alloc::vec::Vec;
use core::arch::aarch64::{
    uint64x2_t, vaddq_u64, vandq_u64, vbicq_u64, vcgtq_s64, vdupq_n_u64, veorq_u64, vgetq_lane_u64,
    vreinterpretq_s64_u64, vshrq_n_u64, vsubq_u64, vtrn1q_u64, vtrn2q_u64,
};
use core::fmt::Debug;
use core::iter::{Product, Sum};
use core::mem::transmute;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use p3_field::exponentiation::exp_10540996611094048183;
use p3_field::op_assign_macros::{
    impl_add_assign, impl_add_base_field, impl_div_methods, impl_mul_base_field, impl_mul_methods,
    impl_packed_value, impl_rng, impl_sub_assign, impl_sub_base_field, impl_sum_prod_base_field,
    ring_sum,
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

/// Vectorized NEON implementation of `Goldilocks` arithmetic.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
#[repr(transparent)]
#[must_use]
pub struct PackedGoldilocksNeon(pub [Goldilocks; WIDTH]);

impl PackedGoldilocksNeon {
    #[inline]
    #[must_use]
    pub(crate) fn to_vector(self) -> uint64x2_t {
        unsafe { transmute(self) }
    }

    #[inline]
    pub(crate) fn from_vector(vector: uint64x2_t) -> Self {
        unsafe { transmute(vector) }
    }

    #[inline]
    const fn broadcast(value: Goldilocks) -> Self {
        Self([value; WIDTH])
    }
}

impl From<Goldilocks> for PackedGoldilocksNeon {
    fn from(x: Goldilocks) -> Self {
        Self::broadcast(x)
    }
}

impl Add for PackedGoldilocksNeon {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::from_vector(add(self.to_vector(), rhs.to_vector()))
    }
}

impl Sub for PackedGoldilocksNeon {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::from_vector(sub(self.to_vector(), rhs.to_vector()))
    }
}

impl Neg for PackedGoldilocksNeon {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self::from_vector(neg(self.to_vector()))
    }
}

impl Mul for PackedGoldilocksNeon {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self::from_vector(mul(self.to_vector(), rhs.to_vector()))
    }
}

impl_add_assign!(PackedGoldilocksNeon);
impl_sub_assign!(PackedGoldilocksNeon);
impl_mul_methods!(PackedGoldilocksNeon);
ring_sum!(PackedGoldilocksNeon);
impl_rng!(PackedGoldilocksNeon);

impl PrimeCharacteristicRing for PackedGoldilocksNeon {
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
        unsafe { reconstitute_from_base(Goldilocks::zero_vec(len * WIDTH)) }
    }
}

impl InjectiveMonomial<7> for PackedGoldilocksNeon {}

impl PermutationMonomial<7> for PackedGoldilocksNeon {
    fn injective_exp_root_n(&self) -> Self {
        exp_10540996611094048183(*self)
    }
}

impl_add_base_field!(PackedGoldilocksNeon, Goldilocks);
impl_sub_base_field!(PackedGoldilocksNeon, Goldilocks);
impl_mul_base_field!(PackedGoldilocksNeon, Goldilocks);
impl_div_methods!(PackedGoldilocksNeon, Goldilocks);
impl_sum_prod_base_field!(PackedGoldilocksNeon, Goldilocks);

impl Algebra<Goldilocks> for PackedGoldilocksNeon {}

impl_packed_value!(PackedGoldilocksNeon, Goldilocks, WIDTH);

unsafe impl PackedField for PackedGoldilocksNeon {
    type Scalar = Goldilocks;
}

/// Interleave two 64-bit vectors at the element level.
/// For block_len=1: [a0, a1] x [b0, b1] -> [a0, b0], [a1, b1]
#[inline]
pub fn interleave_u64(v0: uint64x2_t, v1: uint64x2_t) -> (uint64x2_t, uint64x2_t) {
    // We want this to compile to:
    //      trn1  res0.2d, v0.2d, v1.2d
    //      trn2  res1.2d, v0.2d, v1.2d
    unsafe { (vtrn1q_u64(v0, v1), vtrn2q_u64(v0, v1)) }
}

unsafe impl PackedFieldPow2 for PackedGoldilocksNeon {
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

// NEON arithmetic uses shifted representation (XOR with 2^63) for unsigned comparison.

const SIGN_BIT: uint64x2_t = unsafe { transmute([i64::MIN as u64; WIDTH]) };
const SHIFTED_FIELD_ORDER: uint64x2_t =
    unsafe { transmute([Goldilocks::ORDER_U64 ^ (i64::MIN as u64); WIDTH]) };
const EPSILON_VEC: uint64x2_t = unsafe { transmute([EPSILON; WIDTH]) };

#[inline(always)]
fn shift(x: uint64x2_t) -> uint64x2_t {
    unsafe { veorq_u64(x, SIGN_BIT) }
}

#[inline(always)]
unsafe fn canonicalize_s(x_s: uint64x2_t) -> uint64x2_t {
    unsafe {
        let x_s_signed = vreinterpretq_s64_u64(x_s);
        let order_s_signed = vreinterpretq_s64_u64(SHIFTED_FIELD_ORDER);
        let mask = vcgtq_s64(order_s_signed, x_s_signed);
        let wrapback_amt = vbicq_u64(EPSILON_VEC, mask);
        vaddq_u64(x_s, wrapback_amt)
    }
}

#[inline(always)]
unsafe fn add_no_double_overflow_64_64s_s(x: uint64x2_t, y_s: uint64x2_t) -> uint64x2_t {
    unsafe {
        let res_wrapped_s = vaddq_u64(x, y_s);
        // After XOR shift, signed comparison correctly detects overflow.
        // Overflow occurred iff y_s > res_wrapped_s (as signed, due to shift semantics)
        let y_s_signed = vreinterpretq_s64_u64(y_s);
        let res_s_signed = vreinterpretq_s64_u64(res_wrapped_s);
        let mask = vcgtq_s64(y_s_signed, res_s_signed);
        // wrapback_amt is EPSILON on overflow
        let wrapback_amt = vshrq_n_u64::<32>(mask);
        vaddq_u64(res_wrapped_s, wrapback_amt)
    }
}

/// Goldilocks modular addition.
#[inline]
fn add(x: uint64x2_t, y: uint64x2_t) -> uint64x2_t {
    unsafe {
        let y_s = shift(y);
        let res_s = add_no_double_overflow_64_64s_s(x, canonicalize_s(y_s));
        shift(res_s)
    }
}

/// Goldilocks modular subtraction.
#[inline]
fn sub(x: uint64x2_t, y: uint64x2_t) -> uint64x2_t {
    unsafe {
        let mut y_s = shift(y);
        y_s = canonicalize_s(y_s);
        let x_s = shift(x);
        let y_s_signed = vreinterpretq_s64_u64(y_s);
        let x_s_signed = vreinterpretq_s64_u64(x_s);
        // -1 if underflow (y > x)
        let mask = vcgtq_s64(y_s_signed, x_s_signed);
        let wrapback_amt = vshrq_n_u64::<32>(mask);
        let res_wrapped = vsubq_u64(x_s, y_s);
        vsubq_u64(res_wrapped, wrapback_amt)
    }
}

/// Goldilocks modular negation.
#[inline]
fn neg(y: uint64x2_t) -> uint64x2_t {
    unsafe {
        let y_s = shift(y);
        vsubq_u64(SHIFTED_FIELD_ORDER, canonicalize_s(y_s))
    }
}

/// Halve a vector of Goldilocks field elements.
#[inline(always)]
pub(crate) fn halve(input: uint64x2_t) -> uint64x2_t {
    unsafe {
        let one = vdupq_n_u64(1);
        let zero = vdupq_n_u64(0);
        let half = vdupq_n_u64(P.div_ceil(2));

        let least_bit = vandq_u64(input, one);
        let t = vshrq_n_u64::<1>(input);
        // neg_least_bit is 0 or -1 (all bits 1)
        let neg_least_bit = vsubq_u64(zero, least_bit);
        let maybe_half = vandq_u64(half, neg_least_bit);
        vaddq_u64(t, maybe_half)
    }
}

/// Goldilocks modular multiplication using interleaved dual-lane ASM.
#[inline]
fn mul(x: uint64x2_t, y: uint64x2_t) -> uint64x2_t {
    unsafe {
        let x0 = vgetq_lane_u64::<0>(x);
        let x1 = vgetq_lane_u64::<1>(x);
        let y0 = vgetq_lane_u64::<0>(y);
        let y1 = vgetq_lane_u64::<1>(y);

        let (res_0, res_1) = mul_reduce_dual_asm(x0, y0, x1, y1);

        transmute([res_0, res_1])
    }
}

/// Interleaved dual-lane multiplication and reduction using scalar ASM.
/// Uses shift-based EPSILON multiplication: hi_lo * EPSILON = (hi_lo << 32) - hi_lo
#[inline(always)]
unsafe fn mul_reduce_dual_asm(a0: u64, b0: u64, a1: u64, b1: u64) -> (u64, u64) {
    use core::arch::asm;
    let result0: u64;
    let result1: u64;

    unsafe {
        asm!(
            // Compute both 128-bit products (interleaved for ILP)
            "mul   {lo0}, {a0}, {b0}",
            "mul   {lo1}, {a1}, {b1}",
            "umulh {hi0}, {a0}, {b0}",
            "umulh {hi1}, {a1}, {b1}",

            // hi_hi = hi >> 32
            "lsr   {hi_hi0}, {hi0}, #32",
            "lsr   {hi_hi1}, {hi1}, #32",

            // tmp = lo - hi_hi (with borrow handling)
            "subs  {tmp0}, {lo0}, {hi_hi0}",
            "csetm {adj0:w}, cc",
            "subs  {tmp1}, {lo1}, {hi_hi1}",
            "csetm {adj1:w}, cc",
            "sub   {tmp0}, {tmp0}, {adj0}",
            "sub   {tmp1}, {tmp1}, {adj1}",

            // hi_lo = hi & EPSILON
            "and   {hi_lo0}, {hi0}, {epsilon}",
            "and   {hi_lo1}, {hi1}, {epsilon}",

            // hi_lo_eps = (hi_lo << 32) - hi_lo (avoids multiply)
            "lsl   {t0}, {hi_lo0}, #32",
            "lsl   {t1}, {hi_lo1}, #32",
            "sub   {hi_lo_eps0}, {t0}, {hi_lo0}",
            "sub   {hi_lo_eps1}, {t1}, {hi_lo1}",

            // result = tmp + hi_lo_eps (with overflow handling)
            "adds  {result0}, {tmp0}, {hi_lo_eps0}",
            "csetm {adj0:w}, cs",
            "adds  {result1}, {tmp1}, {hi_lo_eps1}",
            "csetm {adj1:w}, cs",
            "add   {result0}, {result0}, {adj0}",
            "add   {result1}, {result1}, {adj1}",

            a0 = in(reg) a0,
            b0 = in(reg) b0,
            a1 = in(reg) a1,
            b1 = in(reg) b1,
            epsilon = in(reg) EPSILON,
            lo0 = out(reg) _,
            lo1 = out(reg) _,
            hi0 = out(reg) _,
            hi1 = out(reg) _,
            hi_hi0 = out(reg) _,
            hi_hi1 = out(reg) _,
            tmp0 = out(reg) _,
            tmp1 = out(reg) _,
            hi_lo0 = out(reg) _,
            hi_lo1 = out(reg) _,
            t0 = out(reg) _,
            t1 = out(reg) _,
            hi_lo_eps0 = out(reg) _,
            hi_lo_eps1 = out(reg) _,
            adj0 = out(reg) _,
            adj1 = out(reg) _,
            result0 = out(reg) result0,
            result1 = out(reg) result1,
            options(pure, nomem, nostack),
        );
    }

    (result0, result1)
}

/// Goldilocks modular square using interleaved dual-lane ASM.
#[inline]
fn square(x: uint64x2_t) -> uint64x2_t {
    unsafe {
        let x0 = vgetq_lane_u64::<0>(x);
        let x1 = vgetq_lane_u64::<1>(x);

        let (res_0, res_1) = mul_reduce_dual_asm(x0, x0, x1, x1);

        transmute([res_0, res_1])
    }
}

#[cfg(test)]
mod tests {
    use p3_field_testing::test_packed_field;

    use super::{Goldilocks, PackedGoldilocksNeon, WIDTH};

    const SPECIAL_VALS: [Goldilocks; WIDTH] =
        Goldilocks::new_array([0xFFFF_FFFF_0000_0000, 0xFFFF_FFFF_FFFF_FFFF]);

    const ZEROS: PackedGoldilocksNeon = PackedGoldilocksNeon(Goldilocks::new_array([
        0x0000_0000_0000_0000,
        0xFFFF_FFFF_0000_0001, // = P, canonicalizes to 0
    ]));

    const ONES: PackedGoldilocksNeon = PackedGoldilocksNeon(Goldilocks::new_array([
        0x0000_0000_0000_0001,
        0xFFFF_FFFF_0000_0002, // = P + 1, canonicalizes to 1
    ]));

    test_packed_field!(
        crate::PackedGoldilocksNeon,
        &[super::ZEROS],
        &[super::ONES],
        crate::PackedGoldilocksNeon(super::SPECIAL_VALS)
    );
}
