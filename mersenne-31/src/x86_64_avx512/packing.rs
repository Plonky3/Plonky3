use alloc::vec::Vec;
use core::arch::x86_64::{self, __m512i, __mmask16};
use core::iter::{Product, Sum};
use core::mem::transmute;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use p3_field::exponentiation::exp_1717986917;
use p3_field::interleave::{interleave_u32, interleave_u64, interleave_u128, interleave_u256};
use p3_field::op_assign_macros::{
    impl_add_assign, impl_add_base_field, impl_div_methods, impl_mul_base_field, impl_mul_methods,
    impl_packed_value, impl_rng, impl_sub_assign, impl_sub_base_field, impl_sum_prod_base_field,
    ring_sum,
};
use p3_field::{
    Algebra, Field, InjectiveMonomial, PackedField, PackedFieldPow2, PackedValue,
    PermutationMonomial, PrimeCharacteristicRing, impl_packed_field_pow_2, mm512_mod_add,
    mm512_mod_sub,
};
use p3_util::reconstitute_from_base;
use rand::Rng;
use rand::distr::{Distribution, StandardUniform};

use crate::{Mersenne31, mul_2exp_i};

const WIDTH: usize = 16;
pub(crate) const P: __m512i = unsafe { transmute::<[u32; WIDTH], _>([0x7fffffff; WIDTH]) };
const EVENS: __mmask16 = 0b0101010101010101;
const ODDS: __mmask16 = 0b1010101010101010;

/// Vectorized AVX-512F implementation of `Mersenne31` arithmetic.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(transparent)] // Needed to make `transmute`s safe.
#[must_use]
pub struct PackedMersenne31AVX512(pub [Mersenne31; WIDTH]);

impl PackedMersenne31AVX512 {
    #[inline]
    #[must_use]
    /// Get an arch-specific vector representing the packed values.
    pub(crate) fn to_vector(self) -> __m512i {
        unsafe {
            // Safety: `Mersenne31` is `repr(transparent)` so it can be transmuted to `u32`. It
            // follows that `[Mersenne31; WIDTH]` can be transmuted to `[u32; WIDTH]`, which can be
            // transmuted to `__m512i`, since arrays are guaranteed to be contiguous in memory.
            // Finally `PackedMersenne31AVX512` is `repr(transparent)` so it can be transmuted to
            // `[Mersenne31; WIDTH]`.
            transmute(self)
        }
    }

    #[inline]
    /// Make a packed field vector from an arch-specific vector.
    ///
    /// SAFETY: The caller must ensure that each element of `vector` represents a valid
    /// `Mersenne31`. In particular, each element of vector must be in `0..=P`.
    pub(crate) unsafe fn from_vector(vector: __m512i) -> Self {
        unsafe {
            // Safety: It is up to the user to ensure that elements of `vector` represent valid
            // `Mersenne31` values. We must only reason about memory representations. `__m512i` can be
            // transmuted to `[u32; WIDTH]` (since arrays elements are contiguous in memory), which can
            // be transmuted to `[Mersenne31; WIDTH]` (since `Mersenne31` is `repr(transparent)`), which
            // in turn can be transmuted to `PackedMersenne31AVX512` (since `PackedMersenne31AVX512` is also
            // `repr(transparent)`).
            transmute(vector)
        }
    }

    /// Copy `value` to all positions in a packed vector. This is the same as
    /// `From<Mersenne31>::from`, but `const`.
    #[inline]
    const fn broadcast(value: Mersenne31) -> Self {
        Self([value; WIDTH])
    }
}

impl From<Mersenne31> for PackedMersenne31AVX512 {
    #[inline]
    fn from(value: Mersenne31) -> Self {
        Self::broadcast(value)
    }
}

impl Add for PackedMersenne31AVX512 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        let lhs = self.to_vector();
        let rhs = rhs.to_vector();
        let res = mm512_mod_add(lhs, rhs, P);
        unsafe {
            // Safety: `mm512_mod_add` returns values in canonical form when given values in canonical form.
            Self::from_vector(res)
        }
    }
}

impl Sub for PackedMersenne31AVX512 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        let lhs = self.to_vector();
        let rhs = rhs.to_vector();
        let res = mm512_mod_sub(lhs, rhs, P);
        unsafe {
            // Safety: `mm512_mod_sub` returns values in canonical form when given values in canonical form.
            Self::from_vector(res)
        }
    }
}

impl Neg for PackedMersenne31AVX512 {
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

impl Mul for PackedMersenne31AVX512 {
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

impl_add_assign!(PackedMersenne31AVX512);
impl_sub_assign!(PackedMersenne31AVX512);
impl_mul_methods!(PackedMersenne31AVX512);
ring_sum!(PackedMersenne31AVX512);
impl_rng!(PackedMersenne31AVX512);

impl PrimeCharacteristicRing for PackedMersenne31AVX512 {
    type PrimeSubfield = Mersenne31;

    const ZERO: Self = Self::broadcast(Mersenne31::ZERO);
    const ONE: Self = Self::broadcast(Mersenne31::ONE);
    const TWO: Self = Self::broadcast(Mersenne31::TWO);
    const NEG_ONE: Self = Self::broadcast(Mersenne31::NEG_ONE);

    #[inline]
    fn from_prime_subfield(f: Self::PrimeSubfield) -> Self {
        f.into()
    }

    #[inline]
    fn halve(&self) -> Self {
        // 2^{-1} = 2^30 mod P so we implement halve by multiplying by 2^30.
        mul_2exp_i::<30, 1>(*self)
    }

    #[inline(always)]
    fn zero_vec(len: usize) -> Vec<Self> {
        // SAFETY: this is a repr(transparent) wrapper around an array.
        unsafe { reconstitute_from_base(Mersenne31::zero_vec(len * WIDTH)) }
    }

    #[inline(always)]
    fn exp_const_u64<const POWER: u64>(&self) -> Self {
        // We provide specialised code for power 5 as this turns up regularly.
        // The other powers could be specialised similarly but we ignore this for now.
        // These ideas could also be used to speed up the more generic exp_u64.
        match POWER {
            0 => Self::ONE,
            1 => *self,
            2 => self.square(),
            3 => self.cube(),
            4 => self.square().square(),
            5 => unsafe {
                let val = self.to_vector();
                Self::from_vector(exp5(val))
            },
            6 => self.square().cube(),
            7 => {
                let x2 = self.square();
                let x3 = x2 * *self;
                let x4 = x2.square();
                x3 * x4
            }
            _ => self.exp_u64(POWER),
        }
    }
}

// Degree of the smallest permutation polynomial for Mersenne31.
//
// As p - 1 = 2×3^2×7×11×... the smallest choice for a degree D satisfying gcd(p - 1, D) = 1 is 5.
impl InjectiveMonomial<5> for PackedMersenne31AVX512 {}

impl PermutationMonomial<5> for PackedMersenne31AVX512 {
    /// In the field `Mersenne31`, `a^{1/5}` is equal to a^{1717986917}.
    ///
    /// This follows from the calculation `5 * 1717986917 = 4*(2^31 - 2) + 1 = 1 mod p - 1`.
    fn injective_exp_root_n(&self) -> Self {
        exp_1717986917(*self)
    }
}

impl_add_base_field!(PackedMersenne31AVX512, Mersenne31);
impl_sub_base_field!(PackedMersenne31AVX512, Mersenne31);
impl_mul_base_field!(PackedMersenne31AVX512, Mersenne31);
impl_div_methods!(PackedMersenne31AVX512, Mersenne31);
impl_sum_prod_base_field!(PackedMersenne31AVX512, Mersenne31);

impl Algebra<Mersenne31> for PackedMersenne31AVX512 {}

#[inline]
#[must_use]
fn movehdup_epi32(a: __m512i) -> __m512i {
    // The instruction is only available in the floating-point flavor; this distinction is only for
    // historical reasons and no longer matters. We cast to floats, do the thing, and cast back.
    unsafe {
        x86_64::_mm512_castps_si512(x86_64::_mm512_movehdup_ps(x86_64::_mm512_castsi512_ps(a)))
    }
}

#[inline]
#[must_use]
fn mask_movehdup_epi32(src: __m512i, k: __mmask16, a: __m512i) -> __m512i {
    // The instruction is only available in the floating-point flavor; this distinction is only for
    // historical reasons and no longer matters. We cast to floats, do the thing, and cast back.
    unsafe {
        let src = x86_64::_mm512_castsi512_ps(src);
        let a = x86_64::_mm512_castsi512_ps(a);
        x86_64::_mm512_castps_si512(x86_64::_mm512_mask_movehdup_ps(src, k, a))
    }
}

#[inline]
#[must_use]
fn mask_moveldup_epi32(src: __m512i, k: __mmask16, a: __m512i) -> __m512i {
    // The instruction is only available in the floating-point flavor; this distinction is only for
    // historical reasons and no longer matters. We cast to floats, do the thing, and cast back.
    unsafe {
        let src = x86_64::_mm512_castsi512_ps(src);
        let a = x86_64::_mm512_castsi512_ps(a);
        x86_64::_mm512_castps_si512(x86_64::_mm512_mask_moveldup_ps(src, k, a))
    }
}

/// Multiply vectors of Mersenne-31 field elements represented as values in {0, ..., P}.
/// If the inputs do not conform to this representation, the result is undefined.
#[inline]
#[must_use]
fn mul(lhs: __m512i, rhs: __m512i) -> __m512i {
    // We want this to compile to:
    // vpaddd     lhs_evn_dbl, lhs, lhs
    // vmovshdup  rhs_odd, rhs
    // vpsrlq     lhs_odd_dbl, lhs, 31
    // vpmuludq   prod_lo_dbl, lhs_evn_dbl, rhs
    // vpmuludq   prod_odd_dbl, lhs_odd_dbl, rhs_odd
    // vmovdqa32  prod_hi, prod_odd_dbl
    // vmovshdup  prod_hi{EVENS}, prod_lo_dbl
    // vmovsldup  prod_lo_dbl{ODDS}, prod_odd_dbl
    // vpsrld     prod_lo, prod_lo_dbl, 1
    // vpaddd     t, prod_lo, prod_hi
    // vpsubd     u, t, P
    // vpminud    res, t, u
    // throughput: 5.5 cyc/vec (2.91 els/cyc)
    // latency: (lhs->res) 15 cyc, (rhs->res) 14 cyc
    unsafe {
        // vpmuludq only reads the bottom 32 bits of every 64-bit quadword.
        // The even indices are already in the bottom 32 bits of a quadword, so we can leave them.
        let rhs_evn = rhs;
        // Again, vpmuludq only reads the bottom 32 bits so we don't need to clear the top. But we
        // do want to double the lhs.
        let lhs_evn_dbl = x86_64::_mm512_add_epi32(lhs, lhs);
        // Copy the high 32 bits in each quadword of rhs down to the low 32.
        let rhs_odd = movehdup_epi32(rhs);
        // Right shift by 31 is equivalent to moving the high 32 bits down to the low 32, and then
        // doubling it. So these are the odd indices in lhs, but doubled.
        let lhs_odd_dbl = x86_64::_mm512_srli_epi64::<31>(lhs);

        // Multiply odd indices; since lhs_odd_dbl is doubled, these products are also doubled.
        // prod_odd_dbl.quadword[i] = 2 * lhs.doubleword[2 * i + 1] * rhs.doubleword[2 * i + 1]
        let prod_odd_dbl = x86_64::_mm512_mul_epu32(lhs_odd_dbl, rhs_odd);
        // Multiply even indices; these are also doubled.
        // prod_evn_dbl.quadword[i] = 2 * lhs.doubleword[2 * i] * rhs.doubleword[2 * i]
        let prod_evn_dbl = x86_64::_mm512_mul_epu32(lhs_evn_dbl, rhs_evn);

        // Move the low halves of odd products into odd positions; keep the low halves of even
        // products in even positions (where they already are). Note that the products are doubled,
        // so the result is a vector of all the low halves, but doubled.
        let prod_lo_dbl = mask_moveldup_epi32(prod_evn_dbl, ODDS, prod_odd_dbl);
        // Move the high halves of even products into even positions, keeping the high halves of odd
        // products where they are. The products are doubled, but we are looking at (prod >> 32),
        // which cancels out the doubling, so this result is _not_ doubled.
        let prod_hi = mask_movehdup_epi32(prod_odd_dbl, EVENS, prod_evn_dbl);
        // Right shift to undo the doubling.
        let prod_lo = x86_64::_mm512_srli_epi32::<1>(prod_lo_dbl);

        // Standard addition of two 31-bit values.
        mm512_mod_add(prod_lo, prod_hi, P)
    }
}

/// Negate a vector of Mersenne-31 field elements represented as values in {0, ..., P}.
/// If the input does not conform to this representation, the result is undefined.
#[inline]
#[must_use]
fn neg(val: __m512i) -> __m512i {
    // We want this to compile to:
    //      vpxord  res, val, P
    // throughput: .5 cyc/vec (32 els/cyc)
    // latency: 1 cyc

    //   Since val is in {0, ..., P (= 2^31 - 1)}, res = val XOR P = P - val. Then res is in {0,
    // ..., P}.
    unsafe {
        // Safety: If this code got compiled then AVX-512F intrinsics are available.
        x86_64::_mm512_xor_epi32(val, P)
    }
}

/// Reduce a representative in {0, ..., P^2}
/// to a representative in [-P, P]. If the input is greater than P^2, the output will
/// still correspond to the same class but will instead lie in [-P, 2^34].
#[inline(always)]
fn partial_reduce_neg(x: __m512i) -> __m512i {
    unsafe {
        // Get the top bits shifted down.
        let hi = x86_64::_mm512_srli_epi64::<31>(x);

        const LOW31: __m512i = unsafe { transmute::<[u64; 8], _>([0x7fffffff; 8]) };

        // nand instead of and means this returns P - lo.
        let neg_lo = x86_64::_mm512_andnot_si512(x, LOW31);

        // we could also try:
        // let neg_lo = x86_64::_mm512_maskz_andnot_epi32(EVENS, x, P);
        // but this seems to get compiled badly and likes outputting vpternlogd.
        // See: https://godbolt.org/z/WPze9e3f3

        // Compiling with sub_epi64 vs sub_epi32 both produce reasonable code so we use
        // sub_epi64 for the slightly greater flexibility.
        x86_64::_mm512_sub_epi64(hi, neg_lo)
    }
}

/// Compute the square of the Mersenne-31 field elements located in the even indices.
/// These field elements are represented as values in {-P, ..., P}. If the even inputs
/// do not conform to this representation, the result is undefined.
/// The top half of each 64-bit lane is ignored.
/// The top half of each 64-bit lane in the result is 0.
#[inline(always)]
fn square_unred(x: __m512i) -> __m512i {
    unsafe {
        // Safety: If this code got compiled then AVX-512F intrinsics are available.
        let x2 = x86_64::_mm512_mul_epi32(x, x);
        partial_reduce_neg(x2)
    }
}

/// Compute the permutation x -> x^5 on Mersenne-31 field elements
/// represented as values in {0, ..., P}. If the inputs do not conform
/// to this representation, the result is undefined.
#[inline(always)]
pub(crate) fn exp5(x: __m512i) -> __m512i {
    unsafe {
        // Safety: If this code got compiled then AVX-512F intrinsics are available.
        let input_evn = x;
        let input_odd = movehdup_epi32(x);

        let evn_sq = square_unred(input_evn);
        let odd_sq = square_unred(input_odd);

        let evn_4 = square_unred(evn_sq);
        let odd_4 = square_unred(odd_sq);

        let evn_5 = x86_64::_mm512_mul_epi32(evn_4, input_evn);
        let odd_5 = x86_64::_mm512_mul_epi32(odd_4, input_odd);

        // Marked dirty as the top bit needs to be cleared.
        let lo_dirty = mask_moveldup_epi32(evn_5, ODDS, odd_5);

        // We could use 2 adds and mask_movehdup_epi32.
        // instead of an add, a shift and a blend.
        let odd_5_hi = x86_64::_mm512_add_epi64(odd_5, odd_5);
        let evn_5_hi = x86_64::_mm512_srli_epi64::<31>(evn_5);
        let hi = x86_64::_mm512_mask_blend_epi32(ODDS, evn_5_hi, odd_5_hi);

        let zero = x86_64::_mm512_setzero_si512();
        let signs = x86_64::_mm512_movepi32_mask(hi);
        let corr = x86_64::_mm512_mask_sub_epi32(P, signs, zero, P);

        let lo = x86_64::_mm512_and_si512(lo_dirty, P);

        let t = x86_64::_mm512_add_epi32(hi, lo);
        let u = x86_64::_mm512_sub_epi32(t, corr);

        x86_64::_mm512_min_epu32(t, u)
    }
}

impl_packed_value!(PackedMersenne31AVX512, Mersenne31, WIDTH);

unsafe impl PackedField for PackedMersenne31AVX512 {
    type Scalar = Mersenne31;
}

impl_packed_field_pow_2!(
    PackedMersenne31AVX512;
    [
        (1, interleave_u32),
        (2, interleave_u64),
        (4, interleave_u128),
        (8, interleave_u256)
    ],
    WIDTH
);

#[cfg(test)]
mod tests {
    use p3_field_testing::test_packed_field;

    use super::{Mersenne31, PackedMersenne31AVX512};

    /// Zero has a redundant representation, so let's test both.
    const ZEROS: PackedMersenne31AVX512 = PackedMersenne31AVX512(Mersenne31::new_array([
        0x00000000, 0x7fffffff, 0x00000000, 0x7fffffff, 0x00000000, 0x7fffffff, 0x00000000,
        0x7fffffff, 0x00000000, 0x7fffffff, 0x00000000, 0x7fffffff, 0x00000000, 0x7fffffff,
        0x00000000, 0x7fffffff,
    ]));

    const SPECIAL_VALS: PackedMersenne31AVX512 = PackedMersenne31AVX512(Mersenne31::new_array([
        0x00000000, 0x7fffffff, 0x00000001, 0x7ffffffe, 0x00000002, 0x7ffffffd, 0x40000000,
        0x3fffffff, 0x00000000, 0x7fffffff, 0x00000001, 0x7ffffffe, 0x00000002, 0x7ffffffd,
        0x40000000, 0x3fffffff,
    ]));

    test_packed_field!(
        crate::PackedMersenne31AVX512,
        &[super::ZEROS],
        &[crate::PackedMersenne31AVX512::ONE],
        super::SPECIAL_VALS
    );
}
