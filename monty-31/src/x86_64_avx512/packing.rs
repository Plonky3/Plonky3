//! Optimised AVX512 implementation for packed vectors of MontyFields31 elements.
//!
//! We check that this compiles to the expected assembly code in: https://godbolt.org/z/Mz1WGYKWe

use alloc::vec::Vec;
use core::arch::asm;
use core::arch::x86_64::{self, __m256i, __m512i, __mmask8, __mmask16};
use core::array;
use core::hint::unreachable_unchecked;
use core::iter::{Product, Sum};
use core::mem::transmute;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

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

use crate::{
    BinomialExtensionData, FieldParameters, MontyField31, PackedMontyParameters,
    RelativelyPrimePower, halve_avx512,
};

const WIDTH: usize = 16;

pub trait MontyParametersAVX512 {
    const PACKED_P: __m512i;
    const PACKED_MU: __m512i;
}

const EVENS: __mmask16 = 0b0101010101010101;
const EVENS_8: __mmask8 = 0b01010101;

/// Vectorized AVX-512F implementation of `MontyField31` arithmetic.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(transparent)] // Needed to make `transmute`s safe.
#[must_use]
pub struct PackedMontyField31AVX512<PMP: PackedMontyParameters>(pub [MontyField31<PMP>; WIDTH]);

impl<PMP: PackedMontyParameters> PackedMontyField31AVX512<PMP> {
    #[inline]
    #[must_use]
    /// Get an arch-specific vector representing the packed values.
    pub(crate) fn to_vector(self) -> __m512i {
        unsafe {
            // Safety: `MontyField31` is `repr(transparent)` so it can be transmuted to `u32`. It
            // follows that `[MontyField31; WIDTH]` can be transmuted to `[u32; WIDTH]`, which can be
            // transmuted to `__m512i`, since arrays are guaranteed to be contiguous in memory.
            // Finally `PackedMontyField31AVX512` is `repr(transparent)` so it can be transmuted to
            // `[MontyField31; WIDTH]`.
            transmute(self)
        }
    }

    #[inline]
    /// Make a packed field vector from an arch-specific vector.
    ///
    /// SAFETY: The caller must ensure that each element of `vector` represents a valid
    /// `MontyField31`. In particular, each element of vector must be in `0..=P`.
    pub(crate) unsafe fn from_vector(vector: __m512i) -> Self {
        unsafe {
            // Safety: It is up to the user to ensure that elements of `vector` represent valid
            // `MontyField31` values. We must only reason about memory representations. `__m512i` can be
            // transmuted to `[u32; WIDTH]` (since arrays elements are contiguous in memory), which can
            // be transmuted to `[MontyField31; WIDTH]` (since `MontyField31` is `repr(transparent)`), which
            // in turn can be transmuted to `PackedMontyField31AVX512` (since `PackedMontyField31AVX512` is also
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

    /// Copy values from `arr` into the packed vector padding by zeros if necessary.
    #[inline]
    fn from_monty_array<const N: usize>(arr: [MontyField31<PMP>; N]) -> Self
    where
        PMP: FieldParameters,
    {
        assert!(N <= WIDTH);
        let mut out = Self::ZERO;
        out.0[..N].copy_from_slice(&arr);
        out
    }
}

impl<PMP: PackedMontyParameters> From<MontyField31<PMP>> for PackedMontyField31AVX512<PMP> {
    #[inline]
    fn from(value: MontyField31<PMP>) -> Self {
        Self::broadcast(value)
    }
}

impl<PMP: PackedMontyParameters> Add for PackedMontyField31AVX512<PMP> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        let lhs = self.to_vector();
        let rhs = rhs.to_vector();
        let res = mm512_mod_add(lhs, rhs, PMP::PACKED_P);
        unsafe {
            // Safety: `add` returns values in canonical form when given values in canonical form.
            Self::from_vector(res)
        }
    }
}

impl<PMP: PackedMontyParameters> Sub for PackedMontyField31AVX512<PMP> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        let lhs = self.to_vector();
        let rhs = rhs.to_vector();
        let res = mm512_mod_sub(lhs, rhs, PMP::PACKED_P);
        unsafe {
            // Safety: `mm512_mod_sub` returns values in canonical form when given values in canonical form.
            Self::from_vector(res)
        }
    }
}

impl<PMP: PackedMontyParameters> Neg for PackedMontyField31AVX512<PMP> {
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

impl<PMP: PackedMontyParameters> Mul for PackedMontyField31AVX512<PMP> {
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

impl_add_assign!(PackedMontyField31AVX512, (PackedMontyParameters, PMP));
impl_sub_assign!(PackedMontyField31AVX512, (PackedMontyParameters, PMP));
impl_mul_methods!(PackedMontyField31AVX512, (FieldParameters, FP));
ring_sum!(PackedMontyField31AVX512, (FieldParameters, FP));
impl_rng!(PackedMontyField31AVX512, (PackedMontyParameters, PMP));

impl<FP: FieldParameters> PrimeCharacteristicRing for PackedMontyField31AVX512<FP> {
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
        let halved = halve_avx512::<FP>(val);
        unsafe {
            // Safety: `halve_avx512` returns values in canonical form when given values in canonical form.
            Self::from_vector(halved)
        }
    }

    #[inline(always)]
    fn zero_vec(len: usize) -> Vec<Self> {
        // SAFETY: this is a repr(transparent) wrapper around an array.
        unsafe { reconstitute_from_base(MontyField31::<FP>::zero_vec(len * WIDTH)) }
    }

    #[inline]
    fn cube(&self) -> Self {
        let val = self.to_vector();
        unsafe {
            // Safety: `apply_func_to_even_odd` returns values in canonical form when given values in canonical form.
            let res = apply_func_to_even_odd::<FP>(val, packed_exp_3::<FP>);
            Self::from_vector(res)
        }
    }

    #[inline]
    fn xor(&self, rhs: &Self) -> Self {
        let lhs = self.to_vector();
        let rhs = rhs.to_vector();
        let res = xor::<FP>(lhs, rhs);
        unsafe {
            // Safety: `xor` returns values in canonical form when given values in canonical form.
            Self::from_vector(res)
        }
    }

    #[inline]
    fn andn(&self, rhs: &Self) -> Self {
        let lhs = self.to_vector();
        let rhs = rhs.to_vector();
        let res = andn::<FP>(lhs, rhs);
        unsafe {
            // Safety: `andn` returns values in canonical form when given values in canonical form.
            Self::from_vector(res)
        }
    }

    #[inline(always)]
    fn exp_const_u64<const POWER: u64>(&self) -> Self {
        // We provide specialised code for the powers 3, 5, 7 as these turn up regularly.
        // The other powers could be specialised similarly but we ignore this for now.
        // These ideas could also be used to speed up the more generic exp_u64.
        match POWER {
            0 => Self::ONE,
            1 => *self,
            2 => self.square(),
            3 => self.cube(),
            4 => self.square().square(),
            5 => {
                let val = self.to_vector();
                unsafe {
                    // Safety: `apply_func_to_even_odd` returns values in canonical form when given values in canonical form.
                    let res = apply_func_to_even_odd::<FP>(val, packed_exp_5::<FP>);
                    Self::from_vector(res)
                }
            }
            6 => self.square().cube(),
            7 => {
                let val = self.to_vector();
                unsafe {
                    // Safety: `apply_func_to_even_odd` returns values in canonical form when given values in canonical form.
                    let res = apply_func_to_even_odd::<FP>(val, packed_exp_7::<FP>);
                    Self::from_vector(res)
                }
            }
            _ => self.exp_u64(POWER),
        }
    }

    #[inline(always)]
    fn dot_product<const N: usize>(u: &[Self; N], v: &[Self; N]) -> Self {
        general_dot_product::<_, _, _, N>(u, v)
    }
}

impl_add_base_field!(
    PackedMontyField31AVX512,
    MontyField31,
    (PackedMontyParameters, PMP)
);
impl_sub_base_field!(
    PackedMontyField31AVX512,
    MontyField31,
    (PackedMontyParameters, PMP)
);
impl_mul_base_field!(
    PackedMontyField31AVX512,
    MontyField31,
    (PackedMontyParameters, PMP)
);
impl_div_methods!(
    PackedMontyField31AVX512,
    MontyField31,
    (FieldParameters, FP)
);
impl_sum_prod_base_field!(
    PackedMontyField31AVX512,
    MontyField31,
    (FieldParameters, FP)
);

impl<FP: FieldParameters> Algebra<MontyField31<FP>> for PackedMontyField31AVX512<FP> {}

impl<FP: FieldParameters + RelativelyPrimePower<D>, const D: u64> InjectiveMonomial<D>
    for PackedMontyField31AVX512<FP>
{
}

impl<FP: FieldParameters + RelativelyPrimePower<D>, const D: u64> PermutationMonomial<D>
    for PackedMontyField31AVX512<FP>
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
fn confuse_compiler(x: __m512i) -> __m512i {
    let y;
    unsafe {
        asm!(
            "/*{0}*/",
            inlateout(zmm_reg) x => y,
            options(nomem, nostack, preserves_flags, pure),
        );
        // Below tells the compiler the semantics of this so it can still do constant folding, etc.
        // You may ask, doesn't it defeat the point of the inline asm block to tell the compiler
        // what it does? The answer is that we still inhibit the transform we want to avoid, so
        // apparently not. Idk, LLVM works in mysterious ways.
        if transmute::<__m512i, [u32; 16]>(x) != transmute::<__m512i, [u32; 16]>(y) {
            unreachable_unchecked();
        }
    }
    y
}

/// No-op. Prevents the compiler from deducing the value of the vector.
///
/// A variant of [`confuse_compiler`] for use with `__m256i` vectors.
///
/// Similar to `core::hint::black_box`, it can be used to stop the compiler applying undesirable
/// "optimizations". Unlike the built-in `black_box`, it does not force the value to be written to
/// and then read from the stack.
#[inline]
#[must_use]
fn confuse_compiler_256(x: __m256i) -> __m256i {
    let y;
    unsafe {
        asm!(
            "/*{0}*/",
            inlateout(ymm_reg) x => y,
            options(nomem, nostack, preserves_flags, pure),
        );
        // Below tells the compiler the semantics of this so it can still do constant folding, etc.
        // You may ask, doesn't it defeat the point of the inline asm block to tell the compiler
        // what it does? The answer is that we still inhibit the transform we want to avoid, so
        // apparently not. Idk, LLVM works in mysterious ways.
        if transmute::<__m256i, [u32; 8]>(x) != transmute::<__m256i, [u32; 8]>(y) {
            unreachable_unchecked();
        }
    }
    y
}

// MONTGOMERY MULTIPLICATION
//   This implementation is based on [1] but with minor changes. The reduction is as follows:
//
// Constants: P < 2^31
//            B = 2^32
//            μ = P^-1 mod B
// Input: 0 <= C < P B
// Output: 0 <= R < P such that R = C B^-1 (mod P)
//   1. Q := μ C mod B
//   2. D := (C - Q P) / B
//   3. R := if D < 0 then D + P else D
//
// We first show that the division in step 2. is exact. It suffices to show that C = Q P (mod B). By
// definition of Q and μ, we have Q P = μ C P = P^-1 C P = C (mod B). We also have
// C - Q P = C (mod P), so thus D = C B^-1 (mod P).
//
// It remains to show that R is in the correct range. It suffices to show that -P <= D < P. We know
// that 0 <= C < P B and 0 <= Q P < P B. Then -P B < C - QP < P B and -P < D < P, as desired.
//
// [1] Modern Computer Arithmetic, Richard Brent and Paul Zimmermann, Cambridge University Press,
//     2010, algorithm 2.7.

/// Perform a partial Montgomery reduction on each 64 bit element.
/// Input must lie in {0, ..., 2^32P}.
/// The output will lie in {-P, ..., P} and be stored in the upper 32 bits.
#[inline]
#[must_use]
fn partial_monty_red_unsigned_to_signed<MPAVX512: MontyParametersAVX512>(
    input: __m512i,
) -> __m512i {
    unsafe {
        // We throw a confuse compiler here to prevent the compiler from
        // using vpmullq instead of vpmuludq in the computations for q_p.
        // vpmullq has both higher latency and lower throughput.
        let q = confuse_compiler(x86_64::_mm512_mul_epu32(input, MPAVX512::PACKED_MU));
        let q_p = x86_64::_mm512_mul_epu32(q, MPAVX512::PACKED_P);

        // This could equivalently be _mm512_sub_epi64
        x86_64::_mm512_sub_epi32(input, q_p)
    }
}

/// Perform a partial Montgomery reduction on each 64 bit element.
/// Input must lie in {-2^{31}P, ..., 2^31P}.
/// The output will lie in {-P, ..., P} and be stored in the upper 32 bits.
#[inline]
#[must_use]
fn partial_monty_red_signed_to_signed<MPAVX512: MontyParametersAVX512>(input: __m512i) -> __m512i {
    unsafe {
        // We throw a confuse compiler here to prevent the compiler from
        // using vpmullq instead of vpmuludq in the computations for q_p.
        // vpmullq has both higher latency and lower throughput.
        let q = confuse_compiler(x86_64::_mm512_mul_epi32(input, MPAVX512::PACKED_MU));
        let q_p = x86_64::_mm512_mul_epi32(q, MPAVX512::PACKED_P);

        // This could equivalently be _mm512_sub_epi64
        x86_64::_mm512_sub_epi32(input, q_p)
    }
}

/// Viewing the input as a vector of 16 `u32`s, copy the odd elements into the even elements below
/// them. In other words, for all `0 <= i < 8`, set the even elements according to
/// `res[2 * i] := a[2 * i + 1]`, and the odd elements according to
/// `res[2 * i + 1] := a[2 * i + 1]`.
#[inline]
#[must_use]
fn movehdup_epi32(a: __m512i) -> __m512i {
    // The instruction is only available in the floating-point flavor; this distinction is only for
    // historical reasons and no longer matters. We cast to floats, do the thing, and cast back.
    unsafe {
        x86_64::_mm512_castps_si512(x86_64::_mm512_movehdup_ps(x86_64::_mm512_castsi512_ps(a)))
    }
}

/// Viewing the input as a vector of 8 `u32`s, copy the odd elements into the even elements below
/// them. In other words, for all `0 <= i < 4`, set the even elements according to
/// `res[2 * i] := a[2 * i + 1]`, and the odd elements according to
/// `res[2 * i + 1] := a[2 * i + 1]`.
#[inline]
#[must_use]
fn movehdup_epi32_256(a: __m256i) -> __m256i {
    // The instruction is only available in the floating-point flavor; this distinction is only for
    // historical reasons and no longer matters. We cast to floats, do the thing, and cast back.
    unsafe {
        x86_64::_mm256_castps_si256(x86_64::_mm256_movehdup_ps(x86_64::_mm256_castsi256_ps(a)))
    }
}

/// Viewing `a` as a vector of 16 `u32`s, copy the odd elements into the even elements below them,
/// then merge with `src` according to the mask provided. In other words, for all `0 <= i < 8`, set
/// the even elements according to `res[2 * i] := if k[2 * i] { a[2 * i + 1] } else { src[2 * i] }`,
/// and the odd elements according to
/// `res[2 * i + 1] := if k[2 * i + 1] { a[2 * i + 1] } else { src[2 * i + 1] }`.
#[inline]
#[must_use]
fn mask_movehdup_epi32(src: __m512i, k: __mmask16, a: __m512i) -> __m512i {
    // The instruction is only available in the floating-point flavor; this distinction is only for
    // historical reasons and no longer matters.

    // While we can write this using intrinsics, when inlined, the intrinsic often compiles
    // to a vpermt2ps which has worse latency, see https://godbolt.org/z/489aaPhz3.
    // Hence we use inline assembly to force the compiler to do the right thing.
    unsafe {
        let dst: __m512i;
        asm!(
            "vmovshdup {src_dst}{{{k}}}, {a}",
            src_dst = inlateout(zmm_reg) src => dst,
            k = in(kreg) k,
            a = in(zmm_reg) a,
            options(nomem, nostack, preserves_flags, pure),
        );
        dst
    }
}

/// Viewing `a` as a vector of 8 `u32`s, copy the odd elements into the even elements below them,
/// then merge with `src` according to the mask provided. In other words, for all `0 <= i < 4`, set
/// the even elements according to `res[2 * i] := if k[2 * i] { a[2 * i + 1] } else { src[2 * i] }`,
/// and the odd elements according to
/// `res[2 * i + 1] := if k[2 * i + 1] { a[2 * i + 1] } else { src[2 * i + 1] }`.
#[inline]
#[must_use]
fn mask_movehdup_epi32_256(src: __m256i, k: __mmask8, a: __m256i) -> __m256i {
    // The instruction is only available in the floating-point flavor; this distinction is only for
    // historical reasons and no longer matters.

    // While we can write this using intrinsics, when inlined, the intrinsic often compiles
    // to a vpermt2ps which has worse latency, see https://godbolt.org/z/489aaPhz3.
    // Hence we use inline assembly to force the compiler to do the right thing.
    unsafe {
        let dst: __m256i;
        asm!(
            "vmovshdup {src_dst}{{{k}}}, {a}",
            src_dst = inlateout(ymm_reg) src => dst,
            k = in(kreg) k,
            a = in(ymm_reg) a,
            options(nomem, nostack, preserves_flags, pure),
        );
        dst
    }
}

/// Multiply a vector of unsigned field elements return a vector of unsigned field elements lying in [0, P).
///
/// Note that the input does not need to be in canonical form but must satisfy
/// the bound `lhs * rhs < 2^32 * P`. If this bound is not satisfied, the result
/// is undefined.
#[inline]
#[must_use]
fn mul<MPAVX512: MontyParametersAVX512>(lhs: __m512i, rhs: __m512i) -> __m512i {
    // We want this to compile to:
    //      vmovshdup  lhs_odd, lhs
    //      vmovshdup  rhs_odd, rhs
    //      vpmuludq   prod_evn, lhs, rhs
    //      vpmuludq   prod_hi, lhs_odd, rhs_odd
    //      vpmuludq   q_evn, prod_evn, MU
    //      vpmuludq   q_odd, prod_hi, MU
    //      vmovshdup  prod_hi{EVENS}, prod_evn
    //      vpmuludq   q_p_evn, q_evn, P
    //      vpmuludq   q_p_hi, q_odd, P
    //      vmovshdup  q_p_hi{EVENS}, q_p_evn
    //      vpcmpltud  underflow, prod_hi, q_p_hi
    //      vpsubd     res, prod_hi, q_p_hi
    //      vpaddd     res{underflow}, res, P
    // throughput: 6.5 cyc/vec (2.46 els/cyc)
    // latency: 21 cyc
    unsafe {
        // `vpmuludq` only reads the even doublewords, so when we pass `lhs` and `rhs` directly we
        // get the eight products at even positions.
        let lhs_evn = lhs;
        let rhs_evn = rhs;

        // Copy the odd doublewords into even positions to compute the eight products at odd
        // positions.
        // NB: The odd doublewords are ignored by `vpmuludq`, so we have a lot of choices for how to
        // do this; `vmovshdup` is nice because it runs on a memory port if the operand is in
        // memory, thus improving our throughput.
        let lhs_odd = movehdup_epi32(lhs);
        let rhs_odd = movehdup_epi32(rhs);

        let prod_evn = x86_64::_mm512_mul_epu32(lhs_evn, rhs_evn);
        let prod_odd = x86_64::_mm512_mul_epu32(lhs_odd, rhs_odd);

        // We throw a confuse compiler here to prevent the compiler from
        // using vpmullq instead of vpmuludq in the computations for q_p.
        // vpmullq has both higher latency and lower throughput.
        let q_evn = confuse_compiler(x86_64::_mm512_mul_epu32(prod_evn, MPAVX512::PACKED_MU));
        let q_odd = confuse_compiler(x86_64::_mm512_mul_epu32(prod_odd, MPAVX512::PACKED_MU));

        // Get all the high halves as one vector: this is `(lhs * rhs) >> 32`.
        // NB: `vpermt2d` may feel like a more intuitive choice here, but it has much higher
        // latency.
        let prod_hi = mask_movehdup_epi32(prod_odd, EVENS, prod_evn);

        // Normally we'd want to mask to perform % 2**32, but the instruction below only reads the
        // low 32 bits anyway.
        let q_p_evn = x86_64::_mm512_mul_epu32(q_evn, MPAVX512::PACKED_P);
        let q_p_odd = x86_64::_mm512_mul_epu32(q_odd, MPAVX512::PACKED_P);

        // We can ignore all the low halves of `q_p` as they cancel out. Get all the high halves as
        // one vector.
        let q_p_hi = mask_movehdup_epi32(q_p_odd, EVENS, q_p_evn);

        // Subtraction `prod_hi - q_p_hi` modulo `P`.
        // NB: Normally we'd `vpaddd P` and take the `vpminud`, but `vpminud` runs on port 0, which
        // is already under a lot of pressure performing multiplications. To relieve this pressure,
        // we check for underflow to generate a mask, and then conditionally add `P`. The underflow
        // check runs on port 5, increasing our throughput, although it does cost us an additional
        // cycle of latency.
        let underflow = x86_64::_mm512_cmplt_epu32_mask(prod_hi, q_p_hi);
        let t = x86_64::_mm512_sub_epi32(prod_hi, q_p_hi);
        x86_64::_mm512_mask_add_epi32(t, underflow, t, MPAVX512::PACKED_P)
    }
}

/// Multiply a vector of unsigned field elements by a single field element.
///
/// Return a vector of unsigned field elements lying in [0, P).
///
/// Note that the input does not need to be in canonical form but must satisfy
/// the bound `lhs * rhs < 2^32 * P`. If this bound is not satisfied, the result
/// is undefined.
#[inline]
#[must_use]
fn mul_256<MPAVX512: MontyParametersAVX512>(lhs: __m256i, rhs: i32) -> __m256i {
    // We want this to compile to:
    //      vmovshdup  lhs_odd, lhs
    //      vpbroadcastd  rhs, rhs
    //      vpmuludq   prod_evn, lhs, rhs
    //      vpmuludq   prod_hi, lhs_odd, rhs_odd
    //      vpmuludq   q_evn, prod_evn, MU
    //      vpmuludq   q_odd, prod_hi, MU
    //      vmovshdup  prod_hi{EVENS}, prod_evn
    //      vpmuludq   q_p_evn, q_evn, P
    //      vpmuludq   q_p_hi, q_odd, P
    //      vmovshdup  q_p_hi{EVENS}, q_p_evn
    //      vpcmpltud  underflow, prod_hi, q_p_hi
    //      vpsubd     res, prod_hi, q_p_hi
    //      vpaddd     res{underflow}, res, P
    // throughput: 6.5 cyc/vec (2.46 els/cyc)
    // latency: 21 cyc
    unsafe {
        // `vpmuludq` only reads the even doublewords, so when we pass `lhs` directly we
        // get the four products at even positions.
        let lhs_evn = lhs;
        let rhs = x86_64::_mm256_set1_epi32(rhs);

        // Copy the odd doublewords into even positions to compute the four products at odd
        // positions.
        // NB: The odd doublewords are ignored by `vpmuludq`, so we have a lot of choices for how to
        // do this; `vmovshdup` is nice because it runs on a memory port if the operand is in
        // memory, thus improving our throughput.
        let lhs_odd = movehdup_epi32_256(lhs);

        let prod_evn = x86_64::_mm256_mul_epu32(lhs_evn, rhs);
        let prod_odd = x86_64::_mm256_mul_epu32(lhs_odd, rhs);

        let mu_256 = x86_64::_mm512_castsi512_si256(MPAVX512::PACKED_MU);
        let q_evn = confuse_compiler_256(x86_64::_mm256_mul_epu32(prod_evn, mu_256));
        let q_odd = confuse_compiler_256(x86_64::_mm256_mul_epu32(prod_odd, mu_256));

        // Get all the high halves as one vector: this is `(lhs * rhs) >> 32`.
        // NB: `vpermt2d` may feel like a more intuitive choice here, but it has much higher
        // latency.
        let prod_hi = mask_movehdup_epi32_256(prod_odd, EVENS_8, prod_evn);

        // Normally we'd want to mask to perform % 2**32, but the instruction below only reads the
        // low 32 bits anyway.
        let p_256 = x86_64::_mm512_castsi512_si256(MPAVX512::PACKED_P);
        let q_p_evn = x86_64::_mm256_mul_epu32(q_evn, p_256);
        let q_p_odd = x86_64::_mm256_mul_epu32(q_odd, p_256);

        // We can ignore all the low halves of `q_p` as they cancel out. Get all the high halves as
        // one vector.
        let q_p_hi = mask_movehdup_epi32_256(q_p_odd, EVENS_8, q_p_evn);

        // Subtraction `prod_hi - q_p_hi` modulo `P`.
        // NB: Normally we'd `vpaddd P` and take the `vpminud`, but `vpminud` runs on port 0, which
        // is already under a lot of pressure performing multiplications. To relieve this pressure,
        // we check for underflow to generate a mask, and then conditionally add `P`. The underflow
        // check runs on port 5, increasing our throughput, although it does cost us an additional
        // cycle of latency.
        let underflow = x86_64::_mm256_cmplt_epu32_mask(prod_hi, q_p_hi);
        let t = x86_64::_mm256_sub_epi32(prod_hi, q_p_hi);
        x86_64::_mm256_mask_add_epi32(t, underflow, t, p_256)
    }
}

/// Compute the elementary arithmetic generalization of `xor`, namely `xor(l, r) = l + r - 2lr` of
/// vectors in canonical form.
///
/// Inputs are assumed to be in canonical form, if the inputs are not in canonical form, the result is undefined.
#[inline]
#[must_use]
fn xor<MPAVX512: MontyParametersAVX512>(lhs: __m512i, rhs: __m512i) -> __m512i {
    // Refactor the expression as r + 2l(1/2 - r). As MONTY_CONSTANT = 2^32, the internal
    // representation 1/2 is 2^31 mod P so the product in the above expression is represented
    // as 2l(2^31 - r). As 0 < 2l, 2^31 - r < 2^32 and 2l(2^31 - r) < 2^32P, we can compute
    // the factors as 32 bit integers and then multiply and monty reduce as usual.
    //
    // We want this to compile to:
    //      vpaddd     lhs_double, lhs, lhs
    //      vpsubd     sub_rhs, (1 << 31), rhs
    //      vmovshdup  lhs_odd, lhs_double
    //      vmovshdup  rhs_odd, sub_rhs
    //      vpmuludq   prod_evn, lhs_double, sub_rhs
    //      vpmuludq   prod_hi, lhs_odd, rhs_odd
    //      vpmuludq   q_evn, prod_evn, MU
    //      vpmuludq   q_odd, prod_hi, MU
    //      vmovshdup  prod_hi{EVENS}, prod_evn
    //      vpmuludq   q_p_evn, q_evn, P
    //      vpmuludq   q_p_hi, q_odd, P
    //      vmovshdup  q_p_hi{EVENS}, q_p_evn
    //      vpcmpltud  underflow, prod_hi, q_p_hi
    //      vpsubd     res, prod_hi, q_p_hi
    //      vpaddd     res{underflow}, res, P
    //      vpaddd     sum,        rhs,   t
    //      vpsubd     sum_corr,   sum,   pos_neg_P
    //      vpminud    res,        sum,   sum_corr
    // throughput: 9 cyc/vec (1.77 els/cyc)
    // latency: 25 cyc
    unsafe {
        // 0 <= 2*lhs < 2P
        let double_lhs = x86_64::_mm512_add_epi32(lhs, lhs);

        // Note that 2^31 is represented as an i32 as (-2^31).
        // Compiler should realise this is a constant.
        let half = x86_64::_mm512_set1_epi32(-1 << 31);

        // 0 < 2^31 - rhs < 2^31
        let half_sub_rhs = x86_64::_mm512_sub_epi32(half, rhs);

        // 2*lhs (2^31 - rhs) < 2P 2^31 < 2^32P so we can use the multiplication function.
        let mul_res = mul::<MPAVX512>(double_lhs, half_sub_rhs);

        // Unfortunately, AVX512 has no equivalent of vpsignd so we can't do the same
        // signed_add trick as in the AVX2 case. Instead we get a reduced value from mul
        // and add on rhs in the standard way.
        mm512_mod_add(rhs, mul_res, MPAVX512::PACKED_P)
    }
}

/// Compute the elementary arithmetic generalization of `andnot`, namely `andn(l, r) = (1 - l)r` of
/// vectors in canonical form.
///
/// Inputs are assumed to be in canonical form, if the inputs are not in canonical form, the result is undefined.
#[inline]
#[must_use]
fn andn<MPAVX512: MontyParametersAVX512>(lhs: __m512i, rhs: __m512i) -> __m512i {
    // As we are working with MONTY_CONSTANT = 2^32, the internal representation
    // of 1 is 2^32 mod P = 2^32 - P mod P. Hence we compute (2^32 - P - l)r.
    // This product is less than 2^32P so we can apply our monty reduction to this.
    //
    // We want this to compile to:
    //      vpsubd     neg_lhs, 2^32 - P, lhs
    //      vmovshdup  lhs_odd, neg_lhs
    //      vmovshdup  rhs_odd, rhs
    //      vpmuludq   prod_evn, neg_lhs, rhs
    //      vpmuludq   prod_hi, lhs_odd, rhs_odd
    //      vpmuludq   q_evn, prod_evn, MU
    //      vpmuludq   q_odd, prod_hi, MU
    //      vmovshdup  prod_hi{EVENS}, prod_evn
    //      vpmuludq   q_p_evn, q_evn, P
    //      vpmuludq   q_p_hi, q_odd, P
    //      vmovshdup  q_p_hi{EVENS}, q_p_evn
    //      vpcmpltud  underflow, prod_hi, q_p_hi
    //      vpsubd     res, prod_hi, q_p_hi
    //      vpaddd     res{underflow}, res, P
    // throughput: 7 cyc/vec (2.3 els/cyc)
    // latency: 22 cyc
    unsafe {
        // We use 2^32 - P instead of 2^32 to avoid having to worry about 0's in lhs.

        // Compiler should realise that this is a constant.
        let neg_p = x86_64::_mm512_sub_epi32(x86_64::_mm512_setzero_epi32(), MPAVX512::PACKED_P);
        let neg_lhs = x86_64::_mm512_sub_epi32(neg_p, lhs);

        // 2*lhs (2^31 - rhs) < 2P 2^31 < 2^32P so we can use the multiplication function.
        mul::<MPAVX512>(neg_lhs, rhs)
    }
}

/// Square the MontyField31 elements in the even index entries.
/// Inputs must be signed 32-bit integers in [-P, ..., P].
/// Outputs will be a signed integer in (-P, ..., P) copied into both the even and odd indices.
#[inline]
#[must_use]
fn shifted_square<MPAVX512: MontyParametersAVX512>(input: __m512i) -> __m512i {
    // Note that we do not need a restriction on the size of input[i]^2 as
    // 2^30 < P and |i32| <= 2^31 and so => input[i]^2 <= 2^62 < 2^32P.
    unsafe {
        let square = x86_64::_mm512_mul_epi32(input, input);
        let square_red = partial_monty_red_unsigned_to_signed::<MPAVX512>(square);
        movehdup_epi32(square_red)
    }
}

/// Cube the MontyField31 elements in the even index entries.
/// Inputs must be signed 32-bit integers in [-P, ..., P].
/// Outputs will be signed integers in (-P^2, ..., P^2).
#[inline]
#[must_use]
pub(crate) fn packed_exp_3<MPAVX512: MontyParametersAVX512>(input: __m512i) -> __m512i {
    unsafe {
        let square = shifted_square::<MPAVX512>(input);
        x86_64::_mm512_mul_epi32(square, input)
    }
}

/// Take the fifth power of the MontyField31 elements in the even index entries.
/// Inputs must be signed 32-bit integers in [-P, ..., P].
/// Outputs will be signed integers in (-P^2, ..., P^2).
#[inline]
#[must_use]
pub(crate) fn packed_exp_5<MPAVX512: MontyParametersAVX512>(input: __m512i) -> __m512i {
    unsafe {
        let square = shifted_square::<MPAVX512>(input);
        let quad = shifted_square::<MPAVX512>(square);
        x86_64::_mm512_mul_epi32(quad, input)
    }
}

/// Take the seventh power of the MontyField31 elements in the even index entries.
/// Inputs must lie in [-P, ..., P].
/// Outputs will be signed integers in (-P^2, ..., P^2).
#[inline]
#[must_use]
pub(crate) fn packed_exp_7<MPAVX512: MontyParametersAVX512>(input: __m512i) -> __m512i {
    unsafe {
        let square = shifted_square::<MPAVX512>(input);
        let cube_raw = x86_64::_mm512_mul_epi32(square, input);
        let cube_red = partial_monty_red_signed_to_signed::<MPAVX512>(cube_raw);
        let cube = movehdup_epi32(cube_red);
        let quad = shifted_square::<MPAVX512>(square);
        x86_64::_mm512_mul_epi32(quad, cube)
    }
}

/// Apply func to the even and odd indices of the input vector.
///
/// func should only depend in the 32 bit entries in the even indices.
/// The input should conform to the requirements of `func`.
/// The output of func must lie in (-P^2, ..., P^2) after which
/// apply_func_to_even_odd will reduce the outputs to lie in [0, P)
/// and recombine the odd and even parts.
#[inline]
#[must_use]
pub(crate) unsafe fn apply_func_to_even_odd<MPAVX512: MontyParametersAVX512>(
    input: __m512i,
    func: fn(__m512i) -> __m512i,
) -> __m512i {
    unsafe {
        let input_evn = input;
        let input_odd = movehdup_epi32(input);

        let output_even = func(input_evn);
        let output_odd = func(input_odd);

        // We need to recombine these even and odd parts and, at the same time reduce back to
        // an output in [0, P).

        // We throw a confuse compiler here to prevent the compiler from
        // using vpmullq instead of vpmuludq in the computations for q_p.
        // vpmullq has both higher latency and lower throughput.
        let q_evn = confuse_compiler(x86_64::_mm512_mul_epi32(output_even, MPAVX512::PACKED_MU));
        let q_odd = confuse_compiler(x86_64::_mm512_mul_epi32(output_odd, MPAVX512::PACKED_MU));

        // Get all the high halves as one vector: this is `(lhs * rhs) >> 32`.
        // NB: `vpermt2d` may feel like a more intuitive choice here, but it has much higher
        // latency.
        let output_hi = mask_movehdup_epi32(output_odd, EVENS, output_even);

        // Normally we'd want to mask to perform % 2**32, but the instruction below only reads the
        // low 32 bits anyway.
        let q_p_evn = x86_64::_mm512_mul_epi32(q_evn, MPAVX512::PACKED_P);
        let q_p_odd = x86_64::_mm512_mul_epi32(q_odd, MPAVX512::PACKED_P);

        // We can ignore all the low halves of `q_p` as they cancel out. Get all the high halves as
        // one vector.
        let q_p_hi = mask_movehdup_epi32(q_p_odd, EVENS, q_p_evn);

        // Subtraction `output_hi - q_p_hi` modulo `P`.
        // NB: Normally we'd `vpaddd P` and take the `vpminud`, but `vpminud` runs on port 0, which
        // is already under a lot of pressure performing multiplications. To relieve this pressure,
        // we check for underflow to generate a mask, and then conditionally add `P`. The underflow
        // check runs on port 5, increasing our throughput, although it does cost us an additional
        // cycle of latency.
        let underflow = x86_64::_mm512_cmplt_epi32_mask(output_hi, q_p_hi);
        let t = x86_64::_mm512_sub_epi32(output_hi, q_p_hi);
        x86_64::_mm512_mask_add_epi32(t, underflow, t, MPAVX512::PACKED_P)
    }
}

/// Negate a vector of MontyField31 elements in canonical form.
/// If the inputs are not in canonical form, the result is undefined.
#[inline]
#[must_use]
fn neg<MPAVX512: MontyParametersAVX512>(val: __m512i) -> __m512i {
    // We want this to compile to:
    //      vptestmd  nonzero, val, val
    //      vpsubd    res{nonzero}{z}, P, val
    // throughput: 1 cyc/vec (16 els/cyc)
    // latency: 4 cyc

    // NB: This routine prioritizes throughput over latency. An alternative method would be to do
    // sub(0, val), which would result in shorter latency, but also lower throughput.

    //   If val is nonzero, then val is in {1, ..., P - 1} and P - val is in the same range. If val
    // is zero, then the result is zeroed by masking.
    unsafe {
        // Safety: If this code got compiled then AVX-512F intrinsics are available.
        let nonzero = x86_64::_mm512_test_epi32_mask(val, val);
        x86_64::_mm512_maskz_sub_epi32(nonzero, MPAVX512::PACKED_P, val)
    }
}

/// Lets us combine some code for MontyField31<FP> and PackedMontyField31AVX2<FP> elements.
///
/// Provides methods to convert an element into a __m512i element and then shift this __m512i
/// element so that the odd elements now lie in the even positions. Depending on the type of input,
/// the shift might be a no-op.
trait IntoM512<PMP: PackedMontyParameters>: Copy + Into<PackedMontyField31AVX512<PMP>> {
    /// Convert the input into a __m512i element.
    fn as_m512i(&self) -> __m512i;

    /// Convert the input to a __m512i element and shift so that all elements in odd positions
    /// now lie in even positions.
    ///
    /// The values lying in the even positions are undefined.
    #[inline(always)]
    fn as_shifted_m512i(&self) -> __m512i {
        let vec = self.as_m512i();
        movehdup_epi32(vec)
    }
}

impl<PMP: PackedMontyParameters> IntoM512<PMP> for PackedMontyField31AVX512<PMP> {
    #[inline(always)]
    fn as_m512i(&self) -> __m512i {
        self.to_vector()
    }
}

impl<PMP: PackedMontyParameters> IntoM512<PMP> for MontyField31<PMP> {
    #[inline(always)]
    fn as_m512i(&self) -> __m512i {
        unsafe { x86_64::_mm512_set1_epi32(self.value as i32) }
    }

    #[inline(always)]
    fn as_shifted_m512i(&self) -> __m512i {
        unsafe { x86_64::_mm512_set1_epi32(self.value as i32) }
    }
}

/// Compute the elementary function `l0*r0 + l1*r1` given four inputs
/// in canonical form.
///
/// If the inputs are not in canonical form, the result is undefined.
#[inline]
#[must_use]
fn dot_product_2<PMP: PackedMontyParameters, LHS: IntoM512<PMP>, RHS: IntoM512<PMP>>(
    lhs: [LHS; 2],
    rhs: [RHS; 2],
) -> __m512i {
    // The following analysis treats all input arrays as being arrays of PackedMontyField31AVX512<FP>.
    // If one of the arrays contains MontyField31<FP>, we get to avoid the initial vmovshdup.
    //
    // We improve the throughput by combining the monty reductions together. As all inputs are
    // `< P < 2^{31}`, `l0*r0 + l1*r1 < 2P^2 < 2^{32}P` so the montgomery reduction
    // algorithm can be applied to the sum of the products instead of to each product individually.
    //
    // We want this to compile to:
    //      vmovshdup  lhs_odd0, lhs0
    //      vmovshdup  rhs_odd0, rhs0
    //      vmovshdup  lhs_odd1, lhs1
    //      vmovshdup  rhs_odd1, rhs1
    //      vpmuludq   prod_evn0, lhs0, rhs0
    //      vpmuludq   prod_odd0, lhs_odd0, rhs_odd0
    //      vpmuludq   prod_evn1, lhs1, rhs1
    //      vpmuludq   prod_odd1, lhs_odd1, rhs_odd1
    //      vpaddq     dot_evn, prod_evn0, prod_evn1
    //      vpaddq     dot, prod_odd0, prod_odd1
    //      vpmuludq   q_evn, prod_evn, MU
    //      vpmuludq   q_odd, dot, MU
    //      vpmuludq   q_P_evn, q_evn, P
    //      vpmuludq   q_P, q_odd, P
    //      vmovshdup  dot{EVENS} dot_evn
    //      vmovshdup  q_P{EVENS} q_P_evn
    //      vpcmpltud  underflow, dot, q_P
    //      vpsubd     res, dot, q_P
    //      vpaddd     res{underflow}, res, P
    // throughput: 9.5 cyc/vec (1.68 els/cyc)
    // latency: 22 cyc
    unsafe {
        let lhs_evn0 = lhs[0].as_m512i();
        let lhs_odd0 = lhs[0].as_shifted_m512i();
        let lhs_evn1 = lhs[1].as_m512i();
        let lhs_odd1 = lhs[1].as_shifted_m512i();

        let rhs_evn0 = rhs[0].as_m512i();
        let rhs_odd0 = rhs[0].as_shifted_m512i();
        let rhs_evn1 = rhs[1].as_m512i();
        let rhs_odd1 = rhs[1].as_shifted_m512i();

        let mul_evn0 = x86_64::_mm512_mul_epu32(lhs_evn0, rhs_evn0);
        let mul_evn1 = x86_64::_mm512_mul_epu32(lhs_evn1, rhs_evn1);
        let mul_odd0 = x86_64::_mm512_mul_epu32(lhs_odd0, rhs_odd0);
        let mul_odd1 = x86_64::_mm512_mul_epu32(lhs_odd1, rhs_odd1);

        let dot_evn = x86_64::_mm512_add_epi64(mul_evn0, mul_evn1);
        let dot_odd = x86_64::_mm512_add_epi64(mul_odd0, mul_odd1);

        // We throw a confuse compiler here to prevent the compiler from
        // using vpmullq instead of vpmuludq in the computations for q_p.
        // vpmullq has both higher latency and lower throughput.
        let q_evn = confuse_compiler(x86_64::_mm512_mul_epu32(dot_evn, PMP::PACKED_MU));
        let q_odd = confuse_compiler(x86_64::_mm512_mul_epu32(dot_odd, PMP::PACKED_MU));

        // Get all the high halves as one vector: this is `dot(lhs, rhs) >> 32`.
        // NB: `vpermt2d` may feel like a more intuitive choice here, but it has much higher
        // latency.
        let dot = mask_movehdup_epi32(dot_odd, EVENS, dot_evn);

        // Normally we'd want to mask to perform % 2**32, but the instruction below only reads the
        // low 32 bits anyway.
        let q_p_evn = x86_64::_mm512_mul_epu32(q_evn, PMP::PACKED_P);
        let q_p_odd = x86_64::_mm512_mul_epu32(q_odd, PMP::PACKED_P);

        // We can ignore all the low halves of `q_p` as they cancel out. Get all the high halves as
        // one vector.
        let q_p = mask_movehdup_epi32(q_p_odd, EVENS, q_p_evn);

        // Subtraction `prod_hi - q_p_hi` modulo `P`.
        // NB: Normally we'd `vpaddd P` and take the `vpminud`, but `vpminud` runs on port 0, which
        // is already under a lot of pressure performing multiplications. To relieve this pressure,
        // we check for underflow to generate a mask, and then conditionally add `P`. The underflow
        // check runs on port 5, increasing our throughput, although it does cost us an additional
        // cycle of latency.
        let underflow = x86_64::_mm512_cmplt_epu32_mask(dot, q_p);
        let t = x86_64::_mm512_sub_epi32(dot, q_p);
        x86_64::_mm512_mask_add_epi32(t, underflow, t, PMP::PACKED_P)
    }
}

/// Compute the elementary function `l0*r0 + l1*r1 + l2*r2 + l3*r3` given eight inputs
/// in canonical form.
///
/// If the inputs are not in canonical form, the result is undefined.
#[inline]
#[must_use]
fn dot_product_4<PMP: PackedMontyParameters, LHS: IntoM512<PMP>, RHS: IntoM512<PMP>>(
    lhs: [LHS; 4],
    rhs: [RHS; 4],
) -> __m512i {
    // The following analysis treats all input arrays as being arrays of PackedMontyField31AVX512<FP>.
    // If one of the arrays contains MontyField31<FP>, we get to avoid the initial vmovshdup.
    //
    // Similarly to dot_product_2, we improve throughput by combining monty reductions however in this case
    // we will need to slightly adjust the reduction algorithm.
    //
    // As all inputs are `< P < 2^{31}`, the sum satisfies: `C = l0*r0 + l1*r1 + l2*r2 + l3*r3 < 4P^2 < 2*2^{32}P`.
    // Start by computing Q := μ C mod B as usual.
    // We can't proceed as normal however as 2*2^{32}P > C - QP > -2^{32}P which doesn't fit into an i64.
    // Instead we do a reduction on C, defining C' = if C < 2^{32}P: {C} else {C - 2^{32}P}
    // From here we proceed with the standard montgomery reduction with C replaced by C'. It works identically
    // with the Q we already computed as C = C' mod B.
    //
    // We want this to compile to:
    //      vmovshdup  lhs_odd0, lhs0
    //      vmovshdup  rhs_odd0, rhs0
    //      vmovshdup  lhs_odd1, lhs1
    //      vmovshdup  rhs_odd1, rhs1
    //      vmovshdup  lhs_odd2, lhs2
    //      vmovshdup  rhs_odd2, rhs2
    //      vmovshdup  lhs_odd3, lhs3
    //      vmovshdup  rhs_odd3, rhs3
    //      vpmuludq   prod_evn0, lhs0, rhs0
    //      vpmuludq   prod_odd0, lhs_odd0, rhs_odd0
    //      vpmuludq   prod_evn1, lhs1, rhs1
    //      vpmuludq   prod_odd1, lhs_odd1, rhs_odd1
    //      vpmuludq   prod_evn2, lhs2, rhs2
    //      vpmuludq   prod_odd2, lhs_odd2, rhs_odd2
    //      vpmuludq   prod_evn3, lhs3, rhs3
    //      vpmuludq   prod_odd3, lhs_odd3, rhs_odd3
    //      vpaddq     dot_evn01, prod_evn0, prod_evn1
    //      vpaddq     dot_odd01, prod_odd0, prod_odd1
    //      vpaddq     dot_evn23, prod_evn2, prod_evn3
    //      vpaddq     dot_odd23, prod_odd2, prod_odd3
    //      vpaddq     dot_evn, dot_evn01, dot_evn23
    //      vpaddq     dot, dot_odd01, dot_odd23
    //      vpmuludq   q_evn, dot_evn, MU
    //      vpmuludq   q_odd, dot, MU
    //      vpmuludq   q_P_evn, q_evn, P
    //      vpmuludq   q_P, q_odd, P
    //      vmovshdup  dot{EVENS} dot_evn
    //      vpcmpleud  over_P, P, dot
    //      vpsubd     dot{underflow}, dot, P
    //      vmovshdup  q_P{EVENS} q_P_evn
    //      vpcmpltud  underflow, dot, q_P
    //      vpsubd     res, dot, q_P
    //      vpaddd     res{underflow}, res, P
    // throughput: 16.5 cyc/vec (0.97 els/cyc)
    // latency: 23 cyc
    unsafe {
        let lhs_evn0 = lhs[0].as_m512i();
        let lhs_odd0 = lhs[0].as_shifted_m512i();
        let lhs_evn1 = lhs[1].as_m512i();
        let lhs_odd1 = lhs[1].as_shifted_m512i();
        let lhs_evn2 = lhs[2].as_m512i();
        let lhs_odd2 = lhs[2].as_shifted_m512i();
        let lhs_evn3 = lhs[3].as_m512i();
        let lhs_odd3 = lhs[3].as_shifted_m512i();

        let rhs_evn0 = rhs[0].as_m512i();
        let rhs_odd0 = rhs[0].as_shifted_m512i();
        let rhs_evn1 = rhs[1].as_m512i();
        let rhs_odd1 = rhs[1].as_shifted_m512i();
        let rhs_evn2 = rhs[2].as_m512i();
        let rhs_odd2 = rhs[2].as_shifted_m512i();
        let rhs_evn3 = rhs[3].as_m512i();
        let rhs_odd3 = rhs[3].as_shifted_m512i();

        let mul_evn0 = x86_64::_mm512_mul_epu32(lhs_evn0, rhs_evn0);
        let mul_evn1 = x86_64::_mm512_mul_epu32(lhs_evn1, rhs_evn1);
        let mul_evn2 = x86_64::_mm512_mul_epu32(lhs_evn2, rhs_evn2);
        let mul_evn3 = x86_64::_mm512_mul_epu32(lhs_evn3, rhs_evn3);
        let mul_odd0 = x86_64::_mm512_mul_epu32(lhs_odd0, rhs_odd0);
        let mul_odd1 = x86_64::_mm512_mul_epu32(lhs_odd1, rhs_odd1);
        let mul_odd2 = x86_64::_mm512_mul_epu32(lhs_odd2, rhs_odd2);
        let mul_odd3 = x86_64::_mm512_mul_epu32(lhs_odd3, rhs_odd3);

        let dot_evn01 = x86_64::_mm512_add_epi64(mul_evn0, mul_evn1);
        let dot_odd01 = x86_64::_mm512_add_epi64(mul_odd0, mul_odd1);
        let dot_evn23 = x86_64::_mm512_add_epi64(mul_evn2, mul_evn3);
        let dot_odd23 = x86_64::_mm512_add_epi64(mul_odd2, mul_odd3);

        let dot_evn = x86_64::_mm512_add_epi64(dot_evn01, dot_evn23);
        let dot_odd = x86_64::_mm512_add_epi64(dot_odd01, dot_odd23);

        // We throw a confuse compiler here to prevent the compiler from
        // using vpmullq instead of vpmuludq in the computations for q_p.
        // vpmullq has both higher latency and lower throughput.
        let q_evn = confuse_compiler(x86_64::_mm512_mul_epu32(dot_evn, PMP::PACKED_MU));
        let q_odd = confuse_compiler(x86_64::_mm512_mul_epu32(dot_odd, PMP::PACKED_MU));

        // Get all the high halves as one vector: this is `dot(lhs, rhs) >> 32`.
        // NB: `vpermt2d` may feel like a more intuitive choice here, but it has much higher
        // latency.
        let dot = mask_movehdup_epi32(dot_odd, EVENS, dot_evn);

        // The elements in dot lie in [0, 2P) so we need to reduce them to [0, P)
        // NB: Normally we'd `vpsubq P` and take the `vpminud`, but `vpminud` runs on port 0, which
        // is already under a lot of pressure performing multiplications. To relieve this pressure,
        // we check for underflow to generate a mask, and then conditionally add `P`.
        let over_p = x86_64::_mm512_cmple_epu32_mask(PMP::PACKED_P, dot);
        let dot_corr = x86_64::_mm512_mask_sub_epi32(dot, over_p, dot, PMP::PACKED_P);

        // Normally we'd want to mask to perform % 2**32, but the instruction below only reads the
        // low 32 bits anyway.
        let q_p_evn = x86_64::_mm512_mul_epu32(q_evn, PMP::PACKED_P);
        let q_p_odd = x86_64::_mm512_mul_epu32(q_odd, PMP::PACKED_P);

        // We can ignore all the low halves of `q_p` as they cancel out. Get all the high halves as
        // one vector.
        let q_p = mask_movehdup_epi32(q_p_odd, EVENS, q_p_evn);

        // Subtraction `prod_hi - q_p_hi` modulo `P`.
        // NB: Normally we'd `vpaddd P` and take the `vpminud`, but `vpminud` runs on port 0, which
        // is already under a lot of pressure performing multiplications. To relieve this pressure,
        // we check for underflow to generate a mask, and then conditionally add `P`. The underflow
        // check runs on port 5, increasing our throughput, although it does cost us an additional
        // cycle of latency.
        let underflow = x86_64::_mm512_cmplt_epu32_mask(dot_corr, q_p);
        let t = x86_64::_mm512_sub_epi32(dot_corr, q_p);
        x86_64::_mm512_mask_add_epi32(t, underflow, t, PMP::PACKED_P)
    }
}

/// A general fast dot product implementation.
///
/// Maximises the number of calls to `dot_product_4` for dot products involving vectors of length
/// more than 4. The length 64 occurs commonly enough it's useful to have a custom implementation
/// which lets it use a slightly better summation algorithm with lower latency.
#[inline(always)]
fn general_dot_product<
    FP: FieldParameters,
    LHS: IntoM512<FP>,
    RHS: IntoM512<FP>,
    const N: usize,
>(
    lhs: &[LHS],
    rhs: &[RHS],
) -> PackedMontyField31AVX512<FP> {
    assert_eq!(lhs.len(), N);
    assert_eq!(rhs.len(), N);
    match N {
        0 => PackedMontyField31AVX512::<FP>::ZERO,
        1 => (lhs[0]).into() * (rhs[0]).into(),
        2 => {
            let res = dot_product_2([lhs[0], lhs[1]], [rhs[0], rhs[1]]);
            unsafe {
                // Safety: `dot_product_2` returns values in canonical form when given values in canonical form.
                PackedMontyField31AVX512::<FP>::from_vector(res)
            }
        }
        3 => {
            let lhs2 = lhs[2];
            let rhs2 = rhs[2];
            let res = dot_product_2([lhs[0], lhs[1]], [rhs[0], rhs[1]]);
            unsafe {
                // Safety: `dot_product_2` returns values in canonical form when given values in canonical form.
                PackedMontyField31AVX512::<FP>::from_vector(res) + (lhs2.into() * rhs2.into())
            }
        }
        4 => {
            let res = dot_product_4(
                [lhs[0], lhs[1], lhs[2], lhs[3]],
                [rhs[0], rhs[1], rhs[2], rhs[3]],
            );
            unsafe {
                // Safety: `dot_product_4` returns values in canonical form when given values in canonical form.
                PackedMontyField31AVX512::<FP>::from_vector(res)
            }
        }
        64 => {
            let sum_4s: [PackedMontyField31AVX512<FP>; 16] = array::from_fn(|i| {
                let res = dot_product_4(
                    [lhs[4 * i], lhs[4 * i + 1], lhs[4 * i + 2], lhs[4 * i + 3]],
                    [rhs[4 * i], rhs[4 * i + 1], rhs[4 * i + 2], rhs[4 * i + 3]],
                );
                unsafe {
                    // Safety: `dot_product_4` returns values in canonical form when given values in canonical form.
                    PackedMontyField31AVX512::<FP>::from_vector(res)
                }
            });
            PackedMontyField31AVX512::<FP>::sum_array::<16>(&sum_4s)
        }
        _ => {
            let mut acc = {
                let res = dot_product_4(
                    [lhs[0], lhs[1], lhs[2], lhs[3]],
                    [rhs[0], rhs[1], rhs[2], rhs[3]],
                );
                unsafe {
                    // Safety: `dot_product_4` returns values in canonical form when given values in canonical form.
                    PackedMontyField31AVX512::<FP>::from_vector(res)
                }
            };
            for i in (4..(N - 3)).step_by(4) {
                let res = dot_product_4(
                    [lhs[i], lhs[i + 1], lhs[i + 2], lhs[i + 3]],
                    [rhs[i], rhs[i + 1], rhs[i + 2], rhs[i + 3]],
                );
                unsafe {
                    // Safety: `dot_product_4` returns values in canonical form when given values in canonical form.
                    acc += PackedMontyField31AVX512::<FP>::from_vector(res)
                }
            }
            match N & 3 {
                0 => acc,
                1 => {
                    acc + general_dot_product::<_, _, _, 1>(
                        &lhs[(4 * (N / 4))..],
                        &rhs[(4 * (N / 4))..],
                    )
                }
                2 => {
                    acc + general_dot_product::<_, _, _, 2>(
                        &lhs[(4 * (N / 4))..],
                        &rhs[(4 * (N / 4))..],
                    )
                }
                3 => {
                    acc + general_dot_product::<_, _, _, 3>(
                        &lhs[(4 * (N / 4))..],
                        &rhs[(4 * (N / 4))..],
                    )
                }
                _ => unreachable!(),
            }
        }
    }
}

impl_packed_value!(
    PackedMontyField31AVX512,
    MontyField31,
    WIDTH,
    (PackedMontyParameters, PMP)
);

unsafe impl<FP: FieldParameters> PackedField for PackedMontyField31AVX512<FP> {
    type Scalar = MontyField31<FP>;

    #[inline]
    fn packed_linear_combination<const N: usize>(coeffs: &[Self::Scalar], vecs: &[Self]) -> Self {
        general_dot_product::<_, _, _, N>(coeffs, vecs)
    }
}

impl_packed_field_pow_2!(
    PackedMontyField31AVX512, (FieldParameters, FP);
    [
        (1, interleave_u32),
        (2, interleave_u64),
        (4, interleave_u128),
        (8, interleave_u256)
    ],
    WIDTH
);

/// Multiplication in a quartic binomial extension field.
#[inline]
pub(crate) fn quartic_mul_packed<FP, const WIDTH: usize>(
    a: &[MontyField31<FP>; WIDTH],
    b: &[MontyField31<FP>; WIDTH],
    res: &mut [MontyField31<FP>; WIDTH],
) where
    FP: FieldParameters + BinomialExtensionData<WIDTH>,
{
    // TODO: It's plausible that this could be improved by folding the computation of packed_b into
    // the custom AVX512 implementation. Moreover, a custom implementation which mixes AVX512 and
    // AVX2 code might well be able to improve on the one that is here.
    assert_eq!(WIDTH, 4);

    // No point in using packings here as we only have 3 elements. It might be worth using a smaller packed
    // field (e.g AVX2) but right now we don't compile both PackedMontyField31AVX2 and PackedMontyField31AVX512
    // at the same time.
    let w_b1 = FP::mul_w(b[1]);
    let w_b2 = FP::mul_w(b[2]);
    let w_b3 = FP::mul_w(b[3]);

    // Constant term = a0*b0 + w(a1*b3 + a2*b2 + a3*b1)
    // Linear term = a0*b1 + a1*b0 + w(a2*b3 + a3*b2)
    // Square term = a0*b2 + a1*b1 + a2*b0 + w(a3*b3)
    // Cubic term = a0*b3 + a1*b2 + a2*b1 + a3*b0
    // The constant term will be computed in the first 128bits, the linear term in the second 128bits,
    // the square term in the third 128bits and the cubic term in the fourth 128bits.
    let dot_product: [MontyField31<FP>; 8] = unsafe {
        // Surprisingly, testing indicates it's actually faster to just use AVX2 intrinsics here instead of AVX512.
        // Hence this is just copied and pasted from the AVX2 implementation. the issue is likely that there isn't
        // a huge amount of parallelism to be had and so using AVX512 leads to a bunch of extra data fiddiling.
        let lhs_0 = x86_64::_mm256_set1_epi32(a[0].value as i32);
        let lhs_1 = x86_64::_mm256_set1_epi32(a[1].value as i32);
        let lhs_2 = x86_64::_mm256_set1_epi32(a[2].value as i32);
        let lhs_3 = x86_64::_mm256_set1_epi32(a[3].value as i32);

        // We use setr instead of set as setr reverses the order of the arguments
        // for some reason.
        let rhs_0 = x86_64::_mm256_setr_epi32(
            b[0].value as i32,
            0,
            b[1].value as i32,
            0,
            b[2].value as i32,
            0,
            b[3].value as i32,
            0,
        );
        let rhs_1 = x86_64::_mm256_setr_epi32(
            w_b3.value as i32,
            0,
            b[0].value as i32,
            0,
            b[1].value as i32,
            0,
            b[2].value as i32,
            0,
        );
        let rhs_2 = x86_64::_mm256_setr_epi32(
            w_b2.value as i32,
            0,
            w_b3.value as i32,
            0,
            b[0].value as i32,
            0,
            b[1].value as i32,
            0,
        );
        let rhs_3 = x86_64::_mm256_setr_epi32(
            w_b1.value as i32,
            0,
            w_b2.value as i32,
            0,
            w_b3.value as i32,
            0,
            b[0].value as i32,
            0,
        );

        let mul_0 = x86_64::_mm256_mul_epu32(lhs_0, rhs_0);
        let mul_1 = x86_64::_mm256_mul_epu32(lhs_1, rhs_1);
        let mul_2 = x86_64::_mm256_mul_epu32(lhs_2, rhs_2);
        let mul_3 = x86_64::_mm256_mul_epu32(lhs_3, rhs_3);

        let dot_01 = x86_64::_mm256_add_epi64(mul_0, mul_1);
        let dot_23 = x86_64::_mm256_add_epi64(mul_2, mul_3);
        let dot = x86_64::_mm256_add_epi64(dot_01, dot_23);

        let mu_mm256 = x86_64::_mm512_castsi512_si256(FP::PACKED_MU);
        let p_mm256 = x86_64::_mm512_castsi512_si256(FP::PACKED_P);

        // We only care about the top 32 bits of dot as the bottom 32 bits will
        // be cancelled out.
        // Those bits currently lie in [0, 2P) so we reduce them to [0, P)
        let dot_sub = x86_64::_mm256_sub_epi32(dot, p_mm256);
        let dot_prime = x86_64::_mm256_min_epu32(dot, dot_sub);

        let q = x86_64::_mm256_mul_epu32(dot, mu_mm256);
        let q_p = x86_64::_mm256_mul_epu32(q, p_mm256);

        let t = x86_64::_mm256_sub_epi32(dot_prime, q_p);
        let corr = x86_64::_mm256_add_epi32(t, p_mm256);
        transmute(x86_64::_mm256_min_epu32(t, corr))
    };

    res[0] = dot_product[1];
    res[1] = dot_product[3];
    res[2] = dot_product[5];
    res[3] = dot_product[7];
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
    // TODO: It's plausible that this could be improved by folding the computation of packed_b into
    // the custom AVX512 implementation. Moreover, AVX512 is really a bit to large so we are wasting a lot
    // of space. A custom implementation which mixes AVX512 and AVX2 code might well be able to
    // improve one that is here.
    assert_eq!(WIDTH, 5);
    let zero = MontyField31::<FP>::ZERO;
    let w_b1 = FP::mul_w(b[1]);
    let w_b2 = FP::mul_w(b[2]);
    let w_b3 = FP::mul_w(b[3]);
    let w_b4 = FP::mul_w(b[4]);

    // Constant term = a0*b0 + w(a1*b4 + a2*b3 + a3*b2 + a4*b1)
    // Linear term = a0*b1 + a1*b0 + w(a2*b4 + a3*b3 + a4*b2)
    // Square term = a0*b2 + a1*b1 + a2*b0 + w(a3*b4 + a4*b3)
    // Cubic term = a0*b3 + a1*b2 + a2*b1 + a3*b0 + w*a4*b4
    // Quartic term = a0*b4 + a1*b3 + a2*b2 + a3*b1 + a4*b0

    // Each packed vector can do 8 multiplications at once. As we have
    // 25 multiplications to do we will need to use at least 3 packed vectors
    // but we might as well use 4 so we can make use of dot_product_2.
    // TODO: This can probably be improved by using a custom function.
    let lhs = [
        PackedMontyField31AVX512([
            a[0], a[1], a[0], a[1], a[2], a[3], a[0], a[1], a[2], a[3], a[4], a[4], a[4], a[4],
            a[4], zero,
        ]),
        PackedMontyField31AVX512([
            a[2], a[3], a[2], a[3], a[0], a[1], a[2], a[3], a[0], a[1], zero, zero, zero, zero,
            zero, zero,
        ]),
    ];
    let rhs = [
        PackedMontyField31AVX512([
            b[0], w_b4, b[1], b[0], b[0], w_b4, b[3], b[2], b[2], b[1], w_b1, w_b2, w_b3, w_b4,
            b[0], zero,
        ]),
        PackedMontyField31AVX512([
            w_b3, w_b2, w_b4, w_b3, b[2], b[1], b[1], b[0], b[4], b[3], zero, zero, zero, zero,
            zero, zero,
        ]),
    ];

    let dot = unsafe { PackedMontyField31AVX512::from_vector(dot_product_2(lhs, rhs)).0 };

    let sumand1 =
        PackedMontyField31AVX512::from_monty_array([dot[0], dot[2], dot[4], dot[6], dot[8]]);
    let sumand2 =
        PackedMontyField31AVX512::from_monty_array([dot[1], dot[3], dot[5], dot[7], dot[9]]);
    let sumand3 =
        PackedMontyField31AVX512::from_monty_array([dot[10], dot[11], dot[12], dot[13], dot[14]]);
    let sum = sumand1 + sumand2 + sumand3;

    res.copy_from_slice(&sum.0[..5]);
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
    // TODO: This could likely be optimised further with more effort.
    // in particular it would benefit from a custom AVX2 implementation.
    assert_eq!(WIDTH, 8);
    let packed_b = PackedMontyField31AVX512::from_monty_array(*b);
    let w_b = FP::mul_w(packed_b).0;

    // Constant coefficient = a0*b0 + w(a1*b7 + ... + a7*b1)
    // Linear coefficient = a0*b1 + a1*b0 + w(a2*b7 + ... + a7*b2)
    // Square coefficient = a0*b2 + .. + a2*b0 + w(a3*b7 + ... + a7*b3)
    // Cube coefficient = a0*b3 + .. + a3*b0 + w(a4*b7 + ... + a7*b4)
    // Quartic coefficient = a0*b4 + ... + a4*b0 + w(a5*b7 + ... + a7*b5)
    // Quintic coefficient = a0*b5 + ... + a5*b0 + w(a6*b7 + ... + a7*b6)
    // Sextic coefficient = a0*b6 + ... + a6*b0 + w*a7*b7
    // Final coefficient = a0*b7 + ... + a7*b0
    // The i'th 64 bit chunk of the _mm512 vector will compute the i'th coefficient.
    let lhs = [
        PackedMontyField31AVX512([
            a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[0], a[1], a[2], a[3], a[4], a[5],
            a[6], a[7],
        ]),
        PackedMontyField31AVX512([
            a[2], a[3], a[4], a[5], a[6], a[7], a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7],
            a[0], a[1],
        ]),
        PackedMontyField31AVX512([
            a[4], a[5], a[6], a[7], a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[0], a[1],
            a[2], a[3],
        ]),
        PackedMontyField31AVX512([
            a[6], a[7], a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[0], a[1], a[2], a[3],
            a[4], a[5],
        ]),
    ];
    let rhs = [
        PackedMontyField31AVX512([
            b[0], w_b[7], w_b[7], w_b[6], w_b[6], w_b[5], w_b[5], w_b[4], b[4], b[3], b[3], b[2],
            b[2], b[1], b[1], b[0],
        ]),
        PackedMontyField31AVX512([
            w_b[6], w_b[5], w_b[5], w_b[4], w_b[4], w_b[3], b[3], b[2], b[2], b[1], b[1], b[0],
            b[0], w_b[7], b[7], b[6],
        ]),
        PackedMontyField31AVX512([
            w_b[4], w_b[3], w_b[3], w_b[2], b[2], b[1], b[1], b[0], b[0], w_b[7], w_b[7], w_b[6],
            b[6], b[5], b[5], b[4],
        ]),
        PackedMontyField31AVX512([
            w_b[2], w_b[1], b[1], b[0], b[0], w_b[7], w_b[7], w_b[6], w_b[6], w_b[5], b[5], b[4],
            b[4], b[3], b[3], b[2],
        ]),
    ];

    // Now take the dot product of the two vectors.
    let dot = dot_product_4(lhs, rhs);

    // It remains to sum the 32 bit halves of the 64 bit chunks together.
    let total = unsafe {
        // We could do this via putting the relevant pieces into a __mm256 vectors but
        // the data fiddling seems to be more expensive than just doing the naive thing.
        let swizzled_dot = x86_64::_mm512_shuffle_epi32::<0b10110001>(dot);
        let sum = mm512_mod_add(dot, swizzled_dot, FP::PACKED_P);
        PackedMontyField31AVX512::<FP>::from_vector(sum)
    };

    res[0] = total.0[0];
    res[1] = total.0[2];
    res[2] = total.0[4];
    res[3] = total.0[6];
    res[4] = total.0[8];
    res[5] = total.0[10];
    res[6] = total.0[12];
    res[7] = total.0[14];
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
            // This could likely be sped up by a completely custom implementation of mul.
            // Surprisingly, the current version is only slightly faster than:
            // res.iter_mut().zip(a.iter()).for_each(|(r, a)| *r = *a * b);
            let out: [MontyField31<FP>; 8] = unsafe {
                let lhs: __m256i =
                    transmute([a[0].value, a[1].value, a[2].value, a[3].value, 0, 0, 0, 0]);
                let prod = mul_256::<FP>(lhs, b.value as i32);
                transmute(prod)
            };

            res.copy_from_slice(&out[..4]);
        }
        5 => {
            // This could likely be sped up by a completely custom implementation of mul.
            let out: [MontyField31<FP>; 8] = unsafe {
                let lhs: __m256i = transmute([
                    a[0].value, a[1].value, a[2].value, a[3].value, a[4].value, 0, 0, 0,
                ]);
                let prod = mul_256::<FP>(lhs, b.value as i32);
                transmute(prod)
            };

            res.copy_from_slice(&out[..5]);
        }
        8 => {
            // This could likely be sped up by a completely custom implementation of mul.
            let out: [MontyField31<FP>; 8] = unsafe {
                let lhs: __m256i = transmute([
                    a[0].value, a[1].value, a[2].value, a[3].value, a[4].value, a[5].value,
                    a[6].value, a[7].value,
                ]);
                let prod = mul_256::<FP>(lhs, b.value as i32);
                transmute(prod)
            };

            res.copy_from_slice(&out);
        }
        _ => panic!("Unsupported binomial extension degree: {}", WIDTH),
    }
}
