use alloc::vec::Vec;
use core::arch::x86_64::{self, __m256i};
use core::array;
use core::iter::{Product, Sum};
use core::mem::transmute;
use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

use p3_field::{
    Algebra, Field, InjectiveMonomial, PackedField, PackedFieldPow2, PackedValue,
    PermutationMonomial, PrimeCharacteristicRing,
};
use p3_util::convert_vec;
use rand::Rng;
use rand::distr::{Distribution, StandardUniform};

use crate::{FieldParameters, MontyField31, PackedMontyParameters, RelativelyPrimePower};

const WIDTH: usize = 8;

pub trait MontyParametersAVX2 {
    const PACKED_P: __m256i;
    const PACKED_P_HIGH: __m256i;
    const PACKED_MU: __m256i;
}

/// Vectorized AVX2 implementation of `MontyField31<FP>` arithmetic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)] // This is needed to make `transmute`s safe.
pub struct PackedMontyField31AVX2<PMP: PackedMontyParameters>(pub [MontyField31<PMP>; WIDTH]);

impl<PMP: PackedMontyParameters> PackedMontyField31AVX2<PMP> {
    #[inline]
    #[must_use]
    /// Get an arch-specific vector representing the packed values.
    pub(crate) fn to_vector(self) -> __m256i {
        unsafe {
            // Safety: `MontyField31<FP>` is `repr(transparent)` so it can be transmuted to `u32`. It
            // follows that `[MontyField31<FP>; WIDTH]` can be transmuted to `[u32; WIDTH]`, which can be
            // transmuted to `__m256i`, since arrays are guaranteed to be contiguous in memory.
            // Finally `PackedMontyField31AVX2<FP>` is `repr(transparent)` so it can be transmuted to
            // `[MontyField31<FP>; WIDTH]`.
            transmute(self)
        }
    }

    #[inline]
    #[must_use]
    /// Make a packed field vector from an arch-specific vector.
    ///
    /// SAFETY: The caller must ensure that each element of `vector` represents a valid `MontyField31<FP>`.
    /// In particular, each element of vector must be in `0..P` (canonical form).
    pub(crate) unsafe fn from_vector(vector: __m256i) -> Self {
        unsafe {
            // Safety: It is up to the user to ensure that elements of `vector` represent valid
            // `MontyField31<FP>` values. We must only reason about memory representations. `__m256i` can be
            // transmuted to `[u32; WIDTH]` (since arrays elements are contiguous in memory), which can
            // be transmuted to `[MontyField31<FP>; WIDTH]` (since `MontyField31<FP>` is `repr(transparent)`), which in
            // turn can be transmuted to `PackedMontyField31AVX2<FP>` (since `PackedMontyField31AVX2<FP>` is also
            // `repr(transparent)`).
            transmute(vector)
        }
    }

    /// Copy `value` to all positions in a packed vector. This is the same as
    /// `From<MontyField31<FP>>::from`, but `const`.
    #[inline]
    #[must_use]
    const fn broadcast(value: MontyField31<PMP>) -> Self {
        Self([value; WIDTH])
    }
}

impl<PMP: PackedMontyParameters> Add for PackedMontyField31AVX2<PMP> {
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

impl<PMP: PackedMontyParameters> Mul for PackedMontyField31AVX2<PMP> {
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

impl<PMP: PackedMontyParameters> Neg for PackedMontyField31AVX2<PMP> {
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

impl<PMP: PackedMontyParameters> Sub for PackedMontyField31AVX2<PMP> {
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

/// Add two vectors of Monty31 field elements in canonical form.
/// If the inputs are not in canonical form, the result is undefined.
#[inline]
#[must_use]
pub(crate) fn add<MPAVX2: MontyParametersAVX2>(lhs: __m256i, rhs: __m256i) -> __m256i {
    // We want this to compile to:
    //      vpaddd   t, lhs, rhs
    //      vpsubd   u, t, P
    //      vpminud  res, t, u
    // throughput: 1 cyc/vec (8 els/cyc)
    // latency: 3 cyc

    //   Let t := lhs + rhs. We want to return t mod P. Recall that lhs and rhs are in
    // 0, ..., P - 1, so t is in 0, ..., 2 P - 2 (< 2^32). It suffices to return t if t < P and
    // t - P otherwise.
    //   Let u := (t - P) mod 2^32 and r := unsigned_min(t, u).
    //   If t is in 0, ..., P - 1, then u is in (P - 1 <) 2^32 - P, ..., 2^32 - 1 and r = t.
    // Otherwise, t is in P, ..., 2 P - 2, u is in 0, ..., P - 2 (< P) and r = u. Hence, r is t if
    // t < P and t - P otherwise, as desired.

    unsafe {
        // Safety: If this code got compiled then AVX2 intrinsics are available.
        let t = x86_64::_mm256_add_epi32(lhs, rhs);
        let u = x86_64::_mm256_sub_epi32(t, MPAVX2::PACKED_P);
        x86_64::_mm256_min_epu32(t, u)
    }
}

// MONTGOMERY MULTIPLICATION
//   This implementation is based on [1] but with minor changes. The reduction is as follows:
//
// Constants: P < 2^31, prime
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
// It remains to show that R is in the correct range. It suffices to show that -P < D < P. We know
// that 0 <= C < P B and 0 <= Q P < P B. Then -P B < C - QP < P B and -P < D < P, as desired.
//
// [1] Modern Computer Arithmetic, Richard Brent and Paul Zimmermann, Cambridge University Press,
//     2010, algorithm 2.7.

// We provide 2 variants of Montgomery reduction depending on if the inputs are unsigned or signed.
// The unsigned variant follows steps 1 and 2 in the above protocol to produce D in (-P, ..., P).
// For the signed variant we assume -PB/2 < C < PB/2 and let Q := μ C mod B be the unique
// representative in [-B/2, ..., B/2 - 1]. The division in step 2 is clearly still exact and
// |C - Q P| <= |C| + |Q||P| < PB so D still lies in (-P, ..., P).

/// Perform a partial Montgomery reduction on each 64 bit element.
/// Input must lie in {0, ..., 2^32P}.
/// The output will lie in {-P, ..., P} and be stored in the upper 32 bits.
#[inline]
#[must_use]
fn partial_monty_red_unsigned_to_signed<MPAVX2: MontyParametersAVX2>(input: __m256i) -> __m256i {
    unsafe {
        let q = x86_64::_mm256_mul_epu32(input, MPAVX2::PACKED_MU);
        let q_p = x86_64::_mm256_mul_epu32(q, MPAVX2::PACKED_P);

        // By construction, the bottom 32 bits of input and q_p are equal.
        // Thus _mm256_sub_epi32 and _mm256_sub_epi64 should act identically.
        // However for some reason, the compiler gets confused if we use _mm256_sub_epi64
        // and outputs a load of nonsense, see: https://godbolt.org/z/3W8M7Tv84.
        x86_64::_mm256_sub_epi32(input, q_p)
    }
}

/// Perform a partial Montgomery reduction on each 64 bit element.
/// Input must lie in {-2^{31}P, ..., 2^31P}.
/// The output will lie in {-P, ..., P} and be stored in the upper 32 bits.
#[inline]
#[must_use]
fn partial_monty_red_signed_to_signed<MPAVX2: MontyParametersAVX2>(input: __m256i) -> __m256i {
    unsafe {
        let q = x86_64::_mm256_mul_epi32(input, MPAVX2::PACKED_MU);
        let q_p = x86_64::_mm256_mul_epi32(q, MPAVX2::PACKED_P);

        // Unlike the previous case the compiler output is essentially identical
        // between _mm256_sub_epi32 and _mm256_sub_epi64. We use _mm256_sub_epi32
        // again just for consistency.
        x86_64::_mm256_sub_epi32(input, q_p)
    }
}

/// Multiply the MontyField31 field elements in the even index entries.
/// lhs[2i], rhs[2i] must be unsigned 32-bit integers such that
/// lhs[2i] * rhs[2i] lies in {0, ..., 2^32P}.
/// The output will lie in {-P, ..., P} and be stored in output[2i + 1].
#[inline]
#[must_use]
fn monty_mul<MPAVX2: MontyParametersAVX2>(lhs: __m256i, rhs: __m256i) -> __m256i {
    unsafe {
        let prod = x86_64::_mm256_mul_epu32(lhs, rhs);
        partial_monty_red_unsigned_to_signed::<MPAVX2>(prod)
    }
}

/// Multiply the MontyField31 field elements in the even index entries.
/// lhs[2i], rhs[2i] must be signed 32-bit integers such that
/// lhs[2i] * rhs[2i] lies in {-2^31P, ..., 2^31P}.
/// The output will lie in {-P, ..., P} stored in output[2i + 1].
#[inline]
#[must_use]
fn monty_mul_signed<MPAVX2: MontyParametersAVX2>(lhs: __m256i, rhs: __m256i) -> __m256i {
    unsafe {
        let prod = x86_64::_mm256_mul_epi32(lhs, rhs);
        partial_monty_red_signed_to_signed::<MPAVX2>(prod)
    }
}

#[inline]
#[must_use]
fn movehdup_epi32(x: __m256i) -> __m256i {
    // This instruction is only available in the floating-point flavor; this distinction is only for
    // historical reasons and no longer matters. We cast to floats, duplicate, and cast back.
    unsafe {
        x86_64::_mm256_castps_si256(x86_64::_mm256_movehdup_ps(x86_64::_mm256_castsi256_ps(x)))
    }
}

/// Multiply vectors of MontyField31 field elements in canonical form.
/// If the inputs are not in canonical form, the result is undefined.
#[inline]
#[must_use]
fn mul<MPAVX2: MontyParametersAVX2>(lhs: __m256i, rhs: __m256i) -> __m256i {
    // We want this to compile to:
    //      vmovshdup  lhs_odd, lhs
    //      vmovshdup  rhs_odd, rhs
    //      vpmuludq   prod_evn, lhs, rhs
    //      vpmuludq   prod_odd, lhs_odd, rhs_odd
    //      vpmuludq   q_evn, prod_evn, MU
    //      vpmuludq   q_odd, prod_odd, MU
    //      vpmuludq   q_P_evn, q_evn, P
    //      vpmuludq   q_P_odd, q_odd, P
    //      vpsubq     d_evn, prod_evn, q_P_evn
    //      vpsubq     d_odd, prod_odd, q_P_odd
    //      vmovshdup  d_evn_hi, d_evn
    //      vpblendd   t, d_evn_hi, d_odd, aah
    //      vpaddd     u, t, P
    //      vpminud    res, t, u
    // throughput: 4.67 cyc/vec (1.71 els/cyc)
    // latency: 21 cyc
    unsafe {
        let lhs_evn = lhs;
        let rhs_evn = rhs;
        let lhs_odd = movehdup_epi32(lhs);
        let rhs_odd = movehdup_epi32(rhs);

        let d_evn = monty_mul::<MPAVX2>(lhs_evn, rhs_evn);
        let d_odd = monty_mul::<MPAVX2>(lhs_odd, rhs_odd);

        let d_evn_hi = movehdup_epi32(d_evn);
        let t = x86_64::_mm256_blend_epi32::<0b10101010>(d_evn_hi, d_odd);

        let u = x86_64::_mm256_add_epi32(t, MPAVX2::PACKED_P);
        x86_64::_mm256_min_epu32(t, u)
    }
}

/// Lets us combine some code for MontyField31<FP> and PackedMontyField31AVX2<FP> elements.
trait InToM256Vector<PMP: PackedMontyParameters>: Copy + Into<PackedMontyField31AVX2<PMP>> {
    /// Convert the input to a __m256i vector.
    fn get_vector(&self) -> __m256i;

    /// Convert the input to a __m256i vector and shift so that all elements in odd positions
    /// lie in even positions.
    ///
    /// The values lying in the even positions are undefined.
    #[inline(always)]
    fn get_odd_in_even_pos_vector(&self) -> __m256i {
        let vec = self.get_vector();
        movehdup_epi32(vec)
    }
}

impl<PMP: PackedMontyParameters> InToM256Vector<PMP> for PackedMontyField31AVX2<PMP> {
    #[inline(always)]
    fn get_vector(&self) -> __m256i {
        self.to_vector()
    }
}

impl<PMP: PackedMontyParameters> InToM256Vector<PMP> for MontyField31<PMP> {
    #[inline(always)]
    fn get_vector(&self) -> __m256i {
        unsafe { x86_64::_mm256_set1_epi32(self.value as i32) }
    }

    #[inline(always)]
    fn get_odd_in_even_pos_vector(&self) -> __m256i {
        unsafe { x86_64::_mm256_set1_epi32(self.value as i32) }
    }
}

/// Compute the elementary function `l0*r0 + l1*r1` given four inputs
/// in canonical form.
///
/// If the inputs are not in canonical form, the result is undefined.
#[inline]
#[must_use]
fn dot_product_2<PMP: PackedMontyParameters, LHS: InToM256Vector<PMP>, RHS: InToM256Vector<PMP>>(
    lhs: [LHS; 2],
    rhs: [RHS; 2],
) -> __m256i {
    // The following analysis treats all input arrays as being arrays of PackedMontyField31AVX2<FP>.
    // The analysis is identical if one of the arrays contains MontyField31<FP>, we just get to avoid
    // a few vmovshdup so everything is slightly cheaper.
    //
    // The naive method involves two multiplications and a summation taking
    // (14)*2 + 3 = 31 instructions with:
    // throughput: 10.33 cyc/vec (0.77 els/cyc)
    // latency: 24 cyc
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
    //      vpaddq     prod_evn, prod_evn0, prod_evn1
    //      vpaddq     prod_odd, prod_odd0, prod_odd1
    //      vpmuludq   q_evn, prod_evn, MU
    //      vpmuludq   q_odd, prod_odd, MU
    //      vpmuludq   q_P_evn, q_evn, P
    //      vpmuludq   q_P_odd, q_odd, P
    //      vpsubq     d_evn, prod_evn, q_P_evn
    //      vpsubq     d_odd, prod_odd, q_P_odd
    //      vmovshdup  d_evn_hi, d_evn
    //      vpblendd   t, d_evn_hi, d_odd, aah
    //      vpaddd     u, t, P
    //      vpminud    res, t, u
    // throughput: 6.67 cyc/vec (1.20 els/cyc)
    // latency: 22 cyc
    unsafe {
        let lhs_evn0 = lhs[0].get_vector();
        let lhs_odd0 = lhs[0].get_odd_in_even_pos_vector();
        let lhs_evn1 = lhs[1].get_vector();
        let lhs_odd1 = lhs[1].get_odd_in_even_pos_vector();

        let rhs_evn0 = rhs[0].get_vector();
        let rhs_odd0 = rhs[0].get_odd_in_even_pos_vector();
        let rhs_evn1 = rhs[1].get_vector();
        let rhs_odd1 = rhs[1].get_odd_in_even_pos_vector();

        let mul_evn0 = x86_64::_mm256_mul_epu32(lhs_evn0, rhs_evn0);
        let mul_evn1 = x86_64::_mm256_mul_epu32(lhs_evn1, rhs_evn1);
        let mul_odd0 = x86_64::_mm256_mul_epu32(lhs_odd0, rhs_odd0);
        let mul_odd1 = x86_64::_mm256_mul_epu32(lhs_odd1, rhs_odd1);

        let dot_evn = x86_64::_mm256_add_epi64(mul_evn0, mul_evn1);
        let dot_odd = x86_64::_mm256_add_epi64(mul_odd0, mul_odd1);

        let red_evn = partial_monty_red_unsigned_to_signed::<PMP>(dot_evn);
        let red_odd = partial_monty_red_unsigned_to_signed::<PMP>(dot_odd);

        let red_evn_hi = movehdup_epi32(red_evn);
        let t = x86_64::_mm256_blend_epi32::<0b10101010>(red_evn_hi, red_odd);

        let u = x86_64::_mm256_add_epi32(t, PMP::PACKED_P);
        x86_64::_mm256_min_epu32(t, u)
    }
}

/// Compute the elementary function `l0*r0 + l1*r1 + l2*r2 + l3*r3` given eight inputs
/// in canonical form.
///
/// If the inputs are not in canonical form, the result is undefined.
#[inline]
#[must_use]
fn dot_product_4<PMP: PackedMontyParameters, LHS: InToM256Vector<PMP>, RHS: InToM256Vector<PMP>>(
    lhs: [LHS; 4],
    rhs: [RHS; 4],
) -> __m256i {
    // The following analysis treats all input arrays as being arrays of PackedMontyField31AVX2<FP>.
    // The analysis is identical if one of the arrays contains MontyField31<FP>, we just get to avoid
    // a few vmovshdup so everything is slightly cheaper.
    //
    // The naive method involves calling dot_product_2 twice and a summation taking
    // (20)*2 + 3 = 43 instructions with:
    // throughput: 14.33 cyc/vec (0.56 els/cyc)
    // latency: 25 cyc
    //
    // We improve the throughput by combining the monty reductions together. This is a little more
    // complicated than in the dot_product_2 case as we will need to slightly adjust the reduction algorithm.
    //
    // As all inputs are `< P < 2^{31}`, the sum satisfies: `l0*r0 + l1*r1 + l2*r2 + l3*r3 < 4P^2 < 2*2^{32}P`.
    // Now compute Q as usual:
    //
    // Q := μ C mod B
    //
    // We can't proceed as normal however as C - Q P now lies in the range 2*2^{32}P > C - QP > -2^{32}P which
    // doesn't fit into an i64. Moreover, AVX2 does not support u64 comparisons, only i64. We can bootstrap
    // i64 comparisons to u64 ones but it takes some extra instructions so instead of testing C and QP we test
    // C >= 2^{32}P where we can now save 2^{32}P as a constant in the appropriate form. This lets us define
    //
    // C' = if C < 2^{32}P: {C} else {C - 2^{32}P}
    //
    // And we can proceed with the standard identical montgomery reduction on C'. Note that
    // C = C' mod B so we can compute Q while the C check is going on.
    //
    // (2^{32}P, 2^{63})
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
    //      vpaddq     prod_evn01, prod_evn0, prod_evn1
    //      vpaddq     prod_odd01, prod_odd0, prod_odd1
    //      vpaddq     prod_evn23, prod_evn2, prod_evn3
    //      vpaddq     prod_odd23, prod_odd2, prod_odd3
    //      vpaddq     prod_evn, prod_evn01, prod_evn23
    //      vpaddq     prod_odd, prod_odd01, prod_odd23
    //      vpmuludq   q_evn, prod_evn, MU
    //      vpmuludq   q_odd, prod_odd, MU
    //      vpmuludq   q_P_evn, q_evn, P
    //      vpmuludq   q_P_odd, q_odd, P
    //      vpsubq     prod_evn_sub, prod_evn, 2^32P
    //      vpminud    prod_evn_prime, prod_evn, prod_evn_sub
    //      vpsubq     prod_odd_sub, prod_odd, 2^32P
    //      vpminud    prod_odd_prime, prod_odd, prod_odd_sub
    //      vpsubq     d_evn, prod_evn_prime, q_P_evn
    //      vpsubq     d_odd, prod_odd_prime, q_P_odd
    //      vmovshdup  d_evn_hi, d_evn
    //      vpblendd   t, d_evn_hi, d_odd, aah
    //      vpaddd     u, t, P
    //      vpminud    res, t, u
    // throughput: 12 cyc/vec (0.67 els/cyc)
    // latency: 23 cyc
    unsafe {
        let lhs_evn0 = lhs[0].get_vector();
        let lhs_odd0 = lhs[0].get_odd_in_even_pos_vector();
        let lhs_evn1 = lhs[1].get_vector();
        let lhs_odd1 = lhs[1].get_odd_in_even_pos_vector();
        let lhs_evn2 = lhs[2].get_vector();
        let lhs_odd2 = lhs[2].get_odd_in_even_pos_vector();
        let lhs_evn3 = lhs[3].get_vector();
        let lhs_odd3 = lhs[3].get_odd_in_even_pos_vector();

        let rhs_evn0 = rhs[0].get_vector();
        let rhs_odd0 = rhs[0].get_odd_in_even_pos_vector();
        let rhs_evn1 = rhs[1].get_vector();
        let rhs_odd1 = rhs[1].get_odd_in_even_pos_vector();
        let rhs_evn2 = rhs[2].get_vector();
        let rhs_odd2 = rhs[2].get_odd_in_even_pos_vector();
        let rhs_evn3 = rhs[3].get_vector();
        let rhs_odd3 = rhs[3].get_odd_in_even_pos_vector();

        let mul_evn0 = x86_64::_mm256_mul_epu32(lhs_evn0, rhs_evn0);
        let mul_evn1 = x86_64::_mm256_mul_epu32(lhs_evn1, rhs_evn1);
        let mul_evn2 = x86_64::_mm256_mul_epu32(lhs_evn2, rhs_evn2);
        let mul_evn3 = x86_64::_mm256_mul_epu32(lhs_evn3, rhs_evn3);
        let mul_odd0 = x86_64::_mm256_mul_epu32(lhs_odd0, rhs_odd0);
        let mul_odd1 = x86_64::_mm256_mul_epu32(lhs_odd1, rhs_odd1);
        let mul_odd2 = x86_64::_mm256_mul_epu32(lhs_odd2, rhs_odd2);
        let mul_odd3 = x86_64::_mm256_mul_epu32(lhs_odd3, rhs_odd3);

        let dot_evn01 = x86_64::_mm256_add_epi64(mul_evn0, mul_evn1);
        let dot_odd01 = x86_64::_mm256_add_epi64(mul_odd0, mul_odd1);
        let dot_evn23 = x86_64::_mm256_add_epi64(mul_evn2, mul_evn3);
        let dot_odd23 = x86_64::_mm256_add_epi64(mul_odd2, mul_odd3);

        let dot_evn = x86_64::_mm256_add_epi64(dot_evn01, dot_evn23);
        let dot_odd = x86_64::_mm256_add_epi64(dot_odd01, dot_odd23);

        // In dot_evn_sub, red_evn, dot_odd_sub, red_odd:
        // _mm256_sub_epi32 and _mm256_sub_epi64 should act identically.
        // However for some reason, the compiler gets confused if we use _mm256_sub_epi64.
        // and outputs a load of nonsense.
        let q_evn = x86_64::_mm256_mul_epu32(dot_evn, PMP::PACKED_MU);
        let q_p_evn = x86_64::_mm256_mul_epu32(q_evn, PMP::PACKED_P);
        let dot_evn_sub = x86_64::_mm256_sub_epi32(dot_evn, PMP::PACKED_P_HIGH);
        let dot_evn_prime = x86_64::_mm256_min_epu32(dot_evn, dot_evn_sub);
        let red_evn = x86_64::_mm256_sub_epi32(dot_evn_prime, q_p_evn);

        let q_odd = x86_64::_mm256_mul_epu32(dot_odd, PMP::PACKED_MU);
        let q_p_odd = x86_64::_mm256_mul_epu32(q_odd, PMP::PACKED_P);
        let dot_odd_sub = x86_64::_mm256_sub_epi32(dot_odd, PMP::PACKED_P_HIGH);
        let dot_odd_prime = x86_64::_mm256_min_epu32(dot_odd, dot_odd_sub);
        let red_odd = x86_64::_mm256_sub_epi32(dot_odd_prime, q_p_odd);

        let red_evn_hi = movehdup_epi32(red_evn);
        let t = x86_64::_mm256_blend_epi32::<0b10101010>(red_evn_hi, red_odd);

        let u = x86_64::_mm256_add_epi32(t, PMP::PACKED_P);
        x86_64::_mm256_min_epu32(t, u)
    }
}

/// For larger dot products we want to repeat calling dot product 4 as
/// many times as possible.
#[inline(always)]
fn general_dot_product<FP: FieldParameters, LHS: InToM256Vector<FP>, RHS: InToM256Vector<FP>>(
    lhs: &[LHS],
    rhs: &[RHS],
) -> PackedMontyField31AVX2<FP> {
    let n = lhs.len();
    assert_eq!(lhs.len(), rhs.len());
    match n {
        1 => (lhs[0]).into() * (rhs[0]).into(),
        2 => {
            let res = dot_product_2([lhs[0], lhs[1]], [rhs[0], rhs[1]]);
            unsafe {
                // Safety: `dot_product_2` returns values in canonical form when given values in canonical form.
                PackedMontyField31AVX2::<FP>::from_vector(res)
            }
        }
        3 => {
            let lhs2 = lhs[2];
            let rhs2 = rhs[2];
            let res = dot_product_2([lhs[0], lhs[1]], [rhs[0], rhs[1]]);
            unsafe {
                // Safety: `dot_product_2` returns values in canonical form when given values in canonical form.
                PackedMontyField31AVX2::<FP>::from_vector(res) + (lhs2.into() * rhs2.into())
            }
        }
        4 => {
            let res = dot_product_4(
                [lhs[0], lhs[1], lhs[2], lhs[3]],
                [rhs[0], rhs[1], rhs[2], rhs[3]],
            );
            unsafe {
                // Safety: `dot_product_4` returns values in canonical form when given values in canonical form.
                PackedMontyField31AVX2::<FP>::from_vector(res)
            }
        }
        64 => {
            let sum_4s: [PackedMontyField31AVX2<FP>; 16] = array::from_fn(|i| {
                let res = dot_product_4(
                    [lhs[4 * i], lhs[4 * i + 1], lhs[4 * i + 2], lhs[4 * i + 3]],
                    [rhs[4 * i], rhs[4 * i + 1], rhs[4 * i + 2], rhs[4 * i + 3]],
                );
                unsafe { PackedMontyField31AVX2::<FP>::from_vector(res) }
            });
            PackedMontyField31AVX2::<FP>::tree_sum::<16>(&sum_4s)
        }
        _ => {
            let mut acc = PackedMontyField31AVX2::<FP>::ZERO;
            for i in (0..n).step_by(4) {
                let res = dot_product_4(
                    [lhs[i], lhs[i + 1], lhs[i + 2], lhs[i + 3]],
                    [rhs[i], rhs[i + 1], rhs[i + 2], rhs[i + 3]],
                );
                unsafe {
                    // Safety: `dot_product_4` returns values in canonical form when given values in canonical form.
                    acc += PackedMontyField31AVX2::<FP>::from_vector(res)
                }
            }
            let remainder = n % 4;
            for i in (n - remainder)..n {
                acc += (lhs[i]).into() * (rhs[i]).into()
            }
            acc
        }
    }
}

/// Square the MontyField31 field elements in the even index entries.
/// Inputs must be signed 32-bit integers.
/// Outputs will be a signed integer in (-P, ..., P) copied into both the even and odd indices.
#[inline]
#[must_use]
fn shifted_square<MPAVX2: MontyParametersAVX2>(input: __m256i) -> __m256i {
    // Note that we do not need a restriction on the size of input[i]^2 as
    // 2^30 < P and |i32| <= 2^31 and so => input[i]^2 <= 2^62 < 2^32P.
    unsafe {
        let square = x86_64::_mm256_mul_epi32(input, input);
        let square_red = partial_monty_red_unsigned_to_signed::<MPAVX2>(square);
        movehdup_epi32(square_red)
    }
}

/// Cube the MontyField31 field elements in the even index entries.
/// Inputs must be signed 32-bit integers in [-P, ..., P].
/// Outputs will be a signed integer in (-P, ..., P) stored in the odd indices.
#[inline]
#[must_use]
pub(crate) fn packed_exp_3<MPAVX2: MontyParametersAVX2>(input: __m256i) -> __m256i {
    let square = shifted_square::<MPAVX2>(input);
    monty_mul_signed::<MPAVX2>(square, input)
}

/// Take the fifth power of the MontyField31 field elements in the even index entries.
/// Inputs must be signed 32-bit integers in [-P, ..., P].
/// Outputs will be a signed integer in (-P, ..., P) stored in the odd indices.
#[inline]
#[must_use]
pub(crate) fn packed_exp_5<MPAVX2: MontyParametersAVX2>(input: __m256i) -> __m256i {
    let square = shifted_square::<MPAVX2>(input);
    let quad = shifted_square::<MPAVX2>(square);
    monty_mul_signed::<MPAVX2>(quad, input)
}

/// Take the seventh power of the MontyField31 field elements in the even index entries.
/// Inputs must lie in [-P, ..., P].
/// Outputs will also lie in (-P, ..., P) stored in the odd indices.
#[inline]
#[must_use]
pub(crate) fn packed_exp_7<MPAVX2: MontyParametersAVX2>(input: __m256i) -> __m256i {
    let square = shifted_square::<MPAVX2>(input);
    let cube = monty_mul_signed::<MPAVX2>(square, input);
    let cube_shifted = movehdup_epi32(cube);
    let quad = shifted_square::<MPAVX2>(square);

    monty_mul_signed::<MPAVX2>(quad, cube_shifted)
}

/// Apply func to the even and odd indices of the input vector.
/// func should only depend in the 32 bit entries in the even indices.
/// The output of func must lie in (-P, ..., P) and be stored in the odd indices.
/// The even indices of the output of func will not be read.
/// The input should conform to the requirements of `func`.
#[inline]
#[must_use]
pub(crate) unsafe fn apply_func_to_even_odd<MPAVX2: MontyParametersAVX2>(
    input: __m256i,
    func: fn(__m256i) -> __m256i,
) -> __m256i {
    unsafe {
        let input_evn = input;
        let input_odd = movehdup_epi32(input);

        let d_evn = func(input_evn);
        let d_odd = func(input_odd);

        let d_evn_hi = movehdup_epi32(d_evn);
        let t = x86_64::_mm256_blend_epi32::<0b10101010>(d_evn_hi, d_odd);

        let u = x86_64::_mm256_add_epi32(t, MPAVX2::PACKED_P);
        x86_64::_mm256_min_epu32(t, u)
    }
}

/// Negate a vector of MontyField31 field elements in canonical form.
/// If the inputs are not in canonical form, the result is undefined.
#[inline]
#[must_use]
fn neg<MPAVX2: MontyParametersAVX2>(val: __m256i) -> __m256i {
    // We want this to compile to:
    //      vpsubd   t, P, val
    //      vpsignd  res, t, val
    // throughput: .67 cyc/vec (12 els/cyc)
    // latency: 2 cyc

    //   The vpsignd instruction is poorly named, because it doesn't _return_ or _copy_ the sign of
    // anything, but _multiplies_ x by the sign of y (treating both as signed integers). In other
    // words,
    //                       { x            if y >s 0,
    //      vpsignd(x, y) := { 0            if y = 0,
    //                       { -x mod 2^32  if y <s 0.
    //   We define t := P - val and note that t = -val (mod P). When val is in {1, ..., P - 1}, t is
    // similarly in {1, ..., P - 1}, so it's in canonical form. Otherwise, val = 0 and t = P.
    //   This is where we define res := vpsignd(t, val). The sign bit of val is never set so either
    // val = 0 or val >s 0. If val = 0, then res = vpsignd(t, 0) = 0, as desired. Otherwise,
    // res = vpsignd(t, val) = t passes t through.
    unsafe {
        // Safety: If this code got compiled then AVX2 intrinsics are available.
        let t = x86_64::_mm256_sub_epi32(MPAVX2::PACKED_P, val);
        x86_64::_mm256_sign_epi32(t, val)
    }
}

/// Subtract vectors of MontyField31 field elements in canonical form.
/// If the inputs are not in canonical form, the result is undefined.
#[inline]
#[must_use]
pub(crate) fn sub<MPAVX2: MontyParametersAVX2>(lhs: __m256i, rhs: __m256i) -> __m256i {
    // We want this to compile to:
    //      vpsubd   t, lhs, rhs
    //      vpaddd   u, t, P
    //      vpminud  res, t, u
    // throughput: 1 cyc/vec (8 els/cyc)
    // latency: 3 cyc

    //   Let t := lhs - rhs. We want to return t mod P. Recall that lhs and rhs are in
    // 0, ..., P - 1, so t is in (-2^31 <) -P + 1, ..., P - 1 (< 2^31). It suffices to return t if
    // t >= 0 and t + P otherwise.
    //   Let u := (t + P) mod 2^32 and r := unsigned_min(t, u).
    //   If t is in 0, ..., P - 1, then u is in P, ..., 2 P - 1 and r = t.
    // Otherwise, t is in -P + 1, ..., -1; u is in 1, ..., P - 1 (< P) and r = u. Hence, r is t if
    // t < P and t - P otherwise, as desired.
    unsafe {
        // Safety: If this code got compiled then AVX2 intrinsics are available.
        let t = x86_64::_mm256_sub_epi32(lhs, rhs);
        let u = x86_64::_mm256_add_epi32(t, MPAVX2::PACKED_P);
        x86_64::_mm256_min_epu32(t, u)
    }
}

impl<PMP: PackedMontyParameters> From<MontyField31<PMP>> for PackedMontyField31AVX2<PMP> {
    #[inline]
    fn from(value: MontyField31<PMP>) -> Self {
        Self::broadcast(value)
    }
}

impl<PMP: PackedMontyParameters> Default for PackedMontyField31AVX2<PMP> {
    #[inline]
    fn default() -> Self {
        MontyField31::<PMP>::default().into()
    }
}

impl<PMP: PackedMontyParameters> AddAssign for PackedMontyField31AVX2<PMP> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<PMP: PackedMontyParameters> MulAssign for PackedMontyField31AVX2<PMP> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<PMP: PackedMontyParameters> SubAssign for PackedMontyField31AVX2<PMP> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<FP: FieldParameters> Sum for PackedMontyField31AVX2<FP> {
    #[inline]
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.reduce(|lhs, rhs| lhs + rhs).unwrap_or(Self::ZERO)
    }
}

impl<FP: FieldParameters> Product for PackedMontyField31AVX2<FP> {
    #[inline]
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.reduce(|lhs, rhs| lhs * rhs).unwrap_or(Self::ONE)
    }
}

impl<FP: FieldParameters> PrimeCharacteristicRing for PackedMontyField31AVX2<FP> {
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
        unsafe {
            // Safety: `apply_func_to_even_odd` returns values in canonical form when given values in canonical form.
            let res = apply_func_to_even_odd::<FP>(val, packed_exp_3::<FP>);
            Self::from_vector(res)
        }
    }

    #[must_use]
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
        general_dot_product(u, v)
    }

    #[inline(always)]
    fn zero_vec(len: usize) -> Vec<Self> {
        // SAFETY: this is a repr(transparent) wrapper around an array.
        unsafe { convert_vec(MontyField31::<FP>::zero_vec(len * WIDTH)) }
    }
}

impl<FP: FieldParameters> Algebra<MontyField31<FP>> for PackedMontyField31AVX2<FP> {}

impl<FP: FieldParameters + RelativelyPrimePower<D>, const D: u64> InjectiveMonomial<D>
    for PackedMontyField31AVX2<FP>
{
}

impl<FP: FieldParameters + RelativelyPrimePower<D>, const D: u64> PermutationMonomial<D>
    for PackedMontyField31AVX2<FP>
{
    fn injective_exp_root_n(&self) -> Self {
        FP::exp_root_d(*self)
    }
}

impl<PMP: PackedMontyParameters> Add<MontyField31<PMP>> for PackedMontyField31AVX2<PMP> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: MontyField31<PMP>) -> Self {
        self + Self::from(rhs)
    }
}

impl<PMP: PackedMontyParameters> Mul<MontyField31<PMP>> for PackedMontyField31AVX2<PMP> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: MontyField31<PMP>) -> Self {
        self * Self::from(rhs)
    }
}

impl<PMP: PackedMontyParameters> Sub<MontyField31<PMP>> for PackedMontyField31AVX2<PMP> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: MontyField31<PMP>) -> Self {
        self - Self::from(rhs)
    }
}

impl<PMP: PackedMontyParameters> AddAssign<MontyField31<PMP>> for PackedMontyField31AVX2<PMP> {
    #[inline]
    fn add_assign(&mut self, rhs: MontyField31<PMP>) {
        *self += Self::from(rhs)
    }
}

impl<PMP: PackedMontyParameters> MulAssign<MontyField31<PMP>> for PackedMontyField31AVX2<PMP> {
    #[inline]
    fn mul_assign(&mut self, rhs: MontyField31<PMP>) {
        *self *= Self::from(rhs)
    }
}

impl<PMP: PackedMontyParameters> SubAssign<MontyField31<PMP>> for PackedMontyField31AVX2<PMP> {
    #[inline]
    fn sub_assign(&mut self, rhs: MontyField31<PMP>) {
        *self -= Self::from(rhs)
    }
}

impl<FP: FieldParameters> Sum<MontyField31<FP>> for PackedMontyField31AVX2<FP> {
    #[inline]
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = MontyField31<FP>>,
    {
        iter.sum::<MontyField31<FP>>().into()
    }
}

impl<FP: FieldParameters> Product<MontyField31<FP>> for PackedMontyField31AVX2<FP> {
    #[inline]
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = MontyField31<FP>>,
    {
        iter.product::<MontyField31<FP>>().into()
    }
}

impl<FP: FieldParameters> Div<MontyField31<FP>> for PackedMontyField31AVX2<FP> {
    type Output = Self;
    #[allow(clippy::suspicious_arithmetic_impl)]
    #[inline]
    fn div(self, rhs: MontyField31<FP>) -> Self {
        self * rhs.inverse()
    }
}

impl<PMP: PackedMontyParameters> Add<PackedMontyField31AVX2<PMP>> for MontyField31<PMP> {
    type Output = PackedMontyField31AVX2<PMP>;
    #[inline]
    fn add(self, rhs: PackedMontyField31AVX2<PMP>) -> PackedMontyField31AVX2<PMP> {
        PackedMontyField31AVX2::<PMP>::from(self) + rhs
    }
}

impl<PMP: PackedMontyParameters> Mul<PackedMontyField31AVX2<PMP>> for MontyField31<PMP> {
    type Output = PackedMontyField31AVX2<PMP>;
    #[inline]
    fn mul(self, rhs: PackedMontyField31AVX2<PMP>) -> PackedMontyField31AVX2<PMP> {
        PackedMontyField31AVX2::<PMP>::from(self) * rhs
    }
}

impl<PMP: PackedMontyParameters> Sub<PackedMontyField31AVX2<PMP>> for MontyField31<PMP> {
    type Output = PackedMontyField31AVX2<PMP>;
    #[inline]
    fn sub(self, rhs: PackedMontyField31AVX2<PMP>) -> PackedMontyField31AVX2<PMP> {
        PackedMontyField31AVX2::<PMP>::from(self) - rhs
    }
}

impl<PMP: PackedMontyParameters> Distribution<PackedMontyField31AVX2<PMP>> for StandardUniform {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> PackedMontyField31AVX2<PMP> {
        PackedMontyField31AVX2::<PMP>(rng.random())
    }
}

#[inline]
#[must_use]
fn interleave1(a: __m256i, b: __m256i) -> (__m256i, __m256i) {
    // We want this to compile to:
    //      vpsllq    t, a, 32
    //      vpsrlq    u, b, 32
    //      vpblendd  res0, a, u, aah
    //      vpblendd  res1, t, b, aah
    // throughput: 1.33 cyc/2 vec (12 els/cyc)
    // latency: (1 -> 1)  1 cyc
    //          (1 -> 2)  2 cyc
    //          (2 -> 1)  2 cyc
    //          (2 -> 2)  1 cyc
    unsafe {
        // Safety: If this code got compiled then AVX2 intrinsics are available.

        // We currently have:
        //   a = [ a0  a1  a2  a3  a4  a5  a6  a7 ],
        //   b = [ b0  b1  b2  b3  b4  b5  b6  b7 ].
        // First form
        //   t = [ a1   0  a3   0  a5   0  a7   0 ].
        //   u = [  0  b0   0  b2   0  b4   0  b6 ].
        let t = x86_64::_mm256_srli_epi64::<32>(a);
        let u = x86_64::_mm256_slli_epi64::<32>(b);

        // Then
        //   res0 = [ a0  b0  a2  b2  a4  b4  a6  b6 ],
        //   res1 = [ a1  b1  a3  b3  a5  b5  a7  b7 ].
        (
            x86_64::_mm256_blend_epi32::<0b10101010>(a, u),
            x86_64::_mm256_blend_epi32::<0b10101010>(t, b),
        )
    }
}

#[inline]
#[must_use]
fn interleave2(a: __m256i, b: __m256i) -> (__m256i, __m256i) {
    // We want this to compile to:
    //      vpalignr  t, b, a, 8
    //      vpblendd  res0, a, t, cch
    //      vpblendd  res1, t, b, cch
    // throughput: 1 cyc/2 vec (16 els/cyc)
    // latency: 2 cyc

    unsafe {
        // Safety: If this code got compiled then AVX2 intrinsics are available.

        // We currently have:
        //   a = [ a0  a1  a2  a3  a4  a5  a6  a7 ],
        //   b = [ b0  b1  b2  b3  b4  b5  b6  b7 ].
        // First form
        //   t = [ a2  a3  b0  b1  a6  a7  b4  b5 ].
        let t = x86_64::_mm256_alignr_epi8::<8>(b, a);

        // Then
        //   res0 = [ a0  a1  b0  b1  a4  a5  b4  b5 ],
        //   res1 = [ a2  a3  b2  b3  a6  a7  b6  b7 ].
        (
            x86_64::_mm256_blend_epi32::<0b11001100>(a, t),
            x86_64::_mm256_blend_epi32::<0b11001100>(t, b),
        )
    }
}

#[inline]
#[must_use]
fn interleave4(a: __m256i, b: __m256i) -> (__m256i, __m256i) {
    // We want this to compile to:
    //      vperm2i128  t, a, b, 21h
    //      vpblendd    res0, a, t, f0h
    //      vpblendd    res1, t, b, f0h
    // throughput: 1 cyc/2 vec (16 els/cyc)
    // latency: 4 cyc

    unsafe {
        // Safety: If this code got compiled then AVX2 intrinsics are available.

        // We currently have:
        //   a = [ a0  a1  a2  a3  a4  a5  a6  a7 ],
        //   b = [ b0  b1  b2  b3  b4  b5  b6  b7 ].
        // First form
        //   t = [ a4  a5  a6  a7  b0  b1  b2  b3 ].
        let t = x86_64::_mm256_permute2x128_si256::<0x21>(a, b);

        // Then
        //   res0 = [ a0  a1  a2  a3  b0  b1  b2  b3 ],
        //   res1 = [ a4  a5  a6  a7  b4  b5  b6  b7 ].
        (
            x86_64::_mm256_blend_epi32::<0b11110000>(a, t),
            x86_64::_mm256_blend_epi32::<0b11110000>(t, b),
        )
    }
}

unsafe impl<FP: FieldParameters> PackedValue for PackedMontyField31AVX2<FP> {
    type Value = MontyField31<FP>;

    const WIDTH: usize = WIDTH;

    #[inline]
    fn from_slice(slice: &[MontyField31<FP>]) -> &Self {
        assert_eq!(slice.len(), Self::WIDTH);
        unsafe {
            // Safety: `[MontyField31<FP>; WIDTH]` can be transmuted to `PackedMontyField31AVX2<FP>` since the
            // latter is `repr(transparent)`. They have the same alignment, so the reference cast is
            // safe too.
            &*slice.as_ptr().cast()
        }
    }
    #[inline]
    fn from_slice_mut(slice: &mut [MontyField31<FP>]) -> &mut Self {
        assert_eq!(slice.len(), Self::WIDTH);
        unsafe {
            // Safety: `[MontyField31<FP>; WIDTH]` can be transmuted to `PackedMontyField31AVX2<FP>` since the
            // latter is `repr(transparent)`. They have the same alignment, so the reference cast is
            // safe too.
            &mut *slice.as_mut_ptr().cast()
        }
    }

    /// Similar to `core:array::from_fn`.
    #[inline]
    fn from_fn<F: FnMut(usize) -> MontyField31<FP>>(f: F) -> Self {
        let vals_arr: [_; WIDTH] = core::array::from_fn(f);
        Self(vals_arr)
    }

    #[inline]
    fn as_slice(&self) -> &[MontyField31<FP>] {
        &self.0[..]
    }
    #[inline]
    fn as_slice_mut(&mut self) -> &mut [MontyField31<FP>] {
        &mut self.0[..]
    }
}

unsafe impl<FP: FieldParameters> PackedField for PackedMontyField31AVX2<FP> {
    type Scalar = MontyField31<FP>;

    #[inline]
    fn dot_product_scalar_packed(scalar_slice: &[Self::Scalar], packed_slice: &[Self]) -> Self {
        // let mut acc = Self::ZERO;
        // for (&scalar, &packed) in scalar_slice.iter().zip(packed_slice) {
        //     acc += packed * scalar
        // }
        // acc
        general_dot_product(scalar_slice, packed_slice)
    }
}

unsafe impl<FP: FieldParameters> PackedFieldPow2 for PackedMontyField31AVX2<FP> {
    #[inline]
    fn interleave(&self, other: Self, block_len: usize) -> (Self, Self) {
        let (v0, v1) = (self.to_vector(), other.to_vector());
        let (res0, res1) = match block_len {
            1 => interleave1(v0, v1),
            2 => interleave2(v0, v1),
            4 => interleave4(v0, v1),
            8 => (v0, v1),
            _ => panic!("unsupported block_len"),
        };
        unsafe {
            // Safety: all values are in canonical form (we haven't changed them).
            (Self::from_vector(res0), Self::from_vector(res1))
        }
    }
}
