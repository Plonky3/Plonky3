//! The `PCLMULQDQ` backend.

use core::arch::x86_64::{
    __m128i, _mm_clmulepi64_si128, _mm_cvtsi128_si64, _mm_set_epi64x, _mm_setzero_si128,
    _mm_shuffle_epi32, _mm_slli_si128, _mm_unpackhi_epi64, _mm_xor_si128,
};
use core::mem::transmute;

use super::basis::TAIL_128;

/// Selects the low quadword of both operands.
///
/// Bit 0 picks the half of the first argument, bit 4 the half of the second.
const LOW_BY_LOW: i32 = 0x00;

/// Selects the high quadword of both operands.
const HIGH_BY_HIGH: i32 = 0x11;

/// Selects the high quadword of the first operand and the low quadword of the second.
const HIGH_BY_LOW: i32 = 0x01;

/// Selects the low quadword of the first operand and the high quadword of the second.
const LOW_BY_HIGH: i32 = 0x10;

/// The carryless product of two 64-bit polynomials over `GF(2)`.
///
/// The instruction accumulates `b << i` for every set bit `i` of `a`.
/// Bit `j` of the result is therefore the coefficient of `x^j`.
#[inline]
pub(super) fn clmul_64x64(a: u64, b: u64) -> u128 {
    // SAFETY: this module is compiled only when `target_feature = "pclmulqdq"` is enabled for
    // the crate, which is what the carryless multiply requires.
    // The remaining intrinsics are `sse2`, always available on `x86_64`.
    unsafe {
        // Arguments run from the highest lane down.
        // The operand goes second and the unused high lane first.
        let a = _mm_set_epi64x(0, a as i64);
        let b = _mm_set_epi64x(0, b as i64);
        let product = _mm_clmulepi64_si128::<LOW_BY_LOW>(a, b);

        // Extracting the high lane directly would need `sse4.1`, so it is moved down instead.
        let low = _mm_cvtsi128_si64(product) as u64;
        let high = _mm_cvtsi128_si64(_mm_unpackhi_epi64(product, product)) as u64;
        u128::from(low) | (u128::from(high) << 64)
    }
}

/// One Horner step of the reduction: `t0 + t1 x^64` modulo the field polynomial.
///
/// # Algorithm
///
/// Split the second argument into its 64-bit halves and rewrite the part that overflows.
///
/// ```text
///     T  = x^7 + x^2 + x + 1                       the modulus tail, since x^128 = T
///     t1 = t1_lo + t1_hi x^64
///
///     t1 x^64 = t1_lo x^64 + t1_hi x^128
///             = t1_lo x^64 + t1_hi T
/// ```
///
/// The first term is a byte-wise shift by eight.
/// The second has degree at most `63 + 7`, so nothing overflows again.
///
/// # Safety
///
/// The caller must be compiled with the `pclmulqdq` target feature.
#[inline]
unsafe fn fold_shifted(t0: __m128i, t1: __m128i) -> __m128i {
    // SAFETY: guaranteed by the caller.
    unsafe {
        // The tail sits in the low lane, which is where the multiply below reads it from.
        let tail = _mm_set_epi64x(0, TAIL_128 as i64);

        // t0 + t1_lo x^64.
        let shifted = _mm_xor_si128(t0, _mm_slli_si128::<8>(t1));

        // ... + t1_hi T.
        _mm_xor_si128(shifted, _mm_clmulepi64_si128::<HIGH_BY_LOW>(t1, tail))
    }
}

/// Multiplication in `GF(2^128) = GF(2)[x] / (x^128 + x^7 + x^2 + x + 1)`.
///
/// Both operands and the result are in the polynomial basis.
///
/// # Algorithm
///
/// Schoolbook over the 64-bit halves, then Horner in `x^64`.
///
/// ```text
///     a b = lo + mid x^64 + hi x^128
///         = lo + x^64 (mid + x^64 hi)
/// ```
///
/// Six carryless products in all: four for the halves, two for the fold.
///
/// # Performance
///
/// Every intermediate stays in a vector register.
/// The general-purpose file has no 128-bit shift.
/// Folding there costs several instructions per step instead of one.
#[inline]
pub(crate) fn poly_mul_128(a: u128, b: u128) -> u128 {
    // SAFETY: this module is compiled only when `target_feature = "pclmulqdq"` is enabled for
    // the crate.
    // The remaining intrinsics are `sse2`, always available on `x86_64`.
    unsafe {
        let x = transmute::<u128, __m128i>(a);
        let y = transmute::<u128, __m128i>(b);

        // The two diagonal half products.
        let low = _mm_clmulepi64_si128::<LOW_BY_LOW>(x, y);
        let high = _mm_clmulepi64_si128::<HIGH_BY_HIGH>(x, y);

        // The two cross products, which share the weight x^64.
        let middle = _mm_xor_si128(
            _mm_clmulepi64_si128::<HIGH_BY_LOW>(x, y),
            _mm_clmulepi64_si128::<LOW_BY_HIGH>(x, y),
        );

        // Inner fold brings the top limb down, outer fold finishes the reduction.
        transmute::<__m128i, u128>(fold_shifted(low, fold_shifted(middle, high)))
    }
}

/// Squaring in `GF(2^128)`, taking and returning the polynomial representation.
///
/// The cross term of `(p0 + p1 x^64)^2` is `2 p0 p1`, which vanishes in characteristic 2.
/// The 256-bit square is therefore `p0^2 + p1^2 x^128`, with nothing between the halves.
#[inline]
pub(crate) fn poly_square_128(a: u128) -> u128 {
    // SAFETY: as in the multiplication above.
    unsafe {
        let x = transmute::<u128, __m128i>(a);
        let low = _mm_clmulepi64_si128::<LOW_BY_LOW>(x, x);
        let high = _mm_clmulepi64_si128::<HIGH_BY_HIGH>(x, x);

        // The inner fold has no middle coefficient to add to, so its exclusive or is skipped.
        let tail = _mm_set_epi64x(0, TAIL_128 as i64);
        let folded = _mm_xor_si128(
            _mm_slli_si128::<8>(high),
            _mm_clmulepi64_si128::<HIGH_BY_LOW>(high, tail),
        );

        transmute::<__m128i, u128>(fold_shifted(low, folded))
    }
}

/// Sum unreduced products with three carryless multiplies per pair and two for reduction.
#[inline]
pub(crate) fn poly_dot_128(pairs: impl Iterator<Item = (u128, u128)>) -> u128 {
    // SAFETY: this module requires PCLMULQDQ and x86-64 supplies SSE2.
    // Every bit pattern is valid in both the integer and vector representations.
    unsafe {
        let mut low = _mm_setzero_si128();
        let mut high = _mm_setzero_si128();
        let mut middle = _mm_setzero_si128();
        for (a, b) in pairs {
            let x = transmute::<u128, __m128i>(a);
            let y = transmute::<u128, __m128i>(b);
            // Accumulate coefficients with weights 1, x^64, and x^128 separately.
            low = _mm_xor_si128(low, _mm_clmulepi64_si128::<LOW_BY_LOW>(x, y));
            high = _mm_xor_si128(high, _mm_clmulepi64_si128::<HIGH_BY_HIGH>(x, y));
            // Karatsuba accumulates the product of the two sums of halves.
            let mixed_x = _mm_xor_si128(x, _mm_shuffle_epi32::<0x4e>(x));
            let mixed_y = _mm_xor_si128(y, _mm_shuffle_epi32::<0x4e>(y));
            middle = _mm_xor_si128(middle, _mm_clmulepi64_si128::<LOW_BY_LOW>(mixed_x, mixed_y));
        }
        // Recover the cross terms from the three accumulated Karatsuba coefficients.
        middle = _mm_xor_si128(middle, _mm_xor_si128(low, high));
        // Invariant: reduction distributes over XOR, including an empty sum.
        transmute::<__m128i, u128>(fold_shifted(low, fold_shifted(middle, high)))
    }
}

/// Multiply by a polynomial of degree below 64 using three carryless products.
#[inline]
pub(crate) fn poly_mul_128_by_64(a: u128, b: u64) -> u128 {
    // SAFETY: the module requires PCLMULQDQ and x86-64 supplies SSE2.
    unsafe {
        let a = transmute::<u128, __m128i>(a);
        let b = _mm_set_epi64x(0, b as i64);
        // Invariant: a*b = a0*b + x^64*(a1*b), so only one Horner fold is needed.
        let low = _mm_clmulepi64_si128::<LOW_BY_LOW>(a, b);
        let middle = _mm_clmulepi64_si128::<HIGH_BY_LOW>(a, b);
        transmute::<__m128i, u128>(fold_shifted(low, middle))
    }
}
