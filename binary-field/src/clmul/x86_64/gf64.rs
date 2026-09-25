//! `GF(2)[x] / (x^64 + x^4 + x^3 + x + 1)`, one element in the low quadword of a register.

use core::arch::x86_64::{__m128i, _mm_cvtsi64_si128, _mm_cvtsi128_si64};

use crate::clmul::wide::{LOW_BY_LOW, Lanes64, fold};

/// The element in the low quadword, and zero above it.
#[inline(always)]
fn lift(value: u64) -> __m128i {
    // SAFETY: `sse2` is part of the `x86_64` baseline.
    unsafe { _mm_cvtsi64_si128(value as i64) }
}

/// The low quadword.
#[inline(always)]
fn lower(value: __m128i) -> u64 {
    // SAFETY: `sse2` is part of the `x86_64` baseline.
    unsafe { _mm_cvtsi128_si64(value) as u64 }
}

/// Reduces the 128-bit product in a register to the element in its low quadword.
#[inline(always)]
fn reduce(product: __m128i) -> u64 {
    // The fold reads the high half from the low quadword of its second argument.
    lower(fold(product, product.unpack_high(product)))
}

/// Multiplication, taking and returning the polynomial representation.
///
/// One carryless product, then a shift fold that never leaves the vector register.
///
/// A second and third carryless product would fold the high half instead.
///
/// On a multiplier-bound core those cost more than the shifts they replace.
#[inline]
pub(crate) fn poly_mul_64(a: u64, b: u64) -> u64 {
    // The 128-bit product fills the register, then folds back into its low quadword.
    reduce(lift(a).clmul::<LOW_BY_LOW>(lift(b)))
}

/// Squaring, taking and returning the polynomial representation.
#[inline]
pub(crate) fn poly_square_64(a: u64) -> u64 {
    let x = lift(a);

    // The carryless square spreads the bits of x to even positions, then folds.
    reduce(x.clmul::<LOW_BY_LOW>(x))
}

/// Sum unreduced products before paying for one reduction.
#[inline]
pub(crate) fn poly_dot_64(pairs: impl Iterator<Item = (u64, u64)>) -> u64 {
    // Reduction is linear, so the sum stays unreduced until the fold below.
    let sum = pairs.fold(__m128i::zero(), |sum, (a, b)| {
        sum.xor(lift(a).clmul::<LOW_BY_LOW>(lift(b)))
    });

    // One fold for the whole sum.
    reduce(sum)
}
