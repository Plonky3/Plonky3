//! The `PMULL` backend, which lives under the `aes` target feature.

use core::arch::aarch64::{
    uint8x16_t, vdupq_n_u8, vdupq_n_u64, veorq_u8, vextq_u8, vgetq_lane_u64, vmull_high_p64,
    vmull_p64, vreinterpretq_p64_u8, vreinterpretq_u8_u64, vreinterpretq_u64_u8,
};
use core::mem::transmute;

/// The carryless product of two 64-bit polynomials over `GF(2)`.
///
/// `PMULL` accumulates `b << i` for every set bit `i` of `a`, so bit `j` of the result is the
/// coefficient of `x^j` exactly as the rest of this module assumes.
#[inline]
pub(super) fn clmul_64x64(a: u64, b: u64) -> u128 {
    // SAFETY: this module is compiled only when `target_feature = "aes"` is enabled for the
    // crate, and `aes` implies `neon`; together those are what `vmull_p64` requires.
    unsafe { vmull_p64(a, b) }
}

/// The carryless product of the low halves of `a` and `b`.
///
/// # Safety
/// The caller must be compiled with the `aes` target feature.
#[inline]
unsafe fn clmul_low(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    // SAFETY: guaranteed by the caller. The lane extractions feed `PMULL` directly, so the
    // operands stay in vector registers.
    unsafe {
        transmute(vmull_p64(
            vgetq_lane_u64::<0>(vreinterpretq_u64_u8(a)),
            vgetq_lane_u64::<0>(vreinterpretq_u64_u8(b)),
        ))
    }
}

/// The carryless product of the high halves of `a` and `b`.
///
/// # Safety
/// The caller must be compiled with the `aes` target feature.
#[inline]
unsafe fn clmul_high(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    // SAFETY: guaranteed by the caller.
    unsafe {
        transmute(vmull_high_p64(
            vreinterpretq_p64_u8(a),
            vreinterpretq_p64_u8(b),
        ))
    }
}

/// Multiplication in `GF(2^128) = GF(2)[x] / (x^128 + x^7 + x^2 + x + 1)`, both operands and the
/// result in the polynomial basis.
///
/// Schoolbook over the 64-bit halves, then the two-round fold of the modulus, with every 128-bit
/// intermediate held in a vector register: the halves are selected by `PMULL`'s own lane choice
/// and by `EXT`, and the fold multiplies by the modulus tail with `PMULL` rather than shifting.
///
/// The fold is the identity `x^128 ≡ x^7 + x^2 + x + 1`. Writing the high half of the product as
/// `h0 + h1·x^64` and the tail as `T`, the first round gives `h1·T = e0 + e1·x^64` with `e1` of
/// degree at most 6, and the second folds `e1·x^128 ≡ e1·T` back down. Collecting the two terms
/// that are multiplied by `T` leaves `(h0 + e1)·T + e0·x^64`, of degree at most 126, so a single
/// further product finishes the reduction.
#[inline]
pub(super) fn poly_mul_128(a: u128, b: u128) -> u128 {
    // SAFETY: this module is compiled only when `target_feature = "aes"` is enabled for the
    // crate, and `aes` implies `neon`; together those are what every intrinsic below requires.
    unsafe {
        let a = transmute::<u128, uint8x16_t>(a);
        let b = transmute::<u128, uint8x16_t>(b);
        let zero = vdupq_n_u8(0);
        // The tail `x^7 + x^2 + x + 1` of the modulus, in both lanes so `PMULL` can take it
        // against either half.
        let tail = vreinterpretq_u8_u64(vdupq_n_u64(super::basis::TAIL_128 as u64));

        // `EXT #8` swaps the halves, so the two cross products come from the same lane choices
        // as the two diagonal ones.
        let swapped = vextq_u8::<8>(b, b);
        let middle = veorq_u8(clmul_low(a, swapped), clmul_high(a, swapped));

        // `EXT` against zero is a 128-bit shift by 64 in either direction.
        let low = veorq_u8(clmul_low(a, b), vextq_u8::<8>(zero, middle));
        let high = veorq_u8(clmul_high(a, b), vextq_u8::<8>(middle, zero));

        let folded = clmul_high(high, tail);
        let spill = vextq_u8::<8>(folded, zero);
        let carried = vextq_u8::<8>(zero, folded);

        let reduced = veorq_u8(low, clmul_low(veorq_u8(high, spill), tail));
        transmute::<uint8x16_t, u128>(veorq_u8(reduced, carried))
    }
}
