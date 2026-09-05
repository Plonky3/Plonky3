//! The `PMULL` backend, which lives under the `aes` target feature.

use core::arch::aarch64::{
    uint8x16_t, vdupq_n_u8, vdupq_n_u64, veorq_u8, vextq_u8, vgetq_lane_u64, vmull_high_p64,
    vmull_p64, vreinterpretq_p64_u8, vreinterpretq_u8_u64, vreinterpretq_u64_u8,
};
use core::mem::transmute;

/// The carryless product of two 64-bit polynomials over `GF(2)`.
///
/// `PMULL` accumulates `b << i` for every set bit `i` of `a`.
/// Bit `j` of the result is therefore the coefficient of `x^j`.
#[inline]
pub(super) fn clmul_64x64(a: u64, b: u64) -> u128 {
    // SAFETY: this module is compiled only when `target_feature = "aes"` is enabled for the
    // crate, and `aes` implies `neon`.
    // Together those are what the carryless multiply requires.
    unsafe { vmull_p64(a, b) }
}

/// The carryless product of the low halves of two vectors.
///
/// # Safety
///
/// The caller must be compiled with the `aes` target feature.
#[inline]
unsafe fn clmul_low(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    // SAFETY: guaranteed by the caller.
    // The lane extractions feed the multiply directly, so the operands stay in vector registers.
    unsafe {
        transmute(vmull_p64(
            vgetq_lane_u64::<0>(vreinterpretq_u64_u8(a)),
            vgetq_lane_u64::<0>(vreinterpretq_u64_u8(b)),
        ))
    }
}

/// The carryless product of the high halves of two vectors.
///
/// # Safety
///
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

/// Reduces the 256-bit product `low + high x^128` modulo the field polynomial.
///
/// # Algorithm
///
/// The modulus rewrites `x^128` as its tail.
/// Writing the top limb as `h0 + h1 x^64`:
///
/// ```text
///     T          = x^7 + x^2 + x + 1
///     high x^128 = h0 T + h1 T x^64
///     h1 T       = e0 + e1 x^64                degree at most 63 + 7
///     e1 x^128   = e1 T                        fold the spill back down
///     total      = (h0 + e1) T + e0 x^64
/// ```
///
/// One carryless product raises the spill, one more finishes the reduction.
///
/// # Safety
///
/// The caller must be compiled with the `aes` target feature.
#[inline]
unsafe fn fold_high(low: uint8x16_t, high: uint8x16_t) -> uint8x16_t {
    // SAFETY: guaranteed by the caller.
    unsafe {
        let zero = vdupq_n_u8(0);

        // The tail in both lanes, so the multiply can take it against either half.
        let tail = vreinterpretq_u8_u64(vdupq_n_u64(super::basis::TAIL_128 as u64));

        // The top half of the limb, scaled by the tail.
        let folded = clmul_high(high, tail);

        // `EXT` against zero is a 128-bit shift by 64 in either direction.
        let spill = vextq_u8::<8>(folded, zero);
        let carried = vextq_u8::<8>(zero, folded);

        let reduced = veorq_u8(low, clmul_low(veorq_u8(high, spill), tail));
        veorq_u8(reduced, carried)
    }
}

/// Multiplication in `GF(2^128) = GF(2)[x] / (x^128 + x^7 + x^2 + x + 1)`.
///
/// Both operands and the result are in the polynomial basis.
///
/// # Algorithm
///
/// Schoolbook over the 64-bit halves, then the fold of the modulus.
/// Halves are selected by the multiply's own lane choice and by `EXT`.
/// Every 128-bit intermediate therefore stays in a vector register.
#[inline]
pub(crate) fn poly_mul_128(a: u128, b: u128) -> u128 {
    const {
        // `target_arch = "aarch64"` covers the big-endian AArch64 targets too.
        // There the halves of a `u128` and the lanes of a vector run in opposite orders.
        assert!(
            cfg!(target_endian = "little"),
            "the halves of a `u128` are its vector lanes only on little-endian targets"
        );
    }

    // SAFETY: this module is compiled only when `target_feature = "aes"` is enabled for the
    // crate, and `aes` implies `neon`.
    // Together those are what every intrinsic below requires.
    unsafe {
        let a = transmute::<u128, uint8x16_t>(a);
        let b = transmute::<u128, uint8x16_t>(b);
        let zero = vdupq_n_u8(0);

        // Swapping the halves of one operand puts the two cross products on the same lane
        // choices as the two diagonal ones.
        let swapped = vextq_u8::<8>(b, b);
        let middle = veorq_u8(clmul_low(a, swapped), clmul_high(a, swapped));

        // Split the middle coefficient across the two limbs it straddles.
        let low = veorq_u8(clmul_low(a, b), vextq_u8::<8>(zero, middle));
        let high = veorq_u8(clmul_high(a, b), vextq_u8::<8>(middle, zero));

        transmute::<uint8x16_t, u128>(fold_high(low, high))
    }
}

/// Squaring in `GF(2^128)`, taking and returning the polynomial representation.
///
/// The cross term of `(p0 + p1 x^64)^2` is `2 p0 p1`, which vanishes in characteristic 2.
/// The 256-bit square is therefore `p0^2 + p1^2 x^128`, with nothing between the halves.
#[inline]
pub(crate) fn poly_square_128(a: u128) -> u128 {
    const {
        assert!(
            cfg!(target_endian = "little"),
            "the halves of a `u128` are its vector lanes only on little-endian targets"
        );
    }

    // SAFETY: as in the multiplication above.
    unsafe {
        let a = transmute::<u128, uint8x16_t>(a);
        transmute::<uint8x16_t, u128>(fold_high(clmul_low(a, a), clmul_high(a, a)))
    }
}

/// Sum unreduced products before folding the GHASH modulus once.
#[inline]
pub(crate) fn poly_dot_128(pairs: impl Iterator<Item = (u128, u128)>) -> u128 {
    const {
        assert!(cfg!(target_endian = "little"));
    }
    // SAFETY: this module requires AES and NEON.
    // Every bit pattern is valid in both the integer and vector representations.
    unsafe {
        let zero = vdupq_n_u8(0);
        let (mut low, mut high, mut middle) = (zero, zero, zero);
        for (a, b) in pairs {
            let a = transmute::<u128, uint8x16_t>(a);
            let b = transmute::<u128, uint8x16_t>(b);
            // Swapping halves aligns the cross terms with the diagonal multiply instructions.
            let swapped = vextq_u8::<8>(b, b);
            low = veorq_u8(low, clmul_low(a, b));
            high = veorq_u8(high, clmul_high(a, b));
            middle = veorq_u8(
                middle,
                veorq_u8(clmul_low(a, swapped), clmul_high(a, swapped)),
            );
        }
        // Place the cross terms across the two halves of the 256-bit sum.
        low = veorq_u8(low, vextq_u8::<8>(zero, middle));
        high = veorq_u8(high, vextq_u8::<8>(middle, zero));
        transmute::<uint8x16_t, u128>(fold_high(low, high))
    }
}

/// Multiply by a polynomial of degree below 64 using three carryless products.
#[inline]
pub(crate) fn poly_mul_128_by_64(a: u128, b: u64) -> u128 {
    const {
        assert!(cfg!(target_endian = "little"));
    }
    // SAFETY: this module requires AES and NEON.
    unsafe {
        let a = transmute::<u128, uint8x16_t>(a);
        let b = vreinterpretq_u8_u64(vdupq_n_u64(b));
        let zero = vdupq_n_u8(0);
        // Invariant: a*b = a0*b + x^64*(a1*b).
        let low = clmul_low(a, b);
        let middle = clmul_high(a, b);
        let tail = vreinterpretq_u8_u64(vdupq_n_u64(super::basis::TAIL_128 as u64));
        // Raise the low half and replace the overflowing high half using x^128 = 0x87.
        let folded = veorq_u8(vextq_u8::<8>(zero, middle), clmul_high(middle, tail));
        transmute::<uint8x16_t, u128>(veorq_u8(low, folded))
    }
}
