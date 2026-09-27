//! The `PMULL` backend, which lives under the `aes` target feature.

use core::arch::aarch64::{
    uint8x16_t, uint64x2_t, vdupq_n_u8, vdupq_n_u64, veorq_u8, vextq_u8, vgetq_lane_u64,
    vmull_high_p64, vmull_p64, vreinterpretq_p64_u8, vreinterpretq_u8_u64, vreinterpretq_u64_u8,
    vzip1q_u64, vzip2q_u64,
};
use core::mem::transmute;

// `target_arch = "aarch64"` covers the big-endian AArch64 targets too.
// There the halves of a `u128` and the lanes of a vector run in opposite orders, which every
// transmute between the two below relies on.
const _: () = assert!(
    cfg!(target_endian = "little"),
    "the halves of a `u128` are its vector lanes only on little-endian targets"
);

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

/// The sum of two polynomial-basis elements, taken in a vector register.
///
/// The operands of the carryless products live there, so a sum computed in the integer file
/// would cross over and back around every product it feeds.
#[inline(always)]
pub(crate) fn poly_add_128(a: u128, b: u128) -> u128 {
    // SAFETY: this module is compiled only with the aes target feature, which implies neon.
    // Every bit pattern is valid in both the integer and vector representations.
    unsafe {
        transmute::<uint8x16_t, u128>(veorq_u8(
            transmute::<u128, uint8x16_t>(a),
            transmute::<u128, uint8x16_t>(b),
        ))
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

/// A multiplier held fixed across a run of products, beside its companion `t x^64`.
///
/// # Algorithm
///
/// Reduction modulo the field polynomial is a ring homomorphism, so the `x^64` weight of the
/// varying operand's upper half can move onto the fixed one:
///
/// ```text
///     v   = v0 + v1 x^64
///     u   = t x^64 mod p                  the companion, once per multiplier
///
///     t v = t v0 + u v1
///         = (t0 v0 + u0 v1) + (t1 v0 + u1 v1) x^64
///         =       low       +       high      x^64          both halves below x^127
/// ```
///
/// The four half products pair low with low and high with high once the multiplier and its
/// companion are interleaved half by half, so `PMULL` and `PMULL2` take them straight from
/// the registers:
///
/// ```text
///     halves_low  = [t0, u0]     low  = PMULL(v, halves_low)  + PMULL2(v, halves_low)
///     halves_high = [t1, u1]     high = PMULL(v, halves_high) + PMULL2(v, halves_high)
/// ```
///
/// The product spans three limbs, so one step finishes it: `high x^64` is the low limb of
/// `high` raised by `x^64`, plus its top limb times the modulus tail, both below `x^128`.
/// That is five carryless products per element, against six for a general product.
#[derive(Clone, Copy)]
pub(crate) struct SplitMultiplier {
    /// The low halves of the multiplier and of its companion, in that order.
    halves_low: uint8x16_t,
    /// Their high halves, in the same order.
    halves_high: uint8x16_t,
}

impl SplitMultiplier {
    /// Prepare a multiplier, deriving its companion.
    #[inline]
    pub(crate) fn new(t: u128) -> Self {
        // SAFETY: this module is compiled only with the aes target feature, which implies neon.
        // Every bit pattern is valid in both the integer and vector representations.
        let companion = unsafe {
            let t = transmute::<u128, uint8x16_t>(t);
            // `t x^64 = t0 x^64 + t1 T`, and neither term reaches `x^128`.
            let raised = vextq_u8::<8>(vdupq_n_u8(0), t);
            transmute::<uint8x16_t, u128>(veorq_u8(raised, clmul_high(t, tail())))
        };
        Self::from_parts(t, companion)
    }

    /// Prepare a multiplier whose companion `t x^64 mod p` is already known.
    #[inline(always)]
    pub(crate) fn from_parts(t: u128, companion: u128) -> Self {
        // SAFETY: this module is compiled only with the aes target feature, which implies neon.
        // Every bit pattern is valid in both the integer and vector representations.
        unsafe {
            let t = transmute::<u128, uint64x2_t>(t);
            let companion = transmute::<u128, uint64x2_t>(companion);
            Self {
                halves_low: transmute::<uint64x2_t, uint8x16_t>(vzip1q_u64(t, companion)),
                halves_high: transmute::<uint64x2_t, uint8x16_t>(vzip2q_u64(t, companion)),
            }
        }
    }

    /// The reduced product of the multiplier with `v`.
    #[inline(always)]
    pub(crate) fn mul(self, v: u128) -> u128 {
        // SAFETY: this module is compiled only with the aes target feature, which implies neon.
        // Every bit pattern is valid in both the integer and vector representations.
        unsafe {
            let v = transmute::<u128, uint8x16_t>(v);
            let low = veorq_u8(
                clmul_low(v, self.halves_low),
                clmul_high(v, self.halves_low),
            );
            let high = veorq_u8(
                clmul_low(v, self.halves_high),
                clmul_high(v, self.halves_high),
            );
            let raised = vextq_u8::<8>(vdupq_n_u8(0), high);
            transmute::<uint8x16_t, u128>(veorq_u8(low, veorq_u8(raised, clmul_high(high, tail()))))
        }
    }
}

/// The modulus tail in both halves, so either multiply can take it against either half.
///
/// # Safety
///
/// The caller must be compiled with the `neon` target feature.
#[inline(always)]
unsafe fn tail() -> uint8x16_t {
    // SAFETY: guaranteed by the caller.
    unsafe { vreinterpretq_u8_u64(vdupq_n_u64(super::basis::TAIL_128 as u64)) }
}
