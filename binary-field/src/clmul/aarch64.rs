//! The `PMULL` backend, which lives under the `aes` target feature.

use core::arch::aarch64::{
    uint64x2_t, vdupq_n_u64, veorq_u64, vextq_u64, vgetq_lane_u64, vmull_p64,
};

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

/// Register-resident Karatsuba product and PMULL reduction for independent batch lanes.
#[inline]
pub(super) fn poly_mul_128_batch(a: u128, b: u128) -> u128 {
    // SAFETY: this module is compiled only with the aes target feature.
    unsafe {
        let av: uint64x2_t = core::mem::transmute(a);
        let bv: uint64x2_t = core::mem::transmute(b);
        let low: uint64x2_t =
            core::mem::transmute(vmull_p64(vgetq_lane_u64::<0>(av), vgetq_lane_u64::<0>(bv)));
        let high: uint64x2_t =
            core::mem::transmute(vmull_p64(vgetq_lane_u64::<1>(av), vgetq_lane_u64::<1>(bv)));
        let mid: uint64x2_t = core::mem::transmute(vmull_p64(
            vgetq_lane_u64::<0>(av) ^ vgetq_lane_u64::<1>(av),
            vgetq_lane_u64::<0>(bv) ^ vgetq_lane_u64::<1>(bv),
        ));
        let middle = veorq_u64(veorq_u64(mid, low), high);
        let zero = vdupq_n_u64(0);
        let lo = veorq_u64(low, vextq_u64::<1>(zero, middle));
        let hi = veorq_u64(high, vextq_u64::<1>(middle, zero));
        // Fold the high polynomial by x^128 = 0x87. The upper fold spills at
        // most seven bits; one more multiplication by 0x87 reduces those too.
        let f0: uint64x2_t = core::mem::transmute(vmull_p64(vgetq_lane_u64::<0>(hi), 0x87));
        let f1: uint64x2_t = core::mem::transmute(vmull_p64(vgetq_lane_u64::<1>(hi), 0x87));
        let spill: uint64x2_t = core::mem::transmute(vmull_p64(vgetq_lane_u64::<1>(f1), 0x87));
        core::mem::transmute(veorq_u64(
            veorq_u64(lo, f0),
            veorq_u64(vextq_u64::<1>(zero, f1), spill),
        ))
    }
}
