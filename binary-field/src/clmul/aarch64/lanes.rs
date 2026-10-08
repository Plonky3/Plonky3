//! The 128-bit NEON register as a backend of the shared `GF(2^64)` algebra.
//!
//! One register serves both the scalar kernels and the packings.

#[cfg(target_feature = "sha3")]
use core::arch::aarch64::veor3q_u64;
use core::arch::aarch64::{
    uint64x2_t, vcombine_u64, vcreate_u64, vdupq_laneq_u64, vdupq_n_u64, veorq_u64, vextq_u64,
    vgetq_lane_u64, vld1q_dup_u64, vld1q_u64, vmull_high_p64, vmull_p64, vreinterpretq_p64_u64,
    vshlq_n_u64, vshrq_n_u64, vst1q_lane_u64, vst1q_u64, vzip1q_u64, vzip2q_u64,
};
use core::mem::transmute;

use crate::clmul::register::Register128;
use crate::clmul::wide::{
    HIGH_BY_HIGH, HIGH_BY_LOW, LOW_BY_LOW, Lanes64, TAIL_64, reduce_by_multiply,
};

// SAFETY for every method below: this module is compiled only when `aes` is enabled.
//
// `aes` implies `neon`, which every intrinsic here requires.
//
// The three-way exclusive or is compiled only under `sha3`, the feature it requires.
impl Lanes64 for uint64x2_t {
    #[inline(always)]
    fn zero() -> Self {
        unsafe { vdupq_n_u64(0) }
    }

    #[inline(always)]
    fn xor(self, other: Self) -> Self {
        unsafe { veorq_u64(self, other) }
    }

    // `EOR3` sums three registers in one instruction.
    #[cfg(target_feature = "sha3")]
    #[inline(always)]
    fn xor3(self, b: Self, c: Self) -> Self {
        unsafe { veor3q_u64(self, b, c) }
    }

    #[inline(always)]
    fn clmul<const IMM: i32>(self, other: Self) -> Self {
        // `PMULL` multiplies the low quadwords, `PMULL2` the high ones.
        //
        // The mixed choice broadcasts the low quadword of the second operand first.
        //
        // A lane read feeds the multiply directly, so no operand leaves the vector file.
        unsafe {
            let product = match IMM {
                LOW_BY_LOW => vmull_p64(vgetq_lane_u64::<0>(self), vgetq_lane_u64::<0>(other)),
                HIGH_BY_HIGH => {
                    vmull_high_p64(vreinterpretq_p64_u64(self), vreinterpretq_p64_u64(other))
                }
                HIGH_BY_LOW => vmull_high_p64(
                    vreinterpretq_p64_u64(self),
                    vreinterpretq_p64_u64(vdupq_laneq_u64::<0>(other)),
                ),
                _ => vmull_high_p64(
                    vreinterpretq_p64_u64(vdupq_laneq_u64::<0>(self)),
                    vreinterpretq_p64_u64(other),
                ),
            };
            transmute::<u128, Self>(product)
        }
    }

    #[inline(always)]
    fn unpack_low(self, other: Self) -> Self {
        unsafe { vzip1q_u64(self, other) }
    }

    #[inline(always)]
    fn unpack_high(self, other: Self) -> Self {
        unsafe { vzip2q_u64(self, other) }
    }

    #[inline(always)]
    fn shl<const N: i32>(self) -> Self {
        unsafe { vshlq_n_u64::<N>(self) }
    }

    #[inline(always)]
    fn shr<const N: i32>(self) -> Self {
        unsafe { vshrq_n_u64::<N>(self) }
    }

    // Apple cores issue `PMULL` on every vector pipe, as cheaply as an exclusive or.
    //
    // So three instructions fold the product, where the shifts take nine.
    #[inline(always)]
    fn reduce_lane(self) -> Self {
        // SAFETY: `neon` is implied by `aes`, under which this module compiles.
        reduce_by_multiply(self, unsafe { vdupq_n_u64(TAIL_64) })
    }

    // Each parity reduces in place, and one interleave collects the two low quadwords.
    #[inline(always)]
    fn reduce_wide(even: Self, odd: Self) -> Self {
        even.reduce_lane().unpack_low(odd.reduce_lane())
    }
}

// SAFETY for every method below: as for the lane operations above.
//
// Every load and store stays inside the array or the reference it is handed.
impl Register128 for uint64x2_t {
    // Apple cores issue `PMULL` on every vector pipe, as cheaply as an exclusive or.
    const CHEAP_MULTIPLY: bool = false;

    #[inline(always)]
    fn lift(value: u64) -> Self {
        // `FMOV` from a general register clears the upper quadword.
        unsafe { vcombine_u64(vcreate_u64(value), vcreate_u64(0)) }
    }

    #[inline(always)]
    fn lower(self) -> u64 {
        unsafe { vgetq_lane_u64::<0>(self) }
    }

    #[inline(always)]
    fn swap(self) -> Self {
        unsafe { vextq_u64::<1>(self, self) }
    }

    #[inline(always)]
    fn load(a: &[u64; 3]) -> (Self, Self) {
        // Bytes 0 to 15 as one register, and the third coordinate in both quadwords.
        unsafe { (vld1q_u64(a.as_ptr()), vld1q_dup_u64(&a[2])) }
    }

    #[inline(always)]
    fn store(pair: Self, last: Self) -> [u64; 3] {
        let mut out = [0u64; 3];

        // One 16-byte store and one 8-byte store, the shape the loads read.
        unsafe {
            vst1q_u64(out.as_mut_ptr(), pair);
            vst1q_lane_u64::<0>(&mut out[2], last);
        }
        out
    }

    #[inline(always)]
    fn load_scalar(k: &u64) -> Self {
        // `LD1R` fills both quadwords, so the mixed product needs no broadcast of its own.
        unsafe { vld1q_dup_u64(k) }
    }
}
