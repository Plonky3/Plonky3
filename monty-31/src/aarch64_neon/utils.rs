use core::arch::aarch64::{uint32x4_t, vandq_u32, vdupq_n_u32, vhaddq_u32, vtstq_u32};

use crate::PackedMontyParameters;

/// Halve a vector of Monty31 field elements in canonical form using NEON.
///
/// - If `val` is even, computes `val / 2`.
/// - If `val` is odd, computes `(val + P) / 2`.
#[inline(always)]
pub(crate) fn halve_neon<PMP: PackedMontyParameters>(input: uint32x4_t) -> uint32x4_t {
    unsafe {
        let one = vdupq_n_u32(1);

        // Check if the least significant bit is set (i.e., if the number is odd).
        //
        // vtstq_u32 returns a mask of all 1s if the bit is set, and all 0s otherwise.
        let is_odd_mask = vtstq_u32(input, one);

        // Select `P` if the corresponding input is odd, or `0` if it's even.
        let to_add = vandq_u32(PMP::PACKED_P, is_odd_mask);

        // Halving add computes `(input + to_add) >> 1` for each lane.
        // - If input is even: (input + 0) >> 1 = input / 2
        // - If input is odd: (input + P) >> 1 = (input + P) / 2
        vhaddq_u32(input, to_add)
    }
}
