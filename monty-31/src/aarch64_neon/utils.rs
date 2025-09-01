use core::arch::aarch64::{uint32x4_t, vaddq_u32, vandq_u32, vdupq_n_u32, vshrq_n_u32, vtstq_u32};

use crate::PackedMontyParameters;

/// Halve a vector of Monty31 field elements in canonical form using NEON.
///
/// - If `val` is even, computes `val / 2`.
/// - If `val` is odd, computes `(val + P) / 2`.
#[inline(always)]
pub(crate) fn halve_neon<PMP: PackedMontyParameters>(input: uint32x4_t) -> uint32x4_t {
    unsafe {
        let one = vdupq_n_u32(1);
        let half_p = vdupq_n_u32(PMP::PRIME.div_ceil(2));

        // Check if the least significant bit is set (i.e., if the number is odd).
        //
        // vtstq_u32 returns a mask of all 1s if the bit is set, and all 0s otherwise.
        let is_odd_mask = vtstq_u32(input, one);

        // Right shift by 1 to perform the division by 2.
        let val_div_2 = vshrq_n_u32(input, 1);

        // Select `half_p` if odd, 0 otherwise.
        let to_add = vandq_u32(half_p, is_odd_mask);

        // Add the conditional term.
        vaddq_u32(val_div_2, to_add)
    }
}
