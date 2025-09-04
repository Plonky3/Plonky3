use core::arch::aarch64::{
    self, uint16x4_t, uint32x4_t, vandq_u32, vbslq_u32, vceqzq_u32, vdup_n_u16, vdupq_n_s32,
    vdupq_n_u32, vhaddq_u32, vmovn_u32, vshlq_u32, vshrq_n_u32, vsubq_u32, vtstq_u32,
};

use crate::{PackedMontyParameters, TwoAdicData};

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

/// Multiply a vector of Monty31 field elements by `-2^{-N}`.
///
/// # Safety
/// - The prime `P` must be of the form `P = r * 2^j + 1` with `r` odd and `r < 2^15`.
/// - `N` must be between 0 and 15.
/// - Inputs must be in canonical form.
/// - Output will be in `(-P, P)`.
/// - `N + N_PRIME` must equal `TAD::TWO_ADICITY`.
#[inline(always)]
pub unsafe fn mul_neg_2exp_neg_n_neon<
    TAD: TwoAdicData + PackedMontyParameters,
    const N: i32,
    const N_PRIME: u32,
>(
    input: uint32x4_t,
) -> uint32x4_t {
    // Verifies the constants at compile time.
    const {
        assert!(N as u32 + N_PRIME == TAD::TWO_ADICITY as u32);
    }

    unsafe {
        // Decompose the input into high and low bits
        //
        // Create a bitmask for the lower N bits. The compiler will treat this as a constant.
        let mask = vdupq_n_u32((1u32 << N) - 1);
        // Isolate the low N bits of the input.
        let lo = vandq_u32(input, mask);
        // Get the high bits by shifting right by the constant N.
        let hi = vshrq_n_u32::<N>(input);

        // Compute the main term: `r * 2^(j-N) * lo`
        //
        // Create a 64-bit vector containing the odd factor `r` in each lane.
        let odd_factor = vdup_n_u16(TAD::ODD_FACTOR as u16);
        // Narrow `lo` from a u32x4 vector to a u16x4 vector (safe as N <= 15).
        let lo_u16: uint16x4_t = vmovn_u32(lo);
        // Perform a widening multiply: u16 * u16 -> u32.
        let lo_x_r: uint32x4_t = aarch64::vmull_u16(lo_u16, odd_factor);
        // Create a vector for the left shift amount. `j-N` is `N_PRIME`.
        let shift_n_prime = vdupq_n_s32(N_PRIME as i32);
        // Perform the final shift to get the full term.
        let lo_shft_nonzero = vshlq_u32(lo_x_r, shift_n_prime);

        // Select the correct result based on whether `lo` was zero or not.
        //
        // Create a mask that is all 1s where `lo` was 0, and all 0s otherwise.
        let lo_is_zero_mask = vceqzq_u32(lo);
        // Use Bitwise Select:
        // - if a lane in `lo_is_zero_mask` is 1, select from `P`;
        // - otherwise, select from `lo_shft_nonzero`.
        let lo_shft = vbslq_u32(lo_is_zero_mask, TAD::PACKED_P, lo_shft_nonzero);

        // Final subtraction
        //
        // The result is `(P or r*2^(j-N)*lo) - hi`.
        vsubq_u32(lo_shft, hi)
    }
}

/// Multiply a vector of Monty31 elements by `-2^{-N}` where `P = 2^31 - 2^N + 1`.
///
/// This is a specialized version for primes where the odd factor `r` has the form `2^k - 1`,
/// which simplifies the multiplication `r * lo` to `(lo << k) - lo`.
///
/// # Safety
/// - The prime `P` must have the special form `P = 2^31 - 2^N + 1`.
/// - Input must be canonical.
/// - Output will be in `(-P, P)`.
#[inline(always)]
pub unsafe fn mul_neg_2exp_neg_two_adicity_neon<
    TAD: TwoAdicData + PackedMontyParameters,
    const N: u32,
    const N_PRIME: u32,
>(
    input: uint32x4_t,
) -> uint32x4_t {
    // Verifies the constants at compile time.
    const {
        assert!(N == TAD::TWO_ADICITY as u32);
        assert!(N + N_PRIME == 31);
    }

    unsafe {
        // Decompose the input into high and low bits
        //
        // Create a bitmask for the lower N bits.
        let mask = vdupq_n_u32((1u32 << N) - 1);
        // Create a vector for the right shift amount.
        let shift_n = vdupq_n_s32(-(N as i32));
        // Isolate the low N bits of the input.
        let lo = vandq_u32(input, mask);
        // Get the high bits by shifting right.
        let hi = vshlq_u32(input, shift_n);

        // Compute the main term: `r * lo`
        //
        // For this special prime, `r = 2^N_PRIME - 1`.
        // The multiplication `r * lo` simplifies to `(lo * 2^N_PRIME) - lo`.
        let shift_n_prime = vdupq_n_s32(N_PRIME as i32);
        // This is the shift-and-subtract trick to perform the multiplication.
        let r_x_lo = vsubq_u32(vshlq_u32(lo, shift_n_prime), lo);

        // Select the correct result based on whether `lo` was zero
        //
        // Create a mask that is all 1s where `lo` was 0.
        let lo_is_zero_mask = vceqzq_u32(lo);
        // Use Bitwise Select to choose between `P` (if `lo` was 0) and `r_x_lo`.
        let lo_shft = vbslq_u32(lo_is_zero_mask, TAD::PACKED_P, r_x_lo);

        // Final subtraction
        //
        // The result is `(P or r*lo) - hi`.
        vsubq_u32(lo_shft, hi)
    }
}
