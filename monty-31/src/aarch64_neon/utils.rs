use core::arch::aarch64::{
    uint16x4_t, uint32x4_t, vandq_u32, vbslq_u32, vceqzq_u32, vcombine_u16, vdup_n_u8, vdup_n_u16,
    vdupq_n_u32, vget_low_u16, vhaddq_u32, vmovl_u16, vmovn_u16, vmovn_u32, vmull_u8, vmulq_u32,
    vshlq_n_u32, vshrq_n_u32, vsubq_u32, vtstq_u32,
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
    const N_PRIME: i32,
>(
    input: uint32x4_t,
) -> uint32x4_t {
    // Verifies the constants at compile time.
    const {
        assert!(N + N_PRIME == TAD::TWO_ADICITY as i32);
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
        // Pre-shift the odd factor: r' = r << N_PRIME.
        // Under constraints r < 2^15 and N_PRIME ≤ 15, r' fits in 31 bits.
        let odd_factor_shifted = vdupq_n_u32((TAD::ODD_FACTOR as u32) << N_PRIME);
        // lo_mul = lo * (r << N_PRIME)
        let lo_mul = vmulq_u32(lo, odd_factor_shifted);

        // Select the correct result based on whether `lo` was zero or not.
        //
        // Create a mask that is all 1s where `lo` was 0, and all 0s otherwise.
        let lo_is_zero_mask = vceqzq_u32(lo);
        // Use Bitwise Select:
        // - if a lane in `lo_is_zero_mask` is 1, select from `P`;
        // - otherwise, select from `lo_shft_nonzero`.
        let lo_shft = vbslq_u32(lo_is_zero_mask, TAD::PACKED_P, lo_mul);

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
    const N: i32,
    const N_PRIME: i32,
>(
    input: uint32x4_t,
) -> uint32x4_t {
    // Verifies the constants at compile time.
    const {
        assert!(N as u32 == TAD::TWO_ADICITY as u32);
        assert!(N + N_PRIME == 31);
    }

    unsafe {
        // Decompose the input into high and low bits
        //
        // Create a bitmask for the lower N bits.
        let mask = vdupq_n_u32((1u32 << N) - 1);
        // Isolate the low N bits of the input.
        let lo = vandq_u32(input, mask);
        // Get the high bits by shifting right with an immediate value.
        let hi = vshrq_n_u32::<N>(input);

        // Compute the main term: `r * lo`
        //
        // For this special prime, `r = 2^N_PRIME - 1`.
        // The multiplication `r * lo` simplifies to `(lo << N_PRIME) - lo`.
        //
        // This is the shift-and-subtract trick, using an immediate shift for efficiency.
        let r_x_lo = vsubq_u32(vshlq_n_u32::<N_PRIME>(lo), lo);

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

/// Multiply a vector of Monty31 field elements by `-2^{-8}`.
///
/// This is a specialized version of `mul_neg_2exp_neg_n_neon` for `N=8` that can be faster
/// on some microarchitectures. It leverages a widening 8-bit to 16-bit multiplication,
/// which mirrors a similar optimization available in AVX512 using `vpmaddubsw`.
///
/// # Safety
/// - The prime `P` must be of the form `P = r * 2^j + 1` with `r` odd and `r < 2^7`.
/// - Input must be given in canonical form `[0, P)`.
/// - Output will be in `(-P, P)`.
#[inline(always)]
pub unsafe fn mul_neg_2exp_neg_8_neon<
    TAD: TwoAdicData + PackedMontyParameters,
    const N_PRIME: i32,
>(
    input: uint32x4_t,
) -> uint32x4_t {
    // Verifies the constants at compile time.
    const {
        assert!(8 + N_PRIME == TAD::TWO_ADICITY as i32);
    }

    unsafe {
        // Decompose the input into high bits (hi) and low 8 bits (lo).
        // Get the high bits by shifting right by 8.
        let hi = vshrq_n_u32::<8>(input);

        // To perform a widening 8-bit multiply, we must first narrow the 32-bit input vector
        // down to an 8-bit vector containing the low bytes.
        let input_u16_low: uint16x4_t = vmovn_u32(input); // u32x4 -> u16x4 (64-bit vector)

        // We combine our 64-bit vector with zeros.
        let zeros_u16 = vdup_n_u16(0);
        let input_u16x8 = vcombine_u16(input_u16_low, zeros_u16);
        let input_u8 = vmovn_u16(input_u16x8); // u16x8 -> u8x8

        // Create a vector containing the odd factor `r` in each 8-bit lane.
        // The constraint `r < 2^7` ensures this is safe.
        let odd_factor_u8 = vdup_n_u8(TAD::ODD_FACTOR as u8);

        // Perform a widening unsigned multiply: u8 * u8 -> u16.
        // This computes `lo * r` for each lane, storing the result in a 16-bit vector.
        let lo_x_r_u16 = vmull_u8(input_u8, odd_factor_u8); // Result is uint16x8_t

        // We only need the results from the 4 original lanes. These are in the lower half
        // of the resulting 16x8 vector. We extract them and then widen back to 32-bit lanes.
        let lo_x_r_low_u16 = vget_low_u16(lo_x_r_u16); // u16x8 -> u16x4
        let lo_x_r = vmovl_u16(lo_x_r_low_u16); // u16x4 -> u32x4

        // Perform the left shift `* 2^(j-N)` where `j-N` is `N_PRIME`.
        let lo_shft_nonzero = vshlq_n_u32::<N_PRIME>(lo_x_r);

        // Now, create a mask to handle the case where the low 8 bits of the input were zero.
        // Isolate the low 8 bits of the original input.
        let lo_mask_vec = vdupq_n_u32(0xFF);
        let lo = vandq_u32(input, lo_mask_vec);
        // Create a bitmask that is all 1s where `lo` was 0, and all 0s otherwise.
        let lo_is_zero_mask = vceqzq_u32(lo);

        // Use Bitwise Select:
        // - if a lane in `lo_is_zero_mask` is 1, select from `P`;
        // - otherwise, select from `lo_shft_nonzero`.
        let lo_shft = vbslq_u32(lo_is_zero_mask, TAD::PACKED_P, lo_shft_nonzero);

        // Final subtraction: `(P or r*2^(j-N)*lo) - hi`.
        vsubq_u32(lo_shft, hi)
    }
}
