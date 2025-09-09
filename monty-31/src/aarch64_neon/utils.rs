use core::arch::aarch64::{
    self, uint32x4_t, vaddq_u32, vandq_u32, vdupq_n_u32, vhaddq_u32, vminq_u32, vmlsq_n_u32,
    vtstq_u32,
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

/// Multiply a vector of Monty31 field elements by `2^{-N}`.
///
/// # Safety
/// - The prime `P` must be of the form `P = r * 2^j + 1` with `r` odd and `j` equal to `TAD::TWO_ADICITY`.
/// - `N` must be less than or equal to `j`.
/// - Inputs must be in canonical form `[0, P)`.
/// - Output will be in canonical form `[0, P)`.
#[inline(always)]
pub unsafe fn mul_2exp_neg_n_neon<TAD: TwoAdicData + PackedMontyParameters, const N: i32>(
    input: uint32x4_t,
) -> uint32x4_t {
    // Verifies the constants at compile time.
    const {
        assert!(N <= TAD::TWO_ADICITY as i32);
    }

    unsafe {
        // Decompose the input into high and low bits.
        let mask = aarch64::vdupq_n_u32((1u32 << N) - 1);
        let lo = aarch64::vandq_u32(input, mask);
        let hi = aarch64::vshrq_n_u32::<N>(input);

        // This computes `hi - lo * (r * 2^N_PRIME) = hi - lo * (P-1)/2^N`.
        //
        // As `lo < 2^N`, `r * lo * 2^N_PRIME < r * 2^(N + N_PRIME) = r * 2^TWO_ADICITY < P`, so no overflow occurs.
        // The vmlsq_n_u32 instruction fuses the constant multiplication and subtraction.
        let n_prime = TAD::TWO_ADICITY as i32 - N;
        let res = vmlsq_n_u32(hi, lo, (TAD::ODD_FACTOR as u32) << n_prime);

        // The result `res` is now in `(-P, P)`. We perform a branchless
        // reduction to bring it into the canonical range `[0, P)`.
        //
        // If res is negative (represented as a large u32), `res + P` is correct.
        // Otherwise, `res` is correct. `vminq_u32` selects the correct one.
        let u = aarch64::vaddq_u32(res, TAD::PACKED_P);
        aarch64::vminq_u32(res, u)
    }
}

/// Multiply a vector of Monty31 elements by `2^{-N}` where `P = 2^31 - 2^N + 1`.
///
/// This is a specialized version for primes where the odd factor `r` has the form `2^k - 1`,
/// which simplifies the multiplication `r * lo` to `(lo << k) - lo`.
///
/// # Safety
/// - The prime `P` must have the special form `P = 2^31 - 2^N + 1`.
/// - The constants must satisfy `N + N_PRIME = 31`.
/// - Input must be canonical `[0, P)`.
/// - Output will be in canonical form `[0, P)`.
#[inline(always)]
pub unsafe fn mul_2exp_neg_two_adicity_neon<
    TAD: TwoAdicData + PackedMontyParameters,
    const N: i32,
    const N_PRIME: i32,
>(
    input: uint32x4_t,
) -> uint32x4_t {
    // Verifies the constants at compile time.
    const {
        assert!(N == TAD::TWO_ADICITY as i32);
        assert!(N + N_PRIME == 31);
    }

    unsafe {
        // Decompose into high and low bits.
        let mask = vdupq_n_u32((1u32 << N) - 1);
        let lo = vandq_u32(input, mask);
        // Get the high bits by shifting right and add the lo bits.
        let hi_plus_lo = aarch64::vsraq_n_u32::<N>(lo, input);

        // Multiply the lo bits by 2^N_PRIME = (r + 1)
        let lo_shft = aarch64::vshlq_n_u32::<N_PRIME>(lo);

        // Compute input * 2^{-N} = hi + lo - (r + 1) * lo
        let res = aarch64::vsubq_u32(hi_plus_lo, lo_shft);

        // Branchless reduction from `(-P, P)` to `[0, P)`.
        let u = vaddq_u32(res, TAD::PACKED_P);
        vminq_u32(res, u)
    }
}
