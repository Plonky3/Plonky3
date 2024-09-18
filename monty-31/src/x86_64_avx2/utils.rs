use core::arch::x86_64::{self, __m256i};
use core::mem::transmute;

use crate::{MontyParameters, PackedMontyParameters};

/// Halve a vector of Monty31 field elements in canonical form.
/// If the inputs are not in canonical form, the result is undefined.
#[inline(always)]
pub fn halve_avx2<MP: MontyParameters>(input: __m256i) -> __m256i {
    // We want this to compile to:
    //      vpand    least_bit, val, ONE
    //      vpsrld   t, val, 1
    //      vpsignd  maybe_half, HALF, least_bit
    //      vpaddd   res, t, maybe_half
    // throughput: 1.33 cyc/vec
    // latency: 3 cyc

    // Given an element val in [0, P), we want to compute val/2 mod P.
    // If val is even: val/2 mod P = val/2 = val >> 1.
    // If val is odd: val/2 mod P = (val + P)/2 = (val >> 1) + (P + 1)/2
    unsafe {
        // Safety: If this code got compiled then AVX2 intrinsics are available.
        const ONE: __m256i = unsafe { transmute([1; 8]) };
        let half: __m256i = transmute([(MP::PRIME + 1) / 2; 8]); // Compiler realises this is constant.

        let least_bit = x86_64::_mm256_and_si256(input, ONE); // Determine the parity of val.
        let t = x86_64::_mm256_srli_epi32::<1>(input);
        let maybe_half = x86_64::_mm256_sign_epi32(half, least_bit); // This does nothing when least_bit = 1 and sets the corresponding entry to 0 when least_bit = 0
        x86_64::_mm256_add_epi32(t, maybe_half)
    }
}

/// Add two vectors of Monty31 field elements with lhs in canonical form and rhs in (-P, P).
/// To reiterate, the two inputs are not symmetric, lhs must be positive. Return a value in canonical form.
/// If the inputs do not conform to these restrictions, the result is undefined.
#[inline(always)]
pub unsafe fn signed_add_avx2<PMP: PackedMontyParameters>(lhs: __m256i, rhs: __m256i) -> __m256i {
    // We want this to compile to:
    //      vpsignd  pos_neg_P,  P, rhs
    //      vpaddd   sum,        lhs,   rhs
    //      vpsubd   sum_corr,   sum,   pos_neg_P
    //      vpminud  res,        sum,   sum_corr
    // throughput: 1.33 cyc/vec
    // latency: 3 cyc

    // Let t = lhs + rhs mod 2^32, we want to return t mod P while correcting for any possible wraparound.
    // We make use of the fact wrapping addition acts identically on signed and unsigned inputs.

    // If rhs is positive, lhs + rhs < 2P < 2^32 and so we interpret t as a unsigned 32 bit integer.
    //      In this case, t mod P = min_{u32}(t, t - P) where min_{u32} takes the min treating both inputs as unsigned 32 bit integers.
    //      This works as if t >= P then t - P < t and if t < P then, due to wraparound, t - P outputs t - P + 2^32 > t.
    // If rhs is negative, -2^31 < -P < lhs + rhs < P < 2^31 so we interpret t as a signed 32 bit integer.
    //      In this case t mod P = min_{u32}(t, t + P)
    //      This works as if t > 0 then t < t + P and if t < 0 then due to wraparound when we interpret t as an unsigned integer it becomes
    //      2^32 + t > t + P.
    // if rhs = 0 then we can just return t = lhs as it is already in the desired range.
    unsafe {
        // If rhs > 0 set the value to P, if rhs < 0 set it to -P and if rhs = 0 set it to 0.
        let pos_neg_p = x86_64::_mm256_sign_epi32(PMP::PACKED_P, rhs);

        // Compute t = lhs + rhs
        let sum = x86_64::_mm256_add_epi32(lhs, rhs);

        // sum_corr = (t - P) if rhs > 0, t + P if rhs < 0 and t if rhs = 0 as desired.
        let sum_corr = x86_64::_mm256_sub_epi32(sum, pos_neg_p);

        x86_64::_mm256_min_epu32(sum, sum_corr)
    }
}
