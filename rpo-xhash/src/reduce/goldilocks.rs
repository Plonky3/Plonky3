//! Goldilocks reduction: u128 → u64 mod P, where P = 2^64 - 2^32 + 1.
//!
//! Uses the identity 2^64 ≡ 2^32 - 1 (mod P).
//! The hi/lo split strategy: split 64-bit Goldilocks elements into two 32-bit halves,
//! run the MDS on each half independently (as u64), then recombine.

/// Recombine lo and hi halves into a Goldilocks field element.
/// Input: lo_result, hi_result from two independent MDS applications on the halves.
/// Output: (lo + hi << 32) mod P.
#[inline(always)]
pub fn combine_halves(lo: u64, hi: u64) -> u64 {
    let s = lo as u128 + ((hi as u128) << 32);
    let s_hi = (s >> 64) as u64;
    let s_lo = s as u64;
    let z = (s_hi << 32).wrapping_sub(s_hi); // s_hi * (2^32 - 1)
    let (res, over) = s_lo.overflowing_add(z);
    res.wrapping_add(0u32.wrapping_sub(over as u32) as u64)
}
