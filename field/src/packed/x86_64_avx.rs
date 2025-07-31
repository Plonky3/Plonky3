//! A collection of helper methods when AVX2 is available

#[cfg(all(feature = "nightly-features", target_feature = "avx512f"))]
use core::arch::x86_64::__m512i;
use core::arch::x86_64::{self, __m256i};

// Goal: Compute r = lhs + rhs mod P for lhs, rhs <= P < 2^31
// Output should mostly lie in [0, P) but is allowed to equal P if lhs = rhs = P.
//
//   Let t := lhs + rhs. Clearly t \in [0, 2P]
//   Define u := (t - P) mod 2^32 and r := min(t, u)
//   We argue by cases.
//      - If t is in [0, P), then due to wraparound, u is in [2^32 - P, 2^32 - 1). As
//          2^32 - P > P - 1, we conclude that r = t lies in the correct range.
//      - If t is in [P, 2 P], then u is in [0, P] and r = u lies in the correct range.
//   As both t and u are both equal to lhs + rhs mod P, we conclude that
//   r = (lhs + rhs) mod P and lies in the correct range.

/// Add the two packed vectors `a` and `b` modulo `p`.
///
/// This allows us to add 8 elements at once.
///
/// Assumes that `p` is less than `2^31` and `a + b < 2P`.
/// If the inputs are not in this range, the result may be incorrect.
/// The result will be in the range `[0, P]` and equal to `(a + b) mod p`.
/// It will be equal to `P` if and only if `a + b = 2P`.
#[inline(always)]
#[must_use]
pub fn mm256_mod_add(a: __m256i, b: __m256i, p: __m256i) -> __m256i {
    // We want this to compile to:
    //      vpaddd   t, lhs, rhs
    //      vpsubd   u, t, P
    //      vpminud  res, t, u
    // throughput: 1 cyc/vec (8 els/cyc)
    // latency: 3 cyc

    unsafe {
        let t = x86_64::_mm256_add_epi32(a, b);
        let u = x86_64::_mm256_sub_epi32(t, p);
        x86_64::_mm256_min_epu32(t, u)
    }
}

#[cfg(all(feature = "nightly-features", target_feature = "avx512f"))]
/// Add the two packed vectors `a` and `b` modulo `p`.
///
/// This allows us to add 16 elements at once.
///
/// Assumes that `p` is less than `2^31` and `a + b < 2P`.
/// If the inputs are not in this range, the result may be incorrect.
/// The result will be in the range `[0, P]` and equal to `(a + b) mod p`.
/// It will be equal to `P` if and only if `a + b = 2P`.
#[inline(always)]
#[must_use]
pub fn mm512_mod_add(a: __m512i, b: __m512i, p: __m512i) -> __m512i {
    // We want this to compile to:
    //      vpaddd   t, lhs, rhs
    //      vpsubd   u, t, P
    //      vpminud  res, t, u
    // throughput: 1.5 cyc/vec (10.67 els/cyc)
    // latency: 3 cyc

    unsafe {
        let t = x86_64::_mm512_add_epi32(a, b);
        let u = x86_64::_mm512_sub_epi32(t, p);
        x86_64::_mm512_min_epu32(t, u)
    }
}
