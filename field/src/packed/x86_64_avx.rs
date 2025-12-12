//! A collection of helper methods when AVX2 is available

#[cfg(target_feature = "avx512f")]
use core::arch::x86_64::__m512i;
use core::arch::x86_64::{self, __m128i, __m256i};
use core::mem::transmute;

// Goal: Compute r = lhs + rhs mod P for lhs, rhs <= P < 2^31
// Output should mostly lie in [0, P) but is allowed to equal P if lhs = rhs = P.
//
//   Let t := lhs + rhs. Clearly t \in [0, 2P]
//   Define u := (t - P) mod 2^32 and r := min(t, u)  (Note that it is crucial this is an unsigned min)
//   We argue by cases.
//      - If t is in [0, P), then due to wraparound, u is in [2^32 - P, 2^32 - 1). As
//          2^32 - P > P - 1, we conclude that r = t lies in the correct range.
//      - If t is in [P, 2 P], then u is in [0, P] and r = u lies in the correct range.
//   As both t and u are both equal to lhs + rhs mod P, we conclude that
//   r = (lhs + rhs) mod P and lies in the correct range.
//
// An identical idea works for subtraction.
// Set t := lhs - rhs, u := t + P and output r := min(t, u).

/// Add the packed vectors `a` and `b` modulo `p`.
///
/// This allows us to add 4 elements at once.
///
/// Assumes that `p` is less than `2^31` and `a + b <= 2P`.
/// If the inputs are not in this range, the result may be incorrect.
/// The result will be in the range `[0, P]` and equal to `(a + b) mod p`.
/// It will be equal to `P` if and only if `a + b = 2P`.
#[inline(always)]
#[must_use]
fn mm128_mod_add(a: __m128i, b: __m128i, p: __m128i) -> __m128i {
    // We want this to compile to:
    //      paddd   t, lhs, rhs
    //      psubd   u, t, P
    //      pminud  res, t, u
    // throughput: 1 cyc/vec (8 els/cyc)
    // latency: 3 cyc

    unsafe {
        let t = x86_64::_mm_add_epi32(a, b);
        let u = x86_64::_mm_sub_epi32(t, p);
        x86_64::_mm_min_epu32(t, u)
    }
}

/// Subtract the packed vectors `a` and `b` modulo `p`.
///
/// This allows us to subtract 4 elements at once.
///
/// Assumes that `p` is less than `2^31` and `|a - b| <= P`.
/// If the inputs are not in this range, the result may be incorrect.
/// The result will be in the range `[0, P]` and equal to `(a - b) mod p`.
/// It will be equal to `P` if and only if `a - b = P` so provided `a - b < P`
/// the result is guaranteed to be less than `P`.
#[inline(always)]
#[must_use]
fn mm128_mod_sub(a: __m128i, b: __m128i, p: __m128i) -> __m128i {
    // We want this to compile to:
    //      psubd   t, lhs, rhs
    //      paddd   u, t, P
    //      pminud  res, t, u
    // throughput: 1 cyc/vec (8 els/cyc)
    // latency: 3 cyc

    unsafe {
        let t = x86_64::_mm_sub_epi32(a, b);
        let u = x86_64::_mm_add_epi32(t, p);
        x86_64::_mm_min_epu32(t, u)
    }
}

/// Add the packed vectors `a` and `b` modulo `p`.
///
/// This allows us to add 8 elements at once.
///
/// Assumes that `p` is less than `2^31` and `a + b <= 2P`.
/// If the inputs are not in this range, the result may be incorrect.
/// The result will be in the range `[0, P]` and equal to `(a + b) mod p`.
/// It will be equal to `P` if and only if `a + b = 2P` so provided `a + b < 2P`
/// the result is guaranteed to be less than `P`.
#[inline(always)]
#[must_use]
pub fn mm256_mod_add(lhs: __m256i, rhs: __m256i, p: __m256i) -> __m256i {
    // We want this to compile to:
    //      vpaddd   t, lhs, rhs
    //      vpsubd   u, t, P
    //      vpminud  res, t, u
    // throughput: 1 cyc/vec (8 els/cyc)
    // latency: 3 cyc

    unsafe {
        let t = x86_64::_mm256_add_epi32(lhs, rhs);
        let u = x86_64::_mm256_sub_epi32(t, p);
        x86_64::_mm256_min_epu32(t, u)
    }
}

/// Subtract the packed vectors `a` and `b` modulo `p`.
///
/// This allows us to subtract 8 elements at once.
///
/// Assumes that `p` is less than `2^31` and `|a - b| <= P`.
/// If the inputs are not in this range, the result may be incorrect.
/// The result will be in the range `[0, P]` and equal to `(a - b) mod p`.
/// It will be equal to `P` if and only if `a - b = P` so provided `a - b < P`
/// the result is guaranteed to be less than `P`.
#[inline(always)]
#[must_use]
pub fn mm256_mod_sub(lhs: __m256i, rhs: __m256i, p: __m256i) -> __m256i {
    // We want this to compile to:
    //      vpsubd   t, lhs, rhs
    //      vpaddd   u, t, P
    //      vpminud  res, t, u
    // throughput: 1 cyc/vec (8 els/cyc)
    // latency: 3 cyc

    unsafe {
        let t = x86_64::_mm256_sub_epi32(lhs, rhs);
        let u = x86_64::_mm256_add_epi32(t, p);
        x86_64::_mm256_min_epu32(t, u)
    }
}

#[cfg(target_feature = "avx512f")]
/// Add the packed vectors `a` and `b` modulo `p`.
///
/// This allows us to add 16 elements at once.
///
/// Assumes that `p` is less than `2^31` and `a + b <= 2P`.
/// If the inputs are not in this range, the result may be incorrect.
/// The result will be in the range `[0, P]` and equal to `(a + b) mod p`.
/// It will be equal to `P` if and only if `a + b = 2P` so provided `a + b < 2P`
/// the result is guaranteed to be less than `P`.
#[inline(always)]
#[must_use]
pub fn mm512_mod_add(lhs: __m512i, rhs: __m512i, p: __m512i) -> __m512i {
    // We want this to compile to:
    //      vpaddd   t, lhs, rhs
    //      vpsubd   u, t, P
    //      vpminud  res, t, u
    // throughput: 1.5 cyc/vec (10.67 els/cyc)
    // latency: 3 cyc

    unsafe {
        let t = x86_64::_mm512_add_epi32(lhs, rhs);
        let u = x86_64::_mm512_sub_epi32(t, p);
        x86_64::_mm512_min_epu32(t, u)
    }
}

#[cfg(target_feature = "avx512f")]
/// Subtract the packed vectors `a` and `b` modulo `p`.
///
/// This allows us to subtract 16 elements at once.
///
/// Assumes that `p` is less than `2^31` and `|a - b| <= P`.
/// If the inputs are not in this range, the result may be incorrect.
/// The result will be in the range `[0, P]` and equal to `(a - b) mod p`.
/// It will be equal to `P` if and only if `a - b = P` so provided `a - b < P`
/// the result is guaranteed to be less than `P`.
#[inline(always)]
#[must_use]
pub fn mm512_mod_sub(lhs: __m512i, rhs: __m512i, p: __m512i) -> __m512i {
    // We want this to compile to:
    //      vpsubd   t, lhs, rhs
    //      vpaddd   u, t, P
    //      vpminud  res, t, u
    // throughput: 1.5 cyc/vec (10.67 els/cyc)
    // latency: 3 cyc

    unsafe {
        // Safety: If this code got compiled then AVX-512F intrinsics are available.
        let t = x86_64::_mm512_sub_epi32(lhs, rhs);
        let u = x86_64::_mm512_add_epi32(t, p);
        x86_64::_mm512_min_epu32(t, u)
    }
}

/// Add two arrays of integers modulo `P` using packings.
///
/// Assumes that `P` is less than `2^31` and `a + b <= 2P` for all array pairs `a, b`.
/// If the inputs are not in this range, the result may be incorrect.
/// The result will be in the range `[0, P]` and equal to `(a + b) mod P`.
/// It will be equal to `P` if and only if `a + b = 2P` so provided `a + b < 2P`
/// the result is guaranteed to be less than `P`.
///
/// Scalar add is assumed to be a function which implements `a + b % P` with the
/// same specifications as above.
#[inline(always)]
pub fn packed_mod_add<const WIDTH: usize>(
    a: &[u32; WIDTH],
    b: &[u32; WIDTH],
    res: &mut [u32; WIDTH],
    p: u32,
    scalar_add: fn(u32, u32) -> u32,
) {
    match WIDTH {
        1 => res[0] = scalar_add(a[0], b[0]),
        2 => {
            // For degree 2 extensions: use scalar operations
            res[0] = scalar_add(a[0], b[0]);
            res[1] = scalar_add(a[1], b[1]);
        }
        3 => {
            // For degree 3 extensions: use __m128i, wasting 1 lane
            let out: [u32; 4] = unsafe {
                let a: __m128i = transmute([a[0], a[1], a[2], 0u32]);
                let b: __m128i = transmute([b[0], b[1], b[2], 0u32]);
                let p: __m128i = x86_64::_mm_set1_epi32(p as i32);
                transmute(mm128_mod_add(a, b, p))
            };
            res[..3].copy_from_slice(&out[..3]);
        }
        4 => {
            // Perfectly fits into a m128i vector. The compiler is good at
            // optimising this into AVX2 instructions in cases where we need to
            // do multiple additions.
            let out: [u32; 4] = unsafe {
                let a: __m128i = transmute([a[0], a[1], a[2], a[3]]);
                let b: __m128i = transmute([b[0], b[1], b[2], b[3]]);
                let p: __m128i = x86_64::_mm_set1_epi32(p as i32);
                transmute(mm128_mod_add(a, b, p))
            };

            res.copy_from_slice(&out);
        }
        5 => {
            // We fit what we can into a m128i vector. The final add on
            // is done using a scalar addition. This seems to be faster than
            // trying to fit everything into an m256i vector and makes it much
            // easier for the compiler to optimise in cases where it needs to
            // do multiple additions.
            let out: [u32; 4] = unsafe {
                let a: __m128i = transmute([a[0], a[1], a[2], a[3]]);
                let b: __m128i = transmute([b[0], b[1], b[2], b[3]]);
                let p: __m128i = x86_64::_mm_set1_epi32(p as i32);
                transmute(mm128_mod_add(a, b, p))
            };
            res[4] = scalar_add(a[4], b[4]);

            res[..4].copy_from_slice(&out[..4]);
        }
        6 => {
            // For degree 6 extensions: use __m128i for first 4, then scalar for last 2
            let out: [u32; 4] = unsafe {
                let a: __m128i = transmute([a[0], a[1], a[2], a[3]]);
                let b: __m128i = transmute([b[0], b[1], b[2], b[3]]);
                let p: __m128i = x86_64::_mm_set1_epi32(p as i32);
                transmute(mm128_mod_add(a, b, p))
            };
            res[..4].copy_from_slice(&out);
            res[4] = scalar_add(a[4], b[4]);
            res[5] = scalar_add(a[5], b[5]);
        }
        7 => {
            // For degree 7 extensions: use __m256i, wasting 1 lane
            let out: [u32; 8] = unsafe {
                let a: __m256i = transmute([a[0], a[1], a[2], a[3], a[4], a[5], a[6], 0u32]);
                let b: __m256i = transmute([b[0], b[1], b[2], b[3], b[4], b[5], b[6], 0u32]);
                let p: __m256i = x86_64::_mm256_set1_epi32(p as i32);
                transmute(mm256_mod_add(a, b, p))
            };
            res.copy_from_slice(&out[..7]);
        }
        8 => {
            // This perfectly fits into a single m256i vector.
            let out: [u32; 8] = unsafe {
                let a: __m256i = transmute([a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]]);
                let b: __m256i = transmute([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]);
                let p: __m256i = x86_64::_mm256_set1_epi32(p as i32);
                transmute(mm256_mod_add(a, b, p))
            };

            res.copy_from_slice(&out);
        }
        _ => panic!("Currently unsupported width for packed addition."),
    }
}

/// Subtract two arrays of integers modulo `P` using packings.
///
/// Assumes that `p` is less than `2^31` and `|a - b| <= P`.
/// If the inputs are not in this range, the result may be incorrect.
/// The result will be in the range `[0, P]` and equal to `(a - b) mod p`.
/// It will be equal to `P` if and only if `a - b = P` so provided `a - b < P`
/// the result is guaranteed to be less than `P`.
///
/// Scalar sub is assumed to be a function which implements `a - b % P` with the
/// same specifications as above.
///
/// TODO: Add support for extensions of degree 2,3,6,7.
#[inline(always)]
pub fn packed_mod_sub<const WIDTH: usize>(
    a: &[u32; WIDTH],
    b: &[u32; WIDTH],
    res: &mut [u32; WIDTH],
    p: u32,
    scalar_sub: fn(u32, u32) -> u32,
) {
    match WIDTH {
        1 => res[0] = scalar_sub(a[0], b[0]),
        4 => {
            // Perfectly fits into a m128i vector. The compiler is good at
            // optimising this into AVX2 instructions in cases where we need to
            // do multiple additions.
            let out: [u32; 4] = unsafe {
                let a: __m128i = transmute([a[0], a[1], a[2], a[3]]);
                let b: __m128i = transmute([b[0], b[1], b[2], b[3]]);
                let p: __m128i = x86_64::_mm_set1_epi32(p as i32);
                transmute(mm128_mod_sub(a, b, p))
            };

            res.copy_from_slice(&out);
        }
        5 => {
            // We fit what we can into a m128i vector. The final add on
            // is done using a scalar subtraction. This seems to be faster than
            // trying to fit everything into an m256i vector and makes it much
            // easier for the compiler to optimise in cases where it needs to
            // do multiple additions.
            let out: [u32; 4] = unsafe {
                let a: __m128i = transmute([a[0], a[1], a[2], a[3]]);
                let b: __m128i = transmute([b[0], b[1], b[2], b[3]]);
                let p: __m128i = x86_64::_mm_set1_epi32(p as i32);
                transmute(mm128_mod_sub(a, b, p))
            };
            res[4] = scalar_sub(a[4], b[4]);

            res[..4].copy_from_slice(&out[..4]);
        }
        8 => {
            // This perfectly fits into a single m256i vector.
            let out: [u32; 8] = unsafe {
                let a: __m256i = transmute([a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]]);
                let b: __m256i = transmute([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]);
                let p: __m256i = x86_64::_mm256_set1_epi32(p as i32);
                transmute(mm256_mod_sub(a, b, p))
            };

            res.copy_from_slice(&out);
        }
        _ => panic!("Currently unsupported width for packed subtraction."),
    }
}
