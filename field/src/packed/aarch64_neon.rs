//! A collection of helper methods when Neon is available

/// Add the two packed vectors `a` and `b` modulo `p`.
///
/// This allows us to add 4 elements at once.
///
/// Assumes that `p` is less than `2^31` and `a + b <= 2P`.
/// If the inputs are not in this range, the result may be incorrect.
/// The result will be in the range `[0, P]` and equal to `(a + b) mod p`.
/// It will be equal to `P` if and only if `a + b = 2P` so provided `a + b < 2P`
/// the result is guaranteed to be less than `P`.
#[inline]
#[must_use]
pub fn uint32x4_mod_add(a: uint32x4_t, b: uint32x4_t, p: uint32x4_t) -> uint32x4_t {
    // We want this to compile to:
    //      add   t.4s, a.4s, b.4s
    //      sub   u.4s, t.4s, P.4s
    //      umin  res.4s, t.4s, u.4s
    // throughput: .75 cyc/vec (5.33 els/cyc)
    // latency: 6 cyc

    // See field/src/packed/x86_64_avx.rs for a proof of correctness of this algorithm.

    unsafe {
        // Safety: If this code got compiled then NEON intrinsics are available.
        let t = aarch64::vaddq_u32(a, b);
        let u = aarch64::vsubq_u32(t, p);
        aarch64::vminq_u32(t, u)
    }
}
