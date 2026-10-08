//! Paired polynomial-field butterflies without extracting scalar lanes.

use core::arch::aarch64::*;

use p3_binary_field::Poly64;

/// Apply each butterfly to two words at a time, returning the covered prefix length.
#[inline]
pub(super) fn butterfly<const INVERSE: bool>(
    lo: &mut [Poly64],
    hi: &mut [Poly64],
    t: Poly64,
) -> usize {
    let covered = lo.len() / 2 * 2;
    // SAFETY: both slices have the same length and each iteration accesses two valid elements.
    // The module is only compiled when the carryless-multiply feature is enabled.
    unsafe {
        let twiddle = vdupq_n_u64(t.to_bits());
        let modulus = vdupq_n_u64(0x1b);
        for start in (0..covered).step_by(8) {
            // Four independent chains expose multiplication latency to the instruction scheduler.
            let count = (covered - start).min(8) / 2;
            let mut low = [vdupq_n_u64(0); 4];
            let mut high = low;
            let mut products = low;
            for pair in 0..count {
                low[pair] = vld1q_u64(lo.as_ptr().add(start + 2 * pair).cast());
                high[pair] = vld1q_u64(hi.as_ptr().add(start + 2 * pair).cast());
                if INVERSE {
                    high[pair] = veorq_u64(high[pair], low[pair]);
                }
                products[pair] = multiply(high[pair], twiddle, modulus);
            }
            for pair in 0..count {
                low[pair] = veorq_u64(low[pair], products[pair]);
                if !INVERSE {
                    high[pair] = veorq_u64(high[pair], low[pair]);
                }
                vst1q_u64(lo.as_mut_ptr().add(start + 2 * pair).cast(), low[pair]);
                vst1q_u64(hi.as_mut_ptr().add(start + 2 * pair).cast(), high[pair]);
            }
        }
    }
    covered
}

/// Multiply and reduce two independent word products in the vector register file.
///
/// # Safety
/// Requires carryless multiplication enabled by the enclosing module's target gate.
#[inline(always)]
unsafe fn multiply(value: uint64x2_t, twiddle: uint64x2_t, modulus: uint64x2_t) -> uint64x2_t {
    // SAFETY: all operations use registers and the module requires the multiplication feature.
    unsafe {
        let a = vreinterpretq_p64_u64(value);
        let b = vreinterpretq_p64_u64(twiddle);
        let p0 = vreinterpretq_u64_p128(vmull_p64(vgetq_lane_p64::<0>(a), vgetq_lane_p64::<0>(b)));
        let p1 = vreinterpretq_u64_p128(vmull_high_p64(a, b));
        let r = vreinterpretq_p64_u64(modulus);
        // Why: x^64 equals 0x1b modulo x^64 + x^4 + x^3 + x + 1.
        let fold = |p| vreinterpretq_u64_p128(vmull_high_p64(vreinterpretq_p64_u64(p), r));
        let t0 = fold(p0);
        let t1 = fold(p1);
        let u0 = fold(t0);
        let u1 = fold(t1);
        let mut e0 = xor3(p0, t0, u0);
        let mut e1 = xor3(p1, t1, u1);
        // Only the low words survive, so preserve whole vectors until their final zip.
        core::arch::asm!("/* {0:v} {1:v} */", inout(vreg) e0, inout(vreg) e1, options(pure, nomem, nostack, preserves_flags));
        vzip1q_u64(e0, e1)
    }
}

/// Sum three vectors, using the three-input instruction when available.
#[inline(always)]
unsafe fn xor3(a: uint64x2_t, b: uint64x2_t, c: uint64x2_t) -> uint64x2_t {
    // SAFETY: the optional instruction is guarded by its compile-time target feature.
    #[cfg(target_feature = "sha3")]
    {
        unsafe { veor3q_u64(a, b, c) }
    }
    #[cfg(not(target_feature = "sha3"))]
    {
        unsafe { veorq_u64(a, veorq_u64(b, c)) }
    }
}
