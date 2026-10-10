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
        let bulk = covered / 8 * 8;
        for start in (0..bulk).step_by(8) {
            // Constant bounds let all four pairs stay in registers, without initializing tail slots.
            let mut low: [uint64x2_t; 4] =
                core::array::from_fn(|pair| vld1q_u64(lo.as_ptr().add(start + 2 * pair).cast()));
            let mut high: [uint64x2_t; 4] =
                core::array::from_fn(|pair| vld1q_u64(hi.as_ptr().add(start + 2 * pair).cast()));
            if INVERSE {
                high = core::array::from_fn(|pair| veorq_u64(high[pair], low[pair]));
            }
            let products: [uint64x2_t; 4] = high.map(|value| multiply(value, twiddle));
            low = core::array::from_fn(|pair| veorq_u64(low[pair], products[pair]));
            if !INVERSE {
                high = core::array::from_fn(|pair| veorq_u64(high[pair], low[pair]));
            }
            for pair in 0..4 {
                vst1q_u64(lo.as_mut_ptr().add(start + 2 * pair).cast(), low[pair]);
                vst1q_u64(hi.as_mut_ptr().add(start + 2 * pair).cast(), high[pair]);
            }
        }
        for start in (bulk..covered).step_by(2) {
            let low = vld1q_u64(lo.as_ptr().add(start).cast());
            let high = vld1q_u64(hi.as_ptr().add(start).cast());
            let high = if INVERSE { veorq_u64(high, low) } else { high };
            let low = veorq_u64(low, multiply(high, twiddle));
            let high = if INVERSE { high } else { veorq_u64(high, low) };
            vst1q_u64(lo.as_mut_ptr().add(start).cast(), low);
            vst1q_u64(hi.as_mut_ptr().add(start).cast(), high);
        }
    }
    covered
}

/// Multiply and reduce two independent word products in the vector register file.
///
/// # Safety
/// Requires carryless multiplication enabled by the enclosing module's target gate.
#[inline(always)]
unsafe fn multiply(value: uint64x2_t, twiddle: uint64x2_t) -> uint64x2_t {
    // SAFETY: all operations use registers and the module requires the multiplication feature.
    unsafe {
        let a = vreinterpretq_p64_u64(value);
        let b = vreinterpretq_p64_u64(twiddle);
        let p0 = vreinterpretq_u64_p128(vmull_p64(vgetq_lane_p64::<0>(a), vgetq_lane_p64::<0>(b)));
        let p1 = vreinterpretq_u64_p128(vmull_high_p64(a, b));
        let low = vzip1q_u64(p0, p1);
        let high = vzip2q_u64(p0, p1);
        // Why: the modulus factors as x^64 + (1 + x) * (1 + x^3).
        // The high half has degree at most 62, so its first doubling loses no bit.
        let a = veorq_u64(high, vshlq_n_u64::<1>(high));
        let spill = vshrq_n_u64::<61>(a);
        let c = xor3(a, spill, vshlq_n_u64::<1>(spill));
        xor3(low, c, vshlq_n_u64::<3>(c))
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
