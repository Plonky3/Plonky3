//! Sixteen AES-field products per NEON register pair.

use core::arch::aarch64::*;

/// Multiply the whole sixteen-byte blocks, leaving only the scalar tail.
#[inline]
pub(super) fn mul_prefix<'a, 'b>(dst: &'a mut [u8], src: &'b [u8]) -> (&'a mut [u8], &'b [u8]) {
    let covered = dst.len() / 16 * 16;
    let (dst, tail) = dst.split_at_mut(covered);
    let (src, rest) = src.split_at(covered);
    for (a, b) in dst
        .as_chunks_mut::<16>()
        .0
        .iter_mut()
        .zip(src.as_chunks::<16>().0)
    {
        // SAFETY: both arrays cover one full unaligned register and NEON is an AArch64 baseline feature.
        unsafe {
            let a_reg = vld1q_u8(a.as_ptr());
            let b_reg = vld1q_u8(b.as_ptr());
            // Byte carryless multiplication is available without the optional 64-bit AES extension.
            let lo = vmull_p8(
                vreinterpret_p8_u8(vget_low_u8(a_reg)),
                vreinterpret_p8_u8(vget_low_u8(b_reg)),
            );
            let hi = vmull_high_p8(vreinterpretq_p8_u8(a_reg), vreinterpretq_p8_u8(b_reg));
            let product = reduce(vreinterpretq_u16_p16(lo), vreinterpretq_u16_p16(hi));
            vst1q_u8(a.as_mut_ptr(), product);
        }
    }
    (tail, rest)
}

/// Add shifted byte polynomials to complete sixteen-lane blocks.
#[inline]
pub(super) fn add_power_prefix<const POWER: i32>(dst: &mut [u16], src: &[u8]) -> usize {
    let covered = src.len() / 16 * 16;
    for start in (0..covered).step_by(16) {
        // SAFETY: each source block has sixteen bytes and each destination block
        // has sixteen u16 lanes. NEON is an AArch64 baseline feature.
        unsafe {
            let values = vld1q_u8(src.as_ptr().add(start));
            let low = vshll_n_u8::<POWER>(vget_low_u8(values));
            let high = vshll_high_n_u8::<POWER>(values);
            let target = dst.as_mut_ptr().add(start);
            vst1q_u16(target, veorq_u16(vld1q_u16(target), low));
            vst1q_u16(target.add(8), veorq_u16(vld1q_u16(target.add(8)), high));
        }
    }
    covered
}

/// Reduce complete sixteen-lane blocks with the shared nibble-table fold.
#[inline]
pub(super) fn reduce_prefix(src: &[u16], dst: &mut [u8]) -> usize {
    let covered = src.len() / 16 * 16;
    for start in (0..covered).step_by(16) {
        // SAFETY: the source provides sixteen readable u16 lanes and the destination
        // sixteen writable bytes. All loads and stores may be unaligned.
        unsafe {
            let source = src.as_ptr().add(start);
            let out = reduce(vld1q_u16(source), vld1q_u16(source.add(8)));
            vst1q_u8(dst.as_mut_ptr().add(start), out);
        }
    }
    covered
}

/// The AES modulus maps a high byte to its product by `0x1b`.
const fn reduction_table(high_nibble: bool) -> [u8; 16] {
    let mut table = [0; 16];
    let mut i = 0;
    while i < 16 {
        let byte = if high_nibble { (i as u8) << 4 } else { i as u8 };
        table[i] = crate::aes::mul_bytes(byte, 0x1b);
        i += 1;
    }
    table
}

/// Images of the high byte's low nibble under coefficient reduction.
const LOW_NIBBLE: [u8; 16] = reduction_table(false);
/// Images of the high byte's high nibble under coefficient reduction.
const HIGH_NIBBLE: [u8; 16] = reduction_table(true);

/// Reduce sixteen polynomial products with two register-resident nibble lookups.
#[inline(always)]
unsafe fn reduce(lo: uint16x8_t, hi: uint16x8_t) -> uint8x16_t {
    // SAFETY: NEON is an AArch64 baseline feature; both tables hold sixteen bytes.
    unsafe {
        let low = vcombine_u8(vmovn_u16(lo), vmovn_u16(hi));
        let high = vcombine_u8(vshrn_n_u16::<8>(lo), vshrn_n_u16::<8>(hi));
        let low_image = vqtbl1q_u8(
            vld1q_u8(LOW_NIBBLE.as_ptr()),
            vandq_u8(high, vdupq_n_u8(15)),
        );
        let high_image = vqtbl1q_u8(vld1q_u8(HIGH_NIBBLE.as_ptr()), vshrq_n_u8::<4>(high));
        #[cfg(target_feature = "sha3")]
        {
            veor3q_u8(low, low_image, high_image)
        }
        #[cfg(not(target_feature = "sha3"))]
        {
            veorq_u8(low, veorq_u8(low_image, high_image))
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::aes::engine::mul_slice;
    use crate::aes::mul_bytes;

    #[test]
    fn every_byte_product_and_unaligned_tail_match_the_scalar_field() {
        // Each row covers all 256 right operands against one left operand.
        for a in 0..=255 {
            let mut values = [0xa5; 259];
            values[1..258].fill(a);
            let factors: [u8; 257] = core::array::from_fn(|i| i as u8);
            mul_slice(&mut values[1..258], &factors);
            // The odd starting offset and the scalar suffix must leave both guards untouched.
            assert_eq!(values[0], 0xa5);
            assert_eq!(values[258], 0xa5);
            for (value, &b) in values[1..258].iter().zip(&factors) {
                assert_eq!(*value, mul_bytes(a, b));
            }
        }
    }
}
