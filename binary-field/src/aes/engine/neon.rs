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
            let product = vcombine_u8(
                reduce(vreinterpretq_u16_p16(lo)),
                reduce(vreinterpretq_u16_p16(hi)),
            );
            vst1q_u8(a.as_mut_ptr(), product);
        }
    }
    (tail, rest)
}

/// Reduce eight degree-at-most-fourteen products modulo x^8 + x^4 + x^3 + x + 1.
#[inline(always)]
unsafe fn reduce(product: uint16x8_t) -> uint8x8_t {
    // SAFETY: these baseline NEON operations use registers only.
    unsafe {
        // Why: x^8 equals 0x1b in the AES field.
        let fold = |high| {
            veorq_u16(
                veorq_u16(high, vshlq_n_u16::<1>(high)),
                veorq_u16(vshlq_n_u16::<3>(high), vshlq_n_u16::<4>(high)),
            )
        };
        let first = veorq_u16(
            vandq_u16(product, vdupq_n_u16(0xff)),
            fold(vshrq_n_u16::<8>(product)),
        );
        // The first fold has degree at most ten, so its second spill folds entirely below degree eight.
        vmovn_u16(veorq_u16(first, fold(vshrq_n_u16::<8>(first))))
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
