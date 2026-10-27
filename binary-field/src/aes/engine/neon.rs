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
#[inline(always)]
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

/// Add terms with a run-time shift to complete sixteen-lane blocks.
#[inline(always)]
pub(super) fn add_power_dynamic_prefix(dst: &mut [u16], src: &[u8], power: u8) -> usize {
    let covered = src.len() / 16 * 16;
    // SAFETY: NEON is an AArch64 baseline feature.
    let shift = unsafe { vdupq_n_s16(i16::from(power)) };
    for start in (0..covered).step_by(16) {
        // SAFETY: the slices cover sixteen readable bytes and sixteen writable u16 lanes.
        unsafe {
            let values = vld1q_u8(src.as_ptr().add(start));
            let low = vshlq_u16(vmovl_u8(vget_low_u8(values)), shift);
            let high = vshlq_u16(vmovl_high_u8(values), shift);
            let target = dst.as_mut_ptr().add(start);
            vst1q_u16(target, veorq_u16(vld1q_u16(target), low));
            vst1q_u16(target.add(8), veorq_u16(vld1q_u16(target.add(8)), high));
        }
    }
    covered
}

/// Reduce complete sixteen-lane blocks with the shared nibble-table fold.
#[inline(always)]
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

/// Sixteen bytes swept with NEON, for the `F_2`-linear map and the field product.
///
/// AArch64 has no byte-wise affine instruction, so the map is driven by two
/// register-resident nibble tables instead. A map is linear, so a byte's image
/// is the image of its low nibble added to the image of its high nibble, and
/// `vqtbl1q_u8` reads sixteen of each per instruction. That is the same device
/// `reduce` above already uses for the modulus fold.
pub(super) mod lanes {
    use core::arch::aarch64::*;

    use super::reduce;
    use crate::aes::engine::{ByteLanes, ByteMatrix};

    /// The two nibble tables a byte matrix becomes: images of the low and high nibbles.
    #[derive(Clone, Copy)]
    pub(crate) struct NibbleTables {
        low: uint8x16_t,
        high: uint8x16_t,
    }

    /// Sixteen bytes in one NEON register.
    pub(crate) type Wide = uint8x16_t;

    /// The same register, since AArch64 has one width here.
    pub(crate) type Narrow = uint8x16_t;

    // SAFETY: `load` and `store` touch exactly sixteen contiguous bytes, the register width.
    unsafe impl ByteLanes for uint8x16_t {
        const WIDTH: usize = 16;

        type Matrix = NibbleTables;

        #[inline(always)]
        fn matrix(quadword: u64) -> Self::Matrix {
            let images = ByteMatrix::from_quadword(quadword).images();

            // A nibble's image is the sum of the images of its set bits.
            let mut low = [0u8; 16];
            let mut high = [0u8; 16];
            for n in 0..16usize {
                let mut lo = 0u8;
                let mut hi = 0u8;
                for b in 0..4usize {
                    if (n >> b) & 1 == 1 {
                        lo ^= images[b];
                        hi ^= images[b + 4];
                    }
                }
                low[n] = lo;
                high[n] = hi;
            }

            // SAFETY: both arrays are exactly one register of bytes.
            unsafe {
                NibbleTables {
                    low: vld1q_u8(low.as_ptr()),
                    high: vld1q_u8(high.as_ptr()),
                }
            }
        }

        #[inline(always)]
        unsafe fn load(from: *const u8) -> Self {
            // SAFETY: the readability of the address is the caller's obligation.
            unsafe { vld1q_u8(from) }
        }

        #[inline(always)]
        unsafe fn store(to: *mut u8, value: Self) {
            // SAFETY: the writability of the address is the caller's obligation.
            unsafe { vst1q_u8(to, value) }
        }

        #[inline(always)]
        fn mul(self, other: Self) -> Self {
            // SAFETY: NEON is an AArch64 baseline feature; `vmull_p8` needs no extension.
            unsafe {
                let low = vmull_p8(
                    vreinterpret_p8_u8(vget_low_u8(self)),
                    vreinterpret_p8_u8(vget_low_u8(other)),
                );
                let high = vmull_high_p8(vreinterpretq_p8_u8(self), vreinterpretq_p8_u8(other));
                reduce(vreinterpretq_u16_p16(low), vreinterpretq_u16_p16(high))
            }
        }

        #[inline(always)]
        fn affine(self, matrix: Self::Matrix) -> Self {
            // SAFETY: NEON is an AArch64 baseline feature.
            unsafe {
                // Table reads are by index, so only the nibble may reach the lookup.
                let low = vqtbl1q_u8(matrix.low, vandq_u8(self, vdupq_n_u8(0x0f)));
                let high = vqtbl1q_u8(matrix.high, vshrq_n_u8::<4>(self));
                veorq_u8(low, high)
            }
        }

        #[inline(always)]
        fn inverse_then_affine(self, matrix: Self::Matrix) -> Self {
            // `x^254` is the inverse in this field, and it sends zero to zero,
            // which is the behaviour this method owes its caller.
            //
            // `254 = 2 + 4 + ... + 128`, so the product of the seven repeated
            // squarings of `x` is `x^254`, with no branch and no table.
            let mut square = self.mul(self);
            let mut inverse = square;
            for _ in 0..6 {
                square = square.mul(square);
                inverse = inverse.mul(square);
            }
            inverse.affine(matrix)
        }
    }
}
