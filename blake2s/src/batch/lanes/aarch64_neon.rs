//! Four lanes per 128-bit NEON register.

use core::arch::aarch64::*;

use super::Word;
use crate::DIGEST_BYTES;
use crate::batch::compress::{BLOCK_BYTES, BLOCK_WORDS};

/// One state or message word for four lanes.
pub(super) type Vector = uint32x4_t;

/// Lanes in one register.
pub(super) const WIDTH: usize = 4;

/// Byte indices that rotate every 32-bit word right by 8 bits.
const ROTR_8: [u8; 16] = [1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11, 8, 13, 14, 15, 12];

// SAFETY (every block below): this module only compiles when the target enables NEON.
impl Word for uint32x4_t {
    #[inline(always)]
    fn splat(value: u32) -> Self {
        unsafe { vdupq_n_u32(value) }
    }

    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        unsafe { vaddq_u32(self, rhs) }
    }

    #[inline(always)]
    fn and(self, rhs: Self) -> Self {
        unsafe { vandq_u32(self, rhs) }
    }

    #[inline(always)]
    fn xor(self, rhs: Self) -> Self {
        unsafe { veorq_u32(self, rhs) }
    }

    #[inline(always)]
    fn rotr_16(self) -> Self {
        // Swapping the two 16-bit halves of every word is the rotation.
        unsafe { vreinterpretq_u32_u16(vrev32q_u16(vreinterpretq_u16_u32(self))) }
    }

    #[inline(always)]
    fn rotr_12(self) -> Self {
        // Shift left, then insert the right-shifted bits: two instructions, no OR.
        unsafe { vsriq_n_u32::<12>(vshlq_n_u32::<20>(self), self) }
    }

    #[inline(always)]
    fn rotr_8(self) -> Self {
        unsafe {
            let table = vld1q_u8(ROTR_8.as_ptr());
            vreinterpretq_u32_u8(vqtbl1q_u8(vreinterpretq_u8_u32(self), table))
        }
    }

    #[inline(always)]
    fn rotr_7(self) -> Self {
        unsafe { vsriq_n_u32::<7>(vshlq_n_u32::<25>(self), self) }
    }
}

/// Transpose a 4 x 4 matrix of words held one row per register.
#[inline(always)]
fn transpose([a, b, c, d]: [uint32x4_t; 4]) -> [uint32x4_t; 4] {
    // SAFETY: this module only compiles when the target enables NEON.
    unsafe {
        // Interleave 32-bit words of row pairs, then 64-bit pairs of those.
        let ab_lo = vreinterpretq_u64_u32(vzip1q_u32(a, b));
        let ab_hi = vreinterpretq_u64_u32(vzip2q_u32(a, b));
        let cd_lo = vreinterpretq_u64_u32(vzip1q_u32(c, d));
        let cd_hi = vreinterpretq_u64_u32(vzip2q_u32(c, d));
        [
            vreinterpretq_u32_u64(vzip1q_u64(ab_lo, cd_lo)),
            vreinterpretq_u32_u64(vzip2q_u64(ab_lo, cd_lo)),
            vreinterpretq_u32_u64(vzip1q_u64(ab_hi, cd_hi)),
            vreinterpretq_u32_u64(vzip2q_u64(ab_hi, cd_hi)),
        ]
    }
}

/// Load one block from each of four lanes as sixteen message words.
#[inline(always)]
pub(super) fn load_block(rows: &[&[u8; BLOCK_BYTES]; WIDTH]) -> [uint32x4_t; BLOCK_WORDS] {
    // Each block is four registers, and quarter k holds words 4k to 4k + 3.
    //
    // SAFETY: each quarter is 16 readable bytes, and a byte load has no alignment requirement.
    // The target is little-endian, so the bytes of each word land in the order BLAKE2s reads.
    let quarters: [[uint32x4_t; 4]; 4] = core::array::from_fn(|k| {
        transpose(rows.map(|row| unsafe { vreinterpretq_u32_u8(vld1q_u8(row[16 * k..].as_ptr())) }))
    });
    core::array::from_fn(|w| quarters[w / 4][w % 4])
}

/// Write the digests of four lanes.
#[inline(always)]
pub(super) fn store_digests(state: &[uint32x4_t; 8], out: &mut [[u8; DIGEST_BYTES]; WIDTH]) {
    // Words 0 to 3, then words 4 to 7, of every lane.
    let low = transpose([state[0], state[1], state[2], state[3]]);
    let high = transpose([state[4], state[5], state[6], state[7]]);
    for ((digest, low), high) in out.iter_mut().zip(low).zip(high) {
        // SAFETY: each half of a digest is 16 writable bytes, and a byte store needs no alignment.
        unsafe {
            vst1q_u8(digest.as_mut_ptr(), vreinterpretq_u8_u32(low));
            vst1q_u8(digest[16..].as_mut_ptr(), vreinterpretq_u8_u32(high));
        }
    }
}
