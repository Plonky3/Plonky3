//! Bars layer for Monolith-31 (Mersenne31, p = 2^31 - 1).
//!
//! A 31-bit field element is decomposed into 4 buckets of (8, 8, 8, 7) bits.
//! A chi-like S-box is applied to each bucket independently, then the
//! results are recomposed into a field element. The Kintsugi strategy
//! guarantees that the recomposed value is always in [0, p) because:
//! - All-ones in each bucket (which would give p = 2^31 - 1) is a
//!   fixed point of the S-box, so it can never map to an out-of-range value.
//!
//! For efficiency, the four per-bucket S-boxes are packed into two static
//! lookup tables that process 16 bits at a time:
//! - LOOKUP1: 2^16 entries (128 KiB), maps the two low buckets (bits 0..15)
//! - LOOKUP2: 2^15 entries (64 KiB), maps the two high buckets (bits 16..30)
//!
//! Both tables are computed at compile time and embedded in the binary's
//! read-only data section, so construction is zero-cost at runtime.

use p3_field::PrimeField32;
use p3_field::integers::QuotientMap;
use p3_mersenne_31::Mersenne31;
use sha3::Shake128Reader;

use super::MonolithBars;
use crate::util::get_random_u32;

/// The 8-bit chi-like S-box from Daemen's Table A.1.
///
/// For each bit position i within the 8-bit value:
///
/// ```text
///   y_i = x_i XOR ((NOT x_{i+1}) AND x_{i+2} AND x_{i+3})
/// ```
///
/// followed by a left rotation by 1 bit.
///
/// Both 0x00 and 0xFF are fixed points, which is required by the
/// Kintsugi strategy for buckets that are 1-chunks or 0-chunks of p.
const fn s_box(y: u8) -> u8 {
    // chi: y XOR (NOT rot1(y)) AND rot2(y) AND rot3(y)
    let tmp = y ^ !y.rotate_left(1) & y.rotate_left(2) & y.rotate_left(3);
    // Final left rotation by 1 to reduce the number of fixed points.
    tmp.rotate_left(1)
}

/// The 7-bit chi-like S-box for the most-significant bucket.
///
/// Same structure as the 8-bit version but with only 2 rotations
/// (not 3) and all operations masked to 7 bits. The rotations are
/// done manually since Rust's `rotate_left` operates on full u8.
///
/// Both 0x00 and 0x7F are fixed points.
const fn final_s_box(y: u8) -> u8 {
    // Manual 7-bit left rotations.
    let y_rot_1 = (y >> 6) | (y << 1);
    let y_rot_2 = (y >> 5) | (y << 2);

    // chi with 2 rotations, masked to 7 bits.
    let tmp = (y ^ !y_rot_1 & y_rot_2) & 0x7F;
    // Final left rotation by 1, masked to 7 bits.
    ((tmp >> 6) | (tmp << 1)) & 0x7F
}

/// Packed S-box for bits 0..15 (two 8-bit buckets), computed at compile time.
///
/// Entry i encodes S(hi_byte) || S(lo_byte) where hi_byte = i >> 8
/// and lo_byte = i & 0xFF. Indexed by the low 16 bits of the field element.
///
/// Size: 2^16 entries * 2 bytes = 128 KiB in .rodata.
#[allow(long_running_const_eval)]
static LOOKUP1: [u16; 1 << 16] = {
    let mut table = [0u16; 1 << 16];
    let mut i = 0u32;
    while i < (1 << 16) {
        // Pack S-box of high byte and low byte into a single u16.
        let hi = s_box((i >> 8) as u8) as u16;
        let lo = s_box(i as u8) as u16;
        table[i as usize] = (hi << 8) | lo;
        i += 1;
    }
    table
};

/// Packed S-box for bits 16..30 (one 8-bit bucket + one 7-bit bucket),
/// computed at compile time.
///
/// Entry i encodes S'(hi_7bit) || S(lo_byte) where hi_7bit = i >> 8
/// and lo_byte = i & 0xFF. Only 2^15 entries since the top bucket is 7 bits.
///
/// Size: 2^15 entries * 2 bytes = 64 KiB in .rodata.
static LOOKUP2: [u16; 1 << 15] = {
    let mut table = [0u16; 1 << 15];
    let mut i = 0u32;
    while i < (1 << 15) {
        // Pack 7-bit S-box of high bits and 8-bit S-box of low byte.
        let hi = final_s_box((i >> 8) as u8) as u16;
        let lo = s_box(i as u8) as u16;
        table[i as usize] = (hi << 8) | lo;
        i += 1;
    }
    table
};

/// Bars implementation for Mersenne31 using compile-time lookup tables.
///
/// This is a zero-size struct because both lookup tables are static
/// constants embedded in the binary. No runtime initialization needed.
#[derive(Clone, Copy, Debug, Default)]
pub struct MonolithBarsM31;

impl MonolithBarsM31 {
    /// Expose the 8-bit S-box for testing.
    pub const fn s_box(y: u8) -> u8 {
        s_box(y)
    }

    /// Expose the 7-bit S-box for testing.
    pub const fn final_s_box(y: u8) -> u8 {
        final_s_box(y)
    }

    /// Apply the Bars S-box to a single field element using static lookup tables.
    ///
    /// Splits the 31-bit canonical form into low 16 bits and high 15 bits,
    /// looks up each half in the precomputed tables, and recombines.
    #[inline]
    fn bar(el: Mersenne31) -> Mersenne31 {
        let val = el.as_canonical_u32();

        // Split into low 16 bits (buckets 0-1) and high 15 bits (buckets 2-3).
        let low = LOOKUP1[val as u16 as usize];
        let high = LOOKUP2[(val >> 16) as usize];

        // Recombine: high occupies bits 16..30, low occupies bits 0..15.
        let result = ((high as u32) << 16) | low as u32;
        unsafe {
            // Safety: low < 2^16 and high < 2^15, so result < 2^31.
            Mersenne31::from_canonical_unchecked(result)
        }
    }
}

impl<const WIDTH: usize> MonolithBars<Mersenne31, WIDTH> for MonolithBarsM31 {
    const NUM_BARS: usize = 8;
    /// p = 2^31 - 1 = 0x7FFFFFFF in little-endian.
    const PRIME_BYTES: &[u8] = &0x7FFF_FFFFu32.to_le_bytes();
    const LIMB_BITS: &[u8] = &[8, 8, 8, 7];

    #[inline]
    fn bars(&self, state: &mut [Mersenne31; WIDTH]) {
        // Apply the S-box to the first 8 elements only.
        // Elements 8..WIDTH pass through unchanged.
        state.iter_mut().take(8).for_each(|el| *el = Self::bar(*el));
    }

    fn random_field_element(shake: &mut Shake128Reader) -> Mersenne31 {
        // Rejection sampling: draw u32 values until one is < p.
        let mut val = get_random_u32(shake);
        while val >= Mersenne31::ORDER_U32 {
            val = get_random_u32(shake);
        }
        unsafe {
            // Safety: val < ORDER_U32 = 2^31 - 1 by the rejection loop.
            Mersenne31::from_canonical_unchecked(val)
        }
    }
}
