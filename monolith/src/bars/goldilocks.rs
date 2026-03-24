//! Bars layer for Monolith-64 (Goldilocks, p = 2^64 - 2^32 + 1).
//!
//! A 64-bit field element is treated as independent lanes of equal width.
//! The chi-like S-box is applied to all lanes simultaneously using SWAR
//! (SIMD Within A Register) bitwise operations on the u64 value.
//!
//! The lane width is controlled by the const generic LOOKUP_BITS:
//! - LOOKUP_BITS=8: 8 lanes of 8 bits each (standard, matches paper Section 4.3)
//! - LOOKUP_BITS=16: 4 lanes of 16 bits each (alternative instantiation)
//!
//! Both produce valid Monolith-64 instantiations but with different S-box
//! mappings (rotations wrap at different lane boundaries), so they generate
//! different round constants and produce different outputs.

use p3_field::PrimeField64;
use p3_field::integers::QuotientMap;
use p3_goldilocks::Goldilocks;
use sha3::Shake128Reader;

use super::MonolithBars;
use crate::util::get_random_u64;

/// Bars implementation for Goldilocks using bitwise SWAR operations.
///
/// The const generic LOOKUP_BITS controls the lane width:
/// - 8: eight 8-bit lanes per u64 (standard)
/// - 16: four 16-bit lanes per u64 (alternative)
///
/// This is a zero-size struct because the S-box is computed purely
/// from bitwise operations, with no precomputed tables.
#[derive(Clone, Copy, Debug, Default)]
pub struct MonolithBarsGoldilocks<const LOOKUP_BITS: usize>;

impl<const LOOKUP_BITS: usize> MonolithBarsGoldilocks<LOOKUP_BITS> {
    /// Apply the chi-like S-box to all lanes of a u64 in parallel.
    ///
    /// For each lane of LOOKUP_BITS width, the S-box computes:
    ///
    /// ```text
    ///   y = x XOR ((NOT rot1(x)) AND rot2(x) AND rot3(x))
    /// ```
    ///
    /// followed by a left rotation by 1 bit within each lane.
    ///
    /// The rotations wrap within each lane boundary (not across lanes).
    /// All-zero and all-ones lanes are fixed points.
    #[inline]
    pub const fn bar(val: u64) -> u64 {
        match LOOKUP_BITS {
            8 => {
                // 8 lanes of 8 bits each.
                // Rotate left by 1 within each 8-bit lane.
                let rot1 =
                    ((!val & 0x8080_8080_8080_8080) >> 7) | ((!val & 0x7F7F_7F7F_7F7F_7F7F) << 1);
                // Rotate left by 2 within each 8-bit lane.
                let rot2 =
                    ((val & 0xC0C0_C0C0_C0C0_C0C0) >> 6) | ((val & 0x3F3F_3F3F_3F3F_3F3F) << 2);
                // Rotate left by 3 within each 8-bit lane.
                let rot3 =
                    ((val & 0xE0E0_E0E0_E0E0_E0E0) >> 5) | ((val & 0x1F1F_1F1F_1F1F_1F1F) << 3);

                // Chi formula: x XOR ((NOT rot1) AND rot2 AND rot3).
                let tmp = val ^ (rot1 & rot2 & rot3);

                // Final rotation left by 1 within each 8-bit lane.
                ((tmp & 0x8080_8080_8080_8080) >> 7) | ((tmp & 0x7F7F_7F7F_7F7F_7F7F) << 1)
            }
            16 => {
                // 4 lanes of 16 bits each.
                // Rotate left by 1 within each 16-bit lane.
                let rot1 =
                    ((!val & 0x8000_8000_8000_8000) >> 15) | ((!val & 0x7FFF_7FFF_7FFF_7FFF) << 1);
                // Rotate left by 2 within each 16-bit lane.
                let rot2 =
                    ((val & 0xC000_C000_C000_C000) >> 14) | ((val & 0x3FFF_3FFF_3FFF_3FFF) << 2);
                // Rotate left by 3 within each 16-bit lane.
                let rot3 =
                    ((val & 0xE000_E000_E000_E000) >> 13) | ((val & 0x1FFF_1FFF_1FFF_1FFF) << 3);

                // Chi formula: x XOR ((NOT rot1) AND rot2 AND rot3).
                let tmp = val ^ (rot1 & rot2 & rot3);

                // Final rotation left by 1 within each 16-bit lane.
                ((tmp & 0x8000_8000_8000_8000) >> 15) | ((tmp & 0x7FFF_7FFF_7FFF_7FFF) << 1)
            }
            _ => panic!("Unsupported LOOKUP_BITS: must be 8 or 16"),
        }
    }
}

/// Compute the LIMB_BITS array for a given lookup size.
///
/// - LOOKUP_BITS=8: 8 limbs of 8 bits -> `[8, 8, 8, 8, 8, 8, 8, 8]`
/// - LOOKUP_BITS=16: 4 limbs of 16 bits -> `[16, 16, 16, 16]`
const fn limb_bits<const LOOKUP_BITS: usize>() -> &'static [u8] {
    match LOOKUP_BITS {
        8 => &[8, 8, 8, 8, 8, 8, 8, 8],
        16 => &[16, 16, 16, 16],
        _ => panic!("Unsupported LOOKUP_BITS"),
    }
}

impl<const WIDTH: usize, const LOOKUP_BITS: usize> MonolithBars<Goldilocks, WIDTH>
    for MonolithBarsGoldilocks<LOOKUP_BITS>
{
    const NUM_BARS: usize = 4;

    /// p = 2^64 - 2^32 + 1 = 0xFFFFFFFF00000001 in little-endian.
    const PRIME_BYTES: &[u8] = &0xFFFF_FFFF_0000_0001u64.to_le_bytes();

    const LIMB_BITS: &[u8] = limb_bits::<LOOKUP_BITS>();

    #[inline]
    fn bars(&self, state: &mut [Goldilocks; WIDTH]) {
        // Apply the SWAR S-box to the first 4 elements only.
        // Elements 4..WIDTH pass through unchanged.
        for el in state.iter_mut().take(4) {
            let val = el.as_canonical_u64();
            let result = Self::bar(val);
            // Safety: Goldilocks accepts any u64 as a (possibly non-canonical) representative.
            *el = unsafe { Goldilocks::from_canonical_unchecked(result) };
        }
    }

    fn random_field_element(shake: &mut Shake128Reader) -> Goldilocks {
        // Rejection sampling: draw u64 values until one is < p.
        let mut val = get_random_u64(shake);
        while val >= Goldilocks::ORDER_U64 {
            val = get_random_u64(shake);
        }
        unsafe {
            // Safety: val < ORDER_U64 by the rejection loop.
            Goldilocks::from_canonical_unchecked(val)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_swar8_fixed_points() {
        // All-zero bytes: each byte is 0x00, a fixed point.
        assert_eq!(MonolithBarsGoldilocks::<8>::bar(0), 0);
        // All-ones bytes: each byte is 0xFF, a fixed point.
        assert_eq!(
            MonolithBarsGoldilocks::<8>::bar(0xFFFF_FFFF_FFFF_FFFF),
            0xFFFF_FFFF_FFFF_FFFF
        );
    }

    #[test]
    fn test_swar16_fixed_points() {
        // All-zero: each 16-bit lane is 0x0000, a fixed point.
        assert_eq!(MonolithBarsGoldilocks::<16>::bar(0), 0);
        // All-ones: each 16-bit lane is 0xFFFF, a fixed point.
        assert_eq!(
            MonolithBarsGoldilocks::<16>::bar(0xFFFF_FFFF_FFFF_FFFF),
            0xFFFF_FFFF_FFFF_FFFF
        );
    }

    #[test]
    fn test_swar8_matches_scalar() {
        // Verify the 8-bit SWAR matches a naive per-byte implementation.
        for val in [
            0x0102_0304_0506_0708u64,
            0xDEAD_BEEF_CAFE_BABEu64,
            0x8000_0000_0000_0001u64,
        ] {
            let swar_result = MonolithBarsGoldilocks::<8>::bar(val);

            // Compute the same thing byte-by-byte using u8 rotations.
            let bytes = val.to_le_bytes();
            let mut out_bytes = [0u8; 8];
            for (i, &b) in bytes.iter().enumerate() {
                let rot1 = b.rotate_left(1);
                let rot2 = b.rotate_left(2);
                let rot3 = b.rotate_left(3);
                let tmp = b ^ (!rot1 & rot2 & rot3);
                out_bytes[i] = tmp.rotate_left(1);
            }
            let scalar_result = u64::from_le_bytes(out_bytes);

            assert_eq!(swar_result, scalar_result, "mismatch for input {val:#018x}");
        }
    }

    #[test]
    fn test_swar16_matches_scalar() {
        // Verify the 16-bit SWAR matches a naive per-u16 implementation.
        for val in [
            0x0102_0304_0506_0708u64,
            0xDEAD_BEEF_CAFE_BABEu64,
            0x8000_0000_0000_0001u64,
        ] {
            let swar_result = MonolithBarsGoldilocks::<16>::bar(val);

            // Compute the same thing 16-bit-word-by-word.
            let words = [
                (val & 0xFFFF) as u16,
                ((val >> 16) & 0xFFFF) as u16,
                ((val >> 32) & 0xFFFF) as u16,
                ((val >> 48) & 0xFFFF) as u16,
            ];
            let mut out_words = [0u16; 4];
            for (i, &w) in words.iter().enumerate() {
                let rot1 = w.rotate_left(1);
                let rot2 = w.rotate_left(2);
                let rot3 = w.rotate_left(3);
                let tmp = w ^ (!rot1 & rot2 & rot3);
                out_words[i] = tmp.rotate_left(1);
            }
            let scalar_result = (out_words[0] as u64)
                | ((out_words[1] as u64) << 16)
                | ((out_words[2] as u64) << 32)
                | ((out_words[3] as u64) << 48);

            assert_eq!(swar_result, scalar_result, "mismatch for input {val:#018x}");
        }
    }

    #[test]
    fn test_swar8_and_swar16_differ() {
        // The two lane widths must produce different outputs for non-trivial inputs,
        // since rotations wrap at different boundaries.
        let val = 0xDEAD_BEEF_CAFE_BABEu64;
        let result8 = MonolithBarsGoldilocks::<8>::bar(val);
        let result16 = MonolithBarsGoldilocks::<16>::bar(val);
        assert_ne!(result8, result16);
    }
}
