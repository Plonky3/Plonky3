//! One-byte subfield butterfly on NEON.
//!
//! Multiplying by an element of GF(2^8) is GF(2)-linear on each byte.
//! So the product splits over the two nibbles, each one table lookup.
//!
//! Only the butterfly needs the instruction set.
//! Building the tables is plain byte arithmetic, so every target compiles and tests it.

/// Shortest run, in bytes, worth building the lookup tables for.
///
/// # Why this value
///
/// - Sixteen registers of work, picked as a round number.
/// - No benchmark stands behind it.
/// - It amortises eight GF(2^8) products, about thirty exclusive ors, and two loads.
/// - That cost should be repaid well below sixteen registers, so the number is likely high.
/// - Setting it properly means sweeping 16, 64, 128 and 256 bytes on AArch64 hardware.
/// - The narrow transforms expose the difference, since almost every butterfly is short there.
#[cfg(all(
    target_arch = "aarch64",
    target_feature = "neon",
    target_endian = "little"
))]
pub(crate) const MIN_BYTES: usize = 256;

/// Split multiplication by one element of GF(2^8) into a pair of nibble lookup tables.
///
/// # Overview
///
/// Multiplication by a fixed element is GF(2)-linear.
/// So it is pinned down by the images of the eight basis elements 1, 2, 4, ..., 128.
///
/// A byte is the sum of its low nibble and its high nibble shifted up.
/// Linearity splits the product along that same sum:
///
/// ```text
///     x         =  (x & 0xf)      +  (x >> 4) << 4
///
///     t . x     =  low[x & 0xf]   ^  high[x >> 4]
///
///     low[n]    =  t . n                              for n < 16
///     high[n]   =  t . (n << 4)                       for n < 16
/// ```
///
/// Both indices are nibbles, so neither table is ever read out of range.
///
/// # Arguments
///
/// - The images of the eight basis elements under the multiplier, lowest bit first.
///
/// # Returns
///
/// - The table read by the low nibble of a byte.
/// - The table read by the high nibble of a byte.
#[inline]
fn nibble_tables(columns: &[u8; 8]) -> ([u8; 16], [u8; 16]) {
    // Index zero is the image of zero, which is zero in both tables.
    let mut low = [0u8; 16];
    let mut high = [0u8; 16];

    // Fill each table by doubling its filled prefix, one basis bit at a time.
    //
    // Invariant: after the step for bit b, the first 2^(b + 1) entries are final.
    for b in 0..4 {
        for n in 0..(1usize << b) {
            // Setting bit b in an index adds that bit's image to the entry already there.
            //
            //     low:   n  ->  n | 2^b        gains the image of 2^b
            //     high:  n  ->  n | 2^b        gains the image of 2^(4 + b)
            low[n | (1 << b)] = low[n] ^ columns[b];
            high[n | (1 << b)] = high[n] ^ columns[4 + b];
        }
    }
    (low, high)
}

/// Butterfly over the whole registers of two byte runs, scaled by one element of GF(2^8).
///
/// # Arguments
///
/// - The lower run of bytes.
/// - The upper run of bytes.
/// - The images of the eight basis elements under the scaling element, lowest bit first.
///
/// # Returns
///
/// - The number of bytes covered, which is a whole number of registers.
///
/// Whatever the two runs hold past that is left for the caller to finish.
///
/// # Panics
///
/// Panics if the two runs have different lengths.
#[cfg(all(
    target_arch = "aarch64",
    target_feature = "neon",
    target_endian = "little"
))]
#[inline]
pub(crate) fn byte_butterfly<const INVERSE: bool>(
    lo: &mut [u8],
    hi: &mut [u8],
    columns: &[u8; 8],
) -> usize {
    use core::arch::aarch64::{
        uint8x16_t, vandq_u8, vdupq_n_u8, veorq_u8, vld1q_u8, vqtbl1q_u8, vshrq_n_u8,
    };

    use crate::lanes::butterfly_run;

    // Both tables are sixteen bytes, which is exactly one register.
    let (low, high) = nibble_tables(columns);

    // Hoist the two tables and the nibble mask out of the loop, into three registers.
    //
    // SAFETY: this code compiles only where the crate enables the instruction set.
    // Either table is sixteen readable bytes, and the load needs no alignment.
    let (low, high, nibble) = unsafe {
        (
            vld1q_u8(low.as_ptr()),
            vld1q_u8(high.as_ptr()),
            vdupq_n_u8(0x0f),
        )
    };

    butterfly_run::<uint8x16_t, INVERSE>(lo, hi, |x| {
        // SAFETY: as above, the instruction set is enabled wherever this compiles.
        unsafe {
            // Both lookups index with a nibble, so the out-of-range behaviour is never reached.
            //
            //     low  lookup: x & 0xf   ->  scaled low nibble
            //     high lookup: x >> 4    ->  scaled high nibble
            veorq_u8(
                vqtbl1q_u8(low, vandq_u8(x, nibble)),
                vqtbl1q_u8(high, vshrq_n_u8::<4>(x)),
            )
        }
    })
}

#[cfg(test)]
mod tests {
    use p3_binary_field::{BinaryField8, TowerLevel};

    use super::nibble_tables;

    #[test]
    fn the_nibble_tables_are_the_field_product() {
        // Invariant: summing the two lookups reproduces the field product exactly.
        //
        //     low[x & 0xf] ^ high[x >> 4]  ==  t . x
        //
        // Both sides range over all 256 multipliers and all 256 bytes, so this is exhaustive.
        for t in 0..=u8::MAX {
            let t = BinaryField8::from_repr(t);

            // The images of the basis elements 1, 2, 4, ..., 128 determine the whole map.
            let columns = core::array::from_fn(|b| (t * BinaryField8::from_repr(1 << b)).to_repr());

            let (low, high) = nibble_tables(&columns);

            for x in 0..=u8::MAX {
                // The low nibble reads one table, the high nibble the other.
                let got = low[(x & 0x0f) as usize] ^ high[(x >> 4) as usize];

                // The reference is the field multiplication itself, not a second table.
                let want = (t * BinaryField8::from_repr(x)).to_repr();

                assert_eq!(got, want, "t={t:?} x={x:#04x}");
            }
        }
    }
}
