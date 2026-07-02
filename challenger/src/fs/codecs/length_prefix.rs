//! Length prefix used by bounded transcript steps.
//!
//! # Overview
//!
//! A bounded step:
//!
//! - declares a maximum count `max` at pattern-record time,
//! - transmits the actual count on the wire as a big-endian integer.
//!
//! The prefix width is the minimum number of bytes that can hold every value in `0..=max`:
//!
//! ```text
//!     max == 0       → width 0     (only legal length is 0)
//!     max <= 255     → width 1
//!     max <= 65535   → width 2
//!     max <= 2^24-1  → width 3
//!     max <= 2^32-1  → width 4
//!     ...
//! ```
//!
//! Both sides derive the width from `max` independently.
//!
//! No self-describing width field ever travels on the wire.

/// Number of big-endian bytes needed to encode any integer in `0..=max`.
///
/// # Formula
///
/// ```text
///     max == 0 → 0
///     max  > 0 → ceil(log_256(max + 1))
/// ```
#[must_use]
pub const fn bound_byte_width(max: usize) -> usize {
    // Bound 0 only admits one value, so no bytes are needed.
    if max == 0 {
        return 0;
    }
    // Count the significant bits of `max`, then round up to whole bytes.
    let bits = usize::BITS - max.leading_zeros();
    bits.div_ceil(8) as usize
}

/// Encode `len` into `width` big-endian bytes.
///
/// # Returns
///
/// A stack-allocated buffer of 8 bytes.
///
/// Callers consume the first `width` bytes.
///
/// Trailing bytes are zero in this implementation but are not part of the contract.
///
/// # Panics
///
/// - When `width` is greater than 8.
/// - When `len` does not fit in `width` bytes.
#[must_use]
pub fn encode_len_be(len: usize, width: usize) -> [u8; 8] {
    // Defensive upper bound: `usize` is at most 8 bytes on every supported platform.
    assert!(width <= 8, "length-prefix width {width} exceeds 8 bytes");
    // Range check the value against the declared width.
    if width < 8 {
        // Smallest value that does not fit: 1 << (width * 8).
        //
        // For width 0 this is 1, so only `len == 0` is legal.
        let limit = 1usize << (width * 8);
        assert!(
            len < limit,
            "length {len} does not fit in {width} bytes (max {})",
            limit - 1,
        );
    }
    // Convert to a full 8-byte big-endian buffer once.
    //
    // The caller slices off the leading bytes it actually needs.
    let full = (len as u64).to_be_bytes();
    // Left-align the significant bytes inside the returned buffer.
    //
    // ```text
    //     width = 3, len = 0x010203
    //     full   = [00, 00, 00, 00, 00, 01, 02, 03]
    //     output = [01, 02, 03, 00, 00, 00, 00, 00]
    // ```
    let mut out = [0u8; 8];
    out[..width].copy_from_slice(&full[8 - width..]);
    out
}

/// Decode `width` big-endian bytes into a `usize`.
///
/// # Panics
///
/// - When `width` is greater than 8.
/// - When `buf` is shorter than `width`.
#[must_use]
pub fn decode_len_be(buf: &[u8], width: usize) -> usize {
    // Defensive upper bound: `usize` is at most 8 bytes on every supported platform.
    assert!(width <= 8, "length-prefix width {width} exceeds 8 bytes");
    // Caller must supply enough bytes to cover the declared width.
    assert!(
        buf.len() >= width,
        "length-prefix buffer holds {} bytes, need {width}",
        buf.len(),
    );
    // Right-align the prefix into a full 8-byte big-endian buffer.
    //
    // ```text
    //     width = 3, buf[..3] = [01, 02, 03]
    //     padded = [00, 00, 00, 00, 00, 01, 02, 03]
    // ```
    let mut padded = [0u8; 8];
    padded[8 - width..].copy_from_slice(&buf[..width]);
    // Decode as a big-endian unsigned integer.
    u64::from_be_bytes(padded) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn width_table_boundaries() {
        // Invariant: width grows by one byte at every power-of-256 boundary.

        // Bound 0 admits only the empty encoding.
        assert_eq!(bound_byte_width(0), 0);

        // Bounds 1..=255 fit in one byte.
        assert_eq!(bound_byte_width(1), 1);
        assert_eq!(bound_byte_width(255), 1);

        // The 256 boundary crosses into two bytes.
        assert_eq!(bound_byte_width(256), 2);
        assert_eq!(bound_byte_width(65_535), 2);

        // The 2^16 boundary crosses into three bytes.
        assert_eq!(bound_byte_width(65_536), 3);
        assert_eq!(bound_byte_width(16_777_215), 3);

        // The 2^24 boundary crosses into four bytes.
        assert_eq!(bound_byte_width(16_777_216), 4);
        assert_eq!(bound_byte_width(u32::MAX as usize), 4);
    }

    #[test]
    fn big_endian_ordering_is_explicit() {
        // Invariant: the most significant byte sits at the lowest wire index.

        // Mutation: encode the value 0x010203 in three bytes.
        let bytes = encode_len_be(0x010203, 3);

        // The leading byte holds the high-order octet.
        assert_eq!(&bytes[..3], &[0x01, 0x02, 0x03]);
    }

    #[test]
    fn width_zero_only_encodes_zero() {
        // Invariant: a width of zero admits only the empty encoding of 0.

        // Encode the only legal value.
        let bytes = encode_len_be(0, 0);

        // Decoding the empty prefix yields the same value.
        assert_eq!(decode_len_be(&bytes[..0], 0), 0);
    }

    #[test]
    #[should_panic(expected = "does not fit")]
    fn width_zero_rejects_nonzero_len() {
        // Width 0 cannot represent any positive length.
        //
        // Encoding 1 with width 0 must panic.
        let _ = encode_len_be(1, 0);
    }

    #[test]
    #[should_panic(expected = "does not fit")]
    fn width_one_rejects_overflow() {
        // Width 1 covers the range 0..=255.
        //
        // Encoding 256 with width 1 must panic.
        let _ = encode_len_be(256, 1);
    }

    #[test]
    fn round_trip_every_width() {
        // Invariant: decoding undoes encoding for every legal (width, len) pair.
        //
        // Fixture state: one representative low/high pair per supported width.
        //
        // ```text
        //     width 0 → {0}
        //     width 1 → {0, 255}
        //     width 2 → {256, 65535}
        //     width 3 → {65536, 16777215}
        //     width 4 → {16777216, u32::MAX}
        // ```
        for (width, len) in [
            (0usize, 0usize),
            (1, 0),
            (1, 255),
            (2, 256),
            (2, 65_535),
            (3, 65_536),
            (3, 16_777_215),
            (4, 16_777_216),
            (4, u32::MAX as usize),
        ] {
            // Encode then decode and require the original value back.
            let bytes = encode_len_be(len, width);
            let decoded = decode_len_be(&bytes[..width], width);
            assert_eq!(decoded, len, "width={width} len={len}");
        }
    }

    #[test]
    fn width_matches_high_byte_of_max() {
        // Invariant: the derived width is wide enough to encode the bound itself.
        //
        // Fixture state: a sample of representative bounds across each width tier.
        for max in [1usize, 2, 100, 255, 256, 1000, 65535, 65536, 1 << 20] {
            // Derive the width and require the bound to round-trip.
            let w = bound_byte_width(max);
            let bytes = encode_len_be(max, w);
            assert_eq!(decode_len_be(&bytes[..w], w), max);
        }
    }
}
