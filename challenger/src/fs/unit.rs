//! Alphabet abstraction for the duplex-sponge transcript layer.

use core::fmt::Debug;

/// A single unit of a sponge's input/output stream.
pub trait Unit: Copy + Default + Debug + Send + Sync + 'static {
    /// Number of bytes occupied by one unit in canonical little-endian form.
    const BYTE_LEN: usize;

    /// Write one unit into the first byte-length-many bytes of `dst`.
    ///
    /// # Panics
    ///
    /// Panics if `dst` is shorter than the byte length.
    fn write_le(self, dst: &mut [u8]);

    /// Read one unit from the first byte-length-many bytes of `src`.
    ///
    /// # Panics
    ///
    /// Panics if `src` is shorter than the byte length.
    fn read_le(src: &[u8]) -> Self;
}

impl Unit for u8 {
    // One unit is one byte by definition.
    const BYTE_LEN: usize = 1;

    fn write_le(self, dst: &mut [u8]) {
        // Direct copy: no endian conversion is meaningful for a single byte.
        dst[0] = self;
    }

    fn read_le(src: &[u8]) -> Self {
        // Direct read: a single byte is its own little-endian encoding.
        src[0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn u8_round_trips_through_unit_methods() {
        // Invariant: write_le followed by read_le recovers the original byte.
        //
        // Fixture state: a one-byte buffer holding the test value.
        // Mutation: write the value, then read it back.
        let value: u8 = 0xab;
        let mut buf = [0u8; 1];
        value.write_le(&mut buf);
        let read = u8::read_le(&buf);
        assert_eq!(read, value);
    }

    #[test]
    fn u8_byte_len_is_one() {
        assert_eq!(<u8 as Unit>::BYTE_LEN, 1);
    }

    #[test]
    #[should_panic]
    fn u8_write_le_panics_on_zero_length_destination() {
        let mut empty: [u8; 0] = [];
        0u8.write_le(&mut empty);
    }
}
