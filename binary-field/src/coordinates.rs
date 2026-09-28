//! The `F_2`-coordinates of a binary field, read off its little-endian bytes.

use p3_field::{Field, RawDataSerializable};

use crate::poly64::Poly64;
use crate::poly192::Poly192;
use crate::tower::TowerLevel;

/// A binary field whose `F_2`-coordinates are the bits of its little-endian bytes.
///
/// Coordinate `j` is bit `j % 8` of byte `j / 8`.
///
/// Coordinate zero is the coefficient of one in every implementation here.
///
/// The dimension need not be a power of two:
///
/// ```text
///     tower levels   2^k coordinates
///     Poly192        192 coordinates, three 64-bit limbs
/// ```
pub trait BitCoordinates: Field + RawDataSerializable {
    /// The field's dimension over `F_2`.
    const DIMENSION: usize;

    /// Build an element from the next [`RawDataSerializable::NUM_BYTES`] bytes of a stream.
    ///
    /// Bits above [`Self::DIMENSION`] are discarded.
    ///
    /// # Panics
    ///
    /// Panics if the stream ends before a whole element has been read.
    fn from_coordinate_bytes(bytes: impl Iterator<Item = u8>) -> Self;
}

impl<F: TowerLevel> BitCoordinates for F {
    const DIMENSION: usize = 1 << F::LOG_BITS;

    #[inline]
    fn from_coordinate_bytes(bytes: impl Iterator<Item = u8>) -> Self {
        F::from_le_byte_iter(bytes)
    }
}

impl BitCoordinates for Poly192 {
    const DIMENSION: usize = Self::BITS;

    /// The three limbs follow one another, lowest first, as the byte encoding writes them.
    #[inline]
    fn from_coordinate_bytes(mut bytes: impl Iterator<Item = u8>) -> Self {
        Self::new(core::array::from_fn(|_| {
            Poly64::from_le_byte_iter(&mut bytes)
        }))
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use p3_field::PrimeCharacteristicRing;
    use proptest::prelude::*;

    use super::*;
    use crate::{BinaryField128, Ghash128};

    /// Reads an element back from the bytes it writes.
    fn round_trip<F: BitCoordinates>(value: F) -> F {
        F::from_coordinate_bytes(value.into_bytes().into_iter())
    }

    #[test]
    fn dimensions_count_the_coordinates() {
        // A tower level has a power-of-two dimension.
        assert_eq!(<Poly64 as BitCoordinates>::DIMENSION, 64);
        assert_eq!(<BinaryField128 as BitCoordinates>::DIMENSION, 128);
        assert_eq!(<Ghash128 as BitCoordinates>::DIMENSION, 128);

        // The cubic extension has three limbs of sixty-four.
        assert_eq!(<Poly192 as BitCoordinates>::DIMENSION, 192);
        assert_eq!(8 * Poly192::NUM_BYTES, 192);
    }

    #[test]
    fn coordinate_zero_is_the_coefficient_of_one() {
        // One sets the lowest bit of the lowest byte, and nothing else.
        let bytes = Poly192::ONE.into_bytes().into_iter().collect::<Vec<_>>();
        assert_eq!(bytes[0], 1);
        assert!(bytes[1..].iter().all(|&byte| byte == 0));
    }

    #[test]
    fn the_subfield_fills_the_low_coordinates() {
        // Poly64 embeds at the constant limb, so its coordinates are the low sixty-four.
        let value = Poly64::new(0x0123_4567_89AB_CDEF);
        let wide = Poly192::from(value)
            .into_bytes()
            .into_iter()
            .collect::<Vec<_>>();
        assert_eq!(
            wide[..8],
            value.into_bytes().into_iter().collect::<Vec<_>>()[..]
        );
        assert!(wide[8..].iter().all(|&byte| byte == 0));
    }

    proptest! {
        #[test]
        fn bytes_round_trip(limbs in prop::array::uniform3(any::<u64>())) {
            let value = Poly192::new(limbs.map(Poly64::new));
            prop_assert_eq!(round_trip(value), value);
        }

        #[test]
        fn tower_bytes_round_trip(raw in any::<u64>()) {
            let value = Poly64::new(raw);
            prop_assert_eq!(round_trip(value), value);
        }
    }
}
