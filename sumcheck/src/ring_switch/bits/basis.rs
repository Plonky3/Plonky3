//! The `F_2`-coordinates of one tower-level element.

use core::marker::PhantomData;

use p3_binary_field::TowerLevel;

/// The widest tower level these coordinates hold.
///
/// A fixed buffer of this width reads coordinates without allocating.
/// An exactly sized array cannot, while the byte count is a trait const.
const MAX_BYTES: usize = 16;

/// The `F_2`-coordinates of one element, in the basis its bytes define.
///
/// # Overview
///
/// A tower level may hold its elements in any `F_2`-basis of the field.
/// This type fixes the one the representation already carries:
///
/// ```text
///     coordinate j  =  bit j of the little-endian byte string
/// ```
///
/// Coordinate zero is therefore the coefficient of one.
/// That is the convention the tensor algebra's matrix indexing assumes.
/// Nothing here depends on which basis that is, only on readers agreeing.
///
/// # Why a value type
///
/// A coordinate is one bit, so three operations matter.
/// Reading one, setting one, and walking the ones that are set.
///
/// Holding the buffer once keeps callers free of byte arithmetic.
/// It also keeps the set-bit walk in one place.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Coefficients<EF> {
    /// The element's little-endian bytes, zero above the level's own width.
    bytes: [u8; MAX_BYTES],
    /// Marker for the level these coordinates belong to.
    _ef: PhantomData<EF>,
}

impl<EF: TowerLevel> Coefficients<EF> {
    /// Coordinates one element has, the level's dimension over `F_2`.
    pub const DIMENSION: usize = 1 << EF::LOG_BITS;

    /// Number of Boolean variables one element's coordinates index.
    pub const LOG_DIMENSION: usize = EF::LOG_BITS;

    /// All coordinates zero.
    #[must_use]
    pub const fn zero() -> Self {
        Self {
            bytes: [0; MAX_BYTES],
            _ef: PhantomData,
        }
    }

    /// The coordinates of one element.
    ///
    /// # Panics
    ///
    /// Panics if the level is wider than the buffer.
    #[must_use]
    pub fn of(value: EF) -> Self {
        assert!(
            EF::NUM_BYTES <= MAX_BYTES,
            "the tower level is wider than these coordinates hold"
        );
        let mut out = Self::zero();
        for (slot, byte) in out.bytes.iter_mut().zip(value.into_bytes()) {
            *slot = byte;
        }
        out
    }

    /// Whether one coordinate is set.
    ///
    /// Bytes above the level's width stay zero.
    ///
    /// An index below the dimension therefore never reads padding.
    #[must_use]
    pub const fn get(&self, index: usize) -> bool {
        (self.bytes[index / 8] >> (index % 8)) & 1 == 1
    }

    /// Set one coordinate.
    pub const fn set(&mut self, index: usize) {
        self.bytes[index / 8] |= 1 << (index % 8);
    }

    /// The indices of the set coordinates, lowest first.
    ///
    /// This replaces the multiplication a general alphabet needs.
    /// Every accumulation over a bit alphabet is written around it.
    pub fn iter_set(&self) -> impl Iterator<Item = usize> + '_ {
        self.bytes[..EF::NUM_BYTES]
            .iter()
            .enumerate()
            .flat_map(|(position, &byte)| SetBits {
                rest: byte,
                base: position * 8,
            })
    }

    /// Every coordinate in order, as a bit.
    pub fn iter(&self) -> impl Iterator<Item = bool> + '_ {
        (0..Self::DIMENSION).map(|index| self.get(index))
    }

    /// The element these coordinates describe.
    ///
    /// The inverse of reading one, so a round trip here is the identity.
    #[must_use]
    pub fn element(&self) -> EF {
        EF::from_le_byte_iter(self.bytes.iter().copied())
    }
}

/// The set bit positions of one byte, lowest first.
struct SetBits {
    /// Bits not yet yielded.
    rest: u8,
    /// Index of this byte's lowest bit within the whole element.
    base: usize,
}

impl Iterator for SetBits {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        (self.rest != 0).then(|| {
            // The lowest set bit, cleared on the way out.
            let bit = self.rest.trailing_zeros() as usize;
            self.rest &= self.rest - 1;
            self.base + bit
        })
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use p3_binary_field::{BinaryField16, BinaryField128};
    use p3_field::PrimeCharacteristicRing;
    use proptest::prelude::*;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;

    #[test]
    fn a_level_has_one_coordinate_per_bit() {
        // Fixture state: 128 bits give 128 coordinates, 16 bits give 16.
        assert_eq!(Coefficients::<BinaryField128>::DIMENSION, 128);
        assert_eq!(Coefficients::<BinaryField128>::LOG_DIMENSION, 7);
        assert_eq!(Coefficients::<BinaryField16>::DIMENSION, 16);
        assert_eq!(Coefficients::<BinaryField16>::LOG_DIMENSION, 4);
    }

    #[test]
    fn reading_coordinates_and_rebuilding_is_the_identity() {
        // Invariant: a coordinate index means the same in both directions.
        // That is what lets a witness be packed by reinterpretation.
        let mut rng = SmallRng::seed_from_u64(0xB17E);
        for _ in 0..64 {
            let value = rng.random::<BinaryField128>();
            assert_eq!(Coefficients::of(value).element(), value);
        }
    }

    #[test]
    fn the_first_coordinate_is_the_coefficient_of_one() {
        // The tensor algebra's matrix convention names that coordinate zero.
        //
        //     ONE  ->  coordinate 0 set, every other clear
        for coefficients in [
            Coefficients::of(BinaryField128::ONE)
                .iter()
                .collect::<Vec<_>>(),
            Coefficients::of(BinaryField16::ONE)
                .iter()
                .collect::<Vec<_>>(),
        ] {
            assert!(coefficients[0]);
            assert!(coefficients[1..].iter().all(|&bit| !bit));
        }
    }

    #[test]
    fn the_coordinates_are_a_basis_decomposition() {
        // Invariant: an element is the sum of the vectors it selects.
        //
        //     x = sum over set j of beta_j,  beta_j = only bit j set
        //
        // That is what makes these positions a basis and not an encoding.
        let mut rng = SmallRng::seed_from_u64(0xBA515);
        for _ in 0..64 {
            let value = rng.random::<BinaryField128>();
            let mut rebuilt = BinaryField128::ZERO;
            for index in Coefficients::of(value).iter_set() {
                let mut basis = Coefficients::<BinaryField128>::zero();
                basis.set(index);
                rebuilt += basis.element();
            }
            assert_eq!(rebuilt, value);
        }
    }

    #[test]
    fn a_narrow_level_reads_none_of_its_padding() {
        // The buffer is wider than most levels, and padding is no coordinate.
        let mut rng = SmallRng::seed_from_u64(0x9A44);
        for _ in 0..64 {
            let coefficients = Coefficients::of(rng.random::<BinaryField16>());
            assert!(coefficients.iter_set().all(|index| index < 16));
        }
    }

    proptest! {
        #[test]
        fn the_set_walk_finds_exactly_the_set_coordinates(raw: u128) {
            // The set-bit walk is the hot path, decomposition the reference.
            let coefficients = Coefficients::of(BinaryField128::from_repr(raw));

            let walked = coefficients.iter_set().collect::<Vec<_>>();
            let expected = coefficients
                .iter()
                .enumerate()
                .filter_map(|(index, bit)| bit.then_some(index))
                .collect::<Vec<_>>();

            prop_assert_eq!(walked, expected);
        }

        #[test]
        fn setting_a_coordinate_is_what_reading_it_back_reports(index in 0usize..16) {
            // The two accessors are each other's inverse on one coordinate.
            let mut coefficients = Coefficients::<BinaryField16>::zero();
            prop_assert!(!coefficients.get(index));

            coefficients.set(index);
            prop_assert!(coefficients.get(index));
            prop_assert_eq!(coefficients.iter_set().collect::<Vec<_>>(), alloc::vec![index]);
        }
    }
}
