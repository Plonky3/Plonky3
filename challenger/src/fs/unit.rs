//! Alphabet abstraction for the duplex-sponge transcript layer.
//!
//! # Overview
//!
//! The wire format of a transcript is always bytes.
//!
//! The sponge behind it is not: a Keccak sponge eats bytes, a Poseidon2 sponge eats field elements.
//!
//! A `Unit` names the alphabet of one sponge and says how a raw byte string enters it.
//!
//! That single hook is what lets one driver run over both families of challenger.

use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_field::PrimeField64;

use crate::CanObserve;

/// Alphabet of a sponge, together with the rule for absorbing a raw byte string into it.
pub trait Unit {
    /// Element the sponge consumes.
    type Item;

    /// Absorb `bytes` into a sponge whose alphabet is `Self::Item`.
    ///
    /// Implementations must be injective on `bytes`.
    /// Two distinct byte strings must never produce the same absorbed sequence.
    fn observe_bytes<C: CanObserve<Self::Item>>(challenger: &mut C, bytes: &[u8]);
}

impl Unit for u8 {
    type Item = Self;

    fn observe_bytes<C: CanObserve<Self>>(challenger: &mut C, bytes: &[u8]) {
        // A byte sponge takes the string verbatim, so injectivity is immediate.
        challenger.observe_slice(bytes);
    }
}

/// Alphabet of a sponge that speaks the prime field `F` natively.
///
/// Used as the `U` parameter of a transcript driven by `DuplexChallenger` or
/// `SerializingChallenger32/64`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FieldUnit<F>(PhantomData<F>);

impl<F: PrimeField64> FieldUnit<F> {
    /// Bytes packed into one field element.
    ///
    /// One byte short of the modulus width, so every chunk is strictly below `p`
    /// and the packing never wraps.
    ///
    /// ```text
    ///     BabyBear   (31 bits) -> 3 bytes per element
    ///     Goldilocks (64 bits) -> 7 bytes per element
    /// ```
    pub(crate) const fn bytes_per_element() -> usize {
        // `bits()` is not const, so derive the width from the order instead.
        let bits = u64::BITS - F::ORDER_U64.leading_zeros();
        ((bits as usize) - 1) / 8
    }
}

impl<F: PrimeField64> Unit for FieldUnit<F> {
    type Item = F;

    /// Absorb `bytes` as `[len] ++ [chunk_0, chunk_1, ...]`.
    ///
    /// Each chunk is `bytes_per_element()` bytes read little-endian, the last one zero-padded.
    ///
    /// The leading length element makes the encoding injective:
    /// padding is only ambiguous without it.
    ///
    /// # Panics
    ///
    /// When `bytes.len()` does not fit in `F`.
    fn observe_bytes<C: CanObserve<F>>(challenger: &mut C, bytes: &[u8]) {
        let chunk = Self::bytes_per_element();
        // Length below the modulus keeps the length element itself injective.
        assert!(
            (bytes.len() as u128) < F::ORDER_U64 as u128,
            "byte string of {} bytes does not fit in one field element",
            bytes.len(),
        );
        // One element for the length, then one per chunk.
        let mut packed: Vec<F> = Vec::with_capacity(1 + bytes.len().div_ceil(chunk));
        packed.push(F::from_u64(bytes.len() as u64));
        for window in bytes.chunks(chunk) {
            // Little-endian fold of at most `chunk` bytes: value < 2^(8*chunk) < p.
            let mut acc = 0u64;
            for (i, &b) in window.iter().enumerate() {
                acc |= (b as u64) << (8 * i);
            }
            packed.push(F::from_u64(acc));
        }
        challenger.observe_slice(&packed);
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;
    use p3_goldilocks::Goldilocks;

    use super::*;

    /// Records every absorbed element in order.
    #[derive(Default)]
    struct Recorder<T> {
        seen: Vec<T>,
    }

    impl<T> CanObserve<T> for Recorder<T> {
        fn observe(&mut self, value: T) {
            self.seen.push(value);
        }
    }

    #[test]
    fn byte_unit_absorbs_the_string_verbatim() {
        // A byte sponge sees exactly the bytes it was handed.
        let mut rec = Recorder::<u8>::default();
        <u8 as Unit>::observe_bytes(&mut rec, &[1, 2, 3]);
        assert_eq!(rec.seen, vec![1u8, 2, 3]);
    }

    #[test]
    fn field_unit_chunk_width_is_one_byte_below_the_modulus() {
        // 31-bit prime holds 3 whole bytes; 64-bit prime holds 7.
        assert_eq!(FieldUnit::<BabyBear>::bytes_per_element(), 3);
        assert_eq!(FieldUnit::<Goldilocks>::bytes_per_element(), 7);
    }

    #[test]
    fn field_unit_packs_length_then_little_endian_chunks() {
        // Invariant: the absorbed sequence is [len, chunk_0, chunk_1, ...].
        //
        // Fixture state: four bytes over a 3-byte chunk width.
        let mut rec = Recorder::<BabyBear>::default();
        FieldUnit::<BabyBear>::observe_bytes(&mut rec, &[0x01, 0x02, 0x03, 0x04]);

        // Leading element carries the byte count.
        assert_eq!(rec.seen[0], BabyBear::from_u32(4));
        // First chunk is 0x030201 read little-endian.
        assert_eq!(rec.seen[1], BabyBear::from_u32(0x03_02_01));
        // Trailing chunk holds the single remaining byte.
        assert_eq!(rec.seen[2], BabyBear::from_u32(0x04));
        assert_eq!(rec.seen.len(), 3);
    }

    #[test]
    fn field_unit_length_element_separates_padded_neighbours() {
        // Invariant: zero padding alone would collapse these two strings.
        //
        // The leading length element keeps them apart.
        let mut short = Recorder::<BabyBear>::default();
        let mut padded = Recorder::<BabyBear>::default();
        FieldUnit::<BabyBear>::observe_bytes(&mut short, &[0xaa]);
        FieldUnit::<BabyBear>::observe_bytes(&mut padded, &[0xaa, 0x00]);
        assert_ne!(short.seen, padded.seen);
    }

    #[test]
    fn field_unit_absorbs_the_empty_string_as_a_bare_length() {
        // Boundary: an empty payload still absorbs its length element.
        let mut rec = Recorder::<BabyBear>::default();
        FieldUnit::<BabyBear>::observe_bytes(&mut rec, &[]);
        assert_eq!(rec.seen, vec![BabyBear::ZERO]);
    }
}
