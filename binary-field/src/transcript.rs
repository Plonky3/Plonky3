//! Canonical typed-transcript encodings for the byte-aligned levels of the tower.
//!
//! # Overview
//!
//! A typed transcript needs three things of a field.
//!
//! - A compiler-independent name for an algebra over it.
//! - An injective way to absorb a raw byte string into a sponge that speaks it.
//! - A fixed-width canonical wire encoding of one element.
//!
//! # Seeding
//!
//! A seed is a byte string, and the sponge here eats tower elements.
//!
//! The string is therefore packed one element per `wire_len` bytes, least significant byte first.
//! A fixed-width length field goes in ahead of it, so no packed string is a prefix of another.
//!
//! The length is written as eight little-endian bytes, at every level.
//!
//! A level narrower than eight bytes spreads that field over several elements.
//! A wider one holds it in a single zero-padded element.
//!
//! # Cost at the narrow levels
//!
//! One element carries one byte at the narrowest level, so a seed costs one absorption per byte.
//!
//! A typed transcript seeds on a fixed-size string, 106 bytes for the quadratic sumcheck:
//!
//! ```text
//!     level     bytes per element     absorptions
//!     8-bit     1                     114
//!     16-bit    2                     57
//!     128-bit   16                    8
//! ```
//!
//! A protocol that seeds a sub-transcript per round pays that once per round.
//!
//! Only the ring-switch tests and benchmarks instantiate a sumcheck at the byte level today.
//! Every shipped prover and verifier seeds at the 128-bit level.

use alloc::vec::Vec;

use p3_challenger::CanObserve;
use p3_challenger::fs::{TranscriptError, TranscriptField, TypeTag};

use crate::{
    BinaryField8, BinaryField16, BinaryField32, BinaryField64, BinaryField128, TowerLevel,
};

/// Implement the typed-transcript hooks for one byte-aligned level of the tower.
///
/// The level is named by its Rust type, its backing integer, and its bit width.
///
/// Every level shares one seeding rule, so every level is generated from here.
macro_rules! impl_transcript_field {
    ($name:ty, $repr:ty, $bits:literal) => {
        impl TranscriptField for $name {
            fn algebra_tag(degree: usize, basis: [u8; 32]) -> TypeTag {
                TypeTag::BinaryTower {
                    bits: $bits,
                    degree,
                    basis,
                }
            }

            fn observe_seed<C: CanObserve<Self>>(challenger: &mut C, bytes: &[u8]) {
                // Bytes carried by one element of this level.
                const WIDTH: usize = $bits / 8;

                // Read one element off a chunk of at most `WIDTH` bytes, zero-padded on the left.
                //
                // The bytes are reinterpreted in the tower basis, not embedded as an integer.
                // An integer embedding lands in GF(2) and would keep only the parity of each byte.
                let element = |chunk: &[u8]| {
                    let mut padded = [0u8; WIDTH];
                    padded[..chunk.len()].copy_from_slice(chunk);
                    <$name>::from_repr(<$repr>::from_le_bytes(padded))
                };

                // The length goes first, as eight bytes, at every level.
                //
                // A level narrower than eight bytes spreads it over several elements.
                // A wider one holds it in a single zero-padded element.
                for chunk in (bytes.len() as u64).to_le_bytes().chunks(WIDTH) {
                    challenger.observe(element(chunk));
                }

                // Then the string itself, one element per chunk.
                for chunk in bytes.chunks(WIDTH) {
                    challenger.observe(element(chunk));
                }
            }

            fn wire_len() -> usize {
                $bits / 8
            }

            fn encode(value: &Self, out: &mut Vec<u8>) {
                out.extend_from_slice(&value.to_repr().to_be_bytes());
            }

            fn decode(bytes: &[u8]) -> Result<Self, TranscriptError> {
                let prefix = bytes
                    .get(..$bits / 8)
                    .ok_or(TranscriptError::BadProofShape {
                        reason: "not enough bytes for a canonical binary field encoding",
                    })?;
                // The element is exactly as wide as its backing integer, so every pattern is canonical.
                Ok(Self::from_repr(<$repr>::from_be_bytes(
                    prefix.try_into().unwrap(),
                )))
            }
        }
    };
}

impl_transcript_field!(BinaryField8, u8, 8);
impl_transcript_field!(BinaryField16, u16, 16);
impl_transcript_field!(BinaryField32, u32, 32);
impl_transcript_field!(BinaryField64, u64, 64);
impl_transcript_field!(BinaryField128, u128, 128);

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use p3_challenger::testing::Recorder;

    use super::*;

    /// Width of the length field written ahead of a seed, in bytes.
    const SEED_LEN_BYTES: usize = 8;

    /// The elements one byte string packs into at a given level.
    fn packed<F: TranscriptField>(bytes: &[u8]) -> Vec<F> {
        let mut recorder = Recorder::default();
        F::observe_seed(&mut recorder, bytes);
        recorder.into_absorbed()
    }

    #[test]
    fn a_seed_carries_its_length_ahead_of_its_bytes() {
        // At the narrowest level one element holds one byte.
        //
        //     length 3, little-endian over 8 bytes  ->  3, 0, 0, 0, 0, 0, 0, 0
        //     then the string itself                ->  1, 2, 3
        let seen = packed::<BinaryField8>(&[1, 2, 3]);

        assert_eq!(seen.len(), SEED_LEN_BYTES + 3);
        assert_eq!(seen[0], BinaryField8::from_repr(3));
        assert_eq!(&seen[SEED_LEN_BYTES..], &packed_bytes(&[1, 2, 3])[..]);
    }

    /// The tail elements of a packing, without the length field in front.
    fn packed_bytes(bytes: &[u8]) -> Vec<BinaryField8> {
        bytes.iter().map(|&b| BinaryField8::from_repr(b)).collect()
    }

    #[test]
    fn a_shorter_string_is_never_a_prefix_of_a_longer_one() {
        // Invariant: absorption is injective, so a padded string cannot impersonate another.
        //
        // Without the length field, `[1]` and `[1, 0]` would pack to the same elements at
        // every level whose width divides neither.
        assert_ne!(
            packed::<BinaryField8>(&[1]),
            packed::<BinaryField8>(&[1, 0])
        );
        assert_ne!(
            packed::<BinaryField16>(&[1]),
            packed::<BinaryField16>(&[1, 0])
        );
        assert_ne!(
            packed::<BinaryField32>(&[1]),
            packed::<BinaryField32>(&[1, 0])
        );
        assert_ne!(
            packed::<BinaryField64>(&[1]),
            packed::<BinaryField64>(&[1, 0])
        );
        assert_ne!(
            packed::<BinaryField128>(&[1]),
            packed::<BinaryField128>(&[1, 0])
        );
    }

    #[test]
    fn the_widest_level_packs_a_seed_by_the_same_rule_as_the_narrow_ones() {
        // Invariant: one seeding rule covers the whole tower, so the widest level is not special.
        //
        // Sixteen bytes fit one element, so a 20-byte seed spans two of them.
        //
        //     length 20, little-endian over 8 bytes, zero-padded to 16  ->  1 element
        //     bytes 0..16                                               ->  1 element
        //     bytes 16..20, zero-padded on the right                    ->  1 element
        let bytes: Vec<u8> = (0..20).collect();
        let seen = packed::<BinaryField128>(&bytes);

        assert_eq!(seen.len(), 3);
        assert_eq!(seen[0], BinaryField128::from_repr(20));

        // The payload elements read the chunk least significant byte first.
        let mut first = [0u8; 16];
        first.copy_from_slice(&bytes[..16]);
        assert_eq!(
            seen[1],
            BinaryField128::from_repr(u128::from_le_bytes(first))
        );

        let mut last = [0u8; 16];
        last[..4].copy_from_slice(&bytes[16..]);
        assert_eq!(
            seen[2],
            BinaryField128::from_repr(u128::from_le_bytes(last))
        );
    }

    #[test]
    fn a_wire_encoding_round_trips_at_every_level() {
        // Fixture: one element per level, written out and read back.
        let mut out = Vec::new();
        BinaryField8::encode(&BinaryField8::from_repr(0xA5), &mut out);
        assert_eq!(out, vec![0xA5]);
        assert_eq!(
            BinaryField8::decode(&out).unwrap(),
            BinaryField8::from_repr(0xA5)
        );

        let mut out = Vec::new();
        BinaryField16::encode(&BinaryField16::from_repr(0xA5C3), &mut out);
        assert_eq!(out, vec![0xA5, 0xC3]);
        assert_eq!(
            BinaryField16::decode(&out).unwrap(),
            BinaryField16::from_repr(0xA5C3)
        );

        let mut out = Vec::new();
        BinaryField32::encode(&BinaryField32::from_repr(0x0123_4567), &mut out);
        assert_eq!(
            BinaryField32::decode(&out).unwrap(),
            BinaryField32::from_repr(0x0123_4567)
        );

        let mut out = Vec::new();
        BinaryField64::encode(&BinaryField64::from_repr(0x0123_4567_89AB_CDEF), &mut out);
        assert_eq!(
            BinaryField64::decode(&out).unwrap(),
            BinaryField64::from_repr(0x0123_4567_89AB_CDEF)
        );
    }

    #[test]
    fn a_short_wire_encoding_is_rejected() {
        // The caller owes a whole element; a truncated one is malformed input, not a panic.
        assert!(BinaryField64::decode(&[0; 7]).is_err());
        assert!(BinaryField32::decode(&[0; 3]).is_err());
        assert!(BinaryField16::decode(&[0; 1]).is_err());
        assert!(BinaryField8::decode(&[]).is_err());
    }

    #[test]
    fn each_level_names_its_own_width() {
        // Two levels must never share an algebra tag, or one could stand in for the other.
        assert_eq!(
            BinaryField8::algebra_tag(1, [0; 32]),
            TypeTag::BinaryTower {
                bits: 8,
                degree: 1,
                basis: [0; 32]
            }
        );
        assert_ne!(
            BinaryField8::algebra_tag(1, [0; 32]),
            BinaryField16::algebra_tag(1, [0; 32])
        );
        assert_ne!(
            BinaryField32::algebra_tag(1, [0; 32]),
            BinaryField64::algebra_tag(1, [0; 32])
        );
        assert_ne!(
            BinaryField64::algebra_tag(1, [0; 32]),
            BinaryField128::algebra_tag(1, [0; 32])
        );
    }
}
