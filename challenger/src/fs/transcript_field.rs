//! Field representation hooks for typed, native-field transcripts.

use alloc::vec::Vec;

use p3_field::{Field, PrimeField64};

use super::codecs::{decode_field_be_canonical, encode_field_be, field_byte_size};
use super::{FieldUnit, TranscriptError, TypeTag};
use crate::CanObserve;

/// A field with stable encodings for a typed native-field transcript.
///
/// Implementations must make seed absorption injective and use a unique, fixed-width
/// wire encoding for every element. The algebra tag must identify the coefficient
/// field and its representation independently of Rust type names, and bind the
/// supplied extension degree. These contracts are required for transcript binding.
///
/// Prime fields below `2^64` retain their canonical big-endian wire format and
/// length-prefixed little-endian seed packing through the blanket implementation.
pub trait TranscriptField: Field {
    /// Stable identity of an algebra with `degree` coefficients over this field.
    fn algebra_tag(degree: usize) -> TypeTag;

    /// Absorb a byte string injectively into this field's sponge alphabet.
    fn observe_seed<C: CanObserve<Self>>(challenger: &mut C, bytes: &[u8]);

    /// Byte length of the canonical encoding of one element.
    fn wire_len() -> usize;

    /// Append exactly `Self::wire_len()` bytes identifying this element uniquely.
    fn encode(value: &Self, out: &mut Vec<u8>);

    /// Decode the first `Self::wire_len()` bytes, rejecting short or noncanonical inputs.
    fn decode(bytes: &[u8]) -> Result<Self, TranscriptError>;
}

impl<F: PrimeField64> TranscriptField for F {
    fn algebra_tag(degree: usize) -> TypeTag {
        TypeTag::Algebra {
            modulus: F::ORDER_U64,
            degree,
        }
    }

    fn observe_seed<C: CanObserve<F>>(challenger: &mut C, bytes: &[u8]) {
        let chunk = FieldUnit::<F>::bytes_per_element();
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

    fn wire_len() -> usize {
        field_byte_size::<F>()
    }

    fn encode(value: &Self, out: &mut Vec<u8>) {
        encode_field_be(value, out);
    }

    fn decode(bytes: &[u8]) -> Result<Self, TranscriptError> {
        decode_field_be_canonical(bytes)
    }
}
