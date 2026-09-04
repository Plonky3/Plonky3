//! Identity codec for sponges whose alphabet is the field itself.

use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_field::PrimeField64;

use crate::fs::codecs::{Codec, decode_field_be_canonical, encode_field_be, field_byte_size};
use crate::fs::error::TranscriptError;
use crate::{CanObserve, CanSample};

/// Identity codec: the sponge alphabet is the field, so no encoding step is needed.
///
/// The sponge half is a straight pass-through.
///
/// The wire half is the same canonical big-endian encoding every other codec
/// uses, so a proof stays byte-identical whichever sponge produced it.
pub struct FieldToFieldCodec<F>(PhantomData<F>);

impl<C, F> Codec<C, F> for FieldToFieldCodec<F>
where
    F: PrimeField64,
    C: CanObserve<F> + CanSample<F>,
{
    /// A sample is exactly the sponge's own output.
    ///
    /// Its distance from uniform is whatever the sponge's is.
    /// No decoding step adds bias on top.
    /// The 128-bit budget is met by construction, not by an argument about byte counts.
    const SECURITY_BITS: u32 = 128;

    fn wire_len() -> usize {
        field_byte_size::<F>()
    }

    fn observe(challenger: &mut C, value: &F) {
        challenger.observe(*value);
    }

    fn sample(challenger: &mut C) -> F {
        challenger.sample()
    }

    fn encode(value: &F, out: &mut Vec<u8>) {
        encode_field_be(value, out);
    }

    fn decode(bytes: &[u8]) -> Result<F, TranscriptError> {
        decode_field_be_canonical(bytes)
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use p3_baby_bear::BabyBear;
    use p3_field::{PrimeCharacteristicRing, PrimeField32};

    use super::*;

    /// Concrete field exercised in this module's tests.
    type F = BabyBear;

    /// FIFO sponge over `F`.
    #[derive(Default)]
    struct FieldFifo {
        queue: Vec<F>,
    }

    impl CanObserve<F> for FieldFifo {
        fn observe(&mut self, v: F) {
            self.queue.push(v);
        }
    }

    impl CanSample<F> for FieldFifo {
        fn sample(&mut self) -> F {
            self.queue.remove(0)
        }
    }

    #[test]
    fn observe_then_sample_round_trips() {
        // Identity codec: elements come back out in the order they went in.
        let mut s = FieldFifo::default();

        let v1 = F::from_u32(7u32);
        let v2 = F::from_u32(123u32);
        let v3 = F::NEG_ONE;

        FieldToFieldCodec::<F>::observe(&mut s, &v1);
        FieldToFieldCodec::<F>::observe(&mut s, &v2);
        FieldToFieldCodec::<F>::observe(&mut s, &v3);

        assert_eq!(FieldToFieldCodec::<F>::sample(&mut s), v1);
        assert_eq!(FieldToFieldCodec::<F>::sample(&mut s), v2);
        assert_eq!(FieldToFieldCodec::<F>::sample(&mut s), v3);
    }

    #[test]
    fn wire_encoding_matches_the_byte_sponge_codec() {
        // Invariant: the wire format does not depend on the sponge alphabet.
        //
        // A proof written by a field sponge must decode under a byte sponge.
        let value = F::from_u32(0x0012_3456);
        let mut field_side = Vec::new();
        <FieldToFieldCodec<F> as Codec<FieldFifo, F>>::encode(&value, &mut field_side);
        assert_eq!(field_side, [0x00, 0x12, 0x34, 0x56]);
        assert_eq!(
            <FieldToFieldCodec<F> as Codec<FieldFifo, F>>::decode(&field_side).unwrap(),
            value
        );
        assert_eq!(
            field_side.len(),
            <FieldToFieldCodec<F> as Codec<FieldFifo, F>>::wire_len()
        );
    }

    #[test]
    fn decode_rejects_a_non_canonical_encoding() {
        // A value at or above the modulus is malformed input, not a field element.
        let at_modulus = F::ORDER_U32.to_be_bytes();
        assert!(<FieldToFieldCodec<F> as Codec<FieldFifo, F>>::decode(&at_modulus).is_err());
    }
}
