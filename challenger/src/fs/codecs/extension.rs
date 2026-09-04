//! Codec for extension-field elements over a base-field codec.

use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_field::{BasedVectorSpace, Field};

use crate::fs::codecs::Codec;
use crate::fs::error::TranscriptError;

/// Lifts a base-field codec to an extension-field codec, coefficient by coefficient.
///
/// The wire format follows the basis order returned by the field crate:
/// `DIMENSION` base-field encodings back to back.
///
/// Both drivers route every extension step through this type, so there is one
/// implementation of that layout and nothing to keep in sync.
pub struct ExtensionFieldCodec<F, EF, BaseCodec>(PhantomData<(F, EF, BaseCodec)>);

impl<C, F, EF, BaseCodec> Codec<C, EF> for ExtensionFieldCodec<F, EF, BaseCodec>
where
    F: Field,
    EF: Field + BasedVectorSpace<F>,
    BaseCodec: Codec<C, F>,
{
    const SECURITY_BITS: u32 = BaseCodec::SECURITY_BITS;

    fn wire_len() -> usize {
        EF::DIMENSION * BaseCodec::wire_len()
    }

    fn observe(challenger: &mut C, value: &EF) {
        // Walk the basis in stored order so observe is deterministic.
        for coeff in value.as_basis_coefficients_slice() {
            BaseCodec::observe(challenger, coeff);
        }
    }

    fn sample(challenger: &mut C) -> EF {
        // Sample one coefficient per basis index in the matching order.
        EF::from_basis_coefficients_fn(|_| BaseCodec::sample(challenger))
    }

    fn encode(value: &EF, out: &mut Vec<u8>) {
        // One base-field encoding per coefficient, in basis order.
        for coeff in value.as_basis_coefficients_slice() {
            BaseCodec::encode(coeff, out);
        }
    }

    fn decode(bytes: &[u8]) -> Result<EF, TranscriptError> {
        let width = BaseCodec::wire_len();
        // The caller owes us one full element; a short slice is malformed input.
        if bytes.len() < Self::wire_len() {
            return Err(TranscriptError::BadProofShape {
                reason: "not enough bytes for an extension-field encoding",
            });
        }
        // Decode each fixed-width chunk, rejecting any non-canonical coefficient.
        let mut coeffs: Vec<F> = Vec::with_capacity(EF::DIMENSION);
        for i in 0..EF::DIMENSION {
            coeffs.push(BaseCodec::decode(&bytes[i * width..])?);
        }
        // Reconstruct in the same basis order the encoder used.
        EF::from_basis_coefficients_iter(coeffs.into_iter()).ok_or(TranscriptError::BadProofShape {
            reason: "extension element basis size mismatch",
        })
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use p3_baby_bear::BabyBear;
    use p3_field::extension::BinomialExtensionField;
    use p3_field::{BasedVectorSpace, PrimeCharacteristicRing};

    use super::*;
    use crate::fs::codecs::field_to_field::FieldToFieldCodec;
    use crate::{CanObserve, CanSample};

    /// Concrete base field exercised in this module's tests.
    type F = BabyBear;
    /// Degree-4 binomial extension over `F`.
    type EF4 = BinomialExtensionField<F, 4>;
    /// Extension codec under test.
    type Cdc = ExtensionFieldCodec<F, EF4, FieldToFieldCodec<F>>;

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

    /// Degree-4 element with four distinct coefficients.
    fn sample_value() -> EF4 {
        let coeffs: [F; 4] = [
            F::from_u32(11u32),
            F::from_u32(13u32),
            F::from_u32(17u32),
            F::from_u32(19u32),
        ];
        EF4::from_basis_coefficients_iter(coeffs.iter().copied())
            .expect("constructing an extension from its basis coefficients")
    }

    #[test]
    fn extension_codec_round_trips_through_field_sponge() {
        // Observe + sample over a FIFO returns the original element.
        let value = sample_value();

        let mut sponge = FieldFifo::default();
        Cdc::observe(&mut sponge, &value);
        assert_eq!(sponge.queue.len(), 4);

        let sampled: EF4 = Cdc::sample(&mut sponge);
        assert_eq!(sampled, value);

        // Security inherits from the base codec.
        assert_eq!(
            <Cdc as Codec<FieldFifo, EF4>>::SECURITY_BITS,
            <FieldToFieldCodec<F> as Codec<FieldFifo, F>>::SECURITY_BITS,
        );
    }

    #[test]
    fn extension_wire_layout_is_the_base_encoding_repeated() {
        // Invariant: `DIMENSION` base encodings back to back, in basis order.
        let value = sample_value();

        let mut out = Vec::new();
        <Cdc as Codec<FieldFifo, EF4>>::encode(&value, &mut out);
        assert_eq!(out.len(), <Cdc as Codec<FieldFifo, EF4>>::wire_len());
        assert_eq!(out.len(), 4 * 4);

        // Each 4-byte block is one canonical base-field coefficient.
        assert_eq!(&out[..4], &[0, 0, 0, 11]);
        assert_eq!(&out[4..8], &[0, 0, 0, 13]);
        assert_eq!(&out[8..12], &[0, 0, 0, 17]);
        assert_eq!(&out[12..], &[0, 0, 0, 19]);

        // Decoding the block recovers the element.
        assert_eq!(<Cdc as Codec<FieldFifo, EF4>>::decode(&out).unwrap(), value);
    }

    #[test]
    fn decode_rejects_a_truncated_element() {
        // One byte short of a full element is malformed input.
        let mut out = Vec::new();
        <Cdc as Codec<FieldFifo, EF4>>::encode(&sample_value(), &mut out);
        out.pop();
        assert_eq!(
            <Cdc as Codec<FieldFifo, EF4>>::decode(&out),
            Err(TranscriptError::BadProofShape {
                reason: "not enough bytes for an extension-field encoding",
            })
        );
    }
}
