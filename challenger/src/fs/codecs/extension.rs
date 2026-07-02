//! Codec for extension-field elements over a base-field codec.

use core::marker::PhantomData;

use p3_field::{BasedVectorSpace, Field};

use crate::fs::codecs::Codec;

/// Lifts a base-field codec to an extension-field codec, coefficient by coefficient.
///
/// The wire format follows the basis order returned by the field crate.
pub struct ExtensionFieldCodec<F, EF, BaseCodec>(PhantomData<(F, EF, BaseCodec)>);

impl<C, F, EF, BaseCodec> Codec<C, EF> for ExtensionFieldCodec<F, EF, BaseCodec>
where
    F: Field,
    EF: Field + BasedVectorSpace<F>,
    BaseCodec: Codec<C, F>,
{
    const SECURITY_BITS: u32 = BaseCodec::SECURITY_BITS;

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
    fn extension_codec_round_trips_through_field_sponge() {
        // Observe + sample over a FIFO returns the original element.
        let coeffs: [F; 4] = [
            F::from_u32(11u32),
            F::from_u32(13u32),
            F::from_u32(17u32),
            F::from_u32(19u32),
        ];
        let value = EF4::from_basis_coefficients_iter(coeffs.iter().copied())
            .expect("constructing an extension from its basis coefficients");

        type Cdc = ExtensionFieldCodec<F, EF4, FieldToFieldCodec<F>>;

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
}
