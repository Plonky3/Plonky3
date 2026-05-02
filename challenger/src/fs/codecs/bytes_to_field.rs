//! Byte-sponge ↔ prime-field-scalar codec.

use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_field::PrimeField;

use crate::fs::codecs::Codec;
use crate::fs::codecs::decode_field::{
    decode_field_via_extra_bytes, field_byte_size, required_bytes,
};
use crate::{CanObserve, CanSample};

/// Codec that maps a prime-field scalar to bytes and back via the IETF `Ns + 16` rule.
///
/// Statistical distance from uniform on the field is bounded by `2^-128`
/// independently of the prime (DSFS Lemma C.1).
pub struct BytesToFieldCodec<F>(PhantomData<F>);

impl<C, F> Codec<C, F> for BytesToFieldCodec<F>
where
    F: PrimeField,
    C: CanObserve<u8> + CanSample<u8>,
{
    const SECURITY_BITS: u32 = 128;

    fn observe(challenger: &mut C, value: &F) {
        // Big-endian canonical encoding.
        //
        // Left-padded so every element is the same width.
        let big = value.as_canonical_biguint();
        let target = field_byte_size::<F>();
        let raw = big.to_bytes_be();
        let mut buf: Vec<u8> = Vec::with_capacity(target);
        if raw.len() < target {
            buf.resize(target - raw.len(), 0);
        }
        buf.extend_from_slice(&raw);
        challenger.observe_slice(&buf);
    }

    fn sample(challenger: &mut C) -> F {
        // Draw `Ns + 16` bytes;
        //
        // The 16 spares give the `2^-128` bias bound.
        let need = required_bytes::<F>();
        let mut bytes: Vec<u8> = Vec::with_capacity(need);
        for _ in 0..need {
            bytes.push(challenger.sample());
        }
        decode_field_via_extra_bytes::<F>(&bytes)
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use p3_baby_bear::BabyBear;
    use p3_field::{PrimeCharacteristicRing, PrimeField32};
    use proptest::prelude::*;

    use super::*;
    use crate::fs::codecs::decode_field::field_byte_size;

    /// Concrete field exercised in this module's tests.
    type F = BabyBear;

    /// FIFO byte queue: absorbed bytes come back out in order.
    #[derive(Default)]
    struct ByteFifo {
        buf: Vec<u8>,
    }

    impl CanObserve<u8> for ByteFifo {
        fn observe(&mut self, value: u8) {
            self.buf.push(value);
        }
    }

    impl CanSample<u8> for ByteFifo {
        fn sample(&mut self) -> u8 {
            assert!(
                !self.buf.is_empty(),
                "ByteFifo: cannot sample from empty queue"
            );
            self.buf.remove(0)
        }
    }

    #[test]
    fn observe_writes_canonical_big_endian_padded_to_field_byte_size() {
        // Element 1 over F encodes as [0, 0, 0, 1].
        let mut s = ByteFifo::default();
        let one = F::from_u32(1u32);
        BytesToFieldCodec::<F>::observe(&mut s, &one);

        let want_len = field_byte_size::<F>();
        assert_eq!(s.buf.len(), want_len);
        assert_eq!(s.buf[..want_len - 1], [0u8; 3]);
        assert_eq!(s.buf[want_len - 1], 1);
    }

    proptest! {
        #[test]
        fn sample_is_deterministic_function_of_byte_queue(seed in any::<[u8; 32]>()) {
            // Same input bytes -> same field element.
            let mut a = ByteFifo::default();
            let mut b = ByteFifo::default();
            let need = required_bytes::<F>();
            for i in 0..need {
                a.observe(seed[i % seed.len()]);
                b.observe(seed[i % seed.len()]);
            }
            let fa = BytesToFieldCodec::<F>::sample(&mut a);
            let fb = BytesToFieldCodec::<F>::sample(&mut b);
            prop_assert_eq!(fa, fb);
            prop_assert!(fa.as_canonical_u32() < F::ORDER_U32);
        }
    }
}
