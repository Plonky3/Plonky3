//! Field-sponge to byte codec.

use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_field::{PrimeField32, PrimeField64};

use crate::fs::codecs::Codec;
use crate::{CanObserve, CanSample};

/// Maps bytes to/from a 32-bit prime-field sponge.
///
/// Sample takes the low byte of a uniform field element, which biases by `~1/p`.
///
/// For 128-bit-clean output, route via a byte sponge instead.
pub struct FieldToBytesCodec<F>(PhantomData<F>);

impl<C, F> Codec<C, u8> for FieldToBytesCodec<F>
where
    F: PrimeField32,
    C: CanObserve<F> + CanSample<F>,
{
    /// Worst-case truncation bias for a 31-bit field: TV ≤ 2^8 / 2^31 = 2^-23.
    ///
    /// Use a byte sponge if cryptographic-grade uniformity is required.
    const SECURITY_BITS: u32 = 23;

    fn observe(challenger: &mut C, value: &u8) {
        // 0..=255 fits in any prime field with order > 256.
        challenger.observe(F::from_u8(*value));
    }

    fn sample(challenger: &mut C) -> u8 {
        let f: F = challenger.sample();
        f.as_canonical_u32() as u8
    }
}

/// Same construction as the 32-bit variant, for 64-bit prime fields.
pub struct FieldToBytesCodec64<F>(PhantomData<F>);

impl<C, F> Codec<C, u8> for FieldToBytesCodec64<F>
where
    F: PrimeField64,
    C: CanObserve<F> + CanSample<F>,
{
    /// Worst-case truncation bias for a 64-bit field: TV ≤ 2^8 / 2^64 = 2^-56.
    ///
    /// Use a byte sponge if cryptographic-grade uniformity is required.
    const SECURITY_BITS: u32 = 56;

    fn observe(challenger: &mut C, value: &u8) {
        challenger.observe(F::from_u8(*value));
    }

    fn sample(challenger: &mut C) -> u8 {
        let f: F = challenger.sample();
        f.as_canonical_u64() as u8
    }
}

/// Sample `n` bytes from a field sponge.
///
/// Behaviourally identical to looping over the per-byte sample method,
/// but skips the per-byte trait dispatch.
pub fn sample_bytes_from_field_sponge<C, F>(challenger: &mut C, n: usize) -> Vec<u8>
where
    F: PrimeField32,
    C: CanObserve<F> + CanSample<F>,
{
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        out.push(<FieldToBytesCodec<F> as Codec<C, u8>>::sample(challenger));
    }
    out
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;

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
    fn observe_packs_byte_into_canonical_field_element() {
        // Byte 0x42 -> field element with integer value 0x42.
        let mut s = FieldFifo::default();
        FieldToBytesCodec::<F>::observe(&mut s, &0x42u8);
        assert_eq!(s.queue.len(), 1);
        assert_eq!(s.queue[0], F::from_u32(0x42u32));
    }

    #[test]
    fn sample_takes_low_byte_of_field_element() {
        // 0xaabbcc -> low byte 0xcc.
        let mut s = FieldFifo::default();
        s.queue.push(F::from_u32(0x00_aa_bb_ccu32));

        let b = FieldToBytesCodec::<F>::sample(&mut s);
        assert_eq!(b, 0xcc);
    }

    #[test]
    fn sample_bytes_helper_returns_requested_length() {
        // Asking for 7 bytes yields exactly 7 bytes from low-byte truncation.
        let mut s = FieldFifo::default();
        for v in 0u32..7 {
            s.queue.push(F::from_u32(v));
        }
        let bytes = sample_bytes_from_field_sponge::<_, F>(&mut s, 7);
        assert_eq!(bytes.len(), 7);
        for (i, b) in bytes.iter().enumerate() {
            assert_eq!(*b, i as u8);
        }
    }
}
