use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_field::{ExtensionField, PrimeField32};
use p3_maybe_rayon::prelude::*;
use p3_symmetric::{CryptographicHasher, Hash};
use tracing::instrument;

use crate::{
    CanObserve, CanSample, CanSampleBits, FieldChallenger, GrindingChallenger, HashChallenger,
};

#[derive(Clone, Debug)]
pub struct SerializingChallenger32<F, Inner> {
    inner: Inner,
    _marker: PhantomData<F>,
}

impl<F: PrimeField32, Inner: CanObserve<u8>> SerializingChallenger32<F, Inner> {
    pub fn new(inner: Inner) -> Self {
        Self {
            inner,
            _marker: PhantomData,
        }
    }
}

impl<F, H> SerializingChallenger32<F, HashChallenger<u8, H, 32>>
where
    F: PrimeField32,
    H: CryptographicHasher<u8, [u8; 32]>,
{
    pub fn from_hasher(initial_state: Vec<u8>, hasher: H) -> Self {
        Self::new(HashChallenger::new(initial_state, hasher))
    }
}

impl<F: PrimeField32, Inner: CanObserve<u8>> CanObserve<F> for SerializingChallenger32<F, Inner> {
    fn observe(&mut self, value: F) {
        self.inner
            .observe_slice(&value.as_canonical_u32().to_le_bytes());
    }
}

impl<F: PrimeField32, const N: usize, Inner: CanObserve<u8>> CanObserve<Hash<F, u8, N>>
    for SerializingChallenger32<F, Inner>
{
    fn observe(&mut self, values: Hash<F, u8, N>) {
        for value in values {
            self.inner.observe(value);
        }
    }
}

impl<F, EF, Inner> CanSample<EF> for SerializingChallenger32<F, Inner>
where
    F: PrimeField32,
    EF: ExtensionField<F>,
    Inner: CanSample<u8>,
{
    fn sample(&mut self) -> EF {
        let sample_base = |inner: &mut Inner| {
            let bytes = inner.sample_array::<4>();
            F::from_wrapped_u32(u32::from_le_bytes(bytes))
        };
        EF::from_base_fn(|_| sample_base(&mut self.inner))
    }
}

impl<F, Inner> CanSampleBits<usize> for SerializingChallenger32<F, Inner>
where
    F: PrimeField32,
    Inner: CanSample<u8>,
{
    fn sample_bits(&mut self, bits: usize) -> usize {
        debug_assert!(bits < (usize::BITS as usize));
        // Limiting the number of bits to a u32 for
        debug_assert!((1 << bits) <= (u32::MAX as usize));
        let rand_usize = u32::from_le_bytes(self.inner.sample_array::<4>()) as usize;
        rand_usize & ((1 << bits) - 1)
    }
}

impl<F, Inner> GrindingChallenger for SerializingChallenger32<F, Inner>
where
    F: PrimeField32,
    Inner: CanSample<u8> + CanObserve<u8> + Clone + Send + Sync,
{
    type Witness = F;

    #[instrument(name = "grind for proof-of-work witness", skip_all)]
    fn grind(&mut self, bits: usize) -> Self::Witness {
        let witness = (0..F::ORDER_U64)
            .into_par_iter()
            .map(|i| F::from_canonical_u64(i))
            .find_any(|witness| self.clone().check_witness(bits, *witness))
            .expect("failed to find witness");
        assert!(self.check_witness(bits, witness));
        witness
    }
}

impl<F, Inner> FieldChallenger<F> for SerializingChallenger32<F, Inner>
where
    F: PrimeField32,
    Inner: CanSample<u8> + CanObserve<u8> + Clone + Send + Sync,
{
}
