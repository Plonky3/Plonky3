use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_field::{ExtensionField, PrimeField32, PrimeField64};
use p3_maybe_rayon::prelude::*;
use p3_symmetric::{CryptographicHasher, Hash};
use p3_util::log2_ceil_u64;
use tracing::instrument;

use crate::{
    CanObserve, CanSample, CanSampleBits, FieldChallenger, GrindingChallenger, HashChallenger,
};

/// Given a challenger that can observe and sample bytes, produces a challenger that is able to
/// sample and observe field elements of a `PrimeField32`.
///
/// **Observing**:
/// -  Takes a field element will serialize it into a byte array and observe each byte.
///
/// **Sampling**:
/// -  Samples a field element in a prime field of size `p` by sampling uniformly an element in the
///    range (0..1 << log_2(p)). This avoids modulo bias.
#[derive(Clone, Debug)]
pub struct SerializingChallenger32<F, Inner> {
    inner: Inner,
    _marker: PhantomData<F>,
}

/// Given a challenger that can observe and sample bytes, produces a challenger that is able to
/// sample and observe field elements of a `PrimeField64` field.
///
/// **Observing**:
/// -  Takes a field element will serialize it into a byte array and observe each byte.
///
/// **Sampling**:
/// -  Samples a field element in a prime field of size `p` by sampling unofrmly an element in the
///    range (0..1 << log_2(p)). This avoids modulo bias.
#[derive(Clone, Debug)]
pub struct SerializingChallenger64<F, Inner> {
    inner: Inner,
    _marker: PhantomData<F>,
}

impl<F: PrimeField32, Inner: CanObserve<u8>> SerializingChallenger32<F, Inner> {
    pub const fn new(inner: Inner) -> Self {
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

impl<F: PrimeField32, const N: usize, Inner: CanObserve<u8>> CanObserve<Hash<F, u64, N>>
    for SerializingChallenger32<F, Inner>
{
    fn observe(&mut self, values: Hash<F, u64, N>) {
        for value in values {
            self.inner.observe_slice(&value.to_le_bytes());
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
        let modulus = F::ORDER_U64 as u32;
        let log_size = log2_ceil_u64(F::ORDER_U64);
        let pow_of_two_bound = (1 << log_size) - 1;
        // Perform rejection sampling over the uniform range (0..log2_ceil(p))
        let sample_base = |inner: &mut Inner| loop {
            let value = u32::from_le_bytes(inner.sample_array::<4>());
            let value = value & pow_of_two_bound;
            if value < modulus {
                return F::from_canonical_u32(value);
            }
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
        // Limiting the number of bits to the field size
        debug_assert!((1 << bits) <= F::ORDER_U64 as usize);
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

impl<F: PrimeField64, Inner: CanObserve<u8>> SerializingChallenger64<F, Inner> {
    pub const fn new(inner: Inner) -> Self {
        Self {
            inner,
            _marker: PhantomData,
        }
    }
}

impl<F, H> SerializingChallenger64<F, HashChallenger<u8, H, 32>>
where
    F: PrimeField64,
    H: CryptographicHasher<u8, [u8; 32]>,
{
    pub fn from_hasher(initial_state: Vec<u8>, hasher: H) -> Self {
        Self::new(HashChallenger::new(initial_state, hasher))
    }
}

impl<F: PrimeField64, Inner: CanObserve<u8>> CanObserve<F> for SerializingChallenger64<F, Inner> {
    fn observe(&mut self, value: F) {
        self.inner
            .observe_slice(&value.as_canonical_u64().to_le_bytes());
    }
}

impl<F: PrimeField64, const N: usize, Inner: CanObserve<u8>> CanObserve<Hash<F, u8, N>>
    for SerializingChallenger64<F, Inner>
{
    fn observe(&mut self, values: Hash<F, u8, N>) {
        for value in values {
            self.inner.observe(value);
        }
    }
}

impl<F, EF, Inner> CanSample<EF> for SerializingChallenger64<F, Inner>
where
    F: PrimeField64,
    EF: ExtensionField<F>,
    Inner: CanSample<u8>,
{
    fn sample(&mut self) -> EF {
        let modulus = F::ORDER_U64;
        let log_size = log2_ceil_u64(F::ORDER_U64);
        let pow_of_two_bound = (1 << log_size) - 1;
        // Perform rejection sampling over the uniform range (0..log2_ceil(p))
        let sample_base = |inner: &mut Inner| loop {
            let value = u64::from_le_bytes(inner.sample_array::<8>());
            let value = value & pow_of_two_bound;
            if value < modulus {
                return F::from_canonical_u64(value);
            }
        };
        EF::from_base_fn(|_| sample_base(&mut self.inner))
    }
}

impl<F, Inner> CanSampleBits<usize> for SerializingChallenger64<F, Inner>
where
    F: PrimeField64,
    Inner: CanSample<u8>,
{
    fn sample_bits(&mut self, bits: usize) -> usize {
        debug_assert!(bits < (usize::BITS as usize));
        // Limiting the number of bits to the field size
        debug_assert!((1 << bits) <= F::ORDER_U64 as usize);
        let rand_usize = u64::from_le_bytes(self.inner.sample_array::<8>()) as usize;
        rand_usize & ((1 << bits) - 1)
    }
}

impl<F, Inner> GrindingChallenger for SerializingChallenger64<F, Inner>
where
    F: PrimeField64,
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

impl<F, Inner> FieldChallenger<F> for SerializingChallenger64<F, Inner>
where
    F: PrimeField64,
    Inner: CanSample<u8> + CanObserve<u8> + Clone + Send + Sync,
{
}
