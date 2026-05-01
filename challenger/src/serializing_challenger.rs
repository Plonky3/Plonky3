use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_field::{BasedVectorSpace, PrimeField32, PrimeField64};
use p3_maybe_rayon::prelude::*;
use p3_symmetric::{CryptographicHasher, Hash, MerkleCap};
use p3_util::log2_ceil_u64;
use tracing::instrument;

use crate::{
    CanFinalizeDigest, CanObserve, CanSample, CanSampleBits, CanSampleUniformBits, FieldChallenger,
    GrindingChallenger, HashChallenger, ResamplingError,
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
/// -  Samples a field element in a prime field of size `p` by sampling uniformly an element in the
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
    pub const fn from_hasher(initial_state: Vec<u8>, hasher: H) -> Self {
        Self::new(HashChallenger::new(initial_state, hasher))
    }
}

impl<F: PrimeField32, Inner: CanObserve<u8>> CanObserve<F> for SerializingChallenger32<F, Inner> {
    fn observe(&mut self, value: F) {
        self.inner
            .observe_slice(&value.to_unique_u32().to_le_bytes());
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

impl<F: PrimeField32, const N: usize, Inner: CanObserve<u8>> CanObserve<&MerkleCap<F, [u8; N]>>
    for SerializingChallenger32<F, Inner>
{
    fn observe(&mut self, cap: &MerkleCap<F, [u8; N]>) {
        for digest in cap.roots() {
            for value in digest {
                self.inner.observe(*value);
            }
        }
    }
}

impl<F: PrimeField32, const N: usize, Inner: CanObserve<u8>> CanObserve<MerkleCap<F, [u8; N]>>
    for SerializingChallenger32<F, Inner>
{
    fn observe(&mut self, cap: MerkleCap<F, [u8; N]>) {
        self.observe(&cap);
    }
}

impl<F: PrimeField32, const N: usize, Inner: CanObserve<u8>> CanObserve<&MerkleCap<F, [u64; N]>>
    for SerializingChallenger32<F, Inner>
{
    fn observe(&mut self, cap: &MerkleCap<F, [u64; N]>) {
        for digest in cap.roots() {
            for value in digest {
                self.inner.observe_slice(&value.to_le_bytes());
            }
        }
    }
}

impl<F: PrimeField32, const N: usize, Inner: CanObserve<u8>> CanObserve<MerkleCap<F, [u64; N]>>
    for SerializingChallenger32<F, Inner>
{
    fn observe(&mut self, cap: MerkleCap<F, [u64; N]>) {
        self.observe(&cap);
    }
}

impl<F, EF, Inner> CanSample<EF> for SerializingChallenger32<F, Inner>
where
    F: PrimeField32,
    EF: BasedVectorSpace<F>,
    Inner: CanSample<u8>,
{
    fn sample(&mut self) -> EF {
        let modulus = F::ORDER_U32;
        let log_size = log2_ceil_u64(F::ORDER_U64);
        // We use u64 to avoid overflow in the case that log_size = 32.
        let pow_of_two_bound = ((1u64 << log_size) - 1) as u32;
        // Perform rejection sampling over the uniform range (0..log2_ceil(p))
        let sample_base = |inner: &mut Inner| loop {
            let value = u32::from_le_bytes(inner.sample_array());
            let value = value & pow_of_two_bound;
            if value < modulus {
                return unsafe {
                    // This is safe as value < F::ORDER_U32.
                    F::from_canonical_unchecked(value)
                };
            }
        };
        EF::from_basis_coefficients_fn(|_| sample_base(&mut self.inner))
    }
}

impl<F, Inner> CanSampleBits<usize> for SerializingChallenger32<F, Inner>
where
    F: PrimeField32,
    Inner: CanSample<u8>,
{
    fn sample_bits(&mut self, bits: usize) -> usize {
        assert!(bits < (usize::BITS as usize));
        // Limiting the number of bits to the field size
        assert!((1 << bits) <= F::ORDER_U64 as usize);
        let rand_usize = u32::from_le_bytes(self.inner.sample_array()) as usize;
        rand_usize & ((1 << bits) - 1)
    }
}

impl<F, Inner> CanSampleUniformBits<F> for SerializingChallenger32<F, Inner>
where
    F: PrimeField32,
    Inner: CanSample<u8>,
{
    /// Sample uniform bits by masking bytes from the inner stream.
    ///
    /// # Overview
    ///
    /// The inner stream emits cryptographic-hash bytes uniform on `[0, 2^8)`.
    ///
    /// Reading 4 bytes as a 32-bit integer and masking the low `bits` is
    /// exactly uniform on `[0, 2^bits)`.
    ///
    /// No field-element decomposition occurs, so no rejection band exists.
    /// The const generic is therefore inert: this function never errors
    /// and never resamples.
    fn sample_uniform_bits<const RESAMPLE: bool>(
        &mut self,
        bits: usize,
    ) -> Result<usize, ResamplingError> {
        // Byte-sourced sampling is uniform without rejection, so the
        // result is always valid and the error arm is unreachable.
        Ok(self.sample_bits(bits))
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
        assert!(bits < (usize::BITS as usize));
        assert!((1 << bits) < F::ORDER_U32);

        // Trivial case: 0 bits mean no PoW is required and any witness is valid.
        if bits == 0 {
            return F::ZERO;
        }

        let witness = (0..F::ORDER_U32)
            .into_par_iter()
            .map(|i| unsafe {
                // i < F::ORDER_U32 by construction so this is safe.
                F::from_canonical_unchecked(i)
            })
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
    pub const fn from_hasher(initial_state: Vec<u8>, hasher: H) -> Self {
        Self::new(HashChallenger::new(initial_state, hasher))
    }
}

impl<F: PrimeField64, Inner: CanObserve<u8>> CanObserve<F> for SerializingChallenger64<F, Inner> {
    fn observe(&mut self, value: F) {
        self.inner
            .observe_slice(&value.to_unique_u64().to_le_bytes());
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

impl<F: PrimeField64, const N: usize, Inner: CanObserve<u8>> CanObserve<Hash<F, u64, N>>
    for SerializingChallenger64<F, Inner>
{
    fn observe(&mut self, values: Hash<F, u64, N>) {
        for value in values {
            self.inner.observe_slice(&value.to_le_bytes());
        }
    }
}

impl<F: PrimeField64, const N: usize, Inner: CanObserve<u8>> CanObserve<&MerkleCap<F, [u8; N]>>
    for SerializingChallenger64<F, Inner>
{
    fn observe(&mut self, cap: &MerkleCap<F, [u8; N]>) {
        for digest in cap.roots() {
            for value in digest {
                self.inner.observe(*value);
            }
        }
    }
}

impl<F: PrimeField64, const N: usize, Inner: CanObserve<u8>> CanObserve<MerkleCap<F, [u8; N]>>
    for SerializingChallenger64<F, Inner>
{
    fn observe(&mut self, cap: MerkleCap<F, [u8; N]>) {
        self.observe(&cap);
    }
}

impl<F: PrimeField64, const N: usize, Inner: CanObserve<u8>> CanObserve<&MerkleCap<F, [u64; N]>>
    for SerializingChallenger64<F, Inner>
{
    fn observe(&mut self, cap: &MerkleCap<F, [u64; N]>) {
        for digest in cap.roots() {
            for value in digest {
                self.inner.observe_slice(&value.to_le_bytes());
            }
        }
    }
}

impl<F: PrimeField64, const N: usize, Inner: CanObserve<u8>> CanObserve<MerkleCap<F, [u64; N]>>
    for SerializingChallenger64<F, Inner>
{
    fn observe(&mut self, cap: MerkleCap<F, [u64; N]>) {
        self.observe(&cap);
    }
}

impl<F, EF, Inner> CanSample<EF> for SerializingChallenger64<F, Inner>
where
    F: PrimeField64,
    EF: BasedVectorSpace<F>,
    Inner: CanSample<u8>,
{
    fn sample(&mut self) -> EF {
        let modulus = F::ORDER_U64;
        let log_size = log2_ceil_u64(F::ORDER_U64) as u32;
        // We use u128 to avoid overflow in the case that log_size = 64.
        let pow_of_two_bound = ((1u128 << log_size) - 1) as u64;

        // Perform rejection sampling over the uniform range (0..log2_ceil(p))
        let sample_base = |inner: &mut Inner| loop {
            let value = u64::from_le_bytes(inner.sample_array());
            let value = value & pow_of_two_bound;
            if value < modulus {
                return unsafe {
                    // This is safe as value < F::ORDER_U64.
                    F::from_canonical_unchecked(value)
                };
            }
        };
        EF::from_basis_coefficients_fn(|_| sample_base(&mut self.inner))
    }
}

impl<F, Inner> CanSampleBits<usize> for SerializingChallenger64<F, Inner>
where
    F: PrimeField64,
    Inner: CanSample<u8>,
{
    fn sample_bits(&mut self, bits: usize) -> usize {
        assert!(bits < (usize::BITS as usize));
        assert!((1u64 << bits) <= F::ORDER_U64);
        let rand_u64 = u64::from_le_bytes(self.inner.sample_array());
        (rand_u64 & ((1u64 << bits) - 1)) as usize
    }
}

impl<F, Inner> CanSampleUniformBits<F> for SerializingChallenger64<F, Inner>
where
    F: PrimeField64,
    Inner: CanSample<u8>,
{
    /// Sample uniform bits by masking bytes from the inner stream.
    ///
    /// # Overview
    ///
    /// The inner stream emits cryptographic-hash bytes uniform on `[0, 2^8)`.
    ///
    /// Reading 8 bytes as a 64-bit integer and masking the low `bits` is
    /// exactly uniform on `[0, 2^bits)`.
    ///
    /// No field-element decomposition occurs, so no rejection band exists.
    /// The const generic is therefore inert: this function never errors
    /// and never resamples.
    fn sample_uniform_bits<const RESAMPLE: bool>(
        &mut self,
        bits: usize,
    ) -> Result<usize, ResamplingError> {
        // Byte-sourced sampling is uniform without rejection, so the
        // result is always valid and the error arm is unreachable.
        Ok(self.sample_bits(bits))
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
        assert!(bits < 64);
        assert!((1u64 << bits) < F::ORDER_U64);

        // Trivial case: 0 bits mean no PoW is required and any witness is valid.
        if bits == 0 {
            return F::ZERO;
        }

        let witness = (0..F::ORDER_U64)
            .into_par_iter()
            .map(|i| unsafe {
                // i < F::ORDER_U64 by construction so this is safe.
                F::from_canonical_unchecked(i)
            })
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

impl<F, Inner> CanFinalizeDigest for SerializingChallenger32<F, Inner>
where
    Inner: CanFinalizeDigest,
{
    type Digest = Inner::Digest;

    fn finalize(self) -> Self::Digest {
        self.inner.finalize()
    }
}

impl<F, Inner> CanFinalizeDigest for SerializingChallenger64<F, Inner>
where
    Inner: CanFinalizeDigest,
{
    type Digest = Inner::Digest;

    fn finalize(self) -> Self::Digest {
        self.inner.finalize()
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;
    use p3_goldilocks::Goldilocks;
    use p3_symmetric::CryptographicHasher;

    use super::*;
    use crate::HashChallenger;

    /// Toy byte hasher: deterministic length-only fingerprint.
    ///
    /// Enough to drive the challenger plumbing without pulling in a real hash crate.
    #[derive(Clone)]
    struct ByteCountHasher;

    impl CryptographicHasher<u8, [u8; 32]> for ByteCountHasher {
        fn hash_iter<I>(&self, input: I) -> [u8; 32]
        where
            I: IntoIterator<Item = u8>,
        {
            let len = input.into_iter().count() as u8;
            core::array::from_fn(|i| len.wrapping_add(i as u8))
        }

        fn hash_iter_slices<'a, I>(&self, input: I) -> [u8; 32]
        where
            I: IntoIterator<Item = &'a [u8]>,
        {
            let len = input.into_iter().map(<[u8]>::len).sum::<usize>() as u8;
            core::array::from_fn(|i| len.wrapping_add(i as u8))
        }
    }

    type Inner = HashChallenger<u8, ByteCountHasher, 32>;

    #[test]
    fn test_serializing_challenger32_grind_zero_bits_returns_zero() {
        // bits == 0: must short-circuit to ZERO without consuming bytes.
        type F = BabyBear;
        let inner = Inner::new(vec![0, 1, 2, 3], ByteCountHasher);
        let mut challenger = SerializingChallenger32::<F, Inner>::new(inner);

        // Pristine shadow: equal next-byte proves no inner mutation.
        let mut shadow = challenger.clone();

        let witness = challenger.grind(0);

        assert_eq!(witness, F::ZERO);
        let after_grind: u8 = challenger.inner.sample();
        let no_grind: u8 = shadow.inner.sample();
        assert_eq!(after_grind, no_grind);
    }

    #[test]
    fn test_serializing_challenger64_grind_zero_bits_returns_zero() {
        // bits == 0: must short-circuit to ZERO without consuming bytes.
        type F = Goldilocks;
        let inner = Inner::new(vec![0, 1, 2, 3], ByteCountHasher);
        let mut challenger = SerializingChallenger64::<F, Inner>::new(inner);

        // Pristine shadow: equal next-byte proves no inner mutation.
        let mut shadow = challenger.clone();

        let witness = challenger.grind(0);

        assert_eq!(witness, F::ZERO);
        let after_grind: u8 = challenger.inner.sample();
        let no_grind: u8 = shadow.inner.sample();
        assert_eq!(after_grind, no_grind);
    }
}
