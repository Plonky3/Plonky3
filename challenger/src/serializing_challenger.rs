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
        // Evaluate the bound in `u64` to keep the shift within its type width.
        // A `usize` shift by `bits >= 32` overflows on 32-bit targets and would zero the mask.
        assert!(
            (1u64 << bits) < F::ORDER_U64,
            "requested bit count must fit within the field order"
        );
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

impl<F, Inner> SerializingChallenger32<F, Inner>
where
    F: PrimeField32,
    Inner: CanSample<u8> + CanObserve<u8>,
{
    /// Hash everything observed since the last sample into the inner state.
    ///
    /// Discards one sampled byte. A proof-of-work step starts here, so each candidate
    /// witness is hashed against a digest of the transcript rather than its pending input.
    fn squeeze(&mut self) {
        let _: u8 = self.inner.sample();
    }

    /// Absorb `witness` and report whether the next `bits` sampled bits are all zero.
    fn witness_passes(&mut self, bits: usize, witness: F) -> bool {
        self.observe(witness);
        self.sample_bits(bits) == 0
    }
}

impl<F, Inner> GrindingChallenger for SerializingChallenger32<F, Inner>
where
    F: PrimeField32,
    Inner: CanSample<u8> + CanObserve<u8> + Clone + Send + Sync,
{
    type Witness = F;

    #[instrument(name = "grind for proof-of-work witness", skip_all, level = "debug")]
    fn grind(&mut self, bits: usize) -> Self::Witness {
        assert!(bits < (usize::BITS as usize));
        // Evaluate the bound in `u64` to keep the shift within its type width.
        // A `u32` shift by `bits >= 32` would wrap and accept a trivial proof-of-work.
        assert!(
            (1u64 << bits) < F::ORDER_U64,
            "requested bit count must fit within the field order"
        );

        // Trivial case: 0 bits mean no PoW is required and any witness is valid.
        if bits == 0 {
            return F::ZERO;
        }

        let mut squeezed = self.clone();
        squeezed.squeeze();
        let witness = (0..F::ORDER_U32)
            .into_par_iter()
            .map(|i| unsafe {
                // i < F::ORDER_U32 by construction so this is safe.
                F::from_canonical_unchecked(i)
            })
            .find_any(|witness| squeezed.clone().witness_passes(bits, *witness))
            .expect("failed to find witness");
        assert!(self.check_witness(bits, witness));
        witness
    }

    /// Squeeze the transcript, absorb `witness`, and check that the next `bits` sampled bits
    /// are all zero. Zero difficulty accepts any witness without touching the transcript.
    fn check_witness(&mut self, bits: usize, witness: Self::Witness) -> bool {
        if bits == 0 {
            return true;
        }
        self.squeeze();
        self.witness_passes(bits, witness)
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
        assert!((1u64 << bits) < F::ORDER_U64);
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

impl<F, Inner> SerializingChallenger64<F, Inner>
where
    F: PrimeField64,
    Inner: CanSample<u8> + CanObserve<u8>,
{
    /// Hash everything observed since the last sample into the inner state.
    ///
    /// Discards one sampled byte. A proof-of-work step starts here, so each candidate
    /// witness is hashed against a digest of the transcript rather than its pending input.
    fn squeeze(&mut self) {
        let _: u8 = self.inner.sample();
    }

    /// Absorb `witness` and report whether the next `bits` sampled bits are all zero.
    fn witness_passes(&mut self, bits: usize, witness: F) -> bool {
        self.observe(witness);
        self.sample_bits(bits) == 0
    }
}

impl<F, Inner> GrindingChallenger for SerializingChallenger64<F, Inner>
where
    F: PrimeField64,
    Inner: CanSample<u8> + CanObserve<u8> + Clone + Send + Sync,
{
    type Witness = F;

    #[instrument(name = "grind for proof-of-work witness", skip_all, level = "debug")]
    fn grind(&mut self, bits: usize) -> Self::Witness {
        assert!(bits < 64);
        assert!((1u64 << bits) < F::ORDER_U64);

        // Trivial case: 0 bits mean no PoW is required and any witness is valid.
        if bits == 0 {
            return F::ZERO;
        }

        let mut squeezed = self.clone();
        squeezed.squeeze();
        let witness = (0..F::ORDER_U64)
            .into_par_iter()
            .map(|i| unsafe {
                // i < F::ORDER_U64 by construction so this is safe.
                F::from_canonical_unchecked(i)
            })
            .find_any(|witness| squeezed.clone().witness_passes(bits, *witness))
            .expect("failed to find witness");
        assert!(self.check_witness(bits, witness));
        witness
    }

    /// Squeeze the transcript, absorb `witness`, and check that the next `bits` sampled bits
    /// are all zero. Zero difficulty accepts any witness without touching the transcript.
    fn check_witness(&mut self, bits: usize, witness: Self::Witness) -> bool {
        if bits == 0 {
            return true;
        }
        self.squeeze();
        self.witness_passes(bits, witness)
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
    use alloc::sync::Arc;
    use alloc::vec;
    use core::sync::atomic::{AtomicUsize, Ordering};

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

    /// Size of the transcript left pending before a grind in the rehash tests.
    const PENDING_LEN: usize = 1 << 12;

    /// Toy byte hasher that counts the calls whose input spans the pending transcript.
    ///
    /// FNV-1a folded through a SplitMix64 finalizer: input-sensitive enough to grind against.
    #[derive(Clone)]
    struct LongInputCounter(Arc<AtomicUsize>);

    impl CryptographicHasher<u8, [u8; 32]> for LongInputCounter {
        fn hash_iter<I>(&self, input: I) -> [u8; 32]
        where
            I: IntoIterator<Item = u8>,
        {
            let mut len = 0;
            let mut state = 0xcbf2_9ce4_8422_2325_u64;
            for byte in input {
                state = (state ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3);
                len += 1;
            }
            if len >= PENDING_LEN {
                self.0.fetch_add(1, Ordering::Relaxed);
            }
            let mut out = [0; 32];
            for (i, chunk) in out.as_chunks_mut::<8>().0.iter_mut().enumerate() {
                let mut z = state.wrapping_add((i as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15));
                z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                *chunk = (z ^ (z >> 31)).to_le_bytes();
            }
            out
        }
    }

    /// Grind over a long pending transcript and count how often it gets rehashed.
    ///
    /// The search starts from a squeezed copy, and the final check squeezes the real
    /// transcript, so the pending input is hashed at most twice however many candidates
    /// are tried. The verifier must accept the witness and stay in sync with the prover.
    fn assert_grind_rehashes_pending_input_at_most_twice<C>(
        wrap: impl FnOnce(HashChallenger<u8, LongInputCounter, 32>) -> C,
    ) where
        C: GrindingChallenger + CanSampleBits<usize> + Clone,
    {
        const BITS: usize = 8;
        let long_hashes = Arc::new(AtomicUsize::new(0));
        let inner =
            HashChallenger::new(vec![7; PENDING_LEN], LongInputCounter(long_hashes.clone()));
        let mut prover = wrap(inner);
        let mut verifier = prover.clone();

        let witness = prover.grind(BITS);
        assert!(long_hashes.load(Ordering::Relaxed) <= 2);

        assert!(verifier.check_witness(BITS, witness));
        assert_eq!(prover.sample_bits(20), verifier.sample_bits(20));
    }

    #[test]
    fn test_serializing_challenger32_grind_rehashes_pending_input_at_most_twice() {
        assert_grind_rehashes_pending_input_at_most_twice(
            SerializingChallenger32::<BabyBear, _>::new,
        );
    }

    #[test]
    fn test_serializing_challenger64_grind_rehashes_pending_input_at_most_twice() {
        assert_grind_rehashes_pending_input_at_most_twice(
            SerializingChallenger64::<Goldilocks, _>::new,
        );
    }

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

    #[test]
    #[should_panic = "requested bit count must fit within the field order"]
    fn test_serializing_challenger32_sample_bits_rejects_oversized_request() {
        // BabyBear order is ~2^30.9, so a 32-bit request must be rejected.
        // The bound is evaluated in u64, so the guard fires in every build profile.
        type F = BabyBear;
        let inner = Inner::new(vec![0, 1, 2, 3], ByteCountHasher);
        let mut challenger = SerializingChallenger32::<F, Inner>::new(inner);
        let _ = challenger.sample_bits(32);
    }

    #[test]
    #[should_panic = "requested bit count must fit within the field order"]
    fn test_serializing_challenger32_grind_rejects_oversized_request() {
        // Same guard, reached through the proof-of-work entry point.
        type F = BabyBear;
        let inner = Inner::new(vec![0, 1, 2, 3], ByteCountHasher);
        let mut challenger = SerializingChallenger32::<F, Inner>::new(inner);
        let _ = challenger.grind(32);
    }
}
