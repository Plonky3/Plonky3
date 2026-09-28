use alloc::vec;
use alloc::vec::Vec;
use core::array;

use p3_maybe_rayon::prelude::*;
use p3_symmetric::{CryptographicHasher, Hash, MerkleCap};

use crate::grinding_challenger::find_witness_by_cloning;
use crate::{ByteGrindingChallenger, CanFinalizeDigest, CanObserve, CanSample};

/// A generic challenger that uses a cryptographic hash function to generate challenges.
#[derive(Debug)]
pub struct HashChallenger<T, H, const OUT_LEN: usize>
where
    T: Clone,
    H: CryptographicHasher<T, [T; OUT_LEN]>,
{
    /// Buffer to store observed values before hashing.
    input_buffer: Vec<T>,
    /// Buffer to store hashed output values, which are consumed when sampling.
    output_buffer: Vec<T>,
    /// The cryptographic hash function used for generating challenges.
    hasher: H,
}

impl<T, H, const OUT_LEN: usize> Clone for HashChallenger<T, H, OUT_LEN>
where
    T: Clone,
    H: CryptographicHasher<T, [T; OUT_LEN]> + Clone,
{
    fn clone(&self) -> Self {
        Self {
            input_buffer: self.input_buffer.clone(),
            output_buffer: self.output_buffer.clone(),
            hasher: self.hasher.clone(),
        }
    }

    fn clone_from(&mut self, source: &Self) {
        self.input_buffer.clone_from(&source.input_buffer);
        self.output_buffer.clone_from(&source.output_buffer);
        self.hasher.clone_from(&source.hasher);
    }
}

impl<T, H, const OUT_LEN: usize> HashChallenger<T, H, OUT_LEN>
where
    T: Clone,
    H: CryptographicHasher<T, [T; OUT_LEN]>,
{
    pub const fn new(initial_state: Vec<T>, hasher: H) -> Self {
        Self {
            input_buffer: initial_state,
            output_buffer: vec![],
            hasher,
        }
    }

    fn flush(&mut self) {
        let inputs = self.input_buffer.drain(..);
        let output = self.hasher.hash_iter(inputs);

        // Chaining values.
        self.input_buffer.extend_from_slice(&output);
        self.output_buffer.clear();
        self.output_buffer.extend(output);
    }
}

impl<T, H, const OUT_LEN: usize> CanObserve<T> for HashChallenger<T, H, OUT_LEN>
where
    T: Clone,
    H: CryptographicHasher<T, [T; OUT_LEN]>,
{
    fn observe(&mut self, value: T) {
        // Any buffered output is now invalid.
        self.output_buffer.clear();

        self.input_buffer.push(value);
    }
    fn observe_slice(&mut self, values: &[T]) {
        if !values.is_empty() {
            self.output_buffer.clear();
            self.input_buffer.extend_from_slice(values);
        }
    }
}

impl<T, H, const N: usize, const OUT_LEN: usize> CanObserve<[T; N]>
    for HashChallenger<T, H, OUT_LEN>
where
    T: Clone,
    H: CryptographicHasher<T, [T; OUT_LEN]>,
{
    fn observe(&mut self, values: [T; N]) {
        self.output_buffer.clear();
        self.input_buffer.extend(values);
    }
}

impl<F, T, H, const N: usize, const OUT_LEN: usize> CanObserve<Hash<F, T, N>>
    for HashChallenger<T, H, OUT_LEN>
where
    T: Clone,
    H: CryptographicHasher<T, [T; OUT_LEN]>,
{
    fn observe(&mut self, values: Hash<F, T, N>) {
        for value in values {
            self.observe(value);
        }
    }
}

impl<F, T, H, const N: usize, const OUT_LEN: usize> CanObserve<&MerkleCap<F, [T; N]>>
    for HashChallenger<T, H, OUT_LEN>
where
    T: Clone,
    H: CryptographicHasher<T, [T; OUT_LEN]>,
{
    fn observe(&mut self, cap: &MerkleCap<F, [T; N]>) {
        for digest in cap.roots() {
            for value in digest {
                self.observe(value.clone());
            }
        }
    }
}

impl<F, T, H, const N: usize, const OUT_LEN: usize> CanObserve<MerkleCap<F, [T; N]>>
    for HashChallenger<T, H, OUT_LEN>
where
    T: Clone,
    H: CryptographicHasher<T, [T; OUT_LEN]>,
{
    fn observe(&mut self, cap: MerkleCap<F, [T; N]>) {
        self.observe(&cap);
    }
}

// for TrivialPcs
impl<T, H, const OUT_LEN: usize> CanObserve<Vec<Vec<T>>> for HashChallenger<T, H, OUT_LEN>
where
    T: Clone,
    H: CryptographicHasher<T, [T; OUT_LEN]>,
{
    fn observe(&mut self, valuess: Vec<Vec<T>>) {
        for values in valuess {
            for value in values {
                self.observe(value);
            }
        }
    }
}

impl<T, H, const OUT_LEN: usize> CanSample<T> for HashChallenger<T, H, OUT_LEN>
where
    T: Clone,
    H: CryptographicHasher<T, [T; OUT_LEN]>,
{
    fn sample(&mut self) -> T {
        if self.output_buffer.is_empty() {
            self.flush();
        }
        self.output_buffer
            .pop()
            .expect("Output buffer should be non-empty")
    }
    fn sample_into_slice(&mut self, mut values: &mut [T]) {
        while !values.is_empty() {
            if self.output_buffer.is_empty() {
                self.flush();
            }
            assert!(
                !self.output_buffer.is_empty(),
                "Output buffer should be non-empty"
            );
            let count = values.len().min(self.output_buffer.len());
            let start = self.output_buffer.len() - count;
            let (head, tail) = values.split_at_mut(count);
            for (slot, sample) in head.iter_mut().zip(self.output_buffer.drain(start..).rev()) {
                *slot = sample;
            }
            values = tail;
        }
    }

    fn sample_vec(&mut self, n: usize) -> Vec<T> {
        let mut result = Vec::with_capacity(n);
        while result.len() < n {
            if self.output_buffer.is_empty() {
                self.flush();
            }
            assert!(
                !self.output_buffer.is_empty(),
                "Output buffer should be non-empty"
            );
            let count = (n - result.len()).min(self.output_buffer.len());
            let start = self.output_buffer.len() - count;
            result.extend(self.output_buffer.drain(start..).rev());
        }
        result
    }
}

impl<H, const OUT_LEN: usize> ByteGrindingChallenger for HashChallenger<u8, H, OUT_LEN>
where
    H: CryptographicHasher<u8, [u8; OUT_LEN]> + Send + Sync,
{
    /// Hash the candidates in batches of equal-length messages.
    ///
    /// Observing a non-empty slice discards the buffered output, so every sample of candidate `c`
    /// comes from the one flush its first sample triggers, popped from the back of the digest:
    ///
    /// ```text
    ///     digest   = H(input_buffer || encode(c))
    ///     sample k = digest[OUT_LEN - 1 - k]          k = 0..S
    /// ```
    ///
    /// Every message shares the `input_buffer` prefix, so a worker writes that prefix into each
    /// slot of its batch once and only rewrites the encoded candidates between batches:
    ///
    /// ```text
    ///     messages: [ input_buffer | encode(c_0) | input_buffer | encode(c_1) | ... ]
    /// ```
    ///
    /// A batch is one call to [`CryptographicHasher::hash_many`], sized to the hasher's lanes.
    fn find_witness<const W: usize, const S: usize>(
        &self,
        num_candidates: u64,
        encode: impl Fn(u64) -> [u8; W] + Sync,
        accepts: impl Fn([u8; S]) -> bool + Sync,
    ) -> Option<u64> {
        // An empty encoding keeps the buffered output, and more samples than one digest holds reach
        // a second flush. Neither fits the single-flush layout above.
        if W == 0 || S > OUT_LEN {
            return find_witness_by_cloning(self, num_candidates, encode, accepts);
        }

        let prefix_len = self.input_buffer.len();
        let message_len = prefix_len + W;
        let batch_len = H::LANES.max(1);
        let num_batches = num_candidates.div_ceil(batch_len as u64);

        (0..num_batches)
            .into_par_iter()
            .map_init(
                || {
                    let mut messages = Vec::with_capacity(batch_len * message_len);
                    for _ in 0..batch_len {
                        messages.extend_from_slice(&self.input_buffer);
                        messages.extend_from_slice(&[0; W]);
                    }
                    (messages, vec![[0; OUT_LEN]; batch_len])
                },
                |(messages, digests), batch| {
                    let first = batch * batch_len as u64;
                    // The last batch stops at the end of the candidate range.
                    let count = (num_candidates - first).min(batch_len as u64) as usize;

                    for (offset, message) in messages
                        .chunks_exact_mut(message_len)
                        .take(count)
                        .enumerate()
                    {
                        message[prefix_len..].copy_from_slice(&encode(first + offset as u64));
                    }
                    self.hasher
                        .hash_many(&messages[..count * message_len], &mut digests[..count]);

                    // Scanning in candidate order keeps a serial search on the smallest pass.
                    digests[..count]
                        .iter()
                        .position(|digest| accepts(array::from_fn(|k| digest[OUT_LEN - 1 - k])))
                        .map(|offset| first + offset as u64)
                },
            )
            .find_map_any(|found| found)
    }
}

impl<T, H, const OUT_LEN: usize> CanFinalizeDigest for HashChallenger<T, H, OUT_LEN>
where
    T: Clone,
    H: CryptographicHasher<T, [T; OUT_LEN]>,
{
    type Digest = [T; OUT_LEN];

    fn finalize(mut self) -> [T; OUT_LEN] {
        // Unconditionally flush: hash the input buffer and produce the
        // digest from the resulting output.
        //
        // Note: unlike sponge-based challengers, observe never auto-flushes
        // here, so the first sample always changes the chaining values and
        // thus the digest.
        self.flush();
        core::array::from_fn(|i| self.output_buffer[i].clone())
    }
}

#[cfg(test)]
mod tests {
    use p3_blake3::Blake3;
    use p3_field::PrimeCharacteristicRing;
    use p3_goldilocks::Goldilocks;
    use p3_keccak::Keccak256Hash;
    use p3_maybe_rayon::PARALLEL_ENABLED;
    use p3_sha256::Sha256;

    use super::*;

    const OUT_LEN: usize = 2;
    type F = Goldilocks;

    #[derive(Clone)]
    struct TestHasher {}

    impl CryptographicHasher<F, [F; OUT_LEN]> for TestHasher {
        /// A very simple hash iterator. From an input of type `IntoIterator<Item = Goldilocks>`,
        /// it outputs the sum of its elements and its length (as a field element).
        fn hash_iter<I>(&self, input: I) -> [F; OUT_LEN]
        where
            I: IntoIterator<Item = F>,
        {
            let (sum, len) = input
                .into_iter()
                .fold((F::ZERO, 0_usize), |(acc_sum, acc_len), f| {
                    (acc_sum + f, acc_len + 1)
                });
            [sum, F::from_usize(len)]
        }

        /// A very simple slice hash iterator. From an input of type `IntoIterator<Item = &'a [Goldilocks]>`,
        /// it outputs the sum of its elements and its length (as a field element).
        fn hash_iter_slices<'a, I>(&self, input: I) -> [F; OUT_LEN]
        where
            I: IntoIterator<Item = &'a [F]>,
            F: 'a,
        {
            let (sum, len) = input
                .into_iter()
                .fold((F::ZERO, 0_usize), |(acc_sum, acc_len), n| {
                    (
                        acc_sum + n.iter().fold(F::ZERO, |acc, f| acc + *f),
                        acc_len + n.len(),
                    )
                });
            [sum, F::from_usize(len)]
        }
    }

    #[test]
    fn bulk_operations_preserve_the_scalar_stream_at_boundaries() {
        for prefix in [0, 1, 2, 3] {
            for len in [0, 1, 2, 3, 7, 19] {
                let mut bulk = HashChallenger::new(vec![F::ONE; 17], TestHasher {});
                let mut scalar = bulk.clone();
                for _ in 0..prefix {
                    assert_eq!(bulk.sample(), scalar.sample());
                }
                bulk.observe_slice(&[] as &[F]);
                let mut samples = vec![F::ZERO; len];
                bulk.sample_into_slice(&mut samples);
                let expected: Vec<_> = (0..len).map(|_| scalar.sample()).collect();
                assert_eq!(samples, expected);
                assert_eq!(bulk.sample(), scalar.sample());
                let observations = [F::from_u8(3), F::from_u8(9)];
                bulk.observe_slice(&observations);
                for x in observations {
                    scalar.observe(x);
                }
                assert_eq!(
                    bulk.sample_vec(len),
                    (0..len).map(|_| scalar.sample()).collect::<Vec<_>>()
                );
                assert_eq!(bulk.sample(), scalar.sample());
            }
        }
    }

    #[test]
    fn clone_from_preserves_stream_and_reuses_buffers() {
        let mut source = HashChallenger::new(vec![F::ONE; 17], TestHasher {});
        let _: F = source.sample();
        let mut worker = HashChallenger::new(vec![F::ZERO; 64], TestHasher {});
        worker.flush();
        let input_ptr = worker.input_buffer.as_ptr();
        let output_ptr = worker.output_buffer.as_ptr();
        for i in 0..8 {
            worker.clone_from(&source);
            let mut expected = source.clone();
            assert_eq!(worker.input_buffer.as_ptr(), input_ptr);
            assert_eq!(worker.output_buffer.as_ptr(), output_ptr);
            for _ in 0..7 {
                assert_eq!(worker.sample(), expected.sample());
            }
            assert_eq!(worker.output_buffer.as_ptr(), output_ptr);
            let observation = F::from_usize(i);
            worker.observe(observation);
            expected.observe(observation);
            for _ in 0..7 {
                assert_eq!(worker.sample(), expected.sample());
            }
            source.observe(observation);
            let _: F = source.sample();
        }
    }

    #[test]
    fn test_hash_challenger() {
        let initial_state = (1..11_u8).map(F::from_u8).collect::<Vec<_>>();
        let test_hasher = TestHasher {};
        let mut hash_challenger = HashChallenger::new(initial_state.clone(), test_hasher);

        assert_eq!(hash_challenger.input_buffer, initial_state);
        assert_eq!(hash_challenger.output_buffer, vec![]);

        hash_challenger.flush();

        let expected_sum = F::from_u8(55);
        let expected_len = F::from_u8(10);
        assert_eq!(
            hash_challenger.input_buffer,
            vec![expected_sum, expected_len]
        );
        assert_eq!(
            hash_challenger.output_buffer,
            vec![expected_sum, expected_len]
        );

        let new_element = F::from_u8(11);
        hash_challenger.observe(new_element);
        assert_eq!(
            hash_challenger.input_buffer,
            vec![expected_sum, expected_len, new_element]
        );
        assert_eq!(hash_challenger.output_buffer, vec![]);

        let new_expected_len = 3;
        let new_expected_sum = 76;

        let new_element = hash_challenger.sample();
        assert_eq!(new_element, F::from_u8(new_expected_len));
        assert_eq!(
            hash_challenger.output_buffer,
            [F::from_u8(new_expected_sum)]
        );
    }

    #[test]
    fn test_hash_challenger_flush() {
        let initial_state = (1..11_u8).map(F::from_u8).collect::<Vec<_>>();
        let test_hasher = TestHasher {};
        let mut hash_challenger = HashChallenger::new(initial_state, test_hasher);

        // Sample twice to ensure flush happens
        let first_sample = hash_challenger.sample();

        let second_sample = hash_challenger.sample();

        // Verify that the first sample is the length of 1..11, (i.e. 10).
        assert_eq!(first_sample, F::from_u8(10));
        //  Verify that the second sample is the sum of numbers from 1 to 10 (i.e. 55)
        assert_eq!(second_sample, F::from_u8(55));

        // Verify that the output buffer is now empty
        assert!(hash_challenger.output_buffer.is_empty());
    }

    #[test]
    fn test_observe_single_value() {
        let test_hasher = TestHasher {};
        // Initial state non-empty
        let mut hash_challenger = HashChallenger::new(vec![F::from_u8(123)], test_hasher);

        // Observe a single value
        let value = F::from_u8(42);
        hash_challenger.observe(value);

        // Check that the input buffer contains the initial and observed values
        assert_eq!(
            hash_challenger.input_buffer,
            vec![F::from_u8(123), F::from_u8(42)]
        );
        // Check that the output buffer is empty (clears after observation)
        assert!(hash_challenger.output_buffer.is_empty());
    }

    #[test]
    fn test_observe_array() {
        let test_hasher = TestHasher {};
        // Initial state non-empty
        let mut hash_challenger = HashChallenger::new(vec![F::from_u8(123)], test_hasher);

        // Observe an array of values
        let values = [F::from_u8(1), F::from_u8(2), F::from_u8(3)];
        hash_challenger.observe(values);

        // Check that the input buffer contains the values
        assert_eq!(
            hash_challenger.input_buffer,
            vec![F::from_u8(123), F::from_u8(1), F::from_u8(2), F::from_u8(3)]
        );
        // Check that the output buffer is empty (clears after observation)
        assert!(hash_challenger.output_buffer.is_empty());
    }

    #[test]
    fn test_observe_hash_cap_and_nested_vec() {
        let test_hasher = TestHasher {};

        // Observing a Hash absorbs its digest words in order.
        let mut from_hash = HashChallenger::new(vec![], test_hasher.clone());
        from_hash.observe(Hash::<F, F, 3>::from([
            F::from_u8(1),
            F::from_u8(2),
            F::from_u8(3),
        ]));

        let mut from_array = HashChallenger::new(vec![], test_hasher.clone());
        from_array.observe([F::from_u8(1), F::from_u8(2), F::from_u8(3)]);
        assert_eq!(from_hash.input_buffer, from_array.input_buffer);

        // A MerkleCap absorbs every word of every root, by value and by reference.
        let cap = MerkleCap::<F, [F; 2]>::new(vec![
            [F::from_u8(4), F::from_u8(5)],
            [F::from_u8(6), F::from_u8(7)],
        ]);
        let flat = vec![F::from_u8(4), F::from_u8(5), F::from_u8(6), F::from_u8(7)];

        let mut from_cap_ref = HashChallenger::new(vec![], test_hasher.clone());
        from_cap_ref.observe(&cap);
        assert_eq!(from_cap_ref.input_buffer, flat);

        let mut from_cap_owned = HashChallenger::new(vec![], test_hasher.clone());
        from_cap_owned.observe(cap);
        assert_eq!(from_cap_owned.input_buffer, flat);

        // Vec<Vec<T>> is flattened in order.
        let mut from_nested = HashChallenger::new(vec![], test_hasher);
        from_nested.observe(vec![
            vec![F::from_u8(8), F::from_u8(9)],
            vec![F::from_u8(10)],
        ]);
        assert_eq!(
            from_nested.input_buffer,
            vec![F::from_u8(8), F::from_u8(9), F::from_u8(10)]
        );
    }

    #[test]
    fn test_sample_output_buffer() {
        let test_hasher = TestHasher {};
        let initial_state = vec![F::from_u8(5), F::from_u8(10)];
        let mut hash_challenger = HashChallenger::new(initial_state, test_hasher);

        let sample = hash_challenger.sample();
        // Verify that the sample is the length of the initial state
        assert_eq!(sample, F::from_u8(2));
        // Check that the output buffer contains the sum of the initial state
        assert_eq!(hash_challenger.output_buffer, vec![F::from_u8(15)]);
    }

    #[test]
    fn test_flush_empty_buffer() {
        let test_hasher = TestHasher {};
        let mut hash_challenger = HashChallenger::new(vec![], test_hasher);

        // Flush empty buffer
        hash_challenger.flush();

        // Check that the input and output buffers contain the sum and length of the empty buffer
        assert_eq!(hash_challenger.input_buffer, vec![F::ZERO, F::ZERO]);
        assert_eq!(hash_challenger.output_buffer, vec![F::ZERO, F::ZERO]);
    }

    #[test]
    fn test_flush_with_data() {
        let test_hasher = TestHasher {};
        // Initial state non-empty
        let initial_state = vec![F::from_u8(1), F::from_u8(2)];
        let mut hash_challenger = HashChallenger::new(initial_state, test_hasher);

        hash_challenger.flush();

        // Check that the input buffer contains the sum and length of the initial state
        assert_eq!(
            hash_challenger.input_buffer,
            vec![F::from_u8(3), F::from_u8(2)]
        );
        // Check that the output buffer contains the sum and length of the initial state
        assert_eq!(
            hash_challenger.output_buffer,
            vec![F::from_u8(3), F::from_u8(2)]
        );
    }

    #[test]
    fn test_sample_after_observe() {
        let test_hasher = TestHasher {};
        let initial_state = vec![F::from_u8(1), F::from_u8(2)];
        let mut hash_challenger = HashChallenger::new(initial_state, test_hasher);

        // Observe will clear the output buffer
        hash_challenger.observe(F::from_u8(3));

        // Verify that the output buffer is empty
        assert!(hash_challenger.output_buffer.is_empty());

        // Verify the new value is in the input buffer
        assert_eq!(
            hash_challenger.input_buffer,
            vec![F::from_u8(1), F::from_u8(2), F::from_u8(3)]
        );

        let sample = hash_challenger.sample();

        // Length of initial state + observed value
        assert_eq!(sample, F::from_u8(3));
    }

    #[test]
    fn test_sample_with_non_empty_output_buffer() {
        let test_hasher = TestHasher {};
        let mut hash_challenger = HashChallenger::new(vec![], test_hasher);

        hash_challenger.output_buffer = vec![F::from_u8(42), F::from_u8(24)];

        let sample = hash_challenger.sample();

        // Sample will pop the last element from the output buffer
        assert_eq!(sample, F::from_u8(24));

        // Check that the output buffer is now one element shorter
        assert_eq!(hash_challenger.output_buffer, vec![F::from_u8(42)]);
    }

    #[test]
    fn test_finalize() {
        let new_chal = || HashChallenger::new(vec![F::from_u8(1), F::from_u8(2)], TestHasher {});

        // Deterministic: same observations produce same digest.
        let mut h1 = new_chal();
        let mut h2 = new_chal();
        h1.observe(F::from_u8(42));
        h2.observe(F::from_u8(42));
        assert_eq!(h1.finalize(), h2.finalize());

        // Different observations produce different digests.
        let mut h1 = new_chal();
        let mut h2 = new_chal();
        h1.observe(F::from_u8(1));
        h2.observe(F::from_u8(2));
        assert_ne!(h1.finalize(), h2.finalize());
    }

    /// Document how sampling interacts with finalize.
    ///
    /// Sampling pops from the output buffer. When the buffer is exhausted,
    /// the next sample triggers a flush (hash), which changes the chaining
    /// values in the input buffer. Finalize always flushes, so the digest
    /// changes whenever a sample triggered a flush — i.e. every OUT_LEN
    /// samples.
    #[test]
    fn test_finalize_sample_interaction() {
        let digest = |n_samples: usize| {
            let mut c = HashChallenger::new(vec![F::from_u8(1), F::from_u8(2)], TestHasher {});
            c.observe(F::from_u8(42));
            for _ in 0..n_samples {
                let _: F = c.sample();
            }
            c.finalize()
        };

        // The first sample triggers a flush (output buffer was empty after
        // observe), changing the chaining values. Finalize's flush then
        // hashes different input than the 0-sample case.
        assert_ne!(digest(0), digest(1));

        // Samples 1 through OUT_LEN come from the same flush output.
        // They don't trigger another flush, so the chaining values
        // (and thus the digest) are identical.
        assert_eq!(digest(1), digest(OUT_LEN));

        // The (OUT_LEN+1)-th sample exhausts the output buffer and
        // triggers a fresh flush, changing the chaining values again.
        assert_ne!(digest(OUT_LEN), digest(OUT_LEN + 1));

        // Within the second batch, the digest is again stable.
        assert_eq!(digest(OUT_LEN + 1), digest(2 * OUT_LEN));
    }

    /// Keccak-256 reporting `L` lanes, so batches of any width run on every target.
    #[derive(Clone)]
    struct WithLanes<const L: usize>;

    impl<const L: usize> CryptographicHasher<u8, [u8; 32]> for WithLanes<L> {
        const LANES: usize = L;

        fn hash_iter<I>(&self, input: I) -> [u8; 32]
        where
            I: IntoIterator<Item = u8>,
        {
            Keccak256Hash.hash_iter(input)
        }

        fn hash_many(&self, input: &[u8], out: &mut [[u8; 32]]) {
            Keccak256Hash.hash_many(input, out);
        }
    }

    /// Byte challengers whose pending input puts the candidate across the block boundaries of the
    /// batched hashers.
    ///
    /// Each comes with a squeezed copy, the state the serializing challengers search from: one
    /// sampled byte leaves the 32-byte chaining value pending and the rest of the digest buffered
    /// for observing the candidate to discard.
    fn byte_transcripts<H>(hasher: &H) -> Vec<HashChallenger<u8, H, 32>>
    where
        H: CryptographicHasher<u8, [u8; 32]>,
    {
        let mut transcripts = Vec::new();
        for len in [0, 1, 32, 52, 56, 60, 64, 124, 128, 132, 136, 1020] {
            let fresh =
                HashChallenger::new((0..len).map(|i| (i * 7) as u8).collect(), hasher.clone());
            let mut squeezed = fresh.clone();
            let _: u8 = squeezed.sample();
            transcripts.push(fresh);
            transcripts.push(squeezed);
        }
        transcripts
    }

    /// Check the batched search against a fresh clone at every candidate of a few ranges.
    ///
    /// The ranges leave one candidate and one short of a full batch in the last batch.
    ///
    /// The target bytes pin a single candidate, so the search has to return that one:
    /// any other candidate would have sampled different bytes. A target sampled only by a
    /// candidate in the batch past the range must not be found at all.
    fn assert_find_witness_matches_clones<H, const W: usize, const S: usize>(
        challenger: &HashChallenger<u8, H, 32>,
    ) where
        H: CryptographicHasher<u8, [u8; 32]> + Send + Sync,
    {
        let lanes = H::LANES.max(1) as u64;
        // Distinct candidates encode to distinct bytes.
        let encode =
            |c: u64| -> [u8; W] { array::from_fn(|i| (c >> (8 * (i % 8))) as u8 ^ i as u8) };

        for num_candidates in [3 * lanes + 1, 4 * lanes - 1] {
            let expected: Vec<[u8; S]> = (0..num_candidates + lanes)
                .map(|c| {
                    let mut clone = challenger.clone();
                    clone.observe_slice(&encode(c));
                    clone.sample_array()
                })
                .collect();
            let in_range = &expected[..num_candidates as usize];

            for target in &expected {
                let found = challenger.find_witness(num_candidates, encode, |s| s == *target);
                if !in_range.contains(target) {
                    assert_eq!(found, None);
                    continue;
                }
                let found = found.expect("the candidate sampling the target must pass");
                assert_eq!(in_range[found as usize], *target);
                // A serial search returns the smallest candidate sampling the target.
                if !PARALLEL_ENABLED {
                    assert_eq!(
                        in_range.iter().position(|s| s == target),
                        Some(found as usize)
                    );
                }
            }

            assert_eq!(
                challenger.find_witness(num_candidates, encode, |_: [u8; S]| false),
                None
            );
        }

        // An empty range has no candidate to pass, however permissive the check.
        assert_eq!(challenger.find_witness(0, encode, |_: [u8; S]| true), None);
    }

    fn assert_find_witness_matches_clones_for<H>(hasher: &H)
    where
        H: CryptographicHasher<u8, [u8; 32]> + Send + Sync,
    {
        for challenger in byte_transcripts(hasher) {
            // The encodings and sample widths of the 32- and 64-bit serializing challengers.
            assert_find_witness_matches_clones::<H, 4, 4>(&challenger);
            assert_find_witness_matches_clones::<H, 8, 8>(&challenger);
            // A sample as wide as the digest still comes from a single flush.
            assert_find_witness_matches_clones::<H, 4, 32>(&challenger);
            // Shapes the single-flush layout cannot express fall back to the default search.
            assert_find_witness_matches_clones::<H, 0, 4>(&challenger);
            assert_find_witness_matches_clones::<H, 4, 33>(&challenger);
            assert_find_witness_matches_clones::<H, 4, 40>(&challenger);
        }
    }

    #[test]
    fn find_witness_matches_clones_at_any_lane_count() {
        assert_find_witness_matches_clones_for(&WithLanes::<1>);
        assert_find_witness_matches_clones_for(&WithLanes::<3>);
        assert_find_witness_matches_clones_for(&WithLanes::<4>);
    }

    #[test]
    fn find_witness_matches_clones_for_byte_hashers() {
        assert_find_witness_matches_clones_for(&Keccak256Hash);
        assert_find_witness_matches_clones_for(&Sha256);
        assert_find_witness_matches_clones_for(&Blake3);
    }

    #[test]
    fn test_output_buffer_cleared_on_observe() {
        let test_hasher = TestHasher {};
        let mut hash_challenger = HashChallenger::new(vec![], test_hasher);

        // Populate artificially the output buffer
        hash_challenger.output_buffer.push(F::from_u8(42));

        // Ensure the output buffer is populated
        assert!(!hash_challenger.output_buffer.is_empty());

        // Observe a new value
        hash_challenger.observe(F::from_u8(3));

        // Verify that the output buffer is cleared after observing
        assert!(hash_challenger.output_buffer.is_empty());
    }

    #[test]
    fn test_observe_empty_array_clears_output_buffer() {
        let test_hasher = TestHasher {};
        let initial_state = vec![F::from_u8(1), F::from_u8(2)];
        let mut hash_challenger = HashChallenger::new(initial_state.clone(), test_hasher);

        // Populate artificially the output buffer
        hash_challenger.output_buffer.push(F::from_u8(42));
        assert!(!hash_challenger.output_buffer.is_empty());

        // Observing an empty array still invalidates the output buffer
        let values: [F; 0] = [];
        hash_challenger.observe(values);

        assert!(hash_challenger.output_buffer.is_empty());
        // Input buffer is unchanged because no values were appended
        assert_eq!(hash_challenger.input_buffer, initial_state);
    }
}
