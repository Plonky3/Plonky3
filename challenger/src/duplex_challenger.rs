use alloc::vec;
use alloc::vec::Vec;

use p3_field::{BasedVectorSpace, Field, PrimeField64};
use p3_symmetric::{CryptographicPermutation, Hash};

use crate::{CanObserve, CanSample, CanSampleBits, FieldChallenger};

#[derive(Clone, Debug)]
pub struct DuplexChallenger<F, P, const WIDTH: usize, const RATE: usize>
where
    F: Clone,
    P: CryptographicPermutation<[F; WIDTH]>,
{
    pub sponge_state: [F; WIDTH],
    pub input_buffer: Vec<F>,
    pub output_buffer: Vec<F>,
    pub permutation: P,
}

impl<F, P, const WIDTH: usize, const RATE: usize> DuplexChallenger<F, P, WIDTH, RATE>
where
    F: Copy,
    P: CryptographicPermutation<[F; WIDTH]>,
{
    pub fn new(permutation: P) -> Self
    where
        F: Default,
    {
        Self {
            sponge_state: [F::default(); WIDTH],
            input_buffer: vec![],
            output_buffer: vec![],
            permutation,
        }
    }

    fn duplexing(&mut self) {
        assert!(self.input_buffer.len() <= RATE);

        // Overwrite the first r elements with the inputs.
        for (i, val) in self.input_buffer.drain(..).enumerate() {
            self.sponge_state[i] = val;
        }

        // Apply the permutation.
        self.permutation.permute_mut(&mut self.sponge_state);

        self.output_buffer.clear();
        self.output_buffer.extend(&self.sponge_state[..RATE]);
    }
}

impl<F, P, const WIDTH: usize, const RATE: usize> FieldChallenger<F>
    for DuplexChallenger<F, P, WIDTH, RATE>
where
    F: PrimeField64,
    P: CryptographicPermutation<[F; WIDTH]>,
{
}

impl<F, P, const WIDTH: usize, const RATE: usize> CanObserve<F>
    for DuplexChallenger<F, P, WIDTH, RATE>
where
    F: Copy,
    P: CryptographicPermutation<[F; WIDTH]>,
{
    fn observe(&mut self, value: F) {
        // Any buffered output is now invalid.
        self.output_buffer.clear();

        self.input_buffer.push(value);

        if self.input_buffer.len() == RATE {
            self.duplexing();
        }
    }
}

impl<F, P, const N: usize, const WIDTH: usize, const RATE: usize> CanObserve<[F; N]>
    for DuplexChallenger<F, P, WIDTH, RATE>
where
    F: Copy,
    P: CryptographicPermutation<[F; WIDTH]>,
{
    fn observe(&mut self, values: [F; N]) {
        for value in values {
            self.observe(value);
        }
    }
}

impl<F, P, const N: usize, const WIDTH: usize, const RATE: usize> CanObserve<Hash<F, F, N>>
    for DuplexChallenger<F, P, WIDTH, RATE>
where
    F: Copy,
    P: CryptographicPermutation<[F; WIDTH]>,
{
    fn observe(&mut self, values: Hash<F, F, N>) {
        for value in values {
            self.observe(value);
        }
    }
}

// for TrivialPcs
impl<F, P, const WIDTH: usize, const RATE: usize> CanObserve<Vec<Vec<F>>>
    for DuplexChallenger<F, P, WIDTH, RATE>
where
    F: Copy,
    P: CryptographicPermutation<[F; WIDTH]>,
{
    fn observe(&mut self, valuess: Vec<Vec<F>>) {
        for values in valuess {
            for value in values {
                self.observe(value);
            }
        }
    }
}

impl<F, EF, P, const WIDTH: usize, const RATE: usize> CanSample<EF>
    for DuplexChallenger<F, P, WIDTH, RATE>
where
    F: Field,
    EF: BasedVectorSpace<F>,
    P: CryptographicPermutation<[F; WIDTH]>,
{
    fn sample(&mut self) -> EF {
        EF::from_basis_coefficients_fn(|_| {
            // If we have buffered inputs, we must perform a duplexing so that the challenge will
            // reflect them. Or if we've run out of outputs, we must perform a duplexing to get more.
            if !self.input_buffer.is_empty() || self.output_buffer.is_empty() {
                self.duplexing();
            }

            self.output_buffer
                .pop()
                .expect("Output buffer should be non-empty")
        })
    }
}

impl<F, P, const WIDTH: usize, const RATE: usize> CanSampleBits<usize>
    for DuplexChallenger<F, P, WIDTH, RATE>
where
    F: PrimeField64,
    P: CryptographicPermutation<[F; WIDTH]>,
{
    fn sample_bits(&mut self, bits: usize) -> usize {
        assert!(bits < (usize::BITS as usize));
        assert!((1 << bits) < F::ORDER_U64);
        let rand_f: F = self.sample();
        let rand_usize = rand_f.as_canonical_u64() as usize;
        rand_usize & ((1 << bits) - 1)
    }
}

#[cfg(test)]
mod tests {
    use core::iter;

    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;
    use p3_goldilocks::Goldilocks;
    use p3_symmetric::Permutation;

    use super::*;
    use crate::grinding_challenger::GrindingChallenger;

    const WIDTH: usize = 24;
    const RATE: usize = 16;

    type G = Goldilocks;
    type BB = BabyBear;

    #[derive(Clone)]
    struct TestPermutation {}

    impl<F: Clone> Permutation<[F; WIDTH]> for TestPermutation {
        fn permute_mut(&self, input: &mut [F; WIDTH]) {
            input.reverse()
        }
    }

    impl<F: Clone> CryptographicPermutation<[F; WIDTH]> for TestPermutation {}

    #[test]
    fn test_duplex_challenger() {
        type Chal = DuplexChallenger<G, TestPermutation, WIDTH, RATE>;
        let permutation = TestPermutation {};
        let mut duplex_challenger = DuplexChallenger::new(permutation);

        // Observe 12 elements.
        (0..12).for_each(|element| duplex_challenger.observe(G::from_u8(element as u8)));

        let state_after_duplexing: Vec<_> = iter::repeat_n(G::ZERO, 12)
            .chain((0..12).map(G::from_u8).rev())
            .collect();

        let expected_samples: Vec<G> = state_after_duplexing[..16].iter().copied().rev().collect();
        let samples = <Chal as CanSample<G>>::sample_vec(&mut duplex_challenger, 16);
        assert_eq!(samples, expected_samples);
    }

    #[test]
    #[should_panic]
    fn test_duplex_challenger_sample_bits_security() {
        type GoldilocksChal = DuplexChallenger<G, TestPermutation, WIDTH, RATE>;
        let permutation = TestPermutation {};
        let mut duplex_challenger = GoldilocksChal::new(permutation);

        for _ in 0..100 {
            assert!(duplex_challenger.sample_bits(129) < 4);
        }
    }

    #[test]
    #[should_panic]
    fn test_duplex_challenger_sample_bits_security_small_field() {
        type BabyBearChal = DuplexChallenger<BB, TestPermutation, WIDTH, RATE>;
        let permutation = TestPermutation {};
        let mut duplex_challenger = BabyBearChal::new(permutation);

        for _ in 0..100 {
            assert!(duplex_challenger.sample_bits(40) < 1 << 31);
        }
    }

    #[test]
    #[should_panic]
    fn test_duplex_challenger_grind_security() {
        type GoldilocksChal = DuplexChallenger<G, TestPermutation, WIDTH, RATE>;
        let permutation = TestPermutation {};
        let mut duplex_challenger = GoldilocksChal::new(permutation);

        // This should cause sample_bits (and hence grind and check_witness) to
        // panic. If bit sizes were not constrained correctly inside the
        // challenger, (1 << too_many_bits) would loop around, incorrectly
        // grinding and accepting a 1-bit PoW.
        let too_many_bits = usize::BITS as usize;

        let witness = duplex_challenger.grind(too_many_bits);
        assert!(duplex_challenger.check_witness(too_many_bits, witness));
    }

    #[test]
    fn test_observe_single_value() {
        let mut chal = DuplexChallenger::<G, TestPermutation, WIDTH, RATE>::new(TestPermutation {});
        chal.observe(G::from_u8(42));
        assert_eq!(chal.input_buffer, vec![G::from_u8(42)]);
        assert!(chal.output_buffer.is_empty());
    }

    #[test]
    fn test_observe_array_of_values() {
        let mut chal = DuplexChallenger::<G, TestPermutation, WIDTH, RATE>::new(TestPermutation {});
        chal.observe([G::from_u8(1), G::from_u8(2), G::from_u8(3)]);
        assert_eq!(
            chal.input_buffer,
            vec![G::from_u8(1), G::from_u8(2), G::from_u8(3)]
        );
        assert!(chal.output_buffer.is_empty());
    }

    #[test]
    fn test_observe_hash_array() {
        let mut chal = DuplexChallenger::<G, TestPermutation, WIDTH, RATE>::new(TestPermutation {});
        let hash = Hash::<G, G, 4>::from([G::from_u8(10); 4]);
        chal.observe(hash);
        assert_eq!(chal.input_buffer, vec![G::from_u8(10); 4]);
    }

    #[test]
    fn test_observe_nested_vecs() {
        let mut chal = DuplexChallenger::<G, TestPermutation, WIDTH, RATE>::new(TestPermutation {});
        chal.observe(vec![
            vec![G::from_u8(1), G::from_u8(2)],
            vec![G::from_u8(3)],
        ]);
        assert_eq!(
            chal.input_buffer,
            vec![G::from_u8(1), G::from_u8(2), G::from_u8(3)]
        );
    }

    #[test]
    fn test_sample_triggers_duplex() {
        let mut chal = DuplexChallenger::<G, TestPermutation, WIDTH, RATE>::new(TestPermutation {});
        chal.observe(G::from_u8(5));
        assert!(chal.output_buffer.is_empty());
        let _sample: G = chal.sample();
        assert!(!chal.output_buffer.is_empty());
    }

    #[test]
    fn test_sample_multiple_extension_field() {
        use p3_field::extension::BinomialExtensionField;
        type EF = BinomialExtensionField<G, 2>;
        let mut chal = DuplexChallenger::<G, TestPermutation, WIDTH, RATE>::new(TestPermutation {});

        chal.observe(G::from_u8(1));
        chal.observe(G::from_u8(2));
        let _: EF = chal.sample();
        let _: EF = chal.sample();
    }

    #[test]
    fn test_sample_bits_within_bounds() {
        let mut chal = DuplexChallenger::<G, TestPermutation, WIDTH, RATE>::new(TestPermutation {});
        for i in 0..RATE {
            chal.observe(G::from_u8(i as u8));
        }

        // With RATE=16 and input = 0..15, the reversed sponge_state will be 15..0
        // The first RATE elements of that, i.e. output_buffer, are 15..0
        // sample_bits(3) will sample the last of those: G::from_u8(0)

        let bits = 3;
        let value = chal.sample_bits(bits);
        let expected = G::ZERO.as_canonical_u64() as usize & ((1 << bits) - 1);
        assert_eq!(value, expected);
    }

    #[test]
    fn test_sample_bits_trigger_duplex_when_empty() {
        let mut chal = DuplexChallenger::<G, TestPermutation, WIDTH, RATE>::new(TestPermutation {});
        // Force empty buffers
        assert_eq!(chal.input_buffer.len(), 0);
        assert_eq!(chal.output_buffer.len(), 0);

        // sampling bits should not panic, should return 0
        let bits = 2;
        let sample = chal.sample_bits(bits);
        let expected = G::ZERO.as_canonical_u64() as usize & ((1 << bits) - 1);
        assert_eq!(sample, expected);
    }

    #[test]
    fn test_output_buffer_pops_correctly() {
        let mut chal = DuplexChallenger::<G, TestPermutation, WIDTH, RATE>::new(TestPermutation {});

        // Observe RATE elements, causing a duplexing
        for i in 0..RATE {
            chal.observe(G::from_u8(i as u8));
        }

        // we expect the output buffer to be reversed
        let expected = [
            G::from_u8(0),
            G::from_u8(0),
            G::from_u8(0),
            G::from_u8(0),
            G::from_u8(0),
            G::from_u8(0),
            G::from_u8(0),
            G::from_u8(0),
            G::from_u8(15),
            G::from_u8(14),
            G::from_u8(13),
            G::from_u8(12),
            G::from_u8(11),
            G::from_u8(10),
            G::from_u8(9),
            G::from_u8(8),
        ]
        .to_vec();

        assert_eq!(chal.output_buffer, expected);

        let first: G = chal.sample();
        let second: G = chal.sample();

        // sampling pops from end of output buffer
        assert_eq!(first, G::from_u8(8));
        assert_eq!(second, G::from_u8(9));
    }

    #[test]
    fn test_duplexing_only_when_needed() {
        let mut chal = DuplexChallenger::<G, TestPermutation, WIDTH, RATE>::new(TestPermutation {});
        chal.output_buffer = vec![G::from_u8(10), G::from_u8(20)];

        // Sample should not call duplexing; just pop from the buffer
        let sample: G = chal.sample();
        assert_eq!(sample, G::from_u8(20));
        assert_eq!(chal.output_buffer, vec![G::from_u8(10)]);
    }

    #[test]
    fn test_flush_when_input_full() {
        let mut chal = DuplexChallenger::<G, TestPermutation, WIDTH, RATE>::new(TestPermutation {});

        // Observe RATE elements, causing a duplexing
        for i in 0..RATE {
            chal.observe(G::from_u8(i as u8));
        }

        // We expect the output buffer to be reversed
        let expected_output = [
            G::from_u8(0),
            G::from_u8(0),
            G::from_u8(0),
            G::from_u8(0),
            G::from_u8(0),
            G::from_u8(0),
            G::from_u8(0),
            G::from_u8(0),
            G::from_u8(15),
            G::from_u8(14),
            G::from_u8(13),
            G::from_u8(12),
            G::from_u8(11),
            G::from_u8(10),
            G::from_u8(9),
            G::from_u8(8),
        ]
        .to_vec();

        // Input buffer should be drained after duplexing
        assert!(chal.input_buffer.is_empty());

        // Output buffer should match expected state from duplexing
        assert_eq!(chal.output_buffer, expected_output);
    }
}
