use alloc::vec;
use alloc::vec::Vec;

use p3_field::{ExtensionField, Field, PrimeField64};
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
        self.output_buffer.extend(self.sponge_state);
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
    EF: ExtensionField<F>,
    P: CryptographicPermutation<[F; WIDTH]>,
{
    fn sample(&mut self) -> EF {
        EF::from_base_fn(|_| {
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
        debug_assert!(bits < (usize::BITS as usize));
        debug_assert!((1 << bits) < F::ORDER_U64);
        let rand_f: F = self.sample();
        let rand_usize = rand_f.as_canonical_u64() as usize;
        rand_usize & ((1 << bits) - 1)
    }
}

#[cfg(test)]
mod tests {
    use p3_field::AbstractField;
    use p3_goldilocks::Goldilocks;
    use p3_symmetric::Permutation;

    use super::*;

    const WIDTH: usize = 24;
    const RATE: usize = 16;

    type TestArray = [F; WIDTH];
    type F = Goldilocks;

    #[derive(Clone)]
    struct TestPermutation {}

    impl Permutation<TestArray> for TestPermutation {
        fn permute_mut(&self, input: &mut TestArray) {
            input.reverse()
        }
    }

    impl CryptographicPermutation<TestArray> for TestPermutation {}

    #[test]
    fn test_duplex_challenger_randomized() {
        let permutation = TestPermutation {};
        let mut duplex_challenger = DuplexChallenger::new(permutation);

        // Observe WIDTH / 2 elements.
        (0..WIDTH / 2)
            .for_each(|element| duplex_challenger.observe(F::from_canonical_u8(element as u8)));

        let should_be_sponge_state: [F; WIDTH] = [
            vec![F::zero(); WIDTH / 2],
            (0..WIDTH / 2)
                .rev()
                .map(F::from_canonical_usize)
                .collect::<Vec<_>>(),
        ]
        .concat()
        .try_into()
        .unwrap();

        (0..WIDTH / 2).for_each(|element| {
            assert_eq!(
                <DuplexChallenger<F, TestPermutation, WIDTH, RATE> as CanSample<F>>::sample(
                    &mut duplex_challenger
                ),
                F::from_canonical_u8(element as u8)
            );
            assert_eq!(
                duplex_challenger.output_buffer,
                should_be_sponge_state[..WIDTH - element - 1]
            );
            assert_eq!(duplex_challenger.sponge_state, should_be_sponge_state);
        });

        (0..WIDTH / 2).for_each(|i| {
            assert_eq!(
                <DuplexChallenger<F, TestPermutation, WIDTH, RATE> as CanSample<F>>::sample(
                    &mut duplex_challenger
                ),
                F::from_canonical_u8(0)
            );
            assert_eq!(duplex_challenger.input_buffer, vec![]);
            assert_eq!(
                duplex_challenger.output_buffer,
                vec![F::zero(); WIDTH / 2 - i - 1]
            );
            assert_eq!(duplex_challenger.sponge_state, should_be_sponge_state)
        })
    }
}
