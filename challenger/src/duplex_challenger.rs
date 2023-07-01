use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;
use p3_field::Field;
use p3_symmetric::permutation::ArrayPermutation;

use crate::Challenger;

pub struct DuplexChallenger<F: Field, P: ArrayPermutation<F, WIDTH>, const WIDTH: usize> {
    sponge_state: [F; WIDTH],
    input_buffer: Vec<F>,
    output_buffer: Vec<F>,
    permutation: P,
    _phantom_f: PhantomData<F>,
}

impl<F: Field, P: ArrayPermutation<F, WIDTH>, const WIDTH: usize> DuplexChallenger<F, P, WIDTH> {
    pub fn new(permutation: P) -> Self {
        Self {
            sponge_state: [F::ZERO; WIDTH],
            input_buffer: vec![],
            output_buffer: vec![],
            permutation,
            _phantom_f: PhantomData,
        }
    }

    fn duplexing(&mut self) {
        assert!(self.input_buffer.len() <= WIDTH);

        // Overwrite the first r elements with the inputs.
        for (i, val) in self.input_buffer.drain(..).enumerate() {
            self.sponge_state[i] = val;
        }

        // Apply the permutation.
        self.sponge_state = self.permutation.permute(self.sponge_state);

        self.output_buffer.clear();
        self.output_buffer.extend(self.sponge_state);
    }
}

impl<F: Field, P: ArrayPermutation<F, WIDTH>, const WIDTH: usize> Challenger<F>
    for DuplexChallenger<F, P, WIDTH>
{
    fn observe_element(&mut self, element: F) {
        // Any buffered output is now invalid.
        self.output_buffer.clear();

        self.input_buffer.push(element);

        if self.input_buffer.len() == WIDTH {
            self.duplexing();
        }
    }

    fn random_element(&mut self) -> F {
        // If we have buffered inputs, we must perform a duplexing so that the challenge will
        // reflect them. Or if we've run out of outputs, we must perform a duplexing to get more.
        if !self.input_buffer.is_empty() || self.output_buffer.is_empty() {
            self.duplexing();
        }

        self.output_buffer
            .pop()
            .expect("Output buffer should be non-empty")
    }
}

#[cfg(test)]
mod tests {
    use p3_field::AbstractField;
    use p3_goldilocks::Goldilocks;
    use p3_symmetric::permutation::CryptographicPermutation;
    use rand::seq::{index::sample, SliceRandom};

    use super::*;

    type ByteArray = [F; 32];
    type F = Goldilocks;

    const WIDTH: usize = 32;
    struct TestPermutation {}

    impl CryptographicPermutation<ByteArray> for TestPermutation {
        fn permute(&self, input: ByteArray) -> ByteArray {
            let mut output = [F::ZERO; 32];
            let mut rng = rand::thread_rng();
            let mut shuffled_indexes: [usize; 32] =
                (0..32).collect::<Vec<usize>>().try_into().unwrap();
            shuffled_indexes.shuffle(&mut rng);
            (0..32).for_each(|i| output[shuffled_indexes[i]] = input[i]);
            output
        }

        fn permute_mut(&self, input: &mut ByteArray) {
            let mut rng = rand::thread_rng();
            let mut shuffled_indexes: [usize; 32] =
                (0..32).collect::<Vec<usize>>().try_into().unwrap();
            shuffled_indexes.shuffle(&mut rng);
            (0..32).for_each(|i| input[i] = input[shuffled_indexes[i]]);
        }
    }

    impl ArrayPermutation<F, WIDTH> for TestPermutation {}

    #[test]
    fn it_works_duplexing() {
        let mut rng = rand::thread_rng();
        let permutation = TestPermutation {};
        let mut duplex_challenger = DuplexChallenger::new(permutation);
        let sample_range = sample(&mut rng, u32::MAX as usize, 31);
        let mut sample_iter = sample_range.iter();

        (0..31).for_each(|_| {
            let element = sample_iter.next().unwrap();
            duplex_challenger.observe_element(Goldilocks::from_canonical_usize(element))
        });

        let input_buffer = duplex_challenger.input_buffer.clone();

        assert_eq!(duplex_challenger.input_buffer.len(), 31);
        duplex_challenger.duplexing();

        assert_eq!(duplex_challenger.input_buffer.len(), 0);
        assert_eq!(duplex_challenger.output_buffer.len(), 32);
        for i in 0..31 {
            assert!(
                input_buffer.contains(&duplex_challenger.output_buffer[i])
                    || duplex_challenger.output_buffer[i] == F::ZERO
            );
        }

        assert_eq!(duplex_challenger.sponge_state.len(), 32);
        assert_eq!(
            duplex_challenger.sponge_state[..],
            duplex_challenger.output_buffer[..]
        );
    }
}
