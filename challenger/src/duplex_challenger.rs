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

    use super::*;

    const WIDTH: usize = 32;

    type TestArray = [F; WIDTH];
    type F = Goldilocks;

    struct TestPermutation {}

    impl CryptographicPermutation<TestArray> for TestPermutation {
        fn permute(&self, input: TestArray) -> TestArray {
            let mut output = input.clone();
            output.reverse();
            output.try_into().unwrap()
        }

        fn permute_mut(&self, input: &mut TestArray) {
            input.reverse()
        }
    }

    impl ArrayPermutation<F, WIDTH> for TestPermutation {}

    #[test]
    fn test_duplex_challenger() {
        let permutation = TestPermutation {};
        let mut duplex_challenger = DuplexChallenger::new(permutation);

        // observe elements before reaching WIDTH
        (0..WIDTH - 1).for_each(|element| {
            duplex_challenger.observe_element(F::from_canonical_u8(element as u8));
            assert_eq!(duplex_challenger.input_buffer.len(), element + 1);
            assert_eq!(
                duplex_challenger.input_buffer,
                (0..element + 1)
                    .map(|i| F::from_canonical_u8(i as u8))
                    .collect::<Vec<_>>()
            );
            assert_eq!(duplex_challenger.output_buffer, vec![]);
            assert_eq!(duplex_challenger.sponge_state, [F::ZERO; WIDTH]);
        });

        // Test functionality when we observe WIDTH elements
        duplex_challenger.observe_element(F::from_canonical_u8(31));
        assert_eq!(duplex_challenger.input_buffer, vec![]);

        let should_be_output_buffer: [F; WIDTH] = (0..WIDTH as u8)
            .rev()
            .map(|i| F::from_canonical_u8(i))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        assert_eq!(duplex_challenger.output_buffer, should_be_output_buffer);
        assert_eq!(duplex_challenger.sponge_state, should_be_output_buffer);

        // Test functionality when observe, over more than WIDTH elements
        duplex_challenger.observe_element(F::from_canonical_u8(255));
        assert_eq!(duplex_challenger.input_buffer, [F::from_canonical_u8(255)]);
        assert_eq!(duplex_challenger.output_buffer, vec![]);
        assert_eq!(duplex_challenger.sponge_state, should_be_output_buffer);

        // Test functionality when observing additional elements
        duplex_challenger.duplexing();
        let mut should_be_output_buffer =
            (0..31).map(|i| F::from_canonical_u8(i)).collect::<Vec<_>>();
        should_be_output_buffer.push(F::from_canonical_u8(255));
        assert_eq!(duplex_challenger.input_buffer, vec![]);
        assert_eq!(duplex_challenger.output_buffer, should_be_output_buffer);
        let should_be_sponge_state: [F; WIDTH] = should_be_output_buffer.try_into().unwrap();
        assert_eq!(duplex_challenger.sponge_state, should_be_sponge_state);
    }

    #[test]
    fn test_duplex_challenger_randomized() {
        let permutation = TestPermutation {};
        let mut duplex_challenger = DuplexChallenger::new(permutation);

        // Observe WIDTH / 2 elements.
        (0..WIDTH / 2).for_each(|element| {
            duplex_challenger.observe_element(F::from_canonical_u8(element as u8))
        });

        let should_be_sponge_state: [F; WIDTH] = [
            vec![F::ZERO; WIDTH / 2],
            (0..WIDTH / 2)
                .rev()
                .map(|i| F::from_canonical_usize(i))
                .collect::<Vec<_>>(),
        ]
        .concat()
        .try_into()
        .unwrap();

        (0..WIDTH / 2).for_each(|element| {
            assert_eq!(
                duplex_challenger.random_element(),
                F::from_canonical_u8(element as u8)
            );
            assert_eq!(
                duplex_challenger.output_buffer,
                should_be_sponge_state[..WIDTH - element - 1]
            );
            assert_eq!(duplex_challenger.sponge_state, should_be_sponge_state);
        });

        (0..WIDTH / 2).for_each(|i| {
            assert_eq!(duplex_challenger.random_element(), F::from_canonical_u8(0));
            assert_eq!(duplex_challenger.input_buffer, vec![]);
            assert_eq!(
                duplex_challenger.output_buffer,
                vec![F::ZERO; WIDTH / 2 - i - 1]
            );
            assert_eq!(duplex_challenger.sponge_state, should_be_sponge_state)
        })
    }
}
