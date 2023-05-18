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
