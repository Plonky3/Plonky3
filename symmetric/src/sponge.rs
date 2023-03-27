use crate::hash::AlgebraicHash;
use crate::permutation::AlgebraicPermutation;
use alloc::vec::Vec;
use core::marker::PhantomData;
use hyperfield::field::Field;

/// A padding-free, overwrite-mode sponge function.
pub struct PaddingFreeAlgebraicSponge<F, P, const RATE: usize, const CAPACITY: usize>
where
    F: Field,
    P: AlgebraicPermutation<F, { RATE + CAPACITY }>,
{
    permutation: P,
    _phantom_f: PhantomData<F>,
}

impl<F, P, const RATE: usize, const CAPACITY: usize> AlgebraicHash<F, RATE>
    for PaddingFreeAlgebraicSponge<F, P, RATE, CAPACITY>
where
    F: Field,
    P: AlgebraicPermutation<F, { RATE + CAPACITY }>,
{
    fn hash(&self, input: Vec<F>) -> [F; RATE] {
        let mut state = [F::ZERO; RATE + CAPACITY];
        for input_chunk in input.chunks(RATE) {
            state[..input_chunk.len()].copy_from_slice(input_chunk);
            state = self.permutation.permute(state);
        }
        let mut output = [F::ZERO; RATE];
        for i in 0..RATE {
            output[i] = state[i];
        }
        output
    }
}
