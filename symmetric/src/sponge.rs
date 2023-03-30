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
    _phantom_f: PhantomData<F>,
    _phantom_p: PhantomData<P>,
}

impl<F, P, const RATE: usize, const CAPACITY: usize> AlgebraicHash<F, RATE>
    for PaddingFreeAlgebraicSponge<F, P, RATE, CAPACITY>
where
    F: Field,
    P: AlgebraicPermutation<F, { RATE + CAPACITY }>,
{
    fn hash(input: Vec<F>) -> [F; RATE] {
        let mut state = [F::ZERO; RATE + CAPACITY];
        for input_chunk in input.chunks(RATE) {
            state[..input_chunk.len()].copy_from_slice(input_chunk);
            state = P::permute(state);
        }
        let mut output = [F::ZERO; RATE];
        for i in 0..RATE {
            output[i] = state[i];
        }
        output
    }
}
