use crate::hasher::CryptographicHasher;
use crate::permutation::CryptographicPermutation;
use alloc::vec::Vec;
use core::marker::PhantomData;

/// A padding-free, overwrite-mode sponge function.
pub struct PaddingFreeSponge<T, P, const RATE: usize, const CAPACITY: usize>
where
    P: CryptographicPermutation<T, { RATE + CAPACITY }>,
{
    _phantom_f: PhantomData<T>,
    _phantom_p: PhantomData<P>,
}

impl<T: Default + Copy, P, const RATE: usize, const CAPACITY: usize> CryptographicHasher<T, RATE>
    for PaddingFreeSponge<T, P, RATE, CAPACITY>
where
    P: CryptographicPermutation<T, { RATE + CAPACITY }>,
{
    fn hash(input: Vec<T>) -> [T; RATE] {
        let mut state = [T::default(); RATE + CAPACITY];
        for input_chunk in input.chunks(RATE) {
            state[..input_chunk.len()].copy_from_slice(input_chunk);
            state = P::permute(state);
        }
        state[..RATE].try_into().unwrap()
    }
}
