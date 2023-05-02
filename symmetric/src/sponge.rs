use crate::hasher::{CryptographicHasher, IterHasher};
use crate::permutation::ArrayPermutation;
use alloc::vec::Vec;
use core::marker::PhantomData;
use itertools::Itertools;

/// A padding-free, overwrite-mode sponge function.
///
/// WIDTH is the sponge's rate + sponge's capacity
pub struct PaddingFreeSponge<T, P, const WIDTH: usize>
where
    P: ArrayPermutation<T, WIDTH>,
{
    _phantom_f: PhantomData<T>,
    // _phantom_p: PhantomData<P>,
    permutation: P,
}

impl<T: Default + Copy, P, const RATE: usize, const WIDTH: usize>
    CryptographicHasher<Vec<T>, [T; RATE]> for PaddingFreeSponge<T, P, WIDTH>
where
    P: ArrayPermutation<T, WIDTH>,
{
    fn hash(&self, input: &Vec<T>) -> [T; RATE] {
        // static_assert(RATE < WIDTH)
        let mut state = [T::default(); WIDTH];
        for input_chunk in input.chunks(RATE) {
            state[..input_chunk.len()].copy_from_slice(input_chunk);
            state = self.permutation.permute(state);
        }
        state[..RATE].try_into().unwrap()
    }
}

impl<T: Default + Copy, P, const RATE: usize, const WIDTH: usize> IterHasher<T, [T; RATE]>
    for PaddingFreeSponge<T, P, WIDTH>
where
    P: ArrayPermutation<T, WIDTH>,
{
    fn hash_iter<I>(&self, input: I) -> [T; RATE]
    where
        I: IntoIterator<Item = T>,
    {
        // static_assert(RATE < WIDTH)
        let mut state = [T::default(); WIDTH];
        for input_chunk in &input.into_iter().chunks(RATE) {
            state.iter_mut().zip(input_chunk).for_each(|(s, i)| *s = i);
            state = self.permutation.permute(state);
        }
        state[..RATE].try_into().unwrap()
    }
}
