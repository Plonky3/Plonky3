use core::marker::PhantomData;

use itertools::Itertools;

use crate::hasher::CryptographicHasher;
use crate::permutation::ArrayPermutation;

/// A padding-free, overwrite-mode sponge function.
///
/// `WIDTH` is the sponge's rate plus the sponge's capacity.
#[derive(Clone)]
pub struct PaddingFreeSponge<T, P, const WIDTH: usize, const RATE: usize, const OUT: usize> {
    permutation: P,
    _phantom_f: PhantomData<T>,
}

impl<T, P, const WIDTH: usize, const RATE: usize, const OUT: usize>
    PaddingFreeSponge<T, P, WIDTH, RATE, OUT>
{
    pub fn new(permutation: P) -> Self {
        Self {
            permutation,
            _phantom_f: PhantomData,
        }
    }
}

impl<T, P, const WIDTH: usize, const RATE: usize, const OUT: usize> CryptographicHasher<T, [T; OUT]>
    for PaddingFreeSponge<T, P, WIDTH, RATE, OUT>
where
    T: Default + Copy,
    P: ArrayPermutation<T, WIDTH>,
{
    fn hash_iter<I>(&self, input: I) -> [T; OUT]
    where
        I: IntoIterator<Item = T>,
    {
        // static_assert(RATE < WIDTH)
        let mut state = [T::default(); WIDTH];
        for input_chunk in &input.into_iter().chunks(RATE) {
            state.iter_mut().zip(input_chunk).for_each(|(s, i)| *s = i);
            state = self.permutation.permute(state);
        }
        state[..OUT].try_into().unwrap()
    }
}
