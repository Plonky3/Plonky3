use crate::hasher::CryptographicHasher;
use crate::permutation::ArrayPermutation;
use core::marker::PhantomData;
use itertools::Itertools;

/// A padding-free, overwrite-mode sponge function.
///
/// WIDTH is the sponge's rate + sponge's capacity
pub struct PaddingFreeSponge<T, P, const WIDTH: usize> {
    permutation: P,
    _phantom_f: PhantomData<T>,
}

impl<T, P, const WIDTH: usize> PaddingFreeSponge<T, P, WIDTH> {
    pub fn new(permutation: P) -> Self {
        Self {
            permutation,
            _phantom_f: PhantomData,
        }
    }
}

impl<T, P, const RATE: usize, const WIDTH: usize> CryptographicHasher<T, [T; RATE]>
    for PaddingFreeSponge<T, P, WIDTH>
where
    T: Default + Copy,
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
