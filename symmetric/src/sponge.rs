use core::marker::PhantomData;
use itertools::Itertools;
use p3_field::{AbstractField, Field, PrimeField32};

use crate::hasher::CryptographicHasher;
use crate::permutation::CryptographicPermutation;

/// A padding-free, overwrite-mode sponge function.
///
/// `WIDTH` is the sponge's rate plus the sponge's capacity.
#[derive(Clone)]
pub struct PaddingFreeSponge<P, const WIDTH: usize, const RATE: usize, const OUT: usize> {
    permutation: P,
}

impl<P, const WIDTH: usize, const RATE: usize, const OUT: usize>
    PaddingFreeSponge<P, WIDTH, RATE, OUT>
{
    pub fn new(permutation: P) -> Self {
        Self { permutation }
    }
}

impl<T, P, const WIDTH: usize, const RATE: usize, const OUT: usize> CryptographicHasher<T, [T; OUT]>
    for PaddingFreeSponge<P, WIDTH, RATE, OUT>
where
    T: Default + Copy,
    P: CryptographicPermutation<[T; WIDTH]>,
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


#[derive(Clone)]
pub struct PaddingFreeSpongeMultiField<F, PF, P, const WIDTH: usize, const RATE: usize, const OUT: usize> {
    permutation: P,
    num_f_elms: usize,
    alpha: PF,
    _phantom: PhantomData<F>,
}

impl<F, PF, P, const WIDTH: usize, const RATE: usize, const OUT: usize> PaddingFreeSpongeMultiField<F, PF, P, WIDTH, RATE, OUT>
where
    F: PrimeField32,
    PF: Field
{
    pub fn new(permutation: P) -> Self {
        let num_f_elms = PF::bits() / <F as Field>::bits();
        Self { permutation, num_f_elms, alpha: PF::from_canonical_u32(F::ORDER_U32), _phantom: PhantomData }
    }
}

impl<F, PF, P, const WIDTH: usize, const RATE: usize, const OUT: usize> CryptographicHasher<F, [PF; OUT]>
    for PaddingFreeSpongeMultiField<F, PF, P, WIDTH, RATE, OUT>
where
    F: PrimeField32,
    PF: AbstractField + Default + Copy,
    P: CryptographicPermutation<[PF; WIDTH]>,
{
    fn hash_iter<I>(&self, input: I) -> [PF; OUT]
    where
        I: IntoIterator<Item = F>,
    {
        let mut state = [PF::default(); WIDTH];
        for block_chunk in &input.into_iter().chunks(RATE) {
            for (chunk_id, chunk) in (&block_chunk.into_iter().chunks(self.num_f_elms)).into_iter().enumerate() {
                let mut sum = PF::zero();
                for term in chunk {
                    sum = sum * self.alpha + PF::from_canonical_u32(term.as_canonical_u32());
                }
                state[chunk_id] = sum;
            }
            state = self.permutation.permute(state);
        }

        state[..OUT].try_into().unwrap()
    }
}
