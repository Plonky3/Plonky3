use alloc::string::String;
use core::marker::PhantomData;
use itertools::Itertools;
use p3_field::{reduce_64, Field, PrimeField, PrimeField64};

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


/// A padding-free, overwrite-mode sponge function.  Accepts `PrimeField64` elements and has a permutation 
/// using a different `Field` type.
///
/// `WIDTH` is the sponge's rate plus the sponge's capacity.
#[derive(Clone)]
pub struct PaddingFreeSpongeMultiField<F, PF, P, const WIDTH: usize, const RATE: usize, const OUT: usize> {
    permutation: P,
    num_f_elms: usize,
    _phantom: PhantomData<(F, PF)>,
}

impl<F, PF, P, const WIDTH: usize, const RATE: usize, const OUT: usize> PaddingFreeSpongeMultiField<F, PF, P, WIDTH, RATE, OUT>
where
    F: PrimeField64,
    PF: Field
{
    pub fn new(permutation: P) -> Result<Self, String> {
        if F::order() >= PF::order() {
            return Err(String::from("F::order() must be less than PF::order()"));
        }

        let num_f_elms = PF::bits() / F::bits();
        Ok(Self { permutation, num_f_elms, _phantom: PhantomData })
    }
}

impl<F, PF, P, const WIDTH: usize, const RATE: usize, const OUT: usize> CryptographicHasher<F, [PF; OUT]>
    for PaddingFreeSpongeMultiField<F, PF, P, WIDTH, RATE, OUT>
where
    F: PrimeField64,
    PF: PrimeField + Default + Copy,
    P: CryptographicPermutation<[PF; WIDTH]>,
{
    fn hash_iter<I>(&self, input: I) -> [PF; OUT]
    where
        I: IntoIterator<Item = F>,
    {
        let mut state = [PF::default(); WIDTH];
        for block_chunk in &input.into_iter().chunks(RATE) {
            for (chunk_id, chunk) in (&block_chunk.chunks(self.num_f_elms)).into_iter().enumerate() {
                state[chunk_id] = reduce_64(&chunk.collect_vec());
            }
            state = self.permutation.permute(state);
        }

        state[..OUT].try_into().unwrap()
    }
}
