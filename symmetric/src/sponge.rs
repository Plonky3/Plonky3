use alloc::string::String;
use core::marker::PhantomData;

use itertools::Itertools;
use p3_field::{reduce_32, Field, PrimeField, PrimeField32};

use crate::hasher::CryptographicHasher;
use crate::permutation::CryptographicPermutation;

/// A padding-free, overwrite-mode sponge function.
///
/// `WIDTH` is the sponge's rate plus the sponge's capacity.
#[derive(Copy, Clone, Debug)]
pub struct PaddingFreeSponge<P, const WIDTH: usize, const RATE: usize, const OUT: usize> {
    permutation: P,
}

impl<P, const WIDTH: usize, const RATE: usize, const OUT: usize>
    PaddingFreeSponge<P, WIDTH, RATE, OUT>
{
    pub const fn new(permutation: P) -> Self {
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
        let mut input = input.into_iter();

        // Itertools' chunks() is more convenient, but seems to add more overhead,
        // hence the more manual loop.
        'outer: loop {
            for i in 0..RATE {
                if let Some(x) = input.next() {
                    state[i] = x;
                } else {
                    if i != 0 {
                        self.permutation.permute_mut(&mut state);
                    }
                    break 'outer;
                }
            }
            self.permutation.permute_mut(&mut state);
        }

        state[..OUT].try_into().unwrap()
    }
}

/// A padding-free, overwrite-mode sponge function that operates natively over PF but accepts elements
/// of F: PrimeField32.
///
/// `WIDTH` is the sponge's rate plus the sponge's capacity.
#[derive(Clone, Debug)]
pub struct MultiField32PaddingFreeSponge<
    F,
    PF,
    P,
    const WIDTH: usize,
    const RATE: usize,
    const OUT: usize,
> {
    permutation: P,
    num_f_elms: usize,
    _phantom: PhantomData<(F, PF)>,
}

impl<F, PF, P, const WIDTH: usize, const RATE: usize, const OUT: usize>
    MultiField32PaddingFreeSponge<F, PF, P, WIDTH, RATE, OUT>
where
    F: PrimeField32,
    PF: Field,
{
    pub fn new(permutation: P) -> Result<Self, String> {
        if F::order() >= PF::order() {
            return Err(String::from("F::order() must be less than PF::order()"));
        }

        let num_f_elms = PF::bits() / F::bits();
        Ok(Self {
            permutation,
            num_f_elms,
            _phantom: PhantomData,
        })
    }
}

impl<F, PF, P, const WIDTH: usize, const RATE: usize, const OUT: usize>
    CryptographicHasher<F, [PF; OUT]> for MultiField32PaddingFreeSponge<F, PF, P, WIDTH, RATE, OUT>
where
    F: PrimeField32,
    PF: PrimeField + Default + Copy,
    P: CryptographicPermutation<[PF; WIDTH]>,
{
    fn hash_iter<I>(&self, input: I) -> [PF; OUT]
    where
        I: IntoIterator<Item = F>,
    {
        let mut state = [PF::default(); WIDTH];
        for block_chunk in &input.into_iter().chunks(RATE) {
            for (chunk_id, chunk) in (&block_chunk.chunks(self.num_f_elms))
                .into_iter()
                .enumerate()
            {
                state[chunk_id] = reduce_32(&chunk.collect_vec());
            }
            state = self.permutation.permute(state);
        }

        state[..OUT].try_into().unwrap()
    }
}
