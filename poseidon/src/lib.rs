//! The Poseidon permutation.

#![no_std]

extern crate alloc;

use alloc::vec::Vec;

use p3_field::{AbstractField, PrimeField};
use p3_mds::MdsPermutation;
use p3_symmetric::{CryptographicPermutation, Permutation};
use rand::distributions::Standard;
use rand::prelude::Distribution;
use rand::Rng;

/// The Poseidon permutation.
#[derive(Clone, Debug)]
pub struct Poseidon<F, Mds, const WIDTH: usize, const ALPHA: u64> {
    half_num_full_rounds: usize,
    num_partial_rounds: usize,
    constants: Vec<F>,
    mds: Mds,
}

impl<F, Mds, const WIDTH: usize, const ALPHA: u64> Poseidon<F, Mds, WIDTH, ALPHA>
where
    F: PrimeField,
{
    /// Create a new Poseidon configuration.
    ///
    /// # Panics
    /// Number of constants must match WIDTH times `num_rounds`; panics otherwise.
    pub fn new(
        half_num_full_rounds: usize,
        num_partial_rounds: usize,
        constants: Vec<F>,
        mds: Mds,
    ) -> Self {
        let num_rounds = 2 * half_num_full_rounds + num_partial_rounds;
        assert_eq!(constants.len(), WIDTH * num_rounds);
        Self {
            half_num_full_rounds,
            num_partial_rounds,
            constants,
            mds,
        }
    }

    pub fn new_from_rng<R: Rng>(
        half_num_full_rounds: usize,
        num_partial_rounds: usize,
        mds: Mds,
        rng: &mut R,
    ) -> Self
    where
        Standard: Distribution<F>,
    {
        let num_rounds = 2 * half_num_full_rounds + num_partial_rounds;
        let num_constants = WIDTH * num_rounds;
        let constants = rng
            .sample_iter(Standard)
            .take(num_constants)
            .collect::<Vec<_>>();
        Self {
            half_num_full_rounds,
            num_partial_rounds,
            constants,
            mds,
        }
    }

    fn half_full_rounds<AF>(&self, state: &mut [AF; WIDTH], round_ctr: &mut usize)
    where
        AF: AbstractField<F = F>,
        Mds: MdsPermutation<AF, WIDTH>,
    {
        for _ in 0..self.half_num_full_rounds {
            self.constant_layer(state, *round_ctr);
            Self::full_sbox_layer(state);
            self.mds.permute_mut(state);
            *round_ctr += 1;
        }
    }

    fn partial_rounds<AF>(&self, state: &mut [AF; WIDTH], round_ctr: &mut usize)
    where
        AF: AbstractField<F = F>,
        Mds: MdsPermutation<AF, WIDTH>,
    {
        for _ in 0..self.num_partial_rounds {
            self.constant_layer(state, *round_ctr);
            Self::partial_sbox_layer(state);
            self.mds.permute_mut(state);
            *round_ctr += 1;
        }
    }

    fn full_sbox_layer<AF>(state: &mut [AF; WIDTH])
    where
        AF: AbstractField<F = F>,
    {
        for x in state.iter_mut() {
            *x = x.exp_const_u64::<ALPHA>();
        }
    }

    fn partial_sbox_layer<AF>(state: &mut [AF; WIDTH])
    where
        AF: AbstractField<F = F>,
    {
        state[0] = state[0].exp_const_u64::<ALPHA>();
    }

    fn constant_layer<AF>(&self, state: &mut [AF; WIDTH], round: usize)
    where
        AF: AbstractField<F = F>,
    {
        for (i, x) in state.iter_mut().enumerate() {
            *x += AF::from_f(self.constants[round * WIDTH + i]);
        }
    }
}

impl<AF, Mds, const WIDTH: usize, const ALPHA: u64> Permutation<[AF; WIDTH]>
    for Poseidon<AF::F, Mds, WIDTH, ALPHA>
where
    AF: AbstractField,
    AF::F: PrimeField,
    Mds: MdsPermutation<AF, WIDTH>,
{
    fn permute_mut(&self, state: &mut [AF; WIDTH]) {
        let mut round_ctr = 0;
        self.half_full_rounds(state, &mut round_ctr);
        self.partial_rounds(state, &mut round_ctr);
        self.half_full_rounds(state, &mut round_ctr);
    }
}

impl<AF, Mds, const WIDTH: usize, const ALPHA: u64> CryptographicPermutation<[AF; WIDTH]>
    for Poseidon<AF::F, Mds, WIDTH, ALPHA>
where
    AF: AbstractField,
    AF::F: PrimeField,
    Mds: MdsPermutation<AF, WIDTH>,
{
}
