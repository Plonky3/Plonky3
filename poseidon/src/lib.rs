//! The Poseidon permutation.

#![no_std]

extern crate alloc;

use alloc::vec::Vec;

use p3_field::Field;
use p3_mds::MdsPermutation;
use p3_symmetric::permutation::{ArrayPermutation, CryptographicPermutation};
use rand::distributions::Standard;
use rand::prelude::Distribution;
use rand::Rng;

/// The Poseidon permutation.
#[derive(Clone)]
pub struct Poseidon<F, Mds, const WIDTH: usize, const ALPHA: u64>
where
    F: Field,
    Mds: MdsPermutation<F, WIDTH>,
{
    half_num_full_rounds: usize,
    num_partial_rounds: usize,
    constants: Vec<F>,
    mds: Mds,
}

impl<F, Mds, const WIDTH: usize, const ALPHA: u64> Poseidon<F, Mds, WIDTH, ALPHA>
where
    F: Field,
    Mds: MdsPermutation<F, WIDTH>,
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

    fn half_full_rounds(&self, state: &mut [F; WIDTH], round_ctr: &mut usize) {
        for _ in 0..self.half_num_full_rounds {
            self.constant_layer(state, *round_ctr);
            Self::full_sbox_layer(state);
            self.mds.permute_mut(state);
            *round_ctr += 1;
        }
    }

    fn partial_rounds(&self, state: &mut [F; WIDTH], round_ctr: &mut usize) {
        for _ in 0..self.num_partial_rounds {
            self.constant_layer(state, *round_ctr);
            Self::partial_sbox_layer(state);
            self.mds.permute_mut(state);
            *round_ctr += 1;
        }
    }

    fn full_sbox_layer(state: &mut [F; WIDTH]) {
        for x in state.iter_mut() {
            *x = x.exp_u64(ALPHA);
        }
    }

    fn partial_sbox_layer(state: &mut [F; WIDTH]) {
        state[0] = state[0].exp_u64(ALPHA);
    }

    fn constant_layer(&self, state: &mut [F; WIDTH], round: usize) {
        for (i, x) in state.iter_mut().enumerate() {
            *x += self.constants[round * WIDTH + i];
        }
    }
}

impl<F, Mds, const WIDTH: usize, const ALPHA: u64> CryptographicPermutation<[F; WIDTH]>
    for Poseidon<F, Mds, WIDTH, ALPHA>
where
    F: Field,
    Mds: MdsPermutation<F, WIDTH>,
{
    fn permute(&self, mut state: [F; WIDTH]) -> [F; WIDTH] {
        let mut round_ctr = 0;
        self.half_full_rounds(&mut state, &mut round_ctr);
        self.partial_rounds(&mut state, &mut round_ctr);
        self.half_full_rounds(&mut state, &mut round_ctr);
        state
    }
}

impl<F: Field, Mds, const WIDTH: usize, const ALPHA: u64> ArrayPermutation<F, WIDTH>
    for Poseidon<F, Mds, WIDTH, ALPHA>
where
    F: Field,
    Mds: MdsPermutation<F, WIDTH>,
{
}
