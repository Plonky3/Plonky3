//! The Poseidon2 permutation.
//!
//! This implementation was based upon the following resources:
//! - https://github.com/HorizenLabs/poseidon2/blob/main/plain_implementations/src/poseidon2/poseidon2.rs
//! - https://eprint.iacr.org/2023/323.pdf

#![no_std]

extern crate alloc;

use alloc::borrow::ToOwned;
use alloc::vec::Vec;

use p3_field::Field;
use p3_mds::MdsPermutation;
use p3_symmetric::permutation::{ArrayPermutation, CryptographicPermutation};
use rand::distributions::Standard;
use rand::prelude::Distribution;
use rand::Rng;

/// The Poseidon2 permutation.
#[derive(Clone)]
pub struct Poseidon2<F, ExternalMds, InternalMds, const WIDTH: usize, const D: u64>
where
    F: Field,
    ExternalMds: MdsPermutation<F, WIDTH>,
    InternalMds: MdsPermutation<F, WIDTH>,
{
    /// The number of external rounds.
    rounds_f: usize,

    /// The number of internal rounds.
    rounds_p: usize,

    /// The round constants.
    constants: Vec<[F; WIDTH]>,

    /// The linear layer used in external rounds.
    external_mds: ExternalMds,

    /// The linear layer used in internal rounds (only needs diffusion property, not MDS).
    internal_mds: InternalMds,
}

impl<F, ExternalMds, InternalMds, const WIDTH: usize, const D: u64>
    Poseidon2<F, ExternalMds, InternalMds, WIDTH, D>
where
    F: Field,
    ExternalMds: MdsPermutation<F, WIDTH>,
    InternalMds: MdsPermutation<F, WIDTH>,
{
    /// Create a new Poseidon2 configuration.
    pub fn new(
        rounds_f: usize,
        rounds_p: usize,
        constants: Vec<[F; WIDTH]>,
        external_mds: ExternalMds,
        internal_mds: InternalMds,
    ) -> Self {
        Self {
            rounds_f,
            rounds_p,
            constants,
            external_mds,
            internal_mds,
        }
    }

    /// Create a new Poseidon2 configuration with random parameters.
    pub fn new_from_rng<R: Rng>(
        rounds_f: usize,
        rounds_p: usize,
        external_mds: ExternalMds,
        internal_mds: InternalMds,
        rng: &mut R,
    ) -> Self
    where
        Standard: Distribution<F>,
    {
        let mut constants = Vec::new();
        let rounds = rounds_f + rounds_p;
        for _ in 0..rounds {
            let mut round_constant = [F::ZERO; WIDTH];
            #[allow(clippy::needless_range_loop)]
            for j in 0..WIDTH {
                round_constant[j] = rng.sample(Standard);
            }
            constants.push(round_constant);
        }

        Self {
            rounds_f,
            rounds_p,
            constants,
            external_mds,
            internal_mds,
        }
    }

    #[inline]
    fn add_rc(&self, state: &[F], rc: &[F]) -> [F; WIDTH] {
        let mut result = [F::ZERO; WIDTH];
        for i in 0..WIDTH {
            result[i] = state[i] + rc[i];
        }
        result
    }

    #[inline]
    fn sbox_p(&self, input: &F) -> F {
        input.exp_const_u64::<D>()
    }

    #[inline]
    fn sbox(&self, state: &[F]) -> [F; WIDTH] {
        let mut result = [F::ZERO; WIDTH];
        for i in 0..WIDTH {
            result[i] = self.sbox_p(&state[i]);
        }
        result
    }
}

impl<F, ExternalMds, InternalMds, const WIDTH: usize, const D: u64>
    CryptographicPermutation<[F; WIDTH]> for Poseidon2<F, ExternalMds, InternalMds, WIDTH, D>
where
    F: Field,
    ExternalMds: MdsPermutation<F, WIDTH>,
    InternalMds: MdsPermutation<F, WIDTH>,
{
    fn permute(&self, state: [F; WIDTH]) -> [F; WIDTH] {
        let mut state = state.to_owned();

        // The initial linear layer.
        self.external_mds.permute_mut(&mut state);

        // The first half of the external rounds.
        let rounds = self.rounds_f + self.rounds_p;
        let rounds_f_beggining = self.rounds_f / 2;
        for r in 0..rounds_f_beggining {
            state = self.add_rc(&state, &self.constants[r]);
            state = self.sbox(&state);
            self.external_mds.permute_mut(&mut state);
        }

        // The internal rounds.
        let p_end = rounds_f_beggining + self.rounds_p;
        for r in self.rounds_f..p_end {
            state[0] += self.constants[r][0];
            state[0] = self.sbox_p(&state[0]);
            self.internal_mds.permute_mut(&mut state);
        }

        // The second half of the external rounds.
        for r in p_end..rounds {
            state = self.add_rc(&state, &self.constants[r]);
            state = self.sbox(&state);
            self.external_mds.permute_mut(&mut state);
        }

        state
    }
}

impl<F: Field, ExternalMds, InternalMds, const T: usize, const D: u64> ArrayPermutation<F, T>
    for Poseidon2<F, ExternalMds, InternalMds, T, D>
where
    F: Field,
    ExternalMds: MdsPermutation<F, T>,
    InternalMds: MdsPermutation<F, T>,
{
}
