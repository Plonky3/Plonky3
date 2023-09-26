//! The Poseidon2 permutation.
//!
//! This implementation was based upon the following resources:
//! - https://github.com/HorizenLabs/poseidon2/blob/main/plain_implementations/src/poseidon2/poseidon2.rs
//! - https://eprint.iacr.org/2023/323.pdf

#![no_std]

extern crate alloc;

mod babybear;
mod diffusion;
mod goldilocks;
use alloc::vec::Vec;

pub use babybear::DiffusionMatrixBabybear;
pub use diffusion::DiffusionPermutation;
pub use goldilocks::DiffusionMatrixGoldilocks;
use p3_field::AbstractField;
use p3_mds::MdsPermutation;
use p3_symmetric::permutation::CryptographicPermutation;
use rand::distributions::Standard;
use rand::prelude::Distribution;
use rand::Rng;

const SUPPORTED_WIDTHS: [usize; 8] = [2, 3, 4, 8, 12, 16, 20, 24];

/// The Poseidon2 permutation.
#[derive(Clone)]
pub struct Poseidon2<F, Mds, Diffusion, const WIDTH: usize, const D: u64>
where
    F: AbstractField,
    Mds: MdsPermutation<F, WIDTH>,
    Diffusion: DiffusionPermutation<F, WIDTH>,
{
    /// The number of external rounds.
    rounds_f: usize,

    /// The number of internal rounds.
    rounds_p: usize,

    /// The round constants.
    constants: Vec<[F::F; WIDTH]>,

    /// The linear layer used in external rounds.
    external_linear_layer: Mds,

    /// The linear layer used in internal rounds (only needs diffusion property, not MDS).
    internal_linear_layer: Diffusion,
}

impl<F, Mds, Diffusion, const WIDTH: usize, const D: u64> Poseidon2<F, Mds, Diffusion, WIDTH, D>
where
    F: AbstractField,
    Mds: MdsPermutation<F, WIDTH>,
    Diffusion: DiffusionPermutation<F, WIDTH>,
{
    /// Create a new Poseidon2 configuration.
    pub fn new(
        rounds_f: usize,
        rounds_p: usize,
        constants: Vec<[F::F; WIDTH]>,
        external_linear_layer: Mds,
        internal_linear_layer: Diffusion,
    ) -> Self {
        assert!(SUPPORTED_WIDTHS.contains(&WIDTH));
        Self {
            rounds_f,
            rounds_p,
            constants,
            external_linear_layer,
            internal_linear_layer,
        }
    }

    /// Create a new Poseidon2 configuration with random parameters.
    pub fn new_from_rng<R: Rng>(
        rounds_f: usize,
        rounds_p: usize,
        external_mds: Mds,
        internal_mds: Diffusion,
        rng: &mut R,
    ) -> Self
    where
        Standard: Distribution<F::F>,
    {
        let mut constants = Vec::new();
        let rounds = rounds_f + rounds_p;
        for _ in 0..rounds {
            let mut round_constant = [F::F::ZERO; WIDTH];
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
            external_linear_layer: external_mds,
            internal_linear_layer: internal_mds,
        }
    }

    #[inline]
    fn add_rc(&self, state: &mut [F; WIDTH], rc: &[F::F; WIDTH]) {
        state.iter_mut().zip(rc).for_each(|(a, b)| *a += *b);
    }

    #[inline]
    fn sbox_p(&self, input: &F) -> F {
        input.exp_const_u64::<D>()
    }

    #[inline]
    fn sbox(&self, state: &mut [F; WIDTH]) {
        state.iter_mut().for_each(|a| *a = self.sbox_p(a));
    }
}

impl<F, Mds, Diffusion, const WIDTH: usize, const D: u64> CryptographicPermutation<[F; WIDTH]>
    for Poseidon2<F, Mds, Diffusion, WIDTH, D>
where
    F: AbstractField,
    Mds: MdsPermutation<F, WIDTH>,
    Diffusion: DiffusionPermutation<F, WIDTH>,
{
    fn permute(&self, mut state: [F; WIDTH]) -> [F; WIDTH] {
        // The initial linear layer.
        self.external_linear_layer.permute_mut(&mut state);

        // The first half of the external rounds.
        let rounds = self.rounds_f + self.rounds_p;
        let rounds_f_beggining = self.rounds_f / 2;
        for r in 0..rounds_f_beggining {
            self.add_rc(&mut state, &self.constants[r]);
            self.sbox(&mut state);
            self.external_linear_layer.permute_mut(&mut state);
        }

        // The internal rounds.
        let p_end = rounds_f_beggining + self.rounds_p;
        for r in self.rounds_f..p_end {
            state[0] += self.constants[r][0];
            state[0] = self.sbox_p(&state[0]);
            self.internal_linear_layer.permute_mut(&mut state);
        }

        // The second half of the external rounds.
        for r in p_end..rounds {
            self.add_rc(&mut state, &self.constants[r]);
            self.sbox(&mut state);
            self.external_linear_layer.permute_mut(&mut state);
        }

        state
    }
}
