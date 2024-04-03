//! The Poseidon2 permutation.
//!
//! This implementation was based upon the following resources:
//! - https://github.com/HorizenLabs/poseidon2/blob/main/plain_implementations/src/poseidon2/poseidon2.rs
//! - https://eprint.iacr.org/2023/323.pdf

#![no_std]

extern crate alloc;

mod diffusion;
mod matrix;
mod round_constants;
use alloc::vec::Vec;

pub use diffusion::{matmul_internal, DiffusionPermutation};
use matrix::Poseidon2MEMatrix;
use p3_field::{AbstractField, PrimeField};
use p3_symmetric::{CryptographicPermutation, Permutation};
use rand::distributions::{Distribution, Standard};
use rand::Rng;
pub use round_constants::*;

const SUPPORTED_WIDTHS: [usize; 8] = [2, 3, 4, 8, 12, 16, 20, 24];

/// The Poseidon2 permutation.
#[derive(Clone, Debug)]
pub struct Poseidon2<F, Diffusion, const WIDTH: usize, const D: u64> {
    /// The number of external rounds.
    rounds_f: usize,

    /// The external round constants.
    external_constants: Vec<[F; WIDTH]>,

    /// The number of internal rounds.
    rounds_p: usize,

    /// The internal round constants.
    internal_constants: Vec<F>,

    /// The linear layer used in internal rounds (only needs diffusion property, not MDS).
    internal_linear_layer: Diffusion,
}

impl<F, Diffusion, const WIDTH: usize, const D: u64> Poseidon2<F, Diffusion, WIDTH, D>
where
    F: PrimeField,
{
    /// Create a new Poseidon2 configuration.
    pub fn new(
        rounds_f: usize,
        external_constants: Vec<[F; WIDTH]>,
        rounds_p: usize,
        internal_constants: Vec<F>,
        internal_linear_layer: Diffusion,
    ) -> Self {
        assert!(SUPPORTED_WIDTHS.contains(&WIDTH));
        Self {
            rounds_f,
            external_constants,
            rounds_p,
            internal_constants,
            internal_linear_layer,
        }
    }

    /// Create a new Poseidon2 configuration with random parameters.
    pub fn new_from_rng<R: Rng>(
        rounds_f: usize,
        rounds_p: usize,
        internal_mds: Diffusion,
        rng: &mut R,
    ) -> Self
    where
        Standard: Distribution<F>,
    {
        let external_constants = rng.sample_iter(Standard).take(rounds_f).collect::<Vec<[F; WIDTH]>>();
        let internal_constants = rng.sample_iter(Standard).take(rounds_p).collect::<Vec<F>>();

        Self {
            rounds_f,
            external_constants,
            rounds_p,
            internal_constants,
            internal_linear_layer: internal_mds,
        }
    }

    #[inline]
    fn add_rc<AF>(&self, state: &mut [AF; WIDTH], rc: &[AF::F; WIDTH])
    where
        AF: AbstractField<F = F>,
    {
        state
            .iter_mut()
            .zip(rc)
            .for_each(|(a, b)| *a += AF::from_f(*b));
    }

    #[inline]
    fn sbox_p<AF>(&self, input: &AF) -> AF
    where
        AF: AbstractField<F = F>,
    {
        input.exp_const_u64::<D>()
    }

    #[inline]
    fn sbox<AF>(&self, state: &mut [AF; WIDTH])
    where
        AF: AbstractField<F = F>,
    {
        state.iter_mut().for_each(|a| *a = self.sbox_p(a));
    }
}

impl<AF, Diffusion, const WIDTH: usize, const D: u64> Permutation<[AF; WIDTH]>
    for Poseidon2<AF::F, Diffusion, WIDTH, D>
where
    AF: AbstractField,
    AF::F: PrimeField,
    Diffusion: DiffusionPermutation<AF, WIDTH>,
{
    fn permute_mut(&self, state: &mut [AF; WIDTH]) {
        let external_linear_layer = Poseidon2MEMatrix::<WIDTH, D>;

        // The initial linear layer.
        external_linear_layer.permute_mut(state);

        // The first half of the external rounds.
        let rounds_f_beginning = self.rounds_f / 2;
        for r in 0..rounds_f_beginning {
            self.add_rc(state, &self.external_constants[r]);
            self.sbox(state);
            external_linear_layer.permute_mut(state);
        }

        // The internal rounds.
        for r in 0..self.rounds_p {
            state[0] += AF::from_f(self.internal_constants[r]);
            state[0] = self.sbox_p(&state[0]);
            self.internal_linear_layer.permute_mut(state);
        }

        // The second half of the external rounds.
        for r in rounds_f_beginning..self.rounds_f {
            self.add_rc(state, &self.external_constants[r]);
            self.sbox(state);
            external_linear_layer.permute_mut(state);
        }
    }
}

impl<AF, Diffusion, const WIDTH: usize, const D: u64> CryptographicPermutation<[AF; WIDTH]>
    for Poseidon2<AF::F, Diffusion, WIDTH, D>
where
    AF: AbstractField,
    AF::F: PrimeField,
    Diffusion: DiffusionPermutation<AF, WIDTH>,
{
}
