//! The Poseidon2 permutation.
//!
//! This implementation was based upon the following resources:
//! - https://github.com/HorizenLabs/poseidon2/blob/main/plain_implementations/src/poseidon2/poseidon2.rs
//! - https://eprint.iacr.org/2023/323.pdf

#![no_std]

extern crate alloc;

mod diffusion;
mod matrix;
mod round_numbers;
use alloc::vec::Vec;

pub use diffusion::{matmul_internal, DiffusionPermutation};
pub use matrix::*;
use p3_field::{AbstractField, PackedField, PrimeField, PrimeField64};
use p3_symmetric::{CryptographicPermutation, Permutation};
use rand::distributions::{Distribution, Standard};
use rand::Rng;
pub use round_numbers::poseidon2_round_numbers_128;

const SUPPORTED_WIDTHS: [usize; 8] = [2, 3, 4, 8, 12, 16, 20, 24];

/// The Basic Poseidon2 permutation.
#[derive(Clone, Debug)]
pub struct Poseidon2<F, MdsLight, Diffusion, const WIDTH: usize, const D: u64> {
    /// The number of external rounds.
    rounds_f: usize,

    /// The external round constants.
    external_constants: Vec<[F; WIDTH]>,

    /// The linear layer used in External Rounds. Should be either MDS or a
    /// circulant matrix based off an MDS matrix of size 4.
    external_linear_layer: MdsLight,

    /// The number of internal rounds.
    rounds_p: usize,

    /// The internal round constants.
    internal_constants: Vec<F>,

    /// The linear layer used in internal rounds (only needs diffusion property, not MDS).
    internal_linear_layer: Diffusion,
}

impl<F, MdsLight, Diffusion, const WIDTH: usize, const D: u64>
    Poseidon2<F, MdsLight, Diffusion, WIDTH, D>
where
    F: PrimeField,
{
    /// Create a new Poseidon2 configuration.
    pub fn new(
        rounds_f: usize,
        external_constants: Vec<[F; WIDTH]>,
        external_linear_layer: MdsLight,
        rounds_p: usize,
        internal_constants: Vec<F>,
        internal_linear_layer: Diffusion,
    ) -> Self {
        assert!(SUPPORTED_WIDTHS.contains(&WIDTH));
        Self {
            rounds_f,
            external_constants,
            external_linear_layer,
            rounds_p,
            internal_constants,
            internal_linear_layer,
        }
    }

    /// Create a new Poseidon2 configuration with random parameters.
    pub fn new_from_rng<R: Rng>(
        rounds_f: usize,
        external_linear_layer: MdsLight,
        rounds_p: usize,
        internal_linear_layer: Diffusion,
        rng: &mut R,
    ) -> Self
    where
        Standard: Distribution<F> + Distribution<[F; WIDTH]>,
    {
        let external_constants = rng
            .sample_iter(Standard)
            .take(rounds_f)
            .collect::<Vec<[F; WIDTH]>>();
        let internal_constants = rng.sample_iter(Standard).take(rounds_p).collect::<Vec<F>>();

        Self {
            rounds_f,
            external_constants,
            external_linear_layer,
            rounds_p,
            internal_constants,
            internal_linear_layer,
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

impl<F, MdsLight, Diffusion, const WIDTH: usize, const D: u64>
    Poseidon2<F, MdsLight, Diffusion, WIDTH, D>
where
    F: PrimeField64,
{
    /// Create a new Poseidon2 configuration with 128 bit security and random rounds constants.
    pub fn new_from_rng_128<R: Rng>(
        external_linear_layer: MdsLight,
        internal_linear_layer: Diffusion,
        rng: &mut R,
    ) -> Self
    where
        Standard: Distribution<F> + Distribution<[F; WIDTH]>,
    {
        let (rounds_f, rounds_p) = poseidon2_round_numbers_128::<F>(WIDTH, D);

        let external_constants = rng
            .sample_iter(Standard)
            .take(rounds_f)
            .collect::<Vec<[F; WIDTH]>>();
        let internal_constants = rng.sample_iter(Standard).take(rounds_p).collect::<Vec<F>>();

        Self {
            rounds_f,
            external_constants,
            external_linear_layer,
            rounds_p,
            internal_constants,
            internal_linear_layer,
        }
    }
}

impl<AF, MdsLight, Diffusion, const WIDTH: usize, const D: u64> Permutation<[AF; WIDTH]>
    for Poseidon2<AF::F, MdsLight, Diffusion, WIDTH, D>
where
    AF: AbstractField,
    AF::F: PrimeField,
    MdsLight: MdsLightPermutation<AF, WIDTH>,
    Diffusion: DiffusionPermutation<AF, WIDTH>,
{
    fn permute_mut(&self, state: &mut [AF; WIDTH]) {
        // The initial linear layer.
        self.external_linear_layer.permute_mut(state);

        // The first half of the external rounds.
        let rounds_f_half = self.rounds_f / 2;
        for r in 0..rounds_f_half {
            self.add_rc(state, &self.external_constants[r]);
            self.sbox(state);
            self.external_linear_layer.permute_mut(state);
        }

        // The internal rounds.
        for r in 0..self.rounds_p {
            state[0] += AF::from_f(self.internal_constants[r]);
            state[0] = self.sbox_p(&state[0]);
            self.internal_linear_layer.permute_mut(state);
        }

        // The second half of the external rounds.
        for r in rounds_f_half..self.rounds_f {
            self.add_rc(state, &self.external_constants[r]);
            self.sbox(state);
            self.external_linear_layer.permute_mut(state);
        }
    }
}

impl<AF, MdsLight, Diffusion, const WIDTH: usize, const D: u64>
    CryptographicPermutation<[AF; WIDTH]> for Poseidon2<AF::F, MdsLight, Diffusion, WIDTH, D>
where
    AF: AbstractField,
    AF::F: PrimeField,
    MdsLight: MdsLightPermutation<AF, WIDTH>,
    Diffusion: DiffusionPermutation<AF, WIDTH>,
{
}

/// A Poseidon2 abstraction allowing for fast Packed Field implementations.
/// F is the field from which the input variables are drawn.
/// T is the internal working type.
/// TODO: THIS WILL NEED CHANGES AS IMPLEMENTATIONS DEMAND.
/// EVENTUALLY WILL REPLACE MAIN POSEIDON2 TYPE
#[derive(Clone, Debug)]
pub struct Poseidon2Fast<F, T, U, const WIDTH: usize> {
    /// Convert from external to internal representation
    convert_to: fn([F; WIDTH]) -> T,

    /// Convert from internal to external representation
    convert_back: fn(T) -> [F; WIDTH],

    /// The external round constants for the initial set of external rounds.
    initial_external_constants: Vec<T>,

    /// The initial external layer.
    initial_external_layer: fn(&mut T, &[T]),

    /// The internal round constants.
    internal_constants: Vec<U>,

    /// The internal layer.
    internal_layer: fn(&mut T, &[U]),

    /// The external round constants for the final set of external rounds.
    final_external_constants: Vec<T>,

    /// The final external layer.
    final_external_layer: fn(&mut T, &[T]),
}

impl<F, T, U, const WIDTH: usize> Poseidon2Fast<F, T, U, WIDTH>
where
    F: PackedField,
    F::Scalar: PrimeField64,
{
    /// Create a new Poseidon2 configuration.
    /// TODO: Fix up this function.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        convert_to: fn([F; WIDTH]) -> T,
        convert_back: fn(T) -> [F; WIDTH],
        initial_external_constants: Vec<T>,
        initial_external_layer: fn(&mut T, &[T]),
        internal_constants: Vec<U>,
        internal_layer: fn(&mut T, &[U]),
        final_external_constants: Vec<T>,
        final_external_layer: fn(&mut T, &[T]),
    ) -> Self {
        // Need to determine supported widths later.
        // assert!(SUPPORTED_WIDTHS.contains(&WIDTH));
        Self {
            convert_to,
            convert_back,
            initial_external_constants,
            initial_external_layer,
            internal_constants,
            internal_layer,
            final_external_constants,
            final_external_layer,
        }
    }

    /// Create a new Poseidon2 configuration with 128 bit security and random rounds constants.
    pub fn new_from_rng_128<R: Rng, const D: u64>(
        convert_to: fn([F; WIDTH]) -> T,
        convert_back: fn(T) -> [F; WIDTH],
        to_u: fn(F::Scalar) -> U,
        initial_external_layer: fn(&mut T, &[T]),
        internal_layer: fn(&mut T, &[U]),
        final_external_layer: fn(&mut T, &[T]),
        rng: &mut R,
    ) -> Self
    where
        Standard: Distribution<F> + Distribution<[F; WIDTH]>,
        Standard: Distribution<F::Scalar> + Distribution<[F::Scalar; WIDTH]>,
    {
        let (rounds_f, rounds_p) = poseidon2_round_numbers_128::<F::Scalar>(F::WIDTH * WIDTH, D);
        let half_f = rounds_f / 2;

        let initial_external_constants = rng
            .sample_iter(Standard)
            .take(half_f)
            .map(convert_to)
            .collect();

        let final_external_constants = rng
            .sample_iter(Standard)
            .take(half_f)
            .map(convert_to)
            .collect();

        let internal_constants = rng.sample_iter(Standard).take(rounds_p).map(to_u).collect();

        Self {
            convert_to,
            convert_back,
            initial_external_constants,
            initial_external_layer,
            internal_constants,
            internal_layer,
            final_external_constants,
            final_external_layer,
        }
    }
}

impl<F, T, U, const WIDTH: usize> Permutation<[F; WIDTH]> for Poseidon2Fast<F, T, U, WIDTH>
where
    F: Clone + Copy,
    T: Clone + Copy + Sync,
    U: Clone + Sync,
{
    fn permute_mut(&self, state: &mut [F; WIDTH]) {
        let mut internal_rep = (self.convert_to)(*state);
        (self.initial_external_layer)(&mut internal_rep, &self.initial_external_constants);
        (self.internal_layer)(&mut internal_rep, &self.internal_constants);
        (self.final_external_layer)(&mut internal_rep, &self.final_external_constants);
        *state = (self.convert_back)(internal_rep)
    }
}

impl<F, T, U, const WIDTH: usize> CryptographicPermutation<[F; WIDTH]>
    for Poseidon2Fast<F, T, U, WIDTH>
where
    F: Clone + Copy,
    T: Clone + Copy + Sync,
    U: Clone + Sync,
{
}
