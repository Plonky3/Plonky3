//! The Poseidon2 permutation.
//!
//! This implementation was based upon the following resources:
//! - https://github.com/HorizenLabs/poseidon2/blob/main/plain_implementations/src/poseidon2/poseidon2.rs
//! - https://eprint.iacr.org/2023/323.pdf

#![no_std]

extern crate alloc;

mod constants;
mod diffusion;
mod matrix;
mod round_numbers;
use alloc::vec::Vec;
use core::marker::PhantomData;

pub use constants::*;
pub use diffusion::*;
pub use matrix::*;
use p3_field::{AbstractField, PrimeField, PrimeField64};
use p3_symmetric::{CryptographicPermutation, Permutation};
use rand::distributions::{Distribution, Standard};
use rand::Rng;
pub use round_numbers::poseidon2_round_numbers_128;

const SUPPORTED_WIDTHS: [usize; 8] = [2, 3, 4, 8, 12, 16, 20, 24];

/// The Poseidon2 permutation.
#[derive(Clone, Debug)]
pub struct Poseidon2<F, ExternalPerm, InternalPerm, const WIDTH: usize, const D: u64> {
    /// The permutations used in External Rounds.
    external_layer: ExternalPerm,

    /// The permutation used in Internal Rounds.
    internal_layer: InternalPerm,

    _phantom: PhantomData<F>,
}

impl<F, ExternalPerm, InternalPerm, const WIDTH: usize, const D: u64>
    Poseidon2<F, ExternalPerm, InternalPerm, WIDTH, D>
where
    F: PrimeField,
    ExternalPerm: Poseidon2ExternalPackedConstants<F, WIDTH>,
    InternalPerm: Poseidon2InternalPackedConstants<F>,
{
    /// Create a new Poseidon2 configuration.
    /// This internally converts the given constants to the relevant packed versions.
    pub fn new(external_constants: [Vec<[F; WIDTH]>; 2], internal_constants: Vec<F>) -> Self {
        assert!(SUPPORTED_WIDTHS.contains(&WIDTH));
        let external_layer = ExternalPerm::convert_from_field_array(external_constants);
        let internal_layer = InternalPerm::convert_from_field(internal_constants);

        Self {
            external_layer,
            internal_layer,
            _phantom: PhantomData,
        }
    }

    /// Create a new Poseidon2 configuration with random parameters.
    pub fn new_from_rng<R: Rng>(rounds_f: usize, rounds_p: usize, rng: &mut R) -> Self
    where
        Standard: Distribution<F> + Distribution<[F; WIDTH]>,
    {
        let half_f = rounds_f / 2;
        assert_eq!(
            2 * half_f,
            rounds_f,
            "The total number of external rounds should be even"
        );
        let init_external_constants = rng.sample_iter(Standard).take(half_f).collect();
        let final_external_constants = rng.sample_iter(Standard).take(half_f).collect();
        let external_constants = [init_external_constants, final_external_constants];
        let internal_constants = rng.sample_iter(Standard).take(rounds_p).collect::<Vec<F>>();

        Self::new(external_constants, internal_constants)
    }
}

impl<F, ExternalPerm, InternalPerm, const WIDTH: usize, const D: u64>
    Poseidon2<F, ExternalPerm, InternalPerm, WIDTH, D>
where
    F: PrimeField64,
    ExternalPerm: Poseidon2ExternalPackedConstants<F, WIDTH>,
    InternalPerm: Poseidon2InternalPackedConstants<F>,
{
    /// Create a new Poseidon2 configuration with 128 bit security and random rounds constants.
    pub fn new_from_rng_128<R: Rng>(rng: &mut R) -> Self
    where
        Standard: Distribution<F> + Distribution<[F; WIDTH]>,
    {
        let (rounds_f, rounds_p) = poseidon2_round_numbers_128::<F>(WIDTH, D);
        Self::new_from_rng(rounds_f, rounds_p, rng)
    }
}

impl<AF, ExternalPerm, InternalPerm, const WIDTH: usize, const D: u64> Permutation<[AF; WIDTH]>
    for Poseidon2<AF::F, ExternalPerm, InternalPerm, WIDTH, D>
where
    AF: AbstractField,
    AF::F: PrimeField,
    ExternalPerm: ExternalLayer<AF, WIDTH, D>,
    InternalPerm: InternalLayer<AF, WIDTH, D, InternalState = ExternalPerm::InternalState>,
{
    fn permute(&self, state: [AF; WIDTH]) -> [AF; WIDTH] {
        let mut internal_state = self.external_layer.permute_state_initial(state);

        self.internal_layer.permute_state(&mut internal_state);

        self.external_layer.permute_state_final(internal_state)
    }

    fn permute_mut(&self, state: &mut [AF; WIDTH]) {
        *state = self.permute((*state).clone())
    }
}

impl<AF, ExternalPerm, InternalPerm, const WIDTH: usize, const D: u64>
    CryptographicPermutation<[AF; WIDTH]> for Poseidon2<AF::F, ExternalPerm, InternalPerm, WIDTH, D>
where
    AF: AbstractField,
    AF::F: PrimeField,
    ExternalPerm: ExternalLayer<AF, WIDTH, D>,
    InternalPerm: InternalLayer<AF, WIDTH, D, InternalState = ExternalPerm::InternalState>,
{
}
