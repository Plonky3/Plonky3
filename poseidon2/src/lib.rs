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

pub use constants::*;
pub use diffusion::*;
pub use matrix::*;
use p3_field::{AbstractField, Field, PrimeField, PrimeField64};
use p3_symmetric::{CryptographicPermutation, Permutation};
use rand::distributions::{Distribution, Standard};
use rand::Rng;
pub use round_numbers::poseidon2_round_numbers_128;

const SUPPORTED_WIDTHS: [usize; 8] = [2, 3, 4, 8, 12, 16, 20, 24];

/// The Poseidon2 permutation.
#[derive(Clone, Debug)]
pub struct Poseidon2<F, ExternalPerm, InternalPerm, const WIDTH: usize, const D: u64>
where
    F: Field,
    ExternalPerm: Poseidon2ExternalPackedConstants<F, WIDTH>,
    InternalPerm: Poseidon2InternalPackedConstants<F>,
{
    /// The external round constants.
    external_constants: [Vec<[F; WIDTH]>; 2],

    /// The external round constants converted for optimal use with PackedFields.
    external_packed_constants: [Vec<ExternalPerm::ConstantsType>; 2],

    /// The permutations used in External Rounds.
    external_layer: ExternalPerm,

    /// The internal round constants.
    internal_constants: Vec<F>,

    /// The internal round constants converted for optimal use with PackedFields.
    internal_packed_constants: Vec<InternalPerm::ConstantsType>,

    /// The permutation used in Internal Rounds.
    internal_layer: InternalPerm,
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
    pub fn new(
        external_constants: [Vec<[F; WIDTH]>; 2],
        external_layer: ExternalPerm,
        internal_constants: Vec<F>,
        internal_layer: InternalPerm,
    ) -> Self {
        assert!(SUPPORTED_WIDTHS.contains(&WIDTH));
        let external_packed_constants = ExternalPerm::convert_from_field_array_list([
            &external_constants[0],
            &external_constants[1],
        ]);

        let internal_packed_constants = InternalPerm::convert_from_field_list(&internal_constants);

        Self {
            external_constants,
            external_packed_constants,
            external_layer,
            internal_constants,
            internal_packed_constants,
            internal_layer,
        }
    }

    /// Create a new Poseidon2 configuration with random parameters.
    pub fn new_from_rng<R: Rng>(
        rounds_f: usize,
        external_layer: ExternalPerm,
        rounds_p: usize,
        internal_layer: InternalPerm,
        rng: &mut R,
    ) -> Self
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

        Self::new(
            external_constants,
            external_layer,
            internal_constants,
            internal_layer,
        )
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
    pub fn new_from_rng_128<R: Rng>(
        external_layer: ExternalPerm,
        internal_layer: InternalPerm,
        rng: &mut R,
    ) -> Self
    where
        Standard: Distribution<F> + Distribution<[F; WIDTH]>,
    {
        let (rounds_f, rounds_p) = poseidon2_round_numbers_128::<F>(WIDTH, D);
        Self::new_from_rng(rounds_f, external_layer, rounds_p, internal_layer, rng)
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
        let mut internal_state = self.external_layer.permute_state_initial(
            state,
            &self.external_constants[0],
            &self.external_packed_constants[0],
        );

        self.internal_layer.permute_state(
            &mut internal_state,
            &self.internal_constants,
            &self.internal_packed_constants,
        );

        self.external_layer.permute_state_final(
            internal_state,
            &self.external_constants[1],
            &self.external_packed_constants[1],
        )
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
