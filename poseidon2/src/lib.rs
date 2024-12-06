//! The Poseidon2 permutation.
//!
//! This implementation was based upon the following resources:
//! - https://github.com/HorizenLabs/poseidon2/blob/main/plain_implementations/src/poseidon2/poseidon2.rs
//! - https://eprint.iacr.org/2023/323.pdf

#![no_std]

extern crate alloc;

mod external;
mod generic;
mod internal;
mod round_numbers;
use alloc::vec::Vec;
use core::marker::PhantomData;

pub use external::*;
pub use generic::*;
pub use internal::*;
use p3_field::{Field, FieldAlgebra, InjectiveMonomial, PrimeField, PrimeField64};
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

impl<FA, ExternalPerm, InternalPerm, const WIDTH: usize, const D: u64>
    Poseidon2<FA, ExternalPerm, InternalPerm, WIDTH, D>
where
    FA: FieldAlgebra,
    FA::F: PrimeField,
    ExternalPerm: ExternalLayerConstructor<FA, WIDTH>,
    InternalPerm: InternalLayerConstructor<FA>,
{
    /// Create a new Poseidon2 configuration.
    /// This internally converts the given constants to the relevant packed versions.
    pub fn new(
        external_constants: ExternalLayerConstants<FA::F, WIDTH>,
        internal_constants: Vec<FA::F>,
    ) -> Self {
        assert!(SUPPORTED_WIDTHS.contains(&WIDTH));
        let external_layer = ExternalPerm::new_from_constants(external_constants);
        let internal_layer = InternalPerm::new_from_constants(internal_constants);

        Self {
            external_layer,
            internal_layer,
            _phantom: PhantomData,
        }
    }

    /// Create a new Poseidon2 configuration with random parameters.
    pub fn new_from_rng<R: Rng>(rounds_f: usize, rounds_p: usize, rng: &mut R) -> Self
    where
        Standard: Distribution<FA::F> + Distribution<[FA::F; WIDTH]>,
    {
        let external_constants = ExternalLayerConstants::new_from_rng(rounds_f, rng);
        let internal_constants = rng.sample_iter(Standard).take(rounds_p).collect();

        Self::new(external_constants, internal_constants)
    }
}

impl<FA, ExternalPerm, InternalPerm, const WIDTH: usize, const D: u64>
    Poseidon2<FA, ExternalPerm, InternalPerm, WIDTH, D>
where
    FA: FieldAlgebra,
    FA::F: PrimeField64,
    ExternalPerm: ExternalLayerConstructor<FA, WIDTH>,
    InternalPerm: InternalLayerConstructor<FA>,
{
    /// Create a new Poseidon2 configuration with 128 bit security and random rounds constants.
    pub fn new_from_rng_128<R: Rng>(rng: &mut R) -> Self
    where
        Standard: Distribution<FA::F> + Distribution<[FA::F; WIDTH]>,
    {
        let (rounds_f, rounds_p) = poseidon2_round_numbers_128::<FA::F>(WIDTH, D);
        Self::new_from_rng(rounds_f, rounds_p, rng)
    }
}

impl<FA, ExternalPerm, InternalPerm, const WIDTH: usize, const D: u64> Permutation<[FA; WIDTH]>
    for Poseidon2<<FA::F as Field>::Packing, ExternalPerm, InternalPerm, WIDTH, D>
where
    FA: FieldAlgebra + Sync + InjectiveMonomial<D>,
    FA::F: PrimeField + InjectiveMonomial<D>,
    ExternalPerm: ExternalLayer<FA, WIDTH, D>,
    InternalPerm: InternalLayer<FA, WIDTH, D>,
{
    fn permute_mut(&self, state: &mut [FA; WIDTH]) {
        self.external_layer.permute_state_initial(state);
        self.internal_layer.permute_state(state);
        self.external_layer.permute_state_terminal(state);
    }
}

impl<FA, ExternalPerm, InternalPerm, const WIDTH: usize, const D: u64>
    CryptographicPermutation<[FA; WIDTH]>
    for Poseidon2<<FA::F as Field>::Packing, ExternalPerm, InternalPerm, WIDTH, D>
where
    FA: FieldAlgebra + Sync + InjectiveMonomial<D>,
    FA::F: PrimeField + InjectiveMonomial<D>,
    ExternalPerm: ExternalLayer<FA, WIDTH, D>,
    InternalPerm: InternalLayer<FA, WIDTH, D>,
{
}
