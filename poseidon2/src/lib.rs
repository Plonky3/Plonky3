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
mod matrix;
use alloc::vec::Vec;

pub use babybear::DiffusionMatrixBabybear;
pub use diffusion::DiffusionPermutation;
pub use goldilocks::DiffusionMatrixGoldilocks;
use matrix::Poseidon2MEMatrix;
use p3_field::{AbstractField, PrimeField};
use p3_symmetric::{CryptographicPermutation, Permutation};
use rand::distributions::Standard;
use rand::prelude::Distribution;
use rand::Rng;

const SUPPORTED_WIDTHS: [usize; 8] = [2, 3, 4, 8, 12, 16, 20, 24];

/// The Poseidon2 permutation.
#[derive(Clone)]
pub struct Poseidon2<F, Diffusion, const WIDTH: usize, const D: u64> {
    /// The number of external rounds.
    rounds_f: usize,

    /// The number of internal rounds.
    rounds_p: usize,

    /// The round constants.
    constants: Vec<[F; WIDTH]>,

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
        rounds_p: usize,
        constants: Vec<[F; WIDTH]>,
        internal_linear_layer: Diffusion,
    ) -> Self {
        assert!(SUPPORTED_WIDTHS.contains(&WIDTH));
        Self {
            rounds_f,
            rounds_p,
            constants,
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
        let mut constants = Vec::new();
        let rounds = rounds_f + rounds_p;
        for _ in 0..rounds {
            constants.push(rng.gen::<[F; WIDTH]>());
        }

        Self {
            rounds_f,
            rounds_p,
            constants,
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
        let external_linear_layer = Poseidon2MEMatrix::<AF, WIDTH, D>::new();

        // The initial linear layer.
        external_linear_layer.permute_mut(state);

        // The first half of the external rounds.
        let rounds = self.rounds_f + self.rounds_p;
        let rounds_f_beggining = self.rounds_f / 2;
        for r in 0..rounds_f_beggining {
            self.add_rc(state, &self.constants[r]);
            self.sbox(state);
            external_linear_layer.permute_mut(state);
        }

        // The internal rounds.
        let p_end = rounds_f_beggining + self.rounds_p;
        for r in self.rounds_f..p_end {
            state[0] += AF::from_f(self.constants[r][0]);
            state[0] = self.sbox_p(&state[0]);
            self.internal_linear_layer.permute_mut(state);
        }

        // The second half of the external rounds.
        for r in p_end..rounds {
            self.add_rc(state, &self.constants[r]);
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

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;
    use p3_field::AbstractField;
    use p3_symmetric::Permutation;
    use rand::Rng;
    use zkhash::fields::goldilocks::FpGoldiLocks;
    use ark_ff::PrimeField;
    use ark_ff::BigInteger;
    use zkhash::poseidon2::poseidon2::Poseidon2 as Poseidon2Ref;
    use zkhash::poseidon2::poseidon2_instance_goldilocks::{POSEIDON2_GOLDILOCKS_8_PARAMS, RC8};
    use p3_goldilocks::Goldilocks;
    use crate::goldilocks::DiffusionMatrixGoldilocks;

    use crate::Poseidon2;

    type F = Goldilocks;

    fn goldilocks_from_ark_ff(input: FpGoldiLocks) -> F {
        let as_bigint = input.into_bigint();
        let mut as_bytes = as_bigint.to_bytes_le();
        as_bytes.resize(8, 0);
        let as_u64 = u64::from_le_bytes(as_bytes[0..8].try_into().unwrap());
        F::from_wrapped_u64(as_u64)
    }

    #[test]
    fn test_poseidon2_goldilocks() {
        const WIDTH: usize = 8;
        const D: u64 = 7;
        const ROUNDS_F: usize = 8;
        const ROUNDS_P: usize = 22;

        let mut rng = rand::thread_rng();

        let poseidon2_ref = Poseidon2Ref::new(&POSEIDON2_GOLDILOCKS_8_PARAMS);

        let round_constants: Vec<[F; WIDTH]> = RC8.iter().map(|vec| {
            vec.iter().cloned().map(goldilocks_from_ark_ff).collect::<Vec<_>>().try_into().unwrap()
        }).collect();

        let poseidon2: Poseidon2<Goldilocks, DiffusionMatrixGoldilocks, WIDTH, D> = Poseidon2::new(
            ROUNDS_F,
            ROUNDS_P,
            round_constants,
            DiffusionMatrixGoldilocks,
        );

        let random_input_u64 = rng.gen::<[u64; WIDTH]>();
        let random_input_ref = random_input_u64.iter().cloned().map(FpGoldiLocks::from).collect::<Vec<_>>();
        let random_input = random_input_u64.iter().cloned().map(F::from_wrapped_u64).collect::<Vec<_>>();

        let ref_output = poseidon2_ref.permutation(&random_input_ref);
        let ref_output_converted = ref_output.iter().cloned().map(goldilocks_from_ark_ff).collect::<Vec<_>>();
        let ref_output_converted_arr: [F; WIDTH] = ref_output_converted.try_into().unwrap();

        let mut output = random_input.clone().try_into().unwrap();
        poseidon2.permute_mut(&mut output);

        assert_eq!(output, ref_output_converted_arr);
    }

}