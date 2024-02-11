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
use core::marker::Sync;

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
    AF: AbstractField + Sync,
    AF::F: PrimeField,
    Diffusion: DiffusionPermutation<AF, WIDTH>,
{
    fn permute_mut(&self, state: &mut [AF; WIDTH]) {
        let external_linear_layer = Poseidon2MEMatrix::<AF, WIDTH, D>::new();

        // The initial linear layer.
        external_linear_layer.permute_mut(state);

        // The first half of the external rounds.
        let rounds = self.rounds_f + self.rounds_p;
        let rounds_f_beginning = self.rounds_f / 2;
        for r in 0..rounds_f_beginning {
            self.add_rc(state, &self.constants[r]);
            self.sbox(state);
            external_linear_layer.permute_mut(state);
        }

        // The internal rounds.
        let p_end = rounds_f_beginning + self.rounds_p;
        for r in rounds_f_beginning..p_end {
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
    AF: AbstractField + Sync,
    AF::F: PrimeField,
    Diffusion: DiffusionPermutation<AF, WIDTH>,
{
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use ark_ff::{BigInteger, PrimeField};
    use p3_baby_bear::BabyBear;
    use p3_field::AbstractField;
    use p3_goldilocks::Goldilocks;
    use p3_symmetric::Permutation;
    use rand::Rng;
    use zkhash::fields::babybear::FpBabyBear;
    use zkhash::fields::goldilocks::FpGoldiLocks;
    use zkhash::poseidon2::poseidon2::Poseidon2 as Poseidon2Ref;
    use zkhash::poseidon2::poseidon2_instance_babybear::{POSEIDON2_BABYBEAR_16_PARAMS, RC16};
    use zkhash::poseidon2::poseidon2_instance_goldilocks::{
        POSEIDON2_GOLDILOCKS_12_PARAMS, POSEIDON2_GOLDILOCKS_8_PARAMS, RC12, RC8,
    };

    use crate::goldilocks::DiffusionMatrixGoldilocks;
    use crate::{DiffusionMatrixBabybear, Poseidon2};

    fn goldilocks_from_ark_ff(input: FpGoldiLocks) -> Goldilocks {
        let as_bigint = input.into_bigint();
        let mut as_bytes = as_bigint.to_bytes_le();
        as_bytes.resize(8, 0);
        let as_u64 = u64::from_le_bytes(as_bytes[0..8].try_into().unwrap());
        Goldilocks::from_wrapped_u64(as_u64)
    }

    fn babybear_from_ark_ff(input: FpBabyBear) -> BabyBear {
        let as_bigint = input.into_bigint();
        let mut as_bytes = as_bigint.to_bytes_le();
        as_bytes.resize(4, 0);
        let as_u32 = u32::from_le_bytes(as_bytes[0..4].try_into().unwrap());
        BabyBear::from_wrapped_u32(as_u32)
    }

    #[test]
    fn test_poseidon2_goldilocks_width_8() {
        const WIDTH: usize = 8;
        const D: u64 = 7;
        const ROUNDS_F: usize = 8;
        const ROUNDS_P: usize = 22;

        type F = Goldilocks;

        let mut rng = rand::thread_rng();

        // Poiseidon2 reference implementation from zkhash repo.
        let poseidon2_ref = Poseidon2Ref::new(&POSEIDON2_GOLDILOCKS_8_PARAMS);

        // Copy over round constants from zkhash.
        let round_constants: Vec<[F; WIDTH]> = RC8
            .iter()
            .map(|vec| {
                vec.iter()
                    .cloned()
                    .map(goldilocks_from_ark_ff)
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap()
            })
            .collect();

        // Our Poseidon2 implementation.
        let poseidon2: Poseidon2<Goldilocks, DiffusionMatrixGoldilocks, WIDTH, D> = Poseidon2::new(
            ROUNDS_F,
            ROUNDS_P,
            round_constants,
            DiffusionMatrixGoldilocks,
        );

        // Generate random input and convert to both Goldilocks field formats.
        let input_u64 = rng.gen::<[u64; WIDTH]>();
        let input_ref = input_u64
            .iter()
            .cloned()
            .map(FpGoldiLocks::from)
            .collect::<Vec<_>>();
        let input = input_u64.map(F::from_wrapped_u64);

        // Check that the conversion is correct.
        assert!(input_ref
            .iter()
            .zip(input.iter())
            .all(|(a, b)| goldilocks_from_ark_ff(*a) == *b));

        // Run reference implementation.
        let output_ref = poseidon2_ref.permutation(&input_ref);
        let expected: [F; WIDTH] = output_ref
            .iter()
            .cloned()
            .map(goldilocks_from_ark_ff)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        // Run our implementation.
        let mut output = input;
        poseidon2.permute_mut(&mut output);

        assert_eq!(output, expected);
    }

    #[test]
    fn test_poseidon2_goldilocks_width_12() {
        const WIDTH: usize = 12;
        const D: u64 = 7;
        const ROUNDS_F: usize = 8;
        const ROUNDS_P: usize = 22;

        type F = Goldilocks;

        let mut rng = rand::thread_rng();

        // Poiseidon2 reference implementation from zkhash repo.
        let poseidon2_ref = Poseidon2Ref::new(&POSEIDON2_GOLDILOCKS_12_PARAMS);

        // Copy over round constants from zkhash.
        let round_constants: Vec<[F; WIDTH]> = RC12
            .iter()
            .map(|vec| {
                vec.iter()
                    .cloned()
                    .map(goldilocks_from_ark_ff)
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap()
            })
            .collect();

        // Our Poseidon2 implementation.
        let poseidon2: Poseidon2<Goldilocks, DiffusionMatrixGoldilocks, WIDTH, D> = Poseidon2::new(
            ROUNDS_F,
            ROUNDS_P,
            round_constants,
            DiffusionMatrixGoldilocks,
        );

        // Generate random input and convert to both Goldilocks field formats.
        let input_u64 = rng.gen::<[u64; WIDTH]>();
        let input_ref = input_u64
            .iter()
            .cloned()
            .map(FpGoldiLocks::from)
            .collect::<Vec<_>>();
        let input = input_u64.map(F::from_wrapped_u64);

        // Check that the conversion is correct.
        assert!(input_ref
            .iter()
            .zip(input.iter())
            .all(|(a, b)| goldilocks_from_ark_ff(*a) == *b));

        // Run reference implementation.
        let output_ref = poseidon2_ref.permutation(&input_ref);
        let expected: [F; WIDTH] = output_ref
            .iter()
            .cloned()
            .map(goldilocks_from_ark_ff)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        // Run our implementation.
        let mut output = input;
        poseidon2.permute_mut(&mut output);

        assert_eq!(output, expected);
    }

    #[test]
    fn test_poseidon2_babybear_width_16() {
        const WIDTH: usize = 16;
        const D: u64 = 7;
        const ROUNDS_F: usize = 8;
        const ROUNDS_P: usize = 13;

        type F = BabyBear;

        let mut rng = rand::thread_rng();

        // Poiseidon2 reference implementation from zkhash repo.
        let poseidon2_ref = Poseidon2Ref::new(&POSEIDON2_BABYBEAR_16_PARAMS);

        // Copy over round constants from zkhash.
        let round_constants: Vec<[F; WIDTH]> = RC16
            .iter()
            .map(|vec| {
                vec.iter()
                    .cloned()
                    .map(babybear_from_ark_ff)
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap()
            })
            .collect();

        // Our Poseidon2 implementation.
        let poseidon2: Poseidon2<BabyBear, DiffusionMatrixBabybear, WIDTH, D> =
            Poseidon2::new(ROUNDS_F, ROUNDS_P, round_constants, DiffusionMatrixBabybear);

        // Generate random input and convert to both BabyBear field formats.
        let input_u32 = rng.gen::<[u32; WIDTH]>();
        let input_ref = input_u32
            .iter()
            .cloned()
            .map(FpBabyBear::from)
            .collect::<Vec<_>>();
        let input = input_u32.map(F::from_wrapped_u32);

        // Check that the conversion is correct.
        assert!(input_ref
            .iter()
            .zip(input.iter())
            .all(|(a, b)| babybear_from_ark_ff(*a) == *b));

        // Run reference implementation.
        let output_ref = poseidon2_ref.permutation(&input_ref);
        let expected: [F; WIDTH] = output_ref
            .iter()
            .cloned()
            .map(babybear_from_ark_ff)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        // Run our implementation.
        let mut output = input;
        poseidon2.permute_mut(&mut output);

        assert_eq!(output, expected);
    }
}
