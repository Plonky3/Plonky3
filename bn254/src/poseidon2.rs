//! Diffusion matrix for Goldilocks3
//!
//! Reference: https://github.com/HorizenLabs/poseidon2/blob/main/plain_implementations/src/poseidon2/poseidon2_instance_bn256.rs

use alloc::vec::Vec;
use alloc::vec;

use p3_field::AbstractField;
use p3_poseidon2::{matmul_internal, DiffusionPermutation};
use p3_symmetric::Permutation;

use lazy_static::lazy_static;

use crate::BN254;

lazy_static! {
    pub static ref MAT_DIAG3_M_1: Vec<BN254> = vec![
        BN254::one(),
        BN254::one(),
        BN254::two(),
    ];
}

#[derive(Debug, Clone, Default)]
pub struct DiffusionMatrixBN254;

impl<AF: AbstractField<F = BN254>> Permutation<[AF; 3]> for DiffusionMatrixBN254 {
    fn permute_mut(&self, state: &mut [AF; 3]) {
        matmul_internal::<BN254, AF, 3>(state, MAT_DIAG3_M_1.as_slice().try_into().unwrap());
    }
}

impl<AF: AbstractField<F = BN254>> DiffusionPermutation<AF, 3> for DiffusionMatrixBN254{}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use p3_poseidon2::Poseidon2;
    use rand::Rng;
    use zkhash::fields::bn256::FpBN256;
    use zkhash::poseidon2::poseidon2::Poseidon2 as Poseidon2Ref;
    use zkhash::poseidon2::poseidon2_instance_bn256::{POSEIDON2_BN256_PARAMS, RC3};

    use super::*;


    fn bn254_from_ark_ff(input: FpBN256) -> BN254 {
        BN254 { value: input }
    }

    #[test]
    fn test_poseidon2_bn254() {
        const WIDTH: usize = 3;
        const D: u64 = 5;
        const ROUNDS_F: usize = 8;
        const ROUNDS_P: usize = 56;

        type F = BN254;

        let mut rng = rand::thread_rng();

        // Poiseidon2 reference implementation from zkhash repo.
        let poseidon2_ref = Poseidon2Ref::new(&POSEIDON2_BN256_PARAMS);

        // Copy over round constants from zkhash.
        let round_constants: Vec<[F; WIDTH]> = RC3
            .iter()
            .map(|vec| {
                vec.iter()
                    .cloned()
                    .map(bn254_from_ark_ff)
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap()
            })
            .collect();

        // Our Poseidon2 implementation.
        let poseidon2: Poseidon2<BN254, DiffusionMatrixBN254, WIDTH, D> = Poseidon2::new(
            ROUNDS_F,
            ROUNDS_P,
            round_constants,
            DiffusionMatrixBN254,
        );

        // Generate random input and convert to both Goldilocks field formats.
        let input_ark_ff: [FpBN256; WIDTH] = rng.gen::<[FpBN256; WIDTH]>();
        let input = input_ark_ff
            .iter()
            .cloned()
            .map(bn254_from_ark_ff)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        // Run reference implementation.
        let output_ref = poseidon2_ref.permutation(&input_ark_ff);

        let expected: [F; WIDTH] = output_ref
            .iter()
            .cloned()
            .map(bn254_from_ark_ff)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        // Run our implementation.
        let mut output = input;
        poseidon2.permute_mut(&mut output);

        assert_eq!(output, expected);
    }
}
