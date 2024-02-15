use p3_symmetric::Permutation;
use p3_poseidon2::DiffusionPermutation;

use crate::{BabyBear, sum_u64};

// Diffusion matrices for Babybear16 and Babybear24.
//
// Reference: https://github.com/HorizenLabs/poseidon2/blob/main/plain_implementations/src/poseidon2/poseidon2_instance_babybear.rs
// We save them here in their MONTY forms. There true forms can be found in the test section where we test to ensur the MONTY
// form is correct.

const MATRIX_DIAG_16_BABYBEAR_MONTY: [u32; 16] = [
    0x09d7163c, 0x65f27816, 0x5e6f3a6d, 0x64f91652, 0x323b0ccc, 0x63b5320b, 0x5691d672, 0x47e6031d,
    0x119401a9, 0x3f642f5c, 0x0f2c9b71, 0x07f81d96, 0x4fecc2e6, 0x4c13e496, 0x13dd1883, 0x70a6dddc,
];

const MATRIX_DIAG_24_BABYBEAR_MONTY: [u32; 24] = [
    0x3e41b357, 0x1033ed98, 0x49cf6705, 0x5281f318, 0x3c4ec8ee, 0x1967a15c, 0x3ab8089a, 0x68f1a534,
    0x11e010d5, 0x0d63d87e, 0x50d79dd3, 0x4a6bc505, 0x6f46057c, 0x437dd156, 0x60b4b484, 0x440c54cb,
    0x4a155e0a, 0x6a059603, 0x46382d00, 0x1f9b9cab, 0x1ce078ad, 0x12d7bd14, 0x5e4294b4, 0x1b314dce,
];

fn matmul_internal<const WIDTH: usize>(
    state: &mut [BabyBear; WIDTH],
    mat_internal_diag_m_1: [u32; WIDTH],
) {
    let sum = sum_u64(state);
    for i in 0..WIDTH {
        state[i] *= BabyBear{ value: mat_internal_diag_m_1[i] };
        state[i] += sum.clone();
    }
}

#[derive(Debug, Clone, Default)]
pub struct DiffusionMatrixBabybear;

impl Permutation<[BabyBear; 16]> for DiffusionMatrixBabybear {
    fn permute_mut(&self, state: &mut [BabyBear; 16]) {
        matmul_internal::<16>(state, MATRIX_DIAG_16_BABYBEAR_MONTY);
    }
}

impl DiffusionPermutation<BabyBear, 16> for DiffusionMatrixBabybear {}

impl Permutation<[BabyBear; 24]> for DiffusionMatrixBabybear {
    fn permute_mut(&self, state: &mut [BabyBear; 24]) {
        matmul_internal::<24>(state, MATRIX_DIAG_24_BABYBEAR_MONTY);
    }
}

impl DiffusionPermutation<BabyBear, 24> for DiffusionMatrixBabybear {}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;
    use p3_field::AbstractField;
    use p3_symmetric::Permutation;
    use p3_poseidon2::Poseidon2;
    use rand::Rng;
    use ark_ff::{BigInteger, PrimeField};
    use zkhash::fields::babybear::FpBabyBear;
    use zkhash::poseidon2::poseidon2::Poseidon2 as Poseidon2Ref;
    use zkhash::poseidon2::poseidon2_instance_babybear::{POSEIDON2_BABYBEAR_16_PARAMS, RC16};

    use super::{
        BabyBear, DiffusionMatrixBabybear, MATRIX_DIAG_16_BABYBEAR_MONTY,
        MATRIX_DIAG_24_BABYBEAR_MONTY,
    };

    // These are currently saved as their true values. It will be far more efficient to save them in Monty Form.
    const MATRIX_DIAG_16_BABYBEAR: [u32; 16] = [
        0x0a632d94, 0x6db657b7, 0x56fbdc9e, 0x052b3d8a, 0x33745201, 0x5c03108c, 0x0beba37b, 0x258c2e8b,
        0x12029f39, 0x694909ce, 0x6d231724, 0x21c3b222, 0x3c0904a5, 0x01d6acda, 0x27705c83, 0x5231c802,
    ];

    const MATRIX_DIAG_24_BABYBEAR: [u32; 24] = [
        0x409133f0, 0x1667a8a1, 0x06a6c7b6, 0x6f53160e, 0x273b11d1, 0x03176c5d, 0x72f9bbf9, 0x73ceba91,
        0x5cdef81d, 0x01393285, 0x46daee06, 0x065d7ba6, 0x52d72d6f, 0x05dd05e0, 0x3bab4b63, 0x6ada3842,
        0x2fc5fbec, 0x770d61b0, 0x5715aae9, 0x03ef0e90, 0x75b6c770, 0x242adf5f, 0x00d0ca4c, 0x36c0e388,
    ];

    #[test]
    fn const_16() {
        let monty_constant = MATRIX_DIAG_16_BABYBEAR
            .map(BabyBear::from_canonical_u32)
            .map(|x| x.value);

        assert_eq!(monty_constant, MATRIX_DIAG_16_BABYBEAR_MONTY);
    }

    #[test]
    fn const_24() {
        let monty_constant = MATRIX_DIAG_24_BABYBEAR
            .map(BabyBear::from_canonical_u32)
            .map(|x| x.value);

        assert_eq!(monty_constant, MATRIX_DIAG_24_BABYBEAR_MONTY);
    }

    fn babybear_from_ark_ff(input: FpBabyBear) -> BabyBear {
        let as_bigint = input.into_bigint();
        let mut as_bytes = as_bigint.to_bytes_le();
        as_bytes.resize(4, 0);
        let as_u32 = u32::from_le_bytes(as_bytes[0..4].try_into().unwrap());
        BabyBear::from_wrapped_u32(as_u32)
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
