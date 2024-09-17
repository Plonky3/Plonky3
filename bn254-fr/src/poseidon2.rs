//! Diffusion matrix for Bn254
//!
//! Reference: https://github.com/HorizenLabs/poseidon2/blob/main/plain_implementations/src/poseidon2/poseidon2_instance_bn256.rs

use std::sync::OnceLock;

use p3_field::AbstractField;
use p3_poseidon2::{
    external_final_permute_state, external_initial_permute_state, internal_permute_state,
    matmul_internal, ExternalLayer, ExternalLayerConstants, ExternalLayerConstructor, HLMDSMat4,
    InternalLayer, InternalLayerConstructor, Poseidon2,
};

use crate::Bn254Fr;

// As p - 1 is divisible by 3 the smallest D which satisfies gcd(p - 1, D) = 1 is 5.
const BN254_S_BOX_DEGREE: u64 = 5;

/// Poseidon2Bn254 contains the implementation of Poseidon2 for the Bn254Fr field.
/// It acts on arrays of the form [Bn254Fr; WIDTH].
pub type Poseidon2Bn254<const WIDTH: usize> = Poseidon2<
    Bn254Fr,
    Poseidon2ExternalLayerBn254<WIDTH>,
    Poseidon2InternalLayerBn254,
    WIDTH,
    BN254_S_BOX_DEGREE,
>;

#[inline]
fn get_diffusion_matrix_3() -> &'static [Bn254Fr; 3] {
    static MAT_DIAG3_M_1: OnceLock<[Bn254Fr; 3]> = OnceLock::new();
    MAT_DIAG3_M_1.get_or_init(|| [Bn254Fr::one(), Bn254Fr::one(), Bn254Fr::two()])
}

#[derive(Debug, Clone, Default)]
pub struct Poseidon2InternalLayerBn254 {
    internal_constants: Vec<Bn254Fr>,
}

impl InternalLayerConstructor<Bn254Fr> for Poseidon2InternalLayerBn254 {
    fn new_from_constants(internal_constants: Vec<Bn254Fr>) -> Self {
        Self { internal_constants }
    }
}

impl InternalLayer<Bn254Fr, 3, 5> for Poseidon2InternalLayerBn254 {
    type InternalState = [Bn254Fr; 3];

    fn permute_state(&self, state: &mut Self::InternalState) {
        internal_permute_state::<Bn254Fr, 3, 5>(
            state,
            |x| matmul_internal(x, *get_diffusion_matrix_3()),
            &self.internal_constants,
        )
    }
}

#[derive(Default, Clone)]
pub struct Poseidon2ExternalLayerBn254<const WIDTH: usize> {
    initial_external_constants: Vec<[Bn254Fr; WIDTH]>,
    final_external_constants: Vec<[Bn254Fr; WIDTH]>,
}

impl<const WIDTH: usize> ExternalLayerConstructor<Bn254Fr, WIDTH>
    for Poseidon2ExternalLayerBn254<WIDTH>
{
    fn new_from_constants(external_constants: ExternalLayerConstants<Bn254Fr, WIDTH>) -> Self {
        let initial_external_constants = external_constants.get_initial_constants().clone();
        let final_external_constants = external_constants.get_terminal_constants().clone();

        Self {
            initial_external_constants,
            final_external_constants,
        }
    }
}

impl<const WIDTH: usize> ExternalLayer<Bn254Fr, WIDTH, 5> for Poseidon2ExternalLayerBn254<WIDTH> {
    type InternalState = [Bn254Fr; WIDTH];

    fn permute_state_initial(&self, mut state: [Bn254Fr; WIDTH]) -> [Bn254Fr; WIDTH] {
        external_initial_permute_state::<_, _, WIDTH, 5>(
            &mut state,
            &self.initial_external_constants,
            &HLMDSMat4,
        );
        state
    }

    fn permute_state_final(&self, mut state: [Bn254Fr; WIDTH]) -> [Bn254Fr; WIDTH] {
        external_final_permute_state::<_, _, WIDTH, 5>(
            &mut state,
            &self.final_external_constants,
            &HLMDSMat4,
        );
        state
    }
}

#[cfg(test)]
mod tests {
    use ff::PrimeField;
    use p3_poseidon2::ExternalLayerConstants;
    use p3_symmetric::Permutation;
    use rand::Rng;
    use zkhash::ark_ff::{BigInteger, PrimeField as ark_PrimeField};
    use zkhash::fields::bn256::FpBN256 as ark_FpBN256;
    use zkhash::poseidon2::poseidon2::Poseidon2 as Poseidon2Ref;
    use zkhash::poseidon2::poseidon2_instance_bn256::{POSEIDON2_BN256_PARAMS, RC3};

    use super::*;
    use crate::FFBn254Fr;

    fn bn254_from_ark_ff(input: ark_FpBN256) -> Bn254Fr {
        let bytes = input.into_bigint().to_bytes_le();

        let mut res = <FFBn254Fr as PrimeField>::Repr::default();

        for (i, digit) in res.as_mut().iter_mut().enumerate() {
            *digit = bytes[i];
        }

        let value = FFBn254Fr::from_repr(res);

        if value.is_some().into() {
            Bn254Fr {
                value: value.unwrap(),
            }
        } else {
            panic!("Invalid field element")
        }
    }

    #[test]
    fn test_poseidon2_bn254() {
        const WIDTH: usize = 3;
        const ROUNDS_F: usize = 8;
        const ROUNDS_P: usize = 56;

        type F = Bn254Fr;

        let mut rng = rand::thread_rng();

        // Poiseidon2 reference implementation from zkhash repo.
        let poseidon2_ref = Poseidon2Ref::new(&POSEIDON2_BN256_PARAMS);

        // Copy over round constants from zkhash.
        let mut round_constants: Vec<[F; WIDTH]> = RC3
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

        let internal_start = ROUNDS_F / 2;
        let internal_end = (ROUNDS_F / 2) + ROUNDS_P;
        let internal_round_constants = round_constants
            .drain(internal_start..internal_end)
            .map(|vec| vec[0])
            .collect::<Vec<_>>();
        let external_round_constants = ExternalLayerConstants::new(
            round_constants[..(ROUNDS_F / 2)].to_vec(),
            round_constants[(ROUNDS_F / 2)..].to_vec(),
        );
        // Our Poseidon2 implementation.
        let poseidon2 = Poseidon2Bn254::new(external_round_constants, internal_round_constants);

        // Generate random input and convert to both Goldilocks field formats.
        let input_ark_ff = rng.gen::<[ark_FpBN256; WIDTH]>();
        let input: [Bn254Fr; 3] = input_ark_ff
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
