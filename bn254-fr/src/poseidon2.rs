//! Diffusion matrix for Bn254
//!
//! Reference: https://github.com/HorizenLabs/poseidon2/blob/main/plain_implementations/src/poseidon2/poseidon2_instance_bn256.rs

use std::sync::OnceLock;

use p3_field::PrimeCharacteristicRing;
use p3_poseidon2::{
    add_rc_and_sbox_generic, external_initial_permute_state, external_terminal_permute_state,
    internal_permute_state, matmul_internal, ExternalLayer, ExternalLayerConstants,
    ExternalLayerConstructor, HLMDSMat4, InternalLayer, InternalLayerConstructor, Poseidon2,
};

use crate::Bn254Fr;

/// Degree of the chosen permutation polynomial for BN254, used as the Poseidon2 S-Box.
///
/// As p - 1 is divisible by 2 and 3 the smallest choice for a degree D satisfying gcd(p - 1, D) = 1 is 5.
const BN254_S_BOX_DEGREE: u64 = 5;

/// An implementation of the Poseidon2 hash function for the Bn254Fr field.
///
/// It acts on arrays of the form `[Bn254Fr; WIDTH]`.
pub type Poseidon2Bn254<const WIDTH: usize> = Poseidon2<
    Bn254Fr,
    Poseidon2ExternalLayerBn254<WIDTH>,
    Poseidon2InternalLayerBn254,
    WIDTH,
    BN254_S_BOX_DEGREE,
>;

/// Currently we only support a single width for Poseidon2 BN254.
const BN254_WIDTH: usize = 3;

#[inline]
fn get_diffusion_matrix_3() -> &'static [Bn254Fr; 3] {
    static MAT_DIAG3_M_1: OnceLock<[Bn254Fr; 3]> = OnceLock::new();
    MAT_DIAG3_M_1.get_or_init(|| [Bn254Fr::ONE, Bn254Fr::ONE, Bn254Fr::TWO])
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

impl InternalLayer<Bn254Fr, BN254_WIDTH, BN254_S_BOX_DEGREE> for Poseidon2InternalLayerBn254 {
    /// Perform the internal layers of the Poseidon2 permutation on the given state.
    fn permute_state(&self, state: &mut [Bn254Fr; BN254_WIDTH]) {
        internal_permute_state(
            state,
            |x| matmul_internal(x, *get_diffusion_matrix_3()),
            &self.internal_constants,
        )
    }
}

pub type Poseidon2ExternalLayerBn254<const WIDTH: usize> = ExternalLayerConstants<Bn254Fr, WIDTH>;

impl<const WIDTH: usize> ExternalLayerConstructor<Bn254Fr, WIDTH>
    for Poseidon2ExternalLayerBn254<WIDTH>
{
    fn new_from_constants(external_constants: ExternalLayerConstants<Bn254Fr, WIDTH>) -> Self {
        external_constants
    }
}

impl<const WIDTH: usize> ExternalLayer<Bn254Fr, WIDTH, BN254_S_BOX_DEGREE>
    for Poseidon2ExternalLayerBn254<WIDTH>
{
    /// Perform the initial external layers of the Poseidon2 permutation on the given state.
    fn permute_state_initial(&self, state: &mut [Bn254Fr; WIDTH]) {
        external_initial_permute_state(
            state,
            self.get_initial_constants(),
            add_rc_and_sbox_generic,
            &HLMDSMat4,
        );
    }

    /// Perform the terminal external layers of the Poseidon2 permutation on the given state.
    fn permute_state_terminal(&self, state: &mut [Bn254Fr; WIDTH]) {
        external_terminal_permute_state(
            state,
            self.get_terminal_constants(),
            add_rc_and_sbox_generic,
            &HLMDSMat4,
        );
    }
}

#[cfg(test)]
mod tests {
    use ff::PrimeField;
    use num_bigint::BigUint;
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

    fn ark_ff_from_bn254(input: Bn254Fr) -> ark_FpBN256 {
        let bigint = BigUint::from_bytes_le(&input.value.to_bytes());

        ark_FpBN256::from(bigint)
    }

    #[test]
    fn test_poseidon2_bn254() {
        const WIDTH: usize = 3;
        const ROUNDS_F: usize = 8;
        const ROUNDS_P: usize = 56;

        type F = Bn254Fr;

        let mut rng = rand::rng();

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
        let input = rng.random::<[Bn254Fr; WIDTH]>();
        let input_ark_ff = input.map(ark_ff_from_bn254);

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
