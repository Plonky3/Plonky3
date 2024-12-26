//! Diffusion matrix for Bn254
//!
//! Reference: https://github.com/HorizenLabs/poseidon2/blob/main/plain_implementations/src/poseidon2/poseidon2_instance_bn256.rs

use std::sync::OnceLock;

use p3_field::FieldAlgebra;
use p3_poseidon2::{
    add_rc_and_sbox_generic, external_initial_permute_state, external_terminal_permute_state,
    internal_permute_state, matmul_internal, ExternalLayer, ExternalLayerConstants,
    ExternalLayerConstructor, HLMDSMat4, InternalLayer, InternalLayerConstructor, Poseidon2,
};

use crate::Bls12377Fr;

/// Degree of the chosen permutation polynomial for BN254, used as the Poseidon2 S-Box.
///
/// As p - 1 is divisible by 2 and 3 the smallest choice for a degree D satisfying gcd(p - 1, D) = 1 is 5.
/// TODO (nazarevsky): figure this out
const BLS12337_S_BOX_DEGREE: u64 = 5;

/// An implementation of the Poseidon2 hash function for the Bn254Fr field.
///
/// It acts on arrays of the form `[Bn254Fr; WIDTH]`.
pub type Poseidon2Bls12337<const WIDTH: usize> = Poseidon2<
    Bls12377Fr,
    Poseidon2ExternalLayerBls12337<WIDTH>,
    Poseidon2InternalLayerBls12337,
    WIDTH,
    BLS12337_S_BOX_DEGREE,
>;

/// Currently we only support a single width for Poseidon2 BN254.
const BN254_WIDTH: usize = 3;

#[inline]
fn get_diffusion_matrix_3() -> &'static [Bls12377Fr; 3] {
    static MAT_DIAG3_M_1: OnceLock<[Bls12377Fr; 3]> = OnceLock::new();
    MAT_DIAG3_M_1.get_or_init(|| [Bls12377Fr::ONE, Bls12377Fr::ONE, Bls12377Fr::TWO])
}

#[derive(Debug, Clone, Default)]
pub struct Poseidon2InternalLayerBls12337 {
    internal_constants: Vec<Bls12377Fr>,
}

impl InternalLayerConstructor<Bls12377Fr> for Poseidon2InternalLayerBls12337 {
    fn new_from_constants(internal_constants: Vec<Bls12377Fr>) -> Self {
        Self { internal_constants }
    }
}

impl InternalLayer<Bls12377Fr, BN254_WIDTH, BLS12337_S_BOX_DEGREE> for Poseidon2InternalLayerBls12337 {
    /// Perform the internal layers of the Poseidon2 permutation on the given state.
    fn permute_state(&self, state: &mut [Bls12377Fr; BN254_WIDTH]) {
        internal_permute_state::<Bls12377Fr, BN254_WIDTH, BLS12337_S_BOX_DEGREE>(
            state,
            |x| matmul_internal(x, *get_diffusion_matrix_3()),
            &self.internal_constants,
        )
    }
}

pub type Poseidon2ExternalLayerBls12337<const WIDTH: usize> = ExternalLayerConstants<Bls12377Fr, WIDTH>;

impl<const WIDTH: usize> ExternalLayerConstructor<Bls12377Fr, WIDTH>
    for Poseidon2ExternalLayerBls12337<WIDTH>
{
    fn new_from_constants(external_constants: ExternalLayerConstants<Bls12377Fr, WIDTH>) -> Self {
        external_constants
    }
}

impl<const WIDTH: usize> ExternalLayer<Bls12377Fr, WIDTH, BLS12337_S_BOX_DEGREE>
    for Poseidon2ExternalLayerBls12337<WIDTH>
{
    /// Perform the initial external layers of the Poseidon2 permutation on the given state.
    fn permute_state_initial(&self, state: &mut [Bls12377Fr; WIDTH]) {
        external_initial_permute_state(
            state,
            self.get_initial_constants(),
            add_rc_and_sbox_generic::<_, BLS12337_S_BOX_DEGREE>,
            &HLMDSMat4,
        );
    }

    /// Perform the terminal external layers of the Poseidon2 permutation on the given state.
    fn permute_state_terminal(&self, state: &mut [Bls12377Fr; WIDTH]) {
        external_terminal_permute_state(
            state,
            self.get_terminal_constants(),
            add_rc_and_sbox_generic::<_, BLS12337_S_BOX_DEGREE>,
            &HLMDSMat4,
        );
    }
}

// TODO (nazarevsky): the test is not working
#[cfg(test)]
mod tests {
    use ff::PrimeField;
    use p3_poseidon2::ExternalLayerConstants;
    use p3_symmetric::Permutation;
    use rand::Rng;
    use zkhash::ark_ff::{BigInteger, PrimeField as ark_PrimeField};
    use zkhash::fields::bls12::FpBLS12 as ark_Bls12;
    // use zkhash::fields::bn256::FpBN256 as ark_FpBN256;
    use zkhash::poseidon2::poseidon2::Poseidon2 as Poseidon2Ref;
    use zkhash::poseidon2::poseidon2_instance_bls12::{POSEIDON2_BLS_2_PARAMS, POSEIDON2_BLS_3_PARAMS, POSEIDON2_BLS_4_PARAMS, POSEIDON2_BLS_8_PARAMS, RC3};

    use super::*;
    use crate::FFBls12377Fr;

    fn bls12337_from_ark_ff(input: ark_Bls12) -> Bls12377Fr {
        let bytes = input.into_bigint().to_bytes_le();

        let value = FFBls12377Fr::from_le_bytes_mod_order(input.0.to_bytes_le().as_slice());
        // if value.is_some().into() {
        //     Bn254Fr {
        //         value: value.unwrap(),
        //     }
        // } else {
        //     panic!("Invalid field element")
        // }
        let a = Bls12377Fr {
            value: value,
        };

        println!("{}", a.to_string());

        a
    }

    #[test]
    fn test_poseidon2_bn254() {
        const WIDTH: usize = 3;
        const ROUNDS_F: usize = 8;
        const ROUNDS_P: usize = 56;

        type F = Bls12377Fr;

        let mut rng = rand::thread_rng();

        // Poiseidon2 reference implementation from zkhash repo.
        let poseidon2_ref = Poseidon2Ref::new(&POSEIDON2_BLS_3_PARAMS);

        // Copy over round constants from zkhash.
        let mut round_constants: Vec<[F; WIDTH]> = RC3
            .iter()
            .map(|vec| {
                vec.iter()
                    .cloned()
                    .map(bls12337_from_ark_ff)
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
        let poseidon2 = Poseidon2Bls12337::new(external_round_constants, internal_round_constants);

        // Generate random input and convert to both Goldilocks field formats.
        let input_ark_ff = rng.gen::<[ark_Bls12; WIDTH]>();
        let input: [Bls12377Fr; 3] = input_ark_ff
            .iter()
            .cloned()
            .map(bls12337_from_ark_ff)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        // Run reference implementation.
        let output_ref = poseidon2_ref.permutation(&input_ark_ff);

        let expected: [F; WIDTH] = output_ref
            .iter()
            .cloned()
            .map(bls12337_from_ark_ff)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        // Run our implementation.
        let mut output = input;
        poseidon2.permute_mut(&mut output);

        assert_eq!(output, expected);
    }
}
