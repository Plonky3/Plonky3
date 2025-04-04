//! Diffusion matrix for Bls12-377
//!
//! Even tho the reference is for the other field, we used it for BLS12-377Fr considering the common
//! field nature.
//! Reference: https://github.com/HorizenLabs/poseidon2/blob/main/plain_implementations/src/poseidon2/poseidon2_instance_bls12.rs
use std::sync::OnceLock;

use p3_field::FieldAlgebra;
use p3_poseidon2::{
    add_rc_and_sbox_generic, external_initial_permute_state, external_terminal_permute_state,
    internal_permute_state, matmul_internal, ExternalLayer, ExternalLayerConstants,
    ExternalLayerConstructor, HLMDSMat4, InternalLayer, InternalLayerConstructor, Poseidon2,
};

use crate::Bls12_377Fr;

/// Degree of the chosen permutation polynomial for BLS12-377, used as the Poseidon2 S-Box.
///
/// As p - 1 is divisible by 2 and 3 the smallest choice for a degree D satisfying gcd(p - 1, D) = 1 is 5.
const BLS12_337_S_BOX_DEGREE: u64 = 17;

/// An implementation of the Poseidon2 hash function for the [Bls12_377Fr] field.
///
/// It acts on arrays of the form `[Bls12_377Fr; WIDTH]`.
pub type Poseidon2Bls12337<const WIDTH: usize> = Poseidon2<
    Bls12_377Fr,
    Poseidon2ExternalLayerBls12337<WIDTH>,
    Poseidon2InternalLayerBls12337,
    WIDTH,
    BLS12_337_S_BOX_DEGREE,
>;

/// Currently we only support a single width for Poseidon2 Bls12_377Fr.
const BLS12_377_WIDTH: usize = 3;

#[inline]
fn get_diffusion_matrix_3() -> &'static [Bls12_377Fr; 3] {
    static MAT_DIAG3_M_1: OnceLock<[Bls12_377Fr; 3]> = OnceLock::new();
    MAT_DIAG3_M_1.get_or_init(|| [Bls12_377Fr::ONE, Bls12_377Fr::ONE, Bls12_377Fr::TWO])
}

#[derive(Debug, Clone, Default)]
pub struct Poseidon2InternalLayerBls12337 {
    internal_constants: Vec<Bls12_377Fr>,
}

impl InternalLayerConstructor<Bls12_377Fr> for Poseidon2InternalLayerBls12337 {
    fn new_from_constants(internal_constants: Vec<Bls12_377Fr>) -> Self {
        Self { internal_constants }
    }
}

impl InternalLayer<Bls12_377Fr, BLS12_377_WIDTH, BLS12_337_S_BOX_DEGREE>
    for Poseidon2InternalLayerBls12337
{
    /// Perform the internal layers of the Poseidon2 permutation on the given state.
    fn permute_state(&self, state: &mut [Bls12_377Fr; BLS12_377_WIDTH]) {
        internal_permute_state::<Bls12_377Fr, BLS12_377_WIDTH, BLS12_337_S_BOX_DEGREE>(
            state,
            |x| matmul_internal(x, *get_diffusion_matrix_3()),
            &self.internal_constants,
        )
    }
}

pub type Poseidon2ExternalLayerBls12337<const WIDTH: usize> =
    ExternalLayerConstants<Bls12_377Fr, WIDTH>;

impl<const WIDTH: usize> ExternalLayerConstructor<Bls12_377Fr, WIDTH>
    for Poseidon2ExternalLayerBls12337<WIDTH>
{
    fn new_from_constants(external_constants: ExternalLayerConstants<Bls12_377Fr, WIDTH>) -> Self {
        external_constants
    }
}

impl<const WIDTH: usize> ExternalLayer<Bls12_377Fr, WIDTH, BLS12_337_S_BOX_DEGREE>
    for Poseidon2ExternalLayerBls12337<WIDTH>
{
    /// Perform the initial external layers of the Poseidon2 permutation on the given state.
    fn permute_state_initial(&self, state: &mut [Bls12_377Fr; WIDTH]) {
        external_initial_permute_state(
            state,
            self.get_initial_constants(),
            add_rc_and_sbox_generic::<_, BLS12_337_S_BOX_DEGREE>,
            &HLMDSMat4,
        );
    }

    /// Perform the terminal external layers of the Poseidon2 permutation on the given state.
    fn permute_state_terminal(&self, state: &mut [Bls12_377Fr; WIDTH]) {
        external_terminal_permute_state(
            state,
            self.get_terminal_constants(),
            add_rc_and_sbox_generic::<_, BLS12_337_S_BOX_DEGREE>,
            &HLMDSMat4,
        );
    }
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use lazy_static::lazy_static;
    use p3_poseidon2::ExternalLayerConstants;
    use p3_symmetric::Permutation;

    use super::*;
    use crate::rc::RC1;
    use crate::FF_Bls12_377Fr;

    fn bls12337_from_str(input: String) -> Bls12_377Fr {
        Bls12_377Fr {
            value: FF_Bls12_377Fr::from_str(&input).unwrap(),
        }
    }

    struct TestCase {
        input: [Bls12_377Fr; 3],
        output: [Bls12_377Fr; 3],
    }

    lazy_static! {
        static ref test_cases: Vec<TestCase> = vec![
            TestCase{
                input: [
                    Bls12_377Fr::from_canonical_u64(1),
                    Bls12_377Fr::from_canonical_u64(1),
                    Bls12_377Fr::from_canonical_u64(1)
                ],
                output: [
                    Bls12_377Fr{
                        value: FF_Bls12_377Fr::from_str("860866526687489428989707845699174300428968972136191802628058085350960837665").unwrap()
                    },
                    Bls12_377Fr{
                        value: FF_Bls12_377Fr::from_str("5110597475731104621758769815158117224808582320756097893691034814235891573903").unwrap()
                    },
                    Bls12_377Fr{
                        value: FF_Bls12_377Fr::from_str("7729317803422720288984283127941214463170714974376170076017011151108586598169").unwrap()
                    }
                ],
            },
            TestCase{
                input: [
                    Bls12_377Fr::from_canonical_u64(3457435785349743598),
                    Bls12_377Fr::from_canonical_u64(6786347127498807634),
                    Bls12_377Fr::from_canonical_u64(0)
                ],
                output: [
                    Bls12_377Fr{
                        value: FF_Bls12_377Fr::from_str("675583857530871455730788450062580113099802122446214910490159210203931204829").unwrap()
                    },
                    Bls12_377Fr{
                        value: FF_Bls12_377Fr::from_str("5907675690176137521797568886899101531719052562432127804242792579253736799082").unwrap()
                    },
                    Bls12_377Fr{
                        value: FF_Bls12_377Fr::from_str("3580163914861780974242579068588450174896166405220547377222985096420674645846").unwrap()
                    }
                ],
            },
            TestCase{
                input: [
                    Bls12_377Fr::from_canonical_u64(9379455689548),
                    Bls12_377Fr::from_canonical_u64(7608439748),
                    Bls12_377Fr::from_canonical_u64(954338476438754632)
                ],
                output: [
                    Bls12_377Fr{
                        value: FF_Bls12_377Fr::from_str("3055206294377317359307613780575828874003458762314725501486826284300578105158").unwrap()
                    },
                    Bls12_377Fr{
                        value: FF_Bls12_377Fr::from_str("5797369338536356401300628927961554112193154358645694251738402018195827090483").unwrap()
                    },
                    Bls12_377Fr{
                        value: FF_Bls12_377Fr::from_str("6809256165826541738832209298614382604751407820145487553453583098940097598887").unwrap()
                    }
                ],
            },
        ];
    }

    #[test]
    fn test_poseidon2_bls12337() {
        const WIDTH: usize = 3;
        const ROUNDS_F: usize = 8;
        const ROUNDS_P: usize = 56;

        type F = Bls12_377Fr;

        // Copy over round constants from pre generated values.
        let round_constants: Vec<[F; WIDTH]> = RC1
            .iter()
            .map(|vec| {
                vec.iter()
                    .cloned()
                    .map(bls12337_from_str)
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap()
            })
            .collect();

        let internal_start = ROUNDS_F / 2;
        let internal_end = (ROUNDS_F / 2) + ROUNDS_P;

        let internal_round_constants = round_constants
            .clone()
            .drain(internal_start..internal_end)
            .map(|vec| vec[0])
            .collect::<Vec<_>>();

        let external_round_constants = ExternalLayerConstants::new(
            round_constants[..(ROUNDS_F / 2)].to_vec(),
            round_constants[internal_end..].to_vec(),
        );

        // Our Poseidon2 implementation.
        let poseidon2 = Poseidon2Bls12337::new(external_round_constants, internal_round_constants);

        test_cases.iter().for_each(|case| {
            let mut output = case.input;
            poseidon2.permute_mut(&mut output);

            assert_eq!(output, case.output);
        });
    }
}
