use crate::Mersenne31;
use alloc::vec::Vec;
use p3_poseidon2::{Poseidon2ExternalPackedConstants, Poseidon2InternalPackedConstants};

#[derive(Debug, Clone, Default)]
pub struct Poseidon2InternalLayerMersenne31 {
    pub(crate) internal_constants: Vec<Mersenne31>,
}

#[derive(Default, Clone)]
pub struct Poseidon2ExternalLayerMersenne31<const WIDTH: usize> {
    pub(crate) initial_external_constants: Vec<[Mersenne31; WIDTH]>,
    pub(crate) final_external_constants: Vec<[Mersenne31; WIDTH]>,
}

/// We save the round constants in the {-P, ..., 0} representation instead of the standard
/// {0, ..., P} one. This saves several instructions later.
impl Poseidon2InternalPackedConstants<Mersenne31> for Poseidon2InternalLayerMersenne31 {
    fn convert_from_field(internal_constants: Vec<Mersenne31>) -> Self {
        Self { internal_constants }
    }
}

/// We save the round constants in the {-P, ..., 0} representation instead of the standard
/// {0, ..., P} one. This saves several instructions later.
impl<const WIDTH: usize> Poseidon2ExternalPackedConstants<Mersenne31, WIDTH>
    for Poseidon2ExternalLayerMersenne31<WIDTH>
{
    /// Convert elements from the standard form {0, ..., P} to {-P, ..., 0}.
    fn convert_from_field_array(external_constants: [Vec<[Mersenne31; WIDTH]>; 2]) -> Self {
        let [initial_external_constants, final_external_constants] = external_constants;
        Self {
            initial_external_constants,
            final_external_constants,
        }
    }
}
