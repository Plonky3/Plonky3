use alloc::vec::Vec;

use p3_poseidon2::{ExternalLayerConstants, ExternalLayerConstructor, InternalLayerConstructor};
//! This file contains simple wrapper structs on top of which we can implement Poseidon2 Internal/ExternalLayer.

use crate::Mersenne31;

#[derive(Debug, Clone, Default)]
pub struct Poseidon2InternalLayerMersenne31 {
    pub(crate) internal_constants: Vec<Mersenne31>,
}

#[derive(Default, Clone)]
pub struct Poseidon2ExternalLayerMersenne31<const WIDTH: usize> {
    pub(crate) initial_external_constants: Vec<[Mersenne31; WIDTH]>,
    pub(crate) final_external_constants: Vec<[Mersenne31; WIDTH]>,
}

impl InternalLayerConstructor<Mersenne31> for Poseidon2InternalLayerMersenne31 {
    fn new_from_constants(internal_constants: Vec<Mersenne31>) -> Self {
        Self { internal_constants }
    }
}

impl<const WIDTH: usize> ExternalLayerConstructor<Mersenne31, WIDTH>
    for Poseidon2ExternalLayerMersenne31<WIDTH>
{
    fn new_from_constants(external_constants: ExternalLayerConstants<Mersenne31, WIDTH>) -> Self {
        let initial_external_constants = external_constants.get_initial_constants().clone();
        let final_external_constants = external_constants.get_terminal_constants().clone();

        Self {
            initial_external_constants,
            final_external_constants,
        }
    }
}
