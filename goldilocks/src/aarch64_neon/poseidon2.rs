//! Optimized Poseidon2 for Goldilocks on aarch64.
//!
//! Uses ARM inline assembly with latency hiding via interleaved S-box/MDS computation.

use alloc::vec::Vec;

use p3_poseidon2::{
    ExternalLayer, ExternalLayerConstants, ExternalLayerConstructor, InternalLayer,
    InternalLayerConstructor,
};

use super::poseidon2_asm::{
    external_initial_permute_state_asm, external_terminal_permute_state_asm,
    internal_permute_state_asm,
};
use crate::{
    Goldilocks, MATRIX_DIAG_8_GOLDILOCKS, MATRIX_DIAG_12_GOLDILOCKS, MATRIX_DIAG_16_GOLDILOCKS,
    MATRIX_DIAG_20_GOLDILOCKS,
};

/// Degree of the chosen permutation polynomial for Goldilocks.
const GOLDILOCKS_S_BOX_DEGREE: u64 = 7;

/// ASM-optimized internal layer with latency-hiding S-box/MDS interleaving.
#[derive(Debug, Clone, Default)]
pub struct Poseidon2InternalLayerGoldilocksAsm {
    internal_constants: Vec<Goldilocks>,
}

impl InternalLayerConstructor<Goldilocks> for Poseidon2InternalLayerGoldilocksAsm {
    fn new_from_constants(internal_constants: Vec<Goldilocks>) -> Self {
        Self { internal_constants }
    }
}

impl InternalLayer<Goldilocks, 8, GOLDILOCKS_S_BOX_DEGREE> for Poseidon2InternalLayerGoldilocksAsm {
    fn permute_state(&self, state: &mut [Goldilocks; 8]) {
        internal_permute_state_asm(state, MATRIX_DIAG_8_GOLDILOCKS, &self.internal_constants);
    }
}

impl InternalLayer<Goldilocks, 12, GOLDILOCKS_S_BOX_DEGREE>
    for Poseidon2InternalLayerGoldilocksAsm
{
    fn permute_state(&self, state: &mut [Goldilocks; 12]) {
        internal_permute_state_asm(state, MATRIX_DIAG_12_GOLDILOCKS, &self.internal_constants);
    }
}

impl InternalLayer<Goldilocks, 16, GOLDILOCKS_S_BOX_DEGREE>
    for Poseidon2InternalLayerGoldilocksAsm
{
    fn permute_state(&self, state: &mut [Goldilocks; 16]) {
        internal_permute_state_asm(state, MATRIX_DIAG_16_GOLDILOCKS, &self.internal_constants);
    }
}

impl InternalLayer<Goldilocks, 20, GOLDILOCKS_S_BOX_DEGREE>
    for Poseidon2InternalLayerGoldilocksAsm
{
    fn permute_state(&self, state: &mut [Goldilocks; 20]) {
        internal_permute_state_asm(state, MATRIX_DIAG_20_GOLDILOCKS, &self.internal_constants);
    }
}

/// ASM-optimized external layer with pipelined S-box computation.
#[derive(Clone)]
pub struct Poseidon2ExternalLayerGoldilocksAsm<const WIDTH: usize> {
    external_constants: ExternalLayerConstants<Goldilocks, WIDTH>,
}

impl<const WIDTH: usize> ExternalLayerConstructor<Goldilocks, WIDTH>
    for Poseidon2ExternalLayerGoldilocksAsm<WIDTH>
{
    fn new_from_constants(external_constants: ExternalLayerConstants<Goldilocks, WIDTH>) -> Self {
        Self { external_constants }
    }
}

impl<const WIDTH: usize> ExternalLayer<Goldilocks, WIDTH, GOLDILOCKS_S_BOX_DEGREE>
    for Poseidon2ExternalLayerGoldilocksAsm<WIDTH>
{
    fn permute_state_initial(&self, state: &mut [Goldilocks; WIDTH]) {
        external_initial_permute_state_asm(state, self.external_constants.get_initial_constants());
    }

    fn permute_state_terminal(&self, state: &mut [Goldilocks; WIDTH]) {
        external_terminal_permute_state_asm(
            state,
            self.external_constants.get_terminal_constants(),
        );
    }
}

/// Type alias for scalar ASM-optimized Poseidon2.
pub type Poseidon2GoldilocksAsm<const WIDTH: usize> = p3_poseidon2::Poseidon2<
    Goldilocks,
    Poseidon2ExternalLayerGoldilocksAsm<WIDTH>,
    Poseidon2InternalLayerGoldilocksAsm,
    WIDTH,
    GOLDILOCKS_S_BOX_DEGREE,
>;
