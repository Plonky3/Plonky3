//! Optimized Poseidon2 for Goldilocks on aarch64.
//!
//! Uses ARM inline assembly with latency hiding via interleaved S-box/MDS computation.
//! Fully unrolled internal rounds for W8, W12, W16.
//!
//! For packed operations, lanes are extracted to scalar, processed with interleaved
//! dual-lane ASM, then repacked. This is faster than using PackedGoldilocksNeon
//! arithmetic directly because the scalar `add_asm` avoids the modular reduction
//! overhead (overflow detection, canonicalization) present in NEON addition.

use alloc::vec::Vec;

use p3_poseidon2::{
    ExternalLayer, ExternalLayerConstants, ExternalLayerConstructor, InternalLayer,
    InternalLayerConstructor,
};

use super::packing::PackedGoldilocksNeon;
use super::poseidon2_asm::{
    external_initial_permute_state_asm, external_terminal_permute_state_asm,
    internal_permute_state_asm, internal_permute_state_asm_w8, internal_permute_state_asm_w12,
    internal_permute_state_asm_w16, internal_round_dual_asm_w8, internal_round_dual_asm_w16,
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
        // Use fully unrolled W8 version for better performance
        internal_permute_state_asm_w8(state, MATRIX_DIAG_8_GOLDILOCKS, &self.internal_constants);
    }
}

impl InternalLayer<Goldilocks, 12, GOLDILOCKS_S_BOX_DEGREE>
    for Poseidon2InternalLayerGoldilocksAsm
{
    fn permute_state(&self, state: &mut [Goldilocks; 12]) {
        // Use fully unrolled W12 version for better performance
        internal_permute_state_asm_w12(state, MATRIX_DIAG_12_GOLDILOCKS, &self.internal_constants);
    }
}

impl InternalLayer<Goldilocks, 16, GOLDILOCKS_S_BOX_DEGREE>
    for Poseidon2InternalLayerGoldilocksAsm
{
    fn permute_state(&self, state: &mut [Goldilocks; 16]) {
        // Use fully unrolled W16 version for better performance
        internal_permute_state_asm_w16(state, MATRIX_DIAG_16_GOLDILOCKS, &self.internal_constants);
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

// PackedGoldilocksNeon support: extract lanes, run interleaved dual-lane ASM, repack.
// Scalar add_asm is faster than PackedGoldilocksNeon::add due to modular reduction overhead.

use super::poseidon2_asm::{
    external_round_dual_asm, internal_round_dual_asm, internal_round_dual_asm_w12,
    mds_light_permutation_asm,
};

fn internal_permute_packed_asm<const WIDTH: usize>(
    state: &mut [PackedGoldilocksNeon; WIDTH],
    diag: [Goldilocks; WIDTH],
    internal_constants: &[Goldilocks],
) {
    // Extract lanes - keep as raw u64 arrays for ASM
    let mut lane0: [u64; WIDTH] = core::array::from_fn(|i| state[i].0[0].value);
    let mut lane1: [u64; WIDTH] = core::array::from_fn(|i| state[i].0[1].value);
    let diag_raw: [u64; WIDTH] = core::array::from_fn(|i| diag[i].value);

    // Process all internal rounds with true interleaved dual-lane execution
    for &rc in internal_constants {
        unsafe {
            internal_round_dual_asm(&mut lane0, &mut lane1, &diag_raw, rc.value);
        }
    }

    // Pack results back
    for i in 0..WIDTH {
        state[i] = PackedGoldilocksNeon([Goldilocks::new(lane0[i]), Goldilocks::new(lane1[i])]);
    }
}

/// Specialized W8 version using fully unrolled internal round.
fn internal_permute_packed_asm_w8(
    state: &mut [PackedGoldilocksNeon; 8],
    diag: [Goldilocks; 8],
    internal_constants: &[Goldilocks],
) {
    let mut lane0: [u64; 8] = core::array::from_fn(|i| state[i].0[0].value);
    let mut lane1: [u64; 8] = core::array::from_fn(|i| state[i].0[1].value);
    let diag_raw: [u64; 8] = core::array::from_fn(|i| diag[i].value);

    // Use the fully unrolled W8 internal round
    for &rc in internal_constants {
        unsafe {
            internal_round_dual_asm_w8(&mut lane0, &mut lane1, &diag_raw, rc.value);
        }
    }

    for i in 0..8 {
        state[i] = PackedGoldilocksNeon([Goldilocks::new(lane0[i]), Goldilocks::new(lane1[i])]);
    }
}

/// Specialized W12 version using fully unrolled internal round.
fn internal_permute_packed_asm_w12(
    state: &mut [PackedGoldilocksNeon; 12],
    diag: [Goldilocks; 12],
    internal_constants: &[Goldilocks],
) {
    let mut lane0: [u64; 12] = core::array::from_fn(|i| state[i].0[0].value);
    let mut lane1: [u64; 12] = core::array::from_fn(|i| state[i].0[1].value);
    let diag_raw: [u64; 12] = core::array::from_fn(|i| diag[i].value);

    // Use the fully unrolled W12 internal round
    for &rc in internal_constants {
        unsafe {
            internal_round_dual_asm_w12(&mut lane0, &mut lane1, &diag_raw, rc.value);
        }
    }

    for i in 0..12 {
        state[i] = PackedGoldilocksNeon([Goldilocks::new(lane0[i]), Goldilocks::new(lane1[i])]);
    }
}

/// Specialized W16 version using fully unrolled internal round.
fn internal_permute_packed_asm_w16(
    state: &mut [PackedGoldilocksNeon; 16],
    diag: [Goldilocks; 16],
    internal_constants: &[Goldilocks],
) {
    let mut lane0: [u64; 16] = core::array::from_fn(|i| state[i].0[0].value);
    let mut lane1: [u64; 16] = core::array::from_fn(|i| state[i].0[1].value);
    let diag_raw: [u64; 16] = core::array::from_fn(|i| diag[i].value);

    // Use the fully unrolled W16 internal round
    for &rc in internal_constants {
        unsafe {
            internal_round_dual_asm_w16(&mut lane0, &mut lane1, &diag_raw, rc.value);
        }
    }

    for i in 0..16 {
        state[i] = PackedGoldilocksNeon([Goldilocks::new(lane0[i]), Goldilocks::new(lane1[i])]);
    }
}

/// Run external initial permutation on packed state with true interleaved execution.
fn external_initial_permute_packed_asm<const WIDTH: usize>(
    state: &mut [PackedGoldilocksNeon; WIDTH],
    initial_constants: &[[Goldilocks; WIDTH]],
) {
    let mut lane0: [u64; WIDTH] = core::array::from_fn(|i| state[i].0[0].value);
    let mut lane1: [u64; WIDTH] = core::array::from_fn(|i| state[i].0[1].value);

    // Initial MDS (sequential - MDS is mostly additions)
    unsafe {
        mds_light_permutation_asm(&mut lane0);
        mds_light_permutation_asm(&mut lane1);
    }

    // External rounds with true interleaved dual-lane execution
    for rc in initial_constants {
        let rc_raw: [u64; WIDTH] = core::array::from_fn(|i| rc[i].value);
        unsafe {
            external_round_dual_asm(&mut lane0, &mut lane1, &rc_raw);
        }
    }

    for i in 0..WIDTH {
        state[i] = PackedGoldilocksNeon([Goldilocks::new(lane0[i]), Goldilocks::new(lane1[i])]);
    }
}

/// Run external terminal permutation on packed state with true interleaved execution.
fn external_terminal_permute_packed_asm<const WIDTH: usize>(
    state: &mut [PackedGoldilocksNeon; WIDTH],
    terminal_constants: &[[Goldilocks; WIDTH]],
) {
    let mut lane0: [u64; WIDTH] = core::array::from_fn(|i| state[i].0[0].value);
    let mut lane1: [u64; WIDTH] = core::array::from_fn(|i| state[i].0[1].value);

    // External rounds with true interleaved dual-lane execution
    for rc in terminal_constants {
        let rc_raw: [u64; WIDTH] = core::array::from_fn(|i| rc[i].value);
        unsafe {
            external_round_dual_asm(&mut lane0, &mut lane1, &rc_raw);
        }
    }

    for i in 0..WIDTH {
        state[i] = PackedGoldilocksNeon([Goldilocks::new(lane0[i]), Goldilocks::new(lane1[i])]);
    }
}

// Internal layer implementations for PackedGoldilocksNeon

impl InternalLayer<PackedGoldilocksNeon, 8, GOLDILOCKS_S_BOX_DEGREE>
    for Poseidon2InternalLayerGoldilocksAsm
{
    fn permute_state(&self, state: &mut [PackedGoldilocksNeon; 8]) {
        // Use fully unrolled W8 version for better performance
        internal_permute_packed_asm_w8(state, MATRIX_DIAG_8_GOLDILOCKS, &self.internal_constants);
    }
}

impl InternalLayer<PackedGoldilocksNeon, 12, GOLDILOCKS_S_BOX_DEGREE>
    for Poseidon2InternalLayerGoldilocksAsm
{
    fn permute_state(&self, state: &mut [PackedGoldilocksNeon; 12]) {
        // Use fully unrolled W12 version for better performance
        internal_permute_packed_asm_w12(state, MATRIX_DIAG_12_GOLDILOCKS, &self.internal_constants);
    }
}

impl InternalLayer<PackedGoldilocksNeon, 16, GOLDILOCKS_S_BOX_DEGREE>
    for Poseidon2InternalLayerGoldilocksAsm
{
    fn permute_state(&self, state: &mut [PackedGoldilocksNeon; 16]) {
        // Use fully unrolled W16 version for better performance
        internal_permute_packed_asm_w16(state, MATRIX_DIAG_16_GOLDILOCKS, &self.internal_constants);
    }
}

impl InternalLayer<PackedGoldilocksNeon, 20, GOLDILOCKS_S_BOX_DEGREE>
    for Poseidon2InternalLayerGoldilocksAsm
{
    fn permute_state(&self, state: &mut [PackedGoldilocksNeon; 20]) {
        internal_permute_packed_asm(state, MATRIX_DIAG_20_GOLDILOCKS, &self.internal_constants);
    }
}

// External layer implementations for PackedGoldilocksNeon

impl<const WIDTH: usize> ExternalLayer<PackedGoldilocksNeon, WIDTH, GOLDILOCKS_S_BOX_DEGREE>
    for Poseidon2ExternalLayerGoldilocksAsm<WIDTH>
{
    fn permute_state_initial(&self, state: &mut [PackedGoldilocksNeon; WIDTH]) {
        external_initial_permute_packed_asm(state, self.external_constants.get_initial_constants());
    }

    fn permute_state_terminal(&self, state: &mut [PackedGoldilocksNeon; WIDTH]) {
        external_terminal_permute_packed_asm(
            state,
            self.external_constants.get_terminal_constants(),
        );
    }
}

