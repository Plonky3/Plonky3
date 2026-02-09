//! Optimized Poseidon2 for Goldilocks on aarch64.
//!
//! Uses ARM inline assembly with latency hiding via interleaved S-box/MDS computation.
//! Fully unrolled internal rounds for W8, W12, W16.
//!
//! For packed operations, lanes are extracted to scalar, processed with interleaved
//! dual-lane ASM, then repacked. This is faster than using PackedGoldilocksNeon
//! arithmetic directly because the scalar `add_asm` avoids the modular reduction
//! overhead present in NEON addition.

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
        internal_permute_state_asm_w8(state, MATRIX_DIAG_8_GOLDILOCKS, &self.internal_constants);
    }
}

impl InternalLayer<Goldilocks, 12, GOLDILOCKS_S_BOX_DEGREE>
    for Poseidon2InternalLayerGoldilocksAsm
{
    fn permute_state(&self, state: &mut [Goldilocks; 12]) {
        internal_permute_state_asm_w12(state, MATRIX_DIAG_12_GOLDILOCKS, &self.internal_constants);
    }
}

impl InternalLayer<Goldilocks, 16, GOLDILOCKS_S_BOX_DEGREE>
    for Poseidon2InternalLayerGoldilocksAsm
{
    fn permute_state(&self, state: &mut [Goldilocks; 16]) {
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
        internal_permute_packed_asm_w8(state, MATRIX_DIAG_8_GOLDILOCKS, &self.internal_constants);
    }
}

impl InternalLayer<PackedGoldilocksNeon, 12, GOLDILOCKS_S_BOX_DEGREE>
    for Poseidon2InternalLayerGoldilocksAsm
{
    fn permute_state(&self, state: &mut [PackedGoldilocksNeon; 12]) {
        internal_permute_packed_asm_w12(state, MATRIX_DIAG_12_GOLDILOCKS, &self.internal_constants);
    }
}

impl InternalLayer<PackedGoldilocksNeon, 16, GOLDILOCKS_S_BOX_DEGREE>
    for Poseidon2InternalLayerGoldilocksAsm
{
    fn permute_state(&self, state: &mut [PackedGoldilocksNeon; 16]) {
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

#[cfg(test)]
mod tests {
    use p3_field::{PrimeCharacteristicRing, PrimeField64};
    use p3_poseidon2::{ExternalLayerConstants, InternalLayer, Poseidon2};
    use p3_symmetric::Permutation;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;
    use crate::{Poseidon2ExternalLayerGoldilocks, Poseidon2InternalLayerGoldilocks};

    type F = Goldilocks;

    // Test that fully ASM-optimized implementation matches generic scalar
    fn test_asm_matches_generic<const WIDTH: usize>()
    where
        Poseidon2InternalLayerGoldilocks: InternalLayer<F, WIDTH, GOLDILOCKS_S_BOX_DEGREE>,
        Poseidon2InternalLayerGoldilocksAsm: InternalLayer<F, WIDTH, GOLDILOCKS_S_BOX_DEGREE>,
        Poseidon2ExternalLayerGoldilocksAsm<WIDTH>:
            ExternalLayer<Goldilocks, WIDTH, GOLDILOCKS_S_BOX_DEGREE>,
    {
        let mut rng = SmallRng::seed_from_u64(42);

        let external_constants =
            ExternalLayerConstants::<Goldilocks, WIDTH>::new_from_rng(4, &mut rng);
        let internal_constants: Vec<Goldilocks> =
            (0..22).map(|_| F::from_u64(rng.random())).collect();

        // Generic scalar implementation
        let generic_poseidon2: Poseidon2<
            Goldilocks,
            Poseidon2ExternalLayerGoldilocks<WIDTH>,
            Poseidon2InternalLayerGoldilocks,
            WIDTH,
            GOLDILOCKS_S_BOX_DEGREE,
        > = Poseidon2::new(external_constants.clone(), internal_constants.clone());

        // Fully ASM-optimized implementation
        let asm_poseidon2: Poseidon2GoldilocksAsm<WIDTH> =
            Poseidon2::new(external_constants, internal_constants);

        // Test with zeros
        let mut generic_input = [F::ZERO; WIDTH];
        let mut asm_input = [F::ZERO; WIDTH];

        generic_poseidon2.permute_mut(&mut generic_input);
        asm_poseidon2.permute_mut(&mut asm_input);

        for i in 0..WIDTH {
            assert_eq!(
                asm_input[i].as_canonical_u64(),
                generic_input[i].as_canonical_u64(),
                "ASM mismatch at index {i} for zero input"
            );
        }

        // Test with random input
        let mut generic_input: [F; WIDTH] = core::array::from_fn(|_| F::from_u64(rng.random()));
        let mut asm_input = generic_input;

        generic_poseidon2.permute_mut(&mut generic_input);
        asm_poseidon2.permute_mut(&mut asm_input);

        for i in 0..WIDTH {
            assert_eq!(
                asm_input[i].as_canonical_u64(),
                generic_input[i].as_canonical_u64(),
                "ASM mismatch at index {i} for random input"
            );
        }
    }

    #[test]
    fn test_asm_matches_generic_width_8() {
        test_asm_matches_generic::<8>();
    }

    #[test]
    fn test_asm_matches_generic_width_12() {
        test_asm_matches_generic::<12>();
    }

    #[test]
    fn test_asm_matches_generic_width_16() {
        test_asm_matches_generic::<16>();
    }

    #[test]
    fn test_asm_matches_generic_width_20() {
        test_asm_matches_generic::<20>();
    }
}
