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
use core::arch::aarch64::uint64x2_t;

use p3_field::PrimeField64;
use p3_poseidon2::{
    ExternalLayer, ExternalLayerConstants, ExternalLayerConstructor, InternalLayer,
    InternalLayerConstructor, poseidon2_round_numbers_128,
};
use p3_symmetric::{CryptographicPermutation, Permutation};
use rand::distr::{Distribution, StandardUniform};
use rand::{Rng, RngExt};

use super::packing::PackedGoldilocksNeon;
use super::poseidon2_asm::*;
use super::utils::{pack_lanes, unpack_lanes};
use crate::{Goldilocks, MATRIX_DIAG_20_GOLDILOCKS};

/// Degree of the chosen permutation polynomial for Goldilocks.
const GOLDILOCKS_S_BOX_DEGREE: u64 = 7;

/// ASM-optimized internal layer with split-state s0-in-register, pre-converted constants.
#[derive(Debug, Default, Clone)]
pub struct Poseidon2InternalLayerGoldilocksAsm {
    constants_raw: Vec<u64>,
}

impl InternalLayerConstructor<Goldilocks> for Poseidon2InternalLayerGoldilocksAsm {
    fn new_from_constants(internal_constants: Vec<Goldilocks>) -> Self {
        // The field constructors accept any 64-bit value, so a constant can arrive unreduced.
        // The internal round adds it with the variant that assumes a reduced second operand.
        // Reducing once here makes that precondition hold for every round.
        let constants_raw = internal_constants
            .iter()
            .map(Goldilocks::as_canonical_u64)
            .collect();
        Self { constants_raw }
    }
}

const DIAG_RAW_20: [u64; 20] = {
    let mut arr = [0u64; 20];
    let mut i = 0;
    while i < 20 {
        arr[i] = MATRIX_DIAG_20_GOLDILOCKS[i].value;
        i += 1;
    }
    arr
};

impl InternalLayer<Goldilocks, 8, GOLDILOCKS_S_BOX_DEGREE> for Poseidon2InternalLayerGoldilocksAsm {
    fn permute_state(&self, state: &mut [Goldilocks; 8]) {
        let state_raw: &mut [u64; 8] =
            unsafe { &mut *(state as *mut [Goldilocks; 8] as *mut [u64; 8]) };
        internal_permute_state_asm_w8(state_raw, &self.constants_raw);
    }
}

impl InternalLayer<Goldilocks, 12, GOLDILOCKS_S_BOX_DEGREE>
    for Poseidon2InternalLayerGoldilocksAsm
{
    fn permute_state(&self, state: &mut [Goldilocks; 12]) {
        let state_raw: &mut [u64; 12] =
            unsafe { &mut *(state as *mut [Goldilocks; 12] as *mut [u64; 12]) };
        internal_permute_state_asm_w12(state_raw, &self.constants_raw);
    }
}

impl InternalLayer<Goldilocks, 16, GOLDILOCKS_S_BOX_DEGREE>
    for Poseidon2InternalLayerGoldilocksAsm
{
    fn permute_state(&self, state: &mut [Goldilocks; 16]) {
        let state_raw: &mut [u64; 16] =
            unsafe { &mut *(state as *mut [Goldilocks; 16] as *mut [u64; 16]) };
        internal_permute_state_asm_w16(state_raw, &self.constants_raw);
    }
}

impl InternalLayer<Goldilocks, 20, GOLDILOCKS_S_BOX_DEGREE>
    for Poseidon2InternalLayerGoldilocksAsm
{
    fn permute_state(&self, state: &mut [Goldilocks; 20]) {
        let state_raw: &mut [u64; 20] =
            unsafe { &mut *(state as *mut [Goldilocks; 20] as *mut [u64; 20]) };
        internal_permute_state_asm(state_raw, &DIAG_RAW_20, &self.constants_raw);
    }
}

#[derive(Clone)]
pub struct Poseidon2ExternalLayerGoldilocksAsm<const WIDTH: usize> {
    initial_constants_raw: Vec<[u64; WIDTH]>,
    terminal_constants_raw: Vec<[u64; WIDTH]>,
}

impl<const WIDTH: usize> ExternalLayerConstructor<Goldilocks, WIDTH>
    for Poseidon2ExternalLayerGoldilocksAsm<WIDTH>
{
    fn new_from_constants(external_constants: ExternalLayerConstants<Goldilocks, WIDTH>) -> Self {
        // Only the width-8 fused rounds add a constant with the reduced-operand variant.
        // The other widths use the variant that reduces that operand itself.
        // Reducing unconditionally is free here and holds for any future width.
        let initial_constants_raw = external_constants
            .get_initial_constants()
            .iter()
            .map(|rc| core::array::from_fn(|i| rc[i].as_canonical_u64()))
            .collect();
        let terminal_constants_raw = external_constants
            .get_terminal_constants()
            .iter()
            .map(|rc| core::array::from_fn(|i| rc[i].as_canonical_u64()))
            .collect();
        Self {
            initial_constants_raw,
            terminal_constants_raw,
        }
    }
}

impl ExternalLayer<Goldilocks, 8, GOLDILOCKS_S_BOX_DEGREE>
    for Poseidon2ExternalLayerGoldilocksAsm<8>
{
    fn permute_state_initial(&self, state: &mut [Goldilocks; 8]) {
        let state_raw: &mut [u64; 8] =
            unsafe { &mut *(state as *mut [Goldilocks; 8] as *mut [u64; 8]) };
        external_initial_permute_w8(state_raw, &self.initial_constants_raw);
    }

    fn permute_state_terminal(&self, state: &mut [Goldilocks; 8]) {
        let state_raw: &mut [u64; 8] =
            unsafe { &mut *(state as *mut [Goldilocks; 8] as *mut [u64; 8]) };
        external_terminal_permute_w8(state_raw, &self.terminal_constants_raw);
    }
}

impl ExternalLayer<Goldilocks, 12, GOLDILOCKS_S_BOX_DEGREE>
    for Poseidon2ExternalLayerGoldilocksAsm<12>
{
    fn permute_state_initial(&self, state: &mut [Goldilocks; 12]) {
        let state_raw: &mut [u64; 12] =
            unsafe { &mut *(state as *mut [Goldilocks; 12] as *mut [u64; 12]) };
        external_initial_permute_state_asm(state_raw, &self.initial_constants_raw);
    }

    fn permute_state_terminal(&self, state: &mut [Goldilocks; 12]) {
        let state_raw: &mut [u64; 12] =
            unsafe { &mut *(state as *mut [Goldilocks; 12] as *mut [u64; 12]) };
        external_terminal_permute_state_asm(state_raw, &self.terminal_constants_raw);
    }
}

impl ExternalLayer<Goldilocks, 16, GOLDILOCKS_S_BOX_DEGREE>
    for Poseidon2ExternalLayerGoldilocksAsm<16>
{
    fn permute_state_initial(&self, state: &mut [Goldilocks; 16]) {
        let state_raw: &mut [u64; 16] =
            unsafe { &mut *(state as *mut [Goldilocks; 16] as *mut [u64; 16]) };
        external_initial_permute_state_asm(state_raw, &self.initial_constants_raw);
    }

    fn permute_state_terminal(&self, state: &mut [Goldilocks; 16]) {
        let state_raw: &mut [u64; 16] =
            unsafe { &mut *(state as *mut [Goldilocks; 16] as *mut [u64; 16]) };
        external_terminal_permute_state_asm(state_raw, &self.terminal_constants_raw);
    }
}

impl ExternalLayer<Goldilocks, 20, GOLDILOCKS_S_BOX_DEGREE>
    for Poseidon2ExternalLayerGoldilocksAsm<20>
{
    fn permute_state_initial(&self, state: &mut [Goldilocks; 20]) {
        let state_raw: &mut [u64; 20] =
            unsafe { &mut *(state as *mut [Goldilocks; 20] as *mut [u64; 20]) };
        external_initial_permute_state_asm(state_raw, &self.initial_constants_raw);
    }

    fn permute_state_terminal(&self, state: &mut [Goldilocks; 20]) {
        let state_raw: &mut [u64; 20] =
            unsafe { &mut *(state as *mut [Goldilocks; 20] as *mut [u64; 20]) };
        external_terminal_permute_state_asm(state_raw, &self.terminal_constants_raw);
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

impl InternalLayer<PackedGoldilocksNeon, 8, GOLDILOCKS_S_BOX_DEGREE>
    for Poseidon2InternalLayerGoldilocksAsm
{
    fn permute_state(&self, state: &mut [PackedGoldilocksNeon; 8]) {
        let (mut lane0, mut lane1) = unpack_lanes(state);
        internal_permute_split_dual_w8(&mut lane0, &mut lane1, &self.constants_raw);
        pack_lanes(state, &lane0, &lane1);
    }
}

impl InternalLayer<PackedGoldilocksNeon, 12, GOLDILOCKS_S_BOX_DEGREE>
    for Poseidon2InternalLayerGoldilocksAsm
{
    fn permute_state(&self, state: &mut [PackedGoldilocksNeon; 12]) {
        let (mut lane0, mut lane1) = unpack_lanes(state);
        internal_permute_split_dual_w12(&mut lane0, &mut lane1, &self.constants_raw);
        pack_lanes(state, &lane0, &lane1);
    }
}

impl InternalLayer<PackedGoldilocksNeon, 16, GOLDILOCKS_S_BOX_DEGREE>
    for Poseidon2InternalLayerGoldilocksAsm
{
    fn permute_state(&self, state: &mut [PackedGoldilocksNeon; 16]) {
        let (mut lane0, mut lane1) = unpack_lanes(state);
        internal_permute_split_dual_w16(&mut lane0, &mut lane1, &self.constants_raw);
        pack_lanes(state, &lane0, &lane1);
    }
}

impl InternalLayer<PackedGoldilocksNeon, 20, GOLDILOCKS_S_BOX_DEGREE>
    for Poseidon2InternalLayerGoldilocksAsm
{
    fn permute_state(&self, state: &mut [PackedGoldilocksNeon; 20]) {
        let (mut lane0, mut lane1) = unpack_lanes(state);
        internal_permute_split_dual(&mut lane0, &mut lane1, &DIAG_RAW_20, &self.constants_raw);
        pack_lanes(state, &lane0, &lane1);
    }
}

impl ExternalLayer<PackedGoldilocksNeon, 8, GOLDILOCKS_S_BOX_DEGREE>
    for Poseidon2ExternalLayerGoldilocksAsm<8>
{
    fn permute_state_initial(&self, state: &mut [PackedGoldilocksNeon; 8]) {
        let (mut lane0, mut lane1) = unpack_lanes(state);
        external_initial_permute_dual_w8(&mut lane0, &mut lane1, &self.initial_constants_raw);
        pack_lanes(state, &lane0, &lane1);
    }

    fn permute_state_terminal(&self, state: &mut [PackedGoldilocksNeon; 8]) {
        let (mut lane0, mut lane1) = unpack_lanes(state);
        external_terminal_permute_dual_w8(&mut lane0, &mut lane1, &self.terminal_constants_raw);
        pack_lanes(state, &lane0, &lane1);
    }
}

impl ExternalLayer<PackedGoldilocksNeon, 12, GOLDILOCKS_S_BOX_DEGREE>
    for Poseidon2ExternalLayerGoldilocksAsm<12>
{
    fn permute_state_initial(&self, state: &mut [PackedGoldilocksNeon; 12]) {
        let (mut lane0, mut lane1) = unpack_lanes(state);
        external_initial_permute_dual(&mut lane0, &mut lane1, &self.initial_constants_raw);
        pack_lanes(state, &lane0, &lane1);
    }

    fn permute_state_terminal(&self, state: &mut [PackedGoldilocksNeon; 12]) {
        let (mut lane0, mut lane1) = unpack_lanes(state);
        external_terminal_permute_dual(&mut lane0, &mut lane1, &self.terminal_constants_raw);
        pack_lanes(state, &lane0, &lane1);
    }
}

impl ExternalLayer<PackedGoldilocksNeon, 16, GOLDILOCKS_S_BOX_DEGREE>
    for Poseidon2ExternalLayerGoldilocksAsm<16>
{
    fn permute_state_initial(&self, state: &mut [PackedGoldilocksNeon; 16]) {
        let (mut lane0, mut lane1) = unpack_lanes(state);
        external_initial_permute_dual(&mut lane0, &mut lane1, &self.initial_constants_raw);
        pack_lanes(state, &lane0, &lane1);
    }

    fn permute_state_terminal(&self, state: &mut [PackedGoldilocksNeon; 16]) {
        let (mut lane0, mut lane1) = unpack_lanes(state);
        external_terminal_permute_dual(&mut lane0, &mut lane1, &self.terminal_constants_raw);
        pack_lanes(state, &lane0, &lane1);
    }
}

impl ExternalLayer<PackedGoldilocksNeon, 20, GOLDILOCKS_S_BOX_DEGREE>
    for Poseidon2ExternalLayerGoldilocksAsm<20>
{
    fn permute_state_initial(&self, state: &mut [PackedGoldilocksNeon; 20]) {
        let (mut lane0, mut lane1) = unpack_lanes(state);
        external_initial_permute_dual(&mut lane0, &mut lane1, &self.initial_constants_raw);
        pack_lanes(state, &lane0, &lane1);
    }

    fn permute_state_terminal(&self, state: &mut [PackedGoldilocksNeon; 20]) {
        let (mut lane0, mut lane1) = unpack_lanes(state);
        external_terminal_permute_dual(&mut lane0, &mut lane1, &self.terminal_constants_raw);
        pack_lanes(state, &lane0, &lane1);
    }
}

/// Fused Poseidon2 permutation for Goldilocks.
///
/// Instead of unpacking/packing between each of the 3 phases (initial external,
/// internal, terminal external), this performs a single unpack at the start and
/// a single pack at the end, eliminating the redundant lane conversions per
/// packed permutation.
#[derive(Clone, Debug)]
pub struct Poseidon2GoldilocksFused<const WIDTH: usize> {
    internal_constants_raw: Vec<u64>,
    initial_constants_raw: Vec<[u64; WIDTH]>,
    terminal_constants_raw: Vec<[u64; WIDTH]>,
}

impl<const WIDTH: usize> Poseidon2GoldilocksFused<WIDTH> {
    /// Unlike the generic `Poseidon2::new`, this only reads the round constants (to derive
    /// the fused ASM constant tables) rather than storing them, so it takes them by
    /// reference instead of by value.
    pub fn new(
        external_constants: &ExternalLayerConstants<Goldilocks, WIDTH>,
        internal_constants: &[Goldilocks],
    ) -> Self {
        let internal_constants_raw = internal_constants
            .iter()
            .map(Goldilocks::as_canonical_u64)
            .collect();
        let initial_constants_raw = external_constants
            .get_initial_constants()
            .iter()
            .map(|rc| core::array::from_fn(|i| rc[i].as_canonical_u64()))
            .collect();
        let terminal_constants_raw = external_constants
            .get_terminal_constants()
            .iter()
            .map(|rc| core::array::from_fn(|i| rc[i].as_canonical_u64()))
            .collect();
        Self {
            internal_constants_raw,
            initial_constants_raw,
            terminal_constants_raw,
        }
    }

    pub fn new_from_rng<R: Rng>(rounds_f: usize, rounds_p: usize, rng: &mut R) -> Self
    where
        StandardUniform: Distribution<Goldilocks> + Distribution<[Goldilocks; WIDTH]>,
    {
        let external_constants = ExternalLayerConstants::new_from_rng(rounds_f, rng);
        let internal_constants = rng
            .sample_iter(StandardUniform)
            .take(rounds_p)
            .collect::<Vec<_>>();
        Self::new(&external_constants, &internal_constants)
    }

    pub fn new_from_rng_128<R: Rng>(rng: &mut R) -> Self
    where
        StandardUniform: Distribution<Goldilocks> + Distribution<[Goldilocks; WIDTH]>,
    {
        let round_numbers =
            poseidon2_round_numbers_128::<Goldilocks>(WIDTH, GOLDILOCKS_S_BOX_DEGREE);
        let (rounds_f, rounds_p) = round_numbers.unwrap_or_else(|e| panic!("{e}"));
        Self::new_from_rng(rounds_f, rounds_p, rng)
    }
}

impl Permutation<[Goldilocks; 8]> for Poseidon2GoldilocksFused<8> {
    fn permute_mut(&self, state: &mut [Goldilocks; 8]) {
        let state_raw: &mut [u64; 8] =
            unsafe { &mut *(state as *mut [Goldilocks; 8] as *mut [u64; 8]) };
        external_initial_permute_w8(state_raw, &self.initial_constants_raw);
        internal_permute_state_asm_w8(state_raw, &self.internal_constants_raw);
        external_terminal_permute_w8(state_raw, &self.terminal_constants_raw);
    }
}

impl CryptographicPermutation<[Goldilocks; 8]> for Poseidon2GoldilocksFused<8> {}

impl Permutation<[Goldilocks; 12]> for Poseidon2GoldilocksFused<12> {
    fn permute_mut(&self, state: &mut [Goldilocks; 12]) {
        let state_raw: &mut [u64; 12] =
            unsafe { &mut *(state as *mut [Goldilocks; 12] as *mut [u64; 12]) };
        external_initial_permute_state_asm(state_raw, &self.initial_constants_raw);
        internal_permute_state_asm_w12(state_raw, &self.internal_constants_raw);
        external_terminal_permute_state_asm(state_raw, &self.terminal_constants_raw);
    }
}

impl CryptographicPermutation<[Goldilocks; 12]> for Poseidon2GoldilocksFused<12> {}

impl Permutation<[Goldilocks; 16]> for Poseidon2GoldilocksFused<16> {
    fn permute_mut(&self, state: &mut [Goldilocks; 16]) {
        let state_raw: &mut [u64; 16] =
            unsafe { &mut *(state as *mut [Goldilocks; 16] as *mut [u64; 16]) };
        external_initial_permute_state_asm(state_raw, &self.initial_constants_raw);
        internal_permute_state_asm_w16(state_raw, &self.internal_constants_raw);
        external_terminal_permute_state_asm(state_raw, &self.terminal_constants_raw);
    }
}

impl CryptographicPermutation<[Goldilocks; 16]> for Poseidon2GoldilocksFused<16> {}

impl Permutation<[Goldilocks; 20]> for Poseidon2GoldilocksFused<20> {
    fn permute_mut(&self, state: &mut [Goldilocks; 20]) {
        let state_raw: &mut [u64; 20] =
            unsafe { &mut *(state as *mut [Goldilocks; 20] as *mut [u64; 20]) };
        external_initial_permute_state_asm(state_raw, &self.initial_constants_raw);
        internal_permute_state_asm(state_raw, &DIAG_RAW_20, &self.internal_constants_raw);
        external_terminal_permute_state_asm(state_raw, &self.terminal_constants_raw);
    }
}

impl CryptographicPermutation<[Goldilocks; 20]> for Poseidon2GoldilocksFused<20> {}

impl Permutation<[PackedGoldilocksNeon; 8]> for Poseidon2GoldilocksFused<8> {
    fn permute_mut(&self, state: &mut [PackedGoldilocksNeon; 8]) {
        let (mut lane0, mut lane1) = unpack_lanes(state);
        external_initial_permute_dual_w8(&mut lane0, &mut lane1, &self.initial_constants_raw);
        internal_permute_split_dual_w8(&mut lane0, &mut lane1, &self.internal_constants_raw);
        external_terminal_permute_dual_w8(&mut lane0, &mut lane1, &self.terminal_constants_raw);
        pack_lanes(state, &lane0, &lane1);
    }
}

impl CryptographicPermutation<[PackedGoldilocksNeon; 8]> for Poseidon2GoldilocksFused<8> {}

impl Permutation<[PackedGoldilocksNeon; 12]> for Poseidon2GoldilocksFused<12> {
    fn permute_mut(&self, state: &mut [PackedGoldilocksNeon; 12]) {
        let mut sv: [uint64x2_t; 12] = core::array::from_fn(|i| state[i].to_vector());
        external_initial_neon(&mut sv, &self.initial_constants_raw);
        internal_permute_neon_w12(&mut sv, &self.internal_constants_raw);
        external_terminal_neon(&mut sv, &self.terminal_constants_raw);
        for (s, &v) in state.iter_mut().zip(sv.iter()) {
            *s = PackedGoldilocksNeon::from_vector(v);
        }
    }
}

impl CryptographicPermutation<[PackedGoldilocksNeon; 12]> for Poseidon2GoldilocksFused<12> {}

impl Permutation<[PackedGoldilocksNeon; 16]> for Poseidon2GoldilocksFused<16> {
    fn permute_mut(&self, state: &mut [PackedGoldilocksNeon; 16]) {
        let mut sv: [uint64x2_t; 16] = core::array::from_fn(|i| state[i].to_vector());
        external_initial_neon(&mut sv, &self.initial_constants_raw);
        internal_permute_neon_w16(&mut sv, &self.internal_constants_raw);
        external_terminal_neon(&mut sv, &self.terminal_constants_raw);
        for (s, &v) in state.iter_mut().zip(sv.iter()) {
            *s = PackedGoldilocksNeon::from_vector(v);
        }
    }
}

impl CryptographicPermutation<[PackedGoldilocksNeon; 16]> for Poseidon2GoldilocksFused<16> {}

impl Permutation<[PackedGoldilocksNeon; 20]> for Poseidon2GoldilocksFused<20> {
    fn permute_mut(&self, state: &mut [PackedGoldilocksNeon; 20]) {
        let mut sv: [uint64x2_t; 20] = core::array::from_fn(|i| state[i].to_vector());
        external_initial_neon(&mut sv, &self.initial_constants_raw);
        internal_permute_neon(&mut sv, &DIAG_RAW_20, &self.internal_constants_raw);
        external_terminal_neon(&mut sv, &self.terminal_constants_raw);
        for (s, &v) in state.iter_mut().zip(sv.iter()) {
            *s = PackedGoldilocksNeon::from_vector(v);
        }
    }
}

impl CryptographicPermutation<[PackedGoldilocksNeon; 20]> for Poseidon2GoldilocksFused<20> {}

#[cfg(test)]
mod tests {
    use p3_field::{PrimeCharacteristicRing, PrimeField64};
    use p3_poseidon2::{ExternalLayerConstants, InternalLayer, Poseidon2};
    use p3_symmetric::Permutation;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;
    use crate::poseidon1::GOLDILOCKS_S_BOX_DEGREE;
    use crate::{
        GOLDILOCKS_POSEIDON2_HALF_FULL_ROUNDS, GOLDILOCKS_POSEIDON2_PARTIAL_ROUNDS_8, P,
        Poseidon2ExternalLayerGoldilocks, Poseidon2InternalLayerGoldilocks,
    };

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

        let external_constants = ExternalLayerConstants::<Goldilocks, WIDTH>::new_from_rng(
            2 * GOLDILOCKS_POSEIDON2_HALF_FULL_ROUNDS,
            &mut rng,
        );
        let internal_constants: Vec<Goldilocks> = (0..GOLDILOCKS_POSEIDON2_PARTIAL_ROUNDS_8)
            .map(|_| F::from_u64(rng.random()))
            .collect();

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

    fn test_fused_matches_generic<const WIDTH: usize>()
    where
        Poseidon2InternalLayerGoldilocks: InternalLayer<F, WIDTH, GOLDILOCKS_S_BOX_DEGREE>,
        Poseidon2GoldilocksFused<WIDTH>:
            Permutation<[F; WIDTH]> + Permutation<[PackedGoldilocksNeon; WIDTH]>,
    {
        let mut rng = SmallRng::seed_from_u64(42);

        let external_constants = ExternalLayerConstants::<Goldilocks, WIDTH>::new_from_rng(
            2 * GOLDILOCKS_POSEIDON2_HALF_FULL_ROUNDS,
            &mut rng,
        );
        let internal_constants: Vec<Goldilocks> = (0..GOLDILOCKS_POSEIDON2_PARTIAL_ROUNDS_8)
            .map(|_| rng.random())
            .collect();

        let generic_poseidon2: Poseidon2<
            Goldilocks,
            Poseidon2ExternalLayerGoldilocks<WIDTH>,
            Poseidon2InternalLayerGoldilocks,
            WIDTH,
            GOLDILOCKS_S_BOX_DEGREE,
        > = Poseidon2::new(external_constants.clone(), internal_constants.clone());

        let fused =
            Poseidon2GoldilocksFused::<WIDTH>::new(&external_constants, &internal_constants);

        // Scalar: fused vs generic
        let mut generic_input = [F::ZERO; WIDTH];
        let mut fused_input = [F::ZERO; WIDTH];
        generic_poseidon2.permute_mut(&mut generic_input);
        fused.permute_mut(&mut fused_input);
        for i in 0..WIDTH {
            assert_eq!(
                fused_input[i].as_canonical_u64(),
                generic_input[i].as_canonical_u64(),
                "Fused scalar mismatch at index {i} for zero input"
            );
        }

        let mut generic_input: [F; WIDTH] = rng.random();
        let mut fused_input = generic_input;
        generic_poseidon2.permute_mut(&mut generic_input);
        fused.permute_mut(&mut fused_input);
        for i in 0..WIDTH {
            assert_eq!(
                fused_input[i].as_canonical_u64(),
                generic_input[i].as_canonical_u64(),
                "Fused scalar mismatch at index {i} for random input"
            );
        }

        // Packed: fused packed vs scalar (each packed lane should match scalar)
        let scalar_a: [F; WIDTH] = rng.random();
        let scalar_b: [F; WIDTH] = rng.random();

        let mut packed_input: [PackedGoldilocksNeon; WIDTH] =
            core::array::from_fn(|i| PackedGoldilocksNeon([scalar_a[i], scalar_b[i]]));
        fused.permute_mut(&mut packed_input);

        let mut expected_a = scalar_a;
        let mut expected_b = scalar_b;
        fused.permute_mut(&mut expected_a);
        fused.permute_mut(&mut expected_b);

        for i in 0..WIDTH {
            assert_eq!(
                packed_input[i].0[0].as_canonical_u64(),
                expected_a[i].as_canonical_u64(),
                "Fused packed lane0 mismatch at index {i}"
            );
            assert_eq!(
                packed_input[i].0[1].as_canonical_u64(),
                expected_b[i].as_canonical_u64(),
                "Fused packed lane1 mismatch at index {i}"
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
    fn test_asm_accepts_non_canonical_round_constants() {
        // Invariant: a constant stored above the modulus permutes like its reduced twin.
        //
        // The field constructors accept any 64-bit value.
        // Roughly one sampled constant in 2^32 lands above the modulus.
        const WIDTH: usize = 8;
        let mut rng = SmallRng::seed_from_u64(7);

        let external_constants = ExternalLayerConstants::<Goldilocks, WIDTH>::new_from_rng(
            2 * GOLDILOCKS_POSEIDON2_HALF_FULL_ROUNDS,
            &mut rng,
        );
        let mut internal_constants: Vec<Goldilocks> = (0..GOLDILOCKS_POSEIDON2_PARTIAL_ROUNDS_8)
            .map(|_| rng.random())
            .collect();
        // Mutation: two internal round constants get an unreduced representative.
        //
        //     P + 1     -> the field element 1
        //     2^64 - 1  -> P + 2^32 - 2, i.e. the field element 2^32 - 2
        internal_constants[0] = Goldilocks::new(P + 1);
        internal_constants[5] = Goldilocks::new(u64::MAX);
        let mut initial = external_constants.get_initial_constants().to_vec();
        // Mutation: one initial external round constant gets an unreduced representative.
        //
        // The light MDS permutation runs first and leaves the whole state reduced.
        // The addition is then exact, so this constant alone changes nothing in a release build.
        // A debug build still catches it, because the addition asserts its precondition.
        initial[0][3] = Goldilocks::new(u64::MAX);
        let non_canonical_external = ExternalLayerConstants::new(
            initial,
            external_constants.get_terminal_constants().to_vec(),
        );

        let generic: Poseidon2<
            Goldilocks,
            Poseidon2ExternalLayerGoldilocks<WIDTH>,
            Poseidon2InternalLayerGoldilocks,
            WIDTH,
            GOLDILOCKS_S_BOX_DEGREE,
        > = Poseidon2::new(non_canonical_external.clone(), internal_constants.clone());
        let asm: Poseidon2GoldilocksAsm<WIDTH> =
            Poseidon2::new(non_canonical_external, internal_constants.clone());

        for seed in 0..32u64 {
            let mut state = [F::ZERO; WIDTH];
            for (i, v) in state.iter_mut().enumerate() {
                *v = Goldilocks::new(u64::MAX - seed * 131 - i as u64);
            }
            let mut generic_state = state;
            asm.permute_mut(&mut state);
            generic.permute_mut(&mut generic_state);
            assert_eq!(state, generic_state, "state seed {seed}");
        }

        // Fixture state: the internal layer alone, over a state whose first word is unreduced.
        //
        // That first word is the only place an unreduced constant changes the field element.
        // Elsewhere the light MDS permutation runs first and leaves the state reduced.
        let asm_internal =
            Poseidon2InternalLayerGoldilocksAsm::new_from_constants(internal_constants.clone());
        let generic_internal =
            Poseidon2InternalLayerGoldilocks::new_from_constants(internal_constants);
        let mut asm_state = [F::ZERO; WIDTH];
        asm_state[0] = Goldilocks::new(u64::MAX);
        let mut generic_state = asm_state;
        InternalLayer::<F, WIDTH, GOLDILOCKS_S_BOX_DEGREE>::permute_state(
            &asm_internal,
            &mut asm_state,
        );
        InternalLayer::<F, WIDTH, GOLDILOCKS_S_BOX_DEGREE>::permute_state(
            &generic_internal,
            &mut generic_state,
        );
        assert_eq!(asm_state, generic_state);

        // Fixture state: two independent lanes fed through the dual-lane packed rounds.
        //
        //     lane 0 : 2^64 - 1 - i      unreduced, just below the wrap point
        //     lane 1 : P + i             unreduced, just above the modulus
        //
        // The dual-lane rounds are a third path into the reduced-operand addition.
        // They share the constant tables with the scalar path, so nothing else covers them.
        let lane_a: [F; WIDTH] = core::array::from_fn(|i| Goldilocks::new(u64::MAX - i as u64));
        let lane_b: [F; WIDTH] = core::array::from_fn(|i| Goldilocks::new(P + i as u64));
        // Interleave the two lanes into the packed state the SIMD permutation consumes.
        let mut packed: [PackedGoldilocksNeon; WIDTH] =
            core::array::from_fn(|i| PackedGoldilocksNeon([lane_a[i], lane_b[i]]));
        asm.permute_mut(&mut packed);

        // Reference: permute each lane separately through the generic implementation.
        let (mut expected_a, mut expected_b) = (lane_a, lane_b);
        generic.permute_mut(&mut expected_a);
        generic.permute_mut(&mut expected_b);
        // Both lanes must match their scalar reference word for word.
        for i in 0..WIDTH {
            assert_eq!(
                packed[i].0[0], expected_a[i],
                "packed lane0 mismatch at {i}"
            );
            assert_eq!(
                packed[i].0[1], expected_b[i],
                "packed lane1 mismatch at {i}"
            );
        }
    }

    #[test]
    fn test_asm_matches_generic_width_16() {
        test_asm_matches_generic::<16>();
    }

    #[test]
    fn test_asm_matches_generic_width_20() {
        test_asm_matches_generic::<20>();
    }

    #[test]
    fn test_fused_matches_generic_width_8() {
        test_fused_matches_generic::<8>();
    }

    #[test]
    fn test_fused_matches_generic_width_12() {
        test_fused_matches_generic::<12>();
    }

    #[test]
    fn test_fused_matches_generic_width_16() {
        test_fused_matches_generic::<16>();
    }

    #[test]
    fn test_fused_matches_generic_width_20() {
        test_fused_matches_generic::<20>();
    }
}
