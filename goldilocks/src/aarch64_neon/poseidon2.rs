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
    InternalLayerConstructor, poseidon2_round_numbers_128,
};
use p3_symmetric::{CryptographicPermutation, Permutation};
use rand::distr::{Distribution, StandardUniform};
use rand::{Rng, RngExt};

use super::packing::PackedGoldilocksNeon;
use super::poseidon2_asm::{
    external_initial_neon, external_initial_permute_dual, external_initial_permute_dual_w8,
    external_initial_permute_state_asm, external_initial_permute_w8, external_terminal_neon,
    external_terminal_permute_dual, external_terminal_permute_dual_w8,
    external_terminal_permute_state_asm, external_terminal_permute_w8, internal_permute_neon,
    internal_permute_neon_w12, internal_permute_neon_w16, internal_permute_split_dual,
    internal_permute_split_dual_w8, internal_permute_split_dual_w12,
    internal_permute_split_dual_w16, internal_permute_state_asm, internal_permute_state_asm_w8,
    internal_permute_state_asm_w12, internal_permute_state_asm_w16, lanes_to_neon, neon_to_lanes,
};
use crate::{Goldilocks, MATRIX_DIAG_16_GOLDILOCKS, MATRIX_DIAG_20_GOLDILOCKS};

/// Degree of the chosen permutation polynomial for Goldilocks.
const GOLDILOCKS_S_BOX_DEGREE: u64 = 7;

/// ASM-optimized internal layer with split-state s0-in-register, pre-converted constants.
#[derive(Debug, Default, Clone)]
pub struct Poseidon2InternalLayerGoldilocksAsm {
    constants_raw: Vec<u64>,
}

impl InternalLayerConstructor<Goldilocks> for Poseidon2InternalLayerGoldilocksAsm {
    fn new_from_constants(internal_constants: Vec<Goldilocks>) -> Self {
        let constants_raw = internal_constants.iter().map(|c| c.value).collect();
        Self { constants_raw }
    }
}

const DIAG_RAW_16: [u64; 16] = [
    MATRIX_DIAG_16_GOLDILOCKS[0].value,
    MATRIX_DIAG_16_GOLDILOCKS[1].value,
    MATRIX_DIAG_16_GOLDILOCKS[2].value,
    MATRIX_DIAG_16_GOLDILOCKS[3].value,
    MATRIX_DIAG_16_GOLDILOCKS[4].value,
    MATRIX_DIAG_16_GOLDILOCKS[5].value,
    MATRIX_DIAG_16_GOLDILOCKS[6].value,
    MATRIX_DIAG_16_GOLDILOCKS[7].value,
    MATRIX_DIAG_16_GOLDILOCKS[8].value,
    MATRIX_DIAG_16_GOLDILOCKS[9].value,
    MATRIX_DIAG_16_GOLDILOCKS[10].value,
    MATRIX_DIAG_16_GOLDILOCKS[11].value,
    MATRIX_DIAG_16_GOLDILOCKS[12].value,
    MATRIX_DIAG_16_GOLDILOCKS[13].value,
    MATRIX_DIAG_16_GOLDILOCKS[14].value,
    MATRIX_DIAG_16_GOLDILOCKS[15].value,
];

const DIAG_RAW_20: [u64; 20] = [
    MATRIX_DIAG_20_GOLDILOCKS[0].value,
    MATRIX_DIAG_20_GOLDILOCKS[1].value,
    MATRIX_DIAG_20_GOLDILOCKS[2].value,
    MATRIX_DIAG_20_GOLDILOCKS[3].value,
    MATRIX_DIAG_20_GOLDILOCKS[4].value,
    MATRIX_DIAG_20_GOLDILOCKS[5].value,
    MATRIX_DIAG_20_GOLDILOCKS[6].value,
    MATRIX_DIAG_20_GOLDILOCKS[7].value,
    MATRIX_DIAG_20_GOLDILOCKS[8].value,
    MATRIX_DIAG_20_GOLDILOCKS[9].value,
    MATRIX_DIAG_20_GOLDILOCKS[10].value,
    MATRIX_DIAG_20_GOLDILOCKS[11].value,
    MATRIX_DIAG_20_GOLDILOCKS[12].value,
    MATRIX_DIAG_20_GOLDILOCKS[13].value,
    MATRIX_DIAG_20_GOLDILOCKS[14].value,
    MATRIX_DIAG_20_GOLDILOCKS[15].value,
    MATRIX_DIAG_20_GOLDILOCKS[16].value,
    MATRIX_DIAG_20_GOLDILOCKS[17].value,
    MATRIX_DIAG_20_GOLDILOCKS[18].value,
    MATRIX_DIAG_20_GOLDILOCKS[19].value,
];

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
        internal_permute_state_asm_w16(state_raw, &DIAG_RAW_16, &self.constants_raw);
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
        let initial_constants_raw = external_constants
            .get_initial_constants()
            .iter()
            .map(|rc| core::array::from_fn(|i| rc[i].value))
            .collect();
        let terminal_constants_raw = external_constants
            .get_terminal_constants()
            .iter()
            .map(|rc| core::array::from_fn(|i| rc[i].value))
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

/// Extract packed state into two raw u64 lane arrays.
#[inline]
fn unpack_lanes<const WIDTH: usize>(
    state: &[PackedGoldilocksNeon; WIDTH],
) -> ([u64; WIDTH], [u64; WIDTH]) {
    let lane0: [u64; WIDTH] = core::array::from_fn(|i| state[i].0[0].value);
    let lane1: [u64; WIDTH] = core::array::from_fn(|i| state[i].0[1].value);
    (lane0, lane1)
}

/// Pack two raw u64 lane arrays back into packed state.
#[inline]
fn pack_lanes<const WIDTH: usize>(
    state: &mut [PackedGoldilocksNeon; WIDTH],
    lane0: &[u64; WIDTH],
    lane1: &[u64; WIDTH],
) {
    for i in 0..WIDTH {
        state[i] = PackedGoldilocksNeon([Goldilocks::new(lane0[i]), Goldilocks::new(lane1[i])]);
    }
}

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
        internal_permute_split_dual_w16(&mut lane0, &mut lane1, &DIAG_RAW_16, &self.constants_raw);
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

/// Fused Poseidon2 permutation for Goldilocks on aarch64.
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
    pub fn new(
        external_constants: &ExternalLayerConstants<Goldilocks, WIDTH>,
        internal_constants: &[Goldilocks],
    ) -> Self {
        let internal_constants_raw = internal_constants.iter().map(|c| c.value).collect();
        let initial_constants_raw = external_constants
            .get_initial_constants()
            .iter()
            .map(|rc| core::array::from_fn(|i| rc[i].value))
            .collect();
        let terminal_constants_raw = external_constants
            .get_terminal_constants()
            .iter()
            .map(|rc| core::array::from_fn(|i| rc[i].value))
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
        internal_permute_state_asm_w16(state_raw, &DIAG_RAW_16, &self.internal_constants_raw);
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
        let (mut lane0, mut lane1) = unpack_lanes(state);
        let mut sv = lanes_to_neon(&lane0, &lane1);
        external_initial_neon(&mut sv, &self.initial_constants_raw);
        internal_permute_neon_w12(&mut sv, &self.internal_constants_raw);
        external_terminal_neon(&mut sv, &self.terminal_constants_raw);
        neon_to_lanes(&sv, &mut lane0, &mut lane1);
        pack_lanes(state, &lane0, &lane1);
    }
}

impl CryptographicPermutation<[PackedGoldilocksNeon; 12]> for Poseidon2GoldilocksFused<12> {}

impl Permutation<[PackedGoldilocksNeon; 16]> for Poseidon2GoldilocksFused<16> {
    fn permute_mut(&self, state: &mut [PackedGoldilocksNeon; 16]) {
        let (mut lane0, mut lane1) = unpack_lanes(state);
        let mut sv = lanes_to_neon(&lane0, &lane1);
        external_initial_neon(&mut sv, &self.initial_constants_raw);
        internal_permute_neon_w16(&mut sv, &DIAG_RAW_16, &self.internal_constants_raw);
        external_terminal_neon(&mut sv, &self.terminal_constants_raw);
        neon_to_lanes(&sv, &mut lane0, &mut lane1);
        pack_lanes(state, &lane0, &lane1);
    }
}

impl CryptographicPermutation<[PackedGoldilocksNeon; 16]> for Poseidon2GoldilocksFused<16> {}

impl Permutation<[PackedGoldilocksNeon; 20]> for Poseidon2GoldilocksFused<20> {
    fn permute_mut(&self, state: &mut [PackedGoldilocksNeon; 20]) {
        let (mut lane0, mut lane1) = unpack_lanes(state);
        let mut sv = lanes_to_neon(&lane0, &lane1);
        external_initial_neon(&mut sv, &self.initial_constants_raw);
        internal_permute_neon(&mut sv, &DIAG_RAW_20, &self.internal_constants_raw);
        external_terminal_neon(&mut sv, &self.terminal_constants_raw);
        neon_to_lanes(&sv, &mut lane0, &mut lane1);
        pack_lanes(state, &lane0, &lane1);
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

    fn test_fused_matches_generic<const WIDTH: usize>()
    where
        Poseidon2InternalLayerGoldilocks: InternalLayer<F, WIDTH, GOLDILOCKS_S_BOX_DEGREE>,
        Poseidon2GoldilocksFused<WIDTH>:
            Permutation<[F; WIDTH]> + Permutation<[PackedGoldilocksNeon; WIDTH]>,
    {
        let mut rng = SmallRng::seed_from_u64(42);

        let external_constants =
            ExternalLayerConstants::<Goldilocks, WIDTH>::new_from_rng(4, &mut rng);
        let internal_constants: Vec<Goldilocks> =
            (0..22).map(|_| F::from_u64(rng.random())).collect();

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

        let mut generic_input: [F; WIDTH] = core::array::from_fn(|_| F::from_u64(rng.random()));
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
        let scalar_a: [F; WIDTH] = core::array::from_fn(|_| F::from_u64(rng.random()));
        let scalar_b: [F; WIDTH] = core::array::from_fn(|_| F::from_u64(rng.random()));

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
