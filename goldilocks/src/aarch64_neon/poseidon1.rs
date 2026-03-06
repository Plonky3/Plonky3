//! Fused Poseidon1 permutation for Goldilocks on aarch64.

use alloc::vec::Vec;

use p3_poseidon::{
    FullRoundConstants, PartialRoundConstants, full_round_initial_permute_state,
    full_round_terminal_permute_state, partial_permute_state,
};
use p3_symmetric::{CryptographicPermutation, Permutation};

use super::mds::{MdsNeonGoldilocks, mds_neon_w8, mds_neon_w12};
use super::packing::PackedGoldilocksNeon;
use super::poseidon1_asm::*;
use super::poseidon2_asm::{sbox_layer_asm, sbox_layer_dual_asm};
use super::utils::{pack_lanes, unpack_lanes};
use crate::Goldilocks;
use crate::poseidon::PoseidonGoldilocksGeneric;

/// Fused Poseidon1 permutation for Goldilocks.
///
/// Holds the pre-extracted raw `u64` constants from the optimized Poseidon1
/// sparse-matrix decomposition. Storing raw values avoids field-element
/// overhead in the hot inner loop.
#[derive(Clone, Debug)]
pub struct Poseidon1GoldilocksFused<const WIDTH: usize> {
    /// Round constants for the initial full rounds (RF/2 vectors).
    initial_constants_raw: Vec<[u64; WIDTH]>,
    /// Round constants for the terminal full rounds (RF/2 vectors).
    terminal_constants_raw: Vec<[u64; WIDTH]>,
    /// Full-width constant vector for the first partial round.
    first_round_constants_raw: [u64; WIDTH],
    /// Dense transition matrix applied once before entering the partial-round loop.
    m_i_raw: [[u64; WIDTH]; WIDTH],
    /// Per-round first row of the sparse matrix (one per partial round).
    sparse_first_row_raw: Vec<[u64; WIDTH]>,
    /// Per-round sub-diagonal vector for the sparse matmul (one per partial round).
    v_raw: Vec<[u64; WIDTH]>,
    /// Scalar round constants for partial rounds 0 through RP-2.
    ///
    /// The last partial round has no scalar constant (it ends with the S-box only).
    round_constants_raw: Vec<u64>,
}

impl<const WIDTH: usize> Poseidon1GoldilocksFused<WIDTH> {
    /// Create from pre-computed full and partial round constants.
    ///
    /// Extracts the raw `u64` representation from each Goldilocks field
    /// element, building the flat arrays that the ASM kernels consume.
    pub fn new(
        full: FullRoundConstants<Goldilocks, WIDTH>,
        partial: PartialRoundConstants<Goldilocks, WIDTH>,
    ) -> Self {
        // Extract raw u64 values from full-round constant matrices.
        let initial_constants_raw = full
            .initial
            .iter()
            .map(|rc| core::array::from_fn(|i| rc[i].value))
            .collect();
        let terminal_constants_raw = full
            .terminal
            .iter()
            .map(|rc| core::array::from_fn(|i| rc[i].value))
            .collect();

        // Extract the first partial-round constant vector.
        let first_round_constants_raw =
            core::array::from_fn(|i| partial.first_round_constants[i].value);

        // Extract the dense transition matrix.
        let m_i_raw = core::array::from_fn(|i| core::array::from_fn(|j| partial.m_i[i][j].value));

        // Extract per-round sparse matrix data.
        let sparse_first_row_raw = partial
            .sparse_first_row
            .iter()
            .map(|r| core::array::from_fn(|i| r[i].value))
            .collect();
        let v_raw = partial
            .v
            .iter()
            .map(|r| core::array::from_fn(|i| r[i].value))
            .collect();

        // Extract scalar round constants for partial rounds.
        let round_constants_raw = partial.round_constants.iter().map(|c| c.value).collect();

        Self {
            initial_constants_raw,
            terminal_constants_raw,
            first_round_constants_raw,
            m_i_raw,
            sparse_first_row_raw,
            v_raw,
            round_constants_raw,
        }
    }
}

/// Run the initial or terminal full rounds on a raw width-8 state.
///
/// Each full round applies: add constants, S-box on all elements, NEON MDS.
#[inline]
fn full_rounds_scalar_w8(raw: &mut [u64; 8], constants: &[[u64; 8]]) {
    for rc in constants {
        unsafe {
            add_rc_asm(raw, rc);
            sbox_layer_asm(raw);
        }
        *raw = unsafe { mds_neon_w8(raw) };
    }
}

/// Run the initial or terminal full rounds on a raw width-12 state.
///
/// Each full round applies: add constants, S-box on all elements, NEON MDS.
#[inline]
fn full_rounds_scalar_w12(raw: &mut [u64; 12], constants: &[[u64; 12]]) {
    for rc in constants {
        unsafe {
            add_rc_asm(raw, rc);
            sbox_layer_asm(raw);
        }
        *raw = unsafe { mds_neon_w12(raw) };
    }
}

/// Run all partial rounds on a raw width-8 state.
///
/// The partial-round sequence is:
/// 1. Add the first-round full-width constant vector.
/// 2. Apply the dense transition matrix once.
/// 3. For each partial round (except the last):
///    S-box on first element, add scalar constant, sparse matmul.
/// 4. Last partial round: S-box on first element, sparse matmul (no constant).
#[inline]
fn partial_rounds_scalar_w8(
    raw: &mut [u64; 8],
    first_rc: &[u64; 8],
    m_i: &[[u64; 8]; 8],
    sparse_first_row: &[[u64; 8]],
    v: &[[u64; 8]],
    round_constants: &[u64],
) {
    // Add the first-round full-width constant vector.
    unsafe {
        add_rc_asm(raw, first_rc);
    }

    // Apply the dense transition matrix once.
    dense_matmul_asm_w8(raw, m_i);

    // Main partial-round loop: S-box + scalar constant + sparse matmul.
    let rounds_p = sparse_first_row.len();
    for r in 0..rounds_p - 1 {
        unsafe {
            sbox_s0_asm(raw);
            add_scalar_s0_asm(raw, round_constants[r]);
            cheap_matmul_asm_w8(raw, &sparse_first_row[r], &v[r]);
        }
    }

    // Last partial round: no scalar constant.
    unsafe {
        sbox_s0_asm(raw);
        cheap_matmul_asm_w8(raw, &sparse_first_row[rounds_p - 1], &v[rounds_p - 1]);
    }
}

/// Run all partial rounds on a raw width-12 state.
///
/// Same structure as the width-8 variant.
#[inline]
fn partial_rounds_scalar_w12(
    raw: &mut [u64; 12],
    first_rc: &[u64; 12],
    m_i: &[[u64; 12]; 12],
    sparse_first_row: &[[u64; 12]],
    v: &[[u64; 12]],
    round_constants: &[u64],
) {
    unsafe {
        add_rc_asm(raw, first_rc);
    }
    dense_matmul_asm_w12(raw, m_i);

    let rounds_p = sparse_first_row.len();
    for r in 0..rounds_p - 1 {
        unsafe {
            sbox_s0_asm(raw);
            add_scalar_s0_asm(raw, round_constants[r]);
            cheap_matmul_asm_w12(raw, &sparse_first_row[r], &v[r]);
        }
    }
    unsafe {
        sbox_s0_asm(raw);
        cheap_matmul_asm_w12(raw, &sparse_first_row[rounds_p - 1], &v[rounds_p - 1]);
    }
}

/// Run the initial or terminal full rounds on two raw width-8 lanes.
///
/// Uses dual-lane ASM primitives for add_rc and S-box, then NEON MDS per lane.
#[inline]
fn full_rounds_dual_w8(lane0: &mut [u64; 8], lane1: &mut [u64; 8], constants: &[[u64; 8]]) {
    for rc in constants {
        unsafe {
            add_rc_dual_asm(lane0, lane1, rc);
            sbox_layer_dual_asm(lane0, lane1);
        }
        *lane0 = unsafe { mds_neon_w8(lane0) };
        *lane1 = unsafe { mds_neon_w8(lane1) };
    }
}

/// Run the initial or terminal full rounds on two raw width-12 lanes.
///
/// Uses dual-lane ASM primitives for add_rc and S-box, then NEON MDS per lane.
#[inline]
fn full_rounds_dual_w12(lane0: &mut [u64; 12], lane1: &mut [u64; 12], constants: &[[u64; 12]]) {
    for rc in constants {
        unsafe {
            add_rc_dual_asm(lane0, lane1, rc);
            sbox_layer_dual_asm(lane0, lane1);
        }
        *lane0 = unsafe { mds_neon_w12(lane0) };
        *lane1 = unsafe { mds_neon_w12(lane1) };
    }
}

/// Run all partial rounds on two width-8 lanes simultaneously.
///
/// Uses dual-lane S-box and sparse matmul primitives to keep the
/// pipeline full. The scalar constant is added to each lane separately
/// (no dual variant needed for a single-element addition).
#[inline]
fn partial_rounds_dual_w8(
    lane0: &mut [u64; 8],
    lane1: &mut [u64; 8],
    first_rc: &[u64; 8],
    m_i: &[[u64; 8]; 8],
    sparse_first_row: &[[u64; 8]],
    v: &[[u64; 8]],
    round_constants: &[u64],
) {
    // Add the first-round constant to both lanes.
    unsafe {
        add_rc_dual_asm(lane0, lane1, first_rc);
    }

    // Dense transition matrix on both lanes.
    dense_matmul_dual_asm_w8(lane0, lane1, m_i);

    // Main partial-round loop.
    let rounds_p = sparse_first_row.len();
    for r in 0..rounds_p - 1 {
        unsafe {
            sbox_s0_dual_asm(lane0, lane1);
            add_scalar_s0_asm(lane0, round_constants[r]);
            add_scalar_s0_asm(lane1, round_constants[r]);
            cheap_matmul_dual_asm_w8(lane0, lane1, &sparse_first_row[r], &v[r]);
        }
    }

    // Last partial round: no scalar constant.
    unsafe {
        sbox_s0_dual_asm(lane0, lane1);
        cheap_matmul_dual_asm_w8(
            lane0,
            lane1,
            &sparse_first_row[rounds_p - 1],
            &v[rounds_p - 1],
        );
    }
}

/// Run all partial rounds on two width-12 lanes simultaneously.
///
/// Same structure as the width-8 dual variant.
#[inline]
fn partial_rounds_dual_w12(
    lane0: &mut [u64; 12],
    lane1: &mut [u64; 12],
    first_rc: &[u64; 12],
    m_i: &[[u64; 12]; 12],
    sparse_first_row: &[[u64; 12]],
    v: &[[u64; 12]],
    round_constants: &[u64],
) {
    unsafe {
        add_rc_dual_asm(lane0, lane1, first_rc);
    }
    dense_matmul_dual_asm_w12(lane0, lane1, m_i);

    let rounds_p = sparse_first_row.len();
    for r in 0..rounds_p - 1 {
        unsafe {
            sbox_s0_dual_asm(lane0, lane1);
            add_scalar_s0_asm(lane0, round_constants[r]);
            add_scalar_s0_asm(lane1, round_constants[r]);
            cheap_matmul_dual_asm_w12(lane0, lane1, &sparse_first_row[r], &v[r]);
        }
    }
    unsafe {
        sbox_s0_dual_asm(lane0, lane1);
        cheap_matmul_dual_asm_w12(
            lane0,
            lane1,
            &sparse_first_row[rounds_p - 1],
            &v[rounds_p - 1],
        );
    }
}

impl Permutation<[Goldilocks; 8]> for Poseidon1GoldilocksFused<8> {
    fn permute_mut(&self, state: &mut [Goldilocks; 8]) {
        // Zero-cost transmute: Goldilocks is repr(transparent) over u64.
        let raw = unsafe { &mut *(state as *mut [Goldilocks; 8] as *mut [u64; 8]) };

        // Initial full rounds, then partial rounds, then terminal full rounds.
        full_rounds_scalar_w8(raw, &self.initial_constants_raw);
        partial_rounds_scalar_w8(
            raw,
            &self.first_round_constants_raw,
            &self.m_i_raw,
            &self.sparse_first_row_raw,
            &self.v_raw,
            &self.round_constants_raw,
        );
        full_rounds_scalar_w8(raw, &self.terminal_constants_raw);
    }
}

impl CryptographicPermutation<[Goldilocks; 8]> for Poseidon1GoldilocksFused<8> {}

impl Permutation<[PackedGoldilocksNeon; 8]> for Poseidon1GoldilocksFused<8> {
    fn permute_mut(&self, state: &mut [PackedGoldilocksNeon; 8]) {
        // Unpack the two lanes from the packed representation.
        let (mut lane0, mut lane1) = unpack_lanes(state);

        // Run the full permutation on both lanes simultaneously.
        full_rounds_dual_w8(&mut lane0, &mut lane1, &self.initial_constants_raw);
        partial_rounds_dual_w8(
            &mut lane0,
            &mut lane1,
            &self.first_round_constants_raw,
            &self.m_i_raw,
            &self.sparse_first_row_raw,
            &self.v_raw,
            &self.round_constants_raw,
        );
        full_rounds_dual_w8(&mut lane0, &mut lane1, &self.terminal_constants_raw);

        // Repack both lanes into the packed representation.
        pack_lanes(state, &lane0, &lane1);
    }
}

impl CryptographicPermutation<[PackedGoldilocksNeon; 8]> for Poseidon1GoldilocksFused<8> {}

impl Permutation<[Goldilocks; 12]> for Poseidon1GoldilocksFused<12> {
    fn permute_mut(&self, state: &mut [Goldilocks; 12]) {
        let raw = unsafe { &mut *(state as *mut [Goldilocks; 12] as *mut [u64; 12]) };

        full_rounds_scalar_w12(raw, &self.initial_constants_raw);
        partial_rounds_scalar_w12(
            raw,
            &self.first_round_constants_raw,
            &self.m_i_raw,
            &self.sparse_first_row_raw,
            &self.v_raw,
            &self.round_constants_raw,
        );
        full_rounds_scalar_w12(raw, &self.terminal_constants_raw);
    }
}

impl CryptographicPermutation<[Goldilocks; 12]> for Poseidon1GoldilocksFused<12> {}

impl Permutation<[PackedGoldilocksNeon; 12]> for Poseidon1GoldilocksFused<12> {
    fn permute_mut(&self, state: &mut [PackedGoldilocksNeon; 12]) {
        let (mut lane0, mut lane1) = unpack_lanes(state);

        full_rounds_dual_w12(&mut lane0, &mut lane1, &self.initial_constants_raw);
        partial_rounds_dual_w12(
            &mut lane0,
            &mut lane1,
            &self.first_round_constants_raw,
            &self.m_i_raw,
            &self.sparse_first_row_raw,
            &self.v_raw,
            &self.round_constants_raw,
        );
        full_rounds_dual_w12(&mut lane0, &mut lane1, &self.terminal_constants_raw);

        pack_lanes(state, &lane0, &lane1);
    }
}

impl CryptographicPermutation<[PackedGoldilocksNeon; 12]> for Poseidon1GoldilocksFused<12> {}

/// Dual-dispatch wrapper for Goldilocks Poseidon1.
///
/// **Scalar** permutations delegate to the generic LLVM-optimized path
/// with Karatsuba MDS (avoiding the regression from sequential inline ASM).
///
/// **Packed width-8** permutations delegate to the fused dual-lane ASM path
/// with NEON-accelerated MDS for full rounds, achieving ~15% speedup per-perm.
///
/// **Packed width-12** permutations run two sequential permutations using
/// the NEON MDS for full rounds and LLVM-optimized partial rounds, called
/// via the public free functions from `p3_poseidon`.
#[derive(Clone, Debug)]
pub struct Poseidon1GoldilocksDispatch<const WIDTH: usize> {
    /// Karatsuba MDS scalar path — used for scalar permutations.
    generic: PoseidonGoldilocksGeneric<WIDTH>,
    /// Fused dual-lane path — used for w8 packed.
    fused: Poseidon1GoldilocksFused<WIDTH>,
    /// Pre-computed full round constants for NEON MDS packed path.
    full_constants: FullRoundConstants<Goldilocks, WIDTH>,
    /// Pre-computed partial round constants for LLVM-optimized packed path.
    partial_constants: PartialRoundConstants<Goldilocks, WIDTH>,
}

impl<const WIDTH: usize> Poseidon1GoldilocksDispatch<WIDTH> {
    /// Create from generic (Karatsuba), fused, and pre-computed constants.
    pub fn new(
        generic: PoseidonGoldilocksGeneric<WIDTH>,
        fused: Poseidon1GoldilocksFused<WIDTH>,
        full_constants: FullRoundConstants<Goldilocks, WIDTH>,
        partial_constants: PartialRoundConstants<Goldilocks, WIDTH>,
    ) -> Self {
        Self {
            generic,
            fused,
            full_constants,
            partial_constants,
        }
    }
}

// --- Width 8 ---

impl Permutation<[Goldilocks; 8]> for Poseidon1GoldilocksDispatch<8> {
    fn permute_mut(&self, state: &mut [Goldilocks; 8]) {
        self.generic.permute_mut(state);
    }
}

impl CryptographicPermutation<[Goldilocks; 8]> for Poseidon1GoldilocksDispatch<8> {}

impl Permutation<[PackedGoldilocksNeon; 8]> for Poseidon1GoldilocksDispatch<8> {
    fn permute_mut(&self, state: &mut [PackedGoldilocksNeon; 8]) {
        self.fused.permute_mut(state);
    }
}

impl CryptographicPermutation<[PackedGoldilocksNeon; 8]> for Poseidon1GoldilocksDispatch<8> {}

// --- Width 12 ---

impl Permutation<[Goldilocks; 12]> for Poseidon1GoldilocksDispatch<12> {
    fn permute_mut(&self, state: &mut [Goldilocks; 12]) {
        self.generic.permute_mut(state);
    }
}

impl CryptographicPermutation<[Goldilocks; 12]> for Poseidon1GoldilocksDispatch<12> {}

impl Permutation<[PackedGoldilocksNeon; 12]> for Poseidon1GoldilocksDispatch<12> {
    fn permute_mut(&self, state: &mut [PackedGoldilocksNeon; 12]) {
        // Two sequential NEON-MDS permutations per lane, with phases
        // interleaved so that constants stay cache-hot across both lanes.
        let (mut lane0, mut lane1) = unpack_lanes(state);
        let gl0 = unsafe { &mut *(&mut lane0 as *mut [u64; 12] as *mut [Goldilocks; 12]) };
        let gl1 = unsafe { &mut *(&mut lane1 as *mut [u64; 12] as *mut [Goldilocks; 12]) };
        let mds = MdsNeonGoldilocks;

        // Initial full rounds for both lanes (initial constants stay hot).
        full_round_initial_permute_state::<_, _, _, 12, 7>(gl0, &self.full_constants, &mds);
        full_round_initial_permute_state::<_, _, _, 12, 7>(gl1, &self.full_constants, &mds);

        // Partial rounds for both lanes (partial constants stay hot).
        partial_permute_state::<_, _, 12, 7>(gl0, &self.partial_constants);
        partial_permute_state::<_, _, 12, 7>(gl1, &self.partial_constants);

        // Terminal full rounds for both lanes (terminal constants stay hot).
        full_round_terminal_permute_state::<_, _, _, 12, 7>(gl0, &self.full_constants, &mds);
        full_round_terminal_permute_state::<_, _, _, 12, 7>(gl1, &self.full_constants, &mds);

        pack_lanes(state, &lane0, &lane1);
    }
}

impl CryptographicPermutation<[PackedGoldilocksNeon; 12]> for Poseidon1GoldilocksDispatch<12> {}

#[cfg(test)]
mod tests {
    use p3_field::{PrimeCharacteristicRing, PrimeField64};
    use p3_poseidon::PoseidonConstants;
    use p3_symmetric::Permutation;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;
    use crate::mds::{MATRIX_CIRC_MDS_8_COL, MATRIX_CIRC_MDS_12_COL};
    use crate::poseidon::{
        GOLDILOCKS_POSEIDON_RC_8, GOLDILOCKS_POSEIDON_RC_12, default_goldilocks_poseidon_8,
        default_goldilocks_poseidon_12,
    };

    type F = Goldilocks;

    /// Build a width-8 fused permutation from the fixed round constants.
    fn make_fused_w8() -> Poseidon1GoldilocksFused<8> {
        let raw = PoseidonConstants {
            rounds_f: 8,
            rounds_p: 22,
            mds_circ_col: MATRIX_CIRC_MDS_8_COL,
            round_constants: GOLDILOCKS_POSEIDON_RC_8.to_vec(),
        };
        let (full, partial) = raw.to_optimized();
        Poseidon1GoldilocksFused::new(full, partial)
    }

    /// Build a width-12 fused permutation from the fixed round constants.
    fn make_fused_w12() -> Poseidon1GoldilocksFused<12> {
        let raw = PoseidonConstants {
            rounds_f: 8,
            rounds_p: 22,
            mds_circ_col: MATRIX_CIRC_MDS_12_COL,
            round_constants: GOLDILOCKS_POSEIDON_RC_12.to_vec(),
        };
        let (full, partial) = raw.to_optimized();
        Poseidon1GoldilocksFused::new(full, partial)
    }

    /// Verify that the fused width-8 implementation matches the generic one
    /// on both zero and random inputs.
    #[test]
    fn test_fused_matches_generic_w8() {
        let generic = default_goldilocks_poseidon_8();
        let fused = make_fused_w8();
        let mut rng = SmallRng::seed_from_u64(42);

        // Zero input.
        let mut g_state = [F::ZERO; 8];
        let mut f_state = [F::ZERO; 8];
        generic.permute_mut(&mut g_state);
        fused.permute_mut(&mut f_state);
        for i in 0..8 {
            assert_eq!(
                f_state[i].as_canonical_u64(),
                g_state[i].as_canonical_u64(),
                "Fused vs generic mismatch at index {i} (zero input, w8)"
            );
        }

        // Random input.
        let mut g_state: [F; 8] = rng.random();
        let mut f_state = g_state;
        generic.permute_mut(&mut g_state);
        fused.permute_mut(&mut f_state);
        for i in 0..8 {
            assert_eq!(
                f_state[i].as_canonical_u64(),
                g_state[i].as_canonical_u64(),
                "Fused vs generic mismatch at index {i} (random input, w8)"
            );
        }
    }

    /// Same fused-vs-generic verification for width 12.
    #[test]
    fn test_fused_matches_generic_w12() {
        let generic = default_goldilocks_poseidon_12();
        let fused = make_fused_w12();
        let mut rng = SmallRng::seed_from_u64(42);

        let mut g_state = [F::ZERO; 12];
        let mut f_state = [F::ZERO; 12];
        generic.permute_mut(&mut g_state);
        fused.permute_mut(&mut f_state);
        for i in 0..12 {
            assert_eq!(
                f_state[i].as_canonical_u64(),
                g_state[i].as_canonical_u64(),
                "Fused vs generic mismatch at index {i} (zero input, w12)"
            );
        }

        let mut g_state: [F; 12] = rng.random();
        let mut f_state = g_state;
        generic.permute_mut(&mut g_state);
        fused.permute_mut(&mut f_state);
        for i in 0..12 {
            assert_eq!(
                f_state[i].as_canonical_u64(),
                g_state[i].as_canonical_u64(),
                "Fused vs generic mismatch at index {i} (random input, w12)"
            );
        }
    }

    /// Verify that the packed (dual-lane) width-8 path matches running
    /// two independent scalar permutations.
    #[test]
    fn test_packed_matches_scalar_w8() {
        let fused = make_fused_w8();
        let mut rng = SmallRng::seed_from_u64(123);

        // Two independent random scalar inputs.
        let scalar_a: [F; 8] = rng.random();
        let scalar_b: [F; 8] = rng.random();

        // Pack them into a single packed state and permute.
        let mut packed: [PackedGoldilocksNeon; 8] =
            core::array::from_fn(|i| PackedGoldilocksNeon([scalar_a[i], scalar_b[i]]));
        fused.permute_mut(&mut packed);

        // Compute the expected result by running scalar on each independently.
        let mut expected_a = scalar_a;
        let mut expected_b = scalar_b;
        fused.permute_mut(&mut expected_a);
        fused.permute_mut(&mut expected_b);

        // Lane 0 must match the first scalar, lane 1 must match the second.
        for i in 0..8 {
            assert_eq!(
                packed[i].0[0].as_canonical_u64(),
                expected_a[i].as_canonical_u64(),
                "Packed lane0 mismatch at index {i} (w8)"
            );
            assert_eq!(
                packed[i].0[1].as_canonical_u64(),
                expected_b[i].as_canonical_u64(),
                "Packed lane1 mismatch at index {i} (w8)"
            );
        }
    }

    /// Same packed-vs-scalar verification for width 12.
    #[test]
    fn test_packed_matches_scalar_w12() {
        let fused = make_fused_w12();
        let mut rng = SmallRng::seed_from_u64(123);

        let scalar_a: [F; 12] = rng.random();
        let scalar_b: [F; 12] = rng.random();

        let mut packed: [PackedGoldilocksNeon; 12] =
            core::array::from_fn(|i| PackedGoldilocksNeon([scalar_a[i], scalar_b[i]]));
        fused.permute_mut(&mut packed);

        let mut expected_a = scalar_a;
        let mut expected_b = scalar_b;
        fused.permute_mut(&mut expected_a);
        fused.permute_mut(&mut expected_b);

        for i in 0..12 {
            assert_eq!(
                packed[i].0[0].as_canonical_u64(),
                expected_a[i].as_canonical_u64(),
                "Packed lane0 mismatch at index {i} (w12)"
            );
            assert_eq!(
                packed[i].0[1].as_canonical_u64(),
                expected_b[i].as_canonical_u64(),
                "Packed lane1 mismatch at index {i} (w12)"
            );
        }
    }

    /// Known-answer test for width 8 (sequential 0..7 input).
    #[test]
    fn test_fused_kat_w8() {
        let fused = make_fused_w8();
        let mut input: [F; 8] = F::new_array([0, 1, 2, 3, 4, 5, 6, 7]);
        fused.permute_mut(&mut input);

        let expected: [F; 8] = F::new_array([
            9402631278636174262,
            17004635335047130969,
            4753012512511375168,
            2367239025173374637,
            16921623581418299817,
            2583582247381514966,
            12095377248421862739,
            5553966757107483887,
        ]);
        assert_eq!(input, expected);
    }

    /// Known-answer test for width 12 (sequential 0..11 input).
    #[test]
    fn test_fused_kat_w12() {
        let fused = make_fused_w12();
        let mut input: [F; 12] = F::new_array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
        fused.permute_mut(&mut input);

        let expected: [F; 12] = F::new_array([
            71109932875369129,
            6358490863509936162,
            15845417984153754708,
            6622548371661487295,
            16665750330794722584,
            3390336522757414137,
            9832650793018174136,
            5390692944999521363,
            15168680663824027226,
            4054910678692513992,
            14678252141200722212,
            3716442817880027191,
        ]);
        assert_eq!(input, expected);
    }
}
