//! Implementation of Poseidon2, see: https://eprint.iacr.org/2023/323

use alloc::vec::Vec;

use p3_field::{Algebra, InjectiveMonomial, PrimeCharacteristicRing};
use p3_poseidon2::{
    ExternalLayer, ExternalLayerConstants, ExternalLayerConstructor, GenericPoseidon2LinearLayers,
    InternalLayer, InternalLayerConstructor, MDSMat4, Poseidon2, add_rc_and_sbox_generic,
    external_initial_permute_state, external_terminal_permute_state, internal_permute_state,
    matmul_internal,
};

use crate::Goldilocks;

/// Degree of the chosen permutation polynomial for Goldilocks, used as the Poseidon2 S-Box.
///
/// As p - 1 = 2^32 * 3 * 5 * 17 * ... the smallest choice for a degree D satisfying gcd(p - 1, D) = 1 is 7.
const GOLDILOCKS_S_BOX_DEGREE: u64 = 7;

/// An implementation of the Poseidon2 hash function for the Goldilocks field.
///
/// It acts on arrays of the form `[Goldilocks; WIDTH]`.
#[cfg(target_arch = "aarch64")]
pub type Poseidon2Goldilocks<const WIDTH: usize> = crate::Poseidon2GoldilocksFused<WIDTH>;

/// An implementation of the Poseidon2 hash function for the Goldilocks field.
///
/// It acts on arrays of the form `[Goldilocks; WIDTH]`.
#[cfg(not(target_arch = "aarch64"))]
pub type Poseidon2Goldilocks<const WIDTH: usize> = Poseidon2<
    Goldilocks,
    Poseidon2ExternalLayerGoldilocks<WIDTH>,
    Poseidon2InternalLayerGoldilocks,
    WIDTH,
    GOLDILOCKS_S_BOX_DEGREE,
>;

pub const MATRIX_DIAG_8_GOLDILOCKS: [Goldilocks; 8] = Goldilocks::new_array([
    0xfffffffeffffffff, // -2
    0x0000000000000001, // 1
    0x0000000000000002, // 2
    0x7fffffff80000001, // 1/2
    0x0000000000000003, // 3
    0x7fffffff80000000, // -1/2
    0xfffffffefffffffe, // -3
    0xfffffffefffffffd, // -4
]);

pub const MATRIX_DIAG_12_GOLDILOCKS: [Goldilocks; 12] = Goldilocks::new_array([
    0xfffffffeffffffff, // -2
    0x0000000000000001, // 1
    0x0000000000000002, // 2
    0x7fffffff80000001, // 1/2
    0x0000000000000003, // 3
    0x0000000000000004, // 4
    0x7fffffff80000000, // -1/2
    0xfffffffefffffffe, // -3
    0xfffffffefffffffd, // -4
    0xbfffffff40000001, // 1/2^2
    0x3fffffffc0000000, // -1/2^2
    0xdfffffff20000001, // 1/2^3
]);

pub const MATRIX_DIAG_16_GOLDILOCKS: [Goldilocks; 16] = Goldilocks::new_array([
    0xfffffffeffffffff, // -2
    0x0000000000000001, // 1
    0x0000000000000002, // 2
    0x7fffffff80000001, // 1/2
    0x0000000000000003, // 3
    0x0000000000000004, // 4
    0x7fffffff80000000, // -1/2
    0xfffffffefffffffe, // -3
    0xfffffffefffffffd, // -4
    0xdfffffff20000001, // 1/2^3
    0xefffffff10000001, // 1/2^4
    0xf7ffffff08000001, // 1/2^5
    0x1fffffffe0000000, // -1/2^3
    0x0ffffffff0000000, // -1/2^4
    0x07fffffff8000000, // -1/2^5
    0xfffffffe00000002, // 1/2^32
]);

pub const MATRIX_DIAG_20_GOLDILOCKS: [Goldilocks; 20] = Goldilocks::new_array([
    0x95c381fda3b1fa57,
    0xf36fe9eb1288f42c,
    0x89f5dcdfef277944,
    0x106f22eadeb3e2d2,
    0x684e31a2530e5111,
    0x27435c5d89fd148e,
    0x3ebed31c414dbf17,
    0xfd45b0b2d294e3cc,
    0x48c904473a7f6dbf,
    0xe0d1b67809295b4d,
    0xddd1941e9d199dcb,
    0x8cfe534eeb742219,
    0xa6e5261d9e3b8524,
    0x6897ee5ed0f82c1b,
    0x0e7dcd0739ee5f78,
    0x493253f3d0d32363,
    0xbb2737f5845f05c0,
    0xa187e810b06ad903,
    0xb635b995936c4918,
    0x0b3694a940bd2394,
]);

fn internal_layer_mat_mul_goldilocks_8<A: Algebra<Goldilocks>>(state: &mut [A; 8]) {
    let sum: A = state.iter().cloned().sum();

    let s0 = state[0].clone();
    let s1 = state[1].clone();
    let s2 = state[2].clone();
    let s3 = state[3].clone();
    let s4 = state[4].clone();
    let s5 = state[5].clone();
    let s6 = state[6].clone();
    let s7 = state[7].clone();

    // V[0] = -2
    let two_s0 = s0.clone() + s0;
    state[0] = sum.clone() - two_s0;

    // V[1] = 1
    state[1] = sum.clone() + s1;

    // V[2] = 2
    let two_s2 = s2.clone() + s2;
    state[2] = sum.clone() + two_s2;

    // V[3] = 1/2
    state[3] = sum.clone() + s3.halve();

    // V[4] = 3
    let two_s4 = s4.clone() + s4.clone();
    let three_s4 = two_s4 + s4;
    state[4] = sum.clone() + three_s4;

    // V[5] = -1/2
    state[5] = sum.clone() - s5.halve();

    // V[6] = -3
    let two_s6 = s6.clone() + s6.clone();
    let three_s6 = two_s6 + s6;
    state[6] = sum.clone() - three_s6;

    // V[7] = -4
    let two_s7 = s7.clone() + s7;
    let four_s7 = two_s7.clone() + two_s7;
    state[7] = sum - four_s7;
}

fn internal_layer_mat_mul_goldilocks_12<A: Algebra<Goldilocks>>(state: &mut [A; 12]) {
    let sum: A = state.iter().cloned().sum();

    let s0 = state[0].clone();
    let s1 = state[1].clone();
    let s2 = state[2].clone();
    let s3 = state[3].clone();
    let s4 = state[4].clone();
    let s5 = state[5].clone();
    let s6 = state[6].clone();
    let s7 = state[7].clone();
    let s8 = state[8].clone();
    let s9 = state[9].clone();
    let s10 = state[10].clone();
    let s11 = state[11].clone();

    // V[0] = -2
    let two_s0 = s0.clone() + s0;
    state[0] = sum.clone() - two_s0;

    // V[1] = 1
    state[1] = sum.clone() + s1;

    // V[2] = 2
    let two_s2 = s2.clone() + s2;
    state[2] = sum.clone() + two_s2;

    // V[3] = 1/2
    state[3] = sum.clone() + s3.halve();

    // V[4] = 3
    let two_s4 = s4.clone() + s4.clone();
    let three_s4 = two_s4 + s4;
    state[4] = sum.clone() + three_s4;

    // V[5] = 4
    let two_s5 = s5.clone() + s5;
    let four_s5 = two_s5.clone() + two_s5;
    state[5] = sum.clone() + four_s5;

    // V[6] = -1/2
    state[6] = sum.clone() - s6.halve();

    // V[7] = -3
    let two_s7 = s7.clone() + s7.clone();
    let three_s7 = two_s7 + s7;
    state[7] = sum.clone() - three_s7;

    // V[8] = -4
    let two_s8 = s8.clone() + s8;
    let four_s8 = two_s8.clone() + two_s8;
    state[8] = sum.clone() - four_s8;

    // V[9] = 1/2^2
    state[9] = sum.clone() + s9.halve().halve();

    // V[10] = -1/2^2
    state[10] = sum.clone() - s10.halve().halve();

    // V[11] = 1/2^3
    state[11] = sum + s11.halve().halve().halve();
}

fn internal_layer_mat_mul_goldilocks_16<A: Algebra<Goldilocks>>(state: &mut [A; 16]) {
    let sum: A = state.iter().cloned().sum();

    let s0 = state[0].clone();
    let s1 = state[1].clone();
    let s2 = state[2].clone();
    let s3 = state[3].clone();
    let s4 = state[4].clone();
    let s5 = state[5].clone();
    let s6 = state[6].clone();
    let s7 = state[7].clone();
    let s8 = state[8].clone();
    let s9 = state[9].clone();
    let s10 = state[10].clone();
    let s11 = state[11].clone();
    let s12 = state[12].clone();
    let s13 = state[13].clone();
    let s14 = state[14].clone();
    let s15 = state[15].clone();

    // V[0] = -2
    let two_s0 = s0.clone() + s0;
    state[0] = sum.clone() - two_s0;

    // V[1] = 1
    state[1] = sum.clone() + s1;

    // V[2] = 2
    let two_s2 = s2.clone() + s2;
    state[2] = sum.clone() + two_s2;

    // V[3] = 1/2
    state[3] = sum.clone() + s3.halve();

    // V[4] = 3
    let two_s4 = s4.clone() + s4.clone();
    let three_s4 = two_s4 + s4;
    state[4] = sum.clone() + three_s4;

    // V[5] = 4
    let two_s5 = s5.clone() + s5;
    let four_s5 = two_s5.clone() + two_s5;
    state[5] = sum.clone() + four_s5;

    // V[6] = -1/2
    state[6] = sum.clone() - s6.halve();

    // V[7] = -3
    let two_s7 = s7.clone() + s7.clone();
    let three_s7 = two_s7 + s7;
    state[7] = sum.clone() - three_s7;

    // V[8] = -4
    let two_s8 = s8.clone() + s8;
    let four_s8 = two_s8.clone() + two_s8;
    state[8] = sum.clone() - four_s8;

    // V[9] = 1/2^3
    state[9] = sum.clone() + s9.halve().halve().halve();

    // V[10] = 1/2^4
    state[10] = sum.clone() + s10.halve().halve().halve().halve();

    // V[11] = 1/2^5
    state[11] = sum.clone() + s11.halve().halve().halve().halve().halve();

    // V[12] = -1/2^3
    state[12] = sum.clone() - s12.halve().halve().halve();

    // V[13] = -1/2^4
    state[13] = sum.clone() - s13.halve().halve().halve().halve();

    // V[14] = -1/2^5
    state[14] = sum.clone() - s14.halve().halve().halve().halve().halve();

    // V[15] = 1/2^32
    let inv_2_32 = MATRIX_DIAG_16_GOLDILOCKS[15];
    let v15 = s15 * inv_2_32;
    state[15] = sum + v15;
}

/// The internal layers of the Poseidon2 permutation.
#[derive(Debug, Clone, Default)]
pub struct Poseidon2InternalLayerGoldilocks {
    internal_constants: Vec<Goldilocks>,
}

impl InternalLayerConstructor<Goldilocks> for Poseidon2InternalLayerGoldilocks {
    fn new_from_constants(internal_constants: Vec<Goldilocks>) -> Self {
        Self { internal_constants }
    }
}

impl<A: Algebra<Goldilocks> + InjectiveMonomial<GOLDILOCKS_S_BOX_DEGREE>>
    InternalLayer<A, 8, GOLDILOCKS_S_BOX_DEGREE> for Poseidon2InternalLayerGoldilocks
{
    /// Perform the internal layers of the Poseidon2 permutation on the given state.
    fn permute_state(&self, state: &mut [A; 8]) {
        internal_permute_state(
            state,
            internal_layer_mat_mul_goldilocks_8,
            &self.internal_constants,
        );
    }
}

impl<A: Algebra<Goldilocks> + InjectiveMonomial<GOLDILOCKS_S_BOX_DEGREE>>
    InternalLayer<A, 12, GOLDILOCKS_S_BOX_DEGREE> for Poseidon2InternalLayerGoldilocks
{
    /// Perform the internal layers of the Poseidon2 permutation on the given state.
    fn permute_state(&self, state: &mut [A; 12]) {
        internal_permute_state(
            state,
            internal_layer_mat_mul_goldilocks_12,
            &self.internal_constants,
        );
    }
}

impl<A: Algebra<Goldilocks> + InjectiveMonomial<GOLDILOCKS_S_BOX_DEGREE>>
    InternalLayer<A, 16, GOLDILOCKS_S_BOX_DEGREE> for Poseidon2InternalLayerGoldilocks
{
    /// Perform the internal layers of the Poseidon2 permutation on the given state.
    fn permute_state(&self, state: &mut [A; 16]) {
        internal_permute_state(
            state,
            internal_layer_mat_mul_goldilocks_16,
            &self.internal_constants,
        );
    }
}

impl<A: Algebra<Goldilocks> + InjectiveMonomial<GOLDILOCKS_S_BOX_DEGREE>>
    InternalLayer<A, 20, GOLDILOCKS_S_BOX_DEGREE> for Poseidon2InternalLayerGoldilocks
{
    /// Perform the internal layers of the Poseidon2 permutation on the given state.
    fn permute_state(&self, state: &mut [A; 20]) {
        internal_permute_state(
            state,
            |x| matmul_internal(x, MATRIX_DIAG_20_GOLDILOCKS),
            &self.internal_constants,
        );
    }
}

/// The external layers of the Poseidon2 permutation.
#[derive(Clone)]
pub struct Poseidon2ExternalLayerGoldilocks<const WIDTH: usize> {
    pub(crate) external_constants: ExternalLayerConstants<Goldilocks, WIDTH>,
}

impl<const WIDTH: usize> ExternalLayerConstructor<Goldilocks, WIDTH>
    for Poseidon2ExternalLayerGoldilocks<WIDTH>
{
    fn new_from_constants(external_constants: ExternalLayerConstants<Goldilocks, WIDTH>) -> Self {
        Self { external_constants }
    }
}

impl<A: Algebra<Goldilocks> + InjectiveMonomial<GOLDILOCKS_S_BOX_DEGREE>, const WIDTH: usize>
    ExternalLayer<A, WIDTH, GOLDILOCKS_S_BOX_DEGREE> for Poseidon2ExternalLayerGoldilocks<WIDTH>
{
    /// Perform the initial external layers of the Poseidon2 permutation on the given state.
    fn permute_state_initial(&self, state: &mut [A; WIDTH]) {
        external_initial_permute_state(
            state,
            self.external_constants.get_initial_constants(),
            add_rc_and_sbox_generic,
            &MDSMat4,
        );
    }

    /// Perform the terminal external layers of the Poseidon2 permutation on the given state.
    fn permute_state_terminal(&self, state: &mut [A; WIDTH]) {
        external_terminal_permute_state(
            state,
            self.external_constants.get_terminal_constants(),
            add_rc_and_sbox_generic,
            &MDSMat4,
        );
    }
}

/// An implementation of the matrix multiplications in the internal and external layers of Poseidon2.
///
/// This can act on `[A; WIDTH]` for any ring implementing `Algebra<Goldilocks>`.
/// If you have either `[Goldilocks::Packing; WIDTH]` or `[Goldilocks; WIDTH]` it will be much faster
/// to use `Poseidon2Goldilocks<WIDTH>` instead of building a Poseidon2 permutation using this.
#[derive(Clone, Debug, Default)]
pub struct GenericPoseidon2LinearLayersGoldilocks;

impl GenericPoseidon2LinearLayers<8> for GenericPoseidon2LinearLayersGoldilocks {
    fn internal_linear_layer<R: PrimeCharacteristicRing>(state: &mut [R; 8]) {
        let sum: R = state.iter().cloned().sum();
        for i in 0..8 {
            let d = R::from_u64(MATRIX_DIAG_8_GOLDILOCKS[i].value);
            state[i] *= d;
            state[i] += sum.clone();
        }
    }
}

impl GenericPoseidon2LinearLayers<12> for GenericPoseidon2LinearLayersGoldilocks {
    fn internal_linear_layer<R: PrimeCharacteristicRing>(state: &mut [R; 12]) {
        let sum: R = state.iter().cloned().sum();
        for i in 0..12 {
            let d = R::from_u64(MATRIX_DIAG_12_GOLDILOCKS[i].value);
            state[i] *= d;
            state[i] += sum.clone();
        }
    }
}

impl GenericPoseidon2LinearLayers<16> for GenericPoseidon2LinearLayersGoldilocks {
    fn internal_linear_layer<R: PrimeCharacteristicRing>(state: &mut [R; 16]) {
        let sum: R = state.iter().cloned().sum();
        for i in 0..16 {
            let d = R::from_u64(MATRIX_DIAG_16_GOLDILOCKS[i].value);
            state[i] *= d;
            state[i] += sum.clone();
        }
    }
}

impl GenericPoseidon2LinearLayers<20> for GenericPoseidon2LinearLayersGoldilocks {
    fn internal_linear_layer<R: PrimeCharacteristicRing>(state: &mut [R; 20]) {
        let sum: R = state.iter().cloned().sum();
        for i in 0..20 {
            let d = R::from_u64(MATRIX_DIAG_20_GOLDILOCKS[i].value);
            state[i] *= d;
            state[i] += sum.clone();
        }
    }
}

#[cfg(test)]
mod tests {
    use p3_field::PrimeCharacteristicRing;

    use super::*;

    type F = Goldilocks;

    #[test]
    fn test_generic_internal_linear_layer_8_matches_matmul_internal() {
        let mut state_generic = [
            F::from_u64(1),
            F::from_u64(2),
            F::from_u64(3),
            F::from_u64(4),
            F::from_u64(5),
            F::from_u64(6),
            F::from_u64(7),
            F::from_u64(8),
        ];
        let mut state_existing = state_generic;

        GenericPoseidon2LinearLayersGoldilocks::internal_linear_layer(&mut state_generic);
        matmul_internal(&mut state_existing, MATRIX_DIAG_8_GOLDILOCKS);

        assert_eq!(state_generic, state_existing);
    }

    #[test]
    fn test_generic_internal_linear_layer_12_matches_matmul_internal() {
        let mut state_generic = [
            F::from_u64(1),
            F::from_u64(2),
            F::from_u64(3),
            F::from_u64(4),
            F::from_u64(5),
            F::from_u64(6),
            F::from_u64(7),
            F::from_u64(8),
            F::from_u64(9),
            F::from_u64(10),
            F::from_u64(11),
            F::from_u64(12),
        ];
        let mut state_existing = state_generic;

        GenericPoseidon2LinearLayersGoldilocks::internal_linear_layer(&mut state_generic);
        matmul_internal(&mut state_existing, MATRIX_DIAG_12_GOLDILOCKS);

        assert_eq!(state_generic, state_existing);
    }

    #[test]
    fn test_generic_internal_linear_layer_16_matches_matmul_internal() {
        let mut state_generic = [
            F::from_u64(1),
            F::from_u64(2),
            F::from_u64(3),
            F::from_u64(4),
            F::from_u64(5),
            F::from_u64(6),
            F::from_u64(7),
            F::from_u64(8),
            F::from_u64(9),
            F::from_u64(10),
            F::from_u64(11),
            F::from_u64(12),
            F::from_u64(13),
            F::from_u64(14),
            F::from_u64(15),
            F::from_u64(16),
        ];
        let mut state_existing = state_generic;

        GenericPoseidon2LinearLayersGoldilocks::internal_linear_layer(&mut state_generic);
        matmul_internal(&mut state_existing, MATRIX_DIAG_16_GOLDILOCKS);

        assert_eq!(state_generic, state_existing);
    }

    #[test]
    fn test_generic_internal_linear_layer_20_matches_matmul_internal() {
        let mut state_generic = [
            F::from_u64(1),
            F::from_u64(2),
            F::from_u64(3),
            F::from_u64(4),
            F::from_u64(5),
            F::from_u64(6),
            F::from_u64(7),
            F::from_u64(8),
            F::from_u64(9),
            F::from_u64(10),
            F::from_u64(11),
            F::from_u64(12),
            F::from_u64(13),
            F::from_u64(14),
            F::from_u64(15),
            F::from_u64(16),
            F::from_u64(17),
            F::from_u64(18),
            F::from_u64(19),
            F::from_u64(20),
        ];
        let mut state_existing = state_generic;

        GenericPoseidon2LinearLayersGoldilocks::internal_linear_layer(&mut state_generic);
        matmul_internal(&mut state_existing, MATRIX_DIAG_20_GOLDILOCKS);

        assert_eq!(state_generic, state_existing);
    }
}
