//! Implementation of Poseidon2, see: https://eprint.iacr.org/2023/323

//! For now we recreate the implementation given in:
//! https://github.com/HorizenLabs/poseidon2/blob/main/plain_implementations/src/poseidon2/poseidon2_instance_goldilocks.rs
//! This uses the constants below along with using the 4x4 matrix:
//! [[5, 7, 1, 3], [4, 6, 1, 1], [1, 3, 5, 7], [1, 1, 4, 6]]
//! to build the 4t x 4t matrix used for the external (full) rounds).

//! Long term we will use more optimised internal and external linear layers.
use alloc::vec::Vec;

use p3_field::{Algebra, InjectiveMonomial, PrimeCharacteristicRing};
use p3_poseidon2::{
    ExternalLayer, ExternalLayerConstants, ExternalLayerConstructor, GenericPoseidon2LinearLayers,
    HLMDSMat4, InternalLayer, InternalLayerConstructor, MDSMat4, Poseidon2,
    add_rc_and_sbox_generic, external_initial_permute_state, external_terminal_permute_state,
    internal_permute_state, matmul_internal,
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

/// A recreating of the Poseidon2 implementation by Horizen Labs for the Goldilocks field.
///
/// It acts on arrays of the form `[Goldilocks; WIDTH]`
/// The original implementation can be found here: https://github.com/HorizenLabs/poseidon2.
/// This implementation is slightly slower than `Poseidon2Goldilocks` as is uses a slower matrix
/// for the external rounds.
pub type Poseidon2GoldilocksHL<const WIDTH: usize> = Poseidon2<
    Goldilocks,
    Poseidon2ExternalLayerGoldilocksHL<WIDTH>,
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
    let mut two_s0 = s0.clone();
    two_s0 += s0;
    state[0] = sum.clone() - two_s0;

    // V[1] = 1
    state[1] = sum.clone() + s1;

    // V[2] = 2
    let mut two_s2 = s2.clone();
    two_s2 += s2;
    state[2] = sum.clone() + two_s2;

    // V[3] = 1/2
    let half = MATRIX_DIAG_8_GOLDILOCKS[3];
    let mut half_s3 = s3;
    half_s3 *= half;
    state[3] = sum.clone() + half_s3;

    // V[4] = 3
    let mut two_s4 = s4.clone();
    two_s4 += s4.clone();
    let three_s4 = two_s4 + s4;
    state[4] = sum.clone() + three_s4;

    // V[5] = -1/2
    let mut half_s5 = s5;
    half_s5 *= half;
    state[5] = sum.clone() - half_s5;

    // V[6] = -3
    let mut two_s6 = s6.clone();
    two_s6 += s6.clone();
    let three_s6 = two_s6 + s6;
    state[6] = sum.clone() - three_s6;

    // V[7] = -4
    let mut two_s7 = s7.clone();
    two_s7 += s7;
    let mut four_s7 = two_s7.clone();
    four_s7 += two_s7;
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
    let mut two_s0 = s0.clone();
    two_s0 += s0;
    state[0] = sum.clone() - two_s0;

    // V[1] = 1
    state[1] = sum.clone() + s1;

    // V[2] = 2
    let mut two_s2 = s2.clone();
    two_s2 += s2;
    state[2] = sum.clone() + two_s2;

    // V[3] = 1/2
    let half = MATRIX_DIAG_12_GOLDILOCKS[3];
    let mut half_s3 = s3;
    half_s3 *= half;
    state[3] = sum.clone() + half_s3;

    // V[4] = 3
    let mut two_s4 = s4.clone();
    two_s4 += s4.clone();
    let three_s4 = two_s4 + s4;
    state[4] = sum.clone() + three_s4;

    // V[5] = 4
    let mut two_s5 = s5.clone();
    two_s5 += s5;
    let mut four_s5 = two_s5.clone();
    four_s5 += two_s5;
    state[5] = sum.clone() + four_s5;

    // V[6] = -1/2
    let mut half_s6 = s6;
    half_s6 *= half;
    state[6] = sum.clone() - half_s6;

    // V[7] = -3
    let mut two_s7 = s7.clone();
    two_s7 += s7.clone();
    let three_s7 = two_s7 + s7;
    state[7] = sum.clone() - three_s7;

    // V[8] = -4
    let mut two_s8 = s8.clone();
    two_s8 += s8;
    let mut four_s8 = two_s8.clone();
    four_s8 += two_s8;
    state[8] = sum.clone() - four_s8;

    // V[9] = 1/2^2
    let inv_4 = MATRIX_DIAG_12_GOLDILOCKS[9]; // 1/2^2
    let mut inv4_s9 = s9;
    inv4_s9 *= inv_4;
    state[9] = sum.clone() + inv4_s9;

    // V[10] = -1/2^2
    let mut inv4_s10 = s10;
    inv4_s10 *= inv_4;
    state[10] = sum.clone() - inv4_s10;

    // V[11] = 1/2^3
    let inv_8 = MATRIX_DIAG_12_GOLDILOCKS[11];
    let mut inv8_s11 = s11;
    inv8_s11 *= inv_8;
    state[11] = sum + inv8_s11;
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
    let mut two_s0 = s0.clone();
    two_s0 += s0;
    state[0] = sum.clone() - two_s0;

    // V[1] = 1
    state[1] = sum.clone() + s1;

    // V[2] = 2
    let mut two_s2 = s2.clone();
    two_s2 += s2;
    state[2] = sum.clone() + two_s2;

    // V[3] = 1/2
    let half = MATRIX_DIAG_16_GOLDILOCKS[3];
    let mut half_s3 = s3;
    half_s3 *= half;
    state[3] = sum.clone() + half_s3;

    // V[4] = 3
    let mut two_s4 = s4.clone();
    two_s4 += s4.clone();
    let three_s4 = two_s4 + s4;
    state[4] = sum.clone() + three_s4;

    // V[5] = 4
    let mut two_s5 = s5.clone();
    two_s5 += s5;
    let mut four_s5 = two_s5.clone();
    four_s5 += two_s5;
    state[5] = sum.clone() + four_s5;

    // V[6] = -1/2
    let mut half_s6 = s6;
    half_s6 *= half;
    state[6] = sum.clone() - half_s6;

    // V[7] = -3
    let mut two_s7 = s7.clone();
    two_s7 += s7.clone();
    let three_s7 = two_s7 + s7;
    state[7] = sum.clone() - three_s7;

    // V[8] = -4
    let mut two_s8 = s8.clone();
    two_s8 += s8;
    let mut four_s8 = two_s8.clone();
    four_s8 += two_s8;
    state[8] = sum.clone() - four_s8;

    // Inverse power-of-two coefficients use precomputed diagonal entries.
    let inv_8 = MATRIX_DIAG_16_GOLDILOCKS[9]; // 1/2^3
    let inv_16 = MATRIX_DIAG_16_GOLDILOCKS[10]; // 1/2^4
    let inv_32 = MATRIX_DIAG_16_GOLDILOCKS[11]; // 1/2^5
    let neg_inv_8 = MATRIX_DIAG_16_GOLDILOCKS[12]; // -1/2^3
    let neg_inv_16 = MATRIX_DIAG_16_GOLDILOCKS[13]; // -1/2^4
    let neg_inv_32 = MATRIX_DIAG_16_GOLDILOCKS[14]; // -1/2^5
    let inv_2_32 = MATRIX_DIAG_16_GOLDILOCKS[15]; // 1/2^32

    // V[9] = 1/2^3
    let mut v9 = s9;
    v9 *= inv_8;
    state[9] = sum.clone() + v9;

    // V[10] = 1/2^4
    let mut v10 = s10;
    v10 *= inv_16;
    state[10] = sum.clone() + v10;

    // V[11] = 1/2^5
    let mut v11 = s11;
    v11 *= inv_32;
    state[11] = sum.clone() + v11;

    // V[12] = -1/2^3
    let mut v12 = s12;
    v12 *= neg_inv_8;
    state[12] = sum.clone() + v12;

    // V[13] = -1/2^4
    let mut v13 = s13;
    v13 *= neg_inv_16;
    state[13] = sum.clone() + v13;

    // V[14] = -1/2^5
    let mut v14 = s14;
    v14 *= neg_inv_32;
    state[14] = sum.clone() + v14;

    // V[15] = 1/2^32
    let mut v15 = s15;
    v15 *= inv_2_32;
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

/// The external layers of the Poseidon2 permutation used by Horizen Labs.
#[derive(Clone)]
pub struct Poseidon2ExternalLayerGoldilocksHL<const WIDTH: usize> {
    pub(crate) external_constants: ExternalLayerConstants<Goldilocks, WIDTH>,
}

impl<const WIDTH: usize> ExternalLayerConstructor<Goldilocks, WIDTH>
    for Poseidon2ExternalLayerGoldilocksHL<WIDTH>
{
    fn new_from_constants(external_constants: ExternalLayerConstants<Goldilocks, WIDTH>) -> Self {
        Self { external_constants }
    }
}

impl<A: Algebra<Goldilocks> + InjectiveMonomial<GOLDILOCKS_S_BOX_DEGREE>, const WIDTH: usize>
    ExternalLayer<A, WIDTH, GOLDILOCKS_S_BOX_DEGREE> for Poseidon2ExternalLayerGoldilocksHL<WIDTH>
{
    /// Perform the initial external layers of the Poseidon2 permutation on the given state.
    fn permute_state_initial(&self, state: &mut [A; WIDTH]) {
        external_initial_permute_state(
            state,
            self.external_constants.get_initial_constants(),
            add_rc_and_sbox_generic,
            &HLMDSMat4,
        );
    }

    /// Perform the terminal external layers of the Poseidon2 permutation on the given state.
    fn permute_state_terminal(&self, state: &mut [A; WIDTH]) {
        external_terminal_permute_state(
            state,
            self.external_constants.get_terminal_constants(),
            add_rc_and_sbox_generic,
            &HLMDSMat4,
        );
    }
}

pub const HL_GOLDILOCKS_8_EXTERNAL_ROUND_CONSTANTS: [[[u64; 8]; 4]; 2] = [
    [
        [
            0xdd5743e7f2a5a5d9,
            0xcb3a864e58ada44b,
            0xffa2449ed32f8cdc,
            0x42025f65d6bd13ee,
            0x7889175e25506323,
            0x34b98bb03d24b737,
            0xbdcc535ecc4faa2a,
            0x5b20ad869fc0d033,
        ],
        [
            0xf1dda5b9259dfcb4,
            0x27515210be112d59,
            0x4227d1718c766c3f,
            0x26d333161a5bd794,
            0x49b938957bf4b026,
            0x4a56b5938b213669,
            0x1120426b48c8353d,
            0x6b323c3f10a56cad,
        ],
        [
            0xce57d6245ddca6b2,
            0xb1fc8d402bba1eb1,
            0xb5c5096ca959bd04,
            0x6db55cd306d31f7f,
            0xc49d293a81cb9641,
            0x1ce55a4fe979719f,
            0xa92e60a9d178a4d1,
            0x002cc64973bcfd8c,
        ],
        [
            0xcea721cce82fb11b,
            0xe5b55eb8098ece81,
            0x4e30525c6f1ddd66,
            0x43c6702827070987,
            0xaca68430a7b5762a,
            0x3674238634df9c93,
            0x88cee1c825e33433,
            0xde99ae8d74b57176,
        ],
    ],
    [
        [
            0x014ef1197d341346,
            0x9725e20825d07394,
            0xfdb25aef2c5bae3b,
            0xbe5402dc598c971e,
            0x93a5711f04cdca3d,
            0xc45a9a5b2f8fb97b,
            0xfe8946a924933545,
            0x2af997a27369091c,
        ],
        [
            0xaa62c88e0b294011,
            0x058eb9d810ce9f74,
            0xb3cb23eced349ae4,
            0xa3648177a77b4a84,
            0x43153d905992d95d,
            0xf4e2a97cda44aa4b,
            0x5baa2702b908682f,
            0x082923bdf4f750d1,
        ],
        [
            0x98ae09a325893803,
            0xf8a6475077968838,
            0xceb0735bf00b2c5f,
            0x0a1a5d953888e072,
            0x2fcb190489f94475,
            0xb5be06270dec69fc,
            0x739cb934b09acf8b,
            0x537750b75ec7f25b,
        ],
        [
            0xe9dd318bae1f3961,
            0xf7462137299efe1a,
            0xb1f6b8eee9adb940,
            0xbdebcc8a809dfe6b,
            0x40fc1f791b178113,
            0x3ac1c3362d014864,
            0x9a016184bdb8aeba,
            0x95f2394459fbc25e,
        ],
    ],
];
pub const HL_GOLDILOCKS_8_INTERNAL_ROUND_CONSTANTS: [u64; 22] = [
    0x488897d85ff51f56,
    0x1140737ccb162218,
    0xa7eeb9215866ed35,
    0x9bd2976fee49fcc9,
    0xc0c8f0de580a3fcc,
    0x4fb2dae6ee8fc793,
    0x343a89f35f37395b,
    0x223b525a77ca72c8,
    0x56ccb62574aaa918,
    0xc4d507d8027af9ed,
    0xa080673cf0b7e95c,
    0xf0184884eb70dcf8,
    0x044f10b0cb3d5c69,
    0xe9e3f7993938f186,
    0x1b761c80e772f459,
    0x606cec607a1b5fac,
    0x14a0c2e1d45f03cd,
    0x4eace8855398574f,
    0xf905ca7103eff3e6,
    0xf8c8f8d20862c059,
    0xb524fe8bdd678e5a,
    0xfbb7865901a1ec41,
];

#[cfg(test)]
mod tests {
    use core::array;

    use p3_field::PrimeCharacteristicRing;
    use p3_poseidon2::Poseidon2;
    use p3_symmetric::Permutation;

    use super::*;

    type F = Goldilocks;

    // A function which recreates the poseidon2 implementation in
    // https://github.com/HorizenLabs/poseidon2
    fn hl_poseidon2_goldilocks_width_8(input: &mut [F; 8]) {
        const WIDTH: usize = 8;

        // Our Poseidon2 implementation.
        let poseidon2: Poseidon2GoldilocksHL<WIDTH> = Poseidon2::new(
            ExternalLayerConstants::<Goldilocks, WIDTH>::new_from_saved_array(
                HL_GOLDILOCKS_8_EXTERNAL_ROUND_CONSTANTS,
                Goldilocks::new_array,
            ),
            Goldilocks::new_array(HL_GOLDILOCKS_8_INTERNAL_ROUND_CONSTANTS).to_vec(),
        );

        poseidon2.permute_mut(input);
    }

    /// Test on the constant 0 input.
    #[test]
    fn test_poseidon2_width_8_zeros() {
        let mut input: [F; 8] = [Goldilocks::ZERO; 8];

        let expected: [F; 8] = Goldilocks::new_array([
            18411014882916974180,
            11853243659833051879,
            2553980965289355629,
            5435536888074291950,
            11414233414141119281,
            15612551474745760831,
            15745650375519692590,
            4546169000627739578,
        ]);
        hl_poseidon2_goldilocks_width_8(&mut input);
        assert_eq!(input, expected);
    }

    /// Test on the input 0..16.
    #[test]
    fn test_poseidon2_width_8_range() {
        let mut input: [F; 8] = array::from_fn(|i| F::from_u64(i as u64));

        let expected: [F; 8] = Goldilocks::new_array([
            14758079437403499858,
            4768220715988658038,
            9988209636190012306,
            8808631253505580005,
            17526572370116009359,
            1590367810676479047,
            13027328087430412699,
            13357513690486523336,
        ]);
        hl_poseidon2_goldilocks_width_8(&mut input);
        assert_eq!(input, expected);
    }

    /// Test on a roughly random input.
    /// This random input is generated by the following sage code:
    /// set_random_seed(2468)
    /// vector([ZZ.random_element(2**31) for t in range(16)])
    #[test]
    fn test_poseidon2_width_8_random() {
        let mut input: [F; 8] = Goldilocks::new_array([
            5116996373749832116,
            8931548647907683339,
            17132360229780760684,
            11280040044015983889,
            11957737519043010992,
            15695650327991256125,
            17604752143022812942,
            543194415197607509,
        ]);

        let expected: [F; 8] = Goldilocks::new_array([
            9817406215841052104,
            16787690088272864961,
            16566820001699848722,
            7208405131694630795,
            16315106302112132474,
            15663526335160302273,
            8171740919697725040,
            7324539319521186184,
        ]);

        hl_poseidon2_goldilocks_width_8(&mut input);
        assert_eq!(input, expected);
    }

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
