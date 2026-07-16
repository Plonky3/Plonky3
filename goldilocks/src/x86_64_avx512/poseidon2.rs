//! Poseidon2 external/internal layers specialized for `PackedGoldilocksAVX512`.
//!
//! `Poseidon2ExternalLayerGoldilocks`/`Poseidon2InternalLayerGoldilocks` (in the crate root
//! `poseidon2` module) already act correctly on `PackedGoldilocksAVX512` through their generic
//! `Algebra<Goldilocks>` implementation, since `PackedGoldilocksAVX512` implements both
//! `Algebra<Goldilocks>` and `InjectiveMonomial<7>`. That generic path re-broadcasts every
//! round constant from scalar `Goldilocks` into a packed vector on every single
//! `permute_state`/`permute_state_initial`/`permute_state_terminal` call, which is wasted work
//! once the permutation is used to hash many inputs.
//!
//! The types here instead:
//! - broadcast each round constant into a `PackedGoldilocksAVX512` once, at construction time,
//!   and reuse the packed constants across every future permutation call, and
//! - in the internal rounds, compute the lane sum used by the diffusion matrix from
//!   `state[1..WIDTH]` independently of the S-box applied to `state[0]`, so the two
//!   (data-independent) computations can be scheduled without one waiting on the other,
//!   instead of forcing the sum to wait on the S-box's multiply/reduce chain.
//!
//! For scalar `Goldilocks` state, both layers simply delegate to the existing generic
//! implementation, which is already effectively optimal for a single field element.

use alloc::vec::Vec;

use p3_field::{InjectiveMonomial, PrimeCharacteristicRing};
use p3_poseidon2::{
    ExternalLayer, ExternalLayerConstants, ExternalLayerConstructor, InternalLayer,
    InternalLayerConstructor, MDSMat4, external_initial_permute_state,
    external_terminal_permute_state,
};

use crate::poseidon1::GOLDILOCKS_S_BOX_DEGREE;
use crate::x86_64_avx512::packing::PackedGoldilocksAVX512;
use crate::{Goldilocks, Poseidon2ExternalLayerGoldilocks, Poseidon2InternalLayerGoldilocks};

#[inline(always)]
fn add_rc_and_sbox(val: &mut PackedGoldilocksAVX512, rc: PackedGoldilocksAVX512) {
    *val = (*val + rc).injective_exp_n();
}

/// Apply one internal round (add round constant, S-box `state[0]`, diffuse) of the
/// width-8 Goldilocks Poseidon2 internal linear layer to a packed AVX512 state.
///
/// Mirrors `internal_layer_mat_mul_goldilocks_8` in the crate root `poseidon2` module, with
/// the lane sum split into a part independent of the S-box (`sum_tail`, covering
/// `state[1..8]`) and the S-box'd `state[0]`, combined only once both are ready.
#[inline(always)]
fn internal_round_goldilocks_8(
    state: &mut [PackedGoldilocksAVX512; 8],
    rc: PackedGoldilocksAVX512,
) {
    let s1 = state[1];
    let s2 = state[2];
    let s3 = state[3];
    let s4 = state[4];
    let s5 = state[5];
    let s6 = state[6];
    let s7 = state[7];
    let sum_tail = s1 + s2 + s3 + s4 + s5 + s6 + s7;

    add_rc_and_sbox(&mut state[0], rc);
    let s0 = state[0];
    let sum = sum_tail + s0;

    // V[0] = -2
    state[0] = sum - (s0 + s0);
    // V[1] = 1
    state[1] = sum + s1;
    // V[2] = 2
    state[2] = sum + (s2 + s2);
    // V[3] = 1/2
    state[3] = sum + s3.halve();
    // V[4] = 3
    state[4] = sum + (s4 + s4 + s4);
    // V[5] = -1/2
    state[5] = sum - s5.halve();
    // V[6] = -3
    state[6] = sum - (s6 + s6 + s6);
    // V[7] = -4
    let two_s7 = s7 + s7;
    state[7] = sum - (two_s7 + two_s7);
}

/// Apply one internal round of the width-12 Goldilocks Poseidon2 internal linear layer to a
/// packed AVX512 state. See [`internal_round_goldilocks_8`] and
/// `internal_layer_mat_mul_goldilocks_12` in the crate root `poseidon2` module.
#[inline(always)]
fn internal_round_goldilocks_12(
    state: &mut [PackedGoldilocksAVX512; 12],
    rc: PackedGoldilocksAVX512,
) {
    let s1 = state[1];
    let s2 = state[2];
    let s3 = state[3];
    let s4 = state[4];
    let s5 = state[5];
    let s6 = state[6];
    let s7 = state[7];
    let s8 = state[8];
    let s9 = state[9];
    let s10 = state[10];
    let s11 = state[11];
    let sum_tail = s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8 + s9 + s10 + s11;

    add_rc_and_sbox(&mut state[0], rc);
    let s0 = state[0];
    let sum = sum_tail + s0;

    // V[0] = -2
    state[0] = sum - (s0 + s0);
    // V[1] = 1
    state[1] = sum + s1;
    // V[2] = 2
    state[2] = sum + (s2 + s2);
    // V[3] = 1/2
    state[3] = sum + s3.halve();
    // V[4] = 3
    state[4] = sum + (s4 + s4 + s4);
    // V[5] = 4
    let two_s5 = s5 + s5;
    state[5] = sum + (two_s5 + two_s5);
    // V[6] = -1/2
    state[6] = sum - s6.halve();
    // V[7] = -3
    state[7] = sum - (s7 + s7 + s7);
    // V[8] = -4
    let two_s8 = s8 + s8;
    state[8] = sum - (two_s8 + two_s8);
    // V[9] = 1/2^2
    state[9] = sum + s9.halve().halve();
    // V[10] = -1/2^2
    state[10] = sum - s10.halve().halve();
    // V[11] = 1/2^3
    state[11] = sum + s11.halve().halve().halve();
}

/// Apply one internal round of the width-16 Goldilocks Poseidon2 internal linear layer to a
/// packed AVX512 state. See [`internal_round_goldilocks_8`] and
/// `internal_layer_mat_mul_goldilocks_16` in the crate root `poseidon2` module.
#[inline(always)]
fn internal_round_goldilocks_16(
    state: &mut [PackedGoldilocksAVX512; 16],
    rc: PackedGoldilocksAVX512,
) {
    let s1 = state[1];
    let s2 = state[2];
    let s3 = state[3];
    let s4 = state[4];
    let s5 = state[5];
    let s6 = state[6];
    let s7 = state[7];
    let s8 = state[8];
    let s9 = state[9];
    let s10 = state[10];
    let s11 = state[11];
    let s12 = state[12];
    let s13 = state[13];
    let s14 = state[14];
    let s15 = state[15];
    let sum_tail = s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8 + s9 + s10 + s11 + s12 + s13 + s14 + s15;

    add_rc_and_sbox(&mut state[0], rc);
    let s0 = state[0];
    let sum = sum_tail + s0;

    // V[0] = -2
    state[0] = sum - (s0 + s0);
    // V[1] = 1
    state[1] = sum + s1;
    // V[2] = 2
    state[2] = sum + (s2 + s2);
    // V[3] = 1/2
    state[3] = sum + s3.halve();
    // V[4] = 3
    state[4] = sum + (s4 + s4 + s4);
    // V[5] = 4
    let two_s5 = s5 + s5;
    state[5] = sum + (two_s5 + two_s5);
    // V[6] = -1/2
    state[6] = sum - s6.halve();
    // V[7] = -3
    state[7] = sum - (s7 + s7 + s7);
    // V[8] = -4
    let two_s8 = s8 + s8;
    state[8] = sum - (two_s8 + two_s8);
    // V[9] = 1/2^3
    state[9] = sum + s9.halve().halve().halve();
    // V[10] = 1/2^4
    state[10] = sum + s10.halve().halve().halve().halve();
    // V[11] = 1/2^5
    state[11] = sum + s11.halve().halve().halve().halve().halve();
    // V[12] = -1/2^3
    state[12] = sum - s12.halve().halve().halve();
    // V[13] = -1/2^4
    state[13] = sum - s13.halve().halve().halve().halve();
    // V[14] = -1/2^5
    state[14] = sum - s14.halve().halve().halve().halve().halve();
    // V[15] = 1/2^32
    let inv_2_32 = crate::MATRIX_DIAG_16_GOLDILOCKS[15];
    state[15] = sum + s15 * inv_2_32;
}

/// The internal layers of the Poseidon2 permutation, specialized for `PackedGoldilocksAVX512`.
#[derive(Clone, Debug, Default)]
pub struct Poseidon2InternalLayerGoldilocksAVX512 {
    /// Scalar fallback, used for `Permutation<[Goldilocks; WIDTH]>`.
    inner: Poseidon2InternalLayerGoldilocks,
    /// Each internal round constant, pre-broadcast into a packed vector.
    packed_internal_constants: Vec<PackedGoldilocksAVX512>,
}

impl InternalLayerConstructor<Goldilocks> for Poseidon2InternalLayerGoldilocksAVX512 {
    fn new_from_constants(internal_constants: Vec<Goldilocks>) -> Self {
        let packed_internal_constants = internal_constants
            .iter()
            .copied()
            .map(PackedGoldilocksAVX512::from)
            .collect();
        let inner = Poseidon2InternalLayerGoldilocks::new_from_constants(internal_constants);
        Self {
            inner,
            packed_internal_constants,
        }
    }
}

impl InternalLayer<Goldilocks, 8, GOLDILOCKS_S_BOX_DEGREE>
    for Poseidon2InternalLayerGoldilocksAVX512
{
    fn permute_state(&self, state: &mut [Goldilocks; 8]) {
        InternalLayer::<Goldilocks, 8, GOLDILOCKS_S_BOX_DEGREE>::permute_state(&self.inner, state);
    }
}

impl InternalLayer<Goldilocks, 12, GOLDILOCKS_S_BOX_DEGREE>
    for Poseidon2InternalLayerGoldilocksAVX512
{
    fn permute_state(&self, state: &mut [Goldilocks; 12]) {
        InternalLayer::<Goldilocks, 12, GOLDILOCKS_S_BOX_DEGREE>::permute_state(&self.inner, state);
    }
}

impl InternalLayer<Goldilocks, 16, GOLDILOCKS_S_BOX_DEGREE>
    for Poseidon2InternalLayerGoldilocksAVX512
{
    fn permute_state(&self, state: &mut [Goldilocks; 16]) {
        InternalLayer::<Goldilocks, 16, GOLDILOCKS_S_BOX_DEGREE>::permute_state(&self.inner, state);
    }
}

// `MATRIX_DIAG_20_GOLDILOCKS` holds opaque full field elements rather than small
// integers/powers of two (see its doc comment in the crate root `poseidon2` module), so it
// doesn't admit the same cheap-add fast path as widths 8/12/16. Width 20 is otherwise
// unused in this crate (no default constructor, no known-answer test), so both the scalar
// and packed cases simply delegate to the existing generic implementation for correctness
// parity with every other Goldilocks Poseidon2 backend, rather than leaving it unsupported.
impl InternalLayer<Goldilocks, 20, GOLDILOCKS_S_BOX_DEGREE>
    for Poseidon2InternalLayerGoldilocksAVX512
{
    fn permute_state(&self, state: &mut [Goldilocks; 20]) {
        InternalLayer::<Goldilocks, 20, GOLDILOCKS_S_BOX_DEGREE>::permute_state(&self.inner, state);
    }
}

impl InternalLayer<PackedGoldilocksAVX512, 20, GOLDILOCKS_S_BOX_DEGREE>
    for Poseidon2InternalLayerGoldilocksAVX512
{
    fn permute_state(&self, state: &mut [PackedGoldilocksAVX512; 20]) {
        InternalLayer::<PackedGoldilocksAVX512, 20, GOLDILOCKS_S_BOX_DEGREE>::permute_state(
            &self.inner,
            state,
        );
    }
}

impl InternalLayer<PackedGoldilocksAVX512, 8, GOLDILOCKS_S_BOX_DEGREE>
    for Poseidon2InternalLayerGoldilocksAVX512
{
    fn permute_state(&self, state: &mut [PackedGoldilocksAVX512; 8]) {
        for &rc in &self.packed_internal_constants {
            internal_round_goldilocks_8(state, rc);
        }
    }
}

impl InternalLayer<PackedGoldilocksAVX512, 12, GOLDILOCKS_S_BOX_DEGREE>
    for Poseidon2InternalLayerGoldilocksAVX512
{
    fn permute_state(&self, state: &mut [PackedGoldilocksAVX512; 12]) {
        for &rc in &self.packed_internal_constants {
            internal_round_goldilocks_12(state, rc);
        }
    }
}

impl InternalLayer<PackedGoldilocksAVX512, 16, GOLDILOCKS_S_BOX_DEGREE>
    for Poseidon2InternalLayerGoldilocksAVX512
{
    fn permute_state(&self, state: &mut [PackedGoldilocksAVX512; 16]) {
        for &rc in &self.packed_internal_constants {
            internal_round_goldilocks_16(state, rc);
        }
    }
}

/// The external layers of the Poseidon2 permutation, specialized for `PackedGoldilocksAVX512`.
#[derive(Clone)]
pub struct Poseidon2ExternalLayerGoldilocksAVX512<const WIDTH: usize> {
    /// Scalar fallback, used for `Permutation<[Goldilocks; WIDTH]>`.
    inner: Poseidon2ExternalLayerGoldilocks<WIDTH>,
    /// Each initial-round constant vector, pre-broadcast lane-wise into packed vectors.
    packed_initial_external_constants: Vec<[PackedGoldilocksAVX512; WIDTH]>,
    /// Each terminal-round constant vector, pre-broadcast lane-wise into packed vectors.
    packed_terminal_external_constants: Vec<[PackedGoldilocksAVX512; WIDTH]>,
}

impl<const WIDTH: usize> ExternalLayerConstructor<Goldilocks, WIDTH>
    for Poseidon2ExternalLayerGoldilocksAVX512<WIDTH>
{
    fn new_from_constants(external_constants: ExternalLayerConstants<Goldilocks, WIDTH>) -> Self {
        let pack_round = |rc: &[Goldilocks; WIDTH]| rc.map(PackedGoldilocksAVX512::from);
        let packed_initial_external_constants = external_constants
            .get_initial_constants()
            .iter()
            .map(pack_round)
            .collect();
        let packed_terminal_external_constants = external_constants
            .get_terminal_constants()
            .iter()
            .map(pack_round)
            .collect();
        let inner = Poseidon2ExternalLayerGoldilocks::new_from_constants(external_constants);
        Self {
            inner,
            packed_initial_external_constants,
            packed_terminal_external_constants,
        }
    }
}

impl<const WIDTH: usize> ExternalLayer<Goldilocks, WIDTH, GOLDILOCKS_S_BOX_DEGREE>
    for Poseidon2ExternalLayerGoldilocksAVX512<WIDTH>
{
    fn permute_state_initial(&self, state: &mut [Goldilocks; WIDTH]) {
        ExternalLayer::<Goldilocks, WIDTH, GOLDILOCKS_S_BOX_DEGREE>::permute_state_initial(
            &self.inner,
            state,
        );
    }

    fn permute_state_terminal(&self, state: &mut [Goldilocks; WIDTH]) {
        ExternalLayer::<Goldilocks, WIDTH, GOLDILOCKS_S_BOX_DEGREE>::permute_state_terminal(
            &self.inner,
            state,
        );
    }
}

impl<const WIDTH: usize> ExternalLayer<PackedGoldilocksAVX512, WIDTH, GOLDILOCKS_S_BOX_DEGREE>
    for Poseidon2ExternalLayerGoldilocksAVX512<WIDTH>
{
    fn permute_state_initial(&self, state: &mut [PackedGoldilocksAVX512; WIDTH]) {
        external_initial_permute_state(
            state,
            &self.packed_initial_external_constants,
            add_rc_and_sbox,
            &MDSMat4,
        );
    }

    fn permute_state_terminal(&self, state: &mut [PackedGoldilocksAVX512; WIDTH]) {
        external_terminal_permute_state(
            state,
            &self.packed_terminal_external_constants,
            add_rc_and_sbox,
            &MDSMat4,
        );
    }
}

#[cfg(test)]
mod tests {
    use p3_field::PackedValue;
    use p3_symmetric::Permutation;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;
    use crate::{
        default_goldilocks_poseidon2_8, default_goldilocks_poseidon2_12,
        default_goldilocks_poseidon2_16,
    };

    const PACKING_WIDTH: usize = <PackedGoldilocksAVX512 as PackedValue>::WIDTH;

    /// Applying the packed AVX512 permutation lane-by-lane must agree with applying the scalar
    /// permutation independently to each lane, for every lane of a random packed state.
    fn assert_packed_matches_scalar<const WIDTH: usize>(
        scalar_perm: &impl Permutation<[Goldilocks; WIDTH]>,
        packed_perm: &impl Permutation<[PackedGoldilocksAVX512; WIDTH]>,
        rng: &mut SmallRng,
    ) {
        let lanes: [[Goldilocks; WIDTH]; PACKING_WIDTH] = core::array::from_fn(|_| rng.random());

        let mut packed_state: [PackedGoldilocksAVX512; WIDTH] =
            core::array::from_fn(|i| PackedGoldilocksAVX512::from_fn(|l| lanes[l][i]));
        packed_perm.permute_mut(&mut packed_state);

        for (l, lane) in lanes.into_iter().enumerate() {
            let mut scalar_state = lane;
            scalar_perm.permute_mut(&mut scalar_state);
            for i in 0..WIDTH {
                assert_eq!(
                    scalar_state[i],
                    packed_state[i].as_slice()[l],
                    "width {WIDTH}, lane {l}, element {i}"
                );
            }
        }
    }

    #[test]
    fn packed_matches_scalar_width_8() {
        let mut rng = SmallRng::seed_from_u64(1);
        let perm = default_goldilocks_poseidon2_8();
        assert_packed_matches_scalar(&perm, &perm, &mut rng);
    }

    #[test]
    fn packed_matches_scalar_width_12() {
        let mut rng = SmallRng::seed_from_u64(2);
        let perm = default_goldilocks_poseidon2_12();
        assert_packed_matches_scalar(&perm, &perm, &mut rng);
    }

    #[test]
    fn packed_matches_scalar_width_16() {
        let mut rng = SmallRng::seed_from_u64(3);
        let perm = default_goldilocks_poseidon2_16();
        assert_packed_matches_scalar(&perm, &perm, &mut rng);
    }

    /// Width 20 has no default constructor or known-answer test anywhere in this crate (its
    /// diagonal constants are undocumented), so there is no `default_goldilocks_poseidon2_20`
    /// to call here. This only exercises that the scalar and packed `InternalLayer`/
    /// `ExternalLayer` impls agree with each other for width 20 (i.e. that delegating to the
    /// generic implementation was wired correctly), not that the round numbers used are
    /// cryptographically appropriate.
    #[test]
    fn packed_matches_scalar_width_20() {
        let mut rng = SmallRng::seed_from_u64(4);
        let perm: p3_poseidon2::Poseidon2<
            Goldilocks,
            Poseidon2ExternalLayerGoldilocksAVX512<20>,
            Poseidon2InternalLayerGoldilocksAVX512,
            20,
            GOLDILOCKS_S_BOX_DEGREE,
        > = p3_poseidon2::Poseidon2::new_from_rng(8, 22, &mut rng);
        assert_packed_matches_scalar(&perm, &perm, &mut rng);
    }
}
