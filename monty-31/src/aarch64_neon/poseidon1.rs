//! NEON-optimized Poseidon1 layers for MontyField31.

use alloc::vec::Vec;
use core::arch::aarch64::int32x4_t;
use core::marker::PhantomData;
use core::mem::transmute;

use p3_field::PrimeCharacteristicRing;
use p3_mds::karatsuba_convolution::{mds_circulant_karatsuba_16, mds_circulant_karatsuba_24};
use p3_poseidon1::external::{
    FullRoundConstants, FullRoundLayer, FullRoundLayerConstructor, mds_multiply,
};
use p3_poseidon1::internal::{
    PartialRoundConstants, PartialRoundLayer, PartialRoundLayerConstructor,
};

use super::poseidon2::{
    InternalLayer16, InternalLayer24, add_rc_and_sbox, convert_to_vec_neg_form_neon,
};
use crate::{
    FieldParameters, MDSUtils, MontyField31, PackedMontyField31Neon, PackedMontyParameters,
    PartialRoundBaseParameters, RelativelyPrimePower, exp_small,
};

/// NEON-specific trait for Poseidon1 partial round parameters.
pub trait PartialRoundParametersNeon<PMP: PackedMontyParameters, const WIDTH: usize>:
    Clone + Sync
{
}

/// The internal (partial round) layer of Poseidon1 for NEON-packed MontyField31.
#[derive(Clone)]
pub struct Poseidon1InternalLayerMonty31<
    PMP: PackedMontyParameters,
    const WIDTH: usize,
    ILP: PartialRoundBaseParameters<PMP, WIDTH>,
> {
    pub(crate) internal_constants: PartialRoundConstants<MontyField31<PMP>, WIDTH>,
    /// Pre-packed first round constants (broadcast to all NEON lanes).
    packed_first_round_constants: [PackedMontyField31Neon<PMP>; WIDTH],
    /// Pre-packed round constants for partial rounds (RP-1 entries).
    packed_round_constants: Vec<PackedMontyField31Neon<PMP>>,
    /// Pre-packed sparse first-row vectors for each partial round (RP entries).
    packed_sparse_first_row: Vec<[PackedMontyField31Neon<PMP>; WIDTH]>,
    /// Pre-packed rank-1 update vectors for each partial round (RP entries).
    packed_v: Vec<[PackedMontyField31Neon<PMP>; WIDTH]>,
    _phantom: PhantomData<ILP>,
}

/// The external (full round) layer of Poseidon1 for NEON-packed MontyField31.
#[derive(Clone)]
pub struct Poseidon1ExternalLayerMonty31<
    PMP: PackedMontyParameters,
    MU: MDSUtils,
    const WIDTH: usize,
> {
    pub(crate) external_constants: FullRoundConstants<MontyField31<PMP>, WIDTH>,
    /// Pre-packed initial round constants in negative form for fused AddRC+S-box.
    packed_initial_constants: Vec<[int32x4_t; WIDTH]>,
    /// Pre-packed terminal round constants in negative form for fused AddRC+S-box.
    packed_terminal_constants: Vec<[int32x4_t; WIDTH]>,
    /// First column of the circulant MDS matrix (for Karatsuba convolution).
    circulant_col: [MontyField31<PMP>; WIDTH],
    _mds: PhantomData<MU>,
}

impl<FP: FieldParameters, const WIDTH: usize, ILP: PartialRoundBaseParameters<FP, WIDTH>>
    PartialRoundLayerConstructor<MontyField31<FP>, WIDTH>
    for Poseidon1InternalLayerMonty31<FP, WIDTH, ILP>
{
    fn new_from_constants(
        internal_constants: PartialRoundConstants<MontyField31<FP>, WIDTH>,
    ) -> Self {
        let packed_first_round_constants = internal_constants
            .first_round_constants
            .map(PackedMontyField31Neon::from);
        let packed_round_constants = internal_constants
            .round_constants
            .iter()
            .map(|&c| PackedMontyField31Neon::from(c))
            .collect();
        let packed_sparse_first_row = internal_constants
            .sparse_first_row
            .iter()
            .map(|row| row.map(PackedMontyField31Neon::from))
            .collect();
        let packed_v = internal_constants
            .v
            .iter()
            .map(|row| row.map(PackedMontyField31Neon::from))
            .collect();
        Self {
            internal_constants,
            packed_first_round_constants,
            packed_round_constants,
            packed_sparse_first_row,
            packed_v,
            _phantom: PhantomData,
        }
    }
}

impl<FP: FieldParameters, MU: MDSUtils, const WIDTH: usize>
    FullRoundLayerConstructor<MontyField31<FP>, WIDTH>
    for Poseidon1ExternalLayerMonty31<FP, MU, WIDTH>
{
    fn new_from_constants(external_constants: FullRoundConstants<MontyField31<FP>, WIDTH>) -> Self {
        let packed_initial_constants = external_constants
            .initial
            .iter()
            .map(|arr| arr.map(|c| convert_to_vec_neg_form_neon::<FP>(c.value as i32)))
            .collect();
        let packed_terminal_constants = external_constants
            .terminal
            .iter()
            .map(|arr| arr.map(|c| convert_to_vec_neg_form_neon::<FP>(c.value as i32)))
            .collect();
        // Extract the first column of the circulant MDS for Karatsuba.
        let circulant_col = core::array::from_fn(|i| external_constants.dense_mds[i][0]);
        Self {
            external_constants,
            packed_initial_constants,
            packed_terminal_constants,
            circulant_col,
            _mds: PhantomData,
        }
    }
}

impl<FP, ILP, const D: u64> PartialRoundLayer<PackedMontyField31Neon<FP>, 16, D>
    for Poseidon1InternalLayerMonty31<FP, 16, ILP>
where
    FP: FieldParameters + RelativelyPrimePower<D>,
    ILP: PartialRoundBaseParameters<FP, 16> + PartialRoundParametersNeon<FP, 16>,
{
    fn permute_state(&self, state: &mut [PackedMontyField31Neon<FP>; 16]) {
        // 1. Add first round constants.
        for (s, &c) in state
            .iter_mut()
            .zip(self.packed_first_round_constants.iter())
        {
            *s += c;
        }

        // 2. Apply dense transition matrix m_i (once).
        mds_multiply(state, &self.internal_constants.m_i);

        // 3. Partial rounds with latency hiding via InternalLayer16.
        let mut split = InternalLayer16::from_packed_field_array(*state);
        let rounds_p = self.packed_sparse_first_row.len();

        for r in 0..rounds_p {
            // PATH A (high latency): S-box on s0.
            unsafe {
                let s0_signed = split.s0.to_signed_vector();
                let s0_sboxed = exp_small::<FP, D>(s0_signed);
                split.s0 = PackedMontyField31Neon::from_vector(s0_sboxed);
            }

            // Add scalar round constant (except last round).
            if r < rounds_p - 1 {
                split.s0 += self.packed_round_constants[r];
            }

            // PATH B (can overlap with S-box): partial dot product on s_hi.
            let s_hi: &[PackedMontyField31Neon<FP>; 15] = unsafe { transmute(&split.s_hi) };
            let first_row = &self.packed_sparse_first_row[r];
            let first_row_hi: [PackedMontyField31Neon<FP>; 15] =
                core::array::from_fn(|i| first_row[i + 1]);
            let partial_dot = PackedMontyField31Neon::<FP>::dot_product(s_hi, &first_row_hi);

            // SERIAL: complete s0 and rank-1 update.
            let s0_val = split.s0;
            split.s0 = s0_val * first_row[0] + partial_dot;

            let v = &self.packed_v[r];
            let s_hi_mut: &mut [PackedMontyField31Neon<FP>; 15] =
                unsafe { transmute(&mut split.s_hi) };
            for j in 0..15 {
                s_hi_mut[j] += s0_val * v[j];
            }
        }

        // 4. Convert back.
        *state = unsafe { split.to_packed_field_array() };
    }
}

impl<FP, ILP, const D: u64> PartialRoundLayer<PackedMontyField31Neon<FP>, 24, D>
    for Poseidon1InternalLayerMonty31<FP, 24, ILP>
where
    FP: FieldParameters + RelativelyPrimePower<D>,
    ILP: PartialRoundBaseParameters<FP, 24> + PartialRoundParametersNeon<FP, 24>,
{
    fn permute_state(&self, state: &mut [PackedMontyField31Neon<FP>; 24]) {
        // 1. Add first round constants.
        for (s, &c) in state
            .iter_mut()
            .zip(self.packed_first_round_constants.iter())
        {
            *s += c;
        }

        // 2. Apply dense transition matrix m_i (once).
        mds_multiply(state, &self.internal_constants.m_i);

        // 3. Partial rounds with latency hiding via InternalLayer24.
        let mut split = InternalLayer24::from_packed_field_array(*state);
        let rounds_p = self.packed_sparse_first_row.len();

        for r in 0..rounds_p {
            // PATH A (high latency): S-box on s0.
            unsafe {
                let s0_signed = split.s0.to_signed_vector();
                let s0_sboxed = exp_small::<FP, D>(s0_signed);
                split.s0 = PackedMontyField31Neon::from_vector(s0_sboxed);
            }

            // Add scalar round constant (except last round).
            if r < rounds_p - 1 {
                split.s0 += self.packed_round_constants[r];
            }

            // PATH B (can overlap with S-box): partial dot product on s_hi.
            let s_hi: &[PackedMontyField31Neon<FP>; 23] = unsafe { transmute(&split.s_hi) };
            let first_row = &self.packed_sparse_first_row[r];
            let first_row_hi: [PackedMontyField31Neon<FP>; 23] =
                core::array::from_fn(|i| first_row[i + 1]);
            let partial_dot = PackedMontyField31Neon::<FP>::dot_product(s_hi, &first_row_hi);

            // SERIAL: complete s0 and rank-1 update.
            let s0_val = split.s0;
            split.s0 = s0_val * first_row[0] + partial_dot;

            let v = &self.packed_v[r];
            let s_hi_mut: &mut [PackedMontyField31Neon<FP>; 23] =
                unsafe { transmute(&mut split.s_hi) };
            for j in 0..23 {
                s_hi_mut[j] += s0_val * v[j];
            }
        }

        // 4. Convert back.
        *state = unsafe { split.to_packed_field_array() };
    }
}

impl<FP, MU, const D: u64> FullRoundLayer<PackedMontyField31Neon<FP>, 16, D>
    for Poseidon1ExternalLayerMonty31<FP, MU, 16>
where
    FP: FieldParameters + RelativelyPrimePower<D>,
    MU: MDSUtils,
{
    fn permute_state_initial(&self, state: &mut [PackedMontyField31Neon<FP>; 16]) {
        for round_constants in &self.packed_initial_constants {
            for (s, &rc) in state.iter_mut().zip(round_constants.iter()) {
                add_rc_and_sbox::<FP, D>(s, rc);
            }
            mds_circulant_karatsuba_16(state, &self.circulant_col);
        }
    }

    fn permute_state_terminal(&self, state: &mut [PackedMontyField31Neon<FP>; 16]) {
        for round_constants in &self.packed_terminal_constants {
            for (s, &rc) in state.iter_mut().zip(round_constants.iter()) {
                add_rc_and_sbox::<FP, D>(s, rc);
            }
            mds_circulant_karatsuba_16(state, &self.circulant_col);
        }
    }
}

impl<FP, MU, const D: u64> FullRoundLayer<PackedMontyField31Neon<FP>, 24, D>
    for Poseidon1ExternalLayerMonty31<FP, MU, 24>
where
    FP: FieldParameters + RelativelyPrimePower<D>,
    MU: MDSUtils,
{
    fn permute_state_initial(&self, state: &mut [PackedMontyField31Neon<FP>; 24]) {
        for round_constants in &self.packed_initial_constants {
            for (s, &rc) in state.iter_mut().zip(round_constants.iter()) {
                add_rc_and_sbox::<FP, D>(s, rc);
            }
            mds_circulant_karatsuba_24(state, &self.circulant_col);
        }
    }

    fn permute_state_terminal(&self, state: &mut [PackedMontyField31Neon<FP>; 24]) {
        for round_constants in &self.packed_terminal_constants {
            for (s, &rc) in state.iter_mut().zip(round_constants.iter()) {
                add_rc_and_sbox::<FP, D>(s, rc);
            }
            mds_circulant_karatsuba_24(state, &self.circulant_col);
        }
    }
}
