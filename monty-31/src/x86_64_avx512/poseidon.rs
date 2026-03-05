//! AVX512-optimized Poseidon layers for MontyField31.

use alloc::vec::Vec;
use core::arch::x86_64::__m512i;
use core::marker::PhantomData;
use core::mem::transmute;

use p3_mds::karatsuba_convolution::{mds_circulant_karatsuba_16, mds_circulant_karatsuba_24};
use p3_poseidon::external::{
    FullRoundConstants, FullRoundLayer, FullRoundLayerConstructor, mds_multiply,
};
use p3_poseidon::internal::{
    PartialRoundConstants, PartialRoundLayer, PartialRoundLayerConstructor,
};

use super::poseidon2::{
    InternalLayer16, InternalLayer24, add_rc_and_sbox, convert_to_vec_neg_form, exp_small,
};
use crate::{
    FieldParameters, MDSUtils, MontyField31, PackedMontyField31AVX512, PackedMontyParameters,
    PartialRoundBaseParameters, RelativelyPrimePower, apply_func_to_even_odd,
};

/// AVX512-specific trait for Poseidon partial round parameters.
pub trait PartialRoundParametersAVX512<PMP: PackedMontyParameters, const WIDTH: usize>:
    Clone + Sync
{
}

/// The internal (partial round) layer of Poseidon for AVX512-packed MontyField31.
#[derive(Clone)]
pub struct PoseidonInternalLayerMonty31<
    PMP: PackedMontyParameters,
    const WIDTH: usize,
    ILP: PartialRoundBaseParameters<PMP, WIDTH>,
> {
    pub(crate) internal_constants: PartialRoundConstants<MontyField31<PMP>, WIDTH>,
    /// Pre-packed first round constants (broadcast to all AVX2 lanes).
    packed_first_round_constants: [PackedMontyField31AVX512<PMP>; WIDTH],
    /// Pre-packed round constants for partial rounds (RP-1 entries).
    packed_round_constants: Vec<PackedMontyField31AVX512<PMP>>,
    /// Pre-packed sparse first-row vectors for each partial round (RP entries).
    packed_sparse_first_row: Vec<[PackedMontyField31AVX512<PMP>; WIDTH]>,
    /// Pre-packed rank-1 update vectors for each partial round (RP entries).
    packed_v: Vec<[PackedMontyField31AVX512<PMP>; WIDTH]>,
    _phantom: PhantomData<ILP>,
}

/// The external (full round) layer of Poseidon for AVX512-packed MontyField31.
#[derive(Clone)]
pub struct PoseidonExternalLayerMonty31<
    PMP: PackedMontyParameters,
    MU: MDSUtils,
    const WIDTH: usize,
> {
    pub(crate) external_constants: FullRoundConstants<MontyField31<PMP>, WIDTH>,
    /// Pre-packed initial round constants in negative form for fused AddRC+S-box.
    packed_initial_constants: Vec<[__m512i; WIDTH]>,
    /// Pre-packed terminal round constants in negative form for fused AddRC+S-box.
    packed_terminal_constants: Vec<[__m512i; WIDTH]>,
    /// First column of the circulant MDS matrix (for Karatsuba convolution).
    circulant_col: [MontyField31<PMP>; WIDTH],
    _mds: PhantomData<MU>,
}

impl<FP: FieldParameters, const WIDTH: usize, ILP: PartialRoundBaseParameters<FP, WIDTH>>
    PartialRoundLayerConstructor<MontyField31<FP>, WIDTH>
    for PoseidonInternalLayerMonty31<FP, WIDTH, ILP>
{
    fn new_from_constants(
        internal_constants: PartialRoundConstants<MontyField31<FP>, WIDTH>,
    ) -> Self {
        let packed_first_round_constants = internal_constants
            .first_round_constants
            .map(PackedMontyField31AVX512::from);
        let packed_round_constants = internal_constants
            .round_constants
            .iter()
            .map(|&c| PackedMontyField31AVX512::from(c))
            .collect();
        let packed_sparse_first_row = internal_constants
            .sparse_first_row
            .iter()
            .map(|row| row.map(PackedMontyField31AVX512::from))
            .collect();
        let packed_v = internal_constants
            .v
            .iter()
            .map(|row| row.map(PackedMontyField31AVX512::from))
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
    for PoseidonExternalLayerMonty31<FP, MU, WIDTH>
{
    fn new_from_constants(external_constants: FullRoundConstants<MontyField31<FP>, WIDTH>) -> Self {
        let packed_initial_constants = external_constants
            .initial
            .iter()
            .map(|arr| arr.map(|c| convert_to_vec_neg_form::<FP>(c.value as i32)))
            .collect();
        let packed_terminal_constants = external_constants
            .terminal
            .iter()
            .map(|arr| arr.map(|c| convert_to_vec_neg_form::<FP>(c.value as i32)))
            .collect();
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

impl<FP, ILP, const D: u64> PartialRoundLayer<PackedMontyField31AVX512<FP>, 16, D>
    for PoseidonInternalLayerMonty31<FP, 16, ILP>
where
    FP: FieldParameters + RelativelyPrimePower<D>,
    ILP: PartialRoundBaseParameters<FP, 16> + PartialRoundParametersAVX512<FP, 16>,
{
    fn permute_state(&self, state: &mut [PackedMontyField31AVX512<FP>; 16]) {
        for (s, &c) in state
            .iter_mut()
            .zip(self.packed_first_round_constants.iter())
        {
            *s += c;
        }

        mds_multiply(state, &self.internal_constants.m_i);

        let mut split = InternalLayer16::from_packed_field_array(*state);
        let rounds_p = self.packed_sparse_first_row.len();

        for r in 0..rounds_p {
            unsafe {
                let vec_val = split.s0.to_vector();
                let output = apply_func_to_even_odd::<FP>(vec_val, exp_small::<FP, D>);
                split.s0 = PackedMontyField31AVX512::from_vector(output);
            }

            if r < rounds_p - 1 {
                split.s0 += self.packed_round_constants[r];
            }

            let s_hi: &[PackedMontyField31AVX512<FP>; 15] = unsafe { transmute(&split.s_hi) };
            let first_row = &self.packed_sparse_first_row[r];
            let mut partial_dot = s_hi[0] * first_row[1];
            for j in 1..15 {
                partial_dot += s_hi[j] * first_row[j + 1];
            }

            let s0_val = split.s0;
            split.s0 = s0_val * first_row[0] + partial_dot;

            let v = &self.packed_v[r];
            let s_hi_mut: &mut [PackedMontyField31AVX512<FP>; 15] =
                unsafe { transmute(&mut split.s_hi) };
            for j in 0..15 {
                s_hi_mut[j] += s0_val * v[j];
            }
        }

        *state = unsafe { split.to_packed_field_array() };
    }
}

impl<FP, ILP, const D: u64> PartialRoundLayer<PackedMontyField31AVX512<FP>, 24, D>
    for PoseidonInternalLayerMonty31<FP, 24, ILP>
where
    FP: FieldParameters + RelativelyPrimePower<D>,
    ILP: PartialRoundBaseParameters<FP, 24> + PartialRoundParametersAVX512<FP, 24>,
{
    fn permute_state(&self, state: &mut [PackedMontyField31AVX512<FP>; 24]) {
        for (s, &c) in state
            .iter_mut()
            .zip(self.packed_first_round_constants.iter())
        {
            *s += c;
        }

        mds_multiply(state, &self.internal_constants.m_i);

        let mut split = InternalLayer24::from_packed_field_array(*state);
        let rounds_p = self.packed_sparse_first_row.len();

        for r in 0..rounds_p {
            unsafe {
                let vec_val = split.s0.to_vector();
                let output = apply_func_to_even_odd::<FP>(vec_val, exp_small::<FP, D>);
                split.s0 = PackedMontyField31AVX512::from_vector(output);
            }

            if r < rounds_p - 1 {
                split.s0 += self.packed_round_constants[r];
            }

            let s_hi: &[PackedMontyField31AVX512<FP>; 23] = unsafe { transmute(&split.s_hi) };
            let first_row = &self.packed_sparse_first_row[r];
            let mut partial_dot = s_hi[0] * first_row[1];
            for j in 1..23 {
                partial_dot += s_hi[j] * first_row[j + 1];
            }

            let s0_val = split.s0;
            split.s0 = s0_val * first_row[0] + partial_dot;

            let v = &self.packed_v[r];
            let s_hi_mut: &mut [PackedMontyField31AVX512<FP>; 23] =
                unsafe { transmute(&mut split.s_hi) };
            for j in 0..23 {
                s_hi_mut[j] += s0_val * v[j];
            }
        }

        *state = unsafe { split.to_packed_field_array() };
    }
}

impl<FP, MU, const D: u64> FullRoundLayer<PackedMontyField31AVX512<FP>, 16, D>
    for PoseidonExternalLayerMonty31<FP, MU, 16>
where
    FP: FieldParameters + RelativelyPrimePower<D>,
    MU: MDSUtils,
{
    fn permute_state_initial(&self, state: &mut [PackedMontyField31AVX512<FP>; 16]) {
        for round_constants in &self.packed_initial_constants {
            for (s, &rc) in state.iter_mut().zip(round_constants.iter()) {
                add_rc_and_sbox::<FP, D>(s, rc);
            }
            mds_circulant_karatsuba_16(state, &self.circulant_col);
        }
    }

    fn permute_state_terminal(&self, state: &mut [PackedMontyField31AVX512<FP>; 16]) {
        for round_constants in &self.packed_terminal_constants {
            for (s, &rc) in state.iter_mut().zip(round_constants.iter()) {
                add_rc_and_sbox::<FP, D>(s, rc);
            }
            mds_circulant_karatsuba_16(state, &self.circulant_col);
        }
    }
}

impl<FP, MU, const D: u64> FullRoundLayer<PackedMontyField31AVX512<FP>, 24, D>
    for PoseidonExternalLayerMonty31<FP, MU, 24>
where
    FP: FieldParameters + RelativelyPrimePower<D>,
    MU: MDSUtils,
{
    fn permute_state_initial(&self, state: &mut [PackedMontyField31AVX512<FP>; 24]) {
        for round_constants in &self.packed_initial_constants {
            for (s, &rc) in state.iter_mut().zip(round_constants.iter()) {
                add_rc_and_sbox::<FP, D>(s, rc);
            }
            mds_circulant_karatsuba_24(state, &self.circulant_col);
        }
    }

    fn permute_state_terminal(&self, state: &mut [PackedMontyField31AVX512<FP>; 24]) {
        for round_constants in &self.packed_terminal_constants {
            for (s, &rc) in state.iter_mut().zip(round_constants.iter()) {
                add_rc_and_sbox::<FP, D>(s, rc);
            }
            mds_circulant_karatsuba_24(state, &self.circulant_col);
        }
    }
}
