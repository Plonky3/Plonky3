use alloc::vec::Vec;

use p3_field::PrimeField;
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use p3_poseidon2::{DiffusionPermutation, MdsLightPermutation};
use tracing::instrument;

use crate::columns::{num_cols, Poseidon2Cols};
use crate::{FullRound, PartialRound, RoundConstants, SBox};

#[instrument(name = "generate vectorized Poseidon2 trace", skip_all)]
pub fn generate_vectorized_trace_rows<
    F: PrimeField,
    MdsLight: MdsLightPermutation<F, WIDTH>,
    Diffusion: DiffusionPermutation<F, WIDTH>,
    const WIDTH: usize,
    const SBOX_DEGREE: usize,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
    const VECTOR_LEN: usize,
>(
    inputs: Vec<[F; WIDTH]>,
    round_constants: &RoundConstants<F, WIDTH, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>,
    external_linear_layer: &MdsLight,
    internal_linear_layer: &Diffusion,
) -> RowMajorMatrix<F> {
    let n = inputs.len();
    assert!(
        n % VECTOR_LEN == 0 && (n / VECTOR_LEN).is_power_of_two(),
        "Callers expected to pad inputs to VECTOR_LEN times a power of two"
    );

    let nrows = n.div_ceil(VECTOR_LEN);
    let ncols = num_cols::<WIDTH, SBOX_DEGREE, SBOX_REGISTERS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>()
        * VECTOR_LEN;
    let mut trace = RowMajorMatrix::new(F::zero_vec(nrows * ncols), ncols);

    let (prefix, perms, suffix) = unsafe {
        trace.values.align_to_mut::<Poseidon2Cols<
            F,
            WIDTH,
            SBOX_DEGREE,
            SBOX_REGISTERS,
            HALF_FULL_ROUNDS,
            PARTIAL_ROUNDS,
        >>()
    };
    assert!(prefix.is_empty(), "Alignment should match");
    assert!(suffix.is_empty(), "Alignment should match");
    assert_eq!(perms.len(), n);

    perms.par_iter_mut().zip(inputs).for_each(|(perm, input)| {
        generate_trace_rows_for_perm(
            perm,
            input,
            round_constants,
            external_linear_layer,
            internal_linear_layer,
        );
    });

    trace
}

// TODO: Take generic iterable
#[instrument(name = "generate Poseidon2 trace", skip_all)]
pub fn generate_trace_rows<
    F: PrimeField,
    MdsLight: MdsLightPermutation<F, WIDTH>,
    Diffusion: DiffusionPermutation<F, WIDTH>,
    const WIDTH: usize,
    const SBOX_DEGREE: usize,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
>(
    inputs: Vec<[F; WIDTH]>,
    constants: &RoundConstants<F, WIDTH, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>,
    external_linear_layer: &MdsLight,
    internal_linear_layer: &Diffusion,
) -> RowMajorMatrix<F> {
    let n = inputs.len();
    assert!(
        n.is_power_of_two(),
        "Callers expected to pad inputs to a power of two"
    );

    let ncols = num_cols::<WIDTH, SBOX_DEGREE, SBOX_REGISTERS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>();
    let mut trace = RowMajorMatrix::new(F::zero_vec(n * ncols), ncols);

    let (prefix, perms, suffix) = unsafe {
        trace.values.align_to_mut::<Poseidon2Cols<
            F,
            WIDTH,
            SBOX_DEGREE,
            SBOX_REGISTERS,
            HALF_FULL_ROUNDS,
            PARTIAL_ROUNDS,
        >>()
    };
    assert!(prefix.is_empty(), "Alignment should match");
    assert!(suffix.is_empty(), "Alignment should match");
    assert_eq!(perms.len(), n);

    perms.par_iter_mut().zip(inputs).for_each(|(perm, input)| {
        generate_trace_rows_for_perm(
            perm,
            input,
            constants,
            external_linear_layer,
            internal_linear_layer,
        );
    });

    trace
}

/// `rows` will normally consist of 24 rows, with an exception for the final row.
fn generate_trace_rows_for_perm<
    F: PrimeField,
    MdsLight: MdsLightPermutation<F, WIDTH>,
    Diffusion: DiffusionPermutation<F, WIDTH>,
    const WIDTH: usize,
    const SBOX_DEGREE: usize,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
>(
    perm: &mut Poseidon2Cols<
        F,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
    >,
    mut state: [F; WIDTH],
    constants: &RoundConstants<F, WIDTH, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>,
    external_linear_layer: &MdsLight,
    internal_linear_layer: &Diffusion,
) {
    perm.export = F::one();
    perm.inputs = state;

    external_linear_layer.permute_mut(&mut state);

    for (full_round, constants) in perm
        .beginning_full_rounds
        .iter_mut()
        .zip(&constants.beginning_full_round_constants)
    {
        generate_full_round::<F, MdsLight, WIDTH, SBOX_DEGREE, SBOX_REGISTERS>(
            &mut state,
            full_round,
            constants,
            external_linear_layer,
        );
    }

    for (partial_round, constant) in perm
        .partial_rounds
        .iter_mut()
        .zip(&constants.partial_round_constants)
    {
        generate_partial_round::<F, Diffusion, WIDTH, SBOX_DEGREE, SBOX_REGISTERS>(
            &mut state,
            partial_round,
            *constant,
            internal_linear_layer,
        );
    }

    for (full_round, constants) in perm
        .ending_full_rounds
        .iter_mut()
        .zip(&constants.ending_full_round_constants)
    {
        generate_full_round::<F, MdsLight, WIDTH, SBOX_DEGREE, SBOX_REGISTERS>(
            &mut state,
            full_round,
            constants,
            external_linear_layer,
        );
    }
}

#[inline]
fn generate_full_round<
    F: PrimeField,
    MdsLight: MdsLightPermutation<F, WIDTH>,
    const WIDTH: usize,
    const SBOX_DEGREE: usize,
    const SBOX_REGISTERS: usize,
>(
    state: &mut [F; WIDTH],
    full_round: &mut FullRound<F, WIDTH, SBOX_DEGREE, SBOX_REGISTERS>,
    round_constants: &[F; WIDTH],
    external_linear_layer: &MdsLight,
) {
    for (state_i, const_i) in state.iter_mut().zip(round_constants) {
        *state_i += *const_i;
    }
    for (state_i, sbox_i) in state.iter_mut().zip(full_round.sbox.iter_mut()) {
        generate_sbox(sbox_i, state_i);
    }
    external_linear_layer.permute_mut(state);
    full_round.post = *state;
}

#[inline]
fn generate_partial_round<
    F: PrimeField,
    Diffusion: DiffusionPermutation<F, WIDTH>,
    const WIDTH: usize,
    const SBOX_DEGREE: usize,
    const SBOX_REGISTERS: usize,
>(
    state: &mut [F; WIDTH],
    partial_round: &mut PartialRound<F, WIDTH, SBOX_DEGREE, SBOX_REGISTERS>,
    round_constant: F,
    internal_linear_layer: &Diffusion,
) {
    state[0] += round_constant;
    generate_sbox(&mut partial_round.sbox, &mut state[0]);
    partial_round.post_sbox = state[0];
    internal_linear_layer.permute_mut(state);
}

#[inline]
fn generate_sbox<F: PrimeField, const DEGREE: usize, const REGISTERS: usize>(
    sbox: &mut SBox<F, DEGREE, REGISTERS>,
    x: &mut F,
) {
    *x = match (DEGREE, REGISTERS) {
        (3, 0) => x.cube(),
        (5, 0) => x.exp_const_u64::<5>(),
        (7, 0) => x.exp_const_u64::<7>(),
        (5, 1) => {
            let x2 = x.square();
            let x3 = x2 * *x;
            sbox.0[0] = x3;
            x3 * x2
        }
        (7, 1) => {
            let x3 = x.cube();
            sbox.0[0] = x3;
            x3 * x3 * *x
        }
        (11, 2) => {
            let x2 = x.square();
            let x3 = x2 * *x;
            let x9 = x3.cube();
            sbox.0[0] = x3;
            sbox.0[1] = x9;
            x9 * x2
        }
        _ => panic!(
            "Unexpected (DEGREE, REGISTERS) of ({}, {})",
            DEGREE, REGISTERS
        ),
    }
}
