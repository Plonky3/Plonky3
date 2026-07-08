use alloc::vec::Vec;
use core::mem::MaybeUninit;

use p3_field::{PackedValue, PrimeCharacteristicRing, PrimeField};
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixViewMut};
use p3_maybe_rayon::prelude::*;
use p3_poseidon2::GenericPoseidon2LinearLayers;
use tracing::instrument;

use crate::columns::{Poseidon2Cols, num_cols};
use crate::{FullRound, PartialRound, RoundConstants, SBox};

#[instrument(name = "generate vectorized Poseidon2 trace", skip_all)]
pub fn generate_vectorized_trace_rows<
    F: PrimeField,
    LinearLayers: GenericPoseidon2LinearLayers<WIDTH>,
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
    const VECTOR_LEN: usize,
>(
    inputs: Vec<[F; WIDTH]>,
    round_constants: &RoundConstants<F, WIDTH, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>,
    extra_capacity_bits: usize,
) -> RowMajorMatrix<F> {
    let n = inputs.len();
    assert!(
        n.is_multiple_of(VECTOR_LEN) && (n / VECTOR_LEN).is_power_of_two(),
        "Callers expected to pad inputs to VECTOR_LEN times a power of two"
    );

    let nrows = n.div_ceil(VECTOR_LEN);
    let ncols = num_cols::<WIDTH, SBOX_DEGREE, SBOX_REGISTERS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>()
        * VECTOR_LEN;
    let mut vec = Vec::with_capacity((nrows * ncols) << extra_capacity_bits);
    let trace = &mut vec.spare_capacity_mut()[..nrows * ncols];
    let trace = RowMajorMatrixViewMut::new(trace, ncols);

    let (prefix, perms, suffix) = unsafe {
        trace.values.align_to_mut::<Poseidon2Cols<
            MaybeUninit<F>,
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

    generate_perms::<
        F,
        LinearLayers,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
    >(perms, inputs, round_constants);

    unsafe {
        vec.set_len(nrows * ncols);
    }

    RowMajorMatrix::new(vec, ncols)
}

// TODO: Take generic iterable
#[instrument(name = "generate Poseidon2 trace", skip_all)]
pub fn generate_trace_rows<
    F: PrimeField,
    LinearLayers: GenericPoseidon2LinearLayers<WIDTH>,
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
>(
    inputs: Vec<[F; WIDTH]>,
    constants: &RoundConstants<F, WIDTH, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>,
    extra_capacity_bits: usize,
) -> RowMajorMatrix<F> {
    let n = inputs.len();
    assert!(
        n.is_power_of_two(),
        "Callers expected to pad inputs to a power of two"
    );

    let ncols = num_cols::<WIDTH, SBOX_DEGREE, SBOX_REGISTERS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>();
    let mut vec = Vec::with_capacity((n * ncols) << extra_capacity_bits);
    let trace = &mut vec.spare_capacity_mut()[..n * ncols];
    let trace = RowMajorMatrixViewMut::new(trace, ncols);

    let (prefix, perms, suffix) = unsafe {
        trace.values.align_to_mut::<Poseidon2Cols<
            MaybeUninit<F>,
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

    generate_perms::<
        F,
        LinearLayers,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
    >(perms, inputs, constants);

    unsafe {
        vec.set_len(n * ncols);
    }

    RowMajorMatrix::new(vec, ncols)
}

/// Fill every permutation's trace columns from its input.
///
/// Runs `F::Packing::WIDTH` permutations at a time through the packed round functions when
/// the input count divides evenly into that width, falling back to one permutation at a time
/// (via [`generate_trace_rows_for_perm`]) otherwise.
fn generate_perms<
    F: PrimeField,
    LinearLayers: GenericPoseidon2LinearLayers<WIDTH>,
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
>(
    perms: &mut [Poseidon2Cols<
        MaybeUninit<F>,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
    >],
    inputs: Vec<[F; WIDTH]>,
    constants: &RoundConstants<F, WIDTH, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>,
) {
    let packing_width = F::Packing::WIDTH;
    if packing_width > 1 && inputs.len().is_multiple_of(packing_width) {
        perms
            .par_chunks_mut(packing_width)
            .zip(inputs.par_chunks(packing_width))
            .for_each(|(perm_chunk, input_chunk)| {
                generate_trace_rows_for_perm_batch::<
                    F,
                    LinearLayers,
                    WIDTH,
                    SBOX_DEGREE,
                    SBOX_REGISTERS,
                    HALF_FULL_ROUNDS,
                    PARTIAL_ROUNDS,
                >(perm_chunk, input_chunk, constants);
            });
    } else {
        perms.par_iter_mut().zip(inputs).for_each(|(perm, input)| {
            generate_trace_rows_for_perm::<
                F,
                LinearLayers,
                WIDTH,
                SBOX_DEGREE,
                SBOX_REGISTERS,
                HALF_FULL_ROUNDS,
                PARTIAL_ROUNDS,
            >(perm, input, constants);
        });
    }
}

/// `rows` will normally consist of 24 rows, with an exception for the final row.
pub fn generate_trace_rows_for_perm<
    F: PrimeField,
    LinearLayers: GenericPoseidon2LinearLayers<WIDTH>,
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
>(
    perm: &mut Poseidon2Cols<
        MaybeUninit<F>,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
    >,
    mut state: [F; WIDTH],
    constants: &RoundConstants<F, WIDTH, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>,
) {
    perm.inputs
        .iter_mut()
        .zip(state.iter())
        .for_each(|(input, &x)| {
            input.write(x);
        });

    LinearLayers::external_linear_layer(&mut state);

    for (full_round, constants) in perm
        .beginning_full_rounds
        .iter_mut()
        .zip(&constants.beginning_full_round_constants)
    {
        generate_full_round::<_, LinearLayers, WIDTH, SBOX_DEGREE, SBOX_REGISTERS>(
            &mut state, full_round, constants,
        );
    }

    for (partial_round, constant) in perm
        .partial_rounds
        .iter_mut()
        .zip(&constants.partial_round_constants)
    {
        generate_partial_round::<_, LinearLayers, WIDTH, SBOX_DEGREE, SBOX_REGISTERS>(
            &mut state,
            partial_round,
            *constant,
        );
    }

    for (full_round, constants) in perm
        .ending_full_rounds
        .iter_mut()
        .zip(&constants.ending_full_round_constants)
    {
        generate_full_round::<_, LinearLayers, WIDTH, SBOX_DEGREE, SBOX_REGISTERS>(
            &mut state, full_round, constants,
        );
    }
}

/// Generate `F::Packing::WIDTH` permutations at once, running the round functions over
/// `F::Packing` so that every state element processes all lanes with one field operation
/// instead of one permutation at a time.
#[inline]
fn generate_trace_rows_for_perm_batch<
    F: PrimeField,
    LinearLayers: GenericPoseidon2LinearLayers<WIDTH>,
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
>(
    perms: &mut [Poseidon2Cols<
        MaybeUninit<F>,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
    >],
    inputs: &[[F; WIDTH]],
    constants: &RoundConstants<F, WIDTH, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>,
) {
    let width = F::Packing::WIDTH;
    debug_assert_eq!(perms.len(), width);
    debug_assert_eq!(inputs.len(), width);

    for (perm, input) in perms.iter_mut().zip(inputs) {
        perm.inputs.iter_mut().zip(input).for_each(|(c, &x)| {
            c.write(x);
        });
    }

    let mut state: [F::Packing; WIDTH] = F::Packing::pack_columns(inputs);

    LinearLayers::external_linear_layer(&mut state);

    for (round, round_constants) in constants.beginning_full_round_constants.iter().enumerate() {
        generate_full_round_packed::<
            F,
            LinearLayers,
            WIDTH,
            SBOX_DEGREE,
            SBOX_REGISTERS,
            HALF_FULL_ROUNDS,
            PARTIAL_ROUNDS,
        >(
            &mut state,
            perms,
            |p| &mut p.beginning_full_rounds[round],
            round_constants,
        );
    }

    for (round, &round_constant) in constants.partial_round_constants.iter().enumerate() {
        generate_partial_round_packed::<
            F,
            LinearLayers,
            WIDTH,
            SBOX_DEGREE,
            SBOX_REGISTERS,
            HALF_FULL_ROUNDS,
            PARTIAL_ROUNDS,
        >(&mut state, perms, round, round_constant);
    }

    for (round, round_constants) in constants.ending_full_round_constants.iter().enumerate() {
        generate_full_round_packed::<
            F,
            LinearLayers,
            WIDTH,
            SBOX_DEGREE,
            SBOX_REGISTERS,
            HALF_FULL_ROUNDS,
            PARTIAL_ROUNDS,
        >(
            &mut state,
            perms,
            |p| &mut p.ending_full_rounds[round],
            round_constants,
        );
    }
}

#[inline]
fn generate_full_round_packed<
    F: PrimeField,
    LinearLayers: GenericPoseidon2LinearLayers<WIDTH>,
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
>(
    state: &mut [F::Packing; WIDTH],
    perms: &mut [Poseidon2Cols<
        MaybeUninit<F>,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
    >],
    full_round: impl Fn(
        &mut Poseidon2Cols<
            MaybeUninit<F>,
            WIDTH,
            SBOX_DEGREE,
            SBOX_REGISTERS,
            HALF_FULL_ROUNDS,
            PARTIAL_ROUNDS,
        >,
    ) -> &mut FullRound<MaybeUninit<F>, WIDTH, SBOX_DEGREE, SBOX_REGISTERS>,
    round_constants: &[F; WIDTH],
) {
    let mut sbox_registers = [[F::Packing::ZERO; SBOX_REGISTERS]; WIDTH];
    for ((s, &rc), regs) in state
        .iter_mut()
        .zip(round_constants.iter())
        .zip(sbox_registers.iter_mut())
    {
        *s += rc;
        *regs = generate_sbox_packed::<F, SBOX_DEGREE, SBOX_REGISTERS>(s);
    }

    LinearLayers::external_linear_layer(state);

    for (lane, perm) in perms.iter_mut().enumerate() {
        let full_round = full_round(perm);
        for i in 0..WIDTH {
            for (r, reg) in sbox_registers[i].iter().enumerate() {
                full_round.sbox[i].0[r].write(reg.as_slice()[lane]);
            }
            full_round.post[i].write(state[i].as_slice()[lane]);
        }
    }
}

#[inline]
fn generate_partial_round_packed<
    F: PrimeField,
    LinearLayers: GenericPoseidon2LinearLayers<WIDTH>,
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
>(
    state: &mut [F::Packing; WIDTH],
    perms: &mut [Poseidon2Cols<
        MaybeUninit<F>,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
    >],
    round: usize,
    round_constant: F,
) {
    state[0] += round_constant;
    let sbox_registers = generate_sbox_packed::<F, SBOX_DEGREE, SBOX_REGISTERS>(&mut state[0]);

    for (lane, perm) in perms.iter_mut().enumerate() {
        let partial_round = &mut perm.partial_rounds[round];
        for (r, reg) in sbox_registers.iter().enumerate() {
            partial_round.sbox.0[r].write(reg.as_slice()[lane]);
        }
        partial_round.post_sbox.write(state[0].as_slice()[lane]);
    }

    LinearLayers::internal_linear_layer(state);
}

/// Packed analog of [`generate_sbox`]: computes `x -> x^{DEGREE}` over `F::Packing::WIDTH`
/// permutations at once, returning the packed intermediate registers instead of writing them
/// directly (the caller unpacks them into each lane's own trace columns).
#[inline]
fn generate_sbox_packed<F: PrimeField, const DEGREE: u64, const REGISTERS: usize>(
    x: &mut F::Packing,
) -> [F::Packing; REGISTERS] {
    let mut registers = [F::Packing::ZERO; REGISTERS];
    *x = match (DEGREE, REGISTERS) {
        (3, 0) => x.cube(),
        (5, 0) => x.exp_const_u64::<5>(),
        (7, 0) => x.exp_const_u64::<7>(),
        (5, 1) => {
            let x2 = x.square();
            let x3 = x2 * *x;
            registers[0] = x3;
            x3 * x2
        }
        (7, 1) => {
            let x3 = x.cube();
            registers[0] = x3;
            x3 * x3 * *x
        }
        (11, 2) => {
            let x2 = x.square();
            let x3 = x2 * *x;
            let x9 = x3.cube();
            registers[0] = x3;
            registers[1] = x9;
            x9 * x2
        }
        _ => panic!(
            "Unexpected (DEGREE, REGISTERS) of ({}, {})",
            DEGREE, REGISTERS
        ),
    };
    registers
}

#[inline]
fn generate_full_round<
    F: PrimeField,
    LinearLayers: GenericPoseidon2LinearLayers<WIDTH>,
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
>(
    state: &mut [F; WIDTH],
    full_round: &mut FullRound<MaybeUninit<F>, WIDTH, SBOX_DEGREE, SBOX_REGISTERS>,
    round_constants: &[F; WIDTH],
) {
    // Combine addition of round constants and S-box application in a single loop
    for ((state_i, const_i), sbox_i) in state
        .iter_mut()
        .zip(round_constants.iter())
        .zip(full_round.sbox.iter_mut())
    {
        *state_i += *const_i;
        generate_sbox(sbox_i, state_i);
    }

    LinearLayers::external_linear_layer(state);
    full_round
        .post
        .iter_mut()
        .zip(*state)
        .for_each(|(post, x)| {
            post.write(x);
        });
}

#[inline]
fn generate_partial_round<
    F: PrimeField,
    LinearLayers: GenericPoseidon2LinearLayers<WIDTH>,
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
>(
    state: &mut [F; WIDTH],
    partial_round: &mut PartialRound<MaybeUninit<F>, WIDTH, SBOX_DEGREE, SBOX_REGISTERS>,
    round_constant: F,
) {
    state[0] += round_constant;
    generate_sbox(&mut partial_round.sbox, &mut state[0]);
    partial_round.post_sbox.write(state[0]);
    LinearLayers::internal_linear_layer(state);
}

/// Computes the S-box `x -> x^{DEGREE}` and stores the partial data required to
/// verify the computation.
///
/// # Panics
///
/// This method panics if the number of `REGISTERS` is not chosen optimally for the given
/// `DEGREE` or if the `DEGREE` is not supported by the S-box. The supported degrees are
/// `3`, `5`, `7`, and `11`.
#[inline]
fn generate_sbox<F: PrimeField, const DEGREE: u64, const REGISTERS: usize>(
    sbox: &mut SBox<MaybeUninit<F>, DEGREE, REGISTERS>,
    x: &mut F,
) {
    *x = match (DEGREE, REGISTERS) {
        (3, 0) => x.cube(),
        (5, 0) => x.exp_const_u64::<5>(),
        (7, 0) => x.exp_const_u64::<7>(),
        (5, 1) => {
            let x2 = x.square();
            let x3 = x2 * *x;
            sbox.0[0].write(x3);
            x3 * x2
        }
        (7, 1) => {
            let x3 = x.cube();
            sbox.0[0].write(x3);
            x3 * x3 * *x
        }
        (11, 2) => {
            let x2 = x.square();
            let x3 = x2 * *x;
            let x9 = x3.cube();
            sbox.0[0].write(x3);
            sbox.0[1].write(x9);
            x9 * x2
        }
        _ => panic!(
            "Unexpected (DEGREE, REGISTERS) of ({}, {})",
            DEGREE, REGISTERS
        ),
    }
}
