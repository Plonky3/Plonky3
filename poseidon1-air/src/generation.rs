//! Execution trace generation for the Poseidon1 AIR.
//!
//! # Overview
//!
//! This module computes the execution trace for the Poseidon1 permutation.
//! Each permutation fills one row of the trace matrix.
//!
//! The trace is generated in parallel using Rayon. Each row is computed independently.
//!
//! # Trace Layout
//!
//! ```text
//!   Row 0:  [ inputs | beginning_full_rounds | partial_rounds | ending_full_rounds ]
//!   Row 1:  [ inputs | beginning_full_rounds | partial_rounds | ending_full_rounds ]
//!   ...
//!   Row N-1: [ inputs | beginning_full_rounds | partial_rounds | ending_full_rounds ]
//! ```
//!
//! Each row stores:
//!
//! - The initial state (`inputs`).
//! - S-box intermediates and post-states for full rounds.
//! - S-box outputs for partial rounds (sparse matrix decomposition).
use alloc::vec::Vec;
use core::mem::MaybeUninit;
use p3_mds::karatsuba_convolution::{mds_circulant_karatsuba_16, mds_circulant_karatsuba_24};

use p3_field::{PrimeField, dot_product};
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixViewMut};
use p3_maybe_rayon::prelude::*;
use p3_poseidon1::external::mds_multiply;
use tracing::instrument;

use crate::columns::{Poseidon1Cols, num_cols};
use crate::{FullRound, FullRoundConstants, PartialRound, PartialRoundConstants, SBox};

/// Karatsuba MDS multiply for trace generation.
///
/// Uses Karatsuba convolution for supported widths (16, 24), falling back
/// to dense O(t²) multiplication otherwise. Concrete field types satisfy
/// the `Copy` bound required by the Karatsuba implementation.
#[inline]
fn mds_for_trace_gen<F: PrimeField, const WIDTH: usize>(
    state: &mut [F; WIDTH],
    circ_col: &[F; WIDTH],
    dense_mds: &[[F; WIDTH]; WIDTH],
) {
    match WIDTH {
        16 => {
            let state_16: &mut [F; 16] = state.as_mut_slice().try_into().unwrap();
            let col_16: &[F; 16] = circ_col.as_slice().try_into().unwrap();
            mds_circulant_karatsuba_16(state_16, col_16);
        }
        24 => {
            let state_24: &mut [F; 24] = state.as_mut_slice().try_into().unwrap();
            let col_24: &[F; 24] = circ_col.as_slice().try_into().unwrap();
            mds_circulant_karatsuba_24(state_24, col_24);
        }
        _ => mds_multiply(state, dense_mds),
    }
}

/// Generate a trace for multiple Poseidon1 permutations (vectorized layout).
///
/// This variant packs `VECTOR_LEN` permutations into each trace row,
/// increasing the row width by a factor of `VECTOR_LEN`. This is useful
/// when the per-permutation column count is small relative to the desired
/// row width.
///
/// # Arguments
///
/// - `inputs`: A vector of `[F; WIDTH]` initial states, one per permutation.
/// - `round_constants`: Pre-computed AIR round constants.
/// - `extra_capacity_bits`: Log2 of the extra allocation factor for LDE blowup.
///   E.g., `extra_capacity_bits = 2` allocates `4x` the base size.
///
/// # Panics
///
/// Panics if `inputs.len()` is not a multiple of `VECTOR_LEN`, or if
/// `inputs.len() / VECTOR_LEN` is not a power of two.
#[instrument(name = "generate vectorized Poseidon1 trace", skip_all)]
pub fn generate_vectorized_trace_rows<
    F: PrimeField,
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
    const VECTOR_LEN: usize,
>(
    inputs: Vec<[F; WIDTH]>,
    full_constants: &FullRoundConstants<F, WIDTH>,
    partial_constants: &PartialRoundConstants<F, WIDTH>,
    extra_capacity_bits: usize,
) -> RowMajorMatrix<F> {
    let n = inputs.len();
    assert!(
        n.is_multiple_of(VECTOR_LEN) && (n / VECTOR_LEN).is_power_of_two(),
        "Callers expected to pad inputs to VECTOR_LEN times a power of two"
    );

    // Number of rows = total permutations / permutations per row.
    let nrows = n.div_ceil(VECTOR_LEN);

    // Row width = columns per permutation × permutations per row.
    let ncols = num_cols::<WIDTH, SBOX_DEGREE, SBOX_REGISTERS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>()
        * VECTOR_LEN;

    // Allocate with extra capacity for LDE blowup (avoids reallocation during proving).
    let mut vec = Vec::with_capacity((nrows * ncols) << extra_capacity_bits);

    // Use `spare_capacity_mut` to get uninitialized memory without zeroing.
    let trace = &mut vec.spare_capacity_mut()[..nrows * ncols];
    let trace = RowMajorMatrixViewMut::new(trace, ncols);

    // Reinterpret the flat MaybeUninit<F> slice as an array of Poseidon1Cols structs.
    //
    // Each row contains VECTOR_LEN consecutive Poseidon1Cols.
    let (prefix, perms, suffix) = unsafe {
        trace.values.align_to_mut::<Poseidon1Cols<
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

    // Derive circulant column from dense MDS (first column of circulant matrix).
    let circ_col: [F; WIDTH] = core::array::from_fn(|i| full_constants.dense_mds[i][0]);

    // Compute each permutation in parallel (one Poseidon1Cols struct per permutation).
    perms.par_iter_mut().zip(inputs).for_each(|(perm, input)| {
        generate_trace_rows_for_perm::<
            F,
            WIDTH,
            SBOX_DEGREE,
            SBOX_REGISTERS,
            HALF_FULL_ROUNDS,
            PARTIAL_ROUNDS,
        >(perm, input, full_constants, partial_constants, &circ_col);
    });

    // All elements have been written; mark the Vec as initialized.
    unsafe {
        vec.set_len(nrows * ncols);
    }

    RowMajorMatrix::new(vec, ncols)
}

/// Generate a trace for multiple Poseidon1 permutations (one per row).
///
/// Each row of the returned matrix represents one complete permutation.
///
/// # Arguments
///
/// - `inputs`: A vector of `[F; WIDTH]` initial states.
///   Length must be a power of two.
/// - `constants`: Pre-computed AIR round constants.
/// - `extra_capacity_bits`: Log2 of the extra allocation factor for LDE blowup.
///
/// # Panics
///
/// Panics if `inputs.len()` is not a power of two.
#[instrument(name = "generate Poseidon1 trace", skip_all)]
pub fn generate_trace_rows<
    F: PrimeField,
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
>(
    inputs: Vec<[F; WIDTH]>,
    full_constants: &FullRoundConstants<F, WIDTH>,
    partial_constants: &PartialRoundConstants<F, WIDTH>,
    extra_capacity_bits: usize,
) -> RowMajorMatrix<F> {
    let n = inputs.len();
    assert!(
        n.is_power_of_two(),
        "Callers expected to pad inputs to a power of two"
    );

    // One permutation per row.
    let ncols = num_cols::<WIDTH, SBOX_DEGREE, SBOX_REGISTERS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>();

    // Allocate with extra capacity for LDE blowup.
    let mut vec = Vec::with_capacity((n * ncols) << extra_capacity_bits);

    // Get uninitialized memory without zeroing.
    let trace = &mut vec.spare_capacity_mut()[..n * ncols];
    let trace = RowMajorMatrixViewMut::new(trace, ncols);

    // Reinterpret as an array of Poseidon1Cols structs (one per row).
    let (prefix, perms, suffix) = unsafe {
        trace.values.align_to_mut::<Poseidon1Cols<
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

    // Derive circulant column from dense MDS (first column of circulant matrix).
    let circ_col: [F; WIDTH] = core::array::from_fn(|i| full_constants.dense_mds[i][0]);

    // Compute each permutation in parallel.
    perms.par_iter_mut().zip(inputs).for_each(|(perm, input)| {
        generate_trace_rows_for_perm::<
            F,
            WIDTH,
            SBOX_DEGREE,
            SBOX_REGISTERS,
            HALF_FULL_ROUNDS,
            PARTIAL_ROUNDS,
        >(perm, input, full_constants, partial_constants, &circ_col);
    });

    // All elements have been written; mark the Vec as initialized.
    unsafe {
        vec.set_len(n * ncols);
    }

    RowMajorMatrix::new(vec, ncols)
}

/// Generate the trace for a single Poseidon1 permutation.
///
/// Fills one `Poseidon1Cols` struct with the inputs, round intermediates,
/// and post-states for every round of the permutation.
///
/// # Execution Flow
///
/// ```text
///   1. Write inputs
///   2. Beginning full rounds (RF/2 rounds)
///   3. Sparse partial rounds (first-round constants, m_i, then loop)
///   4. Ending full rounds (RF/2 rounds)
/// ```
///
/// The `state` array tracks the live permutation state and is modified in
/// place at each step. The committed columns (`sbox`, `post`, `post_sbox`)
/// are written as side effects.
pub fn generate_trace_rows_for_perm<
    F: PrimeField,
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
>(
    perm: &mut Poseidon1Cols<
        MaybeUninit<F>,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
    >,
    mut state: [F; WIDTH],
    full_constants: &FullRoundConstants<F, WIDTH>,
    partial_constants: &PartialRoundConstants<F, WIDTH>,
    circ_col: &[F; WIDTH],
) {
    // Step 1: Write the initial state into the `inputs` columns.
    perm.inputs
        .iter_mut()
        .zip(state.iter())
        .for_each(|(input, &x)| {
            input.write(x);
        });

    // Step 2: Beginning full rounds (RF/2 rounds).
    for (full_round, rc) in perm
        .beginning_full_rounds
        .iter_mut()
        .zip(&full_constants.initial)
    {
        generate_full_round::<_, WIDTH, SBOX_DEGREE, SBOX_REGISTERS>(
            &mut state,
            full_round,
            rc,
            &circ_col,
            &full_constants.dense_mds,
        );
    }

    // Step 3: Sparse partial rounds.
    // Add first-round constants.
    for (s, &c) in state
        .iter_mut()
        .zip(partial_constants.first_round_constants.iter())
    {
        *s += c;
    }
    // Dense transition matrix m_i (once).
    mds_multiply(&mut state, &partial_constants.m_i);

    // Partial round loop.
    let rounds_p = partial_constants.sparse_first_row.len();
    for round in 0..rounds_p {
        let rc = if round < rounds_p - 1 {
            Some(partial_constants.round_constants[round])
        } else {
            None
        };
        generate_sparse_partial_round::<F, WIDTH, SBOX_DEGREE, SBOX_REGISTERS>(
            &mut state,
            &mut perm.partial_rounds[round],
            rc,
            &partial_constants.sparse_first_row[round],
            &partial_constants.v[round],
        );
    }

    // Step 4: Ending full rounds (RF/2 rounds).
    for (full_round, rc) in perm
        .ending_full_rounds
        .iter_mut()
        .zip(&full_constants.terminal)
    {
        generate_full_round::<_, WIDTH, SBOX_DEGREE, SBOX_REGISTERS>(
            &mut state,
            full_round,
            rc,
            &circ_col,
            &full_constants.dense_mds,
        );
    }
}

/// Execute one full round and write the trace columns.
///
/// For each state element:
/// 1. Add the round constant.
/// 2. Compute the S-box and write intermediates.
///
/// Then multiply by the MDS matrix (using Karatsuba for WIDTH 16 or 24)
/// and write the post-state.
#[inline]
fn generate_full_round<
    F: PrimeField,
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
>(
    state: &mut [F; WIDTH],
    full_round: &mut FullRound<MaybeUninit<F>, WIDTH, SBOX_DEGREE, SBOX_REGISTERS>,
    round_constants: &[F; WIDTH],
    circ_col: &[F; WIDTH],
    dense_mds: &[[F; WIDTH]; WIDTH],
) {
    for ((state_i, const_i), sbox_i) in state
        .iter_mut()
        .zip(round_constants.iter())
        .zip(full_round.sbox.iter_mut())
    {
        // state[i] += rc[i].
        *state_i += *const_i;

        // Compute state[i] = state[i]^DEGREE and write S-box intermediates.
        generate_sbox(sbox_i, state_i);
    }

    // MDS multiply: state = MDS * state.
    // Karatsuba MDS for supported widths, dense fallback otherwise.
    mds_for_trace_gen(state, circ_col, dense_mds);

    // Write the post-state to the trace.
    full_round
        .post
        .iter_mut()
        .zip(*state)
        .for_each(|(post, x)| {
            post.write(x);
        });
}

/// Execute one sparse partial round and write the trace columns.
///
/// 1. S-box on `state[0]` and write intermediates.
/// 2. Write the committed S-box output.
/// 3. Add scalar round constant (if present).
/// 4. Sparse matrix multiply: dot product for new `state[0]`, rank-1 update for rest.
#[inline]
fn generate_sparse_partial_round<
    F: PrimeField,
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
>(
    state: &mut [F; WIDTH],
    partial_round: &mut PartialRound<MaybeUninit<F>, SBOX_DEGREE, SBOX_REGISTERS>,
    round_constant: Option<F>,
    first_row: &[F; WIDTH],
    v: &[F; WIDTH],
) {
    // S-box on state[0].
    generate_sbox(&mut partial_round.sbox, &mut state[0]);

    // Write the committed S-box output.
    partial_round.post_sbox.write(state[0]);

    // Add scalar round constant (if present).
    if let Some(rc) = round_constant {
        state[0] += rc;
    }

    // Sparse matrix multiply.
    let old_s0 = state[0];
    state[0] = dot_product(state.iter().copied(), first_row.iter().copied());
    for i in 1..WIDTH {
        state[i] += old_s0 * v[i - 1];
    }
}

/// Compute the S-box `x → x^DEGREE` and write intermediate values to the trace.
///
/// The intermediate values are the same ones constrained by `eval_sbox` in the
/// AIR. The prover computes them here; the verifier checks them via constraints.
///
/// # Supported Configurations
///
/// | DEGREE | REGISTERS | Intermediates Written | Output                 |
/// |--------|-----------|-----------------------|------------------------|
/// | 3      | 0         | (none)                | `x^3`                  |
/// | 5      | 0         | (none)                | `x^5`                  |
/// | 7      | 0         | (none)                | `x^7`                  |
/// | 5      | 1         | `x^3`                 | `x^3 * x^2 = x^5`      |
/// | 7      | 1         | `x^3`                 | `x^3 * x^3 * x = x^7`  |
/// | 11     | 1         | `x^3`                 | `(x^3)^3 * x^2 = x^11` |
/// | 11     | 2         | `x^3`, `x^9`          | `x^9 * x^2 = x^11`     |
///
/// # Panics
///
/// Panics if `(DEGREE, REGISTERS)` is not one of the supported configurations.
#[inline]
fn generate_sbox<F: PrimeField, const DEGREE: u64, const REGISTERS: usize>(
    sbox: &mut SBox<MaybeUninit<F>, DEGREE, REGISTERS>,
    x: &mut F,
) {
    *x = match (DEGREE, REGISTERS) {
        // Direct computation: no intermediates needed.
        (3, 0) => x.cube(),
        (5, 0) => x.exp_const_u64::<5>(),
        (7, 0) => x.exp_const_u64::<7>(),

        // x^5 with one intermediate (x^3).
        (5, 1) => {
            let x2 = x.square();
            let x3 = x2 * *x;
            sbox.0[0].write(x3);
            x3 * x2
        }

        // x^7 with one intermediate (x^3).
        (7, 1) => {
            let x3 = x.cube();
            sbox.0[0].write(x3);
            x3 * x3 * *x
        }

        // x^11 with one intermediate (x^3).
        // Output: (x^3)^3 * x^2 = x^11.
        // Constraint degree: max(3, 5) = 5.
        (11, 1) => {
            let x3 = x.cube();
            sbox.0[0].write(x3);
            x3.cube() * x.square()
        }

        // x^11 with two intermediates (x^3, x^9).
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
