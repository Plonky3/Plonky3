//! Execution trace generation for the Monolith AIR.
//!
//! # Overview
//!
//! This module computes the execution trace for the Monolith permutation.
//! Each permutation fills one row of the trace matrix.
//!
//! The trace is generated in parallel using Rayon. Each row is computed
//! independently.
//!
//! # Trace Layout
//!
//! ```text
//!   Row 0:  [ inputs | full_rounds[0..R] | final_round ]
//!   Row 1:  [ inputs | full_rounds[0..R] | final_round ]
//!   ...
//!   Row N-1: [ inputs | full_rounds[0..R] | final_round ]
//! ```
//!
//! Each row stores:
//!
//! - The initial state (`inputs`).
//! - Per round: bit decompositions of Bar inputs, committed Bar outputs,
//!   and the post-state after Bricks + Concrete + RC.

use alloc::vec::Vec;
use core::mem::MaybeUninit;

use p3_field::PrimeField64;
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixViewMut};
use p3_maybe_rayon::prelude::*;
use p3_monolith::MonolithBars;
use p3_poseidon1::external::mds_multiply;
use tracing::instrument;

use crate::MonolithAir;
use crate::columns::{MonolithCols, MonolithRoundCols, num_cols};

/// Generate a trace for multiple Monolith permutations (one per row).
///
/// Each row of the returned matrix represents one complete permutation.
///
/// # Arguments
///
/// - `inputs`: A vector of `[F; WIDTH]` initial states. Length must be a power of two.
/// - `air`: The Monolith AIR instance containing round constants and MDS matrix.
/// - `bars`: The field-specific Bars layer implementation.
/// - `extra_capacity_bits`: Log2 of the extra allocation factor for LDE blowup.
///   E.g., `extra_capacity_bits = 2` allocates `4x` the base size.
///
/// # Panics
///
/// Panics if `inputs.len()` is not a power of two.
#[instrument(name = "generate Monolith trace", skip_all)]
pub fn generate_trace_rows<
    F: PrimeField64,
    B: MonolithBars<F, WIDTH> + Sync,
    const WIDTH: usize,
    const NUM_FULL_ROUNDS: usize,
    const NUM_BARS: usize,
    const FIELD_BITS: usize,
>(
    inputs: Vec<[F; WIDTH]>,
    air: &MonolithAir<F, WIDTH, NUM_FULL_ROUNDS, NUM_BARS, FIELD_BITS>,
    bars: &B,
    extra_capacity_bits: usize,
) -> RowMajorMatrix<F> {
    let n = inputs.len();
    assert!(
        n.is_power_of_two(),
        "Callers expected to pad inputs to a power of two"
    );

    // One permutation per row.
    let ncols = num_cols::<WIDTH, NUM_FULL_ROUNDS, NUM_BARS, FIELD_BITS>();

    // Allocate with extra capacity for LDE blowup.
    let mut vec = Vec::with_capacity((n * ncols) << extra_capacity_bits);

    // Get uninitialized memory without zeroing.
    let trace = &mut vec.spare_capacity_mut()[..n * ncols];
    let trace = RowMajorMatrixViewMut::new(trace, ncols);

    // Reinterpret as an array of MonolithCols structs (one per row).
    let (prefix, perms, suffix) = unsafe {
        trace.values.align_to_mut::<MonolithCols<
            MaybeUninit<F>,
            WIDTH,
            NUM_FULL_ROUNDS,
            NUM_BARS,
            FIELD_BITS,
        >>()
    };
    assert!(prefix.is_empty(), "Alignment should match");
    assert!(suffix.is_empty(), "Alignment should match");
    assert_eq!(perms.len(), n);

    // Compute each permutation in parallel.
    perms.par_iter_mut().zip(inputs).for_each(|(perm, input)| {
        generate_trace_rows_for_perm(perm, input, air, bars);
    });

    // All elements have been written; mark the Vec as initialized.
    unsafe {
        vec.set_len(n * ncols);
    }

    RowMajorMatrix::new(vec, ncols)
}

/// Generate the trace for a single Monolith permutation.
///
/// Fills one `MonolithCols` struct with the inputs, round intermediates,
/// and post-states for every round of the permutation.
///
/// # Execution Flow
///
/// ```text
///   1. Write inputs
///   2. Apply initial Concrete (MDS multiply)
///   3. For each full round (NUM_FULL_ROUNDS rounds):
///      a. Save pre-Bars state, decompose into bits
///      b. Apply Bars (S-box on first NUM_BARS elements)
///      c. Apply Bricks (Feistel Type-3 with squaring)
///      d. Apply Concrete (MDS multiply)
///      e. Add round constants
///      f. Write post-state
///   4. Final round (same as above, but no round constants)
/// ```
pub fn generate_trace_rows_for_perm<
    F: PrimeField64,
    B: MonolithBars<F, WIDTH>,
    const WIDTH: usize,
    const NUM_FULL_ROUNDS: usize,
    const NUM_BARS: usize,
    const FIELD_BITS: usize,
>(
    perm: &mut MonolithCols<MaybeUninit<F>, WIDTH, NUM_FULL_ROUNDS, NUM_BARS, FIELD_BITS>,
    mut state: [F; WIDTH],
    air: &MonolithAir<F, WIDTH, NUM_FULL_ROUNDS, NUM_BARS, FIELD_BITS>,
    bars: &B,
) {
    // Step 1: Write the initial state into the `inputs` columns.
    perm.inputs
        .iter_mut()
        .zip(state.iter())
        .for_each(|(col, &val)| {
            col.write(val);
        });

    // Step 2: Apply the initial Concrete layer (MDS multiply, no constants).
    mds_multiply(&mut state, &air.mds_matrix);

    // Step 3: Full rounds (NUM_FULL_ROUNDS rounds with round constants).
    for (round_idx, round_cols) in perm.full_rounds.iter_mut().enumerate() {
        generate_round::<F, B, WIDTH, NUM_BARS, FIELD_BITS>(
            &mut state,
            round_cols,
            &air.mds_matrix,
            Some(&air.round_constants[round_idx]),
            bars,
        );
    }

    // Step 4: Final round (no round constants).
    generate_round::<F, B, WIDTH, NUM_BARS, FIELD_BITS>(
        &mut state,
        &mut perm.final_round,
        &air.mds_matrix,
        None,
        bars,
    );
}

/// Execute one Monolith round and write the trace columns.
///
/// Applies: Bars → Bricks → Concrete → (optional) AddRoundConstants.
///
/// Writes the bit decomposition of Bar inputs, the committed Bar outputs,
/// and the post-state to the trace columns.
#[inline]
fn generate_round<
    F: PrimeField64,
    B: MonolithBars<F, WIDTH>,
    const WIDTH: usize,
    const NUM_BARS: usize,
    const FIELD_BITS: usize,
>(
    state: &mut [F; WIDTH],
    round: &mut MonolithRoundCols<MaybeUninit<F>, WIDTH, NUM_BARS, FIELD_BITS>,
    mds_matrix: &[[F; WIDTH]; WIDTH],
    round_constants: Option<&[F; WIDTH]>,
    bars: &B,
) {
    // Bars layer

    // Decompose each of the NUM_BARS elements into bits (before applying Bars).
    for (bar_bits, &el) in round.bars_input_bits.iter_mut().zip(state.iter()) {
        let bits = decompose_to_bits::<F, FIELD_BITS>(el);
        for (bit_col, &bit_val) in bar_bits.iter_mut().zip(bits.iter()) {
            bit_col.write(bit_val);
        }
    }

    // Apply the Bars S-box (modifies the first NUM_BARS elements in place).
    bars.bars(state);

    // Write the committed Bar outputs.
    for (bar_out, &el) in round.bars_output.iter_mut().zip(state.iter()) {
        bar_out.write(el);
    }

    // Bricks layer
    //
    // Feistel Type-3: state[i] += state[i-1]^2 for i = WIDTH-1 down to 1.
    // Reverse iteration avoids reading already-modified values.
    for i in (1..WIDTH).rev() {
        state[i] += state[i - 1].square();
    }

    // Concrete layer
    mds_multiply(state, mds_matrix);

    // Round constants (if not the final round)
    if let Some(rc) = round_constants {
        for (s, &c) in state.iter_mut().zip(rc.iter()) {
            *s += c;
        }
    }

    // Write post-state
    round
        .post
        .iter_mut()
        .zip(state.iter())
        .for_each(|(col, &val)| {
            col.write(val);
        });
}

/// Decompose a prime field element into `FIELD_BITS` bits (LSB first).
///
/// Each bit is a field element equal to 0 or 1. Uses the canonical `u64`
/// representative (sufficient for Monolith-31 and Monolith-64).
#[inline]
pub(crate) fn decompose_to_bits<F: PrimeField64, const FIELD_BITS: usize>(
    element: F,
) -> [F; FIELD_BITS] {
    let val = element.as_canonical_u64();
    core::array::from_fn(|i| {
        if i < FIELD_BITS && i < 64 {
            F::from_bool(((val >> i) & 1) != 0)
        } else {
            F::ZERO
        }
    })
}

#[cfg(test)]
mod tests {
    use p3_field::{PrimeCharacteristicRing, PrimeField32};
    use p3_mersenne_31::Mersenne31;
    use proptest::prelude::*;

    use super::*;

    type F = Mersenne31;

    proptest! {
        /// Every bit produced by `decompose_to_bits` is boolean.
        #[test]
        fn proptest_decompose_bits_are_boolean(val in 0u32..Mersenne31::ORDER_U32) {
            let bits = decompose_to_bits::<F, 31>(F::from_u32(val));
            for &b in &bits {
                let v = b.as_canonical_u32();
                prop_assert!(v == 0 || v == 1);
            }
        }

        /// Reconstructing from bits recovers the original element.
        #[test]
        fn proptest_decompose_round_trip(val in 0u32..Mersenne31::ORDER_U32) {
            let bits = decompose_to_bits::<F, 31>(F::from_u32(val));

            let mut reconstructed = 0u32;
            for (i, &b) in bits.iter().enumerate() {
                reconstructed |= b.as_canonical_u32() << i;
            }
            prop_assert_eq!(reconstructed, val);
        }

        /// `decompose_to_bits` matches direct bit extraction from the integer.
        #[test]
        fn proptest_decompose_matches_integer_bits(val in 0u32..Mersenne31::ORDER_U32) {
            let bits = decompose_to_bits::<F, 31>(F::from_u32(val));

            for (i, &b) in bits.iter().enumerate() {
                prop_assert_eq!(b.as_canonical_u32(), (val >> i) & 1);
            }
        }
    }
}
