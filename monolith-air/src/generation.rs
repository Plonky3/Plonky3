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
use p3_mds::util::mds_multiply;
use p3_monolith::MonolithBars;
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
            air.limb_bits,
            &air.modulus_lsb_to_msb,
            bars,
        );
    }

    // Step 4: Final round (no round constants).
    generate_round::<F, B, WIDTH, NUM_BARS, FIELD_BITS>(
        &mut state,
        &mut perm.final_round,
        &air.mds_matrix,
        None,
        air.limb_bits,
        &air.modulus_lsb_to_msb,
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
    limb_bits: &[usize],
    modulus_lsb_to_msb: &[bool; FIELD_BITS],
    bars: &B,
) {
    // Phase 1: Bars witnesses (per Bar slot):
    //
    //     - little-endian bit decomposition of the input
    //     - per-bit chi AND product (splits the S-box into degree-3 pieces)
    //     - canonical-pattern walk flags (rules out any encoding `>= p`)
    for (bar_idx, ((bar_bits, bar_chi), bar_mflag)) in round
        .bars_input_bits
        .iter_mut()
        .zip(round.bars_chi_products.iter_mut())
        .zip(round.bars_match_flags.iter_mut())
        .enumerate()
    {
        // Snapshot bits before Bars overwrites the state.
        let bits = decompose_to_bits::<F, FIELD_BITS>(state[bar_idx]);
        for (bit_col, &bit_val) in bar_bits.iter_mut().zip(bits.iter()) {
            bit_col.write(bit_val);
        }

        // Per-bit AND products, walking limbs left to right.
        let mut bit_offset = 0;
        for (limb_idx, &n) in limb_bits.iter().enumerate() {
            // Only the trailing limb may be narrower than 8 bits.
            let is_last_reduced = limb_idx == limb_bits.len() - 1 && n < 8;
            let limb = &bits[bit_offset..bit_offset + n];

            //     sub(k) = (j - k) mod n
            for j in 0..n {
                let sub = |offset: usize| (j + n - (offset % n)) % n;

                //     andn   = (1 - x[j-2]) * x[j-3]
                //     8-bit  : chi = andn * x[j-4]
                //     7-bit  : chi = andn
                let andn = limb[sub(2)].andn(&limb[sub(3)]);
                let chi_product = if is_last_reduced {
                    andn
                } else {
                    andn * limb[sub(4)]
                };
                bar_chi[bit_offset + j].write(chi_product);
            }
            bit_offset += n;
        }

        // Canonical-pattern walk (MSB → LSB):
        //
        //     prev = 1
        //     for i from FIELD_BITS-1 down to 0:
        //         if p_bit[i]: prev *= bits[i]
        //         else       : prev stays                 (and prev * bits[i] must be 0)
        //         match_flag[i] = prev
        //
        // Canonical inputs always end with prev = 0 (some bit position must
        // make them strictly less than p).
        let mut prev = F::ONE;
        for i in (0..FIELD_BITS).rev() {
            if modulus_lsb_to_msb[i] {
                prev *= bits[i];
            }
            bar_mflag[i].write(prev);
        }
    }

    // Phase 2: Bars S-box (overwrites positions 0..u; u..t pass through).
    bars.bars(state);
    for (bar_out, &el) in round.bars_output.iter_mut().zip(state.iter()) {
        bar_out.write(el);
    }

    // Phase 3: Bricks (Feistel squaring).
    //
    //     state[i] += state[i-1]^2   for i = t-1 down to 1
    //
    // Reverse order keeps each step reading the pre-update predecessor.
    for i in (1..WIDTH).rev() {
        state[i] += state[i - 1].square();
    }

    // Phase 4: Concrete (dense MDS image).
    mds_multiply(state, mds_matrix);

    // Phase 5: Round constants (skipped on the final round).
    if let Some(rc) = round_constants {
        for (s, &c) in state.iter_mut().zip(rc.iter()) {
            *s += c;
        }
    }

    // Phase 6: Persist the post-state.
    round
        .post
        .iter_mut()
        .zip(state.iter())
        .for_each(|(col, &val)| {
            col.write(val);
        });
}

/// Little-endian bit decomposition of a prime-field element.
///
/// Each output cell carries a field element equal to 0 or 1.
///
/// # Panics
///
/// Compile-time panic when the requested bit length exceeds 64.
#[inline]
pub(crate) fn decompose_to_bits<F: PrimeField64, const FIELD_BITS: usize>(
    element: F,
) -> [F; FIELD_BITS] {
    // Cap: the canonical form is a `u64`, so it has at most 64 bit positions.
    const { assert!(FIELD_BITS <= 64, "FIELD_BITS must fit in a u64") };

    // Canonical integer image; bit i sits at position i.
    let val = element.as_canonical_u64();
    core::array::from_fn(|i| F::from_bool(((val >> i) & 1) != 0))
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
