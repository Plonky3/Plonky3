//! ARM assembly primitives for the Poseidon1 permutation over Goldilocks.

use super::utils::{add_asm, mul_add_asm, mul_asm};

// ---------------------------------------------------------------------------
// S-box: x -> x^7 (applied to the first element only)
// ---------------------------------------------------------------------------

/// Apply the degree-7 S-box to the first element of the state.
///
/// Computes `x^7` using four multiplications via the addition chain:
///
/// ```text
///     x -> x^2 -> x^3 (= x^2 * x)
///                 x^4 (= x^2 * x^2)
///                 x^7 (= x^3 * x^4)
/// ```
///
/// Only the first element is modified. All other elements are unchanged.
/// This corresponds to the non-linear step of a **partial round**.
#[inline(always)]
pub unsafe fn sbox_s0_asm(state: &mut [u64]) {
    unsafe {
        // Load the first element.
        let s0 = state[0];

        // Square: x^2.
        let s0_2 = mul_asm(s0, s0);

        // Cube: x^3 = x^2 * x.
        let s0_3 = mul_asm(s0_2, s0);

        // Fourth power: x^4 = x^2 * x^2.
        let s0_4 = mul_asm(s0_2, s0_2);

        // Seventh power: x^7 = x^3 * x^4.
        state[0] = mul_asm(s0_3, s0_4);
    }
}

/// Dual-lane S-box on the first element of two independent states.
///
/// Applies the same degree-7 S-box to both first elements. Interleaving
/// the two chains hides the multiplication latency: while one multiply
/// retires, the other is already in flight.
#[inline(always)]
pub unsafe fn sbox_s0_dual_asm(state0: &mut [u64], state1: &mut [u64]) {
    unsafe {
        // Load both first elements.
        let a = state0[0];
        let b = state1[0];

        // Square both.
        let a2 = mul_asm(a, a);
        let b2 = mul_asm(b, b);

        // Cube both: x^3 = x^2 * x.
        let a3 = mul_asm(a2, a);
        let b3 = mul_asm(b2, b);

        // Fourth power both: x^4 = x^2 * x^2.
        let a4 = mul_asm(a2, a2);
        let b4 = mul_asm(b2, b2);

        // Seventh power both: x^7 = x^3 * x^4.
        state0[0] = mul_asm(a3, a4);
        state1[0] = mul_asm(b3, b4);
    }
}

// ---------------------------------------------------------------------------
// Sparse matrix-vector multiply (partial-round linear layer)
// ---------------------------------------------------------------------------

/// Sparse matrix-vector multiply for a width-8 state.
///
/// Implements the partial-round linear layer. The sparse matrix is
/// encoded as its first row and a sub-diagonal vector:
///
/// ```text
///     new[0]  = dot(first_row, state)            (dot product)
///     new[i]  = state[i] + state[0] * v[i-1]   (for i >= 1)
/// ```
///
/// The original first element is captured before the dot product
/// overwrites it. The unrolled form avoids loop overhead and gives
/// the scheduler maximum freedom to reorder independent multiply-adds.
#[inline(always)]
pub unsafe fn cheap_matmul_asm_w8(state: &mut [u64; 8], first_row: &[u64; 8], v: &[u64; 8]) {
    unsafe {
        // Capture the original first element before it gets overwritten.
        let old_s0 = state[0];

        // Dot product: accumulate dot(first_row, state).
        let mut acc = mul_asm(state[0], first_row[0]);
        acc = mul_add_asm(state[1], first_row[1], acc);
        acc = mul_add_asm(state[2], first_row[2], acc);
        acc = mul_add_asm(state[3], first_row[3], acc);
        acc = mul_add_asm(state[4], first_row[4], acc);
        acc = mul_add_asm(state[5], first_row[5], acc);
        acc = mul_add_asm(state[6], first_row[6], acc);
        acc = mul_add_asm(state[7], first_row[7], acc);

        // Tail update: each remaining element gets old_first * v[i-1] added.
        state[1] = mul_add_asm(old_s0, v[0], state[1]);
        state[2] = mul_add_asm(old_s0, v[1], state[2]);
        state[3] = mul_add_asm(old_s0, v[2], state[3]);
        state[4] = mul_add_asm(old_s0, v[3], state[4]);
        state[5] = mul_add_asm(old_s0, v[4], state[5]);
        state[6] = mul_add_asm(old_s0, v[5], state[6]);
        state[7] = mul_add_asm(old_s0, v[6], state[7]);

        // Write the dot-product result into the first slot.
        state[0] = acc;
    }
}

/// Sparse matrix-vector multiply for a width-12 state.
///
/// Same decomposition as the width-8 variant:
/// - Dot product for the new first element.
/// - Scalar multiply-add for every other element.
#[inline(always)]
pub unsafe fn cheap_matmul_asm_w12(state: &mut [u64; 12], first_row: &[u64; 12], v: &[u64; 12]) {
    unsafe {
        // Capture the original first element before it gets overwritten.
        let old_s0 = state[0];

        // Dot product: accumulate dot(first_row, state).
        let mut acc = mul_asm(state[0], first_row[0]);
        acc = mul_add_asm(state[1], first_row[1], acc);
        acc = mul_add_asm(state[2], first_row[2], acc);
        acc = mul_add_asm(state[3], first_row[3], acc);
        acc = mul_add_asm(state[4], first_row[4], acc);
        acc = mul_add_asm(state[5], first_row[5], acc);
        acc = mul_add_asm(state[6], first_row[6], acc);
        acc = mul_add_asm(state[7], first_row[7], acc);
        acc = mul_add_asm(state[8], first_row[8], acc);
        acc = mul_add_asm(state[9], first_row[9], acc);
        acc = mul_add_asm(state[10], first_row[10], acc);
        acc = mul_add_asm(state[11], first_row[11], acc);

        // Tail update: each remaining element gets old_first * v[i-1] added.
        state[1] = mul_add_asm(old_s0, v[0], state[1]);
        state[2] = mul_add_asm(old_s0, v[1], state[2]);
        state[3] = mul_add_asm(old_s0, v[2], state[3]);
        state[4] = mul_add_asm(old_s0, v[3], state[4]);
        state[5] = mul_add_asm(old_s0, v[4], state[5]);
        state[6] = mul_add_asm(old_s0, v[5], state[6]);
        state[7] = mul_add_asm(old_s0, v[6], state[7]);
        state[8] = mul_add_asm(old_s0, v[7], state[8]);
        state[9] = mul_add_asm(old_s0, v[8], state[9]);
        state[10] = mul_add_asm(old_s0, v[9], state[10]);
        state[11] = mul_add_asm(old_s0, v[10], state[11]);

        // Write the dot-product result into the first slot.
        state[0] = acc;
    }
}

/// Dual-lane sparse matrix-vector multiply for a width-8 state.
///
/// Processes two independent states through the same sparse matrix
/// simultaneously. Both lanes share the same first-row and sub-diagonal
/// vectors, since the matrix is fixed for a given partial round.
///
/// Interleaving multiply-adds from both lanes keeps the pipeline full.
#[inline(always)]
pub unsafe fn cheap_matmul_dual_asm_w8(
    s0: &mut [u64; 8],
    s1: &mut [u64; 8],
    first_row: &[u64; 8],
    v: &[u64; 8],
) {
    unsafe {
        // Capture the original first elements from both lanes.
        let old_a = s0[0];
        let old_b = s1[0];

        // Dot products: one per lane, interleaved.
        let mut acc_a = mul_asm(s0[0], first_row[0]);
        let mut acc_b = mul_asm(s1[0], first_row[0]);
        acc_a = mul_add_asm(s0[1], first_row[1], acc_a);
        acc_b = mul_add_asm(s1[1], first_row[1], acc_b);
        acc_a = mul_add_asm(s0[2], first_row[2], acc_a);
        acc_b = mul_add_asm(s1[2], first_row[2], acc_b);
        acc_a = mul_add_asm(s0[3], first_row[3], acc_a);
        acc_b = mul_add_asm(s1[3], first_row[3], acc_b);
        acc_a = mul_add_asm(s0[4], first_row[4], acc_a);
        acc_b = mul_add_asm(s1[4], first_row[4], acc_b);
        acc_a = mul_add_asm(s0[5], first_row[5], acc_a);
        acc_b = mul_add_asm(s1[5], first_row[5], acc_b);
        acc_a = mul_add_asm(s0[6], first_row[6], acc_a);
        acc_b = mul_add_asm(s1[6], first_row[6], acc_b);
        acc_a = mul_add_asm(s0[7], first_row[7], acc_a);
        acc_b = mul_add_asm(s1[7], first_row[7], acc_b);

        // Tail updates: both lanes, interleaved.
        s0[1] = mul_add_asm(old_a, v[0], s0[1]);
        s1[1] = mul_add_asm(old_b, v[0], s1[1]);
        s0[2] = mul_add_asm(old_a, v[1], s0[2]);
        s1[2] = mul_add_asm(old_b, v[1], s1[2]);
        s0[3] = mul_add_asm(old_a, v[2], s0[3]);
        s1[3] = mul_add_asm(old_b, v[2], s1[3]);
        s0[4] = mul_add_asm(old_a, v[3], s0[4]);
        s1[4] = mul_add_asm(old_b, v[3], s1[4]);
        s0[5] = mul_add_asm(old_a, v[4], s0[5]);
        s1[5] = mul_add_asm(old_b, v[4], s1[5]);
        s0[6] = mul_add_asm(old_a, v[5], s0[6]);
        s1[6] = mul_add_asm(old_b, v[5], s1[6]);
        s0[7] = mul_add_asm(old_a, v[6], s0[7]);
        s1[7] = mul_add_asm(old_b, v[6], s1[7]);

        // Write the dot-product results into the first slots.
        s0[0] = acc_a;
        s1[0] = acc_b;
    }
}

/// Dual-lane sparse matrix-vector multiply for a width-12 state.
///
/// Same as the width-8 dual variant but with 12-element states.
/// Uses loops instead of full unrolling since width 12 is large
/// enough that code size matters more than marginal scheduling gains.
#[inline(always)]
pub unsafe fn cheap_matmul_dual_asm_w12(
    s0: &mut [u64; 12],
    s1: &mut [u64; 12],
    first_row: &[u64; 12],
    v: &[u64; 12],
) {
    unsafe {
        // Capture the original first elements from both lanes.
        let old_a = s0[0];
        let old_b = s1[0];

        // Dot products: one per lane, interleaved.
        let mut acc_a = mul_asm(s0[0], first_row[0]);
        let mut acc_b = mul_asm(s1[0], first_row[0]);
        for i in 1..12 {
            acc_a = mul_add_asm(s0[i], first_row[i], acc_a);
            acc_b = mul_add_asm(s1[i], first_row[i], acc_b);
        }

        // Tail updates: both lanes.
        for i in 1..12 {
            s0[i] = mul_add_asm(old_a, v[i - 1], s0[i]);
            s1[i] = mul_add_asm(old_b, v[i - 1], s1[i]);
        }

        // Write the dot-product results into the first slots.
        s0[0] = acc_a;
        s1[0] = acc_b;
    }
}

// ---------------------------------------------------------------------------
// Dense matrix-vector multiply (full-round linear layer)
// ---------------------------------------------------------------------------

/// Dense matrix-vector multiply for a width-8 state.
///
/// Computes `state = M * state` where M is a full 8x8 MDS matrix
/// stored in row-major order. Used in the **full rounds** of the
/// permutation where every element is mixed with every other.
///
/// Each output element is the dot product of one matrix row with the
/// input vector. The input is snapshotted before any writes occur.
pub fn dense_matmul_asm_w8(state: &mut [u64; 8], m: &[[u64; 8]; 8]) {
    unsafe {
        // Snapshot the current state so reads are not clobbered by writes.
        let input = *state;

        // Compute each output element as a dot product of one matrix
        // row with the snapshotted input.
        for i in 0..8 {
            let mut acc = mul_asm(input[0], m[i][0]);
            for j in 1..8 {
                acc = mul_add_asm(input[j], m[i][j], acc);
            }
            state[i] = acc;
        }
    }
}

/// Dense matrix-vector multiply for a width-12 state.
///
/// Same as the width-8 variant but with a 12×12 MDS matrix.
pub fn dense_matmul_asm_w12(state: &mut [u64; 12], m: &[[u64; 12]; 12]) {
    unsafe {
        // Snapshot the current state.
        let input = *state;

        // One dot product per output element.
        for i in 0..12 {
            let mut acc = mul_asm(input[0], m[i][0]);
            for j in 1..12 {
                acc = mul_add_asm(input[j], m[i][j], acc);
            }
            state[i] = acc;
        }
    }
}

/// Dual-lane dense matrix-vector multiply for a width-8 state.
///
/// Multiplies two independent state vectors by the same 8×8 matrix.
/// Both lanes share the matrix but have their own input and output.
///
/// Interleaving the two dot-product chains per row hides latency.
pub fn dense_matmul_dual_asm_w8(s0: &mut [u64; 8], s1: &mut [u64; 8], m: &[[u64; 8]; 8]) {
    unsafe {
        // Snapshot both input vectors.
        let in0 = *s0;
        let in1 = *s1;

        // For each row, compute both dot products in lockstep.
        for i in 0..8 {
            let mut a = mul_asm(in0[0], m[i][0]);
            let mut b = mul_asm(in1[0], m[i][0]);
            for j in 1..8 {
                a = mul_add_asm(in0[j], m[i][j], a);
                b = mul_add_asm(in1[j], m[i][j], b);
            }
            s0[i] = a;
            s1[i] = b;
        }
    }
}

/// Dual-lane dense matrix-vector multiply for a width-12 state.
///
/// Same as the width-8 dual variant but with a 12×12 matrix.
pub fn dense_matmul_dual_asm_w12(s0: &mut [u64; 12], s1: &mut [u64; 12], m: &[[u64; 12]; 12]) {
    unsafe {
        // Snapshot both input vectors.
        let in0 = *s0;
        let in1 = *s1;

        // For each row, compute both dot products in lockstep.
        for i in 0..12 {
            let mut a = mul_asm(in0[0], m[i][0]);
            let mut b = mul_asm(in1[0], m[i][0]);
            for j in 1..12 {
                a = mul_add_asm(in0[j], m[i][j], a);
                b = mul_add_asm(in1[j], m[i][j], b);
            }
            s0[i] = a;
            s1[i] = b;
        }
    }
}

// ---------------------------------------------------------------------------
// Round-constant addition
// ---------------------------------------------------------------------------

/// Add round constants to every element of the state.
///
/// This is the first step of every Poseidon1 round. Each element
/// receives its own constant, added in the Goldilocks field.
///
/// Generic over the state width to work with both width-8 and width-12.
#[inline(always)]
pub unsafe fn add_rc_asm<const WIDTH: usize>(state: &mut [u64; WIDTH], rc: &[u64; WIDTH]) {
    unsafe {
        // Element-wise modular addition.
        for i in 0..WIDTH {
            state[i] = add_asm(state[i], rc[i]);
        }
    }
}

/// Dual-lane round-constant addition.
///
/// Adds the same constants to two independent states. Both lanes
/// share the constants because they are at the same round position.
#[inline(always)]
pub unsafe fn add_rc_dual_asm<const WIDTH: usize>(
    s0: &mut [u64; WIDTH],
    s1: &mut [u64; WIDTH],
    rc: &[u64; WIDTH],
) {
    unsafe {
        // Both lanes receive the same constant at each position.
        for i in 0..WIDTH {
            s0[i] = add_asm(s0[i], rc[i]);
            s1[i] = add_asm(s1[i], rc[i]);
        }
    }
}

/// Add a single round constant to the first element only.
///
/// Used in partial rounds where only the first element enters the
/// S-box and thus only needs its own constant added.
#[inline(always)]
pub unsafe fn add_scalar_s0_asm(state: &mut [u64], rc: u64) {
    unsafe {
        // Only the first element is modified.
        state[0] = add_asm(state[0], rc);
    }
}

#[cfg(test)]
mod tests {
    use p3_field::PrimeField64;
    use proptest::prelude::*;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use super::*;
    use crate::Goldilocks;

    type F = Goldilocks;

    /// Reduce a raw `u64` to its canonical Goldilocks representative.
    ///
    /// Wraps the value into a field element and extracts the unique
    /// representative in `[0, P)`. This is the single source of truth
    /// for comparing ASM outputs (which may carry unreduced values)
    /// against field-level references.
    fn canon(x: u64) -> u64 {
        F::new(x).as_canonical_u64()
    }

    proptest! {
        // ================================================================
        // S-box: first element raised to the 7th power
        // ================================================================

        /// Verify the single-lane S-box against a field-level reference.
        ///
        /// The reference computes x^7 step by step using field multiplication.
        /// Only the first element should change; the rest must be untouched.
        #[test]
        fn test_sbox_s0_asm(vals in prop::array::uniform8(any::<u64>())) {
            // Build the expected x^7 using the field multiplication chain.
            let x = F::new(vals[0]);
            let x2 = x * x;
            let x3 = x2 * x;
            let x4 = x2 * x2;
            let expected_s0 = (x3 * x4).as_canonical_u64();

            // Run the ASM version on a copy.
            let mut state = vals;
            unsafe { sbox_s0_asm(&mut state); }

            // The first element must match x^7.
            prop_assert_eq!(canon(state[0]), expected_s0);

            // Every other element must be unchanged.
            for i in 1..8 {
                prop_assert_eq!(state[i], vals[i]);
            }
        }

        /// Verify the dual-lane S-box matches two independent single-lane calls.
        ///
        /// Runs the single-lane version on each lane separately as the
        /// reference, then checks the dual-lane version produces the same.
        #[test]
        fn test_sbox_s0_dual_asm(
            vals0 in prop::array::uniform8(any::<u64>()),
            vals1 in prop::array::uniform8(any::<u64>()),
        ) {
            // Build the reference by running single-lane on each lane.
            let mut ref0 = vals0;
            let mut ref1 = vals1;
            unsafe {
                sbox_s0_asm(&mut ref0);
                sbox_s0_asm(&mut ref1);
            }

            // Run the dual-lane version.
            let mut s0 = vals0;
            let mut s1 = vals1;
            unsafe { sbox_s0_dual_asm(&mut s0, &mut s1); }

            // Both first elements must match their reference.
            prop_assert_eq!(canon(s0[0]), canon(ref0[0]));
            prop_assert_eq!(canon(s1[0]), canon(ref1[0]));

            // All other elements must be unchanged.
            for i in 1..8 {
                prop_assert_eq!(s0[i], vals0[i]);
                prop_assert_eq!(s1[i], vals1[i]);
            }
        }

        // ================================================================
        // Round-constant addition: element-wise field addition
        // ================================================================

        /// Verify round-constant addition (width 8) against field addition.
        ///
        /// Each element should equal the field sum of the original value
        /// and its corresponding round constant.
        #[test]
        fn test_add_rc_asm_w8(
            vals in prop::array::uniform8(any::<u64>()),
            rc in prop::array::uniform8(any::<u64>()),
        ) {
            // Build the expected result using field addition.
            let expected: [u64; 8] = core::array::from_fn(|i| {
                (F::new(vals[i]) + F::new(rc[i])).as_canonical_u64()
            });

            // Run the ASM version.
            let mut state = vals;
            unsafe { add_rc_asm(&mut state, &rc); }

            // Every element must match.
            for i in 0..8 {
                prop_assert_eq!(canon(state[i]), expected[i]);
            }
        }

        /// Same verification for width 12.
        #[test]
        fn test_add_rc_asm_w12(
            vals in prop::array::uniform12(any::<u64>()),
            rc in prop::array::uniform12(any::<u64>()),
        ) {
            let expected: [u64; 12] = core::array::from_fn(|i| {
                (F::new(vals[i]) + F::new(rc[i])).as_canonical_u64()
            });

            let mut state = vals;
            unsafe { add_rc_asm(&mut state, &rc); }

            for i in 0..12 {
                prop_assert_eq!(canon(state[i]), expected[i]);
            }
        }

        /// Verify dual-lane round-constant addition (width 8) matches
        /// two independent single-lane calls.
        #[test]
        fn test_add_rc_dual_asm_w8(
            vals0 in prop::array::uniform8(any::<u64>()),
            vals1 in prop::array::uniform8(any::<u64>()),
            rc in prop::array::uniform8(any::<u64>()),
        ) {
            // Reference: single-lane on each independently.
            let mut ref0 = vals0;
            let mut ref1 = vals1;
            unsafe {
                add_rc_asm(&mut ref0, &rc);
                add_rc_asm(&mut ref1, &rc);
            }

            // Run the dual-lane version.
            let mut s0 = vals0;
            let mut s1 = vals1;
            unsafe { add_rc_dual_asm(&mut s0, &mut s1, &rc); }

            // Both lanes must match their references.
            for i in 0..8 {
                prop_assert_eq!(canon(s0[i]), canon(ref0[i]));
                prop_assert_eq!(canon(s1[i]), canon(ref1[i]));
            }
        }

        /// Same dual-lane verification for width 12.
        #[test]
        fn test_add_rc_dual_asm_w12(
            vals0 in prop::array::uniform12(any::<u64>()),
            vals1 in prop::array::uniform12(any::<u64>()),
            rc in prop::array::uniform12(any::<u64>()),
        ) {
            let mut ref0 = vals0;
            let mut ref1 = vals1;
            unsafe {
                add_rc_asm(&mut ref0, &rc);
                add_rc_asm(&mut ref1, &rc);
            }

            let mut s0 = vals0;
            let mut s1 = vals1;
            unsafe { add_rc_dual_asm(&mut s0, &mut s1, &rc); }

            for i in 0..12 {
                prop_assert_eq!(canon(s0[i]), canon(ref0[i]));
                prop_assert_eq!(canon(s1[i]), canon(ref1[i]));
            }
        }

        // ================================================================
        // Scalar addition: first element only
        // ================================================================

        /// Verify that adding a scalar to the first element matches
        /// field addition, and that all other elements are untouched.
        #[test]
        fn test_add_scalar_s0_asm(vals in prop::array::uniform8(any::<u64>()), rc: u64) {
            // Expected: field sum of the first element and the constant.
            let expected_s0 = (F::new(vals[0]) + F::new(rc)).as_canonical_u64();

            // Run the ASM version.
            let mut state = vals;
            unsafe { add_scalar_s0_asm(&mut state, rc); }

            // The first element must match.
            prop_assert_eq!(canon(state[0]), expected_s0);

            // Every other element must be unchanged.
            for i in 1..8 {
                prop_assert_eq!(state[i], vals[i]);
            }
        }

        // ================================================================
        // Sparse matrix-vector multiply (partial-round linear layer)
        //
        // The sparse matrix decomposes into:
        //   new[0] = dot(first_row, state)
        //   new[i] = state[i] + state[0] * v[i-1]   for i >= 1
        // ================================================================

        /// Verify the width-8 sparse matmul against a field-level reference.
        ///
        /// Builds the expected result by computing the dot product and
        /// the per-element multiply-add using Goldilocks field operations.
        #[test]
        fn test_cheap_matmul_asm_w8(
            vals in prop::array::uniform8(any::<u64>()),
            first_row in prop::array::uniform8(any::<u64>()),
            v in prop::array::uniform8(any::<u64>()),
        ) {
            // Lift raw values into field elements.
            let f: [F; 8] = vals.map(F::new);
            let fr: [F; 8] = first_row.map(F::new);
            let fv: [F; 8] = v.map(F::new);

            // Capture the original first element.
            let old_s0 = f[0];

            // Dot product for the new first element.
            let new_s0: F = (0..8).map(|i| f[i] * fr[i]).sum();

            // Tail update for elements 1..8.
            let mut expected = f;
            for i in 1..8 {
                expected[i] = f[i] + old_s0 * fv[i - 1];
            }
            expected[0] = new_s0;

            // Run the ASM version.
            let mut state = vals;
            unsafe { cheap_matmul_asm_w8(&mut state, &first_row, &v); }

            // Every element must match.
            for i in 0..8 {
                prop_assert_eq!(canon(state[i]), expected[i].as_canonical_u64());
            }
        }

        /// Same verification for width 12.
        #[test]
        fn test_cheap_matmul_asm_w12(
            vals in prop::array::uniform12(any::<u64>()),
            first_row in prop::array::uniform12(any::<u64>()),
            v in prop::array::uniform12(any::<u64>()),
        ) {
            let f: [F; 12] = vals.map(F::new);
            let fr: [F; 12] = first_row.map(F::new);
            let fv: [F; 12] = v.map(F::new);

            let old_s0 = f[0];
            let new_s0: F = (0..12).map(|i| f[i] * fr[i]).sum();

            let mut expected = f;
            for i in 1..12 {
                expected[i] = f[i] + old_s0 * fv[i - 1];
            }
            expected[0] = new_s0;

            let mut state = vals;
            unsafe { cheap_matmul_asm_w12(&mut state, &first_row, &v); }

            for i in 0..12 {
                prop_assert_eq!(canon(state[i]), expected[i].as_canonical_u64());
            }
        }

        /// Verify the width-8 dual-lane sparse matmul matches two
        /// independent single-lane calls.
        #[test]
        fn test_cheap_matmul_dual_asm_w8(
            vals0 in prop::array::uniform8(any::<u64>()),
            vals1 in prop::array::uniform8(any::<u64>()),
            first_row in prop::array::uniform8(any::<u64>()),
            v in prop::array::uniform8(any::<u64>()),
        ) {
            // Reference: single-lane on each independently.
            let mut ref0 = vals0;
            let mut ref1 = vals1;
            unsafe {
                cheap_matmul_asm_w8(&mut ref0, &first_row, &v);
                cheap_matmul_asm_w8(&mut ref1, &first_row, &v);
            }

            // Run the dual-lane version.
            let mut s0 = vals0;
            let mut s1 = vals1;
            unsafe { cheap_matmul_dual_asm_w8(&mut s0, &mut s1, &first_row, &v); }

            // Both lanes must match their references.
            for i in 0..8 {
                prop_assert_eq!(canon(s0[i]), canon(ref0[i]));
                prop_assert_eq!(canon(s1[i]), canon(ref1[i]));
            }
        }

        /// Same dual-lane verification for width 12.
        #[test]
        fn test_cheap_matmul_dual_asm_w12(
            vals0 in prop::array::uniform12(any::<u64>()),
            vals1 in prop::array::uniform12(any::<u64>()),
            first_row in prop::array::uniform12(any::<u64>()),
            v in prop::array::uniform12(any::<u64>()),
        ) {
            let mut ref0 = vals0;
            let mut ref1 = vals1;
            unsafe {
                cheap_matmul_asm_w12(&mut ref0, &first_row, &v);
                cheap_matmul_asm_w12(&mut ref1, &first_row, &v);
            }

            let mut s0 = vals0;
            let mut s1 = vals1;
            unsafe { cheap_matmul_dual_asm_w12(&mut s0, &mut s1, &first_row, &v); }

            for i in 0..12 {
                prop_assert_eq!(canon(s0[i]), canon(ref0[i]));
                prop_assert_eq!(canon(s1[i]), canon(ref1[i]));
            }
        }

        // ================================================================
        // Dense matrix-vector multiply (full-round linear layer)
        // ================================================================

        /// Verify the width-8 dense matmul against a field-level reference.
        ///
        /// Each output element is the dot product of one matrix row with
        /// the input vector. The matrix is fixed from a deterministic seed.
        #[test]
        fn test_dense_matmul_asm_w8(vals in prop::array::uniform8(any::<u64>())) {
            // Fixed matrix from a deterministic seed.
            let mut rng = SmallRng::seed_from_u64(42);
            let m: [[u64; 8]; 8] = rand::RngExt::random(&mut rng);

            // Reference: standard matrix-vector product using field ops.
            let f: [F; 8] = vals.map(F::new);
            let expected: [F; 8] = core::array::from_fn(|i| {
                (0..8).map(|j| f[j] * F::new(m[i][j])).sum()
            });

            // Run the ASM version.
            let mut state = vals;
            dense_matmul_asm_w8(&mut state, &m);

            // Every element must match.
            for i in 0..8 {
                prop_assert_eq!(canon(state[i]), expected[i].as_canonical_u64());
            }
        }

        /// Same verification for width 12.
        #[test]
        fn test_dense_matmul_asm_w12(vals in prop::array::uniform12(any::<u64>())) {
            let mut rng = SmallRng::seed_from_u64(43);
            let m: [[u64; 12]; 12] = rand::RngExt::random(&mut rng);

            let f: [F; 12] = vals.map(F::new);
            let expected: [F; 12] = core::array::from_fn(|i| {
                (0..12).map(|j| f[j] * F::new(m[i][j])).sum()
            });

            let mut state = vals;
            dense_matmul_asm_w12(&mut state, &m);

            for i in 0..12 {
                prop_assert_eq!(canon(state[i]), expected[i].as_canonical_u64());
            }
        }

        /// Verify the width-8 dual-lane dense matmul matches two
        /// independent single-lane calls.
        #[test]
        fn test_dense_matmul_dual_asm_w8(
            vals0 in prop::array::uniform8(any::<u64>()),
            vals1 in prop::array::uniform8(any::<u64>()),
        ) {
            // Fixed matrix from a deterministic seed.
            let mut rng = SmallRng::seed_from_u64(44);
            let m: [[u64; 8]; 8] = rand::RngExt::random(&mut rng);

            // Reference: single-lane on each independently.
            let mut ref0 = vals0;
            let mut ref1 = vals1;
            dense_matmul_asm_w8(&mut ref0, &m);
            dense_matmul_asm_w8(&mut ref1, &m);

            // Run the dual-lane version.
            let mut s0 = vals0;
            let mut s1 = vals1;
            dense_matmul_dual_asm_w8(&mut s0, &mut s1, &m);

            // Both lanes must match their references.
            for i in 0..8 {
                prop_assert_eq!(canon(s0[i]), canon(ref0[i]));
                prop_assert_eq!(canon(s1[i]), canon(ref1[i]));
            }
        }

        /// Same dual-lane verification for width 12.
        #[test]
        fn test_dense_matmul_dual_asm_w12(
            vals0 in prop::array::uniform12(any::<u64>()),
            vals1 in prop::array::uniform12(any::<u64>()),
        ) {
            let mut rng = SmallRng::seed_from_u64(45);
            let m: [[u64; 12]; 12] = rand::RngExt::random(&mut rng);

            let mut ref0 = vals0;
            let mut ref1 = vals1;
            dense_matmul_asm_w12(&mut ref0, &m);
            dense_matmul_asm_w12(&mut ref1, &m);

            let mut s0 = vals0;
            let mut s1 = vals1;
            dense_matmul_dual_asm_w12(&mut s0, &mut s1, &m);

            for i in 0..12 {
                prop_assert_eq!(canon(s0[i]), canon(ref0[i]));
                prop_assert_eq!(canon(s1[i]), canon(ref1[i]));
            }
        }
    }
}
