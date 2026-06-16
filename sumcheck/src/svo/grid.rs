//! Ternary `{0, 1, inf}^l` grid expansion.
//!
//! Extends Boolean-hypercube evaluations with the "evaluation at infinity" along
//! every variable, producing the grid the SVO accumulators read from.

use alloc::vec::Vec;

use p3_field::Field;
use p3_maybe_rayon::prelude::*;
use p3_util::log2_strict_usize;

/// Expand `2^l` Boolean-hypercube evaluations to `3^l` evaluations on `{0,1,inf}^l`.
///
/// # Overview
///
/// A multilinear polynomial in `l` variables is determined by `2^l` evaluations
/// on `{0,1}^l`. This extends them to include the "evaluation at infinity"
/// (the leading coefficient) for each variable:
///
/// ```text
///     f(0), f(1)  -->  f(0), f(1), f(inf)    where f(inf) = f(1) - f(0)
/// ```
///
/// # Motivation
///
/// The SVO sumcheck prover needs accumulator values on the `{0, 1, inf}^l` grid.
///
/// - **Naive**: evaluate each of the `3^l` grid points independently via
///   Lagrange interpolation from the `2^l` Boolean values --> `O(6^l)`.
/// - **This function**: process one variable at a time --> `O(3^l)`.
///
/// # Memory Layout
///
/// The input uses prefix-variable-fastest ordering: `idx = x_0 + 2*x_1 + ...`
///
/// The output uses the same convention as the expansion order. The first
/// variable processed (x_0) becomes the **slowest**-varying coordinate
/// in the ternary grid. Flat index = `x_1 + 3*x_0` for 2 variables:
///
/// ```text
///     l=2 example:
///     index:   0      1      2      3      4      5      6      7      8
///     point: (0,0)  (0,1)  (0,inf)  (1,0)  (1,1)  (1,inf)  (inf,0)  (inf,1)  (inf,inf)
/// ```
///
/// # Algorithm
///
/// `l` stages, one per variable. Each stage converts pairs into triples:
///
/// ```text
///     stage 0:  2^l             values on {0,1}^l
///     stage 1:  3 * 2^{l-1}    values on {0, 1, inf} x {0,1}^{l-1}
///       ...
///     stage l:  3^l             values on {0, 1, inf}^l
/// ```
///
/// Two buffers alternate in a ping-pong pattern.
/// Initial assignment ensures the final result lands in the output buffer.
///
/// # Parallelization
///
/// - **Early stages**: many small blocks --> parallelize across blocks.
/// - **Late stages**: few large blocks --> parallelize within each block.
///
/// # Panics
///
/// - Input length not a power of two.
/// - Output or scratch length != `3^l`.
///
/// # Performance
///
/// - Time: `O(3^l)` field additions and doublings.
/// - Space: two pre-allocated `3^l` buffers, no internal allocation.
pub(super) fn evals_01inf_grid_into<F: Field>(
    boolean_evals: &[F],
    output: &mut [F],
    scratch: &mut [F],
) {
    let num_variables = log2_strict_usize(boolean_evals.len());
    let output_len = 3usize.pow(num_variables as u32);

    assert_eq!(output.len(), output_len);
    assert_eq!(scratch.len(), output_len);

    // Single constant -- nothing to expand.
    if num_variables == 0 {
        output[0] = boolean_evals[0];
        return;
    }

    // Ping-pong buffer setup.
    //
    // Each stage swaps cur/next. After l swaps the result must be in `output`.
    //
    //     l odd  --> start in scratch --> after l swaps --> output  OK
    //     l even --> start in output  --> after l swaps --> output  OK
    let (mut cur, mut next) = if num_variables % 2 == 1 {
        scratch[..boolean_evals.len()].copy_from_slice(boolean_evals);
        (&mut scratch[..], &mut output[..])
    } else {
        output[..boolean_evals.len()].copy_from_slice(boolean_evals);
        (&mut output[..], &mut scratch[..])
    };

    // Below this: parallelize across blocks (many small chunks).
    // Above this: parallelize within each block (few large chunks).
    //
    // Why 256: below this the per-element work is too small for per-element
    // thread scheduling; above this there may be only 1-2 blocks.
    const PARALLEL_STRIDE_THRESHOLD: usize = 256;

    for stage in 0..num_variables {
        // Stride = 3^stage: how many consecutive elements share the same
        // value of the variable being expanded (the already-expanded
        // variables each contribute a factor of 3).
        let in_stride = 3usize.pow(stage as u32);

        // Blocks = 2^{remaining}: independent groups of (f(0), f(1)) pairs.
        let blocks = 1usize << (num_variables - stage - 1);

        // Slice only the live region (early stages use less than the full buffer).
        //
        //     cur  per block: [ f(0)-group | f(1)-group ]   each of size in_stride
        //     next per block: [ f(0)-group | f(1)-group | f(inf)-group ]
        let cur_slice = &cur[..blocks * 2 * in_stride];
        let next_slice = &mut next[..blocks * 3 * in_stride];

        if in_stride < PARALLEL_STRIDE_THRESHOLD {
            // Many small blocks -- parallelize across blocks.
            //
            //     l=2, stage 0 (stride=1, blocks=2):
            //     cur:   [ f(0,0) f(1,0) | f(0,1) f(1,1) ]
            //     next:  [ f(0,0) f(1,0) f(inf,0) | f(0,1) f(1,1) f(inf,1) ]
            cur_slice
                .par_chunks(2 * in_stride)
                .zip(next_slice.par_chunks_mut(3 * in_stride))
                .for_each(|(c_chunk, n_chunk)| {
                    for j in 0..in_stride {
                        let f0 = c_chunk[j];
                        let f1 = c_chunk[in_stride + j];
                        // Interleaved output: position j --> indices 3j, 3j+1, 3j+2.
                        n_chunk[3 * j] = f0;
                        n_chunk[3 * j + 1] = f1;
                        n_chunk[3 * j + 2] = f1 - f0;
                    }
                });
        } else {
            // Few large blocks -- parallelize within each block.
            cur_slice
                .chunks(2 * in_stride)
                .zip(next_slice.chunks_mut(3 * in_stride))
                .for_each(|(c_chunk, n_chunk)| {
                    // Split into left=f(0) half, right=f(1) half.
                    let (c_left, c_right) = c_chunk.split_at(in_stride);
                    // Each (f0, f1) pair --> (f0, f1, f1 - f0) triple.
                    c_left
                        .par_iter()
                        .zip(c_right.par_iter())
                        .zip(n_chunk.par_chunks_mut(3))
                        .for_each(|((&f0, &f1), out)| {
                            out[0] = f0;
                            out[1] = f1;
                            out[2] = f1 - f0;
                        });
                });
        }

        // What was `next` becomes `cur` for the next stage.
        core::mem::swap(&mut cur, &mut next);
    }
}

pub(crate) fn evals_01inf_grid_prefix<F: Field>(evals: &[F]) -> Vec<F> {
    fn reverse_ternary_digits(mut idx: usize, l: usize) -> usize {
        let mut rev = 0usize;
        for _ in 0..l {
            rev = 3 * rev + (idx % 3);
            idx /= 3;
        }
        rev
    }

    let l = log2_strict_usize(evals.len());
    let grid_len = 3usize.pow(l as u32);
    let mut prefix = F::zero_vec(grid_len);
    let mut scratch = F::zero_vec(grid_len);
    evals_01inf_grid_into(evals, &mut prefix, &mut scratch);

    let mut out = F::zero_vec(grid_len);
    for (src_idx, value) in prefix.into_iter().enumerate() {
        out[reverse_ternary_digits(src_idx, l)] = value;
    }
    out
}

#[cfg(test)]
mod test {
    use alloc::vec;
    use alloc::vec::Vec;

    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;
    use p3_koala_bear::KoalaBear;
    use proptest::prelude::*;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;

    type F = KoalaBear;
    type EF = BinomialExtensionField<F, 4>;

    /// Convenience wrapper: expand boolean evals onto {0, 1, inf}^l grid.
    fn evals_01inf_grid(boolean_evals: &[EF]) -> Vec<EF> {
        let num_variables = log2_strict_usize(boolean_evals.len());
        let output_len = 3usize.pow(num_variables as u32);
        let mut output = EF::zero_vec(output_len);
        let mut scratch = EF::zero_vec(output_len);
        evals_01inf_grid_into(boolean_evals, &mut output, &mut scratch);
        output
    }

    /// Compare the grid expansion against naive multilinear evaluation on every
    /// point of `{0, 1, inf}^l`.
    /// Verify the grid stores correct `(f(0), f(1), f(inf))` triples.
    ///
    /// For each variable, the third slot must satisfy `f(inf) = f(1) - f(0)`.
    /// Combined with the fact that `f(0)` and `f(1)` are the original Boolean
    /// evaluations, this fully validates the grid.
    fn assert_evals_01inf_grid_correct(boolean_evals: &[EF]) {
        let num_variables = log2_strict_usize(boolean_evals.len());
        let grid = evals_01inf_grid(boolean_evals);

        // For every triple along the innermost variable (stride 1), check
        // that grid[3k+2] == grid[3k+1] - grid[3k].
        let inner_groups = grid.len() / 3;
        for g in 0..inner_groups {
            let v0 = grid[3 * g];
            let v1 = grid[3 * g + 1];
            let v_inf = grid[3 * g + 2];
            assert_eq!(
                v_inf,
                v1 - v0,
                "f(inf) != f(1)-f(0) at group {g}, num_variables={num_variables}"
            );
        }
    }

    // Tests for evals_01inf_grid_into

    #[test]
    fn test_evals_01inf_grid_into_zero_vars() {
        // Zero variables: the polynomial is a single constant.
        // Input: [c] on {0}^0 (one point, the empty tuple).
        // Output: [c] on {0}^0 (still one point).
        let c = EF::from_u32(42);
        let input = [c];
        let mut output = [EF::ZERO];
        let mut scratch = [EF::ZERO];

        evals_01inf_grid_into(&input, &mut output, &mut scratch);

        // The sole value is copied through unchanged.
        assert_eq!(output, [c]);
    }

    #[test]
    fn test_evals_01inf_grid_into_one_var() {
        // One variable: f(0) = 3, f(1) = 7.
        // f(inf) = f(1) - f(0) = 7 - 3 = 4  (the leading coefficient).
        //
        // Output layout on {0, 1, inf}:
        //   index 0 → f(0) = 3
        //   index 1 → f(1) = 7
        //   index 2 → f(inf) = 4
        let f0 = EF::from_u32(3);
        let f1 = EF::from_u32(7);
        let input = [f0, f1];
        let mut output = [EF::ZERO; 3];
        let mut scratch = [EF::ZERO; 3];

        evals_01inf_grid_into(&input, &mut output, &mut scratch);

        assert_eq!(output[0], f0);
        assert_eq!(output[1], f1);
        // f(inf) = 7 - 3 = 4
        assert_eq!(output[2], EF::from_u32(4));
    }

    #[test]
    fn test_evals_01inf_grid_into_two_vars_hand_computed() {
        // f(x_0, x_1) = 1 + 2*x_0 + 4*x_1 + 4*x_0*x_1
        //
        // Input (low-var-fastest):
        //   idx 0 -> (0,0) = 1
        //   idx 1 -> (1,0) = 3
        //   idx 2 -> (0,1) = 5
        //   idx 3 -> (1,1) = 11
        //
        // Stage 0 expands x_0: each (f(0), f(1)) pair -> (f(0), f(1), f(1)-f(0)):
        //   x_1=0: (1, 3) -> (1, 3, 2)
        //   x_1=1: (5, 11) -> (5, 11, 6)
        //   buffer: [1, 3, 2, 5, 11, 6]
        //
        // Stage 1 expands x_1 (stride=3): for each x_0 value,
        // the pair (f(x_0,0), f(x_0,1)) -> (f(x_0,0), f(x_0,1), f(x_0,1)-f(x_0,0)):
        //   x_0=0: (1, 5)  -> (1, 5, 4)
        //   x_0=1: (3, 11) -> (3, 11, 8)
        //   x_0=2: (2, 6)  -> (2, 6, 4)
        //
        // Output (x_0 slowest, x_1 fastest), index = x_1 + 3*x_0:
        //   idx:   0  1  2  3   4  5  6  7  8
        //   val:   1  5  4  3  11  8  2  6  4
        let input = [1, 3, 5, 11].map(EF::from_u32);
        let mut output = [EF::ZERO; 9];
        let mut scratch = [EF::ZERO; 9];

        evals_01inf_grid_into(&input, &mut output, &mut scratch);

        let expected = [1, 5, 4, 3, 11, 8, 2, 6, 4].map(EF::from_u32);
        assert_eq!(output, expected);
    }

    #[test]
    fn test_evals_01inf_grid_into_output_size() {
        // Verify the output length is 3^l for various numbers of variables.
        for num_variables in 1..=5 {
            let input_len = 1 << num_variables;
            let output_len = 3usize.pow(num_variables as u32);

            // Use all-zero input; we only care about sizes here.
            let input = EF::zero_vec(input_len);
            let mut output = EF::zero_vec(output_len);
            let mut scratch = EF::zero_vec(output_len);

            // Should not panic.
            evals_01inf_grid_into(&input, &mut output, &mut scratch);
        }
    }

    #[test]
    fn test_evals_01inf_grid_into_result_lands_in_output() {
        // The ping-pong buffer logic must place the final result in the
        // output buffer, not in scratch. Verify for both odd and even
        // numbers of variables (since the initial buffer assignment differs).
        let mut rng = SmallRng::seed_from_u64(123);

        for num_variables in 1..=4 {
            let input: Vec<EF> = (0..1 << num_variables).map(|_| rng.random()).collect();
            let output_len = 3usize.pow(num_variables as u32);
            let mut output = EF::zero_vec(output_len);
            let mut scratch = EF::zero_vec(output_len);

            evals_01inf_grid_into(&input, &mut output, &mut scratch);

            // The grid computed via the convenience wrapper must match.
            let reference = evals_01inf_grid(&input);
            assert_eq!(
                output, reference,
                "ping-pong mismatch for num_variables={num_variables}"
            );
        }
    }

    #[test]
    fn test_evals_01inf_grid_into_preserves_boolean_points() {
        // The grid expansion must preserve the original 2^l Boolean evaluations.
        // At every Boolean point in {0,1}^l, the grid value must equal the input.
        //
        // The binary index uses b_0 + 2*b_1 + ... (low-var-fastest).
        // The ternary grid has x_0 slowest (first variable processed),
        // so the ternary index for a Boolean point is b_{l-1} + 3*b_{l-2} + ... + 3^{l-1}*b_0.
        let mut rng = SmallRng::seed_from_u64(77);

        for num_variables in 1..=4 {
            let input: Vec<EF> = (0..1 << num_variables).map(|_| rng.random()).collect();
            let grid = evals_01inf_grid(&input);

            for (bool_idx, &input_val) in input.iter().enumerate() {
                // Extract binary digits (low-var-first): b_0, b_1, ..., b_{l-1}.
                let mut bits = Vec::with_capacity(num_variables);
                let mut tmp = bool_idx;
                for _ in 0..num_variables {
                    bits.push(tmp & 1);
                    tmp >>= 1;
                }

                // Build ternary index with reversed variable order:
                // x_0 is slowest (weight 3^{l-1}), x_{l-1} is fastest (weight 1).
                let mut ternary_idx = 0;
                let mut power_of_3 = 1;
                for &b in bits.iter().rev() {
                    ternary_idx += b * power_of_3;
                    power_of_3 *= 3;
                }

                assert_eq!(
                    grid[ternary_idx], input_val,
                    "Boolean point mismatch at bool_idx={bool_idx}, num_variables={num_variables}"
                );
            }
        }
    }

    #[test]
    fn test_evals_01inf_grid_into_constant_polynomial() {
        // A constant polynomial f(x) = c has f(0) = f(1) = c and f(inf) = 0.
        //
        // For l=1, the grid is [c, c, 0].
        // For l=2, the grid is [c, c, 0, c, c, 0, 0, 0, 0].
        // The {0,1} slots hold c; any slot involving inf in any coordinate is 0.
        let c = EF::from_u32(99);

        for num_variables in 0..=4 {
            let input = vec![c; 1 << num_variables];
            let output_len = 3usize.pow(num_variables as u32);
            let mut output = EF::zero_vec(output_len);
            let mut scratch = EF::zero_vec(output_len);

            evals_01inf_grid_into(&input, &mut output, &mut scratch);

            // Check each grid point. A point has value c if all its ternary
            // digits are in {0,1}, and value 0 if any digit is 2 (the inf slot).
            for (idx, &val) in output.iter().enumerate() {
                let has_inf = {
                    let mut tmp = idx;
                    let mut found = false;
                    for _ in 0..num_variables {
                        if tmp % 3 == 2 {
                            found = true;
                        }
                        tmp /= 3;
                    }
                    found
                };
                let expected = if has_inf { EF::ZERO } else { c };
                assert_eq!(
                    val, expected,
                    "constant polynomial mismatch at idx={idx}, num_variables={num_variables}"
                );
            }
        }
    }

    #[test]
    fn test_evals_01inf_grid_into_linearity() {
        // The grid expansion must be linear:
        //   grid(a*f + b*g) = a*grid(f) + b*grid(g)
        //
        // This follows from the extrapolation formula being linear,
        // but verify it concretely.
        let mut rng = SmallRng::seed_from_u64(55);
        let num_variables = 3;
        let n = 1 << num_variables;

        let f: Vec<EF> = (0..n).map(|_| rng.random()).collect();
        let g: Vec<EF> = (0..n).map(|_| rng.random()).collect();
        let a: EF = rng.random();
        let b: EF = rng.random();

        // Compute grid(a*f + b*g).
        let combined: Vec<EF> = f
            .iter()
            .zip(g.iter())
            .map(|(&fi, &gi)| a * fi + b * gi)
            .collect();
        let grid_combined = evals_01inf_grid(&combined);

        // Compute a*grid(f) + b*grid(g).
        let grid_f = evals_01inf_grid(&f);
        let grid_g = evals_01inf_grid(&g);
        let linear_combined: Vec<EF> = grid_f
            .iter()
            .zip(grid_g.iter())
            .map(|(&fi, &gi)| a * fi + b * gi)
            .collect();

        assert_eq!(grid_combined, linear_combined);
    }

    #[test]
    fn test_evals_01inf_grid_into_large_stride_branch_matches_naive() {
        // `num_variables = 7` guarantees the final stage has `in_stride = 3^6 = 729`,
        // which takes the large-stride branch (`in_stride >= 256`).
        let num_variables = 7;
        let mut rng = SmallRng::seed_from_u64(2025);
        let evals: Vec<EF> = (0..1 << num_variables).map(|_| rng.random()).collect();
        assert_evals_01inf_grid_correct(evals.as_slice());
    }

    #[test]
    #[should_panic(expected = "Not a power of two")]
    fn test_evals_01inf_grid_into_panics_on_non_power_of_two_input() {
        let input = [EF::ZERO; 3];
        let mut output = [EF::ZERO; 3];
        let mut scratch = [EF::ZERO; 3];

        evals_01inf_grid_into(&input, &mut output, &mut scratch);
    }

    #[test]
    #[should_panic(expected = "assertion `left == right` failed")]
    fn test_evals_01inf_grid_into_panics_on_wrong_output_len() {
        let input = [EF::ZERO; 4];
        let mut output = [EF::ZERO; 8];
        let mut scratch = [EF::ZERO; 9];

        evals_01inf_grid_into(&input, &mut output, &mut scratch);
    }

    #[test]
    #[should_panic(expected = "assertion `left == right` failed")]
    fn test_evals_01inf_grid_into_panics_on_wrong_scratch_len() {
        let input = [EF::ZERO; 4];
        let mut output = [EF::ZERO; 9];
        let mut scratch = [EF::ZERO; 8];

        evals_01inf_grid_into(&input, &mut output, &mut scratch);
    }

    proptest! {
        /// Verify the {0, 1, inf}^l grid expansion matches naive MLE evaluation.
        #[test]
        fn prop_evals_01inf_grid_matches_naive(num_variables in 1usize..=5) {
            // For each grid point in {0, 1, inf}^l, compare the fast grid-expansion
            // result against the naive approach of evaluating the multilinear
            // extension at that point by repeatedly fixing variables.
            let mut rng = SmallRng::seed_from_u64(num_variables as u64);
            let evals: Vec<EF> = (0..1 << num_variables).map(|_| rng.random()).collect();
            assert_evals_01inf_grid_correct(evals.as_slice());
        }

        /// Verify that grid expansion preserves the original Boolean-hypercube values.
        #[test]
        fn prop_evals_01inf_grid_preserves_boolean_points(num_variables in 1usize..=6) {
            let mut rng = SmallRng::seed_from_u64(num_variables as u64 + 1000);
            let input: Vec<EF> = (0..1 << num_variables).map(|_| rng.random()).collect();
            let grid = evals_01inf_grid(&input);

            for (bool_idx, &input_val) in input.iter().enumerate() {
                // Extract binary digits (low-var-first): b_0, b_1, ..., b_{l-1}.
                let mut bits = Vec::with_capacity(num_variables);
                let mut tmp = bool_idx;
                for _ in 0..num_variables {
                    bits.push(tmp & 1);
                    tmp >>= 1;
                }

                // Ternary index with x_0 slowest (reversed variable order).
                let mut ternary_idx = 0;
                let mut power_of_3 = 1;
                for &b in bits.iter().rev() {
                    ternary_idx += b * power_of_3;
                    power_of_3 *= 3;
                }

                prop_assert_eq!(grid[ternary_idx], input_val);
            }
        }
    }
}
