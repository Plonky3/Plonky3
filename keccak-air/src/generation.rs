use alloc::vec::Vec;
use core::array;
use core::mem::transmute;

use p3_air::utils::{u64_to_16_bit_limbs, u64_to_bits_le};
use p3_field::PrimeField64;
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::iter::repeat_n;
use p3_maybe_rayon::prelude::*;
use tracing::instrument;

use crate::columns::{KeccakCols, NUM_KECCAK_COLS};
use crate::{NUM_ROUNDS, R, RC, U64_LIMBS};

// TODO: Take generic iterable
#[instrument(name = "generate Keccak trace", skip_all)]
pub fn generate_trace_rows<F: PrimeField64>(
    inputs: Vec<[u64; 25]>,
    extra_capacity_bits: usize,
) -> RowMajorMatrix<F> {
    let num_rows = (inputs.len() * NUM_ROUNDS).next_power_of_two();
    let trace_length = num_rows * NUM_KECCAK_COLS;

    // We allocate extra_capacity_bits now as this will be needed by the dft.
    let mut long_trace = F::zero_vec(trace_length << extra_capacity_bits);
    long_trace.truncate(trace_length);

    let mut trace = RowMajorMatrix::new(long_trace, NUM_KECCAK_COLS);
    let (prefix, rows, suffix) = unsafe { trace.values.align_to_mut::<KeccakCols<F>>() };
    assert!(prefix.is_empty(), "Alignment should match");
    assert!(suffix.is_empty(), "Alignment should match");
    assert_eq!(rows.len(), num_rows);

    let num_padding_inputs = num_rows.div_ceil(NUM_ROUNDS) - inputs.len();
    let padded_inputs = inputs
        .into_par_iter()
        .chain(repeat_n([0; 25], num_padding_inputs));

    rows.par_chunks_mut(NUM_ROUNDS)
        .zip(padded_inputs)
        .for_each(|(row, input)| {
            generate_trace_rows_for_perm(row, input);
        });

    trace
}

/// `rows` will normally consist of 24 rows, with an exception for the final row.
fn generate_trace_rows_for_perm<F: PrimeField64>(rows: &mut [KeccakCols<F>], input: [u64; 25]) {
    // Convert flat input array to 5x5 matrix.
    // The input uses standard Keccak indexing: input[x + 5*y] corresponds to state[x][y].
    // After transmute, we get row-major layout: transmuted[i][j] = input[i*5 + j].
    // To align with Keccak's state[x][y] = input[x + 5*y], we need to transpose.
    let transmuted: [[u64; 5]; 5] = unsafe { transmute(input) };
    let mut current_state: [[u64; 5]; 5] = array::from_fn(|x| array::from_fn(|y| transmuted[y][x]));

    // initial_state is stored in y-major order for the AIR columns (preimage[y][x]).
    let initial_state: [[[F; 4]; 5]; 5] =
        array::from_fn(|y| array::from_fn(|x| u64_to_16_bit_limbs(current_state[x][y])));

    // Populate the round input for the first round.
    rows[0].a = initial_state;
    rows[0].preimage = initial_state;

    generate_trace_row_for_round(&mut rows[0], 0, &mut current_state);

    for round in 1..rows.len() {
        rows[round].preimage = initial_state;

        // Copy previous row's output to next row's input.
        for y in 0..5 {
            for x in 0..5 {
                for limb in 0..U64_LIMBS {
                    rows[round].a[y][x][limb] = rows[round - 1].a_prime_prime_prime(y, x, limb);
                }
            }
        }

        generate_trace_row_for_round(&mut rows[round], round, &mut current_state);
    }
}

fn generate_trace_row_for_round<F: PrimeField64>(
    row: &mut KeccakCols<F>,
    round: usize,
    current_state: &mut [[u64; 5]; 5],
) {
    row.step_flags[round] = F::ONE;

    // Populate C[x] = xor(A[x, 0], A[x, 1], A[x, 2], A[x, 3], A[x, 4]).
    let state_c: [u64; 5] = current_state.map(|row| row.iter().fold(0, |acc, y| acc ^ y));
    for (x, elem) in state_c.iter().enumerate() {
        row.c[x] = u64_to_bits_le(*elem);
    }

    // Populate C'[x, z] = xor(C[x, z], C[x - 1, z], C[x + 1, z - 1]).
    let state_c_prime: [u64; 5] =
        array::from_fn(|x| state_c[x] ^ state_c[(x + 4) % 5] ^ state_c[(x + 1) % 5].rotate_left(1));
    for (x, elem) in state_c_prime.iter().enumerate() {
        row.c_prime[x] = u64_to_bits_le(*elem);
    }

    // Populate A'. To avoid shifting indices, we rewrite
    //     A'[x, y, z] = xor(A[x, y, z], C[x - 1, z], C[x + 1, z - 1])
    // as
    //     A'[x, y, z] = xor(A[x, y, z], C[x, z], C'[x, z]).
    *current_state =
        array::from_fn(|i| array::from_fn(|j| current_state[i][j] ^ state_c[i] ^ state_c_prime[i]));
    for (x, x_row) in current_state.iter().enumerate() {
        for (y, elem) in x_row.iter().enumerate() {
            row.a_prime[y][x] = u64_to_bits_le(*elem);
        }
    }

    // Rotate the current state to get the B array.
    *current_state = array::from_fn(|i| {
        array::from_fn(|j| {
            let new_i = (i + 3 * j) % 5;
            let new_j = i;
            current_state[new_i][new_j].rotate_left(R[new_i][new_j] as u32)
        })
    });

    // Populate A''.
    // A''[x, y] = xor(B[x, y], andn(B[x + 1, y], B[x + 2, y])).
    *current_state = array::from_fn(|i| {
        array::from_fn(|j| {
            current_state[i][j] ^ ((!current_state[(i + 1) % 5][j]) & current_state[(i + 2) % 5][j])
        })
    });
    for (x, x_row) in current_state.iter().enumerate() {
        for (y, elem) in x_row.iter().enumerate() {
            row.a_prime_prime[y][x] = u64_to_16_bit_limbs(*elem);
        }
    }

    row.a_prime_prime_0_0_bits = u64_to_bits_le(current_state[0][0]);

    // A''[0, 0] is additionally xor'd with RC.
    current_state[0][0] ^= RC[round];

    row.a_prime_prime_prime_0_0_limbs = u64_to_16_bit_limbs(current_state[0][0]);
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_goldilocks::Goldilocks;
    use p3_keccak::KeccakF;
    use p3_symmetric::Permutation;

    use super::*;

    /// Helper function to extract the output state from the trace after all 24 rounds.
    /// The output is stored in `a_prime_prime_prime` for (0,0) and `a_prime_prime` for others.
    fn extract_output_from_trace<F: PrimeField64>(rows: &[KeccakCols<F>]) -> [u64; 25] {
        let last_row = &rows[NUM_ROUNDS - 1];
        let mut output = [0u64; 25];

        for y in 0..5 {
            for x in 0..5 {
                let mut value = 0u64;
                for limb in 0..U64_LIMBS {
                    let limb_val = last_row.a_prime_prime_prime(y, x, limb).as_canonical_u64();
                    value |= limb_val << (limb * 16);
                }
                // Standard Keccak indexing: state[x + 5*y]
                output[x + 5 * y] = value;
            }
        }
        output
    }

    /// Helper function to extract the input preimage from the trace.
    fn extract_input_from_trace<F: PrimeField64>(rows: &[KeccakCols<F>]) -> [u64; 25] {
        let first_row = &rows[0];
        let mut input = [0u64; 25];

        for y in 0..5 {
            for x in 0..5 {
                let mut value = 0u64;
                for limb in 0..U64_LIMBS {
                    let limb_val = first_row.preimage[y][x][limb].as_canonical_u64();
                    value |= limb_val << (limb * 16);
                }
                // Standard Keccak indexing: state[x + 5*y]
                input[x + 5 * y] = value;
            }
        }
        input
    }

    #[test]
    fn test_keccak_permutation_matches_p3_keccak() {
        // Test with a non-trivial input state
        let input: [u64; 25] = core::array::from_fn(|i| i as u64 * 0x0123456789ABCDEFu64);

        // Compute expected output using p3-keccak (reference implementation)
        let mut expected_output = input;
        KeccakF.permute_mut(&mut expected_output);

        // Generate trace using our implementation
        let trace = generate_trace_rows::<Goldilocks>(vec![input], 0);
        let (prefix, rows, suffix) = unsafe { trace.values.align_to::<KeccakCols<Goldilocks>>() };
        assert!(prefix.is_empty());
        assert!(suffix.is_empty());

        // Verify input was stored correctly
        let stored_input = extract_input_from_trace(&rows[..NUM_ROUNDS]);
        assert_eq!(
            stored_input, input,
            "Input state should match the provided input"
        );

        // Verify output matches p3-keccak
        let our_output = extract_output_from_trace(&rows[..NUM_ROUNDS]);
        assert_eq!(
            our_output, expected_output,
            "Keccak-f output should match p3-keccak reference implementation"
        );
    }

    #[test]
    fn test_keccak_permutation_zero_state() {
        // Test with all-zero state
        let input = [0u64; 25];

        let mut expected_output = input;
        KeccakF.permute_mut(&mut expected_output);

        let trace = generate_trace_rows::<Goldilocks>(vec![input], 0);
        let (prefix, rows, suffix) = unsafe { trace.values.align_to::<KeccakCols<Goldilocks>>() };
        assert!(prefix.is_empty());
        assert!(suffix.is_empty());

        let our_output = extract_output_from_trace(&rows[..NUM_ROUNDS]);
        assert_eq!(
            our_output, expected_output,
            "Keccak-f on zero state should match p3-keccak"
        );
    }

    #[test]
    fn test_keccak_permutation_known_vector() {
        // Known test vector: state with only first element set to 1
        let mut input = [0u64; 25];
        input[0] = 1;

        let mut expected_output = input;
        KeccakF.permute_mut(&mut expected_output);

        let trace = generate_trace_rows::<Goldilocks>(vec![input], 0);
        let (prefix, rows, suffix) = unsafe { trace.values.align_to::<KeccakCols<Goldilocks>>() };
        assert!(prefix.is_empty());
        assert!(suffix.is_empty());

        let our_output = extract_output_from_trace(&rows[..NUM_ROUNDS]);
        assert_eq!(
            our_output, expected_output,
            "Keccak-f with input[0]=1 should match p3-keccak"
        );
    }

    #[test]
    fn test_multiple_permutations() {
        // Test multiple permutations in a single trace
        let inputs: Vec<[u64; 25]> = (0..4)
            .map(|i| core::array::from_fn(|j| (i * 25 + j) as u64))
            .collect();

        let expected_outputs: Vec<[u64; 25]> = inputs
            .iter()
            .map(|input| {
                let mut output = *input;
                KeccakF.permute_mut(&mut output);
                output
            })
            .collect();

        let trace = generate_trace_rows::<Goldilocks>(inputs, 0);
        let (prefix, rows, suffix) = unsafe { trace.values.align_to::<KeccakCols<Goldilocks>>() };
        assert!(prefix.is_empty());
        assert!(suffix.is_empty());

        for (i, expected) in expected_outputs.iter().enumerate() {
            let start = i * NUM_ROUNDS;
            let our_output = extract_output_from_trace(&rows[start..start + NUM_ROUNDS]);
            assert_eq!(
                our_output, *expected,
                "Permutation {} should match p3-keccak",
                i
            );
        }
    }

    #[test]
    fn test_input_output_limb_indexing() {
        // Verify that input_limb and output_limb functions use correct indexing
        // This tests the column mapping for preimage and output

        let input: [u64; 25] = core::array::from_fn(|i| i as u64 + 1);
        let trace = generate_trace_rows::<Goldilocks>(vec![input], 0);
        let (prefix, rows, suffix) = unsafe { trace.values.align_to::<KeccakCols<Goldilocks>>() };
        assert!(prefix.is_empty());
        assert!(suffix.is_empty());

        // Check that preimage is stored in y-major order as per Keccak spec
        let first_row = &rows[0];
        for (i_u64, &expected_val) in input.iter().enumerate() {
            let y = i_u64 / 5;
            let x = i_u64 % 5;

            let mut stored_value = 0u64;
            for limb in 0..U64_LIMBS {
                let limb_val = first_row.preimage[y][x][limb].as_canonical_u64();
                stored_value |= limb_val << (limb * 16);
            }

            // input[i_u64] should be stored at preimage[y][x] where i_u64 = x + 5*y
            // So input[x + 5*y] should equal preimage[y][x]
            assert_eq!(
                stored_value, expected_val,
                "preimage[{}][{}] should equal input[{}]",
                y, x, i_u64
            );
        }
    }
}
