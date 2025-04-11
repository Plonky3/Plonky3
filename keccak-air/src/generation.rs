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
    let mut current_state: [[u64; 5]; 5] = unsafe { transmute(input) };

    let initial_state: [[[F; 4]; 5]; 5] =
        array::from_fn(|y| array::from_fn(|x| u64_to_16_bit_limbs(current_state[y][x])));

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

mod tests{
    use super::*;
    use alloc::vec::Vec; 
    use p3_goldilocks::Goldilocks;
    use p3_field::PrimeCharacteristicRing;

      // Create a default KeccakCols instance
    fn default_keccak_cols<F: PrimeField64 + PrimeCharacteristicRing>() -> KeccakCols<F> {
        KeccakCols {
            step_flags: [F::ZERO; NUM_ROUNDS],
            export: F::ZERO,
            preimage: [[[F::ZERO; U64_LIMBS]; 5]; 5],
            a: [[[F::ZERO; U64_LIMBS]; 5]; 5],
            c: [[F::ZERO; 64]; 5],
            c_prime: [[F::ZERO; 64]; 5],
            a_prime: [[[F::ZERO; 64]; 5]; 5],
            a_prime_prime: [[[F::ZERO; U64_LIMBS]; 5]; 5],
            a_prime_prime_0_0_bits: [F::ZERO; 64],
            a_prime_prime_prime_0_0_limbs: [F::ZERO; U64_LIMBS],
        }
    }


     // Verify Keccak permutation with any test vector
    fn verify_keccak_permutation(input: [u64; 25], expected: [u64; 25]) {
        // Generate trace for the Keccak permutation
        let mut rows: Vec<KeccakCols<Goldilocks>> = (0..NUM_ROUNDS).map(|_| default_keccak_cols()).collect();
        generate_trace_rows_for_perm(&mut rows, input);
        
        // Extract the final state from the trace
        let mut final_state = [[0u64; 5]; 5];
        for y in 0..5 {
            for x in 0..5 {
                // Reconstruct the u64 from 4 16-bit limbs
                let mut value = 0u64;
                for i in 0..U64_LIMBS {
                    let limb = rows[NUM_ROUNDS - 1].a_prime_prime_prime(y, x, i);
                    value |= (limb.as_canonical_u64() & 0xFFFF) << (16 * i);
                }
                final_state[y][x] = value;
            }
        }
        // Flatten the 2D array for comparison
        let mut final_flat = [0u64; 25];
        for y in 0..5 {
            for x in 0..5 {
                final_flat[x + 5 * y] = final_state[y][x];
            }
        }
        
        // Verify against expected output
        assert_eq!(final_flat, expected, "Keccak permutation output doesn't match expected values");
    }


    #[test]
    fn test_keccak_trace_round_transition() {
        // Create a test input with unique values
        let mut input = [0u64; 25];
        for i in 0..25 {
            input[i] = (i as u64 + 1) * 0x1111;
        }

        // Generate trace for the Keccak permutation
        let mut rows: Vec<KeccakCols<Goldilocks>> = (0..NUM_ROUNDS).map(|_| default_keccak_cols()).collect();
        generate_trace_rows_for_perm(&mut rows, input);

        // Verify that the output of round 0 (a_prime_prime_prime) becomes the input of round 1 (a)
        for y in 0..5 {
            for x in 0..5 {
                for limb in 0..U64_LIMBS {
                    assert_eq!(
                        rows[1].a[y][x][limb],
                        rows[0].a_prime_prime_prime(y, x, limb),
                        "State transition mismatch at position ({}, {}) limb {} between round 0 and 1",
                        x, y, limb
                    );
                }
            }
        }

        // Verify that the preimage remains the same across rounds
        for round in 1..NUM_ROUNDS {
            for y in 0..5 {
                for x in 0..5 {
                    for limb in 0..U64_LIMBS {
                        assert_eq!(
                            rows[round].preimage[y][x][limb],
                            rows[0].preimage[y][x][limb],
                            "Preimage mismatch at round {}, position ({}, {}) limb {}",
                            round, x, y, limb
                        );
                    }
                }
            }
        }
    }


    #[test]
    fn test_keccak_permutation_zero_input() {
        let input = [0u64; 25];
        let expected = [
            0xF1258F7940E1DDE7, 
            0x84D5CCF933C0478A, 
            0xD598261EA65AA9EE, 
            0xBD1547306F80494D,
            0x8B284E056253D057, 
            0xFF97A42D7F8E6FD4, 
            0x90FEE5A0A44647C4, 
            0x8C5BDA0CD6192E76,
            0xAD30A6F71B19059C, 
            0x30935AB7D08FFC64, 
            0xEB5AA93F2317D635, 
            0xA9A6E6260D712103,
            0x81A57C16DBCF555F, 
            0x43B831CD0347C826, 
            0x01F22F1A11A5569F, 
            0x05E5635A21D9AE61,
            0x64BEFEF28CC970F2, 
            0x613670957BC46611, 
            0xB87C5A554FD00ECB, 
            0x8C3EE88A1CCF32C8,
            0x940C7922AE3A2614, 
            0x1841F924A2C509E4, 
            0x16F53526E70465C2, 
            0x75F644E97F30A13B,
            0xEAF1FF7B5CECA249,
        ];
        verify_keccak_permutation(input, expected);

    }

    
    #[test]
    fn test_keccak_state_indexing() {
        // Create a test state with unique values for each position
        let mut input = [0u64; 25];
        for i in 0..25 {
            input[i] = i as u64 + 1; 
        }
        
        // Convert to 2D state using unsafe transmute (as in the implementation)
        let state_2d: [[u64; 5]; 5] = unsafe { transmute(input) };
        
        // Verify that the indexing matches the expected convention: state[x + 5*y]
        for y in 0..5 {
            for x in 0..5 {
                // Check that the value at state_2d[y][x] is the same as input[x + 5*y]
                // This confirms we're using the correct indexing convention
                assert_eq!(state_2d[y][x], input[x + 5 * y]);
            }
        }
    }
    
   

}


