use alloc::vec;
use alloc::vec::Vec;
use core::iter;

use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;

use crate::columns::{KeccakCols, NUM_KECCAK_COLS};
use crate::constants::rc_value_limb;
use crate::logic::{andn, xor};
use crate::{BITS_PER_LIMB, NUM_ROUNDS, U64_LIMBS};

pub fn generate_trace_rows<F: PrimeField32>(
    inputs: Vec<[u64; 25]>,
    min_rows: usize,
) -> RowMajorMatrix<F> {
    let num_rows = (inputs.len() * NUM_ROUNDS)
        .max(min_rows)
        .next_power_of_two();
    let mut trace = RowMajorMatrix::new(vec![F::ZERO; num_rows * NUM_KECCAK_COLS], NUM_KECCAK_COLS);
    let (prefix, rows, suffix) = unsafe { trace.values.align_to_mut::<KeccakCols<F>>() };
    assert!(prefix.is_empty(), "Data was not aligned");
    assert!(suffix.is_empty(), "Data was not aligned");
    assert_eq!(rows.len(), num_rows);

    let padded_inputs = inputs.into_iter().chain(iter::repeat([0; 25]));
    for (row, input) in rows.chunks_mut(NUM_ROUNDS).zip(padded_inputs) {
        generate_trace_rows_for_perm(row, input);
    }

    trace
}

/// `rows` will normally consist of 24 rows, with an exception for the final row.
fn generate_trace_rows_for_perm<F: PrimeField32>(rows: &mut [KeccakCols<F>], input: [u64; 25]) {
    // Populate the preimage for each row.
    for row in rows.iter_mut() {
        for y in 0..5 {
            for x in 0..5 {
                let input_xy = input[y * 5 + x];
                for limb in 0..U64_LIMBS {
                    row.preimage[y][x][limb] =
                        F::from_canonical_u64((input_xy >> (16 * limb)) & 0xFFFF);
                }
            }
        }
    }

    // Populate the round input for the first round.
    for y in 0..5 {
        for x in 0..5 {
            let input_xy = input[y * 5 + x];
            for limb in 0..U64_LIMBS {
                rows[0].a[y][x][limb] = F::from_canonical_u64((input_xy >> (16 * limb)) & 0xFFFF);
            }
        }
    }

    generate_trace_row_for_round(&mut rows[0], 0);

    for round in 1..rows.len() {
        // Copy previous row's output to next row's input.
        for y in 0..5 {
            for x in 0..5 {
                for limb in 0..U64_LIMBS {
                    rows[round].a[y][x][limb] = rows[round - 1].a_prime_prime_prime(x, y, limb);
                }
            }
        }

        generate_trace_row_for_round(&mut rows[round], round);
    }
}

fn generate_trace_row_for_round<F: PrimeField32>(row: &mut KeccakCols<F>, round: usize) {
    row.step_flags[round] = F::ONE;

    // Populate C[x] = xor(A[x, 0], A[x, 1], A[x, 2], A[x, 3], A[x, 4]).
    for x in 0..5 {
        for z in 0..64 {
            let limb = z / BITS_PER_LIMB;
            let bit_in_limb = z % BITS_PER_LIMB;
            let a = [0, 1, 2, 3, 4].map(|i| {
                let a_limb = row.a[i][x][limb].as_canonical_u32() as u16;
                F::from_bool(((a_limb >> bit_in_limb) & 1) != 0)
            });
            row.c[x][z] = xor(a);
        }
    }

    // Populate C'[x, z] = xor(C[x, z], C[x - 1, z], C[x + 1, z - 1]).
    for x in 0..5 {
        for z in 0..64 {
            row.c_prime[x][z] = xor([
                row.c[x][z],
                row.c[(x + 4) % 5][z],
                row.c[(x + 1) % 5][(z + 63) % 64],
            ]);
        }
    }

    // Populate A'. To avoid shifting indices, we rewrite
    //     A'[x, y, z] = xor(A[x, y, z], C[x - 1, z], C[x + 1, z - 1])
    // as
    //     A'[x, y, z] = xor(A[x, y, z], C[x, z], C'[x, z]).
    for x in 0..5 {
        for y in 0..5 {
            for z in 0..64 {
                let limb = z / BITS_PER_LIMB;
                let bit_in_limb = z % BITS_PER_LIMB;
                let a_limb = row.a[y][x][limb].as_canonical_u32() as u16;
                let a_bit = F::from_bool(((a_limb >> bit_in_limb) & 1) != 0);
                row.a_prime[x][y][z] = xor([a_bit, row.c[x][z], row.c_prime[x][z]]);
            }
        }
    }

    // Populate A''.
    // A''[x, y] = xor(B[x, y], andn(B[x + 1, y], B[x + 2, y])).
    for y in 0..5 {
        for x in 0..5 {
            for limb in 0..U64_LIMBS {
                row.a_prime_prime[y][x][limb] = (limb * BITS_PER_LIMB..(limb + 1) * BITS_PER_LIMB)
                    .rev()
                    .fold(F::ZERO, |acc, z| {
                        let bit = xor([
                            row.b(x, y, z),
                            andn(row.b((x + 1) % 5, y, z), row.b((x + 2) % 5, y, z)),
                        ]);
                        acc.double() + bit
                    });
            }
        }
    }

    // For the XOR, we split A''[0, 0] to bits.
    let mut val = 0;
    for limb in 0..U64_LIMBS {
        let val_limb = row.a_prime_prime[0][0][limb].as_canonical_u32() as u64;
        val |= val_limb << (limb * BITS_PER_LIMB);
    }
    let val_bits: Vec<bool> = (0..64)
        .scan(val, |acc, _| {
            let bit = (*acc & 1) != 0;
            *acc >>= 1;
            Some(bit)
        })
        .collect();
    for (i, bit) in row.a_prime_prime_0_0_bits.iter_mut().enumerate() {
        *bit = F::from_bool(val_bits[i]);
    }

    // A''[0, 0] is additionally xor'd with RC.
    for limb in 0..U64_LIMBS {
        let rc_lo = rc_value_limb(round, limb);
        row.a_prime_prime_prime_0_0_limbs[limb] =
            F::from_canonical_u16(row.a_prime_prime[0][0][limb].as_canonical_u32() as u16 ^ rc_lo);
    }
}
