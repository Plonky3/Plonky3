use alloc::vec::Vec;
use core::array;

use p3_air::utils::u64_to_bits_le;
use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use tracing::instrument;

use super::columns::{KECCAK_BINARY_ROWS_PER_PERM, KeccakBinaryCols, NUM_KECCAK_BINARY_COLS};
use super::rho_pi_source;
use crate::{NUM_ROUNDS, RC};

/// Build the trace of the characteristic-2 Keccak-f AIR for the given permutation inputs.
///
/// Each input is a Keccak-f state indexed by lane `5y + x`.
/// It occupies 25 rows: the input states of the 24 rounds, then the permutation output.
/// The height is the next power of two of `25 * inputs.len()`.
/// Rows past the last permutation are output rows with an all-zero state.
///
/// # Panics
///
/// - The field does not have characteristic 2.
/// - `inputs` is empty.
#[instrument(name = "generate binary Keccak trace", skip_all)]
pub fn generate_binary_trace_rows<F: Field>(
    inputs: Vec<[u64; 25]>,
    extra_capacity_bits: usize,
) -> RowMajorMatrix<F> {
    assert!(
        F::TWO == F::ZERO,
        "the binary Keccak AIR requires a field of characteristic 2"
    );
    assert!(!inputs.is_empty(), "at least one permutation is required");

    let num_perm_rows = inputs.len() * KECCAK_BINARY_ROWS_PER_PERM;
    let num_rows = num_perm_rows.next_power_of_two();
    let trace_length = num_rows * NUM_KECCAK_BINARY_COLS;

    // Reserve the extra capacity the low-degree extension will need.
    let mut long_trace = F::zero_vec(trace_length << extra_capacity_bits);
    long_trace.truncate(trace_length);

    let mut trace = RowMajorMatrix::new(long_trace, NUM_KECCAK_BINARY_COLS);
    let (prefix, rows, suffix) = unsafe { trace.values.align_to_mut::<KeccakBinaryCols<F>>() };
    assert!(prefix.is_empty(), "Alignment should match");
    assert!(suffix.is_empty(), "Alignment should match");
    assert_eq!(rows.len(), num_rows);

    let (perm_rows, padding_rows) = rows.split_at_mut(num_perm_rows);

    perm_rows
        .par_chunks_mut(KECCAK_BINARY_ROWS_PER_PERM)
        .zip(inputs.into_par_iter())
        .for_each(|(rows, input)| generate_perm_rows(rows, input));

    // The state of a padding row is already zero.
    padding_rows
        .par_iter_mut()
        .for_each(|row| row.round_flags[NUM_ROUNDS] = F::ONE);

    trace
}

/// Fill the 25 rows of one permutation, starting from a zeroed buffer.
fn generate_perm_rows<F: Field>(rows: &mut [KeccakBinaryCols<F>], input: [u64; 25]) {
    let (round_rows, output_row) = rows.split_at_mut(NUM_ROUNDS);

    let mut state = input;
    for (round, row) in round_rows.iter_mut().enumerate() {
        row.round_flags[round] = F::ONE;
        write_state(row, &state);
        keccak_round(&mut state, round);
    }

    output_row[0].round_flags[NUM_ROUNDS] = F::ONE;
    write_state(&mut output_row[0], &state);
}

/// Write the bits of a state, indexed by lane `5y + x`, into the state columns of a row.
pub(super) fn write_state<F: Field>(row: &mut KeccakBinaryCols<F>, state: &[u64; 25]) {
    for (y, plane) in row.a.iter_mut().enumerate() {
        for (x, lane) in plane.iter_mut().enumerate() {
            *lane = u64_to_bits_le(state[5 * y + x]);
        }
    }
}

/// Apply round `round` of Keccak-f to a state indexed by lane `5y + x`.
pub(super) fn keccak_round(state: &mut [u64; 25], round: usize) {
    // Theta:
    //     C[x] = xor_y A[x, y]
    //     D[x] = C[x - 1] ^ ROT(C[x + 1], 1)
    let c: [u64; 5] = array::from_fn(|x| (0..5).fold(0, |acc, y| acc ^ state[5 * y + x]));
    let d: [u64; 5] = array::from_fn(|x| c[(x + 4) % 5] ^ c[(x + 1) % 5].rotate_left(1));

    // Rho and pi, applied to the theta output A'[x, y] = A[x, y] ^ D[x].
    let b: [u64; 25] = array::from_fn(|i| {
        let (source_y, source_x, rot) = rho_pi_source(i % 5, i / 5);
        (state[5 * source_y + source_x] ^ d[source_x]).rotate_left(rot as u32)
    });

    // Chi: A''[x, y] = B[x, y] ^ (!B[x + 1, y] & B[x + 2, y]).
    *state = array::from_fn(|i| {
        let (y, x) = (i / 5, i % 5);
        b[i] ^ (!b[5 * y + (x + 1) % 5] & b[5 * y + (x + 2) % 5])
    });

    // Iota.
    state[0] ^= RC[round];
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use core::borrow::Borrow;

    use p3_binary_field::BinaryField128;
    use p3_field::PrimeCharacteristicRing;
    use p3_goldilocks::Goldilocks;
    use p3_keccak::KeccakF;
    use p3_matrix::Matrix;
    use p3_symmetric::Permutation;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;

    type F = BinaryField128;

    /// Borrow row `r` of a trace as typed columns.
    fn row(trace: &RowMajorMatrix<F>, r: usize) -> &KeccakBinaryCols<F> {
        trace.values[r * trace.width..(r + 1) * trace.width].borrow()
    }

    /// Read the state columns of a row back into lanes indexed by `5y + x`.
    fn read_state(row: &KeccakBinaryCols<F>) -> [u64; 25] {
        array::from_fn(|i| {
            row.a[i / 5][i % 5]
                .iter()
                .enumerate()
                .fold(0, |acc, (z, &bit)| {
                    assert!(bit == F::ZERO || bit == F::ONE, "state cell is not a bit");
                    acc | (u64::from(bit == F::ONE) << z)
                })
        })
    }

    /// Index of the single set flag of a row.
    fn row_kind(row: &KeccakBinaryCols<F>) -> usize {
        let set: Vec<usize> = (0..KECCAK_BINARY_ROWS_PER_PERM)
            .filter(|&i| row.round_flags[i] != F::ZERO)
            .collect();
        assert_eq!(set.len(), 1, "round flags are not one-hot");
        assert_eq!(row.round_flags[set[0]], F::ONE);
        set[0]
    }

    #[test]
    fn output_rows_match_keccak_f() {
        // Three random permutations: 75 rows, padded to 128.
        //
        //     rows 25i..25i+24 : input states of rounds 0..23 of permutation i
        //     row  25i+24      : KeccakF(input_i)
        //     rows 75..128     : padding, output flag and zero state
        let mut rng = SmallRng::seed_from_u64(7);
        let inputs: Vec<[u64; 25]> = (0..3).map(|_| rng.random()).collect();

        let trace = generate_binary_trace_rows::<F>(inputs.clone(), 0);
        assert_eq!(trace.width(), NUM_KECCAK_BINARY_COLS);
        assert_eq!(trace.height(), 128);

        for (i, input) in inputs.iter().enumerate() {
            let base = i * KECCAK_BINARY_ROWS_PER_PERM;

            for round in 0..NUM_ROUNDS {
                assert_eq!(row_kind(row(&trace, base + round)), round);
            }
            assert_eq!(read_state(row(&trace, base)), *input);

            let mut expected = *input;
            KeccakF.permute_mut(&mut expected);
            let output = row(&trace, base + NUM_ROUNDS);
            assert_eq!(row_kind(output), NUM_ROUNDS);
            assert_eq!(read_state(output), expected, "permutation {i}");
        }

        for r in 3 * KECCAK_BINARY_ROWS_PER_PERM..trace.height() {
            assert_eq!(row_kind(row(&trace, r)), NUM_ROUNDS);
            assert_eq!(read_state(row(&trace, r)), [0; 25]);
        }
    }

    #[test]
    fn heights_round_up_to_a_power_of_two() {
        for (num_hashes, height) in [(1, 32), (2, 64), (3, 128), (5, 128), (6, 256)] {
            let trace = generate_binary_trace_rows::<F>(vec![[0; 25]; num_hashes], 0);
            assert_eq!(trace.height(), height);
        }
    }

    #[test]
    #[should_panic(expected = "characteristic 2")]
    fn rejects_odd_characteristic() {
        let _ = generate_binary_trace_rows::<Goldilocks>(vec![[0; 25]], 0);
    }

    #[test]
    #[should_panic(expected = "at least one permutation")]
    fn rejects_empty_input() {
        let _ = generate_binary_trace_rows::<F>(Vec::new(), 0);
    }
}
