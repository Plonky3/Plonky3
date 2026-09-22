use alloc::vec;
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

/// Number of trace rows packed into one `u64`.
const BITS_PER_WORD: usize = 64;

/// Column of the output flag `round_flags[NUM_ROUNDS]`, the only set cell of a padding row.
///
/// The round flags lead the row.
const OUTPUT_FLAG_COLUMN: usize = NUM_ROUNDS;

/// Build the trace of [`generate_binary_trace_rows`] packed into one `u64` per 64 trace rows.
///
/// The returned matrix keeps the AIR columns as its width. Physical row `w` stores logical
/// rows `64 * w..64 * w + 63`, with bit zero holding the first logical row. The bits of the
/// last block past the trace height are zero. The witness is built as the words its cells are
/// the bits of, so the generic field parameter names only the characteristic the cells are read
/// in; no cell is ever held in it.
///
/// # Panics
///
/// - The field does not have characteristic 2.
/// - `inputs` is empty.
#[instrument(name = "generate packed binary Keccak trace", skip_all)]
#[allow(clippy::needless_pass_by_value)]
pub fn generate_binary_trace_packed<F: Field>(inputs: Vec<[u64; 25]>) -> RowMajorMatrix<u64> {
    assert!(
        F::TWO == F::ZERO,
        "the binary Keccak AIR requires a field of characteristic 2"
    );
    assert!(!inputs.is_empty(), "at least one permutation is required");

    let num_perm_rows = inputs.len() * KECCAK_BINARY_ROWS_PER_PERM;
    let num_rows = num_perm_rows.next_power_of_two();
    let num_blocks = num_rows.div_ceil(BITS_PER_WORD);

    let mut words = vec![0u64; num_blocks * NUM_KECCAK_BINARY_COLS];
    words
        .par_chunks_exact_mut(NUM_KECCAK_BINARY_COLS)
        .enumerate()
        .for_each_init(
            // One permutation's 25 rows per worker keep temporary storage bounded by 25 rows of
            // witness words, independently of the number of trace rows and blocks.
            || [RowWords::default(); KECCAK_BINARY_ROWS_PER_PERM],
            |rows, (block_index, block)| {
                pack_block(block, block_index, &inputs, num_rows, rows);
            },
        );

    RowMajorMatrix::new(words, NUM_KECCAK_BINARY_COLS)
}

/// The witness words of one row: the one-hot row kind, then the state lane by lane.
///
/// The state is stored as the AIR stores it, lane `5y + x`, so the lanes are the column groups
/// of the row in order.
#[derive(Clone, Copy, Default)]
struct RowWords {
    /// The one-hot row kind, bit `r` for round `r` and bit `NUM_ROUNDS` for an output row.
    flags: u32,
    /// The state at the start of the row.
    state: [u64; 25],
}

/// Fill the words of the 25 rows of one permutation.
///
/// Writes exactly what [`generate_perm_rows`] writes, as the words the cells are read from.
fn generate_perm_words(rows: &mut [RowWords; KECCAK_BINARY_ROWS_PER_PERM], input: [u64; 25]) {
    let (round_rows, output_row) = rows.split_at_mut(NUM_ROUNDS);

    let mut state = input;
    for (round, row) in round_rows.iter_mut().enumerate() {
        row.flags = 1 << round;
        row.state = state;
        keccak_round(&mut state, round);
    }

    output_row[0].flags = 1 << NUM_ROUNDS;
    output_row[0].state = state;
}

/// Pack the logical rows `64 * block_index..64 * block_index + 63` of the trace into `block`.
///
/// Each permutation overlapping the block is regenerated into `rows`, the 25 rows of one
/// permutation as witness words, and the bits of its rows inside the block are read out of them.
fn pack_block(
    block: &mut [u64],
    block_index: usize,
    inputs: &[[u64; 25]],
    num_rows: usize,
    rows: &mut [RowWords; KECCAK_BINARY_ROWS_PER_PERM],
) {
    let num_perm_rows = inputs.len() * KECCAK_BINARY_ROWS_PER_PERM;
    let block_start = block_index * BITS_PER_WORD;
    let block_end = (block_start + BITS_PER_WORD).min(num_rows);
    let perm_rows_end = block_end.min(num_perm_rows);

    if block_start < perm_rows_end {
        let first_perm = block_start / KECCAK_BINARY_ROWS_PER_PERM;
        let last_perm = (perm_rows_end - 1) / KECCAK_BINARY_ROWS_PER_PERM;
        for (offset, &input) in inputs[first_perm..=last_perm].iter().enumerate() {
            generate_perm_words(rows, input);

            let perm_start = (first_perm + offset) * KECCAK_BINARY_ROWS_PER_PERM;
            let start = block_start.max(perm_start);
            let end = perm_rows_end.min(perm_start + KECCAK_BINARY_ROWS_PER_PERM);
            for row in start..end {
                let words = &rows[row - perm_start];
                // The row kind is one-hot, so exactly one flag column takes a bit.
                block[words.flags.trailing_zeros() as usize] |= 1u64 << (row - block_start);

                // A lane's cells are its bits, so its columns take them in one sweep.
                let shift = row - block_start;
                let mut column = KECCAK_BINARY_ROWS_PER_PERM;
                for &lane in &words.state {
                    let columns = &mut block[column..column + u64::BITS as usize];
                    for (index, word) in columns.iter_mut().enumerate() {
                        *word |= ((lane >> index) & 1) << shift;
                    }
                    column += u64::BITS as usize;
                }
            }
        }
    }

    // The state of a padding row is zero.
    for row in block_start.max(num_perm_rows)..block_end {
        block[OUTPUT_FLAG_COLUMN] |= 1u64 << (row - block_start);
    }
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

    use p3_binary_field::{BinaryField128, Gf2};
    use p3_field::PrimeCharacteristicRing;
    use p3_goldilocks::Goldilocks;
    use p3_keccak::KeccakF;
    use p3_matrix::Matrix;
    use p3_symmetric::Permutation;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;
    use crate::binary::KeccakBinaryAir;

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

    /// Bit of a packed trace at a logical row and column.
    fn packed_bit(packed: &RowMajorMatrix<u64>, row: usize, column: usize) -> bool {
        (packed.values[(row / 64) * packed.width + column] >> (row % 64)) & 1 == 1
    }

    /// Require a packed trace to hold exactly the cells of a dense trace, with zero padding bits.
    fn assert_packed_matches_dense(dense: &RowMajorMatrix<F>, packed: &RowMajorMatrix<u64>) {
        let height = dense.height();
        assert_eq!(packed.width, NUM_KECCAK_BINARY_COLS);
        assert_eq!(packed.height(), height.div_ceil(64));
        for row in 0..height {
            for column in 0..NUM_KECCAK_BINARY_COLS {
                assert_eq!(
                    dense.values[row * NUM_KECCAK_BINARY_COLS + column],
                    F::from_bool(packed_bit(packed, row, column)),
                    "row {row}, column {column}"
                );
            }
        }
        if !height.is_multiple_of(64) {
            for word in &packed.values[packed.width * (packed.height() - 1)..] {
                assert_eq!(*word >> (height % 64), 0);
            }
        }
    }

    #[test]
    fn packed_trace_matches_dense_trace() {
        // Permutations of 25 rows straddle the 64-row blocks:
        //
        //     1 permutation      : height 32, one partial block
        //     2 permutations     : height 64, one full block
        //     3, 5 permutations  : height 128, padding rows past row 75 or 125
        //     6, 7 permutations  : height 256, the last block has padding rows only
        //     41 permutations    : height 2048, permutation rows end past row 1024
        //     62 permutations    : height 2048, block 24 mixes permutation and padding rows
        //     64 permutations    : height 2048, permutation rows end on the block boundary 1600
        //     65 permutations    : height 2048, permutation 64 starts on the block boundary 1600
        let mut rng = SmallRng::seed_from_u64(11);
        for num_hashes in [1usize, 2, 3, 5, 6, 7, 41, 62, 64, 65] {
            let inputs: Vec<[u64; 25]> = (0..num_hashes).map(|_| rng.random()).collect();
            let dense = generate_binary_trace_rows::<F>(inputs.clone(), 0);
            let packed = generate_binary_trace_packed::<Gf2>(inputs);
            assert_packed_matches_dense(&dense, &packed);
        }
    }

    #[test]
    fn packed_bits_do_not_depend_on_the_temporary_field() {
        let mut rng = SmallRng::seed_from_u64(12);
        let inputs: Vec<[u64; 25]> = (0..6).map(|_| rng.random()).collect();
        assert_eq!(
            generate_binary_trace_packed::<Gf2>(inputs.clone()).values,
            generate_binary_trace_packed::<F>(inputs).values
        );
    }

    #[test]
    fn packed_random_trace_uses_the_dense_generator_sequence() {
        let air = KeccakBinaryAir::default();
        for num_hashes in [1usize, 3, 6] {
            let dense = air.generate_random_trace_rows::<F>(num_hashes, 0);
            let packed = air.generate_random_trace_packed::<Gf2>(num_hashes);
            assert_packed_matches_dense(&dense, &packed);
        }
    }

    #[test]
    #[should_panic(expected = "characteristic 2")]
    fn packed_generator_rejects_odd_characteristic() {
        let _ = generate_binary_trace_packed::<Goldilocks>(vec![[0; 25]]);
    }

    #[test]
    #[should_panic(expected = "at least one permutation")]
    fn packed_generator_rejects_empty_input() {
        let _ = generate_binary_trace_packed::<Gf2>(Vec::new());
    }
}
