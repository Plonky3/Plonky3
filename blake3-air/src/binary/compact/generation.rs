use alloc::vec;
use alloc::vec::Vec;
use core::array;
use core::borrow::BorrowMut;

use p3_air::utils::u32_to_bits_le;
use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use tracing::instrument;

use super::super::generation::{g_words, generate_g};
use super::super::{Blake3CompressionInput, G_SCHEDULE, NUM_ROUNDS, iv_word};
use super::BLAKE3_BINARY_COMPACT_ROWS_PER_COMPRESSION;
use super::columns::{Blake3BinaryCompactCols, NUM_BLAKE3_BINARY_COMPACT_COLS};
use crate::constants::permute;

/// Generate a compact binary Blake-3 trace with eight logical rows per compression.
///
/// Rows zero through six of each group contain its rounds. Row seven is a zero dummy
/// round. The compression inputs are exposed on the group's first row and the output
/// is recovered from its sixth-offset row.
///
/// # Panics
///
/// Panics unless the field has characteristic two and the input count is a nonzero
/// power of two. Callers must pad inputs to the desired compression count.
#[instrument(name = "generate compact Blake3 binary trace", skip_all)]
pub fn generate_binary_compact_trace_rows<F: Field>(
    inputs: Vec<Blake3CompressionInput>,
    extra_capacity_bits: usize,
) -> RowMajorMatrix<F> {
    validate::<F>(&inputs);
    let group_size = BLAKE3_BINARY_COMPACT_ROWS_PER_COMPRESSION * NUM_BLAKE3_BINARY_COMPACT_COLS;
    let length = inputs.len() * group_size;
    let mut values = F::zero_vec(length << extra_capacity_bits);
    values.truncate(length);
    values
        .par_chunks_exact_mut(group_size)
        .zip(inputs)
        .for_each(|(group, input)| {
            let mut state = initial_state(&input);
            let mut message = input.block;
            let (rows, remainder) = group.as_chunks_mut::<NUM_BLAKE3_BINARY_COMPACT_COLS>();
            debug_assert!(remainder.is_empty());
            for values in rows.iter_mut().take(NUM_ROUNDS) {
                let row: &mut Blake3BinaryCompactCols<F> = values.as_mut_slice().borrow_mut();
                row.chaining_value = input.chaining_value.map(u32_to_bits_le);
                row.block = message.map(u32_to_bits_le);
                row.state = array::from_fn(|i| u32_to_bits_le(state[i / 4][i % 4]));
                for (g, (cols, slots)) in row.round.iter_mut().zip(G_SCHEDULE).enumerate() {
                    generate_g(cols, &mut state, slots, message[2 * g], message[2 * g + 1]);
                }
                permute(&mut message);
            }
        });
    RowMajorMatrix::new(values, NUM_BLAKE3_BINARY_COMPACT_COLS)
}

/// Generate the compact binary trace packed into one `u64` per 64 logical rows.
///
/// Physical row `w`, bit `b` represents logical row `64 * w + b`. Each physical
/// row holds eight compression groups. Padding rows and unused tail bits are zero.
/// Witness arithmetic uses integer words and writes bits directly; it does not
/// allocate a field-valued trace.
///
/// # Panics
///
/// Panics under the same conditions as [`generate_binary_compact_trace_rows`].
#[instrument(name = "generate packed compact Blake3 binary trace", skip_all)]
#[allow(clippy::needless_pass_by_value)]
pub fn generate_binary_compact_trace_packed<F: Field>(
    inputs: Vec<Blake3CompressionInput>,
) -> RowMajorMatrix<u64> {
    validate::<F>(&inputs);
    let mut words = vec![0; inputs.len().div_ceil(8) * NUM_BLAKE3_BINARY_COMPACT_COLS];
    inputs
        .par_chunks(8)
        .zip(words.par_chunks_exact_mut(NUM_BLAKE3_BINARY_COMPACT_COLS))
        .for_each(|(inputs, output)| {
            for (input_index, input) in inputs.iter().enumerate() {
                let mut state = initial_state(input);
                let mut message = input.block;
                for round in 0..NUM_ROUNDS {
                    let row_bit = input_index * BLAKE3_BINARY_COMPACT_ROWS_PER_COMPRESSION + round;
                    let mut column = 0;
                    for &word in input
                        .chaining_value
                        .iter()
                        .chain(&message)
                        .chain(state.iter().flatten())
                    {
                        write_word(output, &mut column, word, 32, row_bit);
                    }
                    for (g, slots) in G_SCHEDULE.into_iter().enumerate() {
                        let witness =
                            g_words(&mut state, slots, message[2 * g], message[2 * g + 1]);
                        write_word(output, &mut column, witness.add1_carries, 31, row_bit);
                        write_word(output, &mut column, witness.d1, 32, row_bit);
                        write_word(output, &mut column, witness.b1, 32, row_bit);
                        write_word(output, &mut column, witness.add2_carries, 31, row_bit);
                        write_word(output, &mut column, witness.d2, 32, row_bit);
                        write_word(output, &mut column, witness.b2, 32, row_bit);
                    }
                    debug_assert_eq!(column, NUM_BLAKE3_BINARY_COMPACT_COLS);
                    permute(&mut message);
                }
            }
        });
    RowMajorMatrix::new(words, NUM_BLAKE3_BINARY_COMPACT_COLS)
}

fn write_word(output: &mut [u64], column: &mut usize, word: u32, width: usize, row_bit: usize) {
    for (bit, value) in output[*column..*column + width].iter_mut().enumerate() {
        *value |= u64::from((word >> bit) & 1) << row_bit;
    }
    *column += width;
}

fn validate<F: Field>(inputs: &[Blake3CompressionInput]) {
    assert_eq!(
        F::TWO,
        F::ZERO,
        "the binary Blake-3 AIR requires a field of characteristic 2"
    );
    assert!(!inputs.is_empty(), "at least one input is required");
    assert!(
        inputs.len().is_power_of_two(),
        "Callers expected to pad inputs to a power of two"
    );
}

fn initial_state(input: &Blake3CompressionInput) -> [[u32; 4]; 4] {
    let cv = input.chaining_value;
    [
        [cv[0], cv[1], cv[2], cv[3]],
        [cv[4], cv[5], cv[6], cv[7]],
        array::from_fn(iv_word),
        [
            input.counter as u32,
            (input.counter >> 32) as u32,
            input.block_len,
            input.flags,
        ],
    ]
}
