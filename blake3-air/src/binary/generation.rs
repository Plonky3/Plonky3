use alloc::vec;
use alloc::vec::Vec;
use core::array;

use p3_air::utils::u32_to_bits_le;
use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use tracing::instrument;

use super::columns::{Blake3BinaryCols, Blake3BinaryGCols, NUM_BLAKE3_BINARY_COLS};
use super::{G_PER_ROUND, G_SCHEDULE, NUM_ROUNDS, iv_word};
use crate::constants::permute;

/// The inputs to one Blake-3 compression.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Blake3CompressionInput {
    /// The chaining value.
    pub chaining_value: [u32; 8],
    /// The message words.
    pub block: [u32; 16],
    /// The block counter.
    pub counter: u64,
    /// Number of message bytes in the block.
    pub block_len: u32,
    /// Domain separation flags.
    pub flags: u32,
}

/// Generate a binary Blake-3 trace with one row per compression.
///
/// # Panics
///
/// Panics if the field does not have characteristic 2, if `inputs` is empty, or if the number of
/// inputs is not a power of two.
#[instrument(name = "generate Blake3 binary trace", skip_all)]
pub fn generate_binary_trace_rows<F: Field>(
    inputs: Vec<Blake3CompressionInput>,
    extra_capacity_bits: usize,
) -> RowMajorMatrix<F> {
    assert_eq!(
        F::TWO,
        F::ZERO,
        "the binary Blake-3 AIR requires a field of characteristic 2"
    );

    let num_rows = inputs.len();
    assert!(num_rows > 0, "at least one input is required");
    assert!(
        num_rows.is_power_of_two(),
        "Callers expected to pad inputs to a power of two"
    );

    let trace_length = num_rows * NUM_BLAKE3_BINARY_COLS;

    // We allocate extra_capacity_bits now as this will be needed by the dft.
    let mut long_trace = F::zero_vec(trace_length << extra_capacity_bits);
    long_trace.truncate(trace_length);

    let mut trace = RowMajorMatrix::new(long_trace, NUM_BLAKE3_BINARY_COLS);
    let (prefix, rows, suffix) = unsafe { trace.values.align_to_mut::<Blake3BinaryCols<F>>() };
    assert!(prefix.is_empty(), "Alignment should match");
    assert!(suffix.is_empty(), "Alignment should match");
    assert_eq!(rows.len(), num_rows);

    rows.par_iter_mut()
        .zip(inputs)
        .for_each(|(row, input)| generate_trace_row(row, &input));

    trace
}

/// Generate a binary Blake-3 trace packed into one `u64` per 64 trace rows.
///
/// The returned matrix keeps the AIR columns as its width. Physical row `w` stores logical
/// rows `64 * w..64 * w + 63`, with bit zero holding the first logical row. The witness is
/// built as the words its cells are the bits of, so the generic field parameter names only the
/// characteristic the cells are read in; no cell is ever held in it.
///
/// # Panics
///
/// Panics if the field does not have characteristic 2, if `inputs` is empty, or if the number of
/// inputs is not a power of two.
#[instrument(name = "generate packed Blake3 binary trace", skip_all)]
#[allow(clippy::needless_pass_by_value)]
pub fn generate_binary_trace_packed<F: Field>(
    inputs: Vec<Blake3CompressionInput>,
) -> RowMajorMatrix<u64> {
    assert_eq!(
        F::TWO,
        F::ZERO,
        "the binary Blake-3 AIR requires a field of characteristic 2"
    );

    let num_rows = inputs.len();
    assert!(num_rows > 0, "at least one input is required");
    assert!(
        num_rows.is_power_of_two(),
        "Callers expected to pad inputs to a power of two"
    );

    let num_blocks = num_rows.div_ceil(64);
    let mut words = vec![0u64; num_blocks * NUM_BLAKE3_BINARY_COLS];
    inputs
        .par_chunks(64)
        .zip(words.par_chunks_exact_mut(NUM_BLAKE3_BINARY_COLS))
        .for_each_init(
            || [0u32; NUM_ROW_WORDS],
            |row, (input_block, block)| {
                // The witness is already words, so a row is generated as words and its bits are
                // read out of registers. One reusable row per worker keeps temporary storage at
                // one word per column group, whatever the trace height.
                for (lane, input) in input_block.iter().enumerate() {
                    generate_row_words(row, input);

                    // Transpose the row into the block: bit `i` of group `g` is the cell of
                    // this lane in the column that group's bit `i` occupies.
                    let mut column = 0;
                    for (&word, &bits) in row.iter().zip(&ROW_WORD_BITS) {
                        let columns = &mut block[column..column + usize::from(bits)];
                        for (index, cell) in columns.iter_mut().enumerate() {
                            *cell |= u64::from((word >> index) & 1) << lane;
                        }
                        column += usize::from(bits);
                    }
                }
            },
        );

    RowMajorMatrix::new(words, NUM_BLAKE3_BINARY_COLS)
}

/// Column groups one row of the binary Blake-3 witness is built from.
///
/// One per stored word: the chaining value, the message block, the four remaining state words,
/// and the six a G step writes.
const NUM_ROW_WORDS: usize = 28 + NUM_ROUNDS * G_PER_ROUND * 6;

/// Cells each column group contributes, in column order.
///
/// Every group is a whole word except the two carry groups of a G step, which have no carry
/// into bit zero.
const ROW_WORD_BITS: [u8; NUM_ROW_WORDS] = {
    let mut bits = [32u8; NUM_ROW_WORDS];
    let mut group = 28;
    while group < NUM_ROW_WORDS {
        bits[group] = 31;
        bits[group + 3] = 31;
        group += 6;
    }
    bits
};

/// Fill one row's witness words, in column order, from a single compression.
///
/// Writes exactly what [`generate_trace_row`] writes, as the words the cells are read from.
fn generate_row_words(row: &mut [u32; NUM_ROW_WORDS], input: &Blake3CompressionInput) {
    let counter_low = input.counter as u32;
    let counter_high = (input.counter >> 32) as u32;

    row[..8].copy_from_slice(&input.chaining_value);
    row[8..24].copy_from_slice(&input.block);
    row[24] = counter_low;
    row[25] = counter_high;
    row[26] = input.block_len;
    row[27] = input.flags;

    let cv = input.chaining_value;
    let mut state = [
        [cv[0], cv[1], cv[2], cv[3]],
        [cv[4], cv[5], cv[6], cv[7]],
        array::from_fn(iv_word),
        [counter_low, counter_high, input.block_len, input.flags],
    ];
    let mut m = input.block;

    let mut group = 28;
    for round_idx in 0..NUM_ROUNDS {
        if round_idx > 0 {
            permute(&mut m);
        }
        for (g, slots) in G_SCHEDULE.into_iter().enumerate() {
            let words = g_words(&mut state, slots, m[2 * g], m[2 * g + 1]);
            row[group] = words.add1_carries;
            row[group + 1] = words.d1;
            row[group + 2] = words.b1;
            row[group + 3] = words.add2_carries;
            row[group + 4] = words.d2;
            row[group + 5] = words.b2;
            group += 6;
        }
    }
}

/// Fill one row with the witness of a single compression.
fn generate_trace_row<F: Field>(row: &mut Blake3BinaryCols<F>, input: &Blake3CompressionInput) {
    let counter_low = input.counter as u32;
    let counter_high = (input.counter >> 32) as u32;

    row.chaining_value = input.chaining_value.map(u32_to_bits_le);
    row.block = input.block.map(u32_to_bits_le);
    row.counter_low = u32_to_bits_le(counter_low);
    row.counter_high = u32_to_bits_le(counter_high);
    row.block_len = u32_to_bits_le(input.block_len);
    row.flags = u32_to_bits_le(input.flags);

    let cv = input.chaining_value;
    let mut state = [
        [cv[0], cv[1], cv[2], cv[3]],
        [cv[4], cv[5], cv[6], cv[7]],
        array::from_fn(iv_word),
        [counter_low, counter_high, input.block_len, input.flags],
    ];
    let mut m = input.block;

    for (round_idx, round) in row.rounds.iter_mut().enumerate() {
        if round_idx > 0 {
            permute(&mut m);
        }
        for (g, (cols, slots)) in round.iter_mut().zip(G_SCHEDULE).enumerate() {
            generate_g(cols, &mut state, slots, m[2 * g], m[2 * g + 1]);
        }
    }
}

impl<F: Field> Blake3BinaryCols<F> {
    /// Recover the compression output of this row from its witness columns.
    ///
    /// The final state words are `a = d1 ^ (d2 <<< 8)`, `b = b2`, `c = b1 ^ (b2 <<< 7)` and
    /// `d = d2` of the last G step that wrote them, and the output is `v[i] ^ v[i + 8]`
    /// followed by `v[i + 8] ^ cv[i]`.
    ///
    /// # Panics
    ///
    /// Panics if a cell read is not a bit.
    pub fn compression_output(&self) -> [u32; 16] {
        let mut v = [0u32; 16];
        for (cols, [ia, ib, ic, id]) in self.rounds[NUM_ROUNDS - 1].iter().zip(G_SCHEDULE) {
            let (d1, b1, d2, b2) = (
                read_word(&cols.d1),
                read_word(&cols.b1),
                read_word(&cols.d2),
                read_word(&cols.b2),
            );
            v[ia] = d1 ^ d2.rotate_left(8);
            v[4 + ib] = b2;
            v[8 + ic] = b1 ^ b2.rotate_left(7);
            v[12 + id] = d2;
        }
        let cv = self.chaining_value.map(|word| read_word(&word));
        array::from_fn(|i| {
            if i < 8 {
                v[i] ^ v[i + 8]
            } else {
                v[i] ^ cv[i - 8]
            }
        })
    }
}

/// Read a word stored as 32 boolean cells, least significant bit first.
///
/// # Panics
///
/// Panics if a cell is not a bit.
fn read_word<F: Field>(bits: &[F; 32]) -> u32 {
    bits.iter().enumerate().fold(0, |word, (i, &bit)| {
        assert!(bit == F::ZERO || bit == F::ONE, "cell is not a bit");
        word | (u32::from(bit == F::ONE) << i)
    })
}

/// The witness words one G step contributes, in column order.
///
/// Each is a word whose low bits are the cells of one column group, lowest bit first.
struct GWords {
    /// Carries into bits `1..32` of `a + b`, shifted down to start at bit zero.
    add1_carries: u32,
    /// The word `d1`.
    d1: u32,
    /// The word `b1`.
    b1: u32,
    /// Carries into bits `1..32` of `a1 + b1`, shifted down to start at bit zero.
    add2_carries: u32,
    /// The output word `d2`.
    d2: u32,
    /// The output word `b2`.
    b2: u32,
}

/// Apply one G step to `state` and return the witness words it contributes.
///
/// `slots` holds the indices of the `a`, `b`, `c`, `d` words within their state rows.
///
/// This is the whole arithmetic of a G step. Both the field rows and the packed words are
/// written from what it returns, so neither can drift from the other.
const fn g_words(
    state: &mut [[u32; 4]; 4],
    [ia, ib, ic, id]: [usize; 4],
    mx: u32,
    my: u32,
) -> GWords {
    let (a, b, c, d) = (state[0][ia], state[1][ib], state[2][ic], state[3][id]);

    let a_plus_b = a.wrapping_add(b);
    let a1 = a_plus_b.wrapping_add(mx);
    let d1 = (d ^ a1).rotate_right(16);
    let c1 = c.wrapping_add(d1);
    let b1 = (b ^ c1).rotate_right(12);

    let a1_plus_b1 = a1.wrapping_add(b1);
    let a2 = a1_plus_b1.wrapping_add(my);
    let d2 = (d1 ^ a2).rotate_right(8);
    let c2 = c1.wrapping_add(d2);
    let b2 = (b1 ^ c2).rotate_right(7);

    state[0][ia] = a2;
    state[1][ib] = b2;
    state[2][ic] = c2;
    state[3][id] = d2;

    GWords {
        add1_carries: carries_word(a, b, a_plus_b),
        d1,
        b1,
        add2_carries: carries_word(a1, b1, a1_plus_b1),
        d2,
        b2,
    }
}

/// Apply one G step to `state`, writing its witness to `cols`.
///
/// `slots` holds the indices of the `a`, `b`, `c`, `d` words within their state rows.
fn generate_g<F: Field>(
    cols: &mut Blake3BinaryGCols<F>,
    state: &mut [[u32; 4]; 4],
    slots: [usize; 4],
    mx: u32,
    my: u32,
) {
    let words = g_words(state, slots, mx, my);

    cols.add1_carries = word_to_bits_le(words.add1_carries);
    cols.d1 = u32_to_bits_le(words.d1);
    cols.b1 = u32_to_bits_le(words.b1);
    cols.add2_carries = word_to_bits_le(words.add2_carries);
    cols.d2 = u32_to_bits_le(words.d2);
    cols.b2 = u32_to_bits_le(words.b2);
}

/// The carries into bits `1..32` of `x + y`, given `sum = x + y mod 2^32`.
///
/// The carry into bit zero is always zero and is not a column, so the word starts at bit one.
const fn carries_word(x: u32, y: u32, sum: u32) -> u32 {
    (sum ^ x ^ y) >> 1
}

/// The low `N` bits of a word, lowest bit first.
fn word_to_bits_le<F: Field, const N: usize>(word: u32) -> [F; N] {
    array::from_fn(|i| F::from_bool((word >> i) & 1 == 1))
}
