use alloc::vec;
use alloc::vec::Vec;
use core::array;
use core::borrow::BorrowMut;

use p3_air::utils::u32_to_bits_le;
use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use tracing::instrument;

use super::air::NUM_INPUT_BITS;
use super::columns::{Blake2sBinaryCols, Blake2sBinaryGCols, NUM_BLAKE2S_BINARY_COLS};
use super::{G_SCHEDULE, NUM_ROUNDS};
use crate::constants::{IV, SIGMA};

/// The inputs to one BLAKE2s compression.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Blake2sCompressionInput {
    /// The chaining value.
    pub chaining_value: [u32; 8],
    /// The message words.
    pub block: [u32; 16],
    /// Total number of message bytes counted through the end of this block.
    pub counter: u64,
    /// Whether this is the final block of the message.
    pub last_block: bool,
    /// Whether this is the last node of a tree, which only tree hashing sets.
    pub last_node: bool,
}

impl Blake2sCompressionInput {
    /// The finalization words, which are all ones when their flag is set.
    const fn flags(&self) -> (u32, u32) {
        (
            if self.last_block { u32::MAX } else { 0 },
            if self.last_node { u32::MAX } else { 0 },
        )
    }
}

/// Generate a binary BLAKE2s trace with one row per compression.
///
/// # Panics
///
/// Panics if the field does not have characteristic 2, if `inputs` is empty, or if the number of
/// inputs is not a power of two.
#[instrument(name = "generate BLAKE2s binary trace", skip_all)]
pub fn generate_binary_trace_rows<F: Field>(
    inputs: Vec<Blake2sCompressionInput>,
    extra_capacity_bits: usize,
) -> RowMajorMatrix<F> {
    assert_eq!(
        F::TWO,
        F::ZERO,
        "the binary BLAKE2s AIR requires a field of characteristic 2"
    );

    let num_rows = inputs.len();
    assert!(num_rows > 0, "at least one input is required");
    assert!(
        num_rows.is_power_of_two(),
        "Callers expected to pad inputs to a power of two"
    );

    let trace_length = num_rows * NUM_BLAKE2S_BINARY_COLS;

    // We allocate extra_capacity_bits now as this will be needed by the dft.
    let mut long_trace = F::zero_vec(trace_length << extra_capacity_bits);
    long_trace.truncate(trace_length);

    let mut trace = RowMajorMatrix::new(long_trace, NUM_BLAKE2S_BINARY_COLS);
    let (prefix, rows, suffix) = unsafe { trace.values.align_to_mut::<Blake2sBinaryCols<F>>() };
    assert!(prefix.is_empty(), "Alignment should match");
    assert!(suffix.is_empty(), "Alignment should match");
    assert_eq!(rows.len(), num_rows);

    rows.par_iter_mut()
        .zip(inputs)
        .for_each(|(row, input)| generate_trace_row(row, &input));

    trace
}

/// Generate a binary BLAKE2s trace packed into one `u64` per 64 trace rows.
///
/// The returned matrix keeps the AIR columns as its width. Physical row `w` stores logical
/// rows `64 * w..64 * w + 63`, with bit zero holding the first logical row. The witness is
/// computed bit-sliced, one lane per logical row, so the generic field parameter names only the
/// characteristic the cells are read in; no cell is ever held in it.
///
/// # Panics
///
/// Panics if the field does not have characteristic 2, if `inputs` is empty, or if the number of
/// inputs is not a power of two.
#[instrument(name = "generate packed BLAKE2s binary trace", skip_all)]
#[allow(clippy::needless_pass_by_value)]
pub fn generate_binary_trace_packed<F: Field>(
    inputs: Vec<Blake2sCompressionInput>,
) -> RowMajorMatrix<u64> {
    assert_eq!(
        F::TWO,
        F::ZERO,
        "the binary BLAKE2s AIR requires a field of characteristic 2"
    );

    let num_rows = inputs.len();
    assert!(num_rows > 0, "at least one input is required");
    assert!(
        num_rows.is_power_of_two(),
        "Callers expected to pad inputs to a power of two"
    );

    let num_blocks = num_rows.div_ceil(64);
    let mut words = vec![0u64; num_blocks * NUM_BLAKE2S_BINARY_COLS];
    inputs
        .par_chunks(64)
        .zip(words.par_chunks_exact_mut(NUM_BLAKE2S_BINARY_COLS))
        .for_each(|(input_block, block)| generate_block(block, input_block));

    RowMajorMatrix::new(words, NUM_BLAKE2S_BINARY_COLS)
}

/// Number of input words of one compression: the chaining value, the message block, the two
/// counter halves and the two finalization flags.
pub(super) const NUM_INPUT_WORDS: usize = 8 + 16 + 4;

/// Fill one packed block with the witness of up to 64 compressions, lane `j` holding input `j`.
///
/// Each word of the block is one column over the block's rows, bit `j` holding lane `j`, so a
/// witness word is held as 32 bit planes. Lanes past the last input are left at zero.
fn generate_block(block: &mut [u64], inputs: &[Blake2sCompressionInput]) {
    // The input words fill the first columns in order, 32 planes each, so two words of every
    // lane transpose into 64 adjacent columns at once.
    let mut words = [[0; NUM_INPUT_WORDS]; 64];
    for (lane, input) in words.iter_mut().zip(inputs) {
        *lane = input_words(input);
    }
    let (pairs, _) = block[..NUM_INPUT_BITS].as_chunks_mut::<64>();
    for (pair, rows) in pairs.iter_mut().enumerate() {
        for (row, lane) in rows.iter_mut().zip(&words) {
            *row = u64::from(lane[2 * pair]) | (u64::from(lane[2 * pair + 1]) << 32);
        }
        transpose_bits(rows);
    }

    let cols: &mut Blake2sBinaryCols<u64> = block.borrow_mut();

    // An absent lane has all-zero inputs. The initialization words are set only in the present
    // lanes, so an absent lane starts from the all-zero state, which every G step maps to
    // itself with no carries.
    let lanes = u64::MAX >> (64 - inputs.len());
    let iv_planes =
        |word: u32| array::from_fn(|bit| if (word >> bit) & 1 == 1 { lanes } else { 0 });
    let cv = cols.chaining_value;
    let parameters = [
        cols.counter_low,
        cols.counter_high,
        cols.last_block,
        cols.last_node,
    ];
    let mut state = [
        [cv[0], cv[1], cv[2], cv[3]],
        [cv[4], cv[5], cv[6], cv[7]],
        array::from_fn(|i| iv_planes(IV[i])),
        array::from_fn(|i| {
            let iv: [u64; 32] = iv_planes(IV[4 + i]);
            array::from_fn(|bit| iv[bit] ^ parameters[i][bit])
        }),
    ];
    let m = cols.block;

    for (round, schedule) in cols.rounds.iter_mut().zip(SIGMA) {
        for (g, (g_cols, slots)) in round.iter_mut().zip(G_SCHEDULE).enumerate() {
            g_planes(
                g_cols,
                &mut state,
                slots,
                &m[schedule[2 * g]],
                &m[schedule[2 * g + 1]],
            );
        }
    }
}

/// The input words of one compression, in column order.
fn input_words(input: &Blake2sCompressionInput) -> [u32; NUM_INPUT_WORDS] {
    let (last_block, last_node) = input.flags();
    let mut words = [0; NUM_INPUT_WORDS];
    words[..8].copy_from_slice(&input.chaining_value);
    words[8..24].copy_from_slice(&input.block);
    words[24] = input.counter as u32;
    words[25] = (input.counter >> 32) as u32;
    words[26] = last_block;
    words[27] = last_node;
    words
}

/// Transpose a 64 x 64 bit matrix in place: bit `j` of `rows[i]` moves to bit `i` of `rows[j]`.
fn transpose_bits(rows: &mut [u64; 64]) {
    // Swap the off-diagonal halves of every aligned square, halving the square each pass.
    let mut width = 32;
    let mut mask = 0x0000_0000_ffff_ffff_u64;
    while width != 0 {
        for base in (0..64).step_by(2 * width) {
            for row in base..base + width {
                let swap = ((rows[row] >> width) ^ rows[row + width]) & mask;
                rows[row] ^= swap << width;
                rows[row + width] ^= swap;
            }
        }
        width >>= 1;
        mask ^= mask << width;
    }
}

/// Apply one G step to the bit planes of `state`, writing its witness planes to `cols`.
///
/// `slots` holds the indices of the `a`, `b`, `c`, `d` words within their state rows. Each
/// operation is [`g_words`] applied to every lane at once.
#[inline]
fn g_planes(
    cols: &mut Blake2sBinaryGCols<u64>,
    [row_a, row_b, row_c, row_d]: &mut [[[u64; 32]; 4]; 4],
    [ia, ib, ic, id]: [usize; 4],
    mx: &[u64; 32],
    my: &[u64; 32],
) {
    let (a, b, c, d) = (
        &mut row_a[ia],
        &mut row_b[ib],
        &mut row_c[ic],
        &mut row_d[id],
    );

    // The witness words are written straight to their columns and `a2`, `c2` to their state
    // slots; `d2` and `b2` are copied from their columns into the state. The carries of the
    // additions that have no carry columns go to `unused`.
    let mut unused = [0; 31];
    let a_plus_b = add_planes(a, b, &mut cols.add1_carries);
    let a1 = add_planes(&a_plus_b, mx, &mut unused);
    cols.d1 = xor_rotate_right(d, &a1, 16);
    let c1 = add_planes(c, &cols.d1, &mut unused);
    cols.b1 = xor_rotate_right(b, &c1, 12);

    let a1_plus_b1 = add_planes(&a1, &cols.b1, &mut cols.add2_carries);
    *a = add_planes(&a1_plus_b1, my, &mut unused);
    cols.d2 = xor_rotate_right(&cols.d1, a, 8);
    *d = cols.d2;
    *c = add_planes(&c1, &cols.d2, &mut unused);
    cols.b2 = xor_rotate_right(&cols.b1, c, 7);
    *b = cols.b2;
}

/// The bit planes of `x + y mod 2^32`.
///
/// The planes of the carries into bits `1..32` are written to `carries`, shifted down to start
/// at plane zero.
#[inline(always)]
fn add_planes(x: &[u64; 32], y: &[u64; 32], carries: &mut [u64; 31]) -> [u64; 32] {
    let mut sum = [0; 32];
    let mut carry = 0;
    for bit in 0..31 {
        let half = x[bit] ^ y[bit];
        sum[bit] = half ^ carry;
        carry = (x[bit] & y[bit]) | (carry & half);
        carries[bit] = carry;
    }
    sum[31] = x[31] ^ y[31] ^ carry;
    sum
}

/// The bit planes of `(x ^ y) >>> amount`.
#[inline(always)]
fn xor_rotate_right(x: &[u64; 32], y: &[u64; 32], amount: usize) -> [u64; 32] {
    array::from_fn(|bit| x[(bit + amount) % 32] ^ y[(bit + amount) % 32])
}

/// Fill one row with the witness of a single compression.
fn generate_trace_row<F: Field>(row: &mut Blake2sBinaryCols<F>, input: &Blake2sCompressionInput) {
    let counter_low = input.counter as u32;
    let counter_high = (input.counter >> 32) as u32;
    let (last_block, last_node) = input.flags();

    row.chaining_value = input.chaining_value.map(u32_to_bits_le);
    row.block = input.block.map(u32_to_bits_le);
    row.counter_low = u32_to_bits_le(counter_low);
    row.counter_high = u32_to_bits_le(counter_high);
    row.last_block = u32_to_bits_le(last_block);
    row.last_node = u32_to_bits_le(last_node);

    let cv = input.chaining_value;
    let parameters = [counter_low, counter_high, last_block, last_node];
    let mut state = [
        [cv[0], cv[1], cv[2], cv[3]],
        [cv[4], cv[5], cv[6], cv[7]],
        array::from_fn(|i| IV[i]),
        array::from_fn(|i| IV[4 + i] ^ parameters[i]),
    ];
    let m = input.block;

    for (round, schedule) in row.rounds.iter_mut().zip(SIGMA) {
        for (g, (cols, slots)) in round.iter_mut().zip(G_SCHEDULE).enumerate() {
            generate_g(
                cols,
                &mut state,
                slots,
                m[schedule[2 * g]],
                m[schedule[2 * g + 1]],
            );
        }
    }
}

impl<F: Field> Blake2sBinaryCols<F> {
    /// Recover the compression output of this row from its witness columns.
    ///
    /// The final state words are `a = d1 ^ (d2 <<< 8)`, `b = b2`, `c = b1 ^ (b2 <<< 7)` and
    /// `d = d2` of the last G step that wrote them, and the output chaining value is
    /// `cv[i] ^ v[i] ^ v[i + 8]`.
    ///
    /// # Panics
    ///
    /// Panics if a cell read is not a bit.
    pub fn compression_output(&self) -> [u32; 8] {
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
        array::from_fn(|i| cv[i] ^ v[i] ^ v[i + 8])
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
/// This is the whole arithmetic of a G step, and the field rows are written from what it returns.
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
    cols: &mut Blake2sBinaryGCols<F>,
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
