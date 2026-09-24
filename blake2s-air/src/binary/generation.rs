use alloc::vec;
use alloc::vec::Vec;
use core::array;
use core::borrow::BorrowMut;

use p3_air::utils::u32_to_bits_le;
use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use tracing::instrument;

use super::columns::{Blake2sBinaryCols, Blake2sBinaryGCols, NUM_BLAKE2S_BINARY_COLS};
use super::{G_SCHEDULE, NUM_ROUNDS};
use crate::constants::{IV, SIGMA};

/// The inputs of one BLAKE2s compression, as in RFC 7693 section 3.2.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Blake2sCompressionInput {
    /// The chaining value `h[0..8]`.
    pub chaining_value: [u32; 8],
    /// The sixteen message words `m[0..16]`, read little-endian from the 64-byte block.
    pub block: [u32; 16],
    /// The byte counter `t`: total bytes hashed through the end of this block.
    pub counter: u64,
    /// Whether this is the final block of the message.
    pub last_block: bool,
    /// Whether this is the last node of a tree hash.
    pub last_node: bool,
}

impl Blake2sCompressionInput {
    /// The two finalization words: all ones when their flag is set, zero otherwise.
    const fn flag_words(&self) -> (u32, u32) {
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
/// - The field does not have characteristic 2.
/// - There are no inputs.
/// - The number of inputs is not a power of two.
#[instrument(name = "generate BLAKE2s binary trace", skip_all)]
pub fn generate_binary_trace_rows<F: Field>(
    inputs: Vec<Blake2sCompressionInput>,
    extra_capacity_bits: usize,
) -> RowMajorMatrix<F> {
    // XOR is field addition only when 1 + 1 = 0.
    assert_eq!(
        F::TWO,
        F::ZERO,
        "the binary BLAKE2s AIR requires a field of characteristic 2"
    );

    // One row per compression, and the height must be a power of two.
    let num_rows = inputs.len();
    assert!(num_rows > 0, "at least one input is required");
    assert!(
        num_rows.is_power_of_two(),
        "Callers expected to pad inputs to a power of two"
    );

    let trace_length = num_rows * NUM_BLAKE2S_BINARY_COLS;

    // Reserve the room the low-degree extension will need, so it does not reallocate.
    let mut long_trace = F::zero_vec(trace_length << extra_capacity_bits);
    long_trace.truncate(trace_length);

    let mut trace = RowMajorMatrix::new(long_trace, NUM_BLAKE2S_BINARY_COLS);

    // Safety: the column struct is `repr(C)` and made only of `F`.
    // It therefore has the layout of `[F; N]`.
    let (prefix, rows, suffix) = unsafe { trace.values.align_to_mut::<Blake2sBinaryCols<F>>() };
    assert!(prefix.is_empty(), "Alignment should match");
    assert!(suffix.is_empty(), "Alignment should match");
    assert_eq!(rows.len(), num_rows);

    // Rows are independent, so they fill in parallel.
    rows.par_iter_mut()
        .zip(inputs)
        .for_each(|(row, input)| generate_trace_row(row, &input));

    trace
}

/// Generate a binary BLAKE2s trace packed into one `u64` per 64 trace rows.
///
/// The width stays the number of AIR columns.
///
/// ```text
///     physical row w, column j:   bit k  =  cell (64 * w + k, j)
/// ```
///
/// The witness is computed bit-sliced, one lane per trace row.
///
/// The field parameter only fixes the characteristic, and no cell is ever held in it.
///
/// # Panics
///
/// - The field does not have characteristic 2.
/// - There are no inputs.
/// - The number of inputs is not a power of two.
#[instrument(name = "generate packed BLAKE2s binary trace", skip_all)]
pub fn generate_binary_trace_packed<F: Field>(
    inputs: &[Blake2sCompressionInput],
) -> RowMajorMatrix<u64> {
    // XOR is field addition only when 1 + 1 = 0.
    assert_eq!(
        F::TWO,
        F::ZERO,
        "the binary BLAKE2s AIR requires a field of characteristic 2"
    );

    // One row per compression, and the height must be a power of two.
    let num_rows = inputs.len();
    assert!(num_rows > 0, "at least one input is required");
    assert!(
        num_rows.is_power_of_two(),
        "Callers expected to pad inputs to a power of two"
    );

    // Each group of 64 inputs fills one physical row, and the groups are independent.
    let num_blocks = num_rows.div_ceil(64);
    let mut words = vec![0u64; num_blocks * NUM_BLAKE2S_BINARY_COLS];
    inputs
        .par_chunks(64)
        .zip(words.par_chunks_exact_mut(NUM_BLAKE2S_BINARY_COLS))
        .for_each(|(input_block, block)| generate_block(block, input_block));

    RowMajorMatrix::new(words, NUM_BLAKE2S_BINARY_COLS)
}

/// Number of 32-bit input words: the chaining value, the message block and the two counter halves.
///
/// The two finalization flags follow them as single cells.
pub(super) const NUM_INPUT_WORDS: usize = 8 + 16 + 2;

// The packed generator transposes the input words two at a time.
const _: () = assert!(NUM_INPUT_WORDS.is_multiple_of(2));

/// Fill one packed physical row with the witness of up to 64 compressions.
///
/// Lane `k` of every column holds input `k`.
///
/// A 32-bit word is therefore held as 32 bit planes.
///
/// Lanes past the last input stay zero.
fn generate_block(block: &mut [u64], inputs: &[Blake2sCompressionInput]) {
    // Phase 1: the input words.
    //
    // They fill the first columns in order, 32 planes per word.
    // Packing two words per lane gives a 64 x 64 bit matrix.
    // One transpose turns it into 64 columns.
    //
    //     before:  rows[k]    = word_{2p+1} of lane k  ||  word_{2p} of lane k
    //     after:   rows[32e + i] = bit i of word_{2p+e}, one bit per lane
    let mut words = [[0; NUM_INPUT_WORDS]; 64];
    for (lane, input) in words.iter_mut().zip(inputs) {
        *lane = input_words(input);
    }
    let (pairs, _) = block[..NUM_INPUT_WORDS * 32].as_chunks_mut::<64>();
    for (pair, rows) in pairs.iter_mut().enumerate() {
        for (row, lane) in rows.iter_mut().zip(&words) {
            *row = u64::from(lane[2 * pair]) | (u64::from(lane[2 * pair + 1]) << 32);
        }
        transpose_bits(rows);
    }

    // Split the row into its fields, so the rounds are written while the inputs are read in place.
    let Blake2sBinaryCols {
        chaining_value: cv,
        block: m,
        counter_low,
        counter_high,
        last_block,
        last_node,
        rounds,
    } = block.borrow_mut();

    // Phase 2: the finalization flags, one plane each, bit k set when input k sets the flag.
    *last_block = lane_mask(inputs, |input| input.last_block);
    *last_node = lane_mask(inputs, |input| input.last_node);

    // Phase 3: the initial state.
    //
    // The constant words are set only in the lanes that hold an input.
    // An absent lane then starts from the all-zero state.
    // Every G step maps that state to itself with no carries, so the lane stays zero.
    let lanes = u64::MAX >> (64 - inputs.len());
    let iv_planes =
        |word: u32| array::from_fn(|bit| if (word >> bit) & 1 == 1 { lanes } else { 0 });

    // Each flag plane is repeated over all 32 bits.
    // This inverts the word in the lanes that set the flag.
    let parameters = [
        *counter_low,
        *counter_high,
        [*last_block; 32],
        [*last_node; 32],
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

    // Phase 4: ten rounds of eight G steps, writing each step's witness planes.
    for (round, schedule) in rounds.iter_mut().zip(SIGMA) {
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

/// A bit plane with bit `k` set when input `k` satisfies the predicate.
fn lane_mask(
    inputs: &[Blake2sCompressionInput],
    flag: impl Fn(&Blake2sCompressionInput) -> bool,
) -> u64 {
    inputs.iter().enumerate().fold(0, |mask, (lane, input)| {
        mask | (u64::from(flag(input)) << lane)
    })
}

/// The 32-bit input words of one compression, in column order.
fn input_words(input: &Blake2sCompressionInput) -> [u32; NUM_INPUT_WORDS] {
    let mut words = [0; NUM_INPUT_WORDS];
    // Columns 0..8: the chaining value.
    words[..8].copy_from_slice(&input.chaining_value);
    // Columns 8..24: the message block.
    words[8..24].copy_from_slice(&input.block);
    // Columns 24 and 25: the low then the high half of the counter.
    words[24] = input.counter as u32;
    words[25] = (input.counter >> 32) as u32;
    words
}

/// Transpose a 64 x 64 bit matrix in place.
///
/// Bit `j` of row `i` moves to bit `i` of row `j`.
fn transpose_bits(rows: &mut [u64; 64]) {
    // Recursive block transpose, done bottom-up in six passes.
    //
    // Each pass swaps the two off-diagonal quarters of every aligned square.
    // The next pass works on squares half as wide.
    //
    //     pass 1: 64 x 64 squares, quarters of 32 x 32
    //     pass 6:  2 x 2  squares, quarters of 1 x 1
    let mut width = 32;
    let mut mask = 0x0000_0000_ffff_ffff_u64;
    while width != 0 {
        for base in (0..64).step_by(2 * width) {
            for row in base..base + width {
                // Swap the high half of the upper row with the low half of the lower row.
                let swap = ((rows[row] >> width) ^ rows[row + width]) & mask;
                rows[row] ^= swap << width;
                rows[row + width] ^= swap;
            }
        }
        width >>= 1;
        // Keep the low half of every chunk of the new width.
        mask ^= mask << width;
    }
}

/// Apply one G step to the bit planes of the state, writing its witness planes.
///
/// The four slots are the positions of the `a`, `b`, `c`, `d` words within their state rows.
///
/// Each operation is the scalar G step applied to all 64 lanes at once.
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

    // Additions without a carry column write their carries to scratch space.
    let mut unused = [0; 31];

    // First half.
    //
    //     a_1 = a + b + m_x      d_1 = (d ^ a_1) >>> 16
    //     c_1 = c + d_1          b_1 = (b ^ c_1) >>> 12
    let a_plus_b = add_planes(a, b, &mut cols.add1_carries);
    let a1 = add_planes(&a_plus_b, mx, &mut unused);
    cols.d1 = xor_rotate_right(d, &a1, 16);
    let c1 = add_planes(c, &cols.d1, &mut unused);
    cols.b1 = xor_rotate_right(b, &c1, 12);

    // Second half, writing the new words straight into the state.
    //
    //     a_2 = a_1 + b_1 + m_y  d_2 = (d_1 ^ a_2) >>> 8
    //     c_2 = c_1 + d_2        b_2 = (b_1 ^ c_2) >>> 7
    let a1_plus_b1 = add_planes(&a1, &cols.b1, &mut cols.add2_carries);
    *a = add_planes(&a1_plus_b1, my, &mut unused);
    cols.d2 = xor_rotate_right(&cols.d1, a, 8);
    *d = cols.d2;
    *c = add_planes(&c1, &cols.d2, &mut unused);
    cols.b2 = xor_rotate_right(&cols.b1, c, 7);
    *b = cols.b2;
}

/// The bit planes of `x + y mod 2^32`, computed by a ripple-carry adder.
///
/// Entry `i` of the carries receives the plane of the carry into bit `i + 1`.
#[inline(always)]
fn add_planes(x: &[u64; 32], y: &[u64; 32], carries: &mut [u64; 31]) -> [u64; 32] {
    let mut sum = [0; 32];
    let mut carry = 0;
    for bit in 0..31 {
        // Full adder on bit i of every lane.
        let half = x[bit] ^ y[bit];
        sum[bit] = half ^ carry;
        carry = (x[bit] & y[bit]) | (carry & half);
        carries[bit] = carry;
    }
    // The carry out of bit 31 is dropped: the sum is mod 2^32.
    sum[31] = x[31] ^ y[31] ^ carry;
    sum
}

/// The bit planes of `(x ^ y) >>> amount`.
#[inline(always)]
fn xor_rotate_right(x: &[u64; 32], y: &[u64; 32], amount: usize) -> [u64; 32] {
    // Rotating right by s moves bit i + s to bit i.
    array::from_fn(|bit| x[(bit + amount) % 32] ^ y[(bit + amount) % 32])
}

/// Fill one row with the witness of a single compression.
fn generate_trace_row<F: Field>(row: &mut Blake2sBinaryCols<F>, input: &Blake2sCompressionInput) {
    let counter_low = input.counter as u32;
    let counter_high = (input.counter >> 32) as u32;
    let (last_block, last_node) = input.flag_words();

    // The input cells.
    row.chaining_value = input.chaining_value.map(u32_to_bits_le);
    row.block = input.block.map(u32_to_bits_le);
    row.counter_low = u32_to_bits_le(counter_low);
    row.counter_high = u32_to_bits_le(counter_high);
    row.last_block = F::from_bool(input.last_block);
    row.last_node = F::from_bool(input.last_node);

    // The initial state, from RFC 7693 section 3.2.
    //
    //     v[0..8]   = h[0..8]
    //     v[8..12]  = IV[0..4]
    //     v[12..16] = IV[4..8] ^ (t_low, t_high, f_0, f_1)
    let cv = input.chaining_value;
    let parameters = [counter_low, counter_high, last_block, last_node];
    let mut state = [
        [cv[0], cv[1], cv[2], cv[3]],
        [cv[4], cv[5], cv[6], cv[7]],
        array::from_fn(|i| IV[i]),
        array::from_fn(|i| IV[4 + i] ^ parameters[i]),
    ];
    let m = input.block;

    // Ten rounds of eight G steps, writing each step's witness.
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
    /// The state `v[0..16]` after the given round, read back from the witness.
    ///
    /// The diagonal steps of a round write every word once, so their outputs are the whole state:
    ///
    /// ```text
    ///     v[i_a]      = a_2 = d_1 ^ (d_2 <<< 8)
    ///     v[4 + i_b]  = b_2
    ///     v[8 + i_c]  = c_2 = b_1 ^ (b_2 <<< 7)
    ///     v[12 + i_d] = d_2
    /// ```
    ///
    /// # Panics
    ///
    /// - The round is not below ten.
    /// - A cell read is not a bit.
    pub(crate) fn state_after_round(&self, round: usize) -> [u32; 16] {
        let mut v = [0u32; 16];
        // Later steps overwrite earlier ones, so the diagonal steps have the last word.
        for (cols, [ia, ib, ic, id]) in self.rounds[round].iter().zip(G_SCHEDULE) {
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
        v
    }

    /// The compression output `h'[0..8]` of this row, read back from the witness.
    ///
    /// RFC 7693 section 3.2 defines it as `h'[i] = h[i] ^ v[i] ^ v[i + 8]` on the final state.
    ///
    /// # Panics
    ///
    /// Panics if a cell read is not a bit.
    pub fn compression_output(&self) -> [u32; 8] {
        let v = self.state_after_round(NUM_ROUNDS - 1);
        let cv = self.chaining_value.map(|word| read_word(&word));
        array::from_fn(|i| cv[i] ^ v[i] ^ v[i + 8])
    }
}

/// Read a word stored as 32 cells, least significant bit first.
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
/// Each word holds the cells of one column group in its low bits, lowest bit first.
struct GWords {
    /// Carries into bits `1..32` of `a + b`, shifted down to start at bit zero.
    add1_carries: u32,
    /// The word `d_1`.
    d1: u32,
    /// The word `b_1`.
    b1: u32,
    /// Carries into bits `1..32` of `a_1 + b_1`, shifted down to start at bit zero.
    add2_carries: u32,
    /// The word `d_2`.
    d2: u32,
    /// The word `b_2`.
    b2: u32,
}

/// Apply one G step to the state and return the witness words it contributes.
///
/// The four slots are the positions of the `a`, `b`, `c`, `d` words within their state rows.
///
/// This is the mixing function G of RFC 7693 section 3.1, with rotations (16, 12, 8, 7).
const fn g_words(
    state: &mut [[u32; 4]; 4],
    [ia, ib, ic, id]: [usize; 4],
    mx: u32,
    my: u32,
) -> GWords {
    let (a, b, c, d) = (state[0][ia], state[1][ib], state[2][ic], state[3][id]);

    // First half, keeping a + b apart to read its carries.
    let a_plus_b = a.wrapping_add(b);
    let a1 = a_plus_b.wrapping_add(mx);
    let d1 = (d ^ a1).rotate_right(16);
    let c1 = c.wrapping_add(d1);
    let b1 = (b ^ c1).rotate_right(12);

    // Second half, keeping a_1 + b_1 apart to read its carries.
    let a1_plus_b1 = a1.wrapping_add(b1);
    let a2 = a1_plus_b1.wrapping_add(my);
    let d2 = (d1 ^ a2).rotate_right(8);
    let c2 = c1.wrapping_add(d2);
    let b2 = (b1 ^ c2).rotate_right(7);

    // The step's outputs replace its inputs in the state.
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

/// Apply one G step to the state, writing its witness as field cells.
///
/// The four slots are the positions of the `a`, `b`, `c`, `d` words within their state rows.
fn generate_g<F: Field>(
    cols: &mut Blake2sBinaryGCols<F>,
    state: &mut [[u32; 4]; 4],
    slots: [usize; 4],
    mx: u32,
    my: u32,
) {
    let words = g_words(state, slots, mx, my);

    // Spread each witness word over its column group, lowest bit first.
    cols.add1_carries = word_to_bits_le(words.add1_carries);
    cols.d1 = u32_to_bits_le(words.d1);
    cols.b1 = u32_to_bits_le(words.b1);
    cols.add2_carries = word_to_bits_le(words.add2_carries);
    cols.d2 = u32_to_bits_le(words.d2);
    cols.b2 = u32_to_bits_le(words.b2);
}

/// The carries into bits `1..32` of `x + y`, given `sum = x + y mod 2^32`.
///
/// Bit `i` of the sum is `x_i ^ y_i ^ carry_i`, so the XOR recovers every carry.
///
/// The carry into bit zero is always zero and has no column, so the result is shifted down by one.
const fn carries_word(x: u32, y: u32, sum: u32) -> u32 {
    (sum ^ x ^ y) >> 1
}

/// The low `N` bits of a word as field cells, lowest bit first.
fn word_to_bits_le<F: Field, const N: usize>(word: u32) -> [F; N] {
    array::from_fn(|i| F::from_bool((word >> i) & 1 == 1))
}
