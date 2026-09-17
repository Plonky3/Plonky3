use alloc::vec::Vec;
use core::array;

use p3_air::utils::u32_to_bits_le;
use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use tracing::instrument;

use super::columns::{Blake3BinaryCols, Blake3BinaryGCols, NUM_BLAKE3_BINARY_COLS};
use super::{G_SCHEDULE, NUM_ROUNDS, iv_word};
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
/// Panics if the field does not have characteristic 2 or if the number of inputs is not
/// a power of two.
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

/// Apply one G step to `state`, writing its witness to `cols`.
///
/// `slots` holds the indices of the `a`, `b`, `c`, `d` words within their state rows.
fn generate_g<F: Field>(
    cols: &mut Blake3BinaryGCols<F>,
    state: &mut [[u32; 4]; 4],
    [ia, ib, ic, id]: [usize; 4],
    mx: u32,
    my: u32,
) {
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

    cols.add1_carries = carry_bits(a, b, a_plus_b);
    cols.d1 = u32_to_bits_le(d1);
    cols.b1 = u32_to_bits_le(b1);
    cols.add2_carries = carry_bits(a1, b1, a1_plus_b1);
    cols.d2 = u32_to_bits_le(d2);
    cols.b2 = u32_to_bits_le(b2);

    state[0][ia] = a2;
    state[1][ib] = b2;
    state[2][ic] = c2;
    state[3][id] = d2;
}

/// The carries into bits `1..32` of `x + y`, given `sum = x + y mod 2^32`.
fn carry_bits<F: Field>(x: u32, y: u32, sum: u32) -> [F; 31] {
    let carries = sum ^ x ^ y;
    array::from_fn(|i| F::from_bool((carries >> (i + 1)) & 1 == 1))
}
