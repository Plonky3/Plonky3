use alloc::vec::Vec;
use core::array;

use p3_air::utils::u32_to_bits_le;
use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use tracing::instrument;

use super::columns::{NUM_SHA256_BINARY_COLS, Sha256BinaryCols};
use crate::{BLOCK_WORDS, INPUT_WORDS, NUM_COMPRESSION_ROUNDS, SHA256_K, STATE_WORDS};

/// Generate a binary SHA-256 trace with one row per compression.
///
/// Each input is laid out as in [`generate_trace_rows`](crate::generate_trace_rows): the 16
/// block words followed by the 8 chaining-value words.
///
/// # Panics
///
/// Panics if the field does not have characteristic 2 or if the number of inputs is not
/// a power of two.
#[instrument(name = "generate SHA-256 binary trace", skip_all)]
pub fn generate_binary_trace_rows<F: Field>(
    inputs: Vec<[u32; INPUT_WORDS]>,
    extra_capacity_bits: usize,
) -> RowMajorMatrix<F> {
    assert_eq!(
        F::TWO,
        F::ZERO,
        "the binary SHA-256 AIR requires a field of characteristic 2"
    );

    let num_rows = inputs.len();
    assert!(
        num_rows.is_power_of_two(),
        "Callers expected to pad inputs to a power of two"
    );

    let trace_length = num_rows * NUM_SHA256_BINARY_COLS;

    // We allocate extra_capacity_bits now as this will be needed by the dft.
    let mut long_trace = F::zero_vec(trace_length << extra_capacity_bits);
    long_trace.truncate(trace_length);

    let mut trace = RowMajorMatrix::new(long_trace, NUM_SHA256_BINARY_COLS);
    let (prefix, rows, suffix) = unsafe { trace.values.align_to_mut::<Sha256BinaryCols<F>>() };
    assert!(prefix.is_empty(), "Alignment should match");
    assert!(suffix.is_empty(), "Alignment should match");
    assert_eq!(rows.len(), num_rows);

    rows.par_iter_mut()
        .zip(inputs)
        .for_each(|(row, input)| generate_trace_row(row, &input));

    trace
}

/// Fill one row with the witness of a single compression.
fn generate_trace_row<F: Field>(row: &mut Sha256BinaryCols<F>, input: &[u32; INPUT_WORDS]) {
    let block: [u32; BLOCK_WORDS] = array::from_fn(|i| input[i]);
    let h_in: [u32; STATE_WORDS] = array::from_fn(|i| input[BLOCK_WORDS + i]);

    // Message schedule.
    let mut w = [0u32; NUM_COMPRESSION_ROUNDS];
    w[..BLOCK_WORDS].copy_from_slice(&block);
    for (i, cols) in row.schedule.iter_mut().enumerate() {
        let t = BLOCK_WORDS + i;
        let (w_t, carries) = add_with_carries(&[
            small_sigma1(w[t - 2]),
            w[t - 7],
            small_sigma0(w[t - 15]),
            w[t - 16],
        ]);
        w[t] = w_t;
        cols.carries = carries;
    }
    row.w = w.map(u32_to_bits_le);

    // Working-state chains, seeded with the chaining value in round-shift order.
    let mut a_chain = [0u32; 4 + NUM_COMPRESSION_ROUNDS];
    let mut e_chain = [0u32; 4 + NUM_COMPRESSION_ROUNDS];
    a_chain[..4].copy_from_slice(&[h_in[3], h_in[2], h_in[1], h_in[0]]);
    e_chain[..4].copy_from_slice(&[h_in[7], h_in[6], h_in[5], h_in[4]]);

    for (t, cols) in row.rounds.iter_mut().enumerate() {
        let (a, b, c, d) = (a_chain[t + 3], a_chain[t + 2], a_chain[t + 1], a_chain[t]);
        let (e, f, g, h) = (e_chain[t + 3], e_chain[t + 2], e_chain[t + 1], e_chain[t]);

        let ch = (e & f) ^ (!e & g);
        let maj = (a & b) ^ (a & c) ^ (b & c);
        let (t1, t1_carries) = add_with_carries(&[h, big_sigma1(e), ch, SHA256_K[t], w[t]]);
        let (new_a, [new_a_carries]) = add_with_carries(&[t1, big_sigma0(a), maj]);

        cols.ch = u32_to_bits_le(ch);
        cols.maj = u32_to_bits_le(maj);
        cols.t1_carries = t1_carries;
        cols.t1 = u32_to_bits_le(t1);
        cols.new_a_carries = new_a_carries;

        a_chain[t + 4] = new_a;
        e_chain[t + 4] = d.wrapping_add(t1);
    }
    row.a_chain = a_chain.map(u32_to_bits_le);
    row.e_chain = e_chain.map(u32_to_bits_le);

    // Output chaining value: H[i] plus the final value of working variable i.
    let final_state: [u32; STATE_WORDS] = array::from_fn(|i| {
        let chain = if i < 4 { &a_chain } else { &e_chain };
        chain[NUM_COMPRESSION_ROUNDS + 3 - i % 4]
    });
    row.h_out = array::from_fn(|i| u32_to_bits_le(h_in[i].wrapping_add(final_state[i])));
}

/// Add the words in order, returning the sum and the carries of every addition but the last.
///
/// Addition `j` adds `words[j + 1]` to the partial sum of `words[0..=j]`. The AIR stores the
/// carries of all but the last, which it reads off the final sum instead.
///
/// # Panics
///
/// Panics if `N` is not two less than the number of words.
fn add_with_carries<F: Field, const N: usize>(words: &[u32]) -> (u32, [[F; 31]; N]) {
    assert_eq!(words.len(), N + 2);
    let mut partial = words[0];
    let carries = array::from_fn(|j| {
        let sum = partial.wrapping_add(words[j + 1]);
        let carries = carry_bits(partial, words[j + 1], sum);
        partial = sum;
        carries
    });
    (partial.wrapping_add(words[N + 1]), carries)
}

impl<F: Field> Sha256BinaryCols<F> {
    /// The output chaining value of this row.
    ///
    /// # Panics
    ///
    /// Panics if a cell read is not a bit.
    pub fn compression_output(&self) -> [u32; STATE_WORDS] {
        self.h_out.each_ref().map(read_word)
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

/// The carries into bits `1..32` of `x + y`, given `sum = x + y mod 2^32`.
fn carry_bits<F: Field>(x: u32, y: u32, sum: u32) -> [F; 31] {
    let carries = sum ^ x ^ y;
    array::from_fn(|i| F::from_bool((carries >> (i + 1)) & 1 == 1))
}

/// `Σ0(x) = (x >>> 2) ^ (x >>> 13) ^ (x >>> 22)`.
const fn big_sigma0(x: u32) -> u32 {
    x.rotate_right(2) ^ x.rotate_right(13) ^ x.rotate_right(22)
}

/// `Σ1(x) = (x >>> 6) ^ (x >>> 11) ^ (x >>> 25)`.
const fn big_sigma1(x: u32) -> u32 {
    x.rotate_right(6) ^ x.rotate_right(11) ^ x.rotate_right(25)
}

/// `σ0(x) = (x >>> 7) ^ (x >>> 18) ^ (x >> 3)`.
const fn small_sigma0(x: u32) -> u32 {
    x.rotate_right(7) ^ x.rotate_right(18) ^ (x >> 3)
}

/// `σ1(x) = (x >>> 17) ^ (x >>> 19) ^ (x >> 10)`.
const fn small_sigma1(x: u32) -> u32 {
    x.rotate_right(17) ^ x.rotate_right(19) ^ (x >> 10)
}
