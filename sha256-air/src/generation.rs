//! Trace generation for the SHA-256 compression AIR.

use alloc::vec::Vec;
use core::array;

use p3_air::utils::u32_to_bits_le;
use p3_field::PrimeField64;
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use tracing::instrument;

use crate::columns::{CHAIN_LEN, NUM_SHA256_COLS, Sha256Cols, Sha256RoundCols};
use crate::constants::{
    BLOCK_WORDS, NUM_COMPRESSION_ROUNDS, SCHEDULE_EXTENSIONS, SHA256_K, STATE_WORDS, u32_to_limbs,
};

/// Number of 32-bit words supplied per row.
///
/// Layout:
/// - Indices 0..16 hold the message block.
/// - Indices 16..24 hold the input chaining state.
pub const INPUT_WORDS: usize = BLOCK_WORDS + STATE_WORDS;

/// Populate a trace matrix where each row encodes one compression.
///
/// # Arguments
///
/// - `inputs`: one 24-word array per row. Length must be a power of two.
/// - `extra_capacity_bits`: log2 blowup factor reserved for the LDE.
///
/// # Returns
///
/// A row-major matrix of field elements ready to be handed to the prover.
///
/// # Panics
///
/// Panics if `inputs.len()` is not a power of two.
#[instrument(name = "generate SHA-256 trace", skip_all)]
pub fn generate_trace_rows<F: PrimeField64>(
    inputs: Vec<[u32; INPUT_WORDS]>,
    extra_capacity_bits: usize,
) -> RowMajorMatrix<F> {
    let num_rows = inputs.len();
    // Power-of-two count is required for the LDE domain to align.
    assert!(
        num_rows.is_power_of_two(),
        "Callers must pad SHA-256 inputs to a power-of-two count"
    );

    // Reserve backing storage for the full blowup plus the logical trace.
    let trace_length = num_rows * NUM_SHA256_COLS;
    let mut long_trace = F::zero_vec(trace_length << extra_capacity_bits);
    // Shrink the logical view back down to the committed row count.
    long_trace.truncate(trace_length);

    // Wrap the flat buffer as a row-major matrix and re-interpret it as an
    // array of typed rows for easy field access.
    let mut trace = RowMajorMatrix::new(long_trace, NUM_SHA256_COLS);
    let (prefix, rows, suffix) = unsafe { trace.values.align_to_mut::<Sha256Cols<F>>() };
    assert!(prefix.is_empty(), "Alignment should match");
    assert!(suffix.is_empty(), "Alignment should match");
    assert_eq!(rows.len(), num_rows);

    // Fill each row in parallel - rows are fully independent.
    rows.par_iter_mut()
        .zip(inputs)
        .for_each(|(row, input)| generate_trace_row_for_compression(row, input));

    trace
}

/// Populate a single row by executing the SHA-256 compression.
///
/// # Algorithm
///
/// 1. Split the 24-word input into a block and an input chaining state.
/// 2. Expand the message schedule: 48 new `W` words plus their per-step
///    sigma intermediates.
/// 3. Run 64 compression rounds, storing each round's packed intermediates
///    and extending the two working-variable chains.
/// 4. Write `H' = H + state` into the output row.
fn generate_trace_row_for_compression<F: PrimeField64>(
    row: &mut Sha256Cols<F>,
    input: [u32; INPUT_WORDS],
) {
    // Step 1: unpack the input into a block and a chaining state.
    let block: [u32; BLOCK_WORDS] = array::from_fn(|i| input[i]);
    let h_in: [u32; STATE_WORDS] = array::from_fn(|i| input[BLOCK_WORDS + i]);

    // Commit the input chaining state in packed form - one entry per H word.
    row.h_in = h_in.map(|h| u32_to_limbs(h).map(F::from_u16));

    // Step 2: message-schedule expansion.
    //
    // Keep the running u32 values in a scratch array so sigma computations
    // stay simple; commit bits to the row as we go.
    let mut w_words = [0u32; NUM_COMPRESSION_ROUNDS];
    for (dst, src) in w_words.iter_mut().zip(block.iter()) {
        *dst = *src;
    }
    // Commit the 16 block words as bits - pure witness, no auxiliary math.
    for (w_bits, &word) in row.w[..BLOCK_WORDS].iter_mut().zip(block.iter()) {
        *w_bits = u32_to_bits_le(word);
    }

    for t in BLOCK_WORDS..NUM_COMPRESSION_ROUNDS {
        // Auxiliary column index (0..48) for the packed columns.
        let i = t - BLOCK_WORDS;

        // Fetch the four schedule operands.
        let w_m15 = w_words[t - 15];
        let w_m2 = w_words[t - 2];
        let w_m7 = w_words[t - 7];
        let w_m16 = w_words[t - 16];

        // small_sigma0(x) = ROTR_7(x) XOR ROTR_18(x) XOR SHR_3(x).
        let s0 = w_m15.rotate_right(7) ^ w_m15.rotate_right(18) ^ (w_m15 >> 3);

        // small_sigma1(x) = ROTR_17(x) XOR ROTR_19(x) XOR SHR_10(x).
        let s1 = w_m2.rotate_right(17) ^ w_m2.rotate_right(19) ^ (w_m2 >> 10);

        // Partial sum - splits the four-term recurrence into two add3-sized
        // pieces.
        let tmp = s1.wrapping_add(w_m7);

        // Final schedule word for this index.
        let w_t = tmp.wrapping_add(s0).wrapping_add(w_m16);
        w_words[t] = w_t;

        // Commit every committed value for step `t`.
        row.w[t] = u32_to_bits_le(w_t);
        row.sched_sigma0[i] = u32_to_limbs(s0).map(F::from_u16);
        row.sched_sigma1[i] = u32_to_limbs(s1).map(F::from_u16);
        row.sched_tmp[i] = u32_to_limbs(tmp).map(F::from_u16);
        row.w_packed[i] = u32_to_limbs(w_t).map(F::from_u16);
    }
    // Sanity check for the extension count.
    debug_assert_eq!(SCHEDULE_EXTENSIONS, NUM_COMPRESSION_ROUNDS - BLOCK_WORDS);

    // Step 3: compression loop.
    //
    // Chain prefix: indices 0..4 hold (d, c, b, a) of the initial state -
    // that is H[3..=0] for the `a` chain and H[7..=4] for the `e` chain.
    row.a_chain[0] = u32_to_bits_le(h_in[3]);
    row.a_chain[1] = u32_to_bits_le(h_in[2]);
    row.a_chain[2] = u32_to_bits_le(h_in[1]);
    row.a_chain[3] = u32_to_bits_le(h_in[0]);
    row.e_chain[0] = u32_to_bits_le(h_in[7]);
    row.e_chain[1] = u32_to_bits_le(h_in[6]);
    row.e_chain[2] = u32_to_bits_le(h_in[5]);
    row.e_chain[3] = u32_to_bits_le(h_in[4]);

    // Working variables, initialised from the input state in canonical
    // SHA-256 order.
    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = h_in;

    for t in 0..NUM_COMPRESSION_ROUNDS {
        let round: &mut Sha256RoundCols<F> = &mut row.rounds[t];

        // big_sigma1(e) = ROTR_6(e) XOR ROTR_11(e) XOR ROTR_25(e).
        let sigma1_e = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);

        // Ch(e, f, g) = (e AND f) XOR (NOT e AND g).
        let ch = (e & f) ^ ((!e) & g);

        // Partial sum: tmp1 = h + big_sigma1 + Ch.
        let tmp1 = h.wrapping_add(sigma1_e).wrapping_add(ch);

        // T1 = tmp1 + K[t] + W[t].
        let t1 = tmp1.wrapping_add(SHA256_K[t]).wrapping_add(w_words[t]);

        // big_sigma0(a) = ROTR_2(a) XOR ROTR_13(a) XOR ROTR_22(a).
        let sigma0_a = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);

        // Maj(a, b, c) = (a AND b) XOR (a AND c) XOR (b AND c).
        let maj = (a & b) ^ (a & c) ^ (b & c);

        // T2 = big_sigma0 + Maj.
        let t2 = sigma0_a.wrapping_add(maj);

        // Shifted slots for the next round.
        let new_a = t1.wrapping_add(t2);
        let new_e = d.wrapping_add(t1);

        // Commit every packed intermediate for this round.
        round.sigma1_e = u32_to_limbs(sigma1_e).map(F::from_u16);
        round.ch = u32_to_limbs(ch).map(F::from_u16);
        round.tmp1 = u32_to_limbs(tmp1).map(F::from_u16);
        round.t1 = u32_to_limbs(t1).map(F::from_u16);
        round.sigma0_a = u32_to_limbs(sigma0_a).map(F::from_u16);
        round.maj = u32_to_limbs(maj).map(F::from_u16);
        round.t2 = u32_to_limbs(t2).map(F::from_u16);
        round.new_a_packed = u32_to_limbs(new_a).map(F::from_u16);
        round.new_e_packed = u32_to_limbs(new_e).map(F::from_u16);

        // Extend the chains so the next round can read the shifted variables
        // directly.
        row.a_chain[t + 4] = u32_to_bits_le(new_a);
        row.e_chain[t + 4] = u32_to_bits_le(new_e);

        // Apply the SHA-256 working-variable shift.
        h = g;
        g = f;
        f = e;
        e = new_e;
        d = c;
        c = b;
        b = a;
        a = new_a;
    }
    // Sanity check: the chain length matches the expected extent.
    debug_assert_eq!(CHAIN_LEN, 4 + NUM_COMPRESSION_ROUNDS);

    // Step 4: finalization.
    //
    // After the loop the working variables line up with the tail of the two
    // chains:
    //     (a, b, c, d) = (a_chain[67], a_chain[66], a_chain[65], a_chain[64])
    //     (e, f, g, h) = (e_chain[67], e_chain[66], e_chain[65], e_chain[64])
    let final_state: [u32; STATE_WORDS] = [a, b, c, d, e, f, g, h];
    for i in 0..STATE_WORDS {
        // Output word i = H[i] + final_state[i] (mod 2^32).
        let h_out_word = h_in[i].wrapping_add(final_state[i]);
        row.h_out[i] = u32_to_limbs(h_out_word).map(F::from_u16);
    }
}
