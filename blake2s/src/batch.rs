//! Hashing many equal-length messages at once.
//!
//! The batch is legal only because the messages are the same length. That makes their block
//! counts, their counters and their final-block flags identical, so one compression can
//! advance all of them and only the message words differ from lane to lane.

use crate::DIGEST_BYTES;
use crate::compress::{BLOCK_BYTES, Block, State, compress_lanes, initial_state};

/// Messages one batched compression advances at once.
///
/// Eight lanes is one 256-bit vector per state word, which is what AVX2 holds and what a
/// 128-bit vector unit such as NEON splits into two. It is a hint to callers grouping
/// messages rather than a requirement: any batch size works, and a remainder that does not
/// fill a group is hashed the same way with fewer lanes.
pub const LANES: usize = 8;

/// Hash `out.len()` equal-length messages laid end to end in `input`.
///
/// # Panics
///
/// Panics if the batch is ragged: the input length must be a whole multiple of the digest
/// count.
pub(crate) fn hash_many(input: &[u8], out: &mut [[u8; DIGEST_BYTES]]) {
    // No digests requested means there is nothing to read from the input.
    if out.is_empty() {
        return;
    }

    assert!(
        input.len().is_multiple_of(out.len()),
        "input length ({}) must be a whole multiple of the digest count ({})",
        input.len(),
        out.len()
    );
    let len = input.len() / out.len();

    // Empty messages all share one digest, and slicing them into groups is meaningless.
    if len == 0 {
        out.fill(crate::Blake2s256::hash(&[]));
        return;
    }

    let mut messages = input.chunks_exact(len);
    for group in out.chunks_mut(LANES) {
        let mut lanes: [&[u8]; LANES] = [&[]; LANES];
        for lane in lanes.iter_mut().take(group.len()) {
            *lane = messages.next().expect("one message per requested digest");
        }
        hash_group(&lanes, group, len);
    }
}

/// Hash one group of at most [`LANES`] messages, writing a digest per lane.
///
/// Lanes past the end of the group hold an empty slice and produce a digest nobody reads,
/// which costs the vector lane that would otherwise sit idle anyway.
fn hash_group(lanes: &[&[u8]; LANES], out: &mut [[u8; DIGEST_BYTES]], len: usize) {
    let mut state: State<LANES> = initial_state();
    let mut counter = 0u64;

    // Every block but the last is compressed with the flag clear. The last one is held back
    // until the message is known to be over, because it is compressed differently.
    let whole_blocks = if len == 0 { 0 } else { (len - 1) / BLOCK_BYTES };
    for index in 0..whole_blocks {
        let offset = index * BLOCK_BYTES;
        counter += BLOCK_BYTES as u64;
        compress_lanes(
            &mut state,
            &gather(lanes, offset, BLOCK_BYTES),
            counter,
            false,
        );
    }

    // The final block is padded with zeros and counts only the message bytes in it. An empty
    // message still compresses one such block, with the counter at zero.
    let offset = whole_blocks * BLOCK_BYTES;
    let remaining = len - offset;
    counter += remaining as u64;
    compress_lanes(&mut state, &gather(lanes, offset, remaining), counter, true);

    for (digest, lane) in out.iter_mut().enumerate() {
        let (words, _) = lane.as_chunks_mut::<4>();
        for (chunk, word) in words.iter_mut().zip(&state) {
            *chunk = word[digest].to_le_bytes();
        }
    }
}

/// Read one block from every lane, zero padded, as little-endian words.
#[inline]
fn gather(lanes: &[&[u8]; LANES], offset: usize, len: usize) -> Block<LANES> {
    let mut block = [[0u32; LANES]; 16];
    for (lane, message) in lanes.iter().enumerate() {
        if message.is_empty() {
            continue;
        }
        let mut bytes = [0u8; BLOCK_BYTES];
        bytes[..len].copy_from_slice(&message[offset..offset + len]);
        let (words, _) = bytes.as_chunks::<4>();
        for (word, source) in block.iter_mut().zip(words) {
            word[lane] = u32::from_le_bytes(*source);
        }
    }
    block
}
