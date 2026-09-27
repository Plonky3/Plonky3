//! Hashing many equal-length messages at once.
//!
//! Equal lengths give every message the same block count, counter and final-block flag.
//!
//! So one compression advances a whole group, one vector lane per message.

mod compress;
mod lanes;

use self::compress::{
    BLOCK_BYTES, BLOCK_WORDS, PARAM_BLOCK_0, STATE_WORDS, compress, initial_state,
};
pub(crate) use self::lanes::LANES;
use self::lanes::{GROUPS, Vector, Word, load_block, store_digests};
use crate::DIGEST_BYTES;

/// Hash `out.len()` equal-length messages laid end to end in `input`.
///
/// # Panics
///
/// Panics if the input length is not a whole multiple of the digest count.
pub(crate) fn hash_many(input: &[u8], out: &mut [[u8; DIGEST_BYTES]]) {
    // No digests requested means there is nothing to read from the input.
    if out.is_empty() {
        return;
    }

    // Every message has the same length, so the split is exact by contract.
    assert!(
        input.len().is_multiple_of(out.len()),
        "input length ({}) must be a whole multiple of the digest count ({})",
        input.len(),
        out.len()
    );
    let len = input.len() / out.len();

    // Each lane reads from its message start to the end of the input.
    //
    // The final block may then be read in place, past the message end, and masked.
    let (groups, rest) = out.as_chunks_mut::<LANES>();
    for (index, digests) in groups.iter_mut().enumerate() {
        let lanes = core::array::from_fn(|lane| &input[(index * LANES + lane) * len..]);
        hash_group(initial_state(PARAM_BLOCK_0), &lanes, len, digests);
    }

    // A short final group repeats its messages to fill the spare lanes.
    //
    // Those lanes compute digests that are never written out.
    if !rest.is_empty() {
        let first = groups.len() * LANES;
        let lanes = core::array::from_fn(|lane| &input[(first + lane % rest.len()) * len..]);
        let mut digests = [[0u8; DIGEST_BYTES]; LANES];
        hash_group(initial_state(PARAM_BLOCK_0), &lanes, len, &mut digests);
        rest.copy_from_slice(&digests[..rest.len()]);
    }
}

/// Hash `len` bytes from the start of every lane, from a given chaining value.
///
/// Each lane slice must hold at least `len` bytes.
fn hash_group(
    mut state: [[Vector; STATE_WORDS]; GROUPS],
    lanes: &[&[u8]; LANES],
    len: usize,
    out: &mut [[u8; DIGEST_BYTES]; LANES],
) {
    // The last block is compressed with the final flag, so it is held back even when full.
    //
    //     len = 130:  [ 64 | 64 | 2 ]   two plain blocks, then a final block of 2 bytes
    //     len = 128:  [ 64 | 64 ]       one plain block, then a final block of 64 bytes
    //     len = 0:    [ ]               one final block of no bytes at all
    let plain_blocks = len.saturating_sub(1) / BLOCK_BYTES;
    for index in 0..plain_blocks {
        let offset = index * BLOCK_BYTES;
        let rows = lanes.map(|lane| lane[offset..][..BLOCK_BYTES].try_into().unwrap());

        // The counter is the byte count up to and including this block.
        let counter = (offset + BLOCK_BYTES) as u64;
        compress(&mut state, &load_block(&rows), counter, false);
    }

    // The final block holds up to 64 message bytes, zero padded.
    let offset = plain_blocks * BLOCK_BYTES;
    let tail = len - offset;

    // Read 64 bytes in place wherever the input reaches that far.
    //
    // Only lanes near the very end of the input fall back to a zero-padded copy.
    let mut spill = [[0u8; BLOCK_BYTES]; LANES];
    for (lane, spill) in lanes.iter().zip(&mut spill) {
        if lane.len() < offset + BLOCK_BYTES {
            spill[..tail].copy_from_slice(&lane[offset..len]);
        }
    }
    let rows = core::array::from_fn(|l| {
        lanes[l]
            .get(offset..offset + BLOCK_BYTES)
            .map_or(&spill[l], |row| row.try_into().unwrap())
    });
    let mut block = load_block(&rows);

    // Zero every byte past the message end.
    //
    // All lanes share the tail length, so one mask per word serves every lane.
    //
    //     tail = 6:  word 0 keeps 4 bytes, word 1 keeps 2, words 2 to 15 keep none
    if tail < BLOCK_BYTES {
        for w in tail / 4..BLOCK_WORDS {
            let kept_bytes = tail.saturating_sub(4 * w);
            let mask = Vector::splat(!(u32::MAX << (8 * kept_bytes)));
            for group in &mut block {
                group[w] = group[w].and(mask);
            }
        }
    }

    // The final counter is the full message length.
    compress(&mut state, &block, len as u64, true);
    store_digests(&state, out);
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use hex_literal::hex;

    use super::*;

    /// RFC 7693 appendix E: BLAKE2s-256 of every self-test digest, in order.
    const SELFTEST_GRAND_HASH: [u8; DIGEST_BYTES] =
        hex!("6A411F08CE25ADCDFB02ABA641451CEC53C598B24F4FC787FBDC88797F4C1DFE");

    /// RFC 7693 appendix E: a deterministic Fibonacci byte sequence.
    fn selftest_seq(len: usize, seed: u32) -> Vec<u8> {
        let mut a = 0xDEAD_4BADu32.wrapping_mul(seed);
        let mut b = 1u32;
        (0..len)
            .map(|_| {
                let t = a.wrapping_add(b);
                a = b;
                b = t;
                (t >> 24) as u8
            })
            .collect()
    }

    /// BLAKE2s with any digest length and key, through the batched kernel.
    ///
    /// Every lane hashes the same input, and every lane must agree.
    fn blake2s(out_len: usize, key: &[u8], message: &[u8]) -> Vec<u8> {
        // A key is one zero-padded block in front of the message.
        let mut input = key.to_vec();
        if !key.is_empty() {
            input.resize(BLOCK_BYTES, 0);
        }
        input.extend_from_slice(message);

        // From the low byte up: digest length, key length, fanout 1, depth 1.
        let param_0 = 0x0101_0000 | (key.len() as u32) << 8 | out_len as u32;

        let mut digests = [[0u8; DIGEST_BYTES]; LANES];
        let lanes = [input.as_slice(); LANES];
        hash_group(initial_state(param_0), &lanes, input.len(), &mut digests);
        assert!(digests.iter().all(|d| d == &digests[0]), "lanes disagree");

        // A shorter digest is a prefix of the chaining value.
        digests[0][..out_len].to_vec()
    }

    #[test]
    fn the_rfc_7693_self_test_matches_its_grand_hash() {
        // Digest lengths and message lengths straight from appendix E.
        let mut digests = Vec::new();
        for out_len in [16, 20, 28, 32] {
            for in_len in [0, 3, 64, 65, 255, 1024] {
                let message = selftest_seq(in_len, in_len as u32);

                // Unkeyed, then keyed with a key as long as the digest.
                digests.extend(blake2s(out_len, &[], &message));
                let key = selftest_seq(out_len, out_len as u32);
                digests.extend(blake2s(out_len, &key, &message));
            }
        }

        // The grand hash is one plain BLAKE2s-256 over the concatenated digests.
        let mut grand = vec![[0u8; DIGEST_BYTES]];
        hash_many(&digests, &mut grand);
        assert_eq!(grand[0], SELFTEST_GRAND_HASH);
    }
}
