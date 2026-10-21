//! Hashing many equal-length messages at once.
//!
//! Equal lengths give every message the same block count, counter and final-block flag.
//!
//! So one compression advances a whole group, one vector lane per message.

mod compress;
mod lanes;

use self::compress::{BLOCK_BYTES, BLOCK_WORDS, PARAM_BLOCK_0, STATE_WORDS, initial_state};
pub(crate) use self::lanes::LANES;
use self::lanes::{GROUPS, Vector, Word, compress, load_block, store_digests};
use crate::DIGEST_BYTES;

/// Neoverse V3 measurements favor scalar hashing over one 16-lane NEON batch for tails of up to six messages.
#[cfg(target_arch = "aarch64")]
const NEON_SCALAR_TAIL_MAX: usize = 6;

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

    // Group g holds messages g * LANES to (g + 1) * LANES, one per lane.
    let (groups, rest) = out.as_chunks_mut::<LANES>();
    for (group, digests) in groups.iter_mut().enumerate() {
        let starts = core::array::from_fn(|lane| (group * LANES + lane) * len);
        hash_group(initial_state(PARAM_BLOCK_0), input, &starts, len, digests);
    }

    // Very short NEON tails avoid advancing four vector groups mostly filled with duplicate messages.
    // Keep full batches on SIMD.
    #[cfg(target_arch = "aarch64")]
    if rest.len() <= NEON_SCALAR_TAIL_MAX {
        let first = groups.len() * LANES;
        for (lane, digest) in rest.iter_mut().enumerate() {
            *digest = crate::Blake2s256::hash(&input[(first + lane) * len..][..len]);
        }
        return;
    }

    // A short final group repeats its messages to fill the spare lanes.
    //
    // Those lanes compute digests that are never written out.
    if !rest.is_empty() {
        let first = groups.len() * LANES;
        let starts = core::array::from_fn(|lane| (first + lane % rest.len()) * len);
        let mut digests = [[0u8; DIGEST_BYTES]; LANES];
        hash_group(
            initial_state(PARAM_BLOCK_0),
            input,
            &starts,
            len,
            &mut digests,
        );
        rest.copy_from_slice(&digests[..rest.len()]);
    }
}

/// Hash the `len`-byte message at each start in the input, from a given chaining value.
fn hash_group(
    mut state: [[Vector; STATE_WORDS]; GROUPS],
    input: &[u8],
    starts: &[usize; LANES],
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
        // The counter is the byte count up to and including this block.
        let counter = ((index + 1) * BLOCK_BYTES) as u64;
        compress(&mut state, &load(input, starts, index, len), counter, false);
    }

    // The final counter is the full message length.
    let block = load(input, starts, plain_blocks, len);
    compress(&mut state, &block, len as u64, true);
    store_digests(&state, out);
}

/// Block `index` of the `len`-byte message at each start, the final block zero padded.
///
/// # Panics
///
/// Panics if a message runs past the end of the input.
#[inline(always)]
fn load(
    input: &[u8],
    starts: &[usize; LANES],
    index: usize,
    len: usize,
) -> [[Vector; BLOCK_WORDS]; GROUPS] {
    // Every message ends inside the input, the furthest one included.
    let furthest = starts.iter().fold(0, |max, &start| max.max(start));
    assert!(
        furthest + len <= input.len(),
        "a message runs past the input"
    );
    let offset = index * BLOCK_BYTES;

    // The block at `offset` of every message, read in place.
    //
    // SAFETY: each use below first checks that the furthest row ends inside the input.
    let in_place = || {
        load_block(&starts.map(|start| unsafe {
            &*input
                .as_ptr()
                .add(start + offset)
                .cast::<[u8; BLOCK_BYTES]>()
        }))
    };

    // A plain block lies wholly inside every message.
    if offset + BLOCK_BYTES < len {
        return in_place();
    }

    // The final block holds up to 64 message bytes.
    //
    // Read 64 bytes in place, past the message end, wherever the input reaches that far.
    // Only messages near the very end of the input fall back to a zero-padded copy.
    let tail = len - offset;
    let mut block = if furthest + offset + BLOCK_BYTES <= input.len() {
        in_place()
    } else {
        let mut spill = [[0u8; BLOCK_BYTES]; LANES];
        for (&start, spill) in starts.iter().zip(&mut spill) {
            spill[..tail].copy_from_slice(&input[start + offset..start + len]);
        }
        load_block(&core::array::from_fn(|l| &spill[l]))
    };

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
    block
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use hex_literal::hex;

    use super::*;
    use crate::tests::official_vectors;

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

        // Every lane reads the one input from its start.
        let mut digests = [[0u8; DIGEST_BYTES]; LANES];
        hash_group(
            initial_state(param_0),
            &input,
            &[0; LANES],
            input.len(),
            &mut digests,
        );
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

    #[test]
    fn the_official_keyed_vectors_match() {
        // The official keyed answers all use the 32-byte key 00 01 .. 1f.
        let key: Vec<u8> = (0..32).collect();
        for (input, digest) in official_vectors(include_str!("../test_vectors/keyed.txt")) {
            assert_eq!(
                blake2s(DIGEST_BYTES, &key, &input),
                digest,
                "{} bytes",
                input.len()
            );
        }
    }
}
