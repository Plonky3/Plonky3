//! The Keccak-f permutation, and hash functions built from it.

#![no_std]

#[cfg(test)]
extern crate alloc;

use p3_symmetric::{CryptographicHasher, CryptographicPermutation, Permutation};
use tiny_keccak::{Hasher, Keccak, keccakf};

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
pub mod avx512;
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
pub use avx512::*;

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    not(target_feature = "avx512f")
))]
pub mod avx2;
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    not(target_feature = "avx512f")
))]
pub use avx2::*;

#[cfg(all(target_arch = "x86_64", not(target_feature = "avx2")))]
pub mod sse2;
#[cfg(all(target_arch = "x86_64", not(target_feature = "avx2")))]
pub use sse2::*;

#[cfg(all(
    target_arch = "aarch64",
    target_feature = "neon",
    target_feature = "sha3"
))]
pub mod neon_sha3;
#[cfg(all(
    target_arch = "aarch64",
    target_feature = "neon",
    target_feature = "sha3"
))]
pub use neon_sha3::*;

#[cfg(all(
    target_arch = "aarch64",
    target_feature = "neon",
    not(target_feature = "sha3")
))]
pub mod neon;
#[cfg(all(
    target_arch = "aarch64",
    target_feature = "neon",
    not(target_feature = "sha3")
))]
pub use neon::*;

#[cfg(not(any(
    all(target_arch = "aarch64", target_feature = "neon"),
    target_arch = "x86_64"
)))]
mod fallback;
#[cfg(not(any(
    all(target_arch = "aarch64", target_feature = "neon"),
    target_arch = "x86_64"
)))]
pub use fallback::*;

/// The Keccak-f permutation.
#[derive(Copy, Clone, Debug)]
pub struct KeccakF;

impl Permutation<[u64; 25]> for KeccakF {
    fn permute_mut(&self, input: &mut [u64; 25]) {
        keccakf(input);
    }
}

impl CryptographicPermutation<[u64; 25]> for KeccakF {}

impl Permutation<[u8; 200]> for KeccakF {
    fn permute(&self, input_u8s: [u8; 200]) -> [u8; 200] {
        let mut state_u64s: [u64; 25] = core::array::from_fn(|i| {
            u64::from_le_bytes(input_u8s[i * 8..][..8].try_into().unwrap())
        });

        keccakf(&mut state_u64s);

        core::array::from_fn(|i| {
            let u64_limb = state_u64s[i / 8];
            u64_limb.to_le_bytes()[i % 8]
        })
    }
}

impl CryptographicPermutation<[u8; 200]> for KeccakF {}

/// Byte rate of the Keccak-256 sponge.
///
/// The permutation holds 1600 bits and the security target reserves twice the 256-bit digest
/// as capacity, which leaves 1088 bits, exactly 136 bytes, absorbed per permutation.
const RATE: usize = (1600 - 2 * 256) / 8;

/// Digest length of Keccak-256 in bytes.
const DIGEST_BYTES: usize = 32;

/// Absorb up to one rate block of a single message into one lane of a vectorized state.
///
/// A vectorized state keeps one independent sponge per lane, side by side.
/// The lane count is whatever the target's permutation is wide.
///
/// ```text
///     state[word][lane]      word = 0..25, lane = 0..L
///
///     state[0]  [ s_0^(0)  s_0^(1)  ...  s_0^(L-1) ]
///     state[1]  [ s_1^(0)  s_1^(1)  ...  s_1^(L-1) ]
///     ...
/// ```
///
/// Absorbing writes only the given lane's column, so the messages never need transposing.
/// Bytes enter the state little-endian, eight to a state word, matching the Keccak convention.
#[inline]
fn absorb_into_lane(state: &mut [[u64; VECTOR_LEN]; 25], lane: usize, block: &[u8]) {
    debug_assert!(block.len() <= RATE);

    // Whole state words come straight from eight consecutive message bytes.
    let (words, tail) = block.as_chunks::<8>();
    for (word_index, word) in words.iter().enumerate() {
        state[word_index][lane] ^= u64::from_le_bytes(*word);
    }

    // A short final run occupies the low bytes of the next state word, the rest staying zero.
    if !tail.is_empty() {
        let mut last = [0u8; 8];
        last[..tail.len()].copy_from_slice(tail);
        state[words.len()][lane] ^= u64::from_le_bytes(last);
    }
}

/// Apply the Keccak padding to one lane after its final partial block was absorbed.
///
/// The rule is `pad10*1` with the original Keccak domain byte, so the block becomes
///
/// ```text
///     [ message bytes | 0x01 | 0x00 ... 0x00 | 0x80 ]
///                        ^                      ^
///                     offset                 rate - 1
/// ```
///
/// A message ending one byte short of the rate puts both marks in the same byte, and since both
/// are applied by exclusive-or that byte simply becomes `0x81`, which is what the rule requires.
#[inline]
fn pad_lane(state: &mut [[u64; VECTOR_LEN]; 25], lane: usize, offset: usize) {
    debug_assert!(offset < RATE);

    // First mark sits immediately after the last absorbed byte of the final block.
    state[offset / 8][lane] ^= 0x01u64 << (8 * (offset % 8));

    // Second mark closes the block at its very last byte.
    state[(RATE - 1) / 8][lane] ^= 0x80u64 << (8 * ((RATE - 1) % 8));
}

/// Read the digest of one lane out of a permuted vectorized state.
#[inline]
fn squeeze_lane(state: &[[u64; VECTOR_LEN]; 25], lane: usize) -> [u8; DIGEST_BYTES] {
    let mut digest = [0u8; DIGEST_BYTES];

    // The digest is the leading bytes of the rate portion, little-endian per state word.
    for (word_index, word) in digest.as_chunks_mut::<8>().0.iter_mut().enumerate() {
        *word = state[word_index][lane].to_le_bytes();
    }

    digest
}

/// The `Keccak` hash functions defined in
/// [Keccak SHA3 submission](https://keccak.team/files/Keccak-submission-3.pdf).
#[derive(Copy, Clone, Debug)]
pub struct Keccak256Hash;

impl CryptographicHasher<u8, [u8; 32]> for Keccak256Hash {
    const LANES: usize = VECTOR_LEN;

    fn hash_iter<I>(&self, input: I) -> [u8; 32]
    where
        I: IntoIterator<Item = u8>,
    {
        const BUFLEN: usize = 512; // Tweakable parameter; determined by experiment
        let mut hasher = Keccak::v256();
        p3_util::apply_to_chunks::<BUFLEN, _, _>(input, |buf| hasher.update(buf));

        let mut output = [0u8; 32];
        hasher.finalize(&mut output);
        output
    }

    fn hash_iter_slices<'a, I>(&self, input: I) -> [u8; 32]
    where
        I: IntoIterator<Item = &'a [u8]>,
    {
        let mut hasher = Keccak::v256();
        for chunk in input {
            hasher.update(chunk);
        }

        let mut output = [0u8; 32];
        hasher.finalize(&mut output);
        output
    }

    fn hash_many(&self, input: &[u8], out: &mut [[u8; 32]]) {
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

        // Empty messages all share one digest and drive no absorb loop at all.
        if len == 0 {
            for digest in out.iter_mut() {
                *digest = self.hash_iter(core::iter::empty());
            }
            return;
        }

        // A message of `len` bytes fills this many whole rate blocks, leaving a shorter final
        // block that carries the padding, possibly empty when the length divides the rate.
        let full_blocks = len / RATE;
        let final_block = len % RATE;

        // One message per lane per permutation.
        // The last group may be short and simply leaves the unused lanes at their initial
        // state, whose digests are never read.
        for (messages, digests) in input
            .chunks(len * VECTOR_LEN)
            .zip(out.chunks_mut(VECTOR_LEN))
        {
            let mut state = [[0u64; VECTOR_LEN]; 25];

            // Absorb the whole blocks in lockstep: every lane contributes its block, then the
            // single vectorized permutation advances all of the sponges together.
            for block in 0..full_blocks {
                for (lane, message) in messages.chunks_exact(len).enumerate() {
                    absorb_into_lane(&mut state, lane, &message[block * RATE..][..RATE]);
                }
                KeccakF.permute_mut(&mut state);
            }

            // Absorb each final partial block and mark it, still without permuting in between.
            for (lane, message) in messages.chunks_exact(len).enumerate() {
                absorb_into_lane(&mut state, lane, &message[full_blocks * RATE..]);
                pad_lane(&mut state, lane, final_block);
            }
            KeccakF.permute_mut(&mut state);

            // Squeeze one digest per requested message.
            for (lane, digest) in digests.iter_mut().enumerate() {
                *digest = squeeze_lane(&state, lane);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use proptest::prelude::*;

    use super::*;

    /// Every message length that changes the shape of the absorb loop.
    ///
    /// The block boundaries are the interesting ones:
    /// - A length one short of the rate folds both padding marks into a single byte.
    /// - A length that is an exact multiple of the rate pads a block carrying no message bytes.
    const SHAPE_LENGTHS: [usize; 14] = [
        0,
        1,
        7,
        8,
        9,
        RATE - 1,
        RATE,
        RATE + 1,
        2 * RATE - 1,
        2 * RATE,
        2 * RATE + 1,
        3 * RATE,
        400,
        1000,
    ];

    /// Hash each message on its own, which is the behaviour the batched path must reproduce.
    fn reference(messages: &[u8], len: usize, count: usize) -> Vec<[u8; 32]> {
        (0..count)
            .map(|k| {
                // A zero-length message still has a digest, so slice defensively.
                let message = if len == 0 {
                    &messages[..0]
                } else {
                    &messages[k * len..(k + 1) * len]
                };
                Keccak256Hash.hash_slice(message)
            })
            .collect()
    }

    #[test]
    fn hash_many_matches_scalar_across_block_shapes() {
        // Batch sizes below, at, and above one full lane group, so the final short group is
        // exercised at every lane count the target might compile to.
        let counts: Vec<usize> = (1..=2 * VECTOR_LEN + 1).collect();

        for len in SHAPE_LENGTHS {
            for &count in &counts {
                // Fixture: a deterministic byte ramp, distinct per position.
                let messages: Vec<u8> = (0..len * count).map(|i| (i * 31 + 7) as u8).collect();

                let mut batched = vec![[0u8; 32]; count];
                Keccak256Hash.hash_many(&messages, &mut batched);

                assert_eq!(
                    batched,
                    reference(&messages, len, count),
                    "len {len}, count {count}"
                );
            }
        }
    }

    #[test]
    fn hash_many_splits_input_by_digest_count() {
        // The message length is derived from the input length divided by the digest count, so
        // 64 bytes and 4 digests must be read as four adjacent 16-byte messages.
        let messages: Vec<u8> = (0..64).map(|i| i as u8).collect();

        // 4 messages of 16 bytes each.
        let mut digests = [[0u8; 32]; 4];
        Keccak256Hash.hash_many(&messages, &mut digests);

        for (k, digest) in digests.iter().enumerate() {
            assert_eq!(*digest, Keccak256Hash.hash_slice(&messages[k * 16..][..16]));
        }
    }

    #[test]
    #[should_panic(expected = "must be a whole multiple")]
    fn hash_many_rejects_ragged_input() {
        // 5 bytes cannot split into 2 equal messages, so the contract is violated up front
        // rather than producing digests over silently misaligned message boundaries.
        let mut digests = [[0u8; 32]; 2];
        Keccak256Hash.hash_many(&[1, 2, 3, 4, 5], &mut digests);
    }

    proptest! {
        #[test]
        fn hash_many_matches_scalar_on_random_batches(
            len in 0usize..=400,
            count in 1usize..=17,
            seed in any::<u64>(),
        ) {
            // Fixture: a cheap deterministic stream so the case shrinks reproducibly.
            let mut x = seed | 1;
            let messages: Vec<u8> = (0..len * count)
                .map(|_| {
                    x ^= x << 13;
                    x ^= x >> 7;
                    x ^= x << 17;
                    x as u8
                })
                .collect();

            let mut batched = vec![[0u8; 32]; count];
            Keccak256Hash.hash_many(&messages, &mut batched);

            prop_assert_eq!(batched, reference(&messages, len, count));
        }
    }
}
