//! The blake3 hash function.

#![no_std]

#[cfg(test)]
extern crate alloc;

// Batching independent messages has exactly one entry point in `blake3`: the `platform` module,
// which the crate marks `#[doc(hidden)]` and labels "undocumented and unstable, for benchmarks
// only". Depending on it is a deliberate risk, taken because nothing stable replaces it.
//
// The stable `hazmat` module is not that replacement. It batches the chunks *inside* one message,
// keyed by each chunk's input offset, and independent messages all start at offset 0, so no
// hazmat call can ever hold more than one of them.
//
// The digest-equality tests below pin every batched digest to `blake3::hash` and to the scalar
// path at every message shape, but they only guard this workspace's CI. `blake3` enters as a
// caret `1.8.4` and the lockfile is not committed, so a 1.x release that reshapes
// `Platform::hash_many` breaks every already-published `p3-blake3` on the next `cargo update`,
// with a patch release the only way out. `platform.rs` did change in 1.8.7, when `arrayref` went.
use blake3::platform::{MAX_SIMD_DEGREE, Platform};
use blake3::{BLOCK_LEN, CHUNK_LEN, IncrementCounter, OUT_LEN};
use p3_symmetric::CryptographicHasher;

/// Initialization vector of BLAKE3, which is also the SHA-256 one.
///
/// A chunk in plain hash mode starts its chaining value here.
/// The `blake3` crate keeps its own copy private, so the words are spelled out.
const IV: [u32; 8] = [
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
];

/// Domain flag set on the first block of a chunk.
const CHUNK_START: u8 = 1 << 0;

/// Domain flag set on the last block of a chunk.
const CHUNK_END: u8 = 1 << 1;

/// Domain flag set on the one compression that produces the output.
///
/// The 32-byte chaining value of a compression carrying it is the BLAKE3 digest.
const ROOT: u8 = 1 << 3;

/// Whole blocks a single-chunk message can hold, which bounds the batched dispatch.
const MAX_BLOCKS_PER_CHUNK: usize = CHUNK_LEN / BLOCK_LEN;

/// Read a chaining value from its 32-byte little-endian form.
#[inline]
fn cv_words(bytes: &[u8; OUT_LEN]) -> [u32; 8] {
    core::array::from_fn(|word| u32::from_le_bytes(bytes[4 * word..][..4].try_into().unwrap()))
}

/// Write a chaining value back out in its 32-byte little-endian form.
#[inline]
fn cv_bytes(words: &[u32; 8]) -> [u8; OUT_LEN] {
    let mut bytes = [0u8; OUT_LEN];
    for (word, out) in words.iter().zip(bytes.as_chunks_mut::<4>().0) {
        *out = word.to_le_bytes();
    }
    bytes
}

/// Advance the `N`-byte whole-block prefix of every message in a batch.
///
/// Every lane starts from the initialization vector at counter zero.
/// That is the state of a chunk sitting at the start of its own message.
///
/// ```text
///     lane 0  [ block 0 | block 1 | ... ]  ->  cv_0
///     lane 1  [ block 0 | block 1 | ... ]  ->  cv_1
///     ...                                      one SIMD call per lane group
/// ```
///
/// # Arguments
///
/// - `platform`: the widest compression the running CPU supports.
/// - `messages`: the batch, one message of `len` bytes after another.
///
/// - `len`: stride from one message to the next, at least `N`.
/// - `flags_end`: extra flags for the last block of the prefix, empty unless the message ends there.
///
/// - `out`: one slot per message, receiving the chaining value after the prefix.
fn hash_prefix<const N: usize>(
    platform: Platform,
    messages: &[u8],
    len: usize,
    flags_end: u8,
    out: &mut [[u8; OUT_LEN]],
) {
    for (group, digests) in messages
        .chunks(len * MAX_SIMD_DEGREE)
        .zip(out.chunks_mut(MAX_SIMD_DEGREE))
    {
        let count = digests.len();

        // Idle lanes of a short final group point at the first message, and their output is dropped.
        // Only the leading `count` entries are ever handed to the compression.
        let lanes: [&[u8; N]; MAX_SIMD_DEGREE] = core::array::from_fn(|lane| {
            let message = if lane < count { lane } else { 0 };
            group[message * len..][..N].try_into().unwrap()
        });

        platform.hash_many(
            &lanes[..count],
            &IV,
            0,
            IncrementCounter::No,
            0,
            CHUNK_START,
            flags_end,
            digests.as_flattened_mut(),
        );
    }
}

/// Pick the batched prefix hash for a block count known only at run time.
///
/// `Platform::hash_many` takes its message length as a const generic.
/// Every block count is therefore its own instantiation, listed out by the macro below.
///
/// The list is finite because a single-chunk prefix holds at most [`MAX_BLOCKS_PER_CHUNK`] blocks.
fn hash_prefix_blocks(
    blocks: usize,
    platform: Platform,
    messages: &[u8],
    len: usize,
    flags_end: u8,
    out: &mut [[u8; OUT_LEN]],
) {
    macro_rules! dispatch {
        ($($blocks:literal),+) => {
            match blocks {
                $($blocks => hash_prefix::<{ $blocks * BLOCK_LEN }>(
                    platform, messages, len, flags_end, out,
                ),)+
                _ => unreachable!("a single-chunk prefix holds at most {MAX_BLOCKS_PER_CHUNK} blocks"),
            }
        };
    }

    dispatch!(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
}

/// The blake3 hash function.
#[derive(Copy, Clone, Debug)]
pub struct Blake3;

impl CryptographicHasher<u8, [u8; 32]> for Blake3 {
    /// Independent chunk states one batched compression call advances.
    ///
    /// `MAX_SIMD_DEGREE` is what `blake3`'s own build compiled in, not what the running CPU can
    /// do: on x86 it is 16 whenever the build script found a C compiler accepting `-mavx512f`
    /// and `-mavx512vl` and 8 otherwise, whatever the host supports; it is 4 under NEON and
    /// under wasm32 with the `wasm32-simd` feature, and 1 with no vector backend at all.
    ///
    /// Reporting that compile-time maximum is the useful answer for a caller grouping messages.
    /// The narrower width that `Platform::detect` finds at run time only splits one group into
    /// several compressions, which costs nothing the grouping was counting on.
    const LANES: usize = MAX_SIMD_DEGREE;

    fn hash_iter<I>(&self, input: I) -> [u8; 32]
    where
        I: IntoIterator<Item = u8>,
    {
        const BUFLEN: usize = 512; // Tweakable parameter; determined by experiment
        let mut hasher = blake3::Hasher::new();
        p3_util::apply_to_chunks::<BUFLEN, _, _>(input, |buf| {
            hasher.update(buf);
        });
        hasher.finalize().into()
    }

    fn hash_iter_slices<'a, I>(&self, input: I) -> [u8; 32]
    where
        I: IntoIterator<Item = &'a [u8]>,
    {
        let mut hasher = blake3::Hasher::new();
        for chunk in input {
            hasher.update(chunk);
        }
        hasher.finalize().into()
    }

    /// Hash a batch of equal-length messages, several of them per compression call.
    ///
    /// A message of at most [`CHUNK_LEN`] bytes is a single BLAKE3 chunk.
    /// Its blocks are a plain chain from the initialization vector, with the chunk counter at zero.
    ///
    /// ```text
    ///     IV -> block 0 -> block 1 -> ... -> block k        flags: CHUNK_START on 0
    ///                                            |                 CHUNK_END | ROOT on k
    ///                                            v
    ///                                         digest
    /// ```
    ///
    /// Lanes share that shape exactly, so one vector call advances a whole group of messages.
    ///
    /// A longer message spans several chunks whose chaining values fold into a tree.
    /// Those go to the scalar hasher, which already drives the SIMD path over the chunks of one message.
    ///
    /// # Panics
    ///
    /// Panics if the batch is ragged: the input length must be a whole multiple of the digest count.
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

        // Empty messages all share one digest and drive no compression loop at all.
        if len == 0 {
            out.fill(self.hash_iter(core::iter::empty()));
            return;
        }

        // Past one chunk the message is a tree, which the batch layout cannot express.
        if len > CHUNK_LEN {
            for (digest, message) in out.iter_mut().zip(input.chunks_exact(len)) {
                *digest = self.hash_slice(message);
            }
            return;
        }

        let platform = Platform::detect();
        let blocks = len / BLOCK_LEN;
        let tail = len % BLOCK_LEN;

        // Whole blocks run in lockstep across the lanes.
        // They carry the end marks only when the message stops on that boundary.
        if blocks > 0 {
            let flags_end = if tail == 0 { CHUNK_END | ROOT } else { 0 };
            hash_prefix_blocks(blocks, platform, input, len, flags_end, out);
        }

        // A length on a block boundary is already finished.
        if tail == 0 {
            return;
        }

        // The short final block is one compression per message, over that message's own chaining value.
        // A message shorter than a block has no prefix, so its single compression opens the chunk too.
        let flags = if blocks == 0 {
            CHUNK_START | CHUNK_END | ROOT
        } else {
            CHUNK_END | ROOT
        };
        // A compression always reads a full block, with the bytes past the message left zero.
        // Every message ends at the same offset, so that zero padding is written once for the batch.
        let mut block = [0u8; BLOCK_LEN];
        for (digest, message) in out.iter_mut().zip(input.chunks_exact(len)) {
            let mut cv = if blocks == 0 { IV } else { cv_words(digest) };

            // The true length is mixed in separately, so the padding cannot collide.
            block[..tail].copy_from_slice(&message[blocks * BLOCK_LEN..]);
            platform.compress_in_place(&mut cv, &block, tail as u8, 0, flags);

            *digest = cv_bytes(&cv);
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use proptest::prelude::*;

    use super::*;

    /// Every message length that changes the shape of the batched path.
    ///
    /// The interesting ones are the boundaries:
    ///
    /// - Below a block there is no batched prefix, only the lone final compression.
    /// - On a block boundary the prefix carries the end marks and no tail compression runs.
    /// - Past a chunk the message becomes a tree and the scalar hasher takes over.
    const SHAPE_LENGTHS: [usize; 11] = [0, 1, 31, 32, 63, 64, 65, 1023, 1024, 1025, 2048];

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
                Blake3.hash_slice(message)
            })
            .collect()
    }

    #[test]
    fn hash_many_matches_scalar_across_block_shapes() {
        // Batch sizes below, at, and above one full lane group.
        // The final short group is then exercised at every lane count the target compiles to.
        let counts: Vec<usize> = (1..=2 * MAX_SIMD_DEGREE + 1).collect();

        for len in SHAPE_LENGTHS {
            for &count in &counts {
                // Fixture: a xorshift stream, whose period keeps every message in the batch
                // distinct. A byte ramp would repeat every 256 bytes and hide a lane swap.
                let mut x = 0x2545_f491_4f6c_dd1du64;
                let messages: Vec<u8> = (0..len * count)
                    .map(|_| {
                        x ^= x << 13;
                        x ^= x >> 7;
                        x ^= x << 17;
                        x as u8
                    })
                    .collect();

                let mut batched = vec![[0u8; 32]; count];
                Blake3.hash_many(&messages, &mut batched);

                assert_eq!(
                    batched,
                    reference(&messages, len, count),
                    "len {len}, count {count}"
                );
            }
        }
    }

    #[test]
    fn hash_many_matches_the_upstream_hash_function() {
        // The digest is a commitment, so it must be plain root BLAKE3 and not a chaining value.
        // Checking against `blake3::hash` pins that independently of this crate's scalar path.
        for len in SHAPE_LENGTHS {
            let message: Vec<u8> = (0..len).map(|i| (i * 17 + 3) as u8).collect();

            let mut batched = [[0u8; 32]; 1];
            Blake3.hash_many(&message, &mut batched);

            assert_eq!(batched[0], *blake3::hash(&message).as_bytes(), "len {len}");
        }
    }

    #[test]
    fn hash_many_splits_input_by_digest_count() {
        // The message length is the input length divided by the digest count.
        // 256 bytes and 4 digests therefore read as four adjacent 64-byte messages.
        let messages: Vec<u8> = (0..256).map(|i| i as u8).collect();

        let mut digests = [[0u8; 32]; 4];
        Blake3.hash_many(&messages, &mut digests);

        for (k, digest) in digests.iter().enumerate() {
            assert_eq!(*digest, Blake3.hash_slice(&messages[k * 64..][..64]));
        }
    }

    #[test]
    fn hash_many_reads_nothing_when_no_digests_are_requested() {
        // A message length cannot be derived from zero digests, so the input is left untouched.
        Blake3.hash_many(&[1, 2, 3], &mut []);
    }

    #[test]
    #[should_panic(expected = "must be a whole multiple")]
    fn hash_many_rejects_ragged_input() {
        // 5 bytes cannot split into 2 equal messages.
        // The contract fails up front rather than misaligning message boundaries.
        let mut digests = [[0u8; 32]; 2];
        Blake3.hash_many(&[1, 2, 3, 4, 5], &mut digests);
    }

    #[test]
    fn chaining_value_round_trips_through_its_byte_form() {
        // The tail compression reads back the prefix digest, so the two conversions must invert.
        let words: [u32; 8] = core::array::from_fn(|i| 0x0123_4567u32.wrapping_mul(i as u32 + 1));
        assert_eq!(cv_words(&cv_bytes(&words)), words);
    }

    proptest! {
        #[test]
        fn hash_many_matches_scalar_on_random_batches(
            len in 0usize..=1200,
            count in 1usize..=2 * MAX_SIMD_DEGREE + 1,
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
            Blake3.hash_many(&messages, &mut batched);

            prop_assert_eq!(batched, reference(&messages, len, count));
        }
    }
}
