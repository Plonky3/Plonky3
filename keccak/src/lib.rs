//! The Keccak-f permutation, and hash functions built from it.

#![no_std]

#[cfg(test)]
extern crate alloc;

use p3_symmetric::{CryptographicHasher, CryptographicPermutation, Permutation};
use tiny_keccak::{Hasher, Keccak, Sha3, keccakf};

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

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub mod wasm32_simd128;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub use wasm32_simd128::*;

#[cfg(not(any(
    all(target_arch = "aarch64", target_feature = "neon"),
    target_arch = "x86_64",
    all(target_arch = "wasm32", target_feature = "simd128")
)))]
mod fallback;
#[cfg(not(any(
    all(target_arch = "aarch64", target_feature = "neon"),
    target_arch = "x86_64",
    all(target_arch = "wasm32", target_feature = "simd128")
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

/// Byte rate of both 256-bit sponges.
///
/// Capacity is twice the 256-bit digest, taken out of the 1600-bit permutation.
/// That leaves 1088 bits, exactly 136 bytes, absorbed per permutation.
const RATE: usize = (1600 - 2 * 256) / 8;

/// Digest length of both 256-bit sponges in bytes.
const DIGEST_BYTES: usize = 32;

/// Index of the state word holding the last byte of the rate.
const LAST_RATE_WORD: usize = (RATE - 1) / 8;

/// Closing padding mark, placed at the last byte of the rate.
const CLOSING_MARK: u64 = 0x80u64 << (8 * ((RATE - 1) % 8));

/// First padding byte of Keccak-256.
///
/// The original submission appends the `pad10*1` rule directly after the message.
/// Its first one bit is the low bit of the byte.
const KECCAK_DOMAIN: u8 = 0x01;

/// First padding byte of SHA3-256.
///
/// FIPS 202 appends the two domain bits `01` before the `pad10*1` rule.
/// Bits enter the byte from the low end, so `0 1 1` reads as `0x06`.
const SHA3_DOMAIN: u8 = 0x06;

/// Exclusive-or one whole state word, every lane at once.
///
/// The permutation loads a state word as a single `VECTOR_LEN`-wide vector, and a partially
/// written word cannot be store-to-load forwarded into that load, so every write here covers
/// the word in full.
#[inline(always)]
fn xor_state_word(word: &mut [u64; VECTOR_LEN], value: [u64; VECTOR_LEN]) {
    for (lane, value) in word.iter_mut().zip(value) {
        *lane ^= value;
    }
}

/// Read the eight message bytes at `offset` in every lane into one state word value.
///
/// Bytes enter the state little-endian, eight to a state word, matching the Keccak convention.
#[inline(always)]
fn gather_word(lanes: &[&[u8]; VECTOR_LEN], offset: usize) -> [u64; VECTOR_LEN] {
    let mut word = [0u64; VECTOR_LEN];
    for (value, lane) in word.iter_mut().zip(lanes) {
        *value = u64::from_le_bytes(lane[offset..][..8].try_into().unwrap());
    }
    word
}

/// Absorb a run of whole state words, one message per lane, starting at `offset` in each.
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
/// Each row is assembled across all lanes first and then stored once, so no state word is
/// built out of partial writes. The transpose happens in registers; the messages themselves
/// are never copied or reordered.
#[inline(always)]
fn absorb_words(
    state: &mut [[u64; VECTOR_LEN]; 25],
    lanes: &[&[u8]; VECTOR_LEN],
    offset: usize,
    words: usize,
) {
    debug_assert!(words <= RATE / 8);

    for (word_index, word) in state[..words].iter_mut().enumerate() {
        xor_state_word(word, gather_word(lanes, offset + 8 * word_index));
    }
}

/// Absorb the final, shorter block of every lane and close it with the padding.
///
/// The first padding byte `d` carries the domain bits of the hash, then the rule's first one bit:
///
/// ```text
///     [ message bytes | d | 0x00 ... 0x00 | 0x80 ]
///                       ^                   ^
///                  block_len            rate - 1
///
///     Keccak-256:  d = 0x01
///     SHA3-256:    d = 0x06
/// ```
///
/// A block ending one byte short of the rate puts both marks in the same byte.
/// Exclusive-or makes that byte `d | 0x80`, exactly what the rule requires.
///
/// The leftover bytes and the marks meet in the word value before it reaches the state,
/// so the closing word is stored once like every other one.
#[inline(always)]
fn absorb_final_block<const DOMAIN: u8>(
    state: &mut [[u64; VECTOR_LEN]; 25],
    lanes: &[&[u8]; VECTOR_LEN],
    offset: usize,
    block_len: usize,
) {
    debug_assert!(block_len < RATE);

    let words = block_len / 8;
    let tail = block_len % 8;
    absorb_words(state, lanes, offset, words);

    // The first mark sits immediately after the last message byte, in the same word as any
    // leftover bytes. The second mark joins it when that word is already the closing word of
    // the rate.
    let mut marks = u64::from(DOMAIN) << (8 * tail);
    if words == LAST_RATE_WORD {
        marks ^= CLOSING_MARK;
    }
    let mut value = [0u64; VECTOR_LEN];
    for (value, lane) in value.iter_mut().zip(lanes) {
        let mut bytes = [0u8; 8];
        bytes[..tail].copy_from_slice(&lane[offset + 8 * words..][..tail]);
        *value = u64::from_le_bytes(bytes) ^ marks;
    }
    xor_state_word(&mut state[words], value);

    // The closing word is otherwise untouched by the message and the first mark, so it takes
    // the second mark alone.
    if words != LAST_RATE_WORD {
        xor_state_word(&mut state[LAST_RATE_WORD], [CLOSING_MARK; VECTOR_LEN]);
    }
}

/// Read the digest of one lane out of a permuted vectorized state.
#[inline(always)]
fn squeeze_lane(state: &[[u64; VECTOR_LEN]; 25], lane: usize) -> [u8; DIGEST_BYTES] {
    let mut digest = [0u8; DIGEST_BYTES];

    // The digest is the leading bytes of the rate portion, little-endian per state word.
    for (word_index, word) in digest.as_chunks_mut::<8>().0.iter_mut().enumerate() {
        *word = state[word_index][lane].to_le_bytes();
    }

    digest
}

/// Hash a batch of equal-length messages, one message per lane per permutation.
///
/// The const parameter is the first padding byte, which alone tells the two 256-bit hashes apart.
///
/// # Panics
///
/// Panics if the input length is not a whole multiple of the digest count.
#[inline(always)]
fn hash_many_with_domain<const DOMAIN: u8>(input: &[u8], out: &mut [[u8; DIGEST_BYTES]]) {
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

    // Empty messages all share one digest.
    // One padding-only block in a single state gives it, with no message to split.
    if len == 0 {
        let mut state = [[0u64; VECTOR_LEN]; 25];
        absorb_final_block::<DOMAIN>(&mut state, &[&[]; VECTOR_LEN], 0, 0);
        KeccakF.permute_mut(&mut state);
        out.fill(squeeze_lane(&state, 0));
        return;
    }

    // Whole rate blocks are absorbed in lockstep, then one shorter final block.
    // That final block carries the padding and is empty when the length divides the rate.
    let full_blocks = len / RATE;
    let final_block = len % RATE;

    // One message per lane per permutation.
    for (messages, digests) in input
        .chunks(len * VECTOR_LEN)
        .zip(out.chunks_mut(VECTOR_LEN))
    {
        let mut state = [[0u64; VECTOR_LEN]; 25];

        // A short last group repeats its first message in the spare lanes, which keeps the
        // lane count fixed at compile time. Those lanes are hashed alongside the requested
        // ones and never squeezed.
        let present = digests.len();
        let lanes: [&[u8]; VECTOR_LEN] = core::array::from_fn(|lane| {
            let index = if lane < present { lane } else { 0 };
            &messages[index * len..][..len]
        });

        // Every lane contributes its block, then one permutation advances all the sponges.
        for block in 0..full_blocks {
            absorb_words(&mut state, &lanes, block * RATE, RATE / 8);
            KeccakF.permute_mut(&mut state);
        }

        // The final partial block carries the padding and permutes once more.
        absorb_final_block::<DOMAIN>(&mut state, &lanes, full_blocks * RATE, final_block);
        KeccakF.permute_mut(&mut state);

        // Squeeze one digest per requested message.
        for (lane, digest) in digests.iter_mut().enumerate() {
            *digest = squeeze_lane(&state, lane);
        }
    }
}

/// Feed a byte stream to a one-message sponge and return its 256-bit digest.
#[inline]
fn finalize_iter<S: Hasher>(mut sponge: S, input: impl IntoIterator<Item = u8>) -> [u8; 32] {
    const BUFLEN: usize = 512; // Tweakable parameter; determined by experiment
    p3_util::apply_to_chunks::<BUFLEN, _, _>(input, |buf| sponge.update(buf));

    let mut output = [0u8; 32];
    sponge.finalize(&mut output);
    output
}

/// Feed a run of byte slices to a one-message sponge and return its 256-bit digest.
#[inline]
fn finalize_slices<'a, S: Hasher>(
    mut sponge: S,
    input: impl IntoIterator<Item = &'a [u8]>,
) -> [u8; 32] {
    for chunk in input {
        sponge.update(chunk);
    }

    let mut output = [0u8; 32];
    sponge.finalize(&mut output);
    output
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
        finalize_iter(Keccak::v256(), input)
    }

    fn hash_iter_slices<'a, I>(&self, input: I) -> [u8; 32]
    where
        I: IntoIterator<Item = &'a [u8]>,
    {
        finalize_slices(Keccak::v256(), input)
    }

    fn hash_many(&self, input: &[u8], out: &mut [[u8; 32]]) {
        hash_many_with_domain::<KECCAK_DOMAIN>(input, out);
    }
}

/// The SHA3-256 hash function of [FIPS 202](https://doi.org/10.6028/NIST.FIPS.202).
///
/// It runs the same sponge as Keccak-256, at the same cost.
///
/// Only the padding differs, so the two digests differ for every input:
///
/// ```text
///     Keccak-256("abc") = 4e03657a ea45a94f ...
///     SHA3-256("abc")   = 3a985da7 4fe225b2 ...
/// ```
#[derive(Copy, Clone, Debug)]
pub struct Sha3_256Hash;

impl CryptographicHasher<u8, [u8; 32]> for Sha3_256Hash {
    const LANES: usize = VECTOR_LEN;

    fn hash_iter<I>(&self, input: I) -> [u8; 32]
    where
        I: IntoIterator<Item = u8>,
    {
        finalize_iter(Sha3::v256(), input)
    }

    fn hash_iter_slices<'a, I>(&self, input: I) -> [u8; 32]
    where
        I: IntoIterator<Item = &'a [u8]>,
    {
        finalize_slices(Sha3::v256(), input)
    }

    fn hash_many(&self, input: &[u8], out: &mut [[u8; 32]]) {
        hash_many_with_domain::<SHA3_DOMAIN>(input, out);
    }
}

#[cfg(test)]
mod nist_tests;

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use hex_literal::hex;
    use proptest::prelude::*;
    use sha3::{Digest, Sha3_256};

    use super::*;

    /// Every message length that changes the shape of the absorb loop.
    ///
    /// The block boundaries are the interesting ones:
    ///
    /// - A length one short of the rate folds both padding marks into a single byte.
    /// - A length that is an exact multiple of the rate pads a block carrying no message bytes.
    /// - A length whose leftover bytes already sit in the closing word of the rate puts both
    ///   marks in that word, in different bytes.
    const SHAPE_LENGTHS: [usize; 16] = [
        0,
        1,
        7,
        8,
        9,
        RATE - 3,
        RATE - 1,
        RATE,
        RATE + 1,
        2 * RATE - 1,
        2 * RATE,
        2 * RATE + 1,
        3 * RATE,
        8 * LAST_RATE_WORD,
        400,
        1000,
    ];

    /// Hash each message on its own, which is the behaviour the batched path must reproduce.
    fn reference<H>(hasher: &H, messages: &[u8], len: usize, count: usize) -> Vec<[u8; 32]>
    where
        H: CryptographicHasher<u8, [u8; 32]>,
    {
        (0..count)
            .map(|k| {
                // A zero-length message still has a digest, so slice defensively.
                let message = if len == 0 {
                    &messages[..0]
                } else {
                    &messages[k * len..(k + 1) * len]
                };
                hasher.hash_slice(message)
            })
            .collect()
    }

    /// Batch every shape length at every count up to two lane groups plus one.
    fn check_block_shapes<H>(hasher: &H)
    where
        H: CryptographicHasher<u8, [u8; 32]>,
    {
        // Batch sizes below, at, and above one full lane group.
        // The final short group is then exercised at every lane count the target compiles to.
        let counts: Vec<usize> = (1..=2 * VECTOR_LEN + 1).collect();

        for len in SHAPE_LENGTHS {
            for &count in &counts {
                // Fixture: a deterministic byte ramp, distinct per position.
                let messages: Vec<u8> = (0..len * count).map(|i| (i * 31 + 7) as u8).collect();

                let mut batched = vec![[0u8; 32]; count];
                hasher.hash_many(&messages, &mut batched);

                assert_eq!(
                    batched,
                    reference(hasher, &messages, len, count),
                    "len {len}, count {count}"
                );
            }
        }
    }

    #[test]
    fn keccak256_hash_many_matches_scalar_across_block_shapes() {
        check_block_shapes(&Keccak256Hash);
    }

    #[test]
    fn sha3_256_hash_many_matches_scalar_across_block_shapes() {
        check_block_shapes(&Sha3_256Hash);
    }

    #[test]
    fn hash_many_splits_input_by_digest_count() {
        // The message length is the input length divided by the digest count.
        // 64 bytes and 4 digests therefore read as four adjacent 16-byte messages.
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
        // 5 bytes cannot split into 2 equal messages.
        // The contract fails up front rather than misaligning message boundaries.
        let mut digests = [[0u8; 32]; 2];
        Sha3_256Hash.hash_many(&[1, 2, 3, 4, 5], &mut digests);
    }

    #[test]
    fn the_two_hashes_differ_only_by_padding() {
        // Both sponges run the same permutation at the same rate.
        // The first padding byte alone separates them:
        //
        //     Keccak-256:  "abc" | 0x01 | 0x00 ... 0x00 | 0x80
        //     SHA3-256:    "abc" | 0x06 | 0x00 ... 0x00 | 0x80
        //
        // Published digests of "abc" for each.
        let keccak = hex!("4e03657aea45a94fc7d47ba826c8d667c0d1e6e33a64a036ec44f58fa12d6c45");
        let sha3 = hex!("3a985da74fe225b2045c172d6bd390bd855f086e3e9d525b46bfe24511431532");

        // Scalar and batched paths agree with the published values.
        let mut batched = [[0u8; 32]; 1];
        Keccak256Hash.hash_many(b"abc", &mut batched);
        assert_eq!(batched[0], keccak);
        assert_eq!(Keccak256Hash.hash_slice(b"abc"), keccak);

        Sha3_256Hash.hash_many(b"abc", &mut batched);
        assert_eq!(batched[0], sha3);
        assert_eq!(Sha3_256Hash.hash_slice(b"abc"), sha3);
    }

    #[test]
    fn empty_messages_hash_to_the_published_digests() {
        // Zero-length messages take the padding-only branch of the batched path.
        // Three of them in one call must all get the digest of the empty string.
        let keccak = hex!("c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470");
        let sha3 = hex!("a7ffc6f8bf1ed76651c14756a061d662f580ff4de43b49fa82d80a4b80f8434a");

        let mut digests = [[0u8; 32]; 3];
        Keccak256Hash.hash_many(&[], &mut digests);
        assert_eq!(digests, [keccak; 3]);

        Sha3_256Hash.hash_many(&[], &mut digests);
        assert_eq!(digests, [sha3; 3]);
    }

    /// A cheap deterministic byte stream, so a failing case shrinks reproducibly.
    fn stream(seed: u64, len: usize) -> Vec<u8> {
        let mut x = seed | 1;
        (0..len)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                x as u8
            })
            .collect()
    }

    proptest! {
        #[test]
        fn keccak256_hash_many_matches_scalar_on_random_batches(
            len in 0usize..=400,
            count in 1usize..=17,
            seed in any::<u64>(),
        ) {
            let messages = stream(seed, len * count);

            let mut batched = vec![[0u8; 32]; count];
            Keccak256Hash.hash_many(&messages, &mut batched);

            prop_assert_eq!(batched, reference(&Keccak256Hash, &messages, len, count));
        }

        #[test]
        fn sha3_256_matches_an_independent_implementation(
            len in 0usize..=600,
            count in 1usize..=2 * VECTOR_LEN + 1,
            seed in any::<u64>(),
        ) {
            // Fixture: `count` random messages of `len` bytes, back to back.
            let messages = stream(seed, len * count);

            // Batched and one-message digests of this crate.
            let mut batched = vec![[0u8; 32]; count];
            Sha3_256Hash.hash_many(&messages, &mut batched);
            let scalar = reference(&Sha3_256Hash, &messages, len, count);

            // Digests of the RustCrypto implementation, which shares no code with this one.
            let expected: Vec<[u8; 32]> = (0..count)
                .map(|k| Sha3_256::digest(&messages[k * len..(k + 1) * len]).into())
                .collect();

            prop_assert_eq!(&batched, &expected);
            prop_assert_eq!(&scalar, &expected);
        }
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    mod wasm_simd_tests {
        use tiny_keccak::keccakf;

        use super::*;

        fn permute_packed(states: [[u64; 25]; 2]) -> [[u64; 25]; 2] {
            let mut packed = core::array::from_fn(|word| [states[0][word], states[1][word]]);
            KeccakF.permute_mut(&mut packed);
            core::array::from_fn(|lane| core::array::from_fn(|word| packed[word][lane]))
        }

        fn permute_scalar(mut states: [[u64; 25]; 2]) -> [[u64; 25]; 2] {
            keccakf(&mut states[0]);
            keccakf(&mut states[1]);
            states
        }

        #[test]
        fn simd_permutation_matches_scalar_for_zero_and_ones() {
            let states = [[0; 25], [u64::MAX; 25]];
            assert_eq!(permute_packed(states), permute_scalar(states));
        }

        #[test]
        fn simd_lanes_do_not_influence_each_other() {
            let lane0 =
                core::array::from_fn(|i| 0x9e37_79b9_7f4a_7c15u64.wrapping_mul(i as u64 + 1));
            let lane1_a = core::array::from_fn(|i| (i as u64).rotate_left(i as u32));
            let lane1_b = core::array::from_fn(|i| !(i as u64).wrapping_mul(0x0101_0101_0101_0101));

            let output_a = permute_packed([lane0, lane1_a]);
            let output_b = permute_packed([lane0, lane1_b]);

            assert_eq!(output_a[0], output_b[0]);
            assert_eq!(output_a, permute_scalar([lane0, lane1_a]));
            assert_eq!(output_b, permute_scalar([lane0, lane1_b]));
        }

        proptest! {
            #[test]
            fn simd_permutation_matches_scalar_for_arbitrary_words(
                states in prop::array::uniform2(prop::array::uniform25(any::<u64>()))
            ) {
                let expected = permute_scalar(states);
                let computed = permute_packed(states);
                prop_assert_eq!(computed, expected);
            }
        }
    }
}
