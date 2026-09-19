//! The SHA2-256 hash function.
//!
//! [`Sha256::hash_many`] and [`Sha256Compress::compress_many`] hash four messages at a time on
//! the targets that have a four-lane backend: wasm32 with `simd128`, x86-64 with `sha` and
//! `sse4.1`, and AArch64 with `neon` and `sha2`. The choice is made at compile time, so an x86-64
//! build needs `-C target-feature=+sha` and an AArch64 Linux build needs `-C target-feature=+sha2`
//! (or `-C target-cpu=native`) to get it; no x86-64 microarchitecture level turns `sha` on by
//! itself. The Apple silicon targets enable `sha2` by default.
//!
//! A build without one hashes a message at a time through `sha2`, which detects the hardware SHA
//! extension (SHA-NI or the ARMv8 SHA-2 extension) at runtime on its own. What the four-lane
//! backends add is the interleaving of four independent streams.

#![no_std]

#[cfg(test)]
extern crate alloc;

use p3_symmetric::{CompressionFunction, CryptographicHasher, PseudoCompressionFunction};
use sha2::Digest;

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm32_simd128;

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "sha",
    target_feature = "sse4.1"
))]
mod x86_64_sha_ni;

#[cfg(all(
    target_arch = "aarch64",
    target_feature = "neon",
    target_feature = "sha2"
))]
mod aarch64_sha2;

/// The four-lane backend this target compiles, if it has one.
///
/// Every batched entry point below goes through this one name, so the target tests appear once
/// per backend rather than once per method.
#[cfg(all(
    target_arch = "aarch64",
    target_feature = "neon",
    target_feature = "sha2"
))]
use aarch64_sha2::ArmSha2 as Backend;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use wasm32_simd128::Simd128 as Backend;
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "sha",
    target_feature = "sse4.1"
))]
use x86_64_sha_ni::ShaNi as Backend;

pub const H256_256: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

/// The SHA2-256 hash function.
#[derive(Copy, Clone, Debug)]
pub struct Sha256;

impl CryptographicHasher<u8, [u8; 32]> for Sha256 {
    #[cfg(any(
        all(target_arch = "wasm32", target_feature = "simd128"),
        all(
            target_arch = "x86_64",
            target_feature = "sha",
            target_feature = "sse4.1"
        ),
        all(
            target_arch = "aarch64",
            target_feature = "neon",
            target_feature = "sha2"
        )
    ))]
    const LANES: usize = four_lane::LANES;

    fn hash_iter<I>(&self, input: I) -> [u8; 32]
    where
        I: IntoIterator<Item = u8>,
    {
        const BUFLEN: usize = 512; // Tweakable parameter; determined by experiment
        let mut hasher = sha2::Sha256::new();
        p3_util::apply_to_chunks::<BUFLEN, _, _>(input, |buf| hasher.update(buf));
        hasher.finalize().into()
    }

    #[inline]
    fn hash_iter_slices<'a, I>(&self, input: I) -> [u8; 32]
    where
        I: IntoIterator<Item = &'a [u8]>,
    {
        let mut hasher = sha2::Sha256::new();
        for chunk in input {
            hasher.update(chunk);
        }
        hasher.finalize().into()
    }

    #[cfg(any(
        all(target_arch = "wasm32", target_feature = "simd128"),
        all(
            target_arch = "x86_64",
            target_feature = "sha",
            target_feature = "sse4.1"
        ),
        all(
            target_arch = "aarch64",
            target_feature = "neon",
            target_feature = "sha2"
        )
    ))]
    fn hash_many(&self, input: &[u8], out: &mut [[u8; 32]]) {
        four_lane::hash_many::<Backend>(input, out);
    }
}

/// SHA2-256 without the padding (pre-processing), intended to be used
/// as a 2-to-1 [PseudoCompressionFunction].
#[derive(Copy, Clone, Debug)]
pub struct Sha256Compress;

impl PseudoCompressionFunction<[u8; 32], 2> for Sha256Compress {
    #[cfg(any(
        all(target_arch = "wasm32", target_feature = "simd128"),
        all(
            target_arch = "x86_64",
            target_feature = "sha",
            target_feature = "sse4.1"
        ),
        all(
            target_arch = "aarch64",
            target_feature = "neon",
            target_feature = "sha2"
        )
    ))]
    const LANES: usize = four_lane::LANES;

    fn compress(&self, input: [[u8; 32]; 2]) -> [u8; 32] {
        let mut state = H256_256;
        // [[u8; 32]; 2] has same memory layout as [u8; 64]
        let block: &[u8; 64] = unsafe { core::mem::transmute(&input) };
        sha2::block_api::compress256(&mut state, core::slice::from_ref(block));

        let mut output = [0u8; 32];
        for (chunk, word) in output.as_chunks_mut::<4>().0.iter_mut().zip(state) {
            chunk.copy_from_slice(&word.to_be_bytes());
        }
        output
    }

    #[cfg(any(
        all(target_arch = "wasm32", target_feature = "simd128"),
        all(
            target_arch = "x86_64",
            target_feature = "sha",
            target_feature = "sse4.1"
        ),
        all(
            target_arch = "aarch64",
            target_feature = "neon",
            target_feature = "sha2"
        )
    ))]
    fn compress_many(&self, inputs: &[[[u8; 32]; 2]], out: &mut [[u8; 32]]) {
        four_lane::compress_many::<Backend>(inputs, out);
    }
}

impl CompressionFunction<[u8; 32], 2> for Sha256Compress {}

/// Padding and batching shared by the four-lane backends.
///
/// A backend supplies only the vector core, through [`FourLane`](four_lane::FourLane). Everything
/// that turns a batch of messages into blocks lives here, so a padding fix cannot reach one
/// backend and miss the others.
#[cfg(any(
    all(target_arch = "wasm32", target_feature = "simd128"),
    all(
        target_arch = "x86_64",
        target_feature = "sha",
        target_feature = "sse4.1"
    ),
    all(
        target_arch = "aarch64",
        target_feature = "neon",
        target_feature = "sha2"
    )
))]
mod four_lane {
    use core::mem::transmute;

    use p3_symmetric::{CryptographicHasher, PseudoCompressionFunction};

    use crate::{Sha256, Sha256Compress};

    /// Number of messages a backend advances per call.
    pub(crate) const LANES: usize = 4;

    /// Bytes in one SHA-256 compression block.
    pub(crate) const BLOCK_BYTES: usize = 64;

    /// The SHA-256 round constants, in round order.
    ///
    /// A backend may read four consecutive entries as one vector, so the natural order is the
    /// useful one and the rows below are eight rounds wide.
    #[rustfmt::skip]
    pub(crate) const ROUND_CONSTANTS: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    ];

    /// The vector core of one backend: four SHA-256 states advanced side by side.
    ///
    /// These three methods are the whole architecture-specific surface. Blocks arrive fully
    /// padded, so an implementation never sees a message length.
    pub(crate) trait FourLane {
        /// Four independent chaining states, held however the backend packs them.
        type State: Copy;

        /// Four copies of the SHA-256 initialization vector.
        fn initial_state() -> Self::State;

        /// Absorb one block into each state, lane `i` taking `blocks[i]`.
        fn compress(state: &mut Self::State, blocks: [&[u8; BLOCK_BYTES]; LANES]);

        /// Serialize the four states as big-endian digests.
        fn write_digests(state: &Self::State, out: &mut [[u8; 32]; LANES]);
    }

    /// Hash four messages of a common length, padding included.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if `messages` does not hold exactly four messages of `len` bytes.
    fn hash_four<B: FourLane>(messages: &[u8], len: usize) -> [[u8; 32]; LANES] {
        debug_assert_eq!(messages.len(), len * LANES);
        let lane_messages: [&[u8]; LANES] =
            core::array::from_fn(|lane| &messages[lane * len..(lane + 1) * len]);
        let mut state = B::initial_state();

        for block in 0..len / BLOCK_BYTES {
            let blocks = core::array::from_fn(|lane| {
                lane_messages[lane][block * BLOCK_BYTES..][..BLOCK_BYTES]
                    .try_into()
                    .unwrap()
            });
            B::compress(&mut state, blocks);
        }

        let remainder = len % BLOCK_BYTES;
        let mut final_blocks = [[0u8; BLOCK_BYTES]; LANES];
        for lane in 0..LANES {
            final_blocks[lane][..remainder]
                .copy_from_slice(&lane_messages[lane][len - remainder..]);
            final_blocks[lane][remainder] = 0x80;
        }

        // The length counter occupies the last eight bytes, so a long tail needs a second block.
        let bit_len = (len as u64).wrapping_mul(8).to_be_bytes();
        if remainder >= BLOCK_BYTES - 8 {
            B::compress(&mut state, final_blocks.each_ref());
            final_blocks = [[0u8; BLOCK_BYTES]; LANES];
        }
        for block in &mut final_blocks {
            block[BLOCK_BYTES - 8..].copy_from_slice(&bit_len);
        }
        B::compress(&mut state, final_blocks.each_ref());

        let mut out = [[0u8; 32]; LANES];
        B::write_digests(&state, &mut out);
        out
    }

    /// Hash `out.len()` equal-length messages laid end to end in `input`.
    ///
    /// Whole groups of four go through the backend and the short tail falls back to the scalar
    /// hasher, so the batch count is unconstrained.
    ///
    /// # Panics
    ///
    /// Panics if `input.len()` is not a whole multiple of `out.len()`.
    pub(crate) fn hash_many<B: FourLane>(input: &[u8], out: &mut [[u8; 32]]) {
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
        let full_groups = out.len() / LANES;
        for group in 0..full_groups {
            let message_start = group * LANES * len;
            let digests = hash_four::<B>(&input[message_start..message_start + LANES * len], len);
            out[group * LANES..(group + 1) * LANES].copy_from_slice(&digests);
        }

        for message in full_groups * LANES..out.len() {
            out[message] = Sha256.hash_slice(&input[message * len..(message + 1) * len]);
        }
    }

    /// Compress each 64-byte pair, without padding, exactly as [`Sha256Compress::compress`] does.
    ///
    /// # Panics
    ///
    /// Panics if the input and output counts differ.
    pub(crate) fn compress_many<B: FourLane>(inputs: &[[[u8; 32]; 2]], out: &mut [[u8; 32]]) {
        assert_eq!(
            inputs.len(),
            out.len(),
            "group count ({}) must equal the output count ({})",
            inputs.len(),
            out.len()
        );

        let full_groups = out.len() / LANES;
        for group in 0..full_groups {
            let mut state = B::initial_state();
            let blocks: [&[u8; BLOCK_BYTES]; LANES] = core::array::from_fn(|lane| {
                // SAFETY: `[[u8; 32]; 2]` and `[u8; 64]` have identical contiguous byte layouts.
                unsafe { transmute(&inputs[group * LANES + lane]) }
            });
            B::compress(&mut state, blocks);

            let mut digests = [[0u8; 32]; LANES];
            B::write_digests(&state, &mut digests);
            out[group * LANES..(group + 1) * LANES].copy_from_slice(&digests);
        }

        for group in full_groups * LANES..out.len() {
            out[group] = Sha256Compress.compress(inputs[group]);
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use hex_literal::hex;
    use p3_symmetric::{CryptographicHasher, PseudoCompressionFunction};
    use proptest::prelude::*;

    use crate::{Sha256, Sha256Compress};

    // Message lengths that change the shape of the padding: 55 fits the mark and the counter in
    // one block and 56 does not, while 64 and 128 pad a block carrying no message bytes at all.
    const SHAPE_LENGTHS: [usize; 12] = [0, 1, 55, 56, 63, 64, 65, 119, 120, 127, 128, 200];

    // Batch sizes below, at, and above the four lanes of the vectorized backends, so the scalar
    // tail after the last full group is exercised at every remainder.
    const BATCH_COUNTS: [usize; 11] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 17];

    // Hash each message on its own, which is the behaviour a batched backend must reproduce.
    fn reference(messages: &[u8], len: usize, count: usize) -> Vec<[u8; 32]> {
        (0..count)
            .map(|k| Sha256.hash_slice(&messages[k * len..k * len + len]))
            .collect()
    }

    // A cheap deterministic stream, so a failing case reproduces exactly.
    fn random_bytes(len: usize, mut state: u64) -> Vec<u8> {
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state as u8
            })
            .collect()
    }

    // SHA-256 compression written directly from FIPS 180-4 §6.2.2, with no intrinsics.
    //
    // On AArch64 `sha2` detects the SHA-2 extension at runtime, so comparing a backend against
    // `Sha256` there compares two users of the same instructions. This is the independent oracle.
    fn spec_compress(state: &mut [u32; 8], block: &[u8; 64]) {
        // Transcribed from FIPS 180-4 §4.2.2 rather than shared with the backends, so a mistake
        // in their table cannot hide by appearing on both sides of the comparison.
        #[rustfmt::skip]
        const K: [u32; 64] = [
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
            0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
            0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
            0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
            0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
            0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
            0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
        ];
        let mut w = [0u32; 64];
        for (word, bytes) in w.iter_mut().zip(block.as_chunks::<4>().0) {
            *word = u32::from_be_bytes(*bytes);
        }
        for t in 16..64 {
            let s0 = w[t - 15].rotate_right(7) ^ w[t - 15].rotate_right(18) ^ (w[t - 15] >> 3);
            let s1 = w[t - 2].rotate_right(17) ^ w[t - 2].rotate_right(19) ^ (w[t - 2] >> 10);
            w[t] = w[t - 16]
                .wrapping_add(s0)
                .wrapping_add(w[t - 7])
                .wrapping_add(s1);
        }

        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;
        for t in 0..64 {
            let big_s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ (!e & g);
            let t1 = h
                .wrapping_add(big_s1)
                .wrapping_add(ch)
                .wrapping_add(K[t])
                .wrapping_add(w[t]);
            let big_s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = big_s0.wrapping_add(maj);
            (h, g, f, e, d, c, b, a) = (g, f, e, d.wrapping_add(t1), c, b, a, t1.wrapping_add(t2));
        }
        for (word, add) in state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
            *word = word.wrapping_add(add);
        }
    }

    // `Sha256Compress::compress` as the specification defines it: one block from the IV.
    fn spec_compress_pair(input: [[u8; 32]; 2]) -> [u8; 32] {
        let mut block = [0u8; 64];
        block[..32].copy_from_slice(&input[0]);
        block[32..].copy_from_slice(&input[1]);
        let mut state = crate::H256_256;
        spec_compress(&mut state, &block);

        let mut out = [0u8; 32];
        for (bytes, word) in out.as_chunks_mut::<4>().0.iter_mut().zip(state) {
            *bytes = word.to_be_bytes();
        }
        out
    }

    // SHA-256 of a whole message as the specification defines it: FIPS 180-4 §5.1.1 padding, then
    // `spec_compress` over every block from the IV. Shares no code with the backends or `sha2`.
    fn spec_hash(message: &[u8]) -> [u8; 32] {
        let mut padded = message.to_vec();
        padded.push(0x80);
        while padded.len() % 64 != 56 {
            padded.push(0);
        }
        padded.extend_from_slice(&((message.len() as u64) * 8).to_be_bytes());

        let mut state = crate::H256_256;
        for block in padded.as_chunks::<64>().0 {
            spec_compress(&mut state, block);
        }

        let mut out = [0u8; 32];
        for (bytes, word) in out.as_chunks_mut::<4>().0.iter_mut().zip(state) {
            *bytes = word.to_be_bytes();
        }
        out
    }

    // Reinterpret a flat byte run as the 64-byte pairs `compress_many` consumes.
    fn compression_inputs(bytes: &[u8]) -> Vec<[[u8; 32]; 2]> {
        bytes
            .as_chunks::<64>()
            .0
            .iter()
            .map(|block| {
                [
                    block[..32].try_into().unwrap(),
                    block[32..].try_into().unwrap(),
                ]
            })
            .collect()
    }

    #[test]
    fn test_hello_world() {
        let input = b"hello world";
        let expected = hex!(
            "
            b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9
        "
        );

        let sha256 = Sha256;
        assert_eq!(sha256.hash_iter(input.to_vec())[..], expected[..]);
    }

    #[test]
    fn test_compress() {
        let left = [0u8; 32];
        // `right` will simulate the SHA256 padding
        let mut right = [0u8; 32];
        right[0] = 1 << 7;
        right[30] = 1; // left has length 256 in bits, L = 0x100

        let expected = Sha256.hash_iter(left);
        let sha256_compress = Sha256Compress;
        assert_eq!(sha256_compress.compress([left, right]), expected);
    }

    #[test]
    fn hash_many_matches_known_vectors() {
        let messages = [
            b"abc".as_slice(),
            b"abd".as_slice(),
            b"abe".as_slice(),
            b"abf".as_slice(),
        ];
        let expected = [
            hex!("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"),
            hex!("a52d159f262b2c6ddb724a61840befc36eb30c88877a4030b65cbe86298449c9"),
            hex!("d81a65c1de02e17d9cfd88d68a8768fd1e3262f5e2fb859382fe33734b3f3ca8"),
            hex!("431b36f2b16be7471a7cce44b22a6d9d4be6faf0a6f4e5f068a6124b951826a9"),
        ];

        let input: Vec<u8> = messages.into_iter().flatten().copied().collect();
        let mut out = [[0u8; 32]; 4];
        Sha256.hash_many(&input, &mut out);

        assert_eq!(out, expected);
    }

    #[test]
    fn hash_many_matches_scalar_across_block_shapes() {
        for len in SHAPE_LENGTHS {
            for count in BATCH_COUNTS {
                let messages = random_bytes(len * count, ((len as u64) << 32) | count as u64 | 1);

                let mut batched = vec![[0u8; 32]; count];
                Sha256.hash_many(&messages, &mut batched);

                assert_eq!(
                    batched,
                    reference(&messages, len, count),
                    "len {len}, count {count}"
                );
            }
        }
    }

    #[test]
    fn hash_many_matches_fips_180_vectors_in_every_lane() {
        // FIPS 180-4 examples: the empty message, one block, one message whose padding spills
        // into a second block, and a two-block message.
        let vectors: [(&[u8], [u8; 32]); 4] = [
            (
                b"",
                hex!("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
            ),
            (
                b"abc",
                hex!("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"),
            ),
            (
                b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
                hex!("248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"),
            ),
            (
                b"abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu",
                hex!("cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1"),
            ),
        ];

        for (message, expected) in vectors {
            // Four copies fill exactly one group, so every lane of the backend is checked.
            let input: Vec<u8> = message.repeat(4);
            let mut out = [[0u8; 32]; 4];
            Sha256.hash_many(&input, &mut out);

            assert_eq!(out, [expected; 4], "message of {} bytes", message.len());
        }
    }

    #[test]
    fn compress_many_matches_specification_at_boundary_blocks() {
        // Blocks whose words hit the carries and the rotations hardest.
        let boundary: [[u8; 64]; 4] = [[0x00; 64], [0xff; 64], [0x80; 64], [0x7f; 64]];
        let inputs = compression_inputs(boundary.as_flattened());

        let mut batched = [[0u8; 32]; 4];
        Sha256Compress.compress_many(&inputs, &mut batched);

        for (input, digest) in inputs.iter().zip(&batched) {
            assert_eq!(*digest, spec_compress_pair(*input));
        }
    }

    #[test]
    fn hash_many_hashes_zero_length_messages() {
        let mut out = [[0u8; 32]; 9];
        Sha256.hash_many(&[], &mut out);

        assert_eq!(out, [Sha256.hash_slice(&[]); 9]);
    }

    #[test]
    fn hash_many_with_empty_output_does_not_validate_input_shape() {
        Sha256.hash_many(&[1, 2, 3], &mut []);
    }

    #[test]
    #[should_panic(expected = "must be a whole multiple")]
    fn hash_many_rejects_ragged_input() {
        let mut out = [[0u8; 32]; 2];
        Sha256.hash_many(&[1, 2, 3, 4, 5], &mut out);
    }

    #[test]
    fn hash_many_keeps_the_lanes_independent() {
        // One full group of four messages, then the same group with only the last one changed.
        let shared = random_bytes(3 * 200, 0x0123_4567_89ab_cdef);
        let mut batch_a = shared.clone();
        batch_a.extend(random_bytes(200, 0xdead_beef_dead_beef));
        let mut batch_b = shared;
        batch_b.extend(random_bytes(200, 0xfeed_face_feed_face));

        let mut out_a = [[0u8; 32]; 4];
        let mut out_b = [[0u8; 32]; 4];
        Sha256.hash_many(&batch_a, &mut out_a);
        Sha256.hash_many(&batch_b, &mut out_b);

        // The untouched messages must agree, and the changed one must not.
        assert_eq!(out_a[..3], out_b[..3]);
        assert_ne!(out_a[3], out_b[3]);
    }

    #[test]
    fn compress_many_matches_scalar_across_batch_counts() {
        for count in BATCH_COUNTS {
            let bytes = random_bytes(count * 64, 0x9e37_79b9_7f4a_7c15 ^ count as u64);
            let inputs = compression_inputs(&bytes);

            let mut batched = vec![[0u8; 32]; count];
            Sha256Compress.compress_many(&inputs, &mut batched);

            for (input, digest) in inputs.iter().zip(&batched) {
                assert_eq!(*digest, Sha256Compress.compress(*input), "count {count}");
            }
        }
    }

    #[test]
    #[should_panic(expected = "group count")]
    fn compress_many_rejects_mismatched_counts() {
        let mut out = [[0u8; 32]; 1];
        Sha256Compress.compress_many(&[], &mut out);
    }

    #[test]
    fn reports_four_lanes_only_on_vectorized_targets() {
        // Every vectorized backend runs four messages at a time.
        //
        // Every other target hashes one.
        let vectorized = cfg!(all(target_arch = "wasm32", target_feature = "simd128"))
            || cfg!(all(
                target_arch = "x86_64",
                target_feature = "sha",
                target_feature = "sse4.1"
            ))
            || cfg!(all(
                target_arch = "aarch64",
                target_feature = "neon",
                target_feature = "sha2"
            ));
        let expected = if vectorized { 4 } else { 1 };

        assert_eq!(
            <Sha256 as CryptographicHasher<u8, [u8; 32]>>::LANES,
            expected
        );
        assert_eq!(
            <Sha256Compress as PseudoCompressionFunction<[u8; 32], 2>>::LANES,
            expected
        );
    }

    proptest! {
        #[test]
        fn hash_many_matches_specification_on_random_batches(
            len in 0usize..=400,
            count in 1usize..=17,
            seed in any::<u64>(),
        ) {
            let messages = random_bytes(len * count, seed | 1);

            let mut batched = vec![[0u8; 32]; count];
            Sha256.hash_many(&messages, &mut batched);

            let expected: Vec<[u8; 32]> = (0..count)
                .map(|k| spec_hash(&messages[k * len..(k + 1) * len]))
                .collect();
            prop_assert_eq!(batched, expected);
        }

        #[test]
        fn hash_many_matches_scalar_on_random_batches(
            len in 0usize..=400,
            count in 1usize..=17,
            seed in any::<u64>(),
        ) {
            let messages = random_bytes(len * count, seed | 1);

            let mut batched = vec![[0u8; 32]; count];
            Sha256.hash_many(&messages, &mut batched);

            prop_assert_eq!(batched, reference(&messages, len, count));
        }

        #[test]
        fn compress_many_matches_specification_on_random_batches(
            count in 0usize..=17,
            seed in any::<u64>(),
        ) {
            let bytes = random_bytes(count * 64, seed | 1);
            let inputs = compression_inputs(&bytes);

            let mut batched = vec![[0u8; 32]; count];
            Sha256Compress.compress_many(&inputs, &mut batched);

            let expected: Vec<[u8; 32]> = inputs.iter().map(|input| spec_compress_pair(*input)).collect();
            prop_assert_eq!(batched, expected);
        }

        #[test]
        fn compress_many_matches_scalar_on_random_batches(
            count in 0usize..=17,
            seed in any::<u64>(),
        ) {
            let bytes = random_bytes(count * 64, seed | 1);
            let inputs = compression_inputs(&bytes);

            let mut batched = vec![[0u8; 32]; count];
            Sha256Compress.compress_many(&inputs, &mut batched);

            let expected: Vec<[u8; 32]> =
                inputs.iter().map(|input| Sha256Compress.compress(*input)).collect();
            prop_assert_eq!(batched, expected);
        }
    }
}
