//! The SHA2-256 hash function.
//!
//! Batched hashing and batched compression run many messages at once where the build enables a batched backend.
//!
//! The backend is picked at compile time, the first match winning:
//!
//! - x86-64 with `avx512f` and `avx512bw`: 32 messages, one per 32-bit lane, and SHA-NI streams for the last few when `sha` is on too;
//! - x86-64 with `sha` and `sse4.1`: four messages, as four interleaved SHA-NI streams;
//! - AArch64 with `neon` and `sha2`: four messages, as four streams of the SHA-2 extension;
//! - wasm32 with `simd128`: four messages, one per lane.
//!
//! No x86-64 microarchitecture level enables `sha`, so SHA-NI needs `-C target-feature=+sha` or `-C target-cpu=native`.
//! AArch64 Linux likewise needs `+sha2`, which the Apple silicon targets enable by default.
//!
//! Any other build hashes one message at a time through `sha2`.
//! That crate detects SHA-NI or the ARMv8 SHA-2 extension at runtime on its own.

#![no_std]

#[cfg(test)]
extern crate alloc;

use p3_symmetric::{CompressionFunction, CryptographicHasher, PseudoCompressionFunction};
use sha2::Digest;

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw"
))]
mod x86_64_avx512;

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
    target_feature = "sse4.1",
    not(all(target_feature = "avx512f", target_feature = "avx512bw"))
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
        ),
        all(
            target_arch = "x86_64",
            target_feature = "avx512f",
            target_feature = "avx512bw"
        )
    ))]
    const LANES: usize = many::LANES;

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
        ),
        all(
            target_arch = "x86_64",
            target_feature = "avx512f",
            target_feature = "avx512bw"
        )
    ))]
    fn hash_many(&self, input: &[u8], out: &mut [[u8; 32]]) {
        many::hash_many(input, out);
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
        ),
        all(
            target_arch = "x86_64",
            target_feature = "avx512f",
            target_feature = "avx512bw"
        )
    ))]
    const LANES: usize = many::LANES;

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
        ),
        all(
            target_arch = "x86_64",
            target_feature = "avx512f",
            target_feature = "avx512bw"
        )
    ))]
    fn compress_many(&self, inputs: &[[[u8; 32]; 2]], out: &mut [[u8; 32]]) {
        many::compress_many(inputs, out);
    }
}

impl CompressionFunction<[u8; 32], 2> for Sha256Compress {}

/// The batched path of an AVX-512 build.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw"
))]
use x86_64_avx512 as many;

/// The batched path of a four-lane build, through the backend it compiles.
#[cfg(any(
    all(target_arch = "wasm32", target_feature = "simd128"),
    all(
        target_arch = "x86_64",
        target_feature = "sha",
        target_feature = "sse4.1",
        not(all(target_feature = "avx512f", target_feature = "avx512bw"))
    ),
    all(
        target_arch = "aarch64",
        target_feature = "neon",
        target_feature = "sha2"
    )
))]
mod many {
    pub(crate) use crate::four_lane::LANES;

    /// Hash equal-length messages laid end to end, four at a time.
    pub(crate) fn hash_many(input: &[u8], out: &mut [[u8; 32]]) {
        crate::four_lane::hash_many::<crate::Backend>(input, out);
    }

    /// Compress each 64-byte pair from the initial hash value, four at a time.
    pub(crate) fn compress_many(inputs: &[[[u8; 32]; 2]], out: &mut [[u8; 32]]) {
        crate::four_lane::compress_many::<crate::Backend>(inputs, out);
    }
}

/// Padding and batching shared by the four-lane backends.
///
/// A backend supplies only the vector core, through [`FourLane`](four_lane::FourLane). Everything
/// that turns a batch of messages into blocks lives here, so a padding fix cannot reach one
/// backend and miss the others.
///
/// An AVX-512 build with SHA-NI also compiles it, for the last few messages of a batch.
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

        /// Leftover messages, after the whole groups, that one call with spare lanes still beats.
        ///
        /// Fewer leftovers are hashed one at a time.
        ///
        /// The default of four never pads, which suits a backend whose streams cost as much as separate calls.
        const PADDED_FROM: usize = LANES;
    }

    /// Hash four messages of a common length, padding included.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if a message is not `len` bytes long.
    #[inline(always)]
    fn hash_four<B: FourLane>(lane_messages: [&[u8]; LANES], len: usize) -> [[u8; 32]; LANES] {
        debug_assert!(lane_messages.iter().all(|message| message.len() == len));
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
    /// Whole groups of four go through the backend.
    ///
    /// The few messages left over go through it too when the backend asks for padding, and one at a time otherwise.
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
        let message = |index: usize| &input[index * len..(index + 1) * len];

        let (groups, rest) = out.as_chunks_mut::<LANES>();
        for (group, digests) in groups.iter_mut().enumerate() {
            *digests = hash_four::<B>(
                core::array::from_fn(|lane| message(group * LANES + lane)),
                len,
            );
        }

        // The spare lanes repeat the last message, and their digests are never written out.
        let first = groups.len() * LANES;
        if rest.len() >= B::PADDED_FROM {
            let last = rest.len() - 1;
            let digests = hash_four::<B>(
                core::array::from_fn(|lane| message(first + lane.min(last))),
                len,
            );
            rest.copy_from_slice(&digests[..rest.len()]);
        } else {
            for (index, digest) in (first..).zip(rest) {
                *digest = Sha256.hash_slice(message(index));
            }
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

        // SAFETY: `[[u8; 32]; 2]` and `[u8; 64]` have identical contiguous byte layouts.
        let block = |index: usize| -> &[u8; BLOCK_BYTES] { unsafe { transmute(&inputs[index]) } };

        let (groups, rest) = out.as_chunks_mut::<LANES>();
        for (group, digests) in groups.iter_mut().enumerate() {
            compress_four::<B>(
                core::array::from_fn(|lane| block(group * LANES + lane)),
                digests,
            );
        }

        // The spare lanes repeat the last block, and their digests are never written out.
        let first = groups.len() * LANES;
        if rest.len() >= B::PADDED_FROM {
            let last = rest.len() - 1;
            let mut digests = [[0u8; 32]; LANES];
            compress_four::<B>(
                core::array::from_fn(|lane| block(first + lane.min(last))),
                &mut digests,
            );
            rest.copy_from_slice(&digests[..rest.len()]);
        } else {
            for (index, digest) in (first..).zip(rest) {
                *digest = Sha256Compress.compress(inputs[index]);
            }
        }
    }

    /// Compress one block per lane from the initial hash value.
    #[inline(always)]
    fn compress_four<B: FourLane>(
        blocks: [&[u8; BLOCK_BYTES]; LANES],
        out: &mut [[u8; 32]; LANES],
    ) {
        let mut state = B::initial_state();
        B::compress(&mut state, blocks);
        B::write_digests(&state, out);
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

    // Message lengths that change the shape of the padding.
    //
    // - 1 to 4: the 0x80 marker lands on each byte of its word in turn.
    // - 55 fits the marker and the length in one block, and 56 does not.
    // - 0, 64 and 128 end with a block that carries no message byte at all.
    const SHAPE_LENGTHS: [usize; 18] = [
        0, 1, 2, 3, 4, 5, 31, 32, 55, 56, 57, 63, 64, 65, 119, 120, 128, 200,
    ];

    // Batch sizes around every group size of every backend.
    //
    // - 4 is the four-lane backends' group, and 3 or 7 leave a padded group of three.
    // - 16 is one AVX-512 register, and 32 two of them.
    // - 8, 9, 20 and 21 sit on each side of the points where AVX-512 changes pass.
    // - 52 and 53 leave 20 and 21 after a whole group of 32.
    const BATCH_COUNTS: [usize; 23] = [
        0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 20, 21, 31, 32, 33, 48, 52, 53, 64, 65, 100,
    ];

    // Enough messages to fill every lane of the compiled backend at least once.
    const EVERY_LANE: usize = {
        let lanes = <Sha256 as CryptographicHasher<u8, [u8; 32]>>::LANES;
        if lanes > 4 { lanes } else { 4 }
    };

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
    pub(crate) fn spec_compress(state: &mut [u32; 8], block: &[u8; 64]) {
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
    fn hash_many_matches_fips_180_2_examples_in_every_lane() {
        // FIPS 180-2 appendix B, plus two shapes the appendix leaves out.
        //
        // - B.1: one block.
        // - B.2: 56 bytes, so the padding spills into a second block of padding only.
        // - B.3: one million "a", 15,625 blocks.
        // - The empty message: one block of padding only.
        // - The 112-byte message of appendix C, two blocks with a partial last one.
        let million_a = vec![b'a'; 1_000_000];
        let vectors: [(&[u8], [u8; 32]); 5] = [
            (
                b"abc",
                hex!("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"),
            ),
            (
                b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
                hex!("248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"),
            ),
            (
                &million_a,
                hex!("cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"),
            ),
            (
                b"",
                hex!("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
            ),
            (
                b"abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu",
                hex!("cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1"),
            ),
        ];

        for (message, expected) in vectors {
            // One copy per lane fills a whole group, so every lane of the backend is checked.
            //
            // The other counts leave a batch end for each shorter pass of the AVX-512 backend.
            // The million-byte message keeps to one group, as it alone is a million bytes per lane.
            let counts: &[usize] = if message.len() > 1000 {
                &[EVERY_LANE]
            } else {
                &[EVERY_LANE, 3, 9, 20, 40]
            };
            for &count in counts {
                let input = message.repeat(count);
                let mut out = vec![[0u8; 32]; count];
                Sha256.hash_many(&input, &mut out);

                assert!(
                    out.iter().all(|digest| *digest == expected),
                    "message of {} bytes, {count} copies",
                    message.len()
                );
            }
        }
    }

    #[test]
    fn compress_many_matches_the_fips_180_2_intermediate_hash() {
        // FIPS 180-2 appendix B.2: the first padded block of the 56-byte message.
        //
        //     56 message bytes || 0x80 || 7 zero bytes
        let mut block = [0u8; 64];
        block[..56].copy_from_slice(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq");
        block[56] = 0x80;

        // Compressing it from the initial hash value gives the appendix's H(1).
        let h1 = hex!("85e655d6417a17953363376a624cde5c76e09589cac5f811cc4b32c1f20e533a");

        // One copy per lane, through the batch and through the one-block path.
        let inputs = compression_inputs(&block.repeat(EVERY_LANE));
        let mut out = vec![[0u8; 32]; EVERY_LANE];
        Sha256Compress.compress_many(&inputs, &mut out);

        assert!(out.iter().all(|digest| *digest == h1));
        assert_eq!(Sha256Compress.compress(inputs[0]), h1);
    }

    #[test]
    fn compress_many_matches_specification_at_boundary_blocks() {
        // Blocks whose words hit the carries and the rotations hardest.
        let boundary: [[u8; 64]; 4] = [[0x00; 64], [0xff; 64], [0x80; 64], [0x7f; 64]];

        // Cycle through them until every lane holds one.
        let inputs: Vec<_> = compression_inputs(boundary.as_flattened())
            .into_iter()
            .cycle()
            .take(EVERY_LANE)
            .collect();

        let mut batched = vec![[0u8; 32]; EVERY_LANE];
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
        // One full group of messages, then the same group with only the last one changed.
        let last = EVERY_LANE - 1;
        let shared = random_bytes(last * 200, 0x0123_4567_89ab_cdef);
        let mut batch_a = shared.clone();
        batch_a.extend(random_bytes(200, 0xdead_beef_dead_beef));
        let mut batch_b = shared;
        batch_b.extend(random_bytes(200, 0xfeed_face_feed_face));

        let mut out_a = vec![[0u8; 32]; EVERY_LANE];
        let mut out_b = vec![[0u8; 32]; EVERY_LANE];
        Sha256.hash_many(&batch_a, &mut out_a);
        Sha256.hash_many(&batch_b, &mut out_b);

        // The untouched messages must agree, and the changed one must not.
        assert_eq!(out_a[..last], out_b[..last]);
        assert_ne!(out_a[last], out_b[last]);
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
    fn reports_the_lane_count_of_the_compiled_backend() {
        // AVX-512 runs 32 messages at a time, the other batched backends four.
        //
        // Every other target hashes one.
        let avx512 = cfg!(all(
            target_arch = "x86_64",
            target_feature = "avx512f",
            target_feature = "avx512bw"
        ));
        let four_lanes = cfg!(all(target_arch = "wasm32", target_feature = "simd128"))
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
        let expected = if avx512 {
            32
        } else if four_lanes {
            4
        } else {
            1
        };

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
            count in 1usize..=100,
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
            count in 1usize..=100,
            seed in any::<u64>(),
        ) {
            let messages = random_bytes(len * count, seed | 1);

            let mut batched = vec![[0u8; 32]; count];
            Sha256.hash_many(&messages, &mut batched);

            prop_assert_eq!(batched, reference(&messages, len, count));
        }

        #[test]
        fn compress_many_matches_specification_on_random_batches(
            count in 0usize..=100,
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
            count in 0usize..=100,
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
