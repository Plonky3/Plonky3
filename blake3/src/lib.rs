//! The BLAKE3 hash function, with a batched path for many messages at once.
//!
//! One message at a time, this wraps the `blake3` crate.
//!
//! Many equal-length messages share their chunks, blocks, counters and flags.
//!
//! So the batched path compresses them in lockstep, one vector lane per message.
//!
//! Fewer messages than lanes fill the spare lanes with their own chunks instead.
//!
//! On x86-64 the batched path picks AVX-512, AVX2 or SSE2 at run time, the widest the CPU has.
//!
//! Other targets pick at build time: NEON on AArch64, SIMD128 on wasm32, and one lane elsewhere.

#![no_std]

#[cfg(test)]
extern crate alloc;
#[cfg(test)]
mod tests;

mod batch;

use blake3::{CHUNK_LEN, OUT_LEN};
use p3_symmetric::CryptographicHasher;

/// Messages the widest compiled backend advances in one batched compression.
///
/// - 32 on x86-64, which is AVX-512 and a multiple of the 16 of AVX2 and the 4 of SSE2.
/// - 16 with NEON, 4 with SIMD128, and 1 elsewhere.
/// - Any batch size works.
/// - Short of a full group, messages longer than a chunk spread their chunks across the lanes.
pub const LANES: usize = batch::LANES;

/// The BLAKE3 hash function, with a 256-bit digest.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct Blake3;

impl CryptographicHasher<u8, [u8; OUT_LEN]> for Blake3 {
    const LANES: usize = LANES;

    fn hash_iter<I>(&self, input: I) -> [u8; OUT_LEN]
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

    fn hash_iter_slices<'a, I>(&self, input: I) -> [u8; OUT_LEN]
    where
        I: IntoIterator<Item = &'a [u8]>,
    {
        let mut hasher = blake3::Hasher::new();
        for chunk in input {
            hasher.update(chunk);
        }
        hasher.finalize().into()
    }

    fn hash_slice(&self, input: &[u8]) -> [u8; OUT_LEN] {
        blake3::hash(input).into()
    }

    /// Hash equal-length messages laid end to end, several of them per compression.
    ///
    /// # Panics
    ///
    /// Panics if the input length is not a whole multiple of the digest count.
    fn hash_many(&self, input: &[u8], out: &mut [[u8; OUT_LEN]]) {
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

        // A lone message takes one lane per chunk at best.
        //
        // With no more chunks than one register has lanes, most lanes would idle.
        //
        // The single-message path then picks a vector as narrow as those chunks.
        let kernel = batch::detect();
        if out.len() == 1 && len.div_ceil(CHUNK_LEN) <= kernel.width {
            out[0] = blake3::hash(input).into();
            return;
        }
        kernel.hash_many(batch::Mode::HASH, input, len, out);
    }
}
