//! The BLAKE3 hash function, with a batched path for many messages at once.
//!
//! One message at a time, this wraps the `blake3` crate.
//!
//! Many equal-length messages share their chunks, blocks, counters and flags.
//!
//! So the batched path compresses them in lockstep, one vector lane per message.
//!
//! The vector backend is picked at build time: AVX-512, AVX2 or SSE2 on x86-64, NEON on AArch64.
//!
//! Other targets run the same kernel one lane at a time.

#![no_std]

#[cfg(test)]
extern crate alloc;
#[cfg(test)]
mod tests;

mod batch;

use blake3::OUT_LEN;
use p3_symmetric::CryptographicHasher;

/// Messages one batched compression advances at once.
///
/// - 32 with AVX-512, 16 with AVX2 or NEON, 4 with SSE2, and 1 elsewhere.
/// - Any batch size works, and a partial register costs one full register of work.
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

        batch::hash_many(batch::Mode::HASH, input, len, out);
    }
}
