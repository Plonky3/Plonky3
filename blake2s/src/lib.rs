//! The BLAKE2s hash function of RFC 7693, with a batched path for many messages at once.
//!
//! One message at a time, this wraps RustCrypto's `blake2`.
//!
//! Many equal-length messages share their block count, byte counter and final-block flag.
//!
//! So the batched path compresses them in lockstep, one vector lane per message.
//!
//! The vector backend is picked at build time: AVX-512, AVX2 or SSE2 on x86-64, NEON on AArch64.
//!
//! Other targets hash a batch one message at a time.

#![no_std]

#[cfg(test)]
extern crate alloc;
#[cfg(test)]
mod tests;

#[cfg_attr(
    not(any(
        target_arch = "x86_64",
        all(
            target_arch = "aarch64",
            target_feature = "neon",
            target_endian = "little"
        )
    )),
    path = "scalar.rs"
)]
mod batch;

use blake2::digest::consts::U32;
use blake2::{Blake2s, Digest};
use p3_symmetric::CryptographicHasher;

/// Messages one batched compression advances at once.
///
/// - 32 with AVX-512, 16 with AVX2, 8 with SSE2 or NEON, and 1 elsewhere.
/// - Any batch size works, and a partial group costs one full group of work.
pub const LANES: usize = batch::LANES;

/// Bytes in a BLAKE2s-256 digest.
pub const DIGEST_BYTES: usize = 32;

/// The BLAKE2s hash function at a 256-bit digest, unkeyed and sequential.
///
/// That is the configuration lean Ethereum's signatures use.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct Blake2s256;

impl Blake2s256 {
    /// Hash one contiguous message.
    #[inline]
    pub fn hash(message: &[u8]) -> [u8; DIGEST_BYTES] {
        Blake2s::<U32>::digest(message).into()
    }
}

impl CryptographicHasher<u8, [u8; DIGEST_BYTES]> for Blake2s256 {
    const LANES: usize = LANES;

    fn hash_iter<I>(&self, input: I) -> [u8; DIGEST_BYTES]
    where
        I: IntoIterator<Item = u8>,
    {
        const BUFLEN: usize = 512; // Tweakable parameter; determined by experiment
        let mut hasher = Blake2s::<U32>::new();
        p3_util::apply_to_chunks::<BUFLEN, _, _>(input, |buf| hasher.update(buf));
        hasher.finalize().into()
    }

    fn hash_iter_slices<'a, I>(&self, input: I) -> [u8; DIGEST_BYTES]
    where
        I: IntoIterator<Item = &'a [u8]>,
    {
        let mut hasher = Blake2s::<U32>::new();
        for slice in input {
            hasher.update(slice);
        }
        hasher.finalize().into()
    }

    /// Hash equal-length messages laid end to end, several of them per compression.
    ///
    /// # Panics
    ///
    /// Panics if the input length is not a whole multiple of the digest count.
    fn hash_many(&self, input: &[u8], out: &mut [[u8; DIGEST_BYTES]]) {
        batch::hash_many(input, out);
    }
}
