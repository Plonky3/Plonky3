//! The BLAKE2s hash function, as specified by RFC 7693.
//!
//! One message at a time this wraps RustCrypto's `blake2`, which is maintained and already
//! vectorized.
//!
//! What it adds is the batched path that crate has no entry point for. Hashing many
//! equal-length messages is the shape a Merkle tree asks for, and those messages share
//! everything a compression depends on except the bytes: the same block count, the same
//! counter, the same final-block flag. So their compressions run in lockstep, one lane per
//! message, over arrays the compiler fills with whatever vector unit the target has.
//!
//! On an M2 that batched path runs about 1.4x the throughput of hashing the same messages
//! one at a time through `blake2`.

#![no_std]

#[cfg(test)]
extern crate alloc;
#[cfg(test)]
mod tests;

mod batch;
mod compress;

pub use batch::LANES;
use blake2::digest::consts::U32;
use blake2::{Blake2s, Digest};
use p3_symmetric::CryptographicHasher;

/// Bytes in a BLAKE2s-256 digest.
pub const DIGEST_BYTES: usize = 32;

/// The BLAKE2s hash function at a 256-bit digest, unkeyed and sequential.
///
/// That is the configuration lean Ethereum's signatures use: no key, no tree mode, and the
/// full 32-byte digest.
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
    /// Messages one batched compression advances at once.
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

    /// Hash a batch of equal-length messages, several of them per compression.
    ///
    /// # Panics
    ///
    /// Panics if the batch is ragged: the input length must be a whole multiple of the
    /// digest count.
    fn hash_many(&self, input: &[u8], out: &mut [[u8; DIGEST_BYTES]]) {
        batch::hash_many(input, out);
    }
}
