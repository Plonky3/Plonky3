//! The BLAKE2s hash function of RFC 7693, with a batched path for many messages at once.
//!
//! One message runs on the general-purpose registers, four independent G chains at a time.
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
mod params;
mod single;

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
        single::hash(message)
    }
}

impl CryptographicHasher<u8, [u8; DIGEST_BYTES]> for Blake2s256 {
    const LANES: usize = LANES;

    fn hash_iter<I>(&self, input: I) -> [u8; DIGEST_BYTES]
    where
        I: IntoIterator<Item = u8>,
    {
        const BUFLEN: usize = 512; // Tweakable parameter; determined by experiment
        let mut hasher = single::Hasher::new();
        p3_util::apply_to_chunks::<BUFLEN, _, _>(input, |buf| hasher.update(buf));
        hasher.finalize()
    }

    fn hash_iter_slices<'a, I>(&self, input: I) -> [u8; DIGEST_BYTES]
    where
        I: IntoIterator<Item = &'a [u8]>,
    {
        // A message given in one piece skips the streaming state.
        let mut slices = input.into_iter();
        let Some(first) = slices.next() else {
            return single::hash(&[]);
        };
        let Some(second) = slices.next() else {
            return single::hash(first);
        };

        let mut hasher = single::Hasher::new();
        for slice in [first, second].into_iter().chain(slices) {
            hasher.update(slice);
        }
        hasher.finalize()
    }

    fn hash_slice(&self, input: &[u8]) -> [u8; DIGEST_BYTES] {
        single::hash(input)
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
