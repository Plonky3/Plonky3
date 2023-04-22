//! The Keccak-f permutation, and hash functions built from it.

#![no_std]

extern crate alloc;

use alloc::vec::Vec;
use p3_symmetric::hasher::IterHasher;
use p3_symmetric::permutation::{ArrayPermutation, CryptographicPermutation};
use tiny_keccak::{keccakf, Hasher, Keccak};

/// The Keccak-f permutation.
pub struct KeccakF;

impl CryptographicPermutation<[u64; 25]> for KeccakF {
    fn permute(&self, mut input: [u64; 25]) -> [u64; 25] {
        keccakf(&mut input);
        input
    }
}

impl ArrayPermutation<u64, 25> for KeccakF {}

/// The `Keccak` hash functions defined in
/// [Keccak SHA3 submission](https://keccak.team/files/Keccak-submission-3.pdf).
pub struct Keccak256Hash;

impl IterHasher<u8, [u8; 32]> for Keccak256Hash {
    fn hash_iter<I>(&self, input: I) -> [u8; 32]
    where
        I: IntoIterator<Item = u8>,
    {
        let mut hasher = Keccak::v256();
        // TODO: Inefficient...
        let input = input.into_iter().collect::<Vec<_>>();
        hasher.update(&input);

        let mut output = [0u8; 32];
        hasher.finalize(&mut output);
        output
    }
}
