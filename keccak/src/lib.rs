//! The Keccak-f permutation, and hash functions built from it.

#![no_std]

extern crate alloc;

use alloc::vec::Vec;

use p3_symmetric::hasher::CryptographicHasher;
use p3_symmetric::permutation::CryptographicPermutation;
use tiny_keccak::{keccakf, Hasher, Keccak};

/// The Keccak-f permutation.
#[derive(Copy, Clone)]
pub struct KeccakF;

impl CryptographicPermutation<[u64; 25]> for KeccakF {
    fn permute(&self, mut input: [u64; 25]) -> [u64; 25] {
        keccakf(&mut input);
        input
    }
}

impl CryptographicPermutation<[u8; 200]> for KeccakF {
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

/// The `Keccak` hash functions defined in
/// [Keccak SHA3 submission](https://keccak.team/files/Keccak-submission-3.pdf).
#[derive(Copy, Clone)]
pub struct Keccak256Hash;

impl CryptographicHasher<u8, [u8; 32]> for Keccak256Hash {
    fn hash_iter<I>(&self, input: I) -> [u8; 32]
    where
        I: IntoIterator<Item = u8>,
    {
        let input = input.into_iter().collect::<Vec<_>>();
        self.hash_iter_slices([input.as_slice()])
    }

    fn hash_iter_slices<'a, I>(&self, input: I) -> [u8; 32]
    where
        I: IntoIterator<Item = &'a [u8]>,
    {
        let mut hasher = Keccak::v256();
        for chunk in input.into_iter() {
            hasher.update(chunk);
        }

        let mut output = [0u8; 32];
        hasher.finalize(&mut output);
        output
    }
}
