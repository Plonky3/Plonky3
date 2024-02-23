//! The blake3 hash function.

#![no_std]

extern crate alloc;

use alloc::vec::Vec;

use p3_symmetric::CryptographicHasher;

/// The blake3 hash function.
#[derive(Copy, Clone)]
pub struct Blake3;

impl CryptographicHasher<u8, [u8; 32]> for Blake3 {
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
        let mut hasher = blake3::Hasher::new();
        for chunk in input.into_iter() {
            hasher.update(chunk);
        }
        hasher.finalize().into()
    }
}
