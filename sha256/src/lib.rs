//! The SHA2-256 hash function.

#![no_std]

extern crate alloc;

use alloc::vec::Vec;

use p3_symmetric::CryptographicHasher;
use sha2::Digest;

/// The SHA2-256 hash function.
#[derive(Copy, Clone, Debug)]
pub struct Sha256;

impl CryptographicHasher<u8, [u8; 32]> for Sha256 {
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
        let mut hasher = sha2::Sha256::new();
        for chunk in input.into_iter() {
            hasher.update(chunk);
        }
        hasher.finalize().into()
    }
}

#[cfg(test)]
mod tests {
    use hex_literal::hex;
    use p3_symmetric::CryptographicHasher;

    use crate::Sha256;

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
}
