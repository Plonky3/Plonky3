//! The SHA2-256 hash function.

#![no_std]

#[cfg(test)]
extern crate alloc;

use p3_symmetric::{CompressionFunction, CryptographicHasher, PseudoCompressionFunction};
use sha2::Digest;

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm32_simd128;

pub const H256_256: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

/// The SHA2-256 hash function.
#[derive(Copy, Clone, Debug)]
pub struct Sha256;

impl CryptographicHasher<u8, [u8; 32]> for Sha256 {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    const LANES: usize = wasm32_simd128::LANES;

    fn hash_iter<I>(&self, input: I) -> [u8; 32]
    where
        I: IntoIterator<Item = u8>,
    {
        const BUFLEN: usize = 512; // Tweakable parameter; determined by experiment
        let mut hasher = sha2::Sha256::new();
        p3_util::apply_to_chunks::<BUFLEN, _, _>(input, |buf| hasher.update(buf));
        hasher.finalize().into()
    }

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

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    fn hash_many(&self, input: &[u8], out: &mut [[u8; 32]]) {
        wasm32_simd128::hash_many(input, out);
    }
}

/// SHA2-256 without the padding (pre-processing), intended to be used
/// as a 2-to-1 [PseudoCompressionFunction].
#[derive(Copy, Clone, Debug)]
pub struct Sha256Compress;

impl PseudoCompressionFunction<[u8; 32], 2> for Sha256Compress {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    const LANES: usize = wasm32_simd128::LANES;

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

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    fn compress_many(&self, inputs: &[[[u8; 32]; 2]], out: &mut [[u8; 32]]) {
        wasm32_simd128::compress_many(inputs, out);
    }
}

impl CompressionFunction<[u8; 32], 2> for Sha256Compress {}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use hex_literal::hex;
    use p3_symmetric::{CryptographicHasher, PseudoCompressionFunction};

    use crate::{Sha256, Sha256Compress};

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
    fn hash_many_matches_scalar_at_padding_boundaries() {
        const LENGTHS: [usize; 9] = [55, 56, 63, 64, 65, 119, 120, 127, 128];
        const COUNTS: [usize; 11] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 17];

        for len in LENGTHS {
            for count in COUNTS {
                let input = random_bytes(len * count, (len as u64) << 32 | count as u64 | 1);
                let mut out = vec![[0u8; 32]; count];
                Sha256.hash_many(&input, &mut out);

                for (message, digest) in input.chunks_exact(len).zip(&out) {
                    assert_eq!(
                        *digest,
                        Sha256.hash_slice(message),
                        "len {len}, count {count}"
                    );
                }
            }
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
    fn compress_many_matches_scalar_for_random_batches() {
        for count in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 17] {
            let bytes = random_bytes(count * 64, 0x1234_5678_9abc_def1 ^ count as u64);
            let inputs: Vec<[[u8; 32]; 2]> = bytes
                .as_chunks::<64>()
                .0
                .iter()
                .map(|block| {
                    [
                        block[..32].try_into().unwrap(),
                        block[32..].try_into().unwrap(),
                    ]
                })
                .collect();
            let mut out = vec![[0u8; 32]; count];
            Sha256Compress.compress_many(&inputs, &mut out);

            for (input, digest) in inputs.iter().zip(&out) {
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
    fn reports_four_lanes_only_for_wasm_simd128() {
        let expected = if cfg!(all(target_arch = "wasm32", target_feature = "simd128")) {
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
}
