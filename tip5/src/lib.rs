//! The Tip5 permutation, and hash functions built from it.
//! https://eprint.iacr.org/2023/107
//! https://github.com/Neptune-Crypto/twenty-first/blob/master/twenty-first/src/shared_math/tip5.rs

#![no_std]

extern crate alloc;

use alloc::vec::Vec;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::tip5::{DIGEST_LENGTH, Tip5State};
use twenty_first::util_types::algebraic_hasher::{AlgebraicHasher, Domain};
use p3_field::PackedField;
use p3_symmetric::hasher::CryptographicHasher;
use p3_symmetric::permutation::{ArrayPermutation, CryptographicPermutation};

#[derive(Default)]
pub struct Tip5Permutation<const WIDTH: usize>;

impl<const WIDTH: usize> CryptographicPermutation<[u64; WIDTH]> for Tip5Permutation<WIDTH> {
    fn permute(&self, mut input: [u64; WIDTH]) -> [u64; WIDTH] {
        let mut sponge = Tip5State::new(Domain::VariableLength);
        // igor: the following is private. waiting for https://github.com/Neptune-Crypto/twenty-first/pull/137
        // twenty_first::shared_math::tip5::Tip5::permutation(&mut sponge);
        sponge.state.into_iter().map(|x| u64::from(x.clone())).collect::<Vec<u64>>().try_into().unwrap()
    }
}

impl<const WIDTH: usize> CryptographicPermutation<[u8; WIDTH]> for Tip5Permutation<WIDTH> {
    fn permute(&self, input_u8s: [u8; WIDTH]) -> [u8; WIDTH] {
        let mut state_u64s: [u64; WIDTH] = core::array::from_fn(|i| {
            u64::from_le_bytes(input_u8s[i * 8..][..8].try_into().unwrap())
        });

        // twenty_first::shared_math::tip5::Tip5::permutation(&mut sponge);

        core::array::from_fn(|i| {
            let u64_limb = state_u64s[i / 8];
            u64_limb.to_le_bytes()[i % 8]
        })
    }
}

impl<const WIDTH: usize> ArrayPermutation<u64, WIDTH> for Tip5Permutation<WIDTH> {}

pub struct Tip5Hasher<const WIDTH: usize> {}

impl<const WIDTH: usize> CryptographicHasher<u64, [u64; DIGEST_LENGTH]> for Tip5Hasher<WIDTH> {
    fn hash_iter<I>(&self, input: I) -> [u64; DIGEST_LENGTH]
        where I: IntoIterator<Item=u64> {
        let input = input.into_iter().map(|x| BFieldElement::new(x)).collect::<Vec<BFieldElement>>();
        let digest = twenty_first::shared_math::tip5::Tip5::hash_varlen(input.as_slice());
        digest.0.into_iter().map(|x| u64::from(x.clone())).collect::<Vec<u64>>().try_into().unwrap()
    }
}

