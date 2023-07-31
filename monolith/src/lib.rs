//! The Monolith permutation, and hash functions built from it.

#![no_std]

extern crate alloc;

use core::u64;

use alloc::sync::Arc;
use alloc::vec::Vec;

use p3_symmetric::compression::PseudoCompressionFunction;
use p3_symmetric::permutation::{ArrayPermutation, CryptographicPermutation};
use zkhash::fields::f31::F31;
use zkhash::monolith_31::monolith_31::Monolith31 as ZKHashMonolith31;
use zkhash::monolith_31::monolith_31_params::Monolith31Params;

// The Monolith-31 permutation.
pub struct Monolith31;

impl CryptographicPermutation<[u64; 25]> for Monolith31 {
    fn permute(&self, mut input: [u64; 25]) -> [u64; 25] {
        let m = ZKHashMonolith31::new(&Arc::new(Monolith31Params::new()));

        let mut input_f: [F31; 25] = input.iter().map(|x| F31::from(*x)).collect::<Vec<_>>().try_into().unwrap();
        m.permutation_u64(&mut input_f);
        input_f.iter().map(|x| x.v as u64).collect::<Vec<_>>().try_into().unwrap()
    }
}

impl ArrayPermutation<u64, 25> for Monolith31 {}

pub struct Monolith31Hash;

impl PseudoCompressionFunction<[u64; 8], 2> for Monolith31Hash {
    fn compress(&self, input: [[u64; 8]; 2]) -> [u64; 8] {
        let m = ZKHashMonolith31::new(&Arc::new(Monolith31Params::new()));

        let input_1 = input[0].iter().map(|x| F31::from(*x)).collect::<Vec<_>>().try_into().unwrap();
        let input_2 = input[1].iter().map(|x| F31::from(*x)).collect::<Vec<_>>().try_into().unwrap();
        let output = m.hash(&input_1, &input_2);

        output.iter().map(|x| x.v as u64).collect::<Vec<_>>().try_into().unwrap()
    }
}