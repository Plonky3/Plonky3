//! The Monolith permutation, and hash functions built from it.

#![no_std]

extern crate alloc;

use core::u64;

use alloc::sync::Arc;
use alloc::vec::Vec;

use p3_field::{AbstractField, PrimeField32};
use p3_mersenne_31::Mersenne31;
use p3_symmetric::compression::PseudoCompressionFunction;
use p3_symmetric::permutation::{ArrayPermutation, CryptographicPermutation};
use zkhash::fields::f31::F31;
use zkhash::monolith_31::monolith_31::Monolith31 as ZKHashMonolith31;
use zkhash::monolith_31::monolith_31_params::Monolith31Params;

// The Monolith-31 permutation.
pub struct Monolith31;

fn m31_to_f31(x: &Mersenne31) -> F31 {
    F31::from(x.as_canonical_u32() as u64)
}

fn f31_to_m31(x: &F31) -> Mersenne31 {
    Mersenne31::from_canonical_u32(x.v)
}

impl CryptographicPermutation<[Mersenne31; 16]> for Monolith31 {
    fn permute(&self, input: [Mersenne31; 16]) -> [Mersenne31; 16] {
        let m = ZKHashMonolith31::new(&Arc::new(Monolith31Params::new()));

        let mut input_f: [F31; 16] = input.iter().map(m31_to_f31).collect::<Vec<_>>().try_into().unwrap();
        m.permutation_u64(&mut input_f);
        input_f.iter().map(f31_to_m31).collect::<Vec<_>>().try_into().unwrap()
    }
}

impl ArrayPermutation<Mersenne31, 16> for Monolith31 {}

pub struct Monolith31Hash;

impl PseudoCompressionFunction<[Mersenne31; 8], 2> for Monolith31Hash {
    fn compress(&self, input: [[Mersenne31; 8]; 2]) -> [Mersenne31; 8] {
        let m = ZKHashMonolith31::new(&Arc::new(Monolith31Params::new()));

        let input_1 = input[0].iter().map(m31_to_f31).collect::<Vec<_>>().try_into().unwrap();
        let input_2 = input[1].iter().map(m31_to_f31).collect::<Vec<_>>().try_into().unwrap();
        let output = m.hash(&input_1, &input_2);

        output.iter().map(f31_to_m31).collect::<Vec<_>>().try_into().unwrap()
    }
}
