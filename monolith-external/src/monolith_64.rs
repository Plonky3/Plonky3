//! The Monolith permutation, and hash functions built from it.

use alloc::sync::Arc;
use alloc::vec::Vec;

use p3_field::{AbstractField, PrimeField64};
use p3_goldilocks::Goldilocks;
use p3_symmetric::compression::PseudoCompressionFunction;
use p3_symmetric::permutation::{ArrayPermutation, CryptographicPermutation};
use zkhash::fields::f64::F64;
use zkhash::monolith_64::monolith_64::Monolith64 as ZKHashMonolith64;
use zkhash::monolith_64::monolith_64_params::Monolith64Params;

// The Monolith-64 permutation.
#[derive(Clone)]
pub struct Monolith64;

fn m64_to_f64(x: &Goldilocks) -> F64 {
    F64::from(x.as_canonical_u64())
}

fn f64_to_m64(x: &F64) -> Goldilocks {
    Goldilocks::from_canonical_u64(x.v)
}

impl CryptographicPermutation<[Goldilocks; 16]> for Monolith64 {
    fn permute(&self, input: [Goldilocks; 16]) -> [Goldilocks; 16] {
        let m = ZKHashMonolith64::new(&Arc::new(Monolith64Params::new()));

        let mut input_f: [F64; 16] = input
            .iter()
            .map(m64_to_f64)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        input_f = m.permutation_u128(&input_f);
        input_f
            .iter()
            .map(f64_to_m64)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap()
    }
}

impl ArrayPermutation<Goldilocks, 16> for Monolith64 {}

#[derive(Clone)]
pub struct Monolith64Hash;

impl PseudoCompressionFunction<[Goldilocks; 8], 2> for Monolith64Hash {
    fn compress(&self, input: [[Goldilocks; 8]; 2]) -> [Goldilocks; 8] {
        let m = ZKHashMonolith64::new(&Arc::new(Monolith64Params::new()));

        let input_1 = input[0]
            .iter()
            .map(m64_to_f64)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let input_2 = input[1]
            .iter()
            .map(m64_to_f64)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let output = m.hash(&input_1, &input_2);

        output
            .iter()
            .map(f64_to_m64)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap()
    }
}
