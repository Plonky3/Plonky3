use p3_field::PrimeField64;
use p3_symmetric::CryptographicPermutation;

use crate::{DuplexChallenger, FieldChallenger};

pub trait GrindingChallenger<F: PrimeField64>: FieldChallenger<F> + Clone {
    // Can be overridden for more efficient methods not involving cloning, depending on the
    // internals of the challenger.
    fn grind(&mut self, bits: usize) -> F {
        for i in 0..F::ORDER_U64 {
            let witness = F::from_canonical_u64(i);
            let mut forked = self.clone();

            if forked.check_witness(bits, witness) {
                self.observe(witness);
                self.sample_bits(bits);
                return witness;
            }
        }

        panic!("failed to find witness")
    }

    #[must_use]
    fn check_witness(&mut self, bits: usize, witness: F) -> bool {
        self.observe(witness);
        self.sample_bits(bits) == 0
    }
}

impl<F, P, const WIDTH: usize> GrindingChallenger<F> for DuplexChallenger<F, P, WIDTH>
where
    F: PrimeField64,
    P: CryptographicPermutation<[F; WIDTH]>,
{
}
