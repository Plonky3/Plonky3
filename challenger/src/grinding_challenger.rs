use p3_field::PrimeField64;
use p3_symmetric::CryptographicPermutation;

use crate::{DuplexChallenger, FieldChallenger};

pub trait GrindingChallenger<F: PrimeField64>: FieldChallenger<F> + Clone {
    // Can be overridden to allow more efficient ways to fork, not cloning the entire state.
    fn fork(&self) -> Self {
        self.clone()
    }

    fn grind(&self, bits: usize) -> F {
        for i in 0..F::ORDER_U64 {
            let witness = F::from_canonical_u64(i);
            let mut forked = self.fork();
            forked.observe(witness);
            if forked.check_witness(bits, witness) {
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
