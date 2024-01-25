use p3_field::PrimeField64;
use p3_maybe_rayon::prelude::*;
use p3_symmetric::CryptographicPermutation;
use tracing::instrument;

use crate::{DuplexChallenger, FieldChallenger};

pub trait GrindingChallenger<F: PrimeField64>: FieldChallenger<F> + Clone {
    // Can be overridden for more efficient methods not involving cloning, depending on the
    // internals of the challenger.
    #[instrument(name = "grind for proof-of-work witness", skip_all)]
    fn grind(&mut self, bits: usize) -> F {
        let witness = (0..F::ORDER_U64)
            .into_par_iter()
            .map(|i| F::from_canonical_u64(i))
            .find_any(|witness| self.clone().check_witness(bits, *witness))
            .expect("failed to find witness");
        assert!(self.check_witness(bits, witness));
        witness
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
