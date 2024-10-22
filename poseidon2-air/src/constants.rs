use alloc::vec::Vec;

use p3_field::Field;
use rand::distributions::{Distribution, Standard};
use rand::Rng;

/// Round constants for Poseidon2, in a format that's convenient for the AIR.
#[derive(Debug)]
pub struct RoundConstants<
    F: Field,
    const WIDTH: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
> {
    pub(crate) beginning_full_round_constants: [[F; WIDTH]; HALF_FULL_ROUNDS],
    pub(crate) partial_round_constants: [F; PARTIAL_ROUNDS],
    pub(crate) ending_full_round_constants: [[F; WIDTH]; HALF_FULL_ROUNDS],
}

impl<F: Field, const WIDTH: usize, const HALF_FULL_ROUNDS: usize, const PARTIAL_ROUNDS: usize>
    RoundConstants<F, WIDTH, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>
{
    pub fn from_rng<R: Rng>(rng: &mut R) -> Self
    where
        Standard: Distribution<F> + Distribution<[F; WIDTH]>,
    {
        let beginning_full_round_constants = rng
            .sample_iter(Standard)
            .take(HALF_FULL_ROUNDS)
            .collect::<Vec<[F; WIDTH]>>()
            .try_into()
            .unwrap();
        let partial_round_constants = rng
            .sample_iter(Standard)
            .take(PARTIAL_ROUNDS)
            .collect::<Vec<F>>()
            .try_into()
            .unwrap();
        let ending_full_round_constants = rng
            .sample_iter(Standard)
            .take(HALF_FULL_ROUNDS)
            .collect::<Vec<[F; WIDTH]>>()
            .try_into()
            .unwrap();
        Self {
            beginning_full_round_constants,
            partial_round_constants,
            ending_full_round_constants,
        }
    }
}
