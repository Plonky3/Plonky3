use alloc::vec::Vec;

use p3_field::Field;
use rand::distr::{Distribution, StandardUniform};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use crate::Polynomial;

// In order to use this auxiliary functionality, activate the `test-utils`
// feature

/// Returns a random polynomial of the exact given degree generated a seeded
/// ChaCha20Rng.
pub fn rand_poly_seeded<F: Field>(degree: usize, seed: Option<u64>) -> Polynomial<F>
where
    StandardUniform: Distribution<F>,
{
    let mut rng = SmallRng::seed_from_u64(seed.unwrap_or(42));

    rand_poly_rng(degree, &mut rng)
}

pub fn rand_poly_rng<F: Field>(degree: usize, rng: &mut impl Rng) -> Polynomial<F>
where
    StandardUniform: Distribution<F>,
{
    let mut coeffs: Vec<F> = (0..degree).map(|_| rng.random()).collect();

    let mut leading_coeff = F::ZERO;

    while leading_coeff == F::ZERO {
        leading_coeff = rng.random();
    }

    coeffs.push(leading_coeff);

    Polynomial::from_coeffs(coeffs)
}
