use alloc::vec::Vec;

use p3_field::Field;
use rand::distr::{Distribution, StandardUniform};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

use crate::Polynomial;

// In order to use this auxiliary functionality, activate the `test-utils`
// feature

/// Returns a random polynomial of the exact given degree.
pub fn rand_poly<F: Field>(degree: usize) -> Polynomial<F>
where
    StandardUniform: Distribution<F>,
{
    let mut rng = rand::rng();

    rand_poly_rng(degree, &mut rng)
}

/// Returns a random polynomial of the exact given degree generated a seeded
/// ChaCha20Rng.
pub fn rand_poly_seeded<F: Field>(degree: usize, seed: u64) -> Polynomial<F>
where
    StandardUniform: Distribution<F>,
{
    let mut rng = ChaCha20Rng::seed_from_u64(seed);

    rand_poly_rng(degree, &mut rng)
}

fn rand_poly_rng<F: Field>(degree: usize, rng: &mut impl Rng) -> Polynomial<F>
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
