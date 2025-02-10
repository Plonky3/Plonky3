use alloc::vec::Vec;
use p3_field::Field;
use rand::{distributions::Standard, prelude::Distribution, Rng};

use crate::Polynomial;

// In order to use this auxiliary functionality, activate the `test-utils`
// feature

/// Returns a random polynomial of the exact given degree.
pub fn rand_poly<F: Field>(degree: usize) -> Polynomial<F>
where
    Standard: Distribution<F>,
{
    let mut rng = rand::thread_rng();

    let mut coeffs: Vec<F> = (0..degree).map(|_| rng.gen()).collect();

    let mut leading_coeff = F::ZERO;

    while leading_coeff == F::ZERO {
        leading_coeff = rng.gen();
    }

    coeffs.push(leading_coeff);

    Polynomial::from_coeffs(coeffs)
}
