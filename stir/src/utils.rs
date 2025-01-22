use core::iter;

use crate::polynomial::Polynomial;
use itertools::Itertools;
use p3_field::{Field, TwoAdicField};

pub(crate) fn compute_pow(security_level: usize, error: f64) -> f64 {
    0f64.max(security_level as f64 - error)
}

pub(crate) fn fold_polynomial<F: TwoAdicField>(
    polynomial: &Polynomial<F>,
    folding_randomness: F,
    log_folding_factor: usize,
) -> Polynomial<F> {
    let folding_factor = 1 << log_folding_factor;
    let fold_size = (polynomial.degree() + 1).div_ceil(folding_factor);

    let folding_powers = iter::successors(Some(F::ONE), |&x| Some(x * folding_randomness))
        .take(folding_factor)
        .collect_vec();

    let mut folded_coeffs = vec![F::ZERO; fold_size];

    // NP TODO remove or move:
    // Example:
    // 1 + 2*x + 3*x^2 + 4*x^3 + 5*x^4 + 6*x^5 + 7*x^6 + 8*x^7
    // folding_factor = 4
    // fold_size = 2
    // folding_powers = [1, r, r^2, r^3]
    // folded_polynomial = (1 + 5x) + r * (2 + 6x) + r^2 * (3 + 7x) + r^3 * (4 + 8x)
    //                   = (1 + 2r + 3r^2 + 4r^3) + (5 + 6r + 7r^2 + 8r^3)x

    for (i, coeff) in polynomial.coeffs().iter().enumerate() {
        folded_coeffs[i / folding_factor] += *coeff * folding_powers[i % folding_factor];
    }

    Polynomial::from_coeffs(folded_coeffs)
}

pub(crate) fn field_element_from_isize<F: Field>(x: isize) -> F {
    let sign = if x >= 0 { F::ONE } else { -F::ONE };
    let value = F::from_canonical_u32(x.abs() as u32);
    sign * value
}

/// Multiply `polynomial` by `1 + coeff * x + coeff^2 * x^2 + ... + coeff^degee * x^degree`
pub(crate) fn multiply_by_power_polynomial<F: Field>(
    polynomial: &Polynomial<F>,
    coeff: F,
    degree: usize,
) -> Polynomial<F> {
    // NP TODO check this works for degree = 0, or early stop, or panic

    //  Let (c, d) = (coeff, degree). Polynomial needs to be multiplied by
    //  ((c*x)^(d + 1) - 1) / (c*x - 1).

    // We first compute polynomial * ((c*x)^(d + 1) - 1), i. e.:
    //   [0 ... 0] || c^(d + 1) * coeffs
    // - coeffs || [0 ... 0]
    let coeff_pow_n_1 = coeff.exp_u64((degree + 1) as u64);
    let mut new_coeffs = vec![F::ZERO; degree + 1];
    new_coeffs.extend(
        polynomial
            .coeffs()
            .iter()
            .map(|&coeff| coeff * coeff_pow_n_1),
    );
    for (c1, c2) in new_coeffs.iter_mut().zip(polynomial.coeffs().iter()) {
        *c1 -= *c2;
    }

    // Now we divide by c*x - 1 by dividing by x - (1/c) and multiplying by c afterwards
    let mut last = *new_coeffs.iter().last().unwrap();
    let coeff_inv = coeff.inverse();
    for new_c in new_coeffs.iter_mut().rev().skip(1) {
        *new_c += coeff_inv * last;
        last = *new_c;
    }

    assert!(new_coeffs.remove(0) == F::ZERO);

    new_coeffs.iter_mut().for_each(|c| *c *= coeff_inv);
    Polynomial::from_coeffs(new_coeffs)
}

#[cfg(test)]
mod tests {

    use p3_baby_bear::BabyBear;
    use p3_field::FieldAlgebra;

    use core::assert_eq;
    use iter::Iterator;
    use rand::Rng;

    use super::*;
    use crate::polynomial::rand_poly;

    type F = BabyBear;

    #[test]
    fn test_fold_polynomial() {
        let polynomial = Polynomial::<F>::from_coeffs(vec![F::ONE; 16]);
        let folding_randomness = F::from_canonical_u32(3);

        // log_folding_factor = 1
        assert_eq!(
            fold_polynomial(&polynomial, folding_randomness, 1).coeffs(),
            vec![4, 4, 4, 4, 4, 4, 4, 4]
                .into_iter()
                .map(F::from_canonical_u32)
                .collect_vec()
        );

        // log_folding_factor = 2
        assert_eq!(
            fold_polynomial(&polynomial, folding_randomness, 2).coeffs(),
            vec![40, 40, 40, 40]
                .into_iter()
                .map(F::from_canonical_u32)
                .collect_vec()
        );
    }

    #[test]
    fn test_fold_backwards() {
        let fold_degree = 5;
        let log_folding_factor = 4;

        let folding_factor = 1 << log_folding_factor;

        let mut rng = rand::thread_rng();

        let folding_randomness: F = rng.gen();

        let folds = (0..folding_factor)
            .map(|_| rand_poly::<F>(fold_degree - 1))
            .collect_vec();

        let powers_of_x = iter::successors(Some(Polynomial::one()), |p| Some(&Polynomial::x() * p))
            .take(folding_factor)
            .collect_vec();

        let polynomial = folds
            .iter()
            .map(|fold| fold.compose_with_exponent(folding_factor))
            .zip(powers_of_x.iter())
            .fold(Polynomial::zero(), |acc, (raised_fold, power_of_x)| {
                // NP TODO maybe method multiply_by_xn(exp)? or have * detect
                // which situation it is
                &acc + &(&raised_fold * power_of_x)
            });

        let powers_of_r = iter::successors(Some(F::ONE), |&x| Some(x * folding_randomness))
            .take(folding_factor)
            .collect_vec();

        let expected_folded_polynomial = folds
            .iter()
            .zip(powers_of_r.iter())
            .map(|(p, r)| p * r)
            .fold(Polynomial::zero(), |acc, p| &acc + &p);

        assert_eq!(
            fold_polynomial(&polynomial, folding_randomness, log_folding_factor),
            expected_folded_polynomial
        );
    }

    #[test]
    fn test_multiply_by_power_polynomial() {
        let degree_polynomial = 5;
        let degree_power_polynomial = 6;

        let mut rng = rand::thread_rng();
        let coeff: F = rng.gen();
        let polynomial = rand_poly(degree_polynomial);

        let expected = &Polynomial::power_polynomial(coeff, degree_power_polynomial) * &polynomial;

        assert_eq!(
            multiply_by_power_polynomial(&polynomial, coeff, degree_power_polynomial),
            expected
        );
    }
}
