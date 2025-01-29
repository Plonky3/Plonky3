use core::iter;

use crate::polynomial::Polynomial;
use itertools::{iterate, izip, Itertools};
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

pub fn fold_evaluations<F: TwoAdicField>(
    evals: Vec<F>,
    point_root: F,
    log_arity: usize,
    // Generator of the coset whose evaluations we are receiving
    omega: F,
    c: F,
) -> F {
    // Let Fold2(g, b) denote the binary (i. e. arity = one) folding of g with
    // coefficient b. Then
    //   Fold(h, 2^arity, coeff) = Fold2(..., Fold2(h, c), c^2, c^4, ..., )
    // where the ellipses denote the arity-fold composition of Fold2.

    // NP TODO remove or change to dbg assert

    let arity = 1 << log_arity;
    assert!(evals.len() == arity);

    // We first construct the list "gammas" of values y_j * c, where y_j runs over half the
    // y_j in Y (i. e. the list of points whose arity-th power is
    // point_root^(arity)). For each {y_j, -y_j} in Y, we only store one of
    // the two. If w is a generator of the unique subgroup of units(F) of order
    // = arity, the list "gammas" has the form
    // {
    //   c * point_root^(-1),
    //   c * (point_root * w)^(-1),
    //   ...,
    //   c * (point_root * w^(arity/2 - 1))^(-1)
    // }

    let inv_point_root = point_root.inverse();
    let inv_omega = omega.inverse();

    let mut gammas = iterate(inv_point_root * c, |&x| x * inv_omega)
        .take(arity / 2)
        .collect_vec();

    let mut result = evals;

    while result.len() > 1 {
        result = fold_evaluations_binary(result, &gammas);
        gammas = gammas[..(gammas.len() / 2)]
            .iter()
            .map(|&gamma| gamma.square())
            .collect_vec();
    }

    result.pop().unwrap()
}

fn fold_evaluations_binary<F: TwoAdicField>(
    evals: Vec<F>,
    // NP TODO think if gammas can be an iter
    gammas: &[F],
) -> Vec<F> {
    let cutoff = evals.len() / 2;
    let low_evals = evals[..cutoff].iter();
    let high_evals = evals[cutoff..].iter();

    izip!(low_evals, high_evals, gammas.iter())
        .map(|(&eval, &eval_inv, &gamma)| fold_evaluation_pair(eval, eval_inv, gamma))
        .collect_vec()
}

fn fold_evaluation_pair<F: TwoAdicField>(eval: F, eval_inv: F, gamma: F) -> F {
    // TODO two inverse
    let inv_two = F::TWO.inverse();
    inv_two * ((F::ONE + gamma) * eval + (F::ONE - gamma) * eval_inv)
}

#[cfg(test)]
mod tests {

    use p3_baby_bear::BabyBear;
    use p3_field::FieldAlgebra;
    use p3_matrix::Matrix;

    use core::assert_eq;
    use iter::Iterator;
    use rand::Rng;

    use super::*;
    use crate::{coset::Radix2Coset, polynomial::rand_poly};

    type BB = BabyBear;

    #[test]
    fn test_fold_polynomial() {
        let polynomial = Polynomial::<BB>::from_coeffs(vec![BB::ONE; 16]);
        let folding_randomness = BB::from_canonical_u32(3);

        // log_folding_factor = 1
        assert_eq!(
            fold_polynomial(&polynomial, folding_randomness, 1).coeffs(),
            vec![4, 4, 4, 4, 4, 4, 4, 4]
                .into_iter()
                .map(BB::from_canonical_u32)
                .collect_vec()
        );

        // log_folding_factor = 2
        assert_eq!(
            fold_polynomial(&polynomial, folding_randomness, 2).coeffs(),
            vec![40, 40, 40, 40]
                .into_iter()
                .map(BB::from_canonical_u32)
                .collect_vec()
        );
    }

    #[test]
    fn test_fold_backwards() {
        let fold_degree = 5;
        let log_folding_factor = 4;

        let folding_factor = 1 << log_folding_factor;

        let mut rng = rand::thread_rng();

        let folding_randomness: BB = rng.gen();

        let folds = (0..folding_factor)
            .map(|_| rand_poly::<BB>(fold_degree - 1))
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

        let powers_of_r = iter::successors(Some(BB::ONE), |&x| Some(x * folding_randomness))
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
        let coeff: BB = rng.gen();
        let polynomial = rand_poly(degree_polynomial);

        let expected = &Polynomial::power_polynomial(coeff, degree_power_polynomial) * &polynomial;

        assert_eq!(
            multiply_by_power_polynomial(&polynomial, coeff, degree_power_polynomial),
            expected
        );
    }

    macro_rules! test_fold_evals_with_log_arity {
        ($log_arity:expr, $polynomial:expr, $folding_randomness:expr) => {{
            let mut rng = rand::thread_rng();
            let domain = Radix2Coset::new(rng.gen(), $log_arity);
            let evaluations = domain
                .iter()
                .map(|x| $polynomial.evaluate(&x))
                .collect_vec();
            let folded_evaluation = fold_evaluations(
                evaluations,
                domain.shift(),
                $log_arity,
                domain.generator(),
                $folding_randomness,
            );
            let folded_polynomial = fold_polynomial(&$polynomial, $folding_randomness, $log_arity);
            assert_eq!(
                folded_evaluation,
                folded_polynomial.evaluate(&domain.shift().exp_power_of_2($log_arity)),
                "log_arity = {}",
                $log_arity
            );
        }};
    }

    #[test]
    fn test_fold_evaluations() {
        let polynomial = rand_poly(1 << 10 - 1);
        let rng = &mut rand::thread_rng();
        let folding_randomness: BB = rng.gen();

        for log_arity in 1..10 {
            test_fold_evals_with_log_arity!(log_arity, polynomial, folding_randomness)
        }
    }

    #[test]
    fn test_fold_evaluations_binary() {
        let log_domain_size = 4;
        let poly_deg = 7;

        let rng = &mut rand::thread_rng();

        let polynomial = rand_poly(poly_deg);

        // Folding coefficient
        let c = rng.gen();

        // Points with the same 16-th power
        let point_root: BB = rng.gen();
        let point = point_root.exp_power_of_2(log_domain_size);
        let omega = BB::two_adic_generator(log_domain_size);
        let roots = iterate(point_root, |&x| x * omega)
            .take(1 << log_domain_size)
            .collect_vec();

        assert!(roots
            .iter()
            .all(|&x| x.exp_power_of_2(log_domain_size) == point));

        // Folding evaluations using the method
        let gammas = roots[0..(1 << log_domain_size) / 2]
            .iter()
            .map(|&root| c * root.inverse())
            .collect_vec();
        let evals = roots
            .iter()
            .map(|&root| polynomial.evaluate(&root))
            .collect_vec();
        let folded_evals = fold_evaluations_binary(evals.clone(), &gammas);

        // Computing folded evaluations by hand
        let folded_poly = fold_polynomial(&polynomial, c, 1);
        let roots_squared = roots[0..(1 << log_domain_size) / 2]
            .iter()
            .map(|&root| root.square())
            .collect_vec();
        let expected_folded_evals = roots_squared
            .iter()
            .map(|root| folded_poly.evaluate(root))
            .collect_vec();

        assert_eq!(folded_evals, expected_folded_evals);
    }
}
