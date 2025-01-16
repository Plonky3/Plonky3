use itertools::Itertools;
use p3_baby_bear::BabyBear;
use p3_field::{Field, FieldAlgebra};
use rand::distributions::Standard;
use rand::prelude::Distribution;
use rand::Rng;

use crate::polynomial::Polynomial;
type BB = BabyBear;

// NP TODO move?
// NP TODO receive Option<random generator>?
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

#[test]
fn test_evaluate() {
    let a_coeffs = vec![3, 2, 5, 6, 8]
        .into_iter()
        .map(BB::from_canonical_u32)
        .collect_vec();

    let a = Polynomial::from_coeffs(a_coeffs);
    let x = BB::from_canonical_u32(1);
    let y = a.evaluate(&x);
    assert_eq!(y, BB::from_canonical_u32(24));
}

#[test]
fn test_sum_sub() {
    // a(x) = 3 + 2x + 5x^2 + 6x^3
    let a_coeffs = vec![3, 2, 5, 6]
        .into_iter()
        .map(BB::from_canonical_u32)
        .collect_vec();

    // b(x) = 1 + 2x + 3x^2 + 4x^3 + 5x^4 + 7x^5 + 8x^6
    let b_coeffs = vec![1, 2, 3, 4, 5, 7, 8]
        .into_iter()
        .map(BB::from_canonical_u32)
        .collect_vec();

    let expected_coeffs_sum = vec![4, 4, 8, 10, 5, 7, 8]
        .into_iter()
        .map(BB::from_canonical_u32)
        .collect_vec();

    let expected_coeffs_sub = vec![2, 0, 2, 2]
        .into_iter()
        .map(BB::from_canonical_u32)
        .chain(vec![
            -BB::from_canonical_u32(5),
            -BB::from_canonical_u32(7),
            -BB::from_canonical_u32(8),
        ])
        .collect_vec();

    let a = Polynomial::from_coeffs(a_coeffs);
    let b = Polynomial::from_coeffs(b_coeffs);

    let sum = &a + &b;
    assert_eq!(sum.coeffs, expected_coeffs_sum);

    let sub = &a - &b;
    assert_eq!(sub.coeffs, expected_coeffs_sub);
}

#[test]
fn test_mul() {
    // a(x) = 3 + 2x + x^2 + x^3 + x^4
    let a_coeffs = vec![3, 2, 1, 1, 1]
        .into_iter()
        .map(BB::from_canonical_u32)
        .collect_vec();

    // b(x) = 8 + 7x + 6x^2
    let b_coeffs = vec![8, 7, 6]
        .into_iter()
        .map(BB::from_canonical_u32)
        .collect_vec();

    // c(x) = 24 + 37x + 40x^2 + 27x^3 + 21x^4 + 13x^5 + 6x^6
    let result_coeffs = vec![24, 37, 40, 27, 21, 13, 6]
        .into_iter()
        .map(BB::from_canonical_u32)
        .collect_vec();

    let a = Polynomial::from_coeffs(a_coeffs);
    let b = Polynomial::from_coeffs(b_coeffs);
    let c = &a * &b;

    assert_eq!(c.coeffs, result_coeffs);
}

#[test]
fn test_divide() {
    // a(x) = 3 - 2x - 5x^2 + 6x^3
    let a_coeffs = vec![
        BB::from_canonical_u32(3),
        -BB::from_canonical_u32(2),
        -BB::from_canonical_u32(5),
        BB::from_canonical_u32(6),
    ];

    // b(x) = -3 + x
    let b_coeffs = vec![-BB::from_canonical_u32(3), BB::from_canonical_u32(1)];

    // q(x) = 37 + 13x + 6x^2
    let q_coeffs = vec![
        BB::from_canonical_u32(37),
        BB::from_canonical_u32(13),
        BB::from_canonical_u32(6),
    ];

    // r(x) = 114
    let r_coeffs = vec![BB::from_canonical_u32(114)];

    let a = Polynomial::from_coeffs(a_coeffs);
    let b = Polynomial::from_coeffs(b_coeffs);

    let (q, r) = &a.divide_with_q_and_r(&b);

    assert_eq!(q.coeffs, q_coeffs);
    assert_eq!(r.coeffs, r_coeffs);
}

#[test]
fn test_naive_interpolate() {
    // p(x) = 3 + 4x + 2x^2 + 7x^3 + 9x^4 + x^5
    let polynomial = Polynomial::<BB>::from_coeffs(
        vec![3, 4, 2, 7, 9, 1]
            .into_iter()
            .map(BB::from_canonical_u32)
            .collect_vec(),
    );

    let points = vec![1, 2, 3, 4, 5, 6]
        .into_iter()
        .map(BB::from_canonical_u32)
        .collect_vec();

    // p(1) = 26, p(2) = 251, p(3) = 1194, p(4) = 3827, p(5) = 9698, p(6) = 21051
    let evals = vec![26, 251, 1194, 3827, 9698, 21051]
        .into_iter()
        .map(BB::from_canonical_u32)
        .collect_vec();

    let poly = Polynomial::<BB>::naive_interpolate(points.into_iter().zip(evals).collect_vec());
    assert_eq!(poly, polynomial);
}

#[test]
fn test_compose_with_exponent() {
    // p(x) = 3 + 2x + 6x^2 + 7x^3 + 9x^4
    let polynomial = Polynomial::<BB>::from_coeffs(
        vec![3, 2, 6, 7, 9]
            .into_iter()
            .map(BB::from_canonical_u32)
            .collect_vec(),
    );

    let exponent = 3;

    // p(x^3) = 3 + 2x^3 + 6x^6 + 7x^9 + 9x^12
    assert_eq!(
        polynomial.compose_with_exponent(exponent).coeffs(),
        vec![3, 0, 0, 2, 0, 0, 6, 0, 0, 7, 0, 0, 9]
            .into_iter()
            .map(BB::from_canonical_u32)
            .collect_vec()
    );
}
