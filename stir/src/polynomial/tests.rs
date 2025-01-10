use itertools::Itertools;
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;

use crate::polynomial::Polynomial;
type F = BabyBear;

#[test]
fn test_evaluate() {
    let a_coeffs = vec![3, 2, 5, 6, 8]
        .into_iter()
        .map(F::from_canonical_u32)
        .collect_vec();

    let a = Polynomial::from_coeffs(a_coeffs);
    let x = F::from_canonical_u32(1);
    let y = a.evaluate(&x);
    assert_eq!(y, F::from_canonical_u32(24));
}

#[test]
fn test_sum_sub() {
    // a(x) = 3 + 2x + 5x^2 + 6x^3
    let a_coeffs = vec![3, 2, 5, 6]
        .into_iter()
        .map(F::from_canonical_u32)
        .collect_vec();

    // b(x) = 1 + 2x + 3x^2 + 4x^3 + 5x^4 + 7x^5 + 8x^6
    let b_coeffs = vec![1, 2, 3, 4, 5, 7, 8]
        .into_iter()
        .map(F::from_canonical_u32)
        .collect_vec();

    // a
    let expected_coeffs_sum = vec![4, 4, 8, 10, 5, 7, 8]
        .into_iter()
        .map(F::from_canonical_u32)
        .collect_vec();

    let expected_coeffs_sub = vec![2, 0, 2, 2]
        .into_iter()
        .map(F::from_canonical_u32)
        .chain(vec![
            -F::from_canonical_u32(5),
            -F::from_canonical_u32(7),
            -F::from_canonical_u32(8),
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
        .map(F::from_canonical_u32)
        .collect_vec();

    // b(x) = 8 + 7x + 6x^2
    let b_coeffs = vec![8, 7, 6]
        .into_iter()
        .map(F::from_canonical_u32)
        .collect_vec();

    // c(x) = 24 + 37x + 40x^2 + 27x^3 + 21x^4 + 13x^5 + 6x^6
    let result_coeffs = vec![24, 37, 40, 27, 21, 13, 6]
        .into_iter()
        .map(F::from_canonical_u32)
        .collect_vec();

    let a = Polynomial::from_coeffs(a_coeffs);
    let b = Polynomial::from_coeffs(b_coeffs);
    let c = &a * &b;

    assert_eq!(c.coeffs, result_coeffs);
}
