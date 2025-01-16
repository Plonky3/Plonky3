use itertools::Itertools;
use p3_baby_bear::BabyBear;
use p3_field::FieldAlgebra;

use super::*;

type F = BabyBear;

#[test]
fn test_interpolate_evals() {
    let coset = Radix2Coset::<F>::new(F::ONE, 3);

    let coeffs = vec![3, 5, 6, 7, 9]
        .into_iter()
        .map(F::from_canonical_u32)
        .collect_vec();

    let polynomial = Polynomial::<F>::from_coeffs(coeffs.clone());

    let evals = (0..1 << 3)
        .map(|i| polynomial.evaluate(&(coset.generator.clone().exp_u64(i) * coset.shift)))
        .collect_vec();

    let interpolation = coset.interpolate_evals(evals);

    assert_eq!(interpolation.coeffs(), coeffs);
}
