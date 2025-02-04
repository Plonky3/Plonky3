use alloc::vec;
use itertools::Itertools;

use p3_baby_bear::BabyBear;
use p3_field::FieldAlgebra;

use rand::Rng;

use super::*;

type BB = BabyBear;

#[test]
fn test_interpolate_evals() {
    let coset = TwoAdicCoset::<BB>::new(BB::ONE, 3);

    let coeffs = vec![3, 5, 6, 7, 9]
        .into_iter()
        .map(BB::from_canonical_u32)
        .collect_vec();

    let polynomial = Polynomial::<BB>::from_coeffs(coeffs.clone());

    let evals = (0..1 << 3)
        .map(|i| polynomial.evaluate(&(coset.generator.clone().exp_u64(i) * coset.shift)))
        .collect_vec();

    let interpolation = coset.interpolate(evals);

    assert_eq!(interpolation.coeffs(), coeffs);
}

#[test]
fn test_coset_iterator() {
    let mut rng = rand::thread_rng();
    let shift: BB = rng.gen();
    let log_size = 3;

    let coset = TwoAdicCoset::<BB>::new(shift, log_size);

    assert_eq!(coset.clone().into_iter().count(), 1 << log_size);
    for (i, e) in coset.iter().enumerate() {
        assert_eq!(coset.element(i.try_into().unwrap()), e);
    }
}
