use alloc::vec;
use itertools::Itertools;

use p3_baby_bear::BabyBear;
use p3_field::{extension::BinomialExtensionField, FieldAlgebra};

use p3_goldilocks::Goldilocks;
use rand::Rng;

use super::*;

type BB = BabyBear;
type GL = Goldilocks;
type GLExt = BinomialExtensionField<GL, 2>;

#[test]
fn test_coset_limit() {
    TwoAdicCoset::<BB>::new(BB::ONE, BB::TWO_ADICITY);
}

#[test]
#[should_panic = "bits <= Self::TWO_ADICITY"]
fn test_coset_too_large() {
    TwoAdicCoset::<BB>::new(BB::ONE, BB::TWO_ADICITY + 1);
}

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
#[should_panic = "is not large enough to be shrunk by a factor of 2^6"]
fn test_shrink_too_much() {
    let coset = TwoAdicCoset::<GL>::new(GL::from_canonical_u16(42), 5);

    for _ in 0..=6 {
        coset.shrink_subgroup(6);
    }
}

#[test]
fn test_shrink_nothing() {
    let coset = TwoAdicCoset::<BB>::new(BB::ZERO, 7);

    let shrunk = coset.shrink_subgroup(0);

    assert_eq!(shrunk.generator, coset.generator);
    assert_eq!(shrunk.shift, coset.shift);
}

#[test]
fn test_shrink_shift() {
    let mut rng = rand::thread_rng();
    let shift: BB = rng.gen();

    let coset = TwoAdicCoset::<BB>::new(shift, 4);
    let shrunk = coset.shrink_coset(2);

    assert_eq!(shrunk.shift, shift.exp_power_of_2(2));
}

#[test]
fn test_shrink_contained() {
    let mut rng = rand::thread_rng();
    let shift: GL = rng.gen();

    let log_shrinking_factor = 3;

    let mut coset = TwoAdicCoset::<GL>::new(shift, 8);
    let shrunk = coset.shrink_subgroup(log_shrinking_factor);

    for (i, e) in shrunk.iter().enumerate() {
        assert_eq!(coset.element(i * (1 << log_shrinking_factor)), e);
    }
}

#[test]
fn test_coset_iterator() {
    let mut rng = rand::thread_rng();
    let shift: BB = rng.gen();
    let log_size = 3;

    let mut coset = TwoAdicCoset::<BB>::new(shift, log_size);

    assert_eq!(coset.clone().into_iter().count(), 1 << log_size);
    for (i, e) in coset.iter().enumerate() {
        assert_eq!(coset.element(i.try_into().unwrap()), e);
    }
}

#[test]
#[should_panic = "exp must be less than the size of the coset."]
fn test_element_exp_too_large() {
    let mut coset = TwoAdicCoset::<BB>::new(BB::ONE, 3);
    coset.element(1 << 3);
}

#[test]
fn test_element() {
    let mut rng = rand::thread_rng();

    let shift: GL = rng.gen();
    let mut coset = TwoAdicCoset::<GL>::new(shift, GL::TWO_ADICITY);

    for _ in 0..100 {
        let exp = rng.gen::<u64>() % (1 << GL::TWO_ADICITY);
        let expected = coset.shift * coset.generator.exp_u64(exp);
        assert_eq!(coset.element(exp as usize), expected);
    }
}

#[test]
fn test_stored_values() {
    // Check the iterated squares of the generator are stored appropriately

    let mut rng = rand::thread_rng();

    let mut coset = TwoAdicCoset::<GLExt>::new(GLExt::from_canonical_u32(424242), 20);

    let n_bits = 11;

    // Some number with its MSB at the n_bits-th bit
    let index = (1 << n_bits) - 1 - rng.gen::<u8>() as usize;

    let expected = coset.shift * coset.generator.exp_u64(index as u64);
    assert_eq!(coset.element(index), expected);

    assert_eq!(coset.generator_iter_squares.len(), n_bits);
}

#[test]
fn test_equality_reflexive() {
    let mut rng = rand::thread_rng();
    let shift = rng.gen();

    let coset1 = TwoAdicCoset::<GLExt>::new(shift, 8);
    assert_eq!(coset1, coset1.clone());
}

#[test]
fn test_stored_values_shrink() {
    let mut coset = TwoAdicCoset::<GL>::new(GL::ONE, 16);

    // After this call, coset will have stored gen^(2^0), ..., gen^(2^9)
    coset.element((1 << 10) - 1);

    // This should keep gen^(2^3), ..., gen^(2^9), which is the same as
    // new_gen^(2^0), ..., new_gen^(2^6) with new_gen being the generator of the
    // shrunk coset (i. e. gen^(2^3))
    let shrunk = coset.shrink_subgroup(3);

    assert_eq!(shrunk.generator_iter_squares.len(), 7);
}

#[test]
fn test_equality_shift() {
    let mut rng = rand::thread_rng();
    let shift = rng.gen();

    let coset1 = TwoAdicCoset::<GLExt>::new(shift, 10);
    let coset2 = coset1.set_shift(coset1.shift + GLExt::ONE);

    assert!(coset1 != coset2);
}

#[test]
fn test_equality_translation() {
    let mut rng = rand::thread_rng();
    let shift = rng.gen();

    let coset1 = TwoAdicCoset::<GLExt>::new(shift, 10);
    let coset2 = coset1.shift_by(coset1.generator.exp_u64(22));

    assert!(coset1 == coset2);
}

#[test]
fn test_contains() {
    let mut rng = rand::thread_rng();
    let shift = rng.gen();

    let log_size = 8;

    let coset = TwoAdicCoset::<BB>::new(shift, log_size);

    let mut d = BB::ONE;

    for _ in 0..(1 << log_size) {
        assert!(coset.contains(coset.shift() * d));
        d *= coset.generator();
    }
}
