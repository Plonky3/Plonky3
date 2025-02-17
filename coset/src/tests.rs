use alloc::vec;

use itertools::Itertools;
use p3_baby_bear::BabyBear;
use p3_field::extension::BinomialExtensionField;
use p3_field::FieldAlgebra;
use p3_goldilocks::Goldilocks;
use rand::Rng;

use super::*;

type BB = BabyBear;
type GL = Goldilocks;
type GLExt = BinomialExtensionField<GL, 2>;

#[test]
// Checks that a coset of the maximum size allwed by the field (implementation)
// can indeed be constructed
fn test_coset_limit() {
    TwoAdicCoset::<BB>::new(BB::ONE, BB::TWO_ADICITY);
}

#[test]
#[should_panic = "bits <= Self::TWO_ADICITY"]
// Checks that attemtping to construct a field larger than allowed by the field
// implementation is disallowed
fn test_coset_too_large() {
    TwoAdicCoset::<BB>::new(BB::ONE, BB::TWO_ADICITY + 1);
}

#[test]
// Checks that the evaluation of a polynomial over a coset works as expected
fn test_evaluate_polynomial() {
    let mut rng = rand::thread_rng();
    let shift = rng.gen();
    let mut coset = TwoAdicCoset::<BB>::new(shift, 3);

    let coeffs = vec![5, 6, 7, 8, 9]
        .into_iter()
        .map(BB::from_canonical_u32)
        .collect_vec();

    let evals = (0..1 << 3)
        .map(|i| {
            coeffs.iter().rfold(BB::ZERO, |result, coeff| {
                result * (coset.generator().exp_u64(i) * shift) + *coeff
            })
        })
        .collect_vec();

    assert_eq!(coset.evaluate_polynomial(coeffs), evals);
}

#[test]
// Checks that interpolation over the coset works as expected
fn test_interpolate_evals() {
    let mut rng = rand::thread_rng();
    let shift = rng.gen();
    let mut coset = TwoAdicCoset::<BB>::new(shift, 3);

    let coeffs = vec![3, 5, 6, 7, 9]
        .into_iter()
        .map(BB::from_canonical_u32)
        .collect_vec();

    let evals = (0..1 << 3)
        .map(|i| {
            coeffs.iter().rfold(BB::ZERO, |result, coeff| {
                result * (coset.generator().exp_u64(i) * coset.shift) + *coeff
            })
        })
        .collect_vec();

    let interpolation_coeffs = coset.interpolate(evals);

    // The interpolation coefficients are not trimmed, so we need to add zeros
    // to the end to make the comparison work
    let expected_interpolation_coeffs =
        [coeffs.clone(), vec![BB::ZERO; (1 << 3) - coeffs.len()]].concat();
    assert_eq!(interpolation_coeffs, expected_interpolation_coeffs);
}

#[test]
#[should_panic = "is not large enough to be shrunk by a factor of 2^6"]
// Checks that attemtping to shrink a coset by any divisor of its size is
// allowed, but doing so by the next power of two is not
fn test_shrink_too_much() {
    let coset = TwoAdicCoset::<GL>::new(GL::from_canonical_u16(42), 5);

    for _ in 0..=6 {
        coset.shrink_subgroup(6);
    }
}

#[test]
// Checks that shrinking by a factor of 2^0 = 1 does nothing
fn test_shrink_nothing() {
    let coset = TwoAdicCoset::<BB>::new(BB::ZERO, 7);

    let shrunk = coset.shrink_subgroup(0);

    assert_eq!(shrunk.generator, coset.generator);
    assert_eq!(shrunk.shift, coset.shift);
}

#[test]
// Checks that shrinking the whole coset results in the expected new shift
fn test_shrink_shift() {
    let mut rng = rand::thread_rng();
    let shift: BB = rng.gen();

    let coset = TwoAdicCoset::<BB>::new(shift, 4);
    let shrunk = coset.shrink_coset(2);

    assert_eq!(shrunk.shift, shift.exp_power_of_2(2));
}

#[test]
// Checks that shrinking the coset by a factor of k results in a new coset whose
// i-th element is the original coset's (i * k)-th element
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
// Checks that the coset iterator yields the expected elements (in the expected
// order)
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
#[should_panic = "index must be less than the size of the coset."]
// Checks that attemtping to access an element at an index larger than the coset's
// size is disallowed (motivation in lib.rs/element)
fn test_element_index_too_large() {
    let mut coset = TwoAdicCoset::<BB>::new(BB::ONE, 3);
    coset.element(1 << 3);
}

#[test]
// Checks that the element method returns the expected values
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
// Checks that the coset stores the expected number of iterated squares of the
// generator
fn test_stored_values() {
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
// Checks that a coset is equal to itself
fn test_equality_reflexive() {
    let mut rng = rand::thread_rng();
    let shift = rng.gen();

    let coset1 = TwoAdicCoset::<GLExt>::new(shift, 8);
    assert_eq!(coset1, coset1.clone());
}

#[test]
// Checks that shrinking a coset keeps the relevant iterated squares of the
// generator
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
// Checks inequality between two arbitrary cosets
fn test_equality_shift() {
    let mut rng = rand::thread_rng();
    let shift = rng.gen();

    let coset1 = TwoAdicCoset::<GLExt>::new(shift, 10);
    let coset2 = coset1.set_shift(coset1.shift + GLExt::ONE);

    assert!(coset1 != coset2);
}

#[test]
// Checks that coset equality is invariant under translation by any element of
// the group, as expected
fn test_equality_translation() {
    let mut rng = rand::thread_rng();
    let shift = rng.gen();

    let coset1 = TwoAdicCoset::<GLExt>::new(shift, 10);
    let coset2 = coset1.shift_by(coset1.generator.exp_u64(22));

    assert!(coset1 == coset2);
}

#[test]
// Checks that the contains method returns true on all elements of the coset
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
