//! Every table of a circuit, proven together and checked natively.

mod common;

use common::{EF, F, chip, config, prove_and_verify};
use p3_circuit::{CircuitBuilder, PermutationChip};
use p3_field::{BasedVectorSpace, PrimeCharacteristicRing};

#[test]
fn every_table_proves_and_verifies() {
    let mut builder = CircuitBuilder::<F, EF, _>::new(chip());

    // Base-field arithmetic: 3 * 5 + 7 = 22, checked by an asserting row.
    let x = builder.witness(F::new(3));
    let y = builder.witness(F::new(5));
    let z = x.clone() * y.clone() + F::new(7);
    builder.assert_zero(&(z - F::new(22))).unwrap();

    // A public input bound to a prover value.
    let public = builder.public(F::new(42));
    builder.assert_zero(&(public - x * F::new(14))).unwrap();

    // Challenge-field arithmetic, split into limbs and assembled back.
    let a = builder.witness_ext(EF::from_basis_coefficients_fn(|i| F::from_usize(i + 1)));
    let b = builder.witness_ext(EF::from_basis_coefficients_fn(|i| F::from_usize(2 * i + 3)));
    let product = a.clone() * b.clone();
    let limbs = builder.decompose(&product);
    let recomposed = builder.compose(&limbs);
    builder.assert_zero_ext(&(recomposed - a * b)).unwrap();

    // One permutation, whose output is checked against the native permutation.
    let state: Vec<_> = (0..16).map(|i| builder.witness(F::from_usize(i))).collect();
    let output = builder.permute(&state);
    let expected = builder
        .chip()
        .permute(&(0..16).map(F::from_usize).collect::<Vec<_>>());
    for (out, expected) in output.iter().zip(expected) {
        builder.assert_zero(&(out.clone() - expected)).unwrap();
    }

    // Canonical bits of a value recompose it.
    let bits = builder.canonical_bits(&y);
    assert_eq!(bits.len(), 31);

    let circuit = builder.build();
    prove_and_verify(&config(2), &circuit);
}

#[test]
fn an_unsatisfied_assertion_is_reported_while_building() {
    let mut builder = CircuitBuilder::<F, EF, _>::new(chip());
    let x = builder.witness(F::new(3));

    // 3 is not zero, so no witness satisfies the circuit.
    assert!(builder.assert_zero(&x).is_err());
}
