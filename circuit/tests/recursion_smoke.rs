//! A circuit proof verified inside a second circuit.

mod common;

use common::{Chip, EF, F, chip, config, prove_and_verify};
use p3_air::BaseAir;
use p3_batch_stark::{ProverData, prove_batch, verify_batch_in};
use p3_circuit::{Circuit, CircuitArith, CircuitBuilder, KeyBinding};

/// A proof of `6 + y = target`, verified by a second circuit.
fn outer_circuit(target: u32, keys: KeyBinding) -> Circuit<F, EF, Chip> {
    let config = config(2);

    // A small first circuit: one public input and a little arithmetic.
    let mut builder = CircuitBuilder::<F, EF, _>::new(chip());
    let x = builder.public(F::new(6));
    let y = builder.witness(F::new(target - 6));
    builder.assert_zero(&(x + y - F::new(target))).unwrap();
    let inner = builder.build();
    let instances = inner.instances::<common::Config>();
    let inner_data = ProverData::from_instances(&config, &instances).unwrap();
    let inner_proof = prove_batch(&config, &instances, &inner_data).unwrap();

    // Its proof, verified by a second circuit.
    let mut arith = CircuitArith::<F, EF, _>::new(chip()).with_key_binding(keys);
    let lifted = arith.public_inputs(&inner.public_values());
    verify_batch_in(
        &mut arith,
        &config,
        &inner.airs,
        &inner_proof,
        &lifted,
        &inner_data.common,
    )
    .expect("the circuit proof verifies inside a circuit");
    arith.build()
}

#[test]
fn a_circuit_proof_verifies_inside_a_circuit() {
    prove_and_verify(&config(2), &outer_circuit(42, KeyBinding::Constant));
}

#[test]
fn the_inner_verifying_key_is_part_of_the_outer_one() {
    // Two inner circuits of one shape, apart from one constant, so their verifying keys differ.
    let keys = |circuit: &Circuit<F, EF, Chip>| -> Vec<_> {
        circuit
            .airs
            .iter()
            .map(|air| air.preprocessed_trace())
            .collect()
    };

    // Held as constants, the inner keys make the outer keys differ.
    let (one, two) = (
        outer_circuit(42, KeyBinding::Constant),
        outer_circuit(43, KeyBinding::Constant),
    );
    assert_ne!(keys(&one), keys(&two));

    // Held as public inputs, the inner keys leave one outer key and move into the public values.
    let (one, two) = (
        outer_circuit(42, KeyBinding::Public),
        outer_circuit(43, KeyBinding::Public),
    );
    assert_eq!(keys(&one), keys(&two));
    assert_ne!(one.public_values(), two.public_values());
    prove_and_verify(&config(2), &one);
}
