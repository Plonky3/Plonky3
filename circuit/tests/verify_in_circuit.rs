//! A batch-STARK proof verified inside a circuit, whose own proof then verifies natively.

mod common;

use common::{EF, F, Fibonacci, chip, config, fibonacci, prove_and_verify};
use p3_air::BaseAir;
use p3_batch_stark::{ProverData, StarkInstance, prove_batch, verify_batch, verify_batch_in};
use p3_circuit::CircuitArith;
use p3_field::PrimeCharacteristicRing;
use p3_uni_stark::{prove, verify, verify_in};

#[test]
fn a_batch_proof_verifies_inside_a_circuit() {
    let config = config(2);

    // The inner proof: one Fibonacci instance of 8 rows.
    let (trace, publics) = fibonacci(F::ZERO, F::ONE, 8);
    let publics = [publics];
    let air = Fibonacci;
    let instances = vec![StarkInstance {
        air: &air,
        trace: &trace,
        public_values: publics[0].clone(),
    }];
    let prover_data = ProverData::from_instances(&config, &instances).unwrap();
    let proof = prove_batch(&config, &instances, &prover_data).unwrap();
    verify_batch(&config, &[air], &proof, &publics, &prover_data.common).unwrap();

    // The same verifier, run on circuit values: the inner public values become circuit inputs.
    let mut arith = CircuitArith::<F, EF, _>::new(chip());
    let lifted = arith.public_inputs(&publics);
    verify_batch_in(
        &mut arith,
        &config,
        &[air],
        &proof,
        &lifted,
        &prover_data.common,
    )
    .expect("the inner proof verifies inside the circuit");

    let circuit = arith.build();
    prove_and_verify(&config, &circuit);
}

#[test]
fn a_tampered_inner_proof_is_rejected_inside_the_circuit() {
    let config = config(2);
    let (trace, publics) = fibonacci(F::ZERO, F::ONE, 8);
    let publics = [publics];
    let air = Fibonacci;
    let instances = vec![StarkInstance {
        air: &air,
        trace: &trace,
        public_values: publics[0].clone(),
    }];
    let prover_data = ProverData::from_instances(&config, &instances).unwrap();
    let mut proof = prove_batch(&config, &instances, &prover_data).unwrap();

    // One opened trace value moves, so the constraints at zeta no longer meet the quotient.
    proof.opened_values.instances[0]
        .base_opened_values
        .trace_local[0] += EF::ONE;

    let mut arith = CircuitArith::<F, EF, _>::new(chip());
    let lifted = arith.public_inputs(&publics);
    assert!(
        verify_batch_in(
            &mut arith,
            &config,
            &[air],
            &proof,
            &lifted,
            &prover_data.common
        )
        .is_err()
    );
}

#[test]
fn a_tampered_circuit_witness_breaks_the_wire_bus() {
    let config = config(2);
    let (trace, publics) = fibonacci(F::ZERO, F::ONE, 8);
    let publics = [publics];
    let air = Fibonacci;
    let instances = vec![StarkInstance {
        air: &air,
        trace: &trace,
        public_values: publics[0].clone(),
    }];
    let prover_data = ProverData::from_instances(&config, &instances).unwrap();
    let proof = prove_batch(&config, &instances, &prover_data).unwrap();

    let mut arith = CircuitArith::<F, EF, _>::new(chip());
    let lifted = arith.public_inputs(&publics);
    verify_batch_in(
        &mut arith,
        &config,
        &[air],
        &proof,
        &lifted,
        &prover_data.common,
    )
    .unwrap();
    let mut circuit = arith.build();
    // A prover-chosen value changes in the value table only, so its readers see another value.
    let value_table = &mut circuit.traces[0];
    value_table.values[0] += F::ONE;

    // Debug builds catch the imbalance while proving; otherwise the native verifier rejects.
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let instances = circuit.instances::<common::Config>();
        let circuit_data = ProverData::from_instances(&config, &instances).unwrap();
        let outer = prove_batch(&config, &instances, &circuit_data).unwrap();
        verify_batch(
            &config,
            &circuit.airs,
            &outer,
            &circuit.public_values(),
            &circuit_data.common,
        )
    }));
    assert!(!matches!(outcome, Ok(Ok(()))));
}

#[test]
fn a_uni_stark_proof_verifies_inside_a_circuit() {
    let config = config(2);
    let (trace, publics) = fibonacci(F::ZERO, F::ONE, 8);
    let proof = prove(&config, &Fibonacci, trace, &publics).unwrap();
    verify(&config, &Fibonacci, &proof, &publics).unwrap();

    // The single-AIR verifier runs on circuit values too.
    let mut arith = CircuitArith::<F, EF, _>::new(chip());
    let lifted = arith.public_inputs(&[publics]).swap_remove(0);
    verify_in(&mut arith, &config, &Fibonacci, &proof, &lifted, None)
        .expect("the inner proof verifies inside the circuit");
    prove_and_verify(&config, &arith.build());
}

#[test]
fn proofs_of_one_shape_give_one_verifying_key() {
    let config = config(2);

    // Two inner proofs of one AIR and one height, with different values throughout.
    let circuits = [(F::ZERO, F::ONE), (F::new(3), F::new(5))].map(|(a, b)| {
        let (trace, publics) = fibonacci(a, b, 8);
        let publics = [publics];
        let instances = vec![StarkInstance {
            air: &Fibonacci,
            trace: &trace,
            public_values: publics[0].clone(),
        }];
        let prover_data = ProverData::from_instances(&config, &instances).unwrap();
        let proof = prove_batch(&config, &instances, &prover_data).unwrap();

        let mut arith = CircuitArith::<F, EF, _>::new(chip());
        let lifted = arith.public_inputs(&publics);
        verify_batch_in(
            &mut arith,
            &config,
            &[Fibonacci],
            &proof,
            &lifted,
            &prover_data.common,
        )
        .unwrap();
        arith.build()
    });

    // The outer circuits differ only in their witness, so their preprocessed columns agree.
    let [one, two] = &circuits;
    assert_eq!(one.airs.len(), two.airs.len());
    for (a, b) in one.airs.iter().zip(&two.airs) {
        assert_eq!(a.preprocessed_trace(), b.preprocessed_trace());
    }
}
