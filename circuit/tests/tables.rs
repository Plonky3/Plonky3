//! Every table of a circuit, proven together and checked natively.

mod common;

use common::{Compress, Config, EF, F, chip, config, prove_and_verify};
use p3_batch_stark::{ProverData, prove_batch, verify_batch};
use p3_circuit::{CircuitBuilder, PermutationChip};
use p3_field::{BasedVectorSpace, PrimeCharacteristicRing};
use p3_koala_bear::default_koalabear_poseidon2_16;
use p3_symmetric::PseudoCompressionFunction;

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

    // A Merkle step on each side, checked against the native compression of the ordered children.
    let compress = Compress::new(default_koalabear_poseidon2_16());
    let digest: [F; 8] = core::array::from_fn(|i| F::from_usize(100 + i));
    let sibling: [F; 8] = core::array::from_fn(|i| F::from_usize(200 + i));
    let wires: Vec<_> = digest.iter().map(|&v| builder.witness(v)).collect();
    for (right, children) in [(false, [digest, sibling]), (true, [sibling, digest])] {
        let bit = builder.witness(F::from_bool(right));
        let parent = builder.merkle_step(&wires, &sibling, &bit);
        for (out, expected) in parent.iter().zip(compress.compress(children)) {
            builder.assert_zero(&(out.clone() - expected)).unwrap();
        }
    }

    // A conditional permutation on each choice: the permuted lanes, or the input lanes.
    let values: Vec<F> = (0..16).map(|i| F::from_usize(300 + i)).collect();
    let permuted = builder.chip().permute(&values);
    let inputs: Vec<_> = values.iter().map(|&v| builder.witness(v)).collect();
    for (take, expected) in [(true, &permuted), (false, &values)] {
        let bit = builder.witness(F::from_bool(take));
        let lanes = builder.permute_if(&inputs, &bit, 8);
        for (out, &expected) in lanes.iter().zip(expected.iter()) {
            builder.assert_zero(&(out.clone() - expected)).unwrap();
        }
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

#[test]
fn a_flipped_merkle_direction_breaks_the_wire_bus() {
    let mut builder = CircuitBuilder::<F, EF, _>::new(chip());
    let digest: Vec<_> = (0..8).map(|i| builder.witness(F::from_usize(i))).collect();
    let bit = builder.witness(F::ZERO);
    let parent = builder.merkle_step(&digest, &[F::ONE; 8], &bit);
    let public = builder.public(builder.value(&parent[0]));
    builder.assert_zero(&(public - parent[0].clone())).unwrap();
    let mut circuit = builder.build();

    // The prover claims the other side: the row now reads the sibling where the digest was.
    let permute = circuit.traces.last_mut().unwrap();
    let bit_column = permute.width - 1;
    permute.values[bit_column] = F::ONE;

    // Debug builds catch the imbalance while proving; otherwise the native verifier rejects.
    let config = config(2);
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let instances = circuit.instances::<Config>();
        let data = ProverData::from_instances(&config, &instances).unwrap();
        let proof = prove_batch(&config, &instances, &data).unwrap();
        verify_batch(
            &config,
            &circuit.airs,
            &proof,
            &circuit.public_values(),
            &data.common,
        )
    }));
    assert!(!matches!(outcome, Ok(Ok(()))));
}

#[test]
fn a_flipped_conditional_choice_breaks_the_wire_bus() {
    let mut builder = CircuitBuilder::<F, EF, _>::new(chip());
    let inputs: Vec<_> = (0..16).map(|i| builder.witness(F::from_usize(i))).collect();
    let take = builder.witness(F::ONE);
    let lanes = builder.permute_if(&inputs, &take, 8);
    let public = builder.public(builder.value(&lanes[0]));
    builder.assert_zero(&(public - lanes[0].clone())).unwrap();
    let mut circuit = builder.build();

    // The prover claims the input lanes were kept: the bit and the keep flag both flip.
    let permute = circuit.traces.last_mut().unwrap();
    let (bit, keep) = (permute.width - 2, permute.width - 1);
    permute.values[bit] = F::ZERO;
    permute.values[keep] = F::ONE;

    let config = config(2);
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let instances = circuit.instances::<Config>();
        let data = ProverData::from_instances(&config, &instances).unwrap();
        let proof = prove_batch(&config, &instances, &data).unwrap();
        verify_batch(
            &config,
            &circuit.airs,
            &proof,
            &circuit.public_values(),
            &data.common,
        )
    }));
    assert!(!matches!(outcome, Ok(Ok(()))));
}
