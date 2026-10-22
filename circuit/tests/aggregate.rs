//! Proofs aggregated into a tree whose root proof verifies natively.

mod common;

use common::{Chip, Config, EF, F, Fibonacci, chip, config, fibonacci, prove_and_verify};
use p3_batch_stark::{BatchProof, CommonData, ProverData, StarkInstance, prove_batch};
use p3_circuit::recursion::{Child, aggregate, public_digest};
use p3_circuit::tables::CircuitAir;
use p3_circuit::{Circuit, KeyBinding};
use p3_field::PrimeCharacteristicRing;

/// A proof with the verifier data and public values its parent needs.
struct Proven<A> {
    airs: Vec<A>,
    common: CommonData<Config>,
    proof: BatchProof<Config>,
    public_values: Vec<Vec<F>>,
}

impl<A> Proven<A> {
    fn child(&self) -> Child<'_, Config, A> {
        Child {
            airs: &self.airs,
            common: &self.common,
            proof: &self.proof,
            public_values: &self.public_values,
        }
    }
}

/// A Fibonacci proof from `(a, b)`.
fn leaf(config: &Config, a: u32, b: u32) -> Proven<Fibonacci> {
    let (trace, publics) = fibonacci(F::new(a), F::new(b), 8);
    let instances = vec![StarkInstance {
        air: &Fibonacci,
        trace: &trace,
        public_values: publics.clone(),
    }];
    let prover_data = ProverData::from_instances(config, &instances).unwrap();
    let proof = prove_batch(config, &instances, &prover_data).unwrap();
    Proven {
        airs: vec![Fibonacci],
        common: prover_data.common,
        proof,
        public_values: vec![publics],
    }
}

/// The proof of a node circuit.
fn prove_node(config: &Config, circuit: Circuit<F, EF, Chip>) -> Proven<CircuitAir<F, EF, Chip>> {
    let instances = circuit.instances::<Config>();
    let prover_data = ProverData::from_instances(config, &instances).unwrap();
    let proof = prove_batch(config, &instances, &prover_data).unwrap();
    let public_values = circuit.public_values();
    drop(instances);
    Proven {
        airs: circuit.airs,
        common: prover_data.common,
        proof,
        public_values,
    }
}

#[test]
#[ignore = "proves circuits that verify circuit proofs; run from heavy CI"]
fn four_proofs_aggregate_into_one_root() {
    let config = config(2);
    let leaves = [(0, 1), (2, 3), (5, 8), (1, 1)].map(|(a, b)| leaf(&config, a, b));

    // The first layer: each node verifies two leaves.
    let nodes = [&leaves[..2], &leaves[2..]].map(|pair| {
        let children: Vec<_> = pair.iter().map(Proven::child).collect();
        let circuit = aggregate(chip(), &config, KeyBinding::Constant, &children).unwrap();
        prove_node(&config, circuit)
    });

    // The root verifies both nodes, and its own proof verifies natively.
    let children: Vec<_> = nodes.iter().map(Proven::child).collect();
    let root = aggregate(chip(), &config, KeyBinding::Constant, &children).unwrap();
    prove_and_verify(&config, &root);

    // The root's public values are the digest of the nodes' digests of the leaves' public values.
    for (node, pair) in nodes.iter().zip(leaves.chunks(2)) {
        let leaf_publics: Vec<&[Vec<F>]> =
            pair.iter().map(|leaf| &leaf.public_values[..]).collect();
        assert_eq!(
            node.public_values.concat(),
            public_digest(&chip(), &leaf_publics)
        );
    }
    let node_publics: Vec<&[Vec<F>]> = nodes.iter().map(|node| &node.public_values[..]).collect();
    assert_eq!(root.publics, public_digest(&chip(), &node_publics));
}

#[test]
fn a_wrong_public_value_fails_aggregation() {
    let config = config(2);
    let mut leaves = [(0, 1), (2, 3)].map(|(a, b)| leaf(&config, a, b));

    // The second leaf claims another result than its proof shows.
    leaves[1].public_values[0][2] += F::ONE;
    let children: Vec<_> = leaves.iter().map(Proven::child).collect();
    assert!(aggregate(chip(), &config, KeyBinding::Constant, &children).is_err());
}

#[test]
#[ignore = "proves circuits that verify circuit proofs; run from heavy CI"]
fn public_keys_come_before_the_digest() {
    let config = config(2);
    let leaves = [(0, 1), (2, 3)].map(|(a, b)| leaf(&config, a, b));
    let nodes = [&leaves[..1], &leaves[1..]].map(|single| {
        let children: Vec<_> = single.iter().map(Proven::child).collect();
        let circuit = aggregate(chip(), &config, KeyBinding::Constant, &children).unwrap();
        prove_node(&config, circuit)
    });

    // The root takes both node keys as public inputs.
    let children: Vec<_> = nodes.iter().map(Proven::child).collect();
    let root = aggregate(chip(), &config, KeyBinding::Public, &children).unwrap();
    prove_and_verify(&config, &root);

    // Whoever checks the root reads each node key, then the digest.
    let mut expected: Vec<F> = Vec::new();
    for node in &nodes {
        let key = &node.common.preprocessed.as_ref().unwrap().commitment;
        expected.extend(key.roots().iter().flatten());
    }
    let node_publics: Vec<&[Vec<F>]> = nodes.iter().map(|node| &node.public_values[..]).collect();
    expected.extend(public_digest(&chip(), &node_publics));
    assert_eq!(root.publics, expected);
}
