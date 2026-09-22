//! Measures what a repeated word gadget costs, inline against composed.
//!
//! The same statement is built twice, then proved end to end both ways.
//!
//! One build writes the gadget's relations out once per use.
//!
//! The other declares the gadget once and gives it a live instance count.
//!
//! For each count the table reports:
//!
//! - committed cells: bits of the padded Boolean trace the commitment covers;
//! - proof bytes: the serialized transcript record;
//! - metadata entries: the compiled wiring each path has to store.
//!
//! The opening count is not a column, because it cannot vary.
//!
//! Both paths discharge the one point the proving routine passes to the commitment.
//!
//! Run it with `cargo run --release -p p3-word-backend --example component_scaling`.

use std::time::Instant;

use p3_binary_field::{BinaryChallenger, BinaryField128};
use p3_binary_pcs::{BinaryPcsConfig, BinaryPcsParams, BooleanPcs};
use p3_challenger::HashChallenger;
use p3_keccak::Keccak256Hash;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher};
use p3_word::{
    AndConstraint, Component, ComponentCall, Composition, ConstraintSystem, Operand, Shift,
    ShiftKind, ShiftedValue, ValueIndex, Word, Word64, ZeroConstraint,
};
use p3_word_backend::{PackedWitness, Statement, WordProofKey};

type EF = BinaryField128;
type MyHash = SerializingHasher<Keccak256Hash>;
type MyCompress = CompressionFunctionFromHasher<Keccak256Hash, 2, 32>;
type MyMmcs = MerkleTreeMmcs<EF, u8, MyHash, MyCompress, 2, 32>;
type Scheme = BooleanPcs<EF, MyMmcs, MyMmcs>;
type Challenger = BinaryChallenger<EF, HashChallenger<u8, Keccak256Hash, 32>>;

/// Base-two logarithm of the bits one committed element holds.
const ABSORBED: usize = 7;

/// Rotation applied to the second input of the gadget's first product.
const ROTATION: usize = 13;

/// Left shift folded into the gadget's linear relation.
const SHIFT: usize = 5;

/// Instance counts the sweep reports.
const SWEEP: [usize; 9] = [4, 8, 16, 32, 64, 128, 256, 512, 1024];

/// The repeated gadget: two products and one linear relation over five words.
fn gadget() -> Component<Word64> {
    let in0 = ValueIndex::public(0).unwrap();
    let in1 = ValueIndex::public(1).unwrap();
    let out = ValueIndex::public(2).unwrap();
    let t = ValueIndex::witness(0).unwrap();
    let u = ValueIndex::witness(1).unwrap();

    let rotated = ShiftedValue::single(in1, Shift::new(ShiftKind::RotateRight, ROTATION).unwrap());
    let shifted = ShiftedValue::single(in0, Shift::new(ShiftKind::LogicalLeft, SHIFT).unwrap());

    let body = ConstraintSystem::new(
        3,
        2,
        vec![ZeroConstraint::new(Operand::new(vec![
            ShiftedValue::plain(u),
            ShiftedValue::plain(t),
            shifted,
        ]))],
        vec![
            AndConstraint::new(
                Operand::single(ShiftedValue::plain(in0)),
                Operand::single(rotated),
                Operand::single(ShiftedValue::plain(t)),
            ),
            AndConstraint::new(
                Operand::single(ShiftedValue::plain(u)),
                Operand::single(ShiftedValue::plain(in1)),
                Operand::single(ShiftedValue::plain(out)),
            ),
        ],
        vec![],
    )
    .unwrap();
    Component::new(body, 2, 1).unwrap()
}

/// Fills every instance with distinct, natively computed words.
fn values(composition: &Composition<Word64>) -> (Vec<Word64>, Vec<Word64>) {
    let mut public = vec![Word64::new(0); composition.public_len()];
    let mut witness = vec![Word64::new(0); composition.witness_len()];
    for instance in 0..composition.calls()[0].instances() {
        let seed = instance as u64 + 1;
        let in0 = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15);
        let in1 = seed.wrapping_mul(0xc2b2_ae3d_27d4_eb4f) | 1;
        let t = in0 & in1.rotate_right(ROTATION as u32);
        let u = t ^ (in0 << SHIFT);

        let interface = composition.interface_mut(&mut public, 0, instance).unwrap();
        interface[0] = Word64::new(in0);
        interface[1] = Word64::new(in1);
        interface[2] = Word64::new(u & in1);
        let private = composition.locals_mut(&mut witness, 0, instance).unwrap();
        private[0] = Word64::new(t);
        private[1] = Word64::new(u);
    }
    (public, witness)
}

fn commitment_scheme(trace_variables: usize) -> Scheme {
    let mmcs = MyMmcs::new(
        MyHash::new(Keccak256Hash),
        MyCompress::new(Keccak256Hash),
        0,
    );
    let params = BinaryPcsParams {
        log_inv_rate: 2,
        pow_bits: 0,
        security_level: 40,
    };
    let config =
        BinaryPcsConfig::try_new_with_folding::<EF, EF>(trace_variables - ABSORBED, params, 1)
            .expect("the sweep arity supports one folding round");
    Scheme::new(config, mmcs.clone(), mmcs, trace_variables).expect("the sweep arity is valid")
}

/// One row of the table.
struct Row {
    /// Bits of the padded Boolean trace the commitment covers.
    committed_cells: usize,
    /// Serialized transcript record.
    proof_bytes: usize,
    /// Compiled wiring entries the path stores.
    metadata: usize,
    /// Wall time to prove, in milliseconds.
    prove_ms: f64,
}

impl Row {
    /// Proves and verifies one statement, then reports what it cost.
    ///
    /// The verification is a self-check on the fixture, not part of the measurement.
    ///
    /// The timing covers proving only, which is the part that grows with the instances.
    fn measure(statement: Statement<Word64>, public: &[Word64], witness: &[Word64]) -> Self {
        let metadata = statement
            .compiled_layout()
            .expect("the layout compiles")
            .footprint()
            .entries();
        let key = WordProofKey::new(statement).expect("the key compiles");
        let scheme = commitment_scheme(key.trace_variables());
        let values =
            PackedWitness::new(key.statement(), public, witness).expect("the shape matches");

        let start = Instant::now();
        let (commitment, proof) = key
            .prove::<EF, EF, _, _>(
                &scheme,
                &values,
                &mut Challenger::from_hasher(Vec::new(), Keccak256Hash),
            )
            .expect("the statement holds");
        let prove_ms = start.elapsed().as_secs_f64() * 1e3;

        key.verify::<EF, EF, _, _>(
            &scheme,
            &commitment,
            public,
            &proof,
            &mut Challenger::from_hasher(Vec::new(), Keccak256Hash),
        )
        .expect("the proof verifies");

        Self {
            // The padded trace is one bit per lane of the committed cube.
            committed_cells: 1 << key.trace_variables(),
            proof_bytes: postcard::to_allocvec(&proof)
                .expect("the record serializes")
                .len(),
            metadata,
            prove_ms,
        }
    }
}

fn main() {
    println!(
        "{:>6} {:>8} {:>12} {:>11} {:>11} {:>11} {:>10} {:>10}",
        "n", "witness", "cells", "bytes/in", "proof B", "meta in", "meta cmp", "prove ms",
    );

    for instances in SWEEP {
        let composition = Composition::new(vec![ComponentCall::new(gadget(), instances)])
            .expect("the sweep fits the compact address space");
        let (public, witness) = values(&composition);
        let flat = composition
            .lower()
            .expect("the lowered system is well formed");

        let inline = Row::measure(Statement::from(flat), &public, &witness);
        let composed = Row::measure(Statement::from(composition), &public, &witness);

        // The two paths describe one statement, so every proof figure must agree.
        assert_eq!(inline.committed_cells, composed.committed_cells);
        assert_eq!(inline.proof_bytes, composed.proof_bytes);

        println!(
            "{:>6} {:>8} {:>12} {:>11.2} {:>11} {:>11} {:>10} {:>10.1}",
            instances,
            witness.len(),
            composed.committed_cells,
            composed.proof_bytes as f64 / instances as f64,
            composed.proof_bytes,
            inline.metadata,
            composed.metadata,
            composed.prove_ms,
        );
    }

    println!();
    println!(
        "committed cells are {} bits per committed word",
        Word64::BITS
    );
    println!("'meta in' is the inline path's compiled wiring, 'meta cmp' the composed path's");
}
