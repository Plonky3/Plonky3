//! Repeated word gadgets proved as instances of one compiled component.
//!
//! Every test here is about the one thing a shared gadget can get wrong.
//!
//! That is the index arithmetic telling one instance from another.
//!
//! The equivalence test pins the composed path to the written-out path it replaces.
//!
//! The rest are the three attacks a shared gadget invites:
//!
//! - a witness that does not match the declared instance count;
//! - two instances reading one cell;
//! - a proof that repeats one instance where two were declared.

use p3_binary_field::{BinaryChallenger, BinaryField128};
use p3_binary_pcs::{BinaryPcsConfig, BinaryPcsParams, BooleanPcs, BooleanPcsError, BooleanProof};
use p3_challenger::HashChallenger;
use p3_commit::Mmcs;
use p3_keccak::Keccak256Hash;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher};
use p3_word::{
    AndConstraint, Component, ComponentCall, Composition, ConstraintSystem, Operand, Segment,
    Shift, ShiftKind, ShiftedValue, ValueIndex, Word64, ZeroConstraint,
};
use p3_word_backend::{PackedWitness, Statement, WordProof, WordProofError, WordProofKey};

type EF = BinaryField128;
type MyHash = SerializingHasher<Keccak256Hash>;
type MyCompress = CompressionFunctionFromHasher<Keccak256Hash, 2, 32>;
type MyMmcs = MerkleTreeMmcs<EF, u8, MyHash, MyCompress, 2, 32>;
type Scheme = BooleanPcs<EF, MyMmcs, MyMmcs>;
type Challenger = BinaryChallenger<EF, HashChallenger<u8, Keccak256Hash, 32>>;
type SchemeProof = BooleanProof<EF, MyMmcs, MyMmcs>;
type SchemeError = BooleanPcsError<EF, <MyMmcs as Mmcs<EF>>::Error>;
type Commitment = <MyMmcs as Mmcs<EF>>::Commitment;
type Record = WordProof<EF, EF, SchemeProof>;

/// Base-two logarithm of the bits one committed element holds.
const ABSORBED: usize = 7;

/// Rotation applied to the second input of the gadget's first product.
const ROTATION: usize = 13;

/// Left shift folded into the gadget's linear relation.
const SHIFT: usize = 5;

// ---------------------------------------------------------------------------
// The gadget
// ---------------------------------------------------------------------------

/// A three-relation word gadget, declared once against component-local slots.
///
/// ```text
/// public  in0 = slot 0, in1 = slot 1, out = slot 2
/// local   t   = slot 0, u   = slot 1
///
/// t = in0 & rotr(in1, 13)
/// u = t ^ (in0 << 5)
/// out = u & in1
/// ```
fn gadget() -> Component<Word64> {
    let in0 = ValueIndex::public(0).expect("slot fits");
    let in1 = ValueIndex::public(1).expect("slot fits");
    let out = ValueIndex::public(2).expect("slot fits");
    let t = ValueIndex::witness(0).expect("slot fits");
    let u = ValueIndex::witness(1).expect("slot fits");

    let rotated = ShiftedValue::single(
        in1,
        Shift::new(ShiftKind::RotateRight, ROTATION).expect("rotation is in range"),
    );
    let shifted = ShiftedValue::single(
        in0,
        Shift::new(ShiftKind::LogicalLeft, SHIFT).expect("shift is in range"),
    );

    let body = ConstraintSystem::new(
        3,
        2,
        // u ^ t ^ (in0 << 5) must vanish.
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
    .expect("the gadget addresses only its declared slots");

    Component::new(body, 2, 1).expect("two inputs and one output span the interface")
}

/// Runs the gadget natively, returning its output word and its two private words.
const fn run_gadget(in0: u64, in1: u64) -> (u64, [u64; 2]) {
    let t = in0 & in1.rotate_right(ROTATION as u32);
    let u = t ^ (in0 << SHIFT);
    (u & in1, [t, u])
}

/// Deterministic, distinct inputs so no instance can stand in for another.
const fn instance_inputs(instance: usize) -> (u64, u64) {
    let seed = instance as u64 + 1;
    (
        seed.wrapping_mul(0x9e37_79b9_7f4a_7c15),
        seed.wrapping_mul(0xc2b2_ae3d_27d4_eb4f) | 1,
    )
}

/// Builds a composition of one call with the requested number of live instances.
fn composition(instances: usize) -> Composition<Word64> {
    Composition::new(vec![ComponentCall::new(gadget(), instances)])
        .expect("the fixture fits the compact address space")
}

/// Fills every instance's interface and private words.
fn honest_values(composition: &Composition<Word64>) -> (Vec<Word64>, Vec<Word64>) {
    let mut public = vec![Word64::new(0); composition.public_len()];
    let mut witness = vec![Word64::new(0); composition.witness_len()];
    let instances = composition.calls()[0].instances();

    for instance in 0..instances {
        let (in0, in1) = instance_inputs(instance);
        let (out, locals) = run_gadget(in0, in1);

        // Each instance writes through its own disjoint slice.
        let interface = composition
            .interface_mut(&mut public, 0, instance)
            .expect("the instance exists");
        interface[0] = Word64::new(in0);
        interface[1] = Word64::new(in1);
        interface[2] = Word64::new(out);

        let private = composition
            .locals_mut(&mut witness, 0, instance)
            .expect("the instance exists");
        private[0] = Word64::new(locals[0]);
        private[1] = Word64::new(locals[1]);
    }
    (public, witness)
}

// ---------------------------------------------------------------------------
// Proving harness
// ---------------------------------------------------------------------------

const fn challenger() -> Challenger {
    Challenger::from_hasher(Vec::new(), Keccak256Hash)
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
            .expect("the fixture arity supports one folding round");
    Scheme::new(config, mmcs.clone(), mmcs, trace_variables).expect("the fixture arity is valid")
}

fn prove(
    key: &WordProofKey<Word64>,
    scheme: &Scheme,
    public: &[Word64],
    witness: &[Word64],
) -> Result<(Commitment, Record), WordProofError<SchemeError>> {
    let values = PackedWitness::new(key.statement(), public, witness)
        .expect("the fixture matches the declared shape");
    key.prove::<EF, EF, _, _>(scheme, &values, &mut challenger())
}

fn verify(
    key: &WordProofKey<Word64>,
    scheme: &Scheme,
    commitment: &Commitment,
    public: &[Word64],
    proof: &Record,
) -> Result<(), WordProofError<SchemeError>> {
    key.verify::<EF, EF, _, _>(scheme, commitment, public, proof, &mut challenger())
}

fn bytes(proof: &Record) -> Vec<u8> {
    postcard::to_allocvec(proof).expect("the transcript record serializes")
}

// ---------------------------------------------------------------------------
// Equivalence with the duplicated-inline statement
// ---------------------------------------------------------------------------

#[test]
fn a_composed_key_and_its_lowered_key_produce_the_same_proof() {
    // Fixture state: four instances, once as a component and once written out.
    let composition = composition(4);
    let flat = composition
        .lower()
        .expect("the lowered system is well formed");
    let (public, witness) = honest_values(&composition);

    let composed_key = WordProofKey::new(composition).expect("the composed key compiles");
    let flat_key = WordProofKey::new(flat).expect("the flat key compiles");
    assert_eq!(composed_key.trace_variables(), flat_key.trace_variables());
    let scheme = commitment_scheme(composed_key.trace_variables());

    let (composed_commitment, composed_proof) =
        prove(&composed_key, &scheme, &public, &witness).expect("the composed statement holds");
    let (flat_commitment, flat_proof) =
        prove(&flat_key, &scheme, &public, &witness).expect("the flat statement holds");

    // The composed path is another description of one statement, not another protocol.
    //
    // The transcript record is therefore the same byte for byte.
    assert_eq!(composed_commitment, flat_commitment);
    assert_eq!(bytes(&composed_proof), bytes(&flat_proof));

    // Either key verifies either proof, which is the same claim from the other side.
    assert!(
        verify(
            &composed_key,
            &scheme,
            &flat_commitment,
            &public,
            &flat_proof
        )
        .is_ok()
    );
    assert!(
        verify(
            &flat_key,
            &scheme,
            &composed_commitment,
            &public,
            &composed_proof
        )
        .is_ok()
    );
}

#[test]
fn the_composed_layout_is_the_lowered_layout() {
    // The compiled wiring is what the reduction reads, so it must match exactly.
    for instances in [1_usize, 2, 3, 5] {
        let composition = composition(instances);
        let flat = composition
            .lower()
            .expect("the lowered system is well formed");

        let composed = Statement::from(composition)
            .compiled_layout()
            .expect("the composed layout compiles");
        let lowered = Statement::from(flat)
            .compiled_layout()
            .expect("the lowered layout compiles");

        assert_eq!(composed.witness().len(), lowered.witness().len());
        assert_eq!(composed.public().len(), lowered.public().len());
        for (segment, composed, lowered) in [
            (Segment::Public, composed.public(), lowered.public()),
            (Segment::Witness, composed.witness(), lowered.witness()),
        ] {
            for word in 0..lowered.len() {
                let actual = composed
                    .keys(word)
                    .expect("the composed segment covers this word")
                    .map(|key| {
                        (
                            key.operation(),
                            key.shifts(),
                            key.references().collect::<Vec<_>>(),
                        )
                    })
                    .collect::<Vec<_>>();
                let expected = lowered
                    .keys(word)
                    .expect("the lowered segment covers this word")
                    .map(|key| {
                        (
                            key.operation(),
                            key.shifts(),
                            key.references().collect::<Vec<_>>(),
                        )
                    })
                    .collect::<Vec<_>>();
                assert_eq!(actual, expected, "{segment:?} word {word}");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The measurement the acceptance criterion asks for
// ---------------------------------------------------------------------------

#[test]
fn compiled_metadata_is_flat_in_the_instance_count() {
    // One component's wiring is stored once, whatever the instance count.
    let mut composed = Vec::new();
    let mut lowered = Vec::new();
    for instances in [1_usize, 2, 4, 8, 16, 32] {
        let composition = composition(instances);
        let flat = composition
            .lower()
            .expect("the lowered system is well formed");

        composed.push(
            Statement::from(composition)
                .compiled_layout()
                .expect("the composed layout compiles")
                .footprint()
                .entries(),
        );
        lowered.push(
            Statement::from(flat)
                .compiled_layout()
                .expect("the lowered layout compiles")
                .footprint()
                .entries(),
        );
    }

    // The composed footprint does not move at all.
    assert!(
        composed.windows(2).all(|pair| pair[0] == pair[1]),
        "composed footprints {composed:?} are not constant"
    );

    // The written-out footprint doubles with every doubling of the instances.
    for pair in lowered.windows(2) {
        assert!(
            pair[1] > pair[0],
            "written-out footprints {lowered:?} do not grow"
        );
    }
    assert!(
        lowered.last().copied().expect("the sweep is not empty")
            > 16 * composed.last().copied().expect("the sweep is not empty"),
        "written-out {lowered:?} against composed {composed:?}"
    );
}

// ---------------------------------------------------------------------------
// Attack: an instance count that does not match the witness
// ---------------------------------------------------------------------------

#[test]
fn a_witness_for_another_instance_count_is_refused() {
    let declared = composition(4);
    let supplied = composition(3);
    let (public, witness) = honest_values(&supplied);

    // Every instance's own words are correct; only the count is wrong.
    assert_eq!(
        supplied
            .lower()
            .expect("the lowered system is well formed")
            .verify(&public, &witness),
        Ok(())
    );

    let key = WordProofKey::new(declared).expect("the composed key compiles");
    assert_eq!(
        PackedWitness::new(key.statement(), &public, &witness),
        Err(p3_word::ShapeError {
            segment: Segment::Public,
            expected: 12,
            actual: 9,
        })
    );
}

#[test]
fn a_proof_of_one_instance_count_does_not_verify_at_another() {
    // Fixture state: two instances proved, three declared at verification.
    let proved = composition(2);
    let (public, witness) = honest_values(&proved);
    let proved_key = WordProofKey::new(proved).expect("the composed key compiles");
    let declared = composition(3);
    let declared_key = WordProofKey::new(declared).expect("the composed key compiles");

    // One commitment wide enough for both, so the arity check cannot fire first.
    let scheme = commitment_scheme(declared_key.trace_variables());
    let (commitment, proof) =
        prove(&proved_key, &scheme, &public, &witness).expect("the statement holds");

    // The verifier's own shape check rejects before any transcript is replayed.
    assert!(matches!(
        verify(&declared_key, &scheme, &commitment, &public, &proof),
        Err(WordProofError::SegmentLength {
            segment: Segment::Public,
            expected: 9,
            actual: 6,
        })
    ));

    // Padding the public words to the declared length does not rescue it either.
    let mut padded = public;
    padded.resize(9, Word64::new(0));
    assert!(verify(&declared_key, &scheme, &commitment, &padded, &proof).is_err());
}

// ---------------------------------------------------------------------------
// Attack: two instances sharing a cell they should not
// ---------------------------------------------------------------------------

#[test]
fn one_instance_cannot_borrow_another_instance_private_word() {
    let composition = composition(2);
    let (public, honest) = honest_values(&composition);
    let key = WordProofKey::new(composition.clone()).expect("the composed key compiles");
    let scheme = commitment_scheme(key.trace_variables());

    // The honest witness proves and verifies.
    let (commitment, proof) = prove(&key, &scheme, &public, &honest).expect("the statement holds");
    assert!(verify(&key, &scheme, &commitment, &public, &proof).is_ok());

    // Instance one now reads instance zero's private words instead of its own.
    let mut shared = honest.clone();
    let borrowed = composition
        .locals(&honest, 0, 0)
        .expect("the instance exists")
        .to_vec();
    composition
        .locals_mut(&mut shared, 0, 1)
        .expect("the instance exists")
        .copy_from_slice(&borrowed);
    assert_ne!(shared, honest);

    // The scalar reference agrees the composed statement no longer holds.
    assert!(
        composition
            .lower()
            .expect("the lowered system is well formed")
            .verify(&public, &shared)
            .is_err()
    );

    // The proof of the shared witness cannot close the vanishing check.
    let (commitment, proof) =
        prove(&key, &scheme, &public, &shared).expect("a transcript is produced regardless");
    // The vanishing check no longer closes against the claimed operands.
    assert!(matches!(
        verify(&key, &scheme, &commitment, &public, &proof),
        Err(WordProofError::RelationClaim)
    ));
}

#[test]
fn every_instance_slot_resolves_to_its_own_word() {
    // Two calls of the same gadget so cross-call aliasing is covered too.
    let composition = Composition::new(vec![
        ComponentCall::new(gadget(), 3),
        ComponentCall::new(gadget(), 4),
    ])
    .expect("the fixture fits the compact address space");

    let mut public = Vec::new();
    let mut witness = Vec::new();
    for (call, instances) in [(0_usize, 3_usize), (1, 4)] {
        for instance in 0..instances {
            for slot in 0..3 {
                let index = ValueIndex::public(slot).expect("slot fits");
                public.push(
                    composition
                        .resolve(call, instance, index)
                        .expect("the instance exists")
                        .position(),
                );
            }
            for slot in 0..2 {
                let index = ValueIndex::witness(slot).expect("slot fits");
                witness.push(
                    composition
                        .resolve(call, instance, index)
                        .expect("the instance exists")
                        .position(),
                );
            }
        }
    }

    // Seven instances, three interface slots and two private slots each.
    assert_eq!(public.len(), 21);
    assert_eq!(witness.len(), 14);
    for mut positions in [public, witness] {
        let total = positions.len();
        positions.sort_unstable();
        positions.dedup();
        assert_eq!(positions.len(), total, "two slots share one word");
    }
}

// ---------------------------------------------------------------------------
// Attack: a proof that repeats one instance where two were declared
// ---------------------------------------------------------------------------

#[test]
fn repeating_one_instance_where_two_were_declared_is_rejected() {
    // Fixture state: two declared instances, both filled from instance zero.
    let composition = composition(2);
    let key = WordProofKey::new(composition.clone()).expect("the composed key compiles");
    let scheme = commitment_scheme(key.trace_variables());

    let (in0, in1) = instance_inputs(0);
    let (out, locals) = run_gadget(in0, in1);
    let mut public = vec![Word64::new(0); composition.public_len()];
    let mut witness = vec![Word64::new(0); composition.witness_len()];
    for instance in 0..2 {
        let interface = composition
            .interface_mut(&mut public, 0, instance)
            .expect("the instance exists");
        interface[0] = Word64::new(in0);
        interface[1] = Word64::new(in1);
        interface[2] = Word64::new(out);
        let private = composition
            .locals_mut(&mut witness, 0, instance)
            .expect("the instance exists");
        private[0] = Word64::new(locals[0]);
        private[1] = Word64::new(locals[1]);
    }

    // Repeating an instance is a valid statement, just not the one the verifier wants.
    //
    // The public interface names the repetition, so the two are told apart.
    let (commitment, proof) =
        prove(&key, &scheme, &public, &witness).expect("the repeated statement holds");
    assert!(verify(&key, &scheme, &commitment, &public, &proof).is_ok());

    // Against the interface the verifier actually holds, the proof fails.
    //
    // The second instance's public words are bound into the transcript first.
    //
    // The repetition therefore cannot pass for two different instances.
    let (declared_public, _) = honest_values(&composition);
    assert_ne!(declared_public, public);
    assert!(verify(&key, &scheme, &commitment, &declared_public, &proof).is_err());
}

#[test]
fn a_second_instance_cannot_reuse_the_first_committed_words() {
    // Fixture state: an interface declaring two distinct instances.
    //
    // The witness behind it only ever computed the first one.
    let composition = composition(2);
    let (public, honest) = honest_values(&composition);
    let key = WordProofKey::new(composition.clone()).expect("the composed key compiles");
    let scheme = commitment_scheme(key.trace_variables());

    let first = composition
        .locals(&honest, 0, 0)
        .expect("the instance exists")
        .to_vec();
    let mut repeated = honest;
    composition
        .locals_mut(&mut repeated, 0, 1)
        .expect("the instance exists")
        .copy_from_slice(&first);

    let (commitment, proof) =
        prove(&key, &scheme, &public, &repeated).expect("a transcript is produced regardless");
    // The vanishing check no longer closes against the claimed operands.
    assert!(matches!(
        verify(&key, &scheme, &commitment, &public, &proof),
        Err(WordProofError::RelationClaim)
    ));
}

// ---------------------------------------------------------------------------
// Public inputs and outputs
// ---------------------------------------------------------------------------

#[test]
fn public_inputs_and_outputs_are_derived_from_the_interface_alone() {
    let composition = composition(3);
    let (public, _) = honest_values(&composition);

    for instance in 0..3 {
        let (in0, in1) = instance_inputs(instance);
        let (out, _) = run_gadget(in0, in1);

        // Positions follow from the declared interface and the instance index.
        assert_eq!(
            composition.inputs(&public, 0, instance),
            Ok([Word64::new(in0), Word64::new(in1)].as_slice())
        );
        assert_eq!(
            composition.outputs(&public, 0, instance),
            Ok([Word64::new(out)].as_slice())
        );
    }
}

#[test]
fn a_substituted_public_output_is_rejected() {
    let composition = composition(2);
    let (public, witness) = honest_values(&composition);
    let key = WordProofKey::new(composition.clone()).expect("the composed key compiles");
    let scheme = commitment_scheme(key.trace_variables());
    let (commitment, proof) = prove(&key, &scheme, &public, &witness).expect("the statement holds");

    // Flipping one bit of the second instance's declared output breaks the proof.
    let mut tampered = public;
    let output = composition
        .interface_mut(&mut tampered, 0, 1)
        .expect("the instance exists");
    output[2] = Word64::new(output[2].get() ^ 1);
    assert!(verify(&key, &scheme, &commitment, &tampered, &proof).is_err());
}

// ---------------------------------------------------------------------------
// Several calls share one commitment and one claim pool
// ---------------------------------------------------------------------------

/// A second, differently shaped gadget: `out = in0 ^ (in1 >> 7)`, with one local.
///
/// ```text
/// public  in0 = slot 0, in1 = slot 1, out = slot 2
/// local   v   = slot 0
///
/// v = in0 ^ (in1 >> 7)
/// out = v & v
/// ```
fn other_gadget() -> Component<Word64> {
    let in0 = ValueIndex::public(0).expect("slot fits");
    let in1 = ValueIndex::public(1).expect("slot fits");
    let out = ValueIndex::public(2).expect("slot fits");
    let v = ValueIndex::witness(0).expect("slot fits");

    let shifted = ShiftedValue::single(
        in1,
        Shift::new(ShiftKind::LogicalRight, 7).expect("shift is in range"),
    );

    let body = ConstraintSystem::new(
        3,
        1,
        vec![ZeroConstraint::new(Operand::new(vec![
            ShiftedValue::plain(v),
            ShiftedValue::plain(in0),
            shifted,
        ]))],
        vec![AndConstraint::new(
            Operand::single(ShiftedValue::plain(v)),
            Operand::single(ShiftedValue::plain(v)),
            Operand::single(ShiftedValue::plain(out)),
        )],
        vec![],
    )
    .expect("the gadget addresses only its declared slots");

    Component::new(body, 2, 1).expect("two inputs and one output span the interface")
}

#[test]
fn two_components_prove_through_one_commitment_and_one_opening() {
    // Fixture state: three instances of one gadget beside five of another.
    let composition = Composition::new(vec![
        ComponentCall::new(gadget(), 3),
        ComponentCall::new(other_gadget(), 5),
    ])
    .expect("the fixture fits the compact address space");

    let mut public = vec![Word64::new(0); composition.public_len()];
    let mut witness = vec![Word64::new(0); composition.witness_len()];

    for instance in 0..3 {
        let (in0, in1) = instance_inputs(instance);
        let (out, locals) = run_gadget(in0, in1);
        let interface = composition
            .interface_mut(&mut public, 0, instance)
            .expect("the instance exists");
        interface[0] = Word64::new(in0);
        interface[1] = Word64::new(in1);
        interface[2] = Word64::new(out);
        let private = composition
            .locals_mut(&mut witness, 0, instance)
            .expect("the instance exists");
        private[0] = Word64::new(locals[0]);
        private[1] = Word64::new(locals[1]);
    }
    for instance in 0..5 {
        let (in0, in1) = instance_inputs(instance + 3);
        let v = in0 ^ (in1 >> 7);
        let interface = composition
            .interface_mut(&mut public, 1, instance)
            .expect("the instance exists");
        interface[0] = Word64::new(in0);
        interface[1] = Word64::new(in1);
        interface[2] = Word64::new(v);
        composition
            .locals_mut(&mut witness, 1, instance)
            .expect("the instance exists")[0] = Word64::new(v);
    }

    // Eight instances of two gadgets: 11 committed words, 24 public words.
    assert_eq!(composition.witness_len(), 11);
    assert_eq!(composition.public_len(), 24);
    assert_eq!(composition.relation_counts(), [8, 11, 0]);

    let flat = composition
        .lower()
        .expect("the lowered system is well formed");
    assert_eq!(flat.verify(&public, &witness), Ok(()));

    let key = WordProofKey::new(composition.clone()).expect("the composed key compiles");
    let scheme = commitment_scheme(key.trace_variables());
    let (commitment, proof) =
        prove(&key, &scheme, &public, &witness).expect("the composed statement holds");
    assert!(verify(&key, &scheme, &commitment, &public, &proof).is_ok());

    // One shared commitment: the lowered statement reaches the same transcript.
    let flat_key = WordProofKey::new(flat).expect("the flat key compiles");
    let (flat_commitment, flat_proof) =
        prove(&flat_key, &scheme, &public, &witness).expect("the flat statement holds");
    assert_eq!(commitment, flat_commitment);
    assert_eq!(bytes(&proof), bytes(&flat_proof));

    // One call's instance cannot stand in for the other call's.
    let mut crossed = witness.clone();
    let borrowed = composition
        .locals(&witness, 0, 0)
        .expect("the instance exists")[0];
    composition
        .locals_mut(&mut crossed, 1, 0)
        .expect("the instance exists")[0] = borrowed;
    let (commitment, proof) =
        prove(&key, &scheme, &public, &crossed).expect("a transcript is produced regardless");
    assert!(matches!(
        verify(&key, &scheme, &commitment, &public, &proof),
        Err(WordProofError::RelationClaim)
    ));
}

#[test]
fn a_call_with_no_live_instances_addresses_nothing() {
    // A dead call must not shift any live call's words or relations.
    let padded = Composition::new(vec![
        ComponentCall::new(other_gadget(), 0),
        ComponentCall::new(gadget(), 4),
    ])
    .expect("the fixture fits the compact address space");
    assert_eq!(padded.witness_base(1), Ok(0));

    let (public, witness) = honest_values(&composition(4));
    let key = WordProofKey::new(padded).expect("the composed key compiles");
    let scheme = commitment_scheme(key.trace_variables());
    let (commitment, proof) =
        prove(&key, &scheme, &public, &witness).expect("the composed statement holds");
    assert!(verify(&key, &scheme, &commitment, &public, &proof).is_ok());
}
