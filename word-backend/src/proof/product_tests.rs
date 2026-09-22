//! End-to-end tests of statements declaring full-width unsigned products.

use alloc::vec;
use alloc::vec::Vec;

use p3_binary_field::{BinaryChallenger, BinaryField128};
use p3_binary_pcs::{BinaryPcsConfig, BinaryPcsParams, BooleanPcs, BooleanPcsError, BooleanProof};
use p3_challenger::HashChallenger;
use p3_commit::Mmcs;
use p3_field::PrimeCharacteristicRing;
use p3_keccak::Keccak256Hash;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher};
use p3_word::{
    AndConstraint, ConstraintKind, ConstraintSystem, IntegerMulConstraint, Operand, Shift,
    ShiftKind, ShiftedValue, ValueIndex, VerificationError, Word, Word32, Word64, ZeroConstraint,
};
use proptest::prelude::*;

use super::{WordProof, WordProofError, WordProofKey};
use crate::{IntegerMulError, PackedWitness, PackedWord};

type EF = BinaryField128;
type MyHash = SerializingHasher<Keccak256Hash>;
type MyCompress = CompressionFunctionFromHasher<Keccak256Hash, 2, 32>;
type MyMmcs = MerkleTreeMmcs<EF, u8, MyHash, MyCompress, 2, 32>;
type Scheme = BooleanPcs<EF, MyMmcs, MyMmcs>;
type Challenger = BinaryChallenger<EF, HashChallenger<u8, Keccak256Hash, 32>>;
type SchemeError = BooleanPcsError<EF, <MyMmcs as Mmcs<EF>>::Error>;
type Commitment = <MyMmcs as Mmcs<EF>>::Commitment;
type Record = WordProof<EF, EF, BooleanProof<EF, MyMmcs, MyMmcs>>;
type Verdict = Result<(), WordProofError<SchemeError>>;
type Mutation = (&'static str, fn(&mut Record));

/// Base-two logarithm of the bits one committed element holds.
const ABSORBED: usize = 7;

fn challenger() -> Challenger {
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

fn public(position: usize) -> ValueIndex {
    ValueIndex::public(position).expect("test position fits")
}

fn committed(position: usize) -> ValueIndex {
    ValueIndex::witness(position).expect("test position fits")
}

fn plain<W: Word>(index: ValueIndex) -> Operand<W> {
    Operand::single(ShiftedValue::plain(index))
}

fn shifted<W: Word>(index: ValueIndex, kind: ShiftKind, amount: usize) -> ShiftedValue<W> {
    ShiftedValue::single(
        index,
        Shift::new(kind, amount).expect("test shift is in range"),
    )
}

/// One key, one commitment scheme, and one honest witness, proved once.
struct Fixture<W: PackedWord> {
    key: WordProofKey<W>,
    scheme: Scheme,
    public: Vec<W>,
    values: PackedWitness<W>,
}

impl<W: PackedWord> Fixture<W> {
    fn new(system: ConstraintSystem<W>, public: Vec<W>, words: &[W]) -> Self {
        let values = PackedWitness::new(&system, &public, words).expect("the shape matches");
        let key = WordProofKey::new(system).expect("a product statement compiles");
        let scheme = commitment_scheme(key.trace_variables());
        Self {
            key,
            scheme,
            public,
            values,
        }
    }

    fn prove(&self) -> (Commitment, Record) {
        self.key
            .prove::<EF, EF, _, _>(&self.scheme, &self.values, &mut challenger())
            .expect("a transcript is produced whether or not the statement holds")
    }

    fn check(&self, commitment: &Commitment, proof: &Record) -> Verdict {
        self.key.verify::<EF, EF, _, _>(
            &self.scheme,
            commitment,
            &self.public,
            proof,
            &mut challenger(),
        )
    }
}

/// A 64-bit statement mixing three products, five bitwise products, and one linear relation.
///
/// Five bitwise rows pad the shared cube to eight rows, while three products pad to four.
///
/// The product claims are therefore lifted into a larger cube before they are batched.
fn word64_statement(
    a: u64,
    b: u64,
    c: u64,
    p: u64,
) -> (ConstraintSystem<Word64>, Vec<Word64>, Vec<Word64>) {
    let products = vec![
        // a * b, all four operands plain.
        IntegerMulConstraint::new(
            plain(committed(0)),
            plain(committed(1)),
            plain(committed(2)),
            plain(committed(3)),
        ),
        // rotr(a, 7) * p, a shifted factor against a public one.
        IntegerMulConstraint::new(
            Operand::single(shifted(committed(0), ShiftKind::RotateRight, 7)),
            plain(public(0)),
            plain(committed(4)),
            plain(committed(5)),
        ),
        // (c ^ b) * c, a two-term factor.
        IntegerMulConstraint::new(
            Operand::new(vec![
                ShiftedValue::plain(committed(6)),
                ShiftedValue::plain(committed(1)),
            ]),
            plain(committed(6)),
            plain(committed(7)),
            plain(committed(8)),
        ),
    ];
    let ands = (0..5)
        .map(|amount| {
            AndConstraint::new(
                plain(committed(0)),
                Operand::single(shifted(committed(1), ShiftKind::LogicalLeft, amount)),
                plain(committed(9 + amount)),
            )
        })
        .collect();
    let linear = ZeroConstraint::new(Operand::new(vec![
        ShiftedValue::plain(committed(2)),
        ShiftedValue::plain(committed(14)),
    ]));
    let system = ConstraintSystem::new(1, 16, vec![linear], ands, products)
        .expect("the fixture addresses only declared words");

    // Native 128-bit arithmetic is the scalar reference for every limb.
    let wide = |x: u64, y: u64| {
        let product = u128::from(x) * u128::from(y);
        [product as u64, (product >> 64) as u64]
    };
    let [lo1, hi1] = wide(a, b);
    let [lo2, hi2] = wide(a.rotate_right(7), p);
    let [lo3, hi3] = wide(c ^ b, c);
    let mut words = vec![a, b, lo1, hi1, lo2, hi2, c, lo3, hi3];
    words.extend((0..5).map(|amount| a & (b << amount)));
    words.extend([lo1, 0]);
    (
        system,
        vec![Word64::new(p)],
        words.into_iter().map(Word64::new).collect(),
    )
}

/// A 32-bit statement mixing three products, one bitwise product, and one linear relation.
fn word32_statement(
    a: u32,
    b: u32,
    p: u32,
) -> (ConstraintSystem<Word32>, Vec<Word32>, Vec<Word32>) {
    let products = vec![
        IntegerMulConstraint::new(
            plain(committed(0)),
            plain(committed(1)),
            plain(committed(2)),
            plain(committed(3)),
        ),
        // (a >> 3) * p, a logical shift against a public factor.
        IntegerMulConstraint::new(
            Operand::single(shifted(committed(0), ShiftKind::LogicalRight, 3)),
            plain(public(0)),
            plain(committed(4)),
            plain(committed(5)),
        ),
        // b * b, one word read as both factors.
        IntegerMulConstraint::new(
            plain(committed(1)),
            plain(committed(1)),
            plain(committed(6)),
            plain(committed(7)),
        ),
    ];
    let and = AndConstraint::new(
        plain(committed(0)),
        plain(committed(1)),
        plain(committed(8)),
    );
    let linear = ZeroConstraint::new(Operand::new(vec![
        ShiftedValue::plain(committed(3)),
        ShiftedValue::plain(committed(9)),
    ]));
    let system = ConstraintSystem::new(1, 16, vec![linear], vec![and], products)
        .expect("the fixture addresses only declared words");

    let wide = |x: u32, y: u32| {
        let product = u64::from(x) * u64::from(y);
        [product as u32, (product >> 32) as u32]
    };
    let [lo1, hi1] = wide(a, b);
    let [lo2, hi2] = wide(a >> 3, p);
    let [lo3, hi3] = wide(b, b);
    let mut words = vec![a, b, lo1, hi1, lo2, hi2, lo3, hi3, a & b, hi1];
    words.resize(16, 0);
    (
        system,
        vec![Word32::new(p)],
        words.into_iter().map(Word32::new).collect(),
    )
}

/// Boundary factors: zero, one, the top bit alone, and the largest word.
const WORD64_EDGES: [u64; 4] = [0, 1, 1 << 63, u64::MAX];

/// The same boundaries at 32 bits.
const WORD32_EDGES: [u32; 4] = [0, 1, 1 << 31, u32::MAX];

fn word64_fixture(a: u64, b: u64, c: u64, p: u64) -> Fixture<Word64> {
    let (system, public, words) = word64_statement(a, b, c, p);
    // The scalar reference accepts the witness before anything is proved.
    assert_eq!(system.verify(&public, &words), Ok(()));
    Fixture::new(system, public, &words)
}

fn word32_fixture(a: u32, b: u32, p: u32) -> Fixture<Word32> {
    let (system, public, words) = word32_statement(a, b, p);
    assert_eq!(system.verify(&public, &words), Ok(()));
    Fixture::new(system, public, &words)
}

fn is_product_rejection(verdict: &Verdict) -> bool {
    matches!(verdict, Err(WordProofError::IntegerMul(_)))
}

#[test]
fn word64_products_at_every_boundary_prove_and_verify() {
    // Each proof carries three products, so four proofs meet every boundary as a factor.
    for (index, a) in WORD64_EDGES.into_iter().enumerate() {
        let b = WORD64_EDGES[3 - index];
        let c = WORD64_EDGES[(index + 1) % 4];
        let p = WORD64_EDGES[(index + 2) % 4];
        let fixture = word64_fixture(a, b, c, p);
        let (commitment, proof) = fixture.prove();
        assert!(
            fixture.check(&commitment, &proof).is_ok(),
            "a = {a:#x}, b = {b:#x}"
        );
    }

    // The largest square fills both limbs, and its low bit is one.
    let fixture = word64_fixture(u64::MAX, u64::MAX, u64::MAX, u64::MAX);
    let (commitment, proof) = fixture.prove();
    assert!(fixture.check(&commitment, &proof).is_ok());
}

#[test]
fn word32_products_at_every_boundary_prove_and_verify() {
    for a in WORD32_EDGES {
        for b in WORD32_EDGES {
            let fixture = word32_fixture(a, b, a ^ 0xDEAD_BEEF);
            let (commitment, proof) = fixture.prove();
            assert!(
                fixture.check(&commitment, &proof).is_ok(),
                "a = {a:#x}, b = {b:#x}"
            );
        }
    }
}

proptest! {
    // Every case is a complete proof, so a handful of cases already crosses every bit.
    #![proptest_config(ProptestConfig::with_cases(4))]

    #[test]
    fn random_word64_products_prove_and_verify(a: u64, b: u64, c: u64, p: u64) {
        let fixture = word64_fixture(a, b, c, p);
        let (commitment, proof) = fixture.prove();
        prop_assert!(fixture.check(&commitment, &proof).is_ok());
    }

    #[test]
    fn random_word32_products_prove_and_verify(a: u32, b: u32, p: u32) {
        let fixture = word32_fixture(a, b, p);
        let (commitment, proof) = fixture.prove();
        prop_assert!(fixture.check(&commitment, &proof).is_ok());
    }

    #[test]
    fn a_wrong_word64_limb_bit_is_rejected(a: u64, b: u64, slot in 2usize..4, bit in 0usize..64) {
        // Mutation: flip one bit of the first product's low or high limb.
        let (system, public, mut words) = word64_statement(a, b, 3, 5);
        words[slot] = Word64::new(words[slot].get() ^ (1 << bit));
        let fixture = Fixture::new(system, public, &words);
        let (commitment, proof) = fixture.prove();
        prop_assert!(fixture.check(&commitment, &proof).is_err());
    }

    #[test]
    fn a_wrong_word32_limb_bit_is_rejected(a: u32, b: u32, slot in 6usize..8, bit in 0usize..32) {
        // Mutation: flip one bit of the square's low or high limb.
        let (system, public, mut words) = word32_statement(a, b, 5);
        words[slot] = Word32::new(words[slot].get() ^ (1 << bit));
        let fixture = Fixture::new(system, public, &words);
        let (commitment, proof) = fixture.prove();
        prop_assert!(is_product_rejection(&fixture.check(&commitment, &proof)));
    }
}

#[test]
fn a_wrong_product_is_rejected_by_the_multiplication_reduction() {
    // Mutation: raise the high limb by one, which no bit-local relation notices.
    let (system, public, mut words) = word64_statement(0xFFFF_0000_1234_5678, 0x9ABC_DEF0, 1, 2);
    words[3] = Word64::new(words[3].get() + 1);
    assert_eq!(
        system.verify(&public, &words),
        Err(VerificationError::Unsatisfied {
            kind: ConstraintKind::IntegerMul,
            constraint: 0,
        })
    );

    let fixture = Fixture::new(system, public, &words);
    let (commitment, proof) = fixture.prove();
    let verdict = fixture.check(&commitment, &proof);
    assert!(is_product_rejection(&verdict), "got {verdict:?}");
}

#[test]
fn the_wraparound_passes_the_lift_and_is_caught_by_the_low_bit() {
    // Mutation: claim all-ones limbs for a zero product.
    //
    // 2^128 - 1 is the group order, so its lift equals the lift of zero.
    let (system, public, mut words) = word64_statement(0, 0x1234, 1, 2);
    words[2] = Word64::new(u64::MAX);
    words[3] = Word64::new(u64::MAX);
    words[14] = Word64::new(u64::MAX);
    assert!(system.verify(&public, &words).is_err());

    // The multiplication reduction accepts, and only the low-bit relation refuses.
    let fixture = Fixture::new(system, public, &words);
    let (commitment, proof) = fixture.prove();
    assert!(matches!(
        fixture.check(&commitment, &proof),
        Err(WordProofError::RelationClaim)
    ));
}

#[test]
fn every_multiplication_record_value_is_load_bearing() {
    let fixture = word64_fixture(0x0123_4567_89AB_CDEF, 0xFEDC_BA98_7654_3210, 7, 9);
    let (commitment, proof) = fixture.prove();

    // Each mutation perturbs one value the multiplication record carries.
    let mutations: [Mutation; 9] = [
        ("root", |proof| {
            proof.integer_mul.as_mut().unwrap().root += EF::ONE;
        }),
        ("factor leaf value", |proof| {
            proof.integer_mul.as_mut().unwrap().factor.values[1] += EF::ONE;
        }),
        ("result leaf value", |proof| {
            proof.integer_mul.as_mut().unwrap().result.values[0] += EF::ONE;
        }),
        ("factor leaf round", |proof| {
            proof.integer_mul.as_mut().unwrap().factor.leaf.round_polys[0][0] += EF::ONE;
        }),
        ("result leaf sum", |proof| {
            proof.integer_mul.as_mut().unwrap().result.leaf.claimed_sum += EF::ONE;
        }),
        ("first factor half", |proof| {
            proof.integer_mul.as_mut().unwrap().factor.layers[0].halves[0] += EF::ONE;
        }),
        ("last result half", |proof| {
            let layers = &mut proof.integer_mul.as_mut().unwrap().result.layers;
            layers.last_mut().unwrap().halves[1] += EF::ONE;
        }),
        ("last factor layer round", |proof| {
            let layers = &mut proof.integer_mul.as_mut().unwrap().factor.layers;
            layers.last_mut().unwrap().sumcheck.round_polys[0][2] += EF::ONE;
        }),
        ("first result layer sum", |proof| {
            let layers = &mut proof.integer_mul.as_mut().unwrap().result.layers;
            layers[0].sumcheck.claimed_sum += EF::ONE;
        }),
    ];

    for (name, mutate) in mutations {
        let mut tampered = proof.clone();
        mutate(&mut tampered);
        let verdict = fixture.check(&commitment, &tampered);
        assert!(
            is_product_rejection(&verdict),
            "{name} must be rejected, got {verdict:?}"
        );
    }

    // The untouched record still verifies, so each rejection is the mutation's alone.
    assert!(fixture.check(&commitment, &proof).is_ok());
}

#[test]
fn every_product_operand_evaluation_is_load_bearing() {
    // Mutation: perturb each of the four product evaluations the vanishing check ends on.
    let fixture = word32_fixture(0x8000_0001, 0x7FFF_FFFF, 0x1357_9BDF);
    let (commitment, proof) = fixture.prove();
    for slot in 4..8 {
        let mut tampered = proof.clone();
        tampered.operands[slot] += EF::ONE;
        assert!(
            matches!(
                fixture.check(&commitment, &tampered),
                Err(WordProofError::RelationClaim)
            ),
            "slot {slot}"
        );
    }

    // Swapping the two factor claims keeps their product, but not their claim points.
    let mut swapped = proof.clone();
    swapped.operands.swap(4, 5);
    assert_ne!(swapped.operands, proof.operands);
    assert!(matches!(
        fixture.check(&commitment, &swapped),
        Err(WordProofError::RelationClaim)
    ));

    // A batched sum the product claims do not fix is refused before any round is read.
    let mut shifted_sum = proof;
    shifted_sum.zerocheck.claimed_sum += EF::ONE;
    assert!(matches!(
        fixture.check(&commitment, &shifted_sum),
        Err(WordProofError::RelationSum)
    ));
}

#[test]
fn a_multiplication_record_must_match_the_statement() {
    let fixture = word32_fixture(3, 5, 7);
    let (commitment, proof) = fixture.prove();

    // Dropping the record cannot hide the products.
    let mut dropped = proof.clone();
    dropped.integer_mul = None;
    assert!(matches!(
        fixture.check(&commitment, &dropped),
        Err(WordProofError::ProductRecord)
    ));

    // A record with one layer too few is refused before its transcript is replayed.
    let mut shallow = proof;
    shallow.integer_mul.as_mut().unwrap().factor.layers.pop();
    assert!(matches!(
        fixture.check(&commitment, &shallow),
        Err(WordProofError::IntegerMul(IntegerMulError::TreeDepth {
            tree: "factor",
            expected: 10,
            actual: 9,
        }))
    ));
}

#[test]
fn a_record_attached_to_a_product_free_statement_is_refused() {
    // Fixture state: one linear relation, and a record borrowed from a product statement.
    let value = ValueIndex::witness(0).expect("test position fits");
    let linear = ZeroConstraint::new(plain::<Word32>(value));
    let system = ConstraintSystem::new(0, 16, vec![linear], vec![], vec![]).unwrap();
    let fixture = Fixture::new(system, vec![], &[Word32::new(0); 16]);
    let (commitment, mut proof) = fixture.prove();
    let (_, donor) = word32_fixture(3, 5, 7).prove();
    proof.integer_mul = donor.integer_mul;

    assert!(matches!(
        fixture.check(&commitment, &proof),
        Err(WordProofError::ProductRecord)
    ));
}
