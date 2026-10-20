//! End-to-end completeness and soundness tests of the reduction.

use alloc::vec;
use alloc::vec::Vec;

use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_binary_field::{BinaryChallenger, BinaryField128};
use p3_challenger::{CanSample, DuplexChallenger, HashChallenger};
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, PrimeCharacteristicRing};
use p3_keccak::Keccak256Hash;
use p3_multilinear_util::poly::Poly;
use proptest::prelude::*;
use rand::SeedableRng;
use rand::rngs::SmallRng;

use crate::{
    BitShift, BlockDiagonal, Ccs, CcsProof, Constraint, Operator, RowSelector, SparseMatrix, Term,
    TraceColumn, VerificationError, prove, verify,
};

/// Base field.
type F = BabyBear;
/// Challenge field.
type EF = BinomialExtensionField<F, 4>;
/// Fiat-Shamir challenger.
type Challenger = DuplexChallenger<F, Poseidon2BabyBear<16>, 16, 8>;

/// Named proof mutation and its expected rejection.
type Tamper = (&'static str, fn(&mut CcsProof<F, EF>), VerificationError);

fn challenger() -> Challenger {
    // Fixed permutation: both sides start from the same sponge.
    let mut rng = SmallRng::seed_from_u64(0xC0FFEE);
    Challenger::new(Poseidon2BabyBear::new_from_rng_128(&mut rng))
}

/// Prove, verify, open the claim, and check both sponges end in step.
fn round_trip(ccs: &Ccs<F>, z: &[F], pow_bits: usize) -> CcsProof<F, EF> {
    let mut prover_challenger = challenger();
    let (proof, prover_claim) = prove::<F, EF, _, _>(ccs, z, pow_bits, &mut prover_challenger);

    let mut verifier_challenger = challenger();
    let verifier_claim =
        verify(ccs, &proof, pow_bits, &mut verifier_challenger).expect("an honest proof verifies");

    // Same claim on both sides.
    assert_eq!(prover_claim, verifier_claim);

    // The claim is the witness extension at its point.
    let opened = Poly::new(z.to_vec()).eval_base(&verifier_claim.point);
    assert_eq!(opened, verifier_claim.value);

    // Sponges stay in step.
    assert_eq!(
        CanSample::<F>::sample(&mut prover_challenger),
        CanSample::<F>::sample(&mut verifier_challenger)
    );
    proof
}

fn check_proof(ccs: &Ccs<F>, proof: &CcsProof<F, EF>) -> Result<(), VerificationError> {
    // Fresh sponge, claim discarded.
    verify(ccs, proof, 0, &mut challenger()).map(drop)
}

/// Fibonacci AIR: a_next = b and b_next = a + b on every transition.
fn fibonacci<K: Field>(log_height: usize) -> Ccs<K> {
    let cell = |column, offset| Operator::from(TraceColumn::new(log_height, 1, column, offset));

    // Matrices: a, b, a_next, b_next.
    let operators = vec![cell(0, 0), cell(1, 0), cell(0, 1), cell(1, 1)];
    let rules = vec![
        Constraint::new(vec![Term::new(K::ONE, [2]), Term::new(K::NEG_ONE, [1])]),
        Constraint::new(vec![
            Term::new(K::ONE, [3]),
            Term::new(K::NEG_ONE, [0]),
            Term::new(K::NEG_ONE, [1]),
        ]),
    ];
    Ccs::new(
        operators,
        rules
            .into_iter()
            .map(|k| k.on(RowSelector::Transition))
            .collect(),
    )
    .unwrap()
}

/// Column-major Fibonacci trace.
fn fibonacci_trace<K: Field>(log_height: usize) -> Vec<K> {
    let height = 1 << log_height;
    let (mut a, mut b) = (vec![K::ONE], vec![K::ONE]);

    // (a, b) -> (b, a + b).
    for row in 1..height {
        a.push(b[row - 1]);
        b.push(a[row - 1] + b[row - 1]);
    }
    a.extend(b);
    a
}

#[test]
fn fibonacci_air_round_trips() {
    // Fixture state: 16 rows, two linear rules, degree 2 with the selector.
    let ccs = fibonacci::<F>(4);
    let z = fibonacci_trace::<F>(4);
    assert_eq!(ccs.check(&z), Ok(()));
    round_trip(&ccs, &z, 0);
}

#[test]
fn fibonacci_air_round_trips_with_grinding() {
    // Fixture state: grinding in every sumcheck round.
    let ccs = fibonacci::<F>(3);
    let z = fibonacci_trace::<F>(3);
    let proof = round_trip(&ccs, &z, 3);

    // One PoW witness per round.
    assert_eq!(proof.zerocheck.pow_witnesses.len(), 3);
    assert_eq!(proof.linear.pow_witnesses.len(), 4);
}

#[test]
fn fibonacci_air_round_trips_in_characteristic_two() {
    // Invariant: nothing assumes odd characteristic.
    type B = BinaryField128;
    type Ch = BinaryChallenger<B, HashChallenger<u8, Keccak256Hash, 32>>;
    let fresh = || Ch::from_hasher(Vec::new(), Keccak256Hash);

    let ccs = fibonacci::<B>(4);
    let z = fibonacci_trace::<B>(4);
    assert_eq!(ccs.check(&z), Ok(()));

    let (proof, claim) = prove::<B, B, _, _>(&ccs, &z, 0, &mut fresh());
    let verified = verify(&ccs, &proof, 0, &mut fresh()).unwrap();
    assert_eq!(claim, verified);
    assert_eq!(Poly::new(z).eval_base(&verified.point), verified.value);
}

/// Copies of the R1CS x * y = 12, x + y = 7, each over (1, x, y, 0).
fn copied_product_and_sum(log_copies: usize) -> Ccs<F> {
    let block = |entries| BlockDiagonal::new(log_copies, SparseMatrix::new(1, 2, entries)).into();
    let a = block(vec![(0, 1, F::ONE), (1, 1, F::ONE), (1, 2, F::ONE)]);
    let b = block(vec![(0, 2, F::ONE), (1, 0, F::ONE)]);
    let c = block(vec![(0, 0, F::from_u32(12)), (1, 0, F::from_u32(7))]);
    Ccs::new(
        vec![a, b, c],
        vec![Constraint::new(vec![
            Term::new(F::ONE, [0, 1]),
            Term::new(F::NEG_ONE, [2]),
        ])],
    )
    .unwrap()
}

#[test]
fn copied_r1cs_round_trips() {
    // Fixture state: 32 copies alternating (3, 4) and (4, 3).
    let ccs = copied_product_and_sum(5);
    let z: Vec<F> = (0..32)
        .flat_map(|copy| {
            if copy % 2 == 0 {
                [1, 3, 4, 0]
            } else {
                [1, 4, 3, 0]
            }
        })
        .map(F::from_u32)
        .collect();
    round_trip(&ccs, &z, 0);
}

/// Bitwise w = x AND rotr_3(y) on 8-bit words, with Boolean x and y bits.
fn rotated_and(log_records: usize) -> Ccs<F> {
    // Record columns: x, y, w, padding, 8 bits each.
    let pick = |base: usize| {
        let entries = (0..8).map(|bit| (bit, base + bit, F::ONE)).collect();
        BlockDiagonal::new(log_records, SparseMatrix::new(3, 5, entries)).into()
    };
    let rotated_y = {
        let rotate = BlockDiagonal::<F>::bit_shift(0, 3, BitShift::RotateRight(3));
        let entries = rotate
            .block()
            .entries()
            .iter()
            .map(|&(bit, src, c)| (bit, 8 + src, c))
            .collect();
        BlockDiagonal::new(log_records, SparseMatrix::new(3, 5, entries)).into()
    };

    // Matrices: x, y, w, rotr_3(y).
    let operators = vec![pick(0), pick(8), pick(16), rotated_y];
    let constraints = vec![
        // x o rotr_3(y) - w = 0.
        Constraint::new(vec![Term::new(F::ONE, [0, 3]), Term::new(F::NEG_ONE, [2])]),
        // x o x - x = 0.
        Constraint::new(vec![Term::new(F::ONE, [0, 0]), Term::new(F::NEG_ONE, [0])]),
        // y o y - y = 0.
        Constraint::new(vec![Term::new(F::ONE, [1, 1]), Term::new(F::NEG_ONE, [1])]),
    ];
    Ccs::new(operators, constraints).unwrap()
}

/// One rotated-AND record, bit zero first.
fn rotated_and_record(x: u8, y: u8) -> [F; 32] {
    let w = x & y.rotate_right(3);
    let bits = |v: u8| (0..8).map(move |i| F::from_bool((v >> i) & 1 == 1));
    let mut record = [F::ZERO; 32];
    for (slot, bit) in record.iter_mut().zip(bits(x).chain(bits(y)).chain(bits(w))) {
        *slot = bit;
    }
    record
}

#[test]
fn rotated_and_round_trips() {
    // Fixture state: 16 records of varied words.
    let ccs = rotated_and(4);
    let z: Vec<F> = (0..16u8)
        .flat_map(|i| rotated_and_record(i.wrapping_mul(37), (i ^ 0xa5).wrapping_mul(91)))
        .collect();
    assert_eq!(ccs.check(&z), Ok(()));
    round_trip(&ccs, &z, 0);
}

#[test]
fn a_cubic_constraint_with_weights_round_trips() {
    // Fixture state: 3 * a * b * c_next - 3 * d = 0 on transitions, degree 4.
    let cell = |column, offset| Operator::from(TraceColumn::new(3, 2, column, offset));
    let ccs = Ccs::new(
        vec![cell(0, 0), cell(1, 0), cell(2, 1), cell(3, 0)],
        vec![
            Constraint::new(vec![
                Term::new(F::from_u32(3), [0, 1, 2]),
                Term::new(-F::from_u32(3), [3]),
            ])
            .on(RowSelector::Transition),
        ],
    )
    .unwrap();
    assert_eq!(ccs.degree(), 4);

    // Small a, b, c, then d from the rule.
    let a: Vec<F> = (0..8).map(|i| F::from_u32(i + 2)).collect();
    let b: Vec<F> = (0..8).map(|i| F::from_u32(3 * i + 1)).collect();
    let c: Vec<F> = (0..8).map(|i| F::from_u32(i * i + 5)).collect();
    let d: Vec<F> = (0..8)
        .map(|i| {
            if i + 1 < 8 {
                a[i] * b[i] * c[i + 1]
            } else {
                F::from_u32(99)
            }
        })
        .collect();
    let z: Vec<F> = [a, b, c, d].concat();
    round_trip(&ccs, &z, 0);
}

#[test]
fn boundary_selectors_bind_exactly_their_row() {
    // Fixture state: random columns a and b agreeing only on rows 0 and 15.
    //
    // a = b is selected on the first and last rows only.
    let mut rng = SmallRng::seed_from_u64(7);
    let mut a: Vec<F> = (0..16).map(|_| rand::RngExt::random(&mut rng)).collect();
    let b: Vec<F> = (0..16).map(|_| rand::RngExt::random(&mut rng)).collect();
    a[0] = b[0];
    a[15] = b[15];

    let cell = |column| Operator::from(TraceColumn::new(4, 1, column, 0));
    let equal = Constraint::new(vec![Term::new(F::ONE, [0]), Term::new(F::NEG_ONE, [1])]);
    let ccs = Ccs::new(
        vec![cell(0), cell(1)],
        vec![
            equal.clone().on(RowSelector::First),
            equal.on(RowSelector::Last),
        ],
    )
    .unwrap();
    let z: Vec<F> = [a, b].concat();
    round_trip(&ccs, &z, 0);

    // Mutation: break row 0, then row 15; each selector catches its own row.
    for row in [0, 15] {
        let mut bad = z.clone();
        bad[row] += F::ONE;
        assert!(ccs.check(&bad).is_err());
        let proof = prove::<F, EF, _, _>(&ccs, &bad, 0, &mut challenger()).0;
        assert_eq!(
            check_proof(&ccs, &proof),
            Err(VerificationError::ZerocheckMismatch),
            "row {row}"
        );
    }

    // Mutation: a middle row is unconstrained.
    let mut free = z;
    free[7] += F::ONE;
    assert_eq!(ccs.check(&free), Ok(()));
    round_trip(&ccs, &free, 0);
}

#[test]
fn a_single_row_system_round_trips() {
    // Invariant: zero zerocheck rounds still reduce correctly.
    //
    // Fixture state: one row, x * y = 12 with witness (1, 3, 4, 0).
    let a = SparseMatrix::new(0, 2, vec![(0, 1, F::ONE)]);
    let b = SparseMatrix::new(0, 2, vec![(0, 2, F::ONE)]);
    let c = SparseMatrix::new(0, 2, vec![(0, 0, F::from_u32(12))]);
    let ccs = Ccs::new(
        vec![a.into(), b.into(), c.into()],
        vec![Constraint::new(vec![
            Term::new(F::ONE, [0, 1]),
            Term::new(F::NEG_ONE, [2]),
        ])],
    )
    .unwrap();
    let proof = round_trip(&ccs, &[1, 3, 4, 0].map(F::from_u32), 0);
    assert_eq!(proof.zerocheck.num_rounds(), 0);
}

#[test]
fn the_bare_reduction_proves_a_homogeneous_relation() {
    // Invariant: with no public cell, a zero constant slot satisfies every R1CS row.
    //
    // Fixture state: x * y = 12, x + y = 7 over (one, x, y, 0), with one = 0, x = 0, y = 9.
    //
    // The committed proof pins the constant; the bare reduction cannot.
    let ccs = copied_product_and_sum(0);
    let bogus = [0, 0, 9, 0].map(F::from_u32);
    assert_eq!(ccs.check(&bogus), Ok(()));
    round_trip(&ccs, &bogus, 0);
}

#[test]
fn a_wrong_witness_is_rejected() {
    // Mutation: b[5] += 1 breaks rows 4 and 5.
    let ccs = fibonacci::<F>(4);
    let mut z = fibonacci_trace::<F>(4);
    z[16 + 5] += F::ONE;

    // The zerocheck sum is no longer zero.
    let proof = prove::<F, EF, _, _>(&ccs, &z, 0, &mut challenger()).0;
    assert_eq!(
        check_proof(&ccs, &proof),
        Err(VerificationError::ZerocheckMismatch)
    );
}

#[test]
fn a_wrong_last_row_is_rejected_as_the_next_row_of_the_row_before() {
    // Invariant: the last row is still read as the next row of the row before.
    //
    // Mutation: change a last-row cell.
    let ccs = fibonacci::<F>(3);
    let mut z = fibonacci_trace::<F>(3);
    z[7] += F::ONE;
    assert!(ccs.check(&z).is_err());
    let proof = prove::<F, EF, _, _>(&ccs, &z, 0, &mut challenger()).0;
    assert_eq!(
        check_proof(&ccs, &proof),
        Err(VerificationError::ZerocheckMismatch)
    );
}

#[test]
fn every_tampered_field_is_rejected() {
    // Fixture state: one honest Fibonacci proof.
    //
    // Mutation: one per proof field.
    let ccs = fibonacci::<F>(3);
    let honest = round_trip(&ccs, &fibonacci_trace::<F>(3), 0);

    let cases: [Tamper; 6] = [
        (
            "zerocheck claims a nonzero sum",
            |p| p.zerocheck.claimed_sum = EF::ONE,
            VerificationError::NonzeroZerocheckClaim,
        ),
        (
            "a zerocheck round moves",
            |p| p.zerocheck.round_polys[1][0] += EF::ONE,
            VerificationError::ZerocheckMismatch,
        ),
        (
            "a matrix output moves",
            |p| p.operand_evals[2] += EF::ONE,
            VerificationError::ZerocheckMismatch,
        ),
        (
            "a matrix output goes missing",
            |p| p.operand_evals.truncate(3),
            VerificationError::OperandCount {
                expected: 4,
                actual: 3,
            },
        ),
        (
            "the linear sumcheck starts elsewhere",
            |p| p.linear.claimed_sum += EF::ONE,
            VerificationError::LinearClaimMismatch,
        ),
        (
            "the witness value moves",
            |p| p.witness_eval += EF::ONE,
            VerificationError::LinearMismatch,
        ),
    ];

    for (what, mutate, expected) in cases {
        let mut proof = honest.clone();
        mutate(&mut proof);
        assert_eq!(check_proof(&ccs, &proof), Err(expected), "{what}");
    }
}

#[test]
fn a_proof_does_not_transfer_to_another_statement() {
    // Invariant: the statement encoding is bound into the seed.
    //
    // Mutation: b_next - a - 2b instead of b_next - a - b.
    let ccs = fibonacci::<F>(3);
    let proof = round_trip(&ccs, &fibonacci_trace::<F>(3), 0);

    let cell = |column, offset| Operator::from(TraceColumn::new(3, 1, column, offset));
    let other = Ccs::new(
        vec![cell(0, 0), cell(1, 0), cell(0, 1), cell(1, 1)],
        vec![
            Constraint::new(vec![Term::new(F::ONE, [2]), Term::new(F::NEG_ONE, [1])]),
            Constraint::new(vec![
                Term::new(F::ONE, [3]),
                Term::new(F::NEG_ONE, [0]),
                Term::new(-F::TWO, [1]),
            ]),
        ]
        .into_iter()
        .map(|k| k.on(RowSelector::Transition))
        .collect(),
    )
    .unwrap();
    assert!(check_proof(&other, &proof).is_err());
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(24))]

    #[test]
    fn random_product_traces_round_trip_and_reject_any_flipped_cell(
        a in prop::collection::vec(any::<u32>(), 32),
        b in prop::collection::vec(any::<u32>(), 32),
        offset in 0usize..4,
        flip in 0usize..128,
    ) {
        // Fixture state: 32 rows with c = a * b_{+offset}, zero where b runs out.
        let a: Vec<F> = a.into_iter().map(F::from_u32).collect();
        let b: Vec<F> = b.into_iter().map(F::from_u32).collect();
        let c: Vec<F> = (0..32)
            .map(|i| if i + offset < 32 { a[i] * b[i + offset] } else { F::ZERO })
            .collect();
        let d = vec![F::ZERO; 32];
        let mut z: Vec<F> = [a, b, c, d].concat();

        let cell = |column, offset| Operator::from(TraceColumn::new(5, 2, column, offset));
        let ccs = Ccs::new(
            vec![cell(0, 0), cell(1, offset), cell(2, 0)],
            vec![Constraint::new(vec![
                Term::new(F::ONE, [0, 1]),
                Term::new(F::NEG_ONE, [2]),
            ])],
        )
        .unwrap();

        // Completeness.
        round_trip(&ccs, &z, 0);

        // Mutation: flip one cell; check soundness only if the trace breaks.
        z[flip] += F::ONE;
        if ccs.check(&z).is_err() {
            let proof = prove::<F, EF, _, _>(&ccs, &z, 0, &mut challenger()).0;
            prop_assert!(check_proof(&ccs, &proof).is_err());
        }
    }
}
