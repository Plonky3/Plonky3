//! A binary zerocheck end to end, from the packed witness to one opening point.

use p3_binary_field::{BinaryChallenger, BinaryField8, BinaryField128, TowerLevel};
use p3_challenger::{CanObserve, HashChallenger};
use p3_field::PrimeCharacteristicRing;
use p3_keccak::Keccak256Hash;
use p3_multilinear_util::poly::Poly;
use p3_sumcheck::univariate_skip::{
    BinaryZerocheck, CHUNK_BITS, Conjunction, ZerocheckError, ZerocheckProof,
};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

/// The subspace the skip round runs over lives in a byte field.
type F = BinaryField8;

/// Challenges and every value the rounds carry live in the 128-bit field above it.
type EF = BinaryField128;

/// Keccak-backed Fiat-Shamir, the same instantiation the binary sumcheck tests use.
type Challenger = BinaryChallenger<EF, HashChallenger<u8, Keccak256Hash, 32>>;

/// Variables the skip round binds in one go.
const LOG_SKIP: usize = 6;

/// Total variables, rows and skipped together.
const LOG_HEIGHT: usize = 10;

const fn fresh_challenger() -> Challenger {
    Challenger::from_hasher(Vec::new(), Keccak256Hash)
}

/// One edit applied to a finished proof, to check the verifier refuses it.
type Mutation = fn(&mut ZerocheckProof<EF>);

/// A bit-valued witness for the conjunction constraint, packed row by row.
struct Witness {
    /// The three operands, in the order the constraint reads them.
    operands: [Vec<u8>; 3],
}

impl Witness {
    /// Draw a witness whose cells all satisfy the constraint.
    fn random(rng: &mut SmallRng, num_rows: usize, row_bytes: usize) -> Self {
        // Two free operands, and the third pinned to their bitwise conjunction.
        let a = (0..num_rows * row_bytes)
            .map(|_| rng.random::<u8>())
            .collect::<Vec<_>>();
        let b = (0..num_rows * row_bytes)
            .map(|_| rng.random::<u8>())
            .collect::<Vec<_>>();
        let c = a.iter().zip(&b).map(|(&x, &y)| x & y).collect::<Vec<_>>();
        Self {
            operands: [a, b, c],
        }
    }

    /// The packed operands, as the prover takes them.
    fn packed(&self) -> [&[u8]; 3] {
        [&self.operands[0], &self.operands[1], &self.operands[2]]
    }

    /// One operand read as a multilinear over the whole hypercube.
    ///
    /// This stands in for a commitment opening.
    ///
    /// A real verifier would read these from the commitment, never from the witness.
    fn multilinear(&self, operand: usize) -> Poly<EF> {
        let packed = &self.operands[operand];
        Poly::new(
            (0..packed.len() * CHUNK_BITS)
                .map(|index| {
                    let bit = (packed[index / CHUNK_BITS] >> (index % CHUNK_BITS)) & 1;
                    if bit == 1 { EF::ONE } else { EF::ZERO }
                })
                .collect::<Vec<_>>(),
        )
    }
}

/// The zerocheck every test runs.
fn zerocheck(pow_bits: usize) -> BinaryZerocheck<F, Conjunction> {
    BinaryZerocheck::new(LOG_SKIP, Conjunction, pow_bits).unwrap()
}

/// Build a witness of the configured shape.
fn fixture(seed: u64, pow_bits: usize) -> (BinaryZerocheck<F, Conjunction>, Witness) {
    let mut rng = SmallRng::seed_from_u64(seed);
    let check = zerocheck(pow_bits);
    let num_rows = 1 << (LOG_HEIGHT - LOG_SKIP);
    let witness = Witness::random(&mut rng, num_rows, check.round().row_bytes());
    (check, witness)
}

/// Verify a proof and discharge its claim the way a commitment would.
///
/// Both halves report, so a test can tell a refused proof from a claim no opening answers.
fn verify_and_discharge(
    check: &BinaryZerocheck<F, Conjunction>,
    witness: &Witness,
    proof: &ZerocheckProof<EF>,
) -> Result<(), ZerocheckError> {
    let mut challenger = fresh_challenger();
    let claim = check.verify::<EF, _>(proof, LOG_HEIGHT, &mut challenger)?;

    // The commitment would open each operand at the claimed point.
    //
    // The claim owns the recombination, so a caller cannot batch them differently.
    //
    // Nor can a caller silently leave the comparison out.
    let openings = (0..3)
        .map(|operand| witness.multilinear(operand).eval_base(&claim.point))
        .collect::<Vec<_>>();

    claim.discharge(&openings)
}

#[test]
fn a_satisfied_constraint_ends_on_a_point_the_commitment_confirms() {
    // Fixture state: 10 variables, 6 skipped in one round, 4 residual rounds.
    //
    //     packed witness -> skip round -> residual rounds -> opening reduction -> point
    //
    // Invariant: the operands really take the claimed value at the point the run ends on.
    //
    // That is the whole chain closing.
    let (check, witness) = fixture(0x21C, 0);
    let mut challenger = fresh_challenger();
    let (proof, prover_claim) =
        check.prove::<EF, _>(&witness.packed(), LOG_HEIGHT, &mut challenger);

    verify_and_discharge(&check, &witness, &proof).unwrap();

    // Both sides reach the same point and the same value, with nothing passed between them.
    let mut challenger = fresh_challenger();
    let verifier_claim = check
        .verify::<EF, _>(&proof, LOG_HEIGHT, &mut challenger)
        .unwrap();
    assert_eq!(prover_claim.point, verifier_claim.point);
    assert_eq!(prover_claim.value, verifier_claim.value);
    assert_eq!(prover_claim.gamma, verifier_claim.gamma);
}

#[test]
fn the_claimed_point_covers_every_variable() {
    // The opening point has to name the whole hypercube, residual variables first.
    //
    //     4 residual + 6 skipped = 10
    //
    // A point of the wrong width would open a different polynomial.
    let (check, witness) = fixture(0x111, 0);
    let mut challenger = fresh_challenger();
    let (_, claim) = check.prove::<EF, _>(&witness.packed(), LOG_HEIGHT, &mut challenger);

    assert_eq!(claim.point.num_variables(), LOG_HEIGHT);
}

#[test]
fn grinding_guards_every_challenge_end_to_end() {
    // Fixture state: 4 bits of difficulty, cheap enough for a test.
    let (check, witness) = fixture(0x6D1, 4);
    let mut challenger = fresh_challenger();
    let (proof, _) = check.prove::<EF, _>(&witness.packed(), LOG_HEIGHT, &mut challenger);

    assert!(proof.skip_pow.is_some());
    verify_and_discharge(&check, &witness, &proof).unwrap();
}

#[test]
fn a_broken_constraint_is_rejected() {
    // Mutation: flip one bit of the claimed conjunction.
    //
    // The round polynomial stops vanishing on the skipped subspace.
    //
    // The verifier's reconstruction still does, so the two disagree at the challenge.
    let (check, mut witness) = fixture(0xBAD, 0);
    witness.operands[2][0] ^= 1;

    let mut challenger = fresh_challenger();
    let (proof, _) = check.prove::<EF, _>(&witness.packed(), LOG_HEIGHT, &mut challenger);

    // Either the replay refuses it, or the claim does not match what the operands open to.
    assert!(verify_and_discharge(&check, &witness, &proof).is_err());
}

#[test]
fn tampered_operand_blends_are_rejected() {
    // Mutation: change one blended value the proof carries.
    //
    // These cross the wire because the residual rounds leave one equation in three unknowns.
    //
    // That equation is what checks them, rather than the binding alone.
    let (check, witness) = fixture(0xB1E, 0);
    let mut challenger = fresh_challenger();
    let (mut proof, _) = check.prove::<EF, _>(&witness.packed(), LOG_HEIGHT, &mut challenger);

    proof.blends[0] += EF::ONE;
    assert_eq!(
        verify_and_discharge(&check, &witness, &proof).unwrap_err(),
        ZerocheckError::BlendConstraintMismatch
    );
}

#[test]
fn a_tampered_round_message_is_rejected() {
    // The message is bound before the skip challenge is drawn.
    //
    // A tamper moves the challenge, and the proof no longer answers what gets asked.
    let (check, witness) = fixture(0x7A3, 0);
    let mut challenger = fresh_challenger();
    let (mut proof, _) = check.prove::<EF, _>(&witness.packed(), LOG_HEIGHT, &mut challenger);

    proof.message[7] += EF::ONE;
    assert!(verify_and_discharge(&check, &witness, &proof).is_err());
}

#[test]
fn a_message_of_the_wrong_width_is_rejected() {
    // The described step declares its width.
    //
    // The replay therefore refuses the message rather than reading a shorter polynomial.
    let (check, witness) = fixture(0x9E1, 0);
    let mut challenger = fresh_challenger();
    let (mut proof, _) = check.prove::<EF, _>(&witness.packed(), LOG_HEIGHT, &mut challenger);

    proof.message.pop();
    assert!(verify_and_discharge(&check, &witness, &proof).is_err());
}

#[test]
fn a_malformed_residual_proof_is_rejected_rather_than_panicking() {
    // The transcript driver panics on drop if it is left unfinalized.
    //
    // That check is live in release builds whenever panics unwind.
    //
    // Each mutation fails a delegated replay at a different point.
    //
    // Every one of them has to come back as a rejection.
    let (check, witness) = fixture(0xF00, 0);
    let mut challenger = fresh_challenger();
    let (honest, _) = check.prove::<EF, _>(&witness.packed(), LOG_HEIGHT, &mut challenger);

    let mutations: [(&str, Mutation); 5] = [
        ("one residual round popped", |proof| {
            proof.residual.round_polys.pop();
        }),
        ("every residual round dropped", |proof| {
            proof.residual.round_polys.clear();
        }),
        ("one residual round too wide", |proof| {
            proof.residual.round_polys[0].push(EF::ONE);
        }),
        ("one opening round popped", |proof| {
            proof.opening.round_polys.pop();
        }),
        ("a grinding witness at zero difficulty", |proof| {
            proof.skip_pow = Some(EF::ONE);
        }),
    ];

    for (what, mutate) in mutations {
        let mut proof = honest.clone();
        mutate(&mut proof);
        assert!(
            verify_and_discharge(&check, &witness, &proof).is_err(),
            "must reject: {what}"
        );
    }
}

#[test]
fn a_proof_replayed_under_a_different_height_is_rejected() {
    // The height reaches the seed through the residual width.
    //
    // Both sides therefore diverge from the very first draw.
    let (check, witness) = fixture(0x4B2, 0);
    let mut challenger = fresh_challenger();
    let (proof, _) = check.prove::<EF, _>(&witness.packed(), LOG_HEIGHT, &mut challenger);

    let mut challenger = fresh_challenger();
    let outcome = check.verify::<EF, _>(&proof, LOG_HEIGHT + 1, &mut challenger);
    assert!(outcome.is_err());
}

#[test]
fn the_prover_cannot_choose_the_zerocheck_point() {
    // The point is drawn inside the reduction, from the transcript, after the commitment.
    //
    // Nothing in the proof carries it, so there is no field a prover could set.
    //
    // That is the obligation the skip round used to leave to its caller.
    //
    // Two runs on the same witness under the same transcript therefore agree exactly.
    let (check, witness) = fixture(0x90, 0);

    let mut first = fresh_challenger();
    let (_, one) = check.prove::<EF, _>(&witness.packed(), LOG_HEIGHT, &mut first);
    let mut second = fresh_challenger();
    let (_, two) = check.prove::<EF, _>(&witness.packed(), LOG_HEIGHT, &mut second);

    assert_eq!(one.point, two.point);
    assert_eq!(one.value, two.value);

    // And a different commitment absorbed first moves the point, so it is genuinely bound to it.
    let mut third = fresh_challenger();
    third.observe(EF::from_repr(7));
    let (_, other) = check.prove::<EF, _>(&witness.packed(), LOG_HEIGHT, &mut third);
    assert_ne!(one.point, other.point);
}
