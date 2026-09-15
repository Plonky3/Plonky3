//! A whole binary zerocheck: one univariate-skip round, then the ordinary sumcheck driver.

use p3_binary_field::{BinaryChallenger, BinaryField8, BinaryField128};
use p3_challenger::HashChallenger;
use p3_field::{Field, PrimeCharacteristicRing};
use p3_keccak::Keccak256Hash;
use p3_multilinear_util::poly::Poly;
use p3_sumcheck::generic_degree::{GenericDegreeProof, RoundProver};
use p3_sumcheck::univariate_skip::{
    CHUNK_BITS, SkipRound, UnivariateSkipProverTranscript, UnivariateSkipShape,
    UnivariateSkipVerifierTranscript,
};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

/// The subspace the skip round runs over lives in a byte field.
type F = BinaryField8;

/// Challenges and every value the residual rounds carry live in the 128-bit field above it.
type EF = BinaryField128;

/// Keccak-backed Fiat-Shamir, the same instantiation the binary sumcheck tests use.
type Challenger = BinaryChallenger<EF, HashChallenger<u8, Keccak256Hash, 32>>;

/// Variables the skip round binds in one go.
const LOG_SKIP: usize = 6;

/// Variables the residual sumcheck binds one at a time.
const LOG_ROWS: usize = 4;

/// Per-variable degree of the residual summand.
///
/// The equality weight contributes one, and the product of the two operands contributes two.
const RESIDUAL_DEGREE: usize = 3;

const fn fresh_challenger() -> Challenger {
    Challenger::from_hasher(Vec::new(), Keccak256Hash)
}

/// The description both sides derive from their own configuration, never from a proof.
const fn shape(round: &SkipRound<F>, pow_bits: usize) -> UnivariateSkipShape {
    UnivariateSkipShape::new(
        round.log_size(),
        round.domain().log_extended(),
        LOG_ROWS,
        RESIDUAL_DEGREE,
        pow_bits,
    )
}

/// A bit-valued witness for the conjunction constraint, packed row by row.
struct Witness {
    /// Packed rows of the first operand.
    a: Vec<u8>,
    /// Packed rows of the second operand.
    b: Vec<u8>,
    /// Packed rows of their claimed conjunction.
    c: Vec<u8>,
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
        Self { a, b, c }
    }

    /// The three packed operands, in the order the constraint reads them.
    const fn operands(&self) -> [&Vec<u8>; 3] {
        [&self.a, &self.b, &self.c]
    }
}

/// One edit applied to a finished proof, to check the verifier refuses it.
type Mutation = fn(&mut Proof);

/// Everything the prover sends.
struct Proof {
    /// The round polynomial on the transmitted points.
    message: Vec<EF>,
    /// The grinding witness guarding the skip challenge, when grinding is enabled.
    pow_witness: Option<EF>,
    /// The delegated sumcheck over the variables the round did not bind.
    residual: GenericDegreeProof<EF, EF>,
}

/// The residual summand, folded one variable at a time.
///
/// The claim it proves is
///
/// ```text
///     sum_x eq(r, x) * ( A(x) * B(x) + C(x) )
/// ```
///
/// where the operands are the committed rows already read at the skip round's challenge.
struct ResidualProver {
    /// The zerocheck's equality weight over the rows.
    eq: Poly<EF>,
    /// The three operands, read at the skip round's challenge.
    operands: [Poly<EF>; 3],
}

impl RoundProver<EF> for ResidualProver {
    fn fold(&mut self, r: EF) {
        // Binding a variable halves every factor in step.
        self.eq.fix_prefix_var_mut(r);
        for operand in &mut self.operands {
            operand.fix_prefix_var_mut(r);
        }
    }

    fn round_poly(&self) -> Vec<EF> {
        // A degree-three round polynomial is reported at three nodes.
        //
        // The value at the fourth is recoverable from the running claim.
        [0, 2, 3]
            .into_iter()
            .map(|node| {
                let node = EF::interpolation_node(node);
                let half = self.eq.num_evals() / 2;

                // Each factor is read between its two halves at the node, then combined.
                (0..half)
                    .map(|index| {
                        let at = |poly: &Poly<EF>| {
                            let values = poly.as_slice();
                            values[index] + (values[index + half] - values[index]) * node
                        };
                        let [a, b, c] = core::array::from_fn(|which| at(&self.operands[which]));
                        at(&self.eq) * (a * b + c)
                    })
                    .sum()
            })
            .collect()
    }
}

/// Build the round message from the packed witness.
fn round_message(round: &SkipRound<F>, witness: &Witness, eq: &Poly<EF>) -> Vec<EF> {
    let num_rows = eq.num_evals();
    let stride = round.num_transmitted();

    // Read every row as a univariate and extend it off the skipped subspace.
    //
    // The rows are bits and the domain is a byte field, so nothing leaves the byte field here.
    let extended = witness.operands().map(|packed| {
        let mut out = F::zero_vec(num_rows * stride);
        round.extend_rows(packed, &mut out);
        out
    });

    // Compose the three extensions into the constraint's own values.
    //
    // Characteristic two makes subtraction addition, so the constraint is a product plus a term.
    let composed = (0..num_rows * stride)
        .map(|index| extended[0][index] * extended[1][index] + extended[2][index])
        .collect::<Vec<_>>();

    // Weigh the rows by the zerocheck's equality polynomial.
    round.round_message::<EF>(&composed, eq.as_slice())
}

/// Run the prover through both phases, under one transcript.
fn prove(
    round: &SkipRound<F>,
    witness: &Witness,
    zerocheck_point: &[EF],
    pow_bits: usize,
) -> Proof {
    let eq = Poly::new_from_point(zerocheck_point, EF::ONE);
    let message = round_message(round, witness, &eq);
    prove_with_message(round, witness, eq, message, pow_bits).0
}

/// Prove with a caller-supplied round message, and report the challenge it induced.
///
/// An honest prover always passes the message the witness produces.
///
/// Taking it as an argument is what lets a test play a prover that sends something else.
fn prove_with_message(
    round: &SkipRound<F>,
    witness: &Witness,
    eq: Poly<EF>,
    message: Vec<EF>,
    pow_bits: usize,
) -> (Proof, EF) {
    // The skip round and the sumcheck it delegates to run under one transcript.
    let mut challenger = fresh_challenger();
    let mut transcript = UnivariateSkipProverTranscript::<Challenger, EF, EF>::new(
        &mut challenger,
        shape(round, pow_bits),
    );

    // Binding the message before the challenge is what stops a prover choosing one to suit.
    let (lambda, pow_witness) = transcript.round_message(&message);

    // One table reads every row of every operand at that challenge.
    let selector = round.selector::<EF>(lambda);
    let operands = witness.operands().map(|packed| selector.bind(packed));

    // The residual claim is the message read back at the same challenge.
    let claimed_sum = round
        .evaluate(&message, lambda)
        .expect("the prover built the message at this round's width");
    let (residual, _) = transcript.residual_sumcheck(|challenger| {
        ResidualProver { eq, operands }.prove::<EF, _>(
            challenger,
            LOG_ROWS,
            RESIDUAL_DEGREE,
            pow_bits,
            claimed_sum,
        )
    });
    transcript.finish();

    (
        Proof {
            message,
            pow_witness,
            residual,
        },
        lambda,
    )
}

/// Replay the transcript and run every check the verifier owes.
///
/// Returns whether the proof is accepted.
fn verify(
    round: &SkipRound<F>,
    witness: &Witness,
    zerocheck_point: &[EF],
    proof: &Proof,
    pow_bits: usize,
) -> bool {
    let mut challenger = fresh_challenger();
    let mut transcript = UnivariateSkipVerifierTranscript::<Challenger, EF, EF>::new(
        &mut challenger,
        shape(round, pow_bits),
    );

    // Replay the round: bind the message, re-check the grind, draw the same challenge.
    let Ok(lambda) = transcript.round_message(&proof.message, proof.pow_witness) else {
        return false;
    };

    // The round polynomial is defined to vanish on the subspace and match the message off it.
    //
    // Reading it at the challenge therefore needs nothing further from the prover.
    let Ok(claimed_sum) = round.evaluate(&proof.message, lambda) else {
        transcript.abort();
        return false;
    };
    if proof.residual.claimed_sum != claimed_sum {
        transcript.abort();
        return false;
    }

    // Replay the residual rounds inside the same bracket the prover used.
    //
    // The delegated rejection travels out as an error, which releases the outer driver too.
    let replayed = transcript.residual_sumcheck(|challenger| {
        proof
            .residual
            .verify(challenger, LOG_ROWS, RESIDUAL_DEGREE, pow_bits)
    });
    let Ok((point, final_sum)) = replayed else {
        return false;
    };
    transcript.finish();

    // The verifier would reach the operand values through a commitment opening.
    //
    // This test stands in for that by reading the committed rows directly.
    let selector = round.selector::<EF>(lambda);
    let operands = witness.operands().map(|packed| selector.bind(packed));
    let [a, b, c] = core::array::from_fn(|which| operands[which].eval_base(&point));
    let eq = Poly::new_from_point(zerocheck_point, EF::ONE).eval_base(&point);

    final_sum == eq * (a * b + c)
}

/// Build a fresh round, witness, and zerocheck challenge point from one seed.
fn fixture(seed: u64) -> (SkipRound<F>, Witness, Vec<EF>) {
    let mut rng = SmallRng::seed_from_u64(seed);
    let round = SkipRound::<F>::new(LOG_SKIP, 2).unwrap();
    let witness = Witness::random(&mut rng, 1 << LOG_ROWS, round.row_bytes());
    let point = (0..LOG_ROWS).map(|_| rng.random::<EF>()).collect();
    (round, witness, point)
}

#[test]
fn a_satisfied_constraint_is_proved_and_accepted() {
    // Fixture state: 2^4 rows of 2^6 bits, so a 10-variable zerocheck over bits.
    //
    //     10 variables = 6 skipped in one round + 4 ordinary rounds
    //
    // Invariant: an honest witness produces a proof every check accepts.
    let (round, witness, point) = fixture(0x5C1);
    let proof = prove(&round, &witness, &point, 0);
    assert!(verify(&round, &witness, &point, &proof, 0));
}

#[test]
fn grinding_guards_the_skip_challenge_end_to_end() {
    // The skip challenge fixes the entire residual claim, so it is worth grinding for.
    //
    // Fixture state: 4 bits of difficulty, cheap enough to run in a test.
    let (round, witness, point) = fixture(0x6D1);
    let proof = prove(&round, &witness, &point, 4);
    assert!(proof.pow_witness.is_some());
    assert!(verify(&round, &witness, &point, &proof, 4));
}

#[test]
fn the_skip_round_removes_its_variables_from_the_sumcheck() {
    // The whole purpose of the round is that the ordinary driver never sees the skipped variables.
    //
    //     without the skip:  10 round polynomials, the first widening 2^9 cells to the field
    //     with the skip:      1 message of 64 values, then 4 round polynomials
    let (round, witness, point) = fixture(0x11);
    let proof = prove(&round, &witness, &point, 0);
    assert_eq!(proof.residual.round_polys.len(), LOG_ROWS);
    assert_eq!(proof.message.len(), 1 << LOG_SKIP);
}

#[test]
fn a_broken_constraint_is_rejected() {
    // Mutation: flip one bit of the claimed conjunction.
    //
    //     c[cell] ^= 1
    //
    // The round polynomial then stops vanishing on the skipped subspace.
    //
    // The verifier's reconstruction still does, so the two disagree and the check fails.
    let (round, mut witness, point) = fixture(0xBAD);
    witness.c[0] ^= 1;
    let proof = prove(&round, &witness, &point, 0);
    assert!(!verify(&round, &witness, &point, &proof, 0));
}

#[test]
fn a_tampered_round_message_is_rejected() {
    // Mutation: corrupt one transmitted value after the honest prover produced it.
    //
    // The message is bound before the challenge is drawn, so the tamper moves the challenge.
    //
    // The residual proof then no longer answers the question that gets asked.
    let (round, witness, point) = fixture(0x7A3);
    let mut proof = prove(&round, &witness, &point, 0);
    proof.message[7] += EF::ONE;
    assert!(!verify(&round, &witness, &point, &proof, 0));
}

#[test]
fn a_message_of_the_wrong_width_is_rejected() {
    // Mutation: drop one transmitted value.
    //
    // The described step declares its width.
    //
    // The replay therefore refuses the message rather than reading a shorter polynomial.
    let (round, witness, point) = fixture(0x9E1);
    let mut proof = prove(&round, &witness, &point, 0);
    proof.message.pop();
    assert!(!verify(&round, &witness, &point, &proof, 0));
}

#[test]
fn a_proof_replayed_under_a_different_shape_is_rejected() {
    // Mutation: verify against a narrower skip than the prover used.
    //
    // The shape reaches the seed, so the two sides diverge from the very first draw.
    let (round, witness, point) = fixture(0x4B2);
    let proof = prove(&round, &witness, &point, 0);

    let narrower = SkipRound::<F>::new(LOG_SKIP - 1, 2).unwrap();
    assert!(!verify(&narrower, &witness, &point, &proof, 0));
}

#[test]
fn the_packing_convention_is_the_one_the_round_documents() {
    // The round reads bit `j` of byte `b` as the cell at skipped index `8b + j`.
    //
    // Getting this backwards would still prove something, just not the intended statement.
    //
    // It is pinned here against the subspace read that recovers the packed bits.
    //
    //     packed row: [0b0000_0010, 0, ...]  ->  cell 1 is set, all others clear
    let round = SkipRound::<F>::new(LOG_SKIP, 2).unwrap();
    let mut packed = vec![0u8; round.row_bytes()];
    packed[0] = 0b0000_0010;

    for column in 0..1 << LOG_SKIP {
        let point = EF::from(round.domain().subspace()[column]);
        let bound = round.selector::<EF>(point).bind(&packed);
        let expected = if column == 1 { EF::ONE } else { EF::ZERO };
        assert_eq!(bound.as_slice()[0], expected, "column={column}");
    }

    // The second byte holds cells 8 through 15, which is what the chunk width fixes.
    assert_eq!(CHUNK_BITS, 8);
}

#[test]
fn a_message_adapted_to_the_challenge_is_rejected() {
    // This is the attack binding exists to stop, and the only test that exercises it.
    //
    // A prover with a broken witness learns the challenge an honest message induces.
    //
    // It then edits the message so the read-back matches the sum it can prove there.
    //
    //     message bound first:  editing it moves the challenge, and the edit no longer fits
    //     message unbound:      the edit lands on the challenge it was built for
    //
    // Tampering after the fact does not test this.
    //
    // It leaves the residual proof inconsistent, so it is refused either way.
    let (round, mut witness, point) = fixture(0xADA);
    witness.c[0] ^= 1;
    let eq = Poly::new_from_point(&point, EF::ONE);
    let honest = round_message(&round, &witness, &eq);

    // Learn the challenge the honest message induces.
    let (_, lambda) = prove_with_message(&round, &witness, eq.clone(), honest.clone(), 0);

    // The residual sum the broken witness really has at that challenge.
    let selector = round.selector::<EF>(lambda);
    let operands = witness.operands().map(|packed| selector.bind(packed));
    let truth = (0..eq.num_evals())
        .map(|row| {
            let a = operands[0].as_slice()[row];
            let b = operands[1].as_slice()[row];
            let c = operands[2].as_slice()[row];
            eq.as_slice()[row] * (a * b + c)
        })
        .sum::<EF>();

    // Patch the last entry so the read-back at that challenge equals the provable sum.
    //
    // The read-back is linear in the message, so one entry carries the whole correction.
    let last = honest.len() - 1;
    let mut unit = vec![EF::ZERO; honest.len()];
    unit[last] = EF::ONE;
    let gap = truth - round.evaluate(&honest, lambda).unwrap();
    let mut cheat = honest;
    cheat[last] += gap * round.evaluate(&unit, lambda).unwrap().inverse();

    // The edit does fit the challenge it was built for.
    assert_eq!(round.evaluate(&cheat, lambda).unwrap(), truth);

    // Running the real transcript on it draws a different challenge, so the edit misses.
    let (proof, adapted) = prove_with_message(&round, &witness, eq, cheat, 0);
    assert_ne!(
        adapted, lambda,
        "binding the message must move the challenge"
    );
    assert!(!verify(&round, &witness, &point, &proof, 0));
}

#[test]
fn a_malformed_residual_proof_is_rejected_rather_than_panicking() {
    // The outer driver panics on drop if it is left unfinalized.
    //
    // That check is live in release builds whenever panics unwind.
    //
    // A delegated rejection therefore has to release it.
    //
    // Otherwise a malformed proof aborts the verifier instead of being refused.
    //
    // Each mutation below makes the residual replay fail at a different point.
    let (round, witness, point) = fixture(0xF00);
    let honest = prove(&round, &witness, &point, 0);

    let mutations: [(&str, Mutation); 5] = [
        ("one round popped", |proof| {
            proof.residual.round_polys.pop();
        }),
        ("every round dropped", |proof| {
            proof.residual.round_polys.clear();
        }),
        ("one round one element too wide", |proof| {
            proof.residual.round_polys[0].push(EF::ONE);
        }),
        ("an extra round appended", |proof| {
            let extra = proof.residual.round_polys[0].clone();
            proof.residual.round_polys.push(extra);
        }),
        ("a grinding witness at zero difficulty", |proof| {
            proof.pow_witness = Some(EF::ONE);
        }),
    ];

    for (what, mutate) in mutations {
        let mut proof = Proof {
            message: honest.message.clone(),
            pow_witness: honest.pow_witness,
            residual: honest.residual.clone(),
        };
        mutate(&mut proof);
        assert!(
            !verify(&round, &witness, &point, &proof, 0),
            "must reject: {what}"
        );
    }
}

#[test]
fn a_grinding_witness_at_zero_difficulty_is_rejected() {
    // Proofs stay canonical: one statement must not have two accepting forms.
    //
    // At zero difficulty the description has no grinding step.
    //
    // A witness is therefore not merely useless, it is refused.
    let (round, witness, point) = fixture(0xCA1);
    let mut proof = prove(&round, &witness, &point, 0);
    assert_eq!(proof.pow_witness, None);

    proof.pow_witness = Some(EF::from_u64(12345));
    assert!(!verify(&round, &witness, &point, &proof, 0));
}
