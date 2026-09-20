//! An executable specification of the generic-degree sumcheck, written from its format description.
//!
//! The shipped verifier is never called to decide anything here.
//!
//! What is shared is the sponge and the step schedule, which are the format, not the reduction.
//!
//! Everything above them is rebuilt from the description.
//!
//! The omitted second evaluation, the node set and the interpolation are all redone by hand.
//!
//! A disagreement there is therefore a real disagreement rather than an echo.

use p3_binary_field::{BinaryChallenger, BinaryField128};
use p3_challenger::HashChallenger;
use p3_field::{Field, PrimeCharacteristicRing};
use p3_keccak::Keccak256Hash;
use p3_sumcheck::generic_degree::{
    GenericDegreeProof, GenericDegreeShape, ProverTranscript, VerifierTranscript,
};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

type F = BinaryField128;
type Challenger = BinaryChallenger<F, HashChallenger<u8, Keccak256Hash, 32>>;

const LOG_HEIGHT: usize = 6;
const DEGREE: usize = 3;

const fn fresh_challenger() -> Challenger {
    Challenger::from_hasher(Vec::new(), Keccak256Hash)
}

/// The nodes the format interpolates a round polynomial over.
fn nodes(degree: usize) -> Vec<F> {
    (0..=degree).map(F::interpolation_node).collect()
}

/// Plain Lagrange interpolation, written out rather than reused.
fn interpolate(xs: &[F], ys: &[F], at: F) -> F {
    let mut acc = F::ZERO;
    for (i, &yi) in ys.iter().enumerate() {
        let mut term = yi;
        for (j, &xj) in xs.iter().enumerate() {
            if i != j {
                term *= (at - xj) * (xs[i] - xj).inverse();
            }
        }
        acc += term;
    }
    acc
}

/// One multilinear, as its values on the hypercube in natural order.
type Table = Vec<F>;

/// One edit applied to a finished proof, to check the specification refuses it.
type Edit = fn(&mut GenericDegreeProof<F, F>);

/// Binds the last variable of a table to a point.
fn fold(table: &[F], t: F) -> Table {
    let half = table.len() / 2;
    (0..half)
        .map(|i| table[i] + t * (table[half + i] + table[i]))
        .collect()
}

/// Sum over the hypercube of the product of the tables.
fn cube_sum(tables: &[Table]) -> F {
    (0..tables[0].len())
        .map(|x| tables.iter().map(|t| t[x]).product::<F>())
        .sum()
}

/// The round polynomial at one node, from the definition.
fn round_value(tables: &[Table], t: F) -> F {
    let folded: Vec<Table> = tables.iter().map(|table| fold(table, t)).collect();
    cube_sum(&folded)
}

/// Produces a proof from the definition of the protocol.
fn specification_prover(mut tables: Vec<Table>, pow_bits: usize) -> GenericDegreeProof<F, F> {
    let num_rounds = tables[0].len().ilog2() as usize;
    let degree = tables.len();
    let shape = GenericDegreeShape::new(num_rounds, degree, pow_bits);
    let claimed_sum = cube_sum(&tables);

    let mut challenger = fresh_challenger();
    let mut transcript =
        ProverTranscript::<Challenger, F, F>::new(&mut challenger, shape, claimed_sum);

    let xs = nodes(degree);
    let mut round_polys = Vec::new();
    let mut pow_witnesses = Vec::new();
    for _ in 0..num_rounds {
        // The format transmits every node but the second, which the invariant recovers.
        let evals: Vec<F> = core::iter::once(xs[0])
            .chain(xs[2..].iter().copied())
            .map(|t| round_value(&tables, t))
            .collect();
        let (challenge, witness) = transcript.round(&evals);
        round_polys.push(evals);
        pow_witnesses.extend(witness);
        tables = tables.iter().map(|t| fold(t, challenge)).collect();
    }
    transcript.finish();

    GenericDegreeProof {
        claimed_sum,
        round_polys,
        pow_witnesses,
    }
}

/// Verifies a proof from the definition, returning the challenges and the final claim.
fn specification_verifier(
    proof: &GenericDegreeProof<F, F>,
    num_rounds: usize,
    degree: usize,
    pow_bits: usize,
) -> Result<(Vec<F>, F), String> {
    if degree == 0 {
        return Err("a degree-zero round polynomial carries nothing".into());
    }
    if proof.round_polys.len() != num_rounds {
        return Err("wrong round count".into());
    }
    let expected_witnesses = if pow_bits > 0 { num_rounds } else { 0 };
    if proof.pow_witnesses.len() != expected_witnesses {
        return Err("wrong witness count".into());
    }
    // Widths are checked before anything is absorbed, so a refusal leaves no half-used sponge.
    if proof.round_polys.iter().any(|evals| evals.len() != degree) {
        return Err("wrong evaluation count".into());
    }

    let shape = GenericDegreeShape::new(num_rounds, degree, pow_bits);
    let mut challenger = fresh_challenger();
    let mut transcript =
        VerifierTranscript::<Challenger, F, F>::new(&mut challenger, shape, proof.claimed_sum);

    let xs = nodes(degree);
    let mut claim = proof.claimed_sum;
    let mut challenges = Vec::new();
    for round in 0..num_rounds {
        let evals = &proof.round_polys[round];
        let witness = (pow_bits > 0).then(|| proof.pow_witnesses[round]);
        let challenge = transcript
            .round(evals, witness)
            .map_err(|e| e.to_string())?;

        // The second node is the one the format omits, recovered from the running claim.
        let mut ys = Vec::with_capacity(degree + 1);
        ys.push(evals[0]);
        ys.push(claim - evals[0]);
        ys.extend_from_slice(&evals[1..]);

        claim = interpolate(&xs, &ys, challenge);
        challenges.push(challenge);
    }
    transcript.finish();
    Ok((challenges, claim))
}

/// The product of the tables at one point, which is what closes the reduction.
fn evaluate(tables: &[Table], point: &[F]) -> F {
    tables
        .iter()
        .map(|table| {
            let mut current = table.to_vec();
            for &r in point {
                current = fold(&current, r);
            }
            current[0]
        })
        .product()
}

fn instance(seed: u64) -> Vec<Table> {
    let mut rng = SmallRng::seed_from_u64(seed);
    (0..DEGREE)
        .map(|_| (0..1 << LOG_HEIGHT).map(|_| rng.random()).collect())
        .collect()
}

#[test]
fn the_specification_pair_closes_the_reduction() {
    for pow_bits in [0, 4] {
        let tables = instance(0x5EC1F1CA + pow_bits as u64);
        let proof = specification_prover(tables.clone(), pow_bits);
        let (point, claim) =
            specification_verifier(&proof, LOG_HEIGHT, DEGREE, pow_bits).expect("accepts");
        assert_eq!(claim, evaluate(&tables, &point), "pow bits {pow_bits}");
    }
}

#[test]
fn the_shipped_verifier_agrees_with_the_specification() {
    for pow_bits in [0, 4] {
        let tables = instance(0xA11CE + pow_bits as u64);
        let proof = specification_prover(tables, pow_bits);

        let (spec_point, spec_claim) =
            specification_verifier(&proof, LOG_HEIGHT, DEGREE, pow_bits).expect("accepts");

        let mut challenger = fresh_challenger();
        let (point, claim) = proof
            .verify(&mut challenger, LOG_HEIGHT, DEGREE, pow_bits)
            .expect("the shipped verifier accepts a specification-conformant proof");

        assert_eq!(point.as_slice(), spec_point.as_slice());
        assert_eq!(claim, spec_claim);
    }
}

#[test]
fn the_specification_refuses_every_edit_to_a_finished_proof() {
    let tables = instance(0xBADC0DE);
    let good = specification_prover(tables.clone(), 4);

    // Each edit is one thing an attacker could change about a shipped proof.
    let edits: [(&str, Edit); 5] = [
        ("a round value", |p| p.round_polys[2][1] += F::ONE),
        ("the claimed sum", |p| p.claimed_sum += F::ONE),
        ("the round order", |p| p.round_polys.swap(0, 1)),
        ("a round width", |p| {
            p.round_polys[0].pop();
        }),
        ("a grinding witness", |p| p.pow_witnesses[3] += F::ONE),
    ];

    for (what, edit) in edits {
        let mut proof = good.clone();
        edit(&mut proof);
        let verdict = specification_verifier(&proof, LOG_HEIGHT, DEGREE, 4);
        let accepted = matches!(
            verdict,
            Ok((ref point, claim)) if claim == evaluate(&tables, point)
        );
        assert!(!accepted, "the specification accepted an edit to {what}");
    }
}
