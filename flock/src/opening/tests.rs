//! Opening tests: commit, lincheck, quirky claims, ring switch, WHIR, verify.

use alloc::vec::Vec;

use p3_challenger::CanSample;
use p3_field::PrimeCharacteristicRing;
use p3_keccak::Keccak256Hash;
use p3_multilinear_util::poly::Poly;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

use super::*;
use crate::lincheck::LincheckProof;
use crate::point::QuirkyPoint;
use crate::skip::K_SKIP;
use crate::tests::{random_circuit, random_rows};
use crate::zerocheck::{MIN_LOG_N, ZerocheckClaims};
use crate::{Batch, Circuit, PublicPorts, Witness, lincheck};

/// A fresh sponge.
fn challenger() -> FlockChallenger {
    FlockChallenger::from_hasher(Vec::new(), Keccak256Hash)
}

/// A quirky point over `m` variables drawn from the sponge.
fn draw_point(sponge: &mut FlockChallenger, m: usize) -> QuirkyPoint {
    let mut draw = || Challenge::new(core::array::from_fn(|_| sponge.sample()));
    QuirkyPoint {
        skip: draw(),
        rest: (0..m - K_SKIP).map(|_| draw()).collect(),
    }
}

/// Commit, reduce one circuit batch by the lincheck, and open its claim beside a direct one.
struct Run {
    pcs: WitnessPcs,
    log_instances: usize,
    commitment: Commitment,
    claims: Vec<QuirkyClaim>,
    proof: OpeningProof,
}

/// The prover side of one run, its claims checked against the witness.
fn run(
    seed: u64,
    gates: u32,
    log_instances: usize,
) -> (Run, Circuit, ZerocheckClaims, LincheckProof) {
    let mut rng = SmallRng::seed_from_u64(seed);
    let circuit = random_circuit(&mut rng, gates);
    let log_instances = log_instances.max(MIN_LOG_N.saturating_sub(circuit.k_log()));
    let (rows, padding) = random_rows(&mut rng, 1 << log_instances);
    let w = Witness::generate(&circuit, log_instances, &rows, &padding, |row, words| {
        words.copy_from_slice(row);
    });
    let k_log = circuit.k_log();
    let m = k_log + log_instances;
    // One opening carries the lincheck claim and the zerocheck's c-claim.
    let pcs = <WitnessPcs>::with_claims(m, &PcsParams::default(), 2).unwrap();

    // Commit, then draw the zerocheck point after the commitment.
    let mut sponge = challenger();
    let (commitment, data) = pcs.commit(w.z().to_vec(), &mut sponge).unwrap();
    let point = draw_point(&mut sponge, m);
    let zc = ZerocheckClaims {
        a: point.evaluate_packed(w.az(), k_log),
        b: point.evaluate_packed(w.bz(), k_log),
        c: point.evaluate_packed(w.z(), k_log),
        point: point.clone(),
    };
    let batch = Batch::new(&circuit, log_instances).unwrap();
    let (lincheck, claim, folded) =
        lincheck::prove_folded(&batch, w.z(), &zc, &PublicPorts::default(), &mut sponge);

    // The lincheck's claim, and the zerocheck's c-claim as a second quirky claim.
    let claims = alloc::vec![claim, QuirkyClaim { point, value: zc.c }];
    for claim in &claims {
        assert_eq!(claim.point.evaluate_packed(w.z(), k_log), claim.value);
    }
    // The first claim's split comes from the folded witness, the second's from the commitment.
    let split = pcs.suffix_tensors(&folded, &claims[0].point, log_instances);
    let proof = pcs
        .open(data, &claims, &[split.as_ref(), None], &mut sponge)
        .unwrap();
    (
        Run {
            pcs,
            log_instances,
            commitment,
            claims,
            proof,
        },
        circuit,
        zc,
        lincheck,
    )
}

/// The verifier side: replay the commitment, the lincheck, then the opening.
fn verify_run(
    run: &Run,
    circuit: &Circuit,
    zc: &ZerocheckClaims,
    lincheck: &LincheckProof,
    claims: &[QuirkyClaim],
    proof: &OpeningProof,
) -> Result<(), OpeningError> {
    let mut sponge = challenger();
    run.pcs.observe(&run.commitment, &mut sponge);
    let batch = Batch::new(circuit, run.log_instances).unwrap();
    let point = draw_point(&mut sponge, batch.log_n());
    assert_eq!(point, zc.point);
    let claim =
        lincheck::verify(&batch, zc, &PublicPorts::default(), lincheck, &mut sponge).unwrap();
    assert_eq!(claim, run.claims[0]);
    run.pcs.verify(&run.commitment, claims, proof, &mut sponge)
}

#[test]
fn a_committed_batch_opens_at_its_quirky_claims() {
    for (seed, gates, log_instances) in [(1, 600, 4), (2, 200, 6)] {
        let (r, circuit, zc, lincheck) = run(seed, gates, log_instances);
        verify_run(&r, &circuit, &zc, &lincheck, &r.claims, &r.proof)
            .unwrap_or_else(|e| panic!("seed {seed}: {e:?}"));
    }
}

#[test]
fn a_false_claim_or_a_tampered_proof_is_refused() {
    let (r, circuit, zc, lincheck) = run(3, 600, 4);
    let verify = |claims: &[QuirkyClaim], proof: &OpeningProof| {
        verify_run(&r, &circuit, &zc, &lincheck, claims, proof)
    };
    verify(&r.claims, &r.proof).unwrap();

    // Mutation: each claim's value, then its skip coordinate.
    for index in 0..r.claims.len() {
        let mut claims = r.claims.clone();
        claims[index].value += Challenge::ONE;
        assert!(verify(&claims, &r.proof).is_err(), "value {index}");
        let mut claims = r.claims.clone();
        claims[index].point.skip += Challenge::ONE;
        assert!(verify(&claims, &r.proof).is_err(), "skip {index}");
    }

    // Mutation: a tensor row, an out-of-domain answer, the first fold, the final polynomial.
    let mut proof = r.proof.clone();
    let mut rows: Vec<Val> = proof.reduction.claims[0].tensor.rows().to_vec();
    rows[7] += Val::ONE;
    proof.reduction.claims[0].tensor = rows.try_into().unwrap();
    assert!(verify(&r.claims, &proof).is_err(), "tensor row");
    // The commitment takes no out-of-domain sample, so a smuggled answer is refused.
    assert!(r.proof.opening.initial_ood_answers.is_empty());
    let mut proof = r.proof.clone();
    proof.opening.initial_ood_answers.push(Challenge::ONE);
    assert!(verify(&r.claims, &proof).is_err(), "out-of-domain answer");
    let mut proof = r.proof.clone();
    proof.opening.initial_sumcheck.polynomial_evaluations[0][0] += Challenge::ONE;
    assert!(verify(&r.claims, &proof).is_err(), "first fold");
    let mut proof = r.proof.clone();
    let mut last = proof.opening.final_poly.clone().unwrap().into_evals();
    last[0] += Challenge::ONE;
    proof.opening.final_poly = Some(Poly::new(last));
    assert!(verify(&r.claims, &proof).is_err(), "final polynomial");

    // A claim of the wrong width is refused before the transcript moves.
    let mut claims = r.claims.clone();
    claims[1].point.rest.pop();
    assert!(matches!(
        verify(&claims, &r.proof),
        Err(OpeningError::PointArity { .. })
    ));
}

#[test]
fn the_prover_refuses_a_claim_its_witness_does_not_satisfy() {
    let mut rng = SmallRng::seed_from_u64(4);
    let z: Vec<u64> = (0..1 << 8).map(|_| rng.random()).collect();
    let witness = Witness::from_tables(z.clone(), z.clone(), z.clone(), 6);
    let pcs = <WitnessPcs>::new(14, &PcsParams::default()).unwrap();
    let mut sponge = challenger();
    let (_, data) = pcs.commit(witness.z().to_vec(), &mut sponge).unwrap();
    let point = draw_point(&mut sponge, 14);
    let value = point.evaluate_packed(&z, 8) + Challenge::ONE;
    assert!(matches!(
        pcs.open(data, &[QuirkyClaim { point, value }], &[], &mut sponge),
        Err(OpeningError::FalseClaim { index: 0 })
    ));

    // A witness of another width is refused.
    let narrow = Witness::from_tables(z[..128].to_vec(), Vec::new(), Vec::new(), 5);
    assert!(matches!(
        pcs.commit(narrow.z().to_vec(), &mut challenger()),
        Err(OpeningError::WitnessLength { .. })
    ));
}

#[test]
fn a_witness_narrower_than_one_element_has_no_commitment() {
    assert!(matches!(
        <WitnessPcs>::new(K_SKIP - 1, &PcsParams::default()),
        Err(OpeningError::TooFewVariables { num_variables: 5 })
    ));
}

#[test]
fn a_split_from_the_folded_witness_reproduces_the_computed_opening() {
    // Invariant: the split the lincheck's fold yields is the one the commitment would compute.
    let log_instances = 4;
    let mut rng = SmallRng::seed_from_u64(5);
    let circuit = random_circuit(&mut rng, 6000);
    let log_instances = log_instances.max(MIN_LOG_N.saturating_sub(circuit.k_log()));
    let (rows, padding) = random_rows(&mut rng, 1 << log_instances);
    let w = Witness::generate(&circuit, log_instances, &rows, &padding, |row, words| {
        words.copy_from_slice(row);
    });
    let k_log = circuit.k_log();
    let m = k_log + log_instances;
    let pcs = <WitnessPcs>::new(m, &PcsParams::default()).unwrap();
    let mut sponge = challenger();
    let (_, data) = pcs.commit(w.z().to_vec(), &mut sponge).unwrap();
    let point = draw_point(&mut sponge, m);
    let zc = ZerocheckClaims {
        a: point.evaluate_packed(w.az(), k_log),
        b: point.evaluate_packed(w.bz(), k_log),
        c: point.evaluate_packed(w.z(), k_log),
        point,
    };
    let batch = Batch::new(&circuit, log_instances).unwrap();
    let (_, claim, folded) =
        lincheck::prove_folded(&batch, w.z(), &zc, &PublicPorts::default(), &mut sponge);
    let split = pcs.suffix_tensors(&folded, &claim.point, log_instances);
    assert!(split.is_some(), "k_log {k_log} leaves no split");

    let claims = [claim];
    let computed = pcs
        .open(data.clone(), &claims, &[], &mut sponge.clone())
        .unwrap();
    let read = pcs
        .open(data, &claims, &[split.as_ref()], &mut sponge)
        .unwrap();
    // The elements and the first fold come before any grinding, whose witness a parallel search may pick freely.
    assert_eq!(read.reduction.claims, computed.reduction.claims);
    assert_eq!(
        read.opening.initial_sumcheck.polynomial_evaluations,
        computed.opening.initial_sumcheck.polynomial_evaluations
    );
}
