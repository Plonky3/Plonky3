//! Opening tests: commit, lincheck, quirky claims, ring switch, WHIR, verify.

use alloc::vec::Vec;

use p3_challenger::CanSample;
use p3_field::PrimeCharacteristicRing;
use p3_keccak::Keccak256Hash;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

use super::*;
use crate::lincheck::LincheckProof;
use crate::point::QuirkyPoint;
use crate::tests::{random_circuit, random_rows};
use crate::zerocheck::{MIN_LOG_N, ZerocheckClaims};
use crate::{Batch, Circuit, lincheck};

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
    let w = crate::Witness::generate(&circuit, log_instances, &rows, &padding, |row, words| {
        words.copy_from_slice(row);
    });
    let k_log = circuit.k_log();
    let m = k_log + log_instances;
    let pcs = WitnessPcs::new(m, &PcsParams::default()).unwrap();

    // Commit, then draw the zerocheck point after the commitment.
    let mut sponge = challenger();
    let (commitment, data) = pcs.commit(&w, &mut sponge).unwrap();
    let point = draw_point(&mut sponge, m);
    let zc = ZerocheckClaims {
        a: point.evaluate_packed(w.az(), k_log),
        b: point.evaluate_packed(w.bz(), k_log),
        c: point.evaluate_packed(w.z(), k_log),
        point: point.clone(),
    };
    let batch = Batch::new(&circuit, log_instances).unwrap();
    let (lincheck, claim) = lincheck::prove(&batch, &w, &zc, &mut sponge);

    // The lincheck's claim, and the zerocheck's c-claim as a second quirky claim.
    let claims = alloc::vec![claim, QuirkyClaim { point, value: zc.c }];
    for claim in &claims {
        assert_eq!(claim.point.evaluate_packed(w.z(), k_log), claim.value);
    }
    let proof = pcs.open(data, &claims, &mut sponge).unwrap();
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
    let claim = lincheck::verify(&batch, zc, lincheck, &mut sponge).unwrap();
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

    // Mutation: a tensor row, the surviving value, a round of the ring switch.
    let mut proof = r.proof.clone();
    let mut rows: Vec<Val> = proof.reduction.claims[0].tensor.rows().to_vec();
    rows[7] += Val::ONE;
    proof.reduction.claims[0].tensor = rows.try_into().unwrap();
    assert!(verify(&r.claims, &proof).is_err(), "tensor row");
    let mut proof = r.proof.clone();
    proof.reduction.final_eval += Challenge::ONE;
    assert!(verify(&r.claims, &proof).is_err(), "final value");

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
    let pcs = WitnessPcs::new(14, &PcsParams::default()).unwrap();
    let mut sponge = challenger();
    let (_, data) = pcs.commit(&witness, &mut sponge).unwrap();
    let point = draw_point(&mut sponge, 14);
    let value = point.evaluate_packed(&z, 8) + Challenge::ONE;
    assert!(matches!(
        pcs.open(data, &[QuirkyClaim { point, value }], &mut sponge),
        Err(OpeningError::FalseClaim { index: 0 })
    ));

    // A witness of another width is refused.
    let narrow = Witness::from_tables(z[..128].to_vec(), Vec::new(), Vec::new(), 5);
    assert!(matches!(
        pcs.commit(&narrow, &mut challenger()),
        Err(OpeningError::WitnessLength { .. })
    ));
}

#[test]
fn a_witness_narrower_than_one_element_has_no_commitment() {
    assert!(matches!(
        WitnessPcs::new(K_SKIP - 1, &PcsParams::default()),
        Err(OpeningError::TooFewVariables { num_variables: 5 })
    ));
}
