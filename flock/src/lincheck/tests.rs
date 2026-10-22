//! Lincheck tests: completeness, soundness, and the fast paths against their definitions.

use alloc::vec;
use alloc::vec::Vec;

use p3_binary_field::{Poly64, Poly192};
use p3_challenger::CanSample;
use p3_field::PrimeCharacteristicRing;
use p3_keccak::Keccak256Hash;
use p3_multilinear_util::poly::Poly;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

use super::transcript::LincheckShape;
use super::*;
use crate::skip::{SKIP_SIZE, lagrange_weights};
use crate::tests::{random_circuit, random_rows};
use crate::zerocheck::{MIN_LOG_N, ZerocheckClaims};
use crate::{Circuit, FlockChallenger, Port, PublicPorts, Witness};

/// Challenge field.
type E = Challenge;

/// A fresh sponge.
fn challenger() -> FlockChallenger {
    FlockChallenger::from_hasher(Vec::new(), Keccak256Hash)
}

/// A quirky point over `m` variables, drawn from a transcript.
fn sampled_point(seed: u8, m: usize) -> QuirkyPoint {
    let mut sponge = FlockChallenger::from_hasher(vec![seed], Keccak256Hash);
    let mut draw = || -> E {
        let c: [Poly64; 3] = core::array::from_fn(|_| sponge.sample());
        Poly192::new(c)
    };
    QuirkyPoint {
        skip: draw(),
        rest: (0..m - K_SKIP).map(|_| draw()).collect(),
    }
}

/// The extension of packed bits at a quirky point, bit by bit from the definition.
fn slow_eval(bits: &[u64], point: &QuirkyPoint) -> E {
    let lagrange = lagrange_weights(&point.skip, SKIP_SIZE);
    let eq_rest = Poly::new_from_point(&point.rest, E::ONE);
    (0..64 * bits.len())
        .filter(|&x| (bits[x / 64] >> (x % 64)) & 1 == 1)
        .map(|x| lagrange[x % SKIP_SIZE] * eq_rest.as_slice()[x / SKIP_SIZE])
        .sum()
}

/// A circuit batch with its witness and the zerocheck's honest claims.
struct Fixture {
    circuit: Circuit,
    log_instances: usize,
    witness: Witness,
    claims: ZerocheckClaims,
}

impl Fixture {
    fn new(seed: u64, max_gates: u32, log_instances: usize) -> Self {
        let mut rng = SmallRng::seed_from_u64(seed);
        let circuit = random_circuit(&mut rng, max_gates);
        // The batch reaches the zerocheck's minimum size.
        let log_instances = log_instances.max(MIN_LOG_N.saturating_sub(circuit.k_log()));
        let (rows, padding) = random_rows(&mut rng, (1 << log_instances) - 1);
        let witness = Witness::generate(&circuit, log_instances, &rows, &padding, |row, words| {
            words.copy_from_slice(row);
        });
        let point = sampled_point(seed as u8, circuit.k_log() + log_instances);
        let claims = ZerocheckClaims {
            a: slow_eval(witness.az(), &point),
            b: slow_eval(witness.bz(), &point),
            c: slow_eval(witness.z(), &point),
            point,
        };
        Self {
            circuit,
            log_instances,
            witness,
            claims,
        }
    }

    fn batch(&self) -> Batch<'_> {
        Batch::new(&self.circuit, self.log_instances).unwrap()
    }

    fn prove(&self, witness: &Witness) -> (LincheckProof, QuirkyClaim) {
        prove(
            &self.batch(),
            witness,
            &self.claims,
            &PublicPorts::default(),
            &mut challenger(),
        )
    }

    fn verify(&self, proof: &LincheckProof) -> Result<QuirkyClaim, LincheckError> {
        self.verify_claims(&self.claims, proof)
    }

    fn verify_claims(
        &self,
        claims: &ZerocheckClaims,
        proof: &LincheckProof,
    ) -> Result<QuirkyClaim, LincheckError> {
        verify(
            &self.batch(),
            claims,
            &PublicPorts::default(),
            proof,
            &mut challenger(),
        )
    }
}

#[test]
fn the_fast_evaluation_is_the_definition() {
    // Fixture state: one instance, fewer than a stripe, whole stripes, whole tiles.
    let mut rng = SmallRng::seed_from_u64(0xE7A1);
    for (k_log, log_instances) in [(6, 0), (7, 2), (8, 3), (6, 6), (9, 7)] {
        let bits: Vec<u64> = (0..1 << (k_log + log_instances - 6))
            .map(|_| rng.random())
            .collect();
        let point = sampled_point(k_log as u8, k_log + log_instances);
        assert_eq!(
            point.evaluate_packed(&bits, k_log),
            slow_eval(&bits, &point),
            "{k_log} {log_instances}"
        );
    }
}

#[test]
fn the_backward_walk_is_the_transpose_of_the_forward_walk() {
    // Invariant: the backward walk gives A_0^T E + alpha B_0^T E, E the quirky eq table.
    let mut rng = SmallRng::seed_from_u64(0xBAC4);
    let circuit = random_circuit(&mut rng, 200);
    let k_log = circuit.k_log();
    let n = 1 << k_log;
    let point = sampled_point(3, k_log);
    let eq = point.eq_weights(point.rest.len());
    let alpha: E = rng.random();

    // Column j of A_0 and B_0 by a forward walk of its unit vector.
    let dense: Vec<E> = (0..n)
        .map(|j| {
            let mut unit = vec![Poly64::ZERO; n];
            unit[j] = Poly64::ONE;
            let mut sum = E::ZERO;
            circuit.walk_rows(&unit, |s, a, b| sum += eq[s] * (alpha * b + a));
            sum
        })
        .collect();
    let mut folded = vec![E::ZERO; n];
    circuit.add_fold_transpose(E::ONE, alpha, &eq, &mut folded);
    assert_eq!(folded, dense);
}

#[test]
fn an_honest_batch_reduces_to_a_true_claim() {
    // Fixture state, (seed, gates, log instances): batches raised to the minimum size, up to whole GFNI tiles.
    for (seed, gates, log_instances) in [
        (1, 30, 0),
        (2, 200, 2),
        (3, 600, 3),
        (4, 1500, 6),
        (5, 300, 7),
    ] {
        let f = Fixture::new(seed, gates, log_instances);
        let (proof, claim) = f.prove(&f.witness);
        let verified = f
            .verify(&proof)
            .unwrap_or_else(|e| panic!("seed {seed}: {e:?}"));
        assert_eq!(verified, claim, "seed {seed}");
        assert_eq!(
            claim.point.rest.len() + K_SKIP,
            f.circuit.k_log() + f.log_instances
        );
        assert_eq!(
            claim.value,
            slow_eval(f.witness.z(), &claim.point),
            "seed {seed}"
        );

        // The slices are the witness at the instance and sumcheck coordinates, per low bit pattern.
        let rest = &claim.point.rest;
        let eq_rest = Poly::new_from_point(rest, E::ONE);
        for (v, &slice) in proof.slices.iter().enumerate() {
            let want: E = (0..eq_rest.num_evals())
                .filter(|&i| {
                    let x = i * SKIP_SIZE + v;
                    (f.witness.z()[x / 64] >> (x % 64)) & 1 == 1
                })
                .map(|i| eq_rest.as_slice()[i])
                .sum();
            assert_eq!(slice, want, "seed {seed}, slice {v}");
        }
    }
}

#[test]
fn every_tampered_proof_field_is_refused() {
    let f = Fixture::new(7, 600, 3);
    let (proof, _) = f.prove(&f.witness);
    assert!(f.circuit.k_log() > K_SKIP, "the fixture has rounds");

    // Mutation: each round coefficient, then each slice, one off.
    for round in 0..proof.rounds.len() {
        for coefficient in 0..2 {
            let mut bad = proof.clone();
            bad.rounds[round][coefficient] += E::ONE;
            assert_eq!(
                f.verify(&bad),
                Err(LincheckError::FinalCheck),
                "round {round}"
            );
        }
    }
    for slice in 0..SKIP_SIZE {
        let mut bad = proof.clone();
        bad.slices[slice] += E::ONE;
        // Every slice carries a nonzero Lagrange weight in the C term.
        assert_eq!(
            f.verify(&bad),
            Err(LincheckError::FinalCheck),
            "slice {slice}"
        );
    }

    // Mutation: a round or a slice missing.
    let mut short = proof.clone();
    short.rounds.pop();
    assert!(matches!(
        f.verify(&short),
        Err(LincheckError::RoundCount { .. })
    ));
    let mut short = proof;
    short.slices.pop();
    assert!(matches!(
        f.verify(&short),
        Err(LincheckError::SliceCount { .. })
    ));
}

#[test]
fn a_false_input_claim_is_refused() {
    let f = Fixture::new(8, 400, 4);
    let (proof, _) = f.prove(&f.witness);
    for which in 0..4 {
        let mut claims = f.claims.clone();
        match which {
            0 => claims.a += E::ONE,
            1 => claims.b += E::ONE,
            2 => claims.c += E::ONE,
            _ => claims.point.skip += E::ONE,
        }
        assert_eq!(
            f.verify_claims(&claims, &proof),
            Err(LincheckError::FinalCheck),
            "{which}"
        );
        // A prover replaying the false claims honestly fails the same way.
        let (proof, _) = prove(
            &f.batch(),
            &f.witness,
            &claims,
            &PublicPorts::default(),
            &mut challenger(),
        );
        assert_eq!(
            f.verify_claims(&claims, &proof),
            Err(LincheckError::FinalCheck),
            "{which}"
        );
    }

    // A point of the wrong width is refused before the transcript moves.
    let mut claims = f.claims.clone();
    claims.point.rest.pop();
    assert!(matches!(
        f.verify_claims(&claims, &proof),
        Err(LincheckError::PointArity { .. })
    ));
}

#[test]
fn a_witness_off_by_one_bit_is_refused() {
    // Mutation: one committed product bit of one instance flipped, the claims kept honest.
    let f = Fixture::new(9, 600, 4);
    let slot = f.circuit.useful_bits() - 1;
    let words = (1 << f.circuit.k_log()) / 64;
    for instance in [0, 5, 15] {
        let mut w = f.witness.clone();
        w.tables_mut()[0][instance * words + slot / 64] ^= 1 << (slot % 64);
        let (proof, _) = f.prove(&w);
        assert_eq!(
            f.verify(&proof),
            Err(LincheckError::FinalCheck),
            "{instance}"
        );
    }

    // Mutation: the constant wire cleared in one instance, which the pin alone catches.
    let pin = f.circuit.const_pos();
    let mut w = f.witness.clone();
    w.tables_mut()[0][3 * words + pin / 64] ^= 1 << (pin % 64);
    let (proof, _) = f.prove(&w);
    assert_eq!(f.verify(&proof), Err(LincheckError::FinalCheck));
}

#[test]
fn the_transcript_binds_the_circuit_and_the_batch() {
    let f = Fixture::new(10, 400, 3);
    let (proof, _) = f.prove(&f.witness);
    // Another circuit of the same width replays other challenges.
    let other = random_circuit(&mut SmallRng::seed_from_u64(11), 400);
    if other.k_log() == f.circuit.k_log() {
        let batch = Batch::new(&other, f.log_instances).unwrap();
        assert!(
            verify(
                &batch,
                &f.claims,
                &PublicPorts::default(),
                &proof,
                &mut challenger()
            )
            .is_err()
        );
    }
    // A sponge that saw something first replays other challenges.
    let mut seeded = FlockChallenger::from_hasher(vec![1], Keccak256Hash);
    assert!(
        verify(
            &f.batch(),
            &f.claims,
            &PublicPorts::default(),
            &proof,
            &mut seeded
        )
        .is_err()
    );
}

#[test]
fn the_transcript_binds_the_public_values() {
    let f = Fixture::new(12, 400, 3);
    let batch = f.batch();
    let words = 1 << (f.circuit.k_log() - 6);
    // The first word of every instance, with bit `flip` of instance 0 toggled.
    let public = |flip: u64| {
        PublicPorts::new(&batch, &[Port::Input(0)], |i, out| {
            out[0] = f.witness.z()[i * words] ^ if i == 0 { flip } else { 0 };
        })
    };
    // The lincheck's own seed differs, not only the outer statement's.
    let seed = |flip| LincheckShape::new(&batch, &public(flip)).domain_separator();
    assert_ne!(seed(1), seed(0));
}
