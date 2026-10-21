//! Zerocheck tests against a slow reference prover and a slow quirky-extension evaluator.

use alloc::vec::Vec;

use p3_binary_field::{Poly64, Poly192};
use p3_challenger::CanSample;
use p3_field::PrimeCharacteristicRing;
use p3_keccak::Keccak256Hash;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

use super::fixed::eq_coordinates;
use super::transcript::{ProverTranscript, ZerocheckShape};
use super::*;
use crate::skip::{SKIP_SIZE, lagrange_weights, phi8, round1_at};
use crate::tests::{Challenger, random_circuit, random_rows};
use crate::{Batch, BatchError, Builder, Circuit, Witness};

/// A fresh sponge.
fn fresh() -> Challenger {
    Challenger::from_hasher(Vec::new(), Keccak256Hash)
}

/// The bit at global index `i` of packed words.
fn bit(words: &[u64], i: usize) -> bool {
    words[i / 64] >> (i % 64) & 1 == 1
}

/// The quirky extension of packed bits at a point, by its definition.
///
/// Lagrange over `S` for the six low bits, the MSB-first multilinear eq for the rest.
fn quirky_eval(words: &[u64], point: &QuirkyPoint) -> Poly192 {
    let lagrange = lagrange_weights(point.skip, SKIP_SIZE);
    let eq = Poly::new_from_point(&point.rest, Poly192::ONE);
    assert_eq!(eq.as_slice().len(), words.len(), "one eq weight per row");
    eq.as_slice()
        .iter()
        .enumerate()
        .map(|(x, &e)| {
            let row: Poly192 = (0..SKIP_SIZE)
                .filter(|&s| bit(words, 64 * x + s))
                .map(|s| lagrange[s])
                .sum();
            e * row
        })
        .sum()
}

/// The zerocheck by its definition: every message summed naively over the cube.
fn reference_prove(
    batch: &Batch<'_>,
    witness: &Witness,
    challenger: &mut Challenger,
    pow_bits: usize,
) -> ZerocheckProof {
    let shape = ZerocheckShape::new(batch, pow_bits);
    let n_rows = 1 << shape.n_rounds();
    let c_words: Vec<u64> = witness
        .az()
        .iter()
        .zip(witness.bz())
        .map(|(a, b)| a & b)
        .collect();
    let tables = [witness.az(), witness.bz(), &c_words];
    let mut transcript = ProverTranscript::new(challenger, shape);
    let r = eq_coordinates(&transcript.outer());

    // Round 1: each row's interpolant on `S`, evaluated on `Lambda`, under the MSB-first eq on reversed `r`.
    let reversed: Vec<Poly192> = r.iter().rev().copied().collect();
    let eq = Poly::new_from_point(&reversed, Poly192::ONE);
    let at_lambda: Vec<Vec<Poly192>> = (0..SKIP_SIZE)
        .map(|i| lagrange_weights(Poly192::from(phi8((SKIP_SIZE + i) as u8)), SKIP_SIZE))
        .collect();
    let row_at = |words: &[u64], x: usize, weights: &[Poly192]| -> Poly192 {
        (0..SKIP_SIZE)
            .filter(|&s| bit(words, 64 * x + s))
            .map(|s| weights[s])
            .sum()
    };
    let round1: Vec<Poly192> = at_lambda
        .iter()
        .map(|w| {
            (0..n_rows)
                .map(|x| {
                    let [a, b, c] = tables.map(|t| row_at(t, x, w));
                    eq.as_slice()[x] * (a * b + c)
                })
                .sum()
        })
        .collect();
    let (z, round1_pow) = transcript.round1(&round1);

    // The multilinear rounds on the tables folded at `z`.
    let lagrange = lagrange_weights(z, SKIP_SIZE);
    let mut folded =
        tables.map(|t| Poly::new((0..n_rows).map(|x| row_at(t, x, &lagrange)).collect()));
    let mut claim = round1_at(z, &round1);
    let mut rounds = Vec::new();
    for j in 0..r.len() {
        let eq = Point::new(r[j + 1..].to_vec()).equality_weights_lsb();
        let [a, b, c] = folded.each_ref().map(Poly::as_slice);
        let at = |t: &[Poly192], x: usize, u: usize| t[2 * x + u];
        let (mut g0, mut g1, mut g_inf) = (Poly192::ZERO, Poly192::ZERO, Poly192::ZERO);
        for (x, &e) in eq.iter().enumerate() {
            g0 += e * (at(a, x, 0) * at(b, x, 0) + at(c, x, 0));
            g1 += e * (at(a, x, 1) * at(b, x, 1) + at(c, x, 1));
            g_inf += e * (at(a, x, 0) + at(a, x, 1)) * (at(b, x, 0) + at(b, x, 1));
        }
        let coeffs = [g0 + g1 + g_inf, g_inf];
        let (chi, pow) = transcript.round(&coeffs);
        claim = g0 + chi * (coeffs[0] + chi * coeffs[1]);
        for t in &mut folded {
            t.fix_suffix_var_mut(chi);
        }
        rounds.push(ZerocheckRound { coeffs, pow });
    }
    let (a, b) = (folded[0].as_slice()[0], folded[1].as_slice()[0]);
    let evals = [a, b, claim + a * b];
    transcript.finish(&evals);
    ZerocheckProof {
        round1,
        round1_pow,
        rounds,
        evals,
    }
}

/// A random circuit and a batch of random instances bringing it to about `m` variables.
///
/// The last `missing` instances take the padding row.
fn random_batch(
    rng: &mut SmallRng,
    max_gates: u32,
    m: usize,
    missing: usize,
) -> (Circuit, usize, Witness) {
    let c = random_circuit(rng, max_gates);
    let log_n = m.saturating_sub(c.k_log()).max(1);
    let (rows, padding) = random_rows(rng, (1 << log_n) - missing);
    let w = Witness::generate(&c, log_n, &rows, &padding, |row, words| {
        words.copy_from_slice(row);
    });
    (c, log_n, w)
}

/// A small circuit: `x AND y` then `variant` more products, each with `y` rotated by `shift`.
fn small_circuit(variant: usize, shift: usize) -> Circuit {
    let mut b = Builder::new(&[64, 64], &[64]);
    let (x, y) = (b.input(0).to_vec(), b.input(1).to_vec());
    for i in 0..64 {
        let mut w = b.and(x[i], y[i]);
        for _ in 0..variant {
            w = b.and(w, y[(i + shift) % 64]);
        }
        b.output(0, i, w);
    }
    b.finish()
}

/// Whether the claims are the true extensions of `A z`, `B z` and `z`.
fn all_true(claims: &ZerocheckClaims, w: &Witness) -> bool {
    claims.a == quirky_eval(w.az(), &claims.point)
        && claims.b == quirky_eval(w.bz(), &claims.point)
        && claims.c == quirky_eval(w.z(), &claims.point)
}

#[test]
fn honest_batches_verify_to_the_true_extensions() {
    // Invariant: the verifier accepts, and its claims are the prover's and the witness's extensions.
    let mut rng = SmallRng::seed_from_u64(0x2E0);
    for (m, pow_bits) in [(13, 0), (14, 2), (15, 0), (17, 0)] {
        let (c, log_n, w) = random_batch(&mut rng, 400, m, 1);
        let batch = Batch::new(&c, log_n).unwrap();
        let (proof, claims) = prove(&batch, &w, pow_bits, &mut fresh());
        assert_eq!(proof.rounds.len(), batch.log_n() - K_SKIP);
        let verified =
            verify(&batch, &proof, pow_bits, &mut fresh()).unwrap_or_else(|e| panic!("m={m}: {e}"));
        assert_eq!(verified, claims, "m={m}");
        assert!(all_true(&claims, &w), "m={m}");
    }
}

#[test]
fn the_fast_prover_is_the_reference_prover() {
    // Invariant: every message, challenge and evaluation equals the naive sums', on full and padded batches.
    //
    // A parallel grind may find another witness, so the proofs are compared without grinding.
    let mut rng = SmallRng::seed_from_u64(0x2E1);
    for round in 0..6 {
        let (c, log_n, w) = random_batch(
            &mut rng,
            300 + 200 * round as u32,
            13 + round % 4,
            round % 2,
        );
        let batch = Batch::new(&c, log_n).unwrap();
        let fast = prove(&batch, &w, 0, &mut fresh()).0;
        let slow = reference_prove(&batch, &w, &mut fresh(), 0);
        assert_eq!(fast, slow, "round {round}");
    }
}

#[test]
fn table_passes_reach_the_true_extensions() {
    // Invariant: a cube deep enough for paired table passes still lands on the true extensions.
    //
    // Fixture state: 22 variables, so the level-6 tables hold 2^16 values.
    let mut rng = SmallRng::seed_from_u64(0x2E2);
    let (c, log_n, w) = random_batch(&mut rng, 600, 22, 0);
    let batch = Batch::new(&c, log_n).unwrap();
    let (proof, claims) = prove(&batch, &w, 0, &mut fresh());
    assert_eq!(verify(&batch, &proof, 0, &mut fresh()), Ok(claims.clone()));
    assert!(all_true(&claims, &w));
}

#[test]
fn every_tampered_field_is_refused() {
    // Invariant: changing any value or length of an honest proof makes the verifier refuse.
    let mut rng = SmallRng::seed_from_u64(0x2E3);
    let (c, log_n, w) = random_batch(&mut rng, 400, 14, 0);
    let batch = Batch::new(&c, log_n).unwrap();
    let bump = |pow: Option<Poly64>| Some(pow.unwrap_or_default() + Poly64::ONE);
    for pow_bits in [0, 3] {
        let (proof, _) = prove(&batch, &w, pow_bits, &mut fresh());
        let check = |p: &ZerocheckProof| verify(&batch, p, pow_bits, &mut fresh());
        assert!(check(&proof).is_ok());
        let mut mutations: Vec<(&str, ZerocheckProof)> = Vec::new();
        let mut mutate = |label, f: &dyn Fn(&mut ZerocheckProof)| {
            let mut p = proof.clone();
            f(&mut p);
            mutations.push((label, p));
        };
        for i in [0, 17, SKIP_SIZE - 1] {
            mutate("round 1 value", &|p| p.round1[i] += Poly192::ONE);
        }
        mutate("round 1 grinding", &|p| p.round1_pow = bump(p.round1_pow));
        for j in [0, 3, proof.rounds.len() / 2, proof.rounds.len() - 1] {
            for k in 0..2 {
                mutate("round coefficient", &|p| {
                    p.rounds[j].coeffs[k] += Poly192::ONE;
                });
            }
            mutate("round grinding", &|p| {
                p.rounds[j].pow = bump(p.rounds[j].pow);
            });
        }
        for k in 0..3 {
            mutate("evaluation", &|p| p.evals[k] += Poly192::ONE);
        }
        mutate("short round 1", &|p| p.round1.truncate(SKIP_SIZE - 1));
        mutate("missing round", &|p| p.rounds.truncate(p.rounds.len() - 1));
        mutate("extra round", &|p| p.rounds.push(p.rounds[0]));
        for (label, p) in mutations {
            assert!(check(&p).is_err(), "{label} accepted, pow_bits={pow_bits}");
        }
    }
}

#[test]
fn a_wrong_witness_bit_leaves_a_false_claim() {
    // Invariant: a batch breaking `a b = z` at one bit is accepted, but never with every claim true.
    //
    // The prover reads only `a` and `b`, so a consumer pinning the claims to the witness refuses.
    let mut rng = SmallRng::seed_from_u64(0x2E4);
    for trial in 0..9 {
        let (c, log_n, mut w) = random_batch(&mut rng, 400, 13 + trial % 3, 0);
        // Flip one bit of `z`, `A z` or `B z` where the flip breaks `a b = z`.
        let i = loop {
            let instance = rng.random_range(0..1usize << log_n);
            let i = (instance << c.k_log()) + rng.random_range(0..c.useful_bits());
            // A factor's flip changes the product only where the other factor is one.
            let breaks = match trial % 3 {
                0 => true,
                1 => bit(w.bz(), i),
                _ => bit(w.az(), i),
            };
            if breaks {
                break i;
            }
        };
        let [z, az, bz] = w.tables_mut();
        let table = match trial % 3 {
            0 => z,
            1 => az,
            _ => bz,
        };
        table[i / 64] ^= 1 << (i % 64);
        assert_ne!(
            bit(w.az(), i) & bit(w.bz(), i),
            bit(w.z(), i),
            "the statement is false"
        );

        let batch = Batch::new(&c, log_n).unwrap();
        let (proof, claims) = prove(&batch, &w, 0, &mut fresh());
        let verified = verify(&batch, &proof, 0, &mut fresh());
        assert_eq!(verified, Ok(claims.clone()), "trial {trial}");
        assert!(!all_true(&claims, &w), "trial {trial}: every claim true");
    }
}

#[test]
fn the_transcript_binds_the_statement() {
    // Invariant: a different circuit, batch size or grinding draws different challenges.
    let (c, other, rewired) = (
        small_circuit(1, 1),
        small_circuit(2, 1),
        small_circuit(1, 2),
    );
    assert_eq!(c.k_log(), other.k_log(), "two statements of one shape");
    assert_eq!(c.num_ands(), rewired.num_ands(), "one operand apart");
    let log_n = MIN_LOG_N + 2 - c.k_log();
    let first = |circuit: &Circuit, log_n: usize, pow_bits: usize| -> Vec<Poly192> {
        let mut ch = fresh();
        let batch = Batch::new(circuit, log_n).unwrap();
        let mut t = ProverTranscript::new(&mut ch, ZerocheckShape::new(&batch, pow_bits));
        let mut drawn = t.outer();
        drawn.push(t.round1(&[Poly192::ZERO; SKIP_SIZE]).0);
        t.abort();
        drawn
    };
    let base = first(&c, log_n, 0);
    assert_eq!(base, first(&c, log_n, 0), "the transcript is deterministic");
    assert_ne!(base, first(&other, log_n, 0), "circuit");
    assert_ne!(base, first(&rewired, log_n, 0), "one operand");
    assert_ne!(base[..2], first(&c, log_n + 1, 0)[..2], "batch size");
    assert_ne!(base, first(&c, log_n, 1), "grinding");

    // A proof verifies for its own statement only, from a fresh sponge.
    let mut rng = SmallRng::seed_from_u64(0x2E5);
    let rows: Vec<[u64; 2]> = (0..1 << log_n).map(|_| rng.random()).collect();
    let w = Witness::generate(&c, log_n, &rows, &[0; 2], |row, words| {
        words.copy_from_slice(row);
    });
    let batch = Batch::new(&c, log_n).unwrap();
    let (proof, _) = prove(&batch, &w, 0, &mut fresh());
    assert!(verify(&batch, &proof, 0, &mut fresh()).is_ok());
    assert!(
        verify(&Batch::new(&other, log_n).unwrap(), &proof, 0, &mut fresh()).is_err(),
        "circuit"
    );
    let mut used = fresh();
    let _: Poly64 = used.sample();
    assert!(
        verify(&batch, &proof, 0, &mut used).is_err(),
        "a used sponge"
    );
}

#[test]
fn too_small_a_batch_is_refused() {
    // Invariant: a batch below the skip and the fixed coordinates has no zerocheck.
    let c = small_circuit(0, 0);
    let log_n = MIN_LOG_N - 1 - c.k_log();
    assert_eq!(
        Batch::new(&c, log_n).unwrap_err(),
        BatchError::TooFewVariables {
            log_n: MIN_LOG_N - 1
        }
    );
    assert!(Batch::new(&c, log_n + 1).is_ok());
}

#[test]
fn a_batch_past_the_address_space_is_refused() {
    let c = small_circuit(0, 0);
    let k_log = c.k_log();
    let too_many = |log_instances| BatchError::TooManyVariables {
        k_log,
        log_instances,
    };
    // Mutation: the variable count wraps around `usize`.
    assert_eq!(
        Batch::new(&c, usize::MAX).unwrap_err(),
        too_many(usize::MAX)
    );
    // Mutation: the bit count no longer fits a `usize` index.
    let largest = usize::BITS as usize - 1 - k_log;
    assert_eq!(
        Batch::new(&c, largest + 1).unwrap_err(),
        too_many(largest + 1)
    );
    // The largest representable batch is accepted.
    assert!(Batch::new(&c, largest).is_ok());
}
