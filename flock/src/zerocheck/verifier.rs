//! The zerocheck verifier: replay the rounds down to claims at one point.

use alloc::vec;
use alloc::vec::Vec;

use p3_binary_field::Poly64;
use p3_challenger::{CanObserve, CanSample, GrindingChallenger};
use p3_field::Dup;
use p3_verifier_arith::{Lift, Native};

use super::fixed::eq_coordinates;
use super::transcript::{VerifierTranscript, ZerocheckShape};
use super::{ZerocheckClaims, ZerocheckError, ZerocheckProof};
use crate::point::QuirkyPoint;
use crate::skip::{SKIP_SIZE, round1_at};
use crate::{Batch, Challenge, Val};

/// Replay a batch's zerocheck down to its claims.
///
/// Acceptance means only that the claims carry the transcript's final value: the caller must pin them.
///
/// # Errors
///
/// - A message has the wrong length, or a grinding witness is wrong.
/// - The evaluations miss the final claim.
pub fn verify<C>(
    batch: &Batch<'_>,
    proof: &ZerocheckProof,
    pow_bits: usize,
    challenger: &mut C,
) -> Result<ZerocheckClaims, ZerocheckError>
where
    C: CanObserve<Poly64> + CanSample<Poly64> + GrindingChallenger<Witness = Poly64>,
{
    verify_in(&mut Native::new(), batch, proof, pow_bits, challenger)
}

/// As [`verify`], within an arithmetic.
///
/// # Errors
///
/// As [`verify`].
pub fn verify_in<A, C>(
    arith: &mut A,
    batch: &Batch<'_>,
    proof: &ZerocheckProof,
    pow_bits: usize,
    challenger: &mut C,
) -> Result<ZerocheckClaims<A::EF>, ZerocheckError>
where
    A: Lift<Val = Val, Challenge = Challenge>,
    C: CanObserve<Poly64> + CanSample<Poly64> + GrindingChallenger<Witness = Poly64>,
{
    let shape = ZerocheckShape::new(batch, pow_bits);
    // Every structural check runs before the transcript moves.
    if proof.round1.len() != SKIP_SIZE {
        return Err(ZerocheckError::Round1Len {
            expected: SKIP_SIZE,
            actual: proof.round1.len(),
        });
    }
    if proof.rounds.len() != shape.n_rounds() {
        return Err(ZerocheckError::RoundCount {
            expected: shape.n_rounds(),
            actual: proof.rounds.len(),
        });
    }
    let mut transcript = arith.seeding(|| VerifierTranscript::new(challenger, shape));
    match replay(arith, &mut transcript, proof, pow_bits) {
        Ok(claims) => {
            for eval in [&claims.a, &claims.b, &claims.c] {
                arith.will_observe_ext(eval);
            }
            transcript.finish(&proof.evals);
            Ok(claims)
        }
        Err(e) => {
            transcript.abort();
            Err(e)
        }
    }
}

/// Announce a grinding witness the transcript absorbs next, when grinding is on.
fn announce_pow<A: Lift<Val = Val>>(arith: &mut A, pow_bits: usize, pow: Option<Val>) {
    if let Some(witness) = pow.filter(|_| pow_bits > 0) {
        let witness = arith.load(witness);
        arith.will_observe(vec![witness]);
    }
}

/// The verifier's walk through the transcript, up to the evaluations.
fn replay<A, C>(
    arith: &mut A,
    transcript: &mut VerifierTranscript<'_, C>,
    proof: &ZerocheckProof,
    pow_bits: usize,
) -> Result<ZerocheckClaims<A::EF>, ZerocheckError>
where
    A: Lift<Val = Val, Challenge = Challenge>,
    C: CanObserve<Poly64> + CanSample<Poly64> + GrindingChallenger<Witness = Poly64>,
{
    // Phase 1: the eq point past the skip.
    let outer: Vec<A::EF> = transcript
        .outer()
        .into_iter()
        .map(|r| arith.sampled_ext(r))
        .collect();
    let r = eq_coordinates(&outer);

    // Phase 2: round 1, known on `Lambda` and zero on `S`, at the skip challenge.
    let round1 = arith.load_ext_slice(&proof.round1).into_owned();
    for value in &round1 {
        arith.will_observe_ext(value);
    }
    announce_pow(arith, pow_bits, proof.round1_pow);
    let z = arith.sampled_ext(transcript.round1(&proof.round1, proof.round1_pow)?);
    let mut claim = round1_at(&z, &round1);

    // Phase 3: each round's constant from `(1 + r) G(0) + r G(1) = claim`, then its challenge.
    let mut rest: Vec<A::EF> = Vec::with_capacity(r.len());
    for (r_eq, round) in r.iter().zip(&proof.rounds) {
        let [c1, c2] = round.coeffs.map(|c| arith.load_ext(c));
        let c0 = claim + r_eq.dup() * (c1.dup() + c2.dup());
        arith.will_observe_ext(&c1);
        arith.will_observe_ext(&c2);
        announce_pow(arith, pow_bits, round.pow);
        let chi = arith.sampled_ext(transcript.round(&round.coeffs, round.pow)?);
        claim = c0 + chi.dup() * (c1 + chi.dup() * c2);
        rest.push(chi);
    }

    // Phase 4: the evaluations must reproduce the final claim.
    let [a, b, c] = proof.evals.map(|e| arith.load_ext(e));
    arith.assert_zero(a.dup() * b.dup() + c.dup() - claim, || {
        ZerocheckError::TerminalMismatch
    })?;
    rest.reverse();
    Ok(ZerocheckClaims {
        point: QuirkyPoint { skip: z, rest },
        a,
        b,
        c,
    })
}
