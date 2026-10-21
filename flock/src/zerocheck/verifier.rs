//! The zerocheck verifier: replay the rounds down to claims at one point.

use alloc::vec::Vec;

use p3_binary_field::{Poly64, Poly192};
use p3_challenger::{CanObserve, CanSample, GrindingChallenger};

use super::fixed::eq_coordinates;
use super::transcript::{VerifierTranscript, ZerocheckShape};
use super::{ZerocheckClaims, ZerocheckError, ZerocheckProof};
use crate::Batch;
use crate::point::QuirkyPoint;
use crate::skip::{SKIP_SIZE, round1_at};

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
    let mut transcript = VerifierTranscript::new(challenger, shape);
    match replay(&mut transcript, proof) {
        Ok(claims) => {
            transcript.finish(&proof.evals);
            Ok(claims)
        }
        Err(e) => {
            transcript.abort();
            Err(e)
        }
    }
}

/// The verifier's walk through the transcript, up to the evaluations.
fn replay<C>(
    transcript: &mut VerifierTranscript<'_, C>,
    proof: &ZerocheckProof,
) -> Result<ZerocheckClaims, ZerocheckError>
where
    C: CanObserve<Poly64> + CanSample<Poly64> + GrindingChallenger<Witness = Poly64>,
{
    // Phase 1: the eq point past the skip.
    let r = eq_coordinates(&transcript.outer());

    // Phase 2: round 1, known on `Lambda` and zero on `S`, at the skip challenge.
    let z = transcript.round1(&proof.round1, proof.round1_pow)?;
    let mut claim = round1_at(z, &proof.round1);

    // Phase 3: each round's constant from `(1 + r) G(0) + r G(1) = claim`, then its challenge.
    let mut rest: Vec<Poly192> = Vec::with_capacity(r.len());
    for (&r_eq, round) in r.iter().zip(&proof.rounds) {
        let [c1, c2] = round.coeffs;
        let c0 = claim + r_eq * (c1 + c2);
        let chi = transcript.round(&round.coeffs, round.pow)?;
        claim = c0 + chi * (c1 + chi * c2);
        rest.push(chi);
    }

    // Phase 4: the evaluations must reproduce the final claim.
    let [a, b, c] = proof.evals;
    if a * b + c != claim {
        return Err(ZerocheckError::TerminalMismatch);
    }
    rest.reverse();
    Ok(ZerocheckClaims {
        point: QuirkyPoint { skip: z, rest },
        a,
        b,
        c,
    })
}
