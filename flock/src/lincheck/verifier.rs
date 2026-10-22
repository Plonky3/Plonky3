//! The lincheck verifier: replay the product sumcheck, then close it by one forward walk of the circuit.

use alloc::vec::Vec;

use p3_challenger::{CanObserve, CanSample};
use p3_field::{Dup, PrimeCharacteristicRing};
use p3_multilinear_util::point::Point;
use p3_sumcheck::lagrange::extrapolate_01inf;
use p3_verifier_arith::Native;
use tracing::instrument;

use super::transcript::{LincheckShape, VerifierTranscript};
use super::{LincheckError, LincheckProof, slice_claim, split_point};
use crate::point::{QuirkyClaim, QuirkyPoint};
use crate::ports::{PublicPorts, eq_at};
use crate::skip::{SKIP_SIZE, interpolate};
use crate::zerocheck::ZerocheckClaims;
use crate::{Batch, FlockArith, Val};

/// Check a lincheck, with the batch's public words, and return the claim on `z` the caller must still open.
///
/// The public words cost work linear in their table.
///
/// # Errors
///
/// A point, round or slice count that does not fit, or a final value the circuit or the public words miss.
pub fn verify<C>(
    batch: &Batch<'_>,
    claims: &ZerocheckClaims,
    public: &PublicPorts,
    proof: &LincheckProof,
    challenger: &mut C,
) -> Result<QuirkyClaim, LincheckError>
where
    C: CanObserve<Val> + CanSample<Val>,
{
    verify_in(&mut Native::new(), batch, claims, public, proof, challenger)
}

/// As [`verify`], within an arithmetic.
///
/// # Errors
///
/// As [`verify`].
#[instrument(skip_all, name = "lincheck verify", fields(log_n = batch.log_n()))]
pub fn verify_in<A, C>(
    arith: &mut A,
    batch: &Batch<'_>,
    claims: &ZerocheckClaims<A::EF>,
    public: &PublicPorts,
    proof: &LincheckProof,
    challenger: &mut C,
) -> Result<QuirkyClaim<A::EF>, LincheckError>
where
    A: FlockArith,
    C: CanObserve<Val> + CanSample<Val>,
{
    let shape = LincheckShape::new(batch, public);
    let rounds = shape.rounds();
    // Every structural check runs before the transcript moves.
    let (outer, inner) = split_point(&claims.point, batch)?;
    public.check(batch)?;
    if proof.rounds.len() != rounds {
        return Err(LincheckError::RoundCount {
            expected: rounds,
            actual: proof.rounds.len(),
        });
    }
    if proof.slices.len() != SKIP_SIZE {
        return Err(LincheckError::SliceCount {
            expected: SKIP_SIZE,
            actual: proof.slices.len(),
        });
    }

    // The input claims are announced, then bound by their native values.
    let evals = [&claims.a, &claims.b, &claims.c];
    for value in claims
        .point
        .coordinates()
        .collect::<Vec<_>>()
        .iter()
        .chain(evals)
    {
        arith.will_observe_ext(value);
    }
    let native_point = QuirkyPoint {
        skip: arith.value_ext(&claims.point.skip),
        rest: claims
            .point
            .rest
            .iter()
            .map(|c| arith.value_ext(c))
            .collect(),
    };
    let native_evals = evals.map(|e| arith.value_ext(e));
    let mut transcript = arith.seeding(|| VerifierTranscript::new(challenger, &shape));
    let (ports, alpha) =
        transcript.claims(&native_point, native_evals, shape.port_coordinates())?;
    let ports: Vec<A::EF> = ports.into_iter().map(|c| arith.sampled_ext(c)).collect();
    let alpha = arith.sampled_ext(alpha);
    let (alpha_sq, alpha_cube) = (alpha.square(), alpha.cube());
    let alpha_4 = alpha_sq.square();

    // The target: the batched claims, the pin's target one, the public words' sum computed here.
    let mut running = claims.a.dup()
        + alpha.dup() * claims.b.dup()
        + alpha_sq.dup() * claims.c.dup()
        + alpha_cube.dup();
    if let Some((skip, region)) = ports.split_first() {
        running += alpha_4.dup() * public.evaluate(arith, outer, skip, region);
    }
    let mut point = Vec::with_capacity(rounds);
    for &[q0, q_inf] in &proof.rounds {
        let (lifted_0, lifted_inf) = (arith.load_ext(q0), arith.load_ext(q_inf));
        arith.will_observe_ext(&lifted_0);
        arith.will_observe_ext(&lifted_inf);
        let r = arith.sampled_ext(transcript.round([q0, q_inf]));
        running = extrapolate_01inf(lifted_0.dup(), running + lifted_0, lifted_inf, r.dup());
        point.push(r);
    }
    let s = arith.load_ext_slice(&proof.slices).into_owned();
    for slice in &s {
        arith.will_observe_ext(slice);
    }
    let skip = arith.sampled_ext(transcript.slices(&proof.slices)?);

    // The terminal identity: the circuit's form, then C, the pin and the public words in closed form.
    let circuit = batch.circuit();
    let form = arith.circuit_form(circuit, &claims.point.skip, inner, &point, &s, &alpha);
    let c_term = Point::eval_eq(inner, &point) * interpolate(&claims.point.skip, &s);
    // Column `64 j + v` weighs `eq(point, j) s[v]`.
    let pin = circuit.const_pos();
    let pin_term = eq_at(arith, &point, &[pin / 64]).remove(0) * s[pin % 64].dup();
    let port_term = match ports.split_first() {
        Some((skip, region)) => {
            let bits = public.region_bits();
            let (high, low) = point.split_at(point.len() - bits);
            let outside: A::EF = high.iter().map(|r| A::EF::ONE + r.dup()).product();
            let words = eq_at(arith, low, public.words());
            let weighted: A::EF = public
                .weights(arith, region)
                .into_iter()
                .zip(words)
                .map(|((_, weight), word)| weight * word)
                .sum();
            outside * weighted * interpolate(skip, &s)
        }
        None => A::EF::ZERO,
    };
    let expected = form + alpha_sq * c_term + alpha_cube * pin_term + alpha_4 * port_term;
    if let Err(error) = arith.assert_zero(running - expected, || LincheckError::FinalCheck) {
        transcript.abort();
        return Err(error);
    }
    transcript.finish();
    Ok(slice_claim(skip, outer, &point, &s))
}
