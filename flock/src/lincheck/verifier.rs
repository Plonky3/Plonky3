//! The lincheck verifier: replay the product sumcheck, then close it by one forward walk of the circuit.

use alloc::vec::Vec;

use p3_challenger::{CanObserve, CanSample};
use p3_field::PrimeCharacteristicRing;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use p3_sumcheck::lagrange::extrapolate_01inf;
use tracing::instrument;

use super::transcript::{LincheckShape, VerifierTranscript};
use super::{LincheckError, LincheckProof, slice_claim, split_point};
use crate::point::QuirkyClaim;
use crate::ports::PublicPorts;
use crate::skip::{SKIP_SIZE, interpolate, lagrange_weights};
use crate::zerocheck::ZerocheckClaims;
use crate::{Batch, Challenge, Val};

/// Check a lincheck, with the batch's public words, and return the claim on `z` the caller must still open.
///
/// The public words cost work linear in their table.
///
/// # Errors
///
/// A point, round or slice count that does not fit, or a final value the circuit or the public words miss.
#[instrument(skip_all, name = "lincheck verify", fields(log_n = batch.log_n()))]
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
    let mut transcript = VerifierTranscript::new(challenger, &shape);
    let (ports, alpha) = transcript.claims(
        &claims.point,
        [claims.a, claims.b, claims.c],
        shape.port_coordinates(),
    )?;
    let (alpha_sq, alpha_cube) = (alpha.square(), alpha.cube());
    let alpha_4 = alpha_sq.square();

    // The target: the batched claims, the pin's target one, the public words' sum computed here.
    let mut running = claims.a + alpha * claims.b + alpha_sq * claims.c + alpha_cube;
    if let Some((&skip, region)) = ports.split_first() {
        running += alpha_4 * public.evaluate(outer, skip, region);
    }
    let mut point = Vec::with_capacity(rounds);
    for &[q0, q_inf] in &proof.rounds {
        let r = transcript.round([q0, q_inf]);
        running = extrapolate_01inf(q0, running + q0, q_inf, r);
        point.push(r);
    }
    let skip = transcript.slices(&proof.slices)?;

    // The terminal identity: the circuit's form, then C and the pin in closed form.
    let s = &proof.slices;
    let eq_inner = claims.point.eq_weights(inner.len());
    let columns: Vec<Challenge> = Poly::new_from_point(&point, Challenge::ONE)
        .iter()
        .flat_map(|&e| s.iter().map(move |&v| e * v))
        .collect();
    let circuit = batch.circuit();
    let mut form = Challenge::ZERO;
    circuit.walk_rows(&columns, |row, a, b| {
        form += eq_inner[row] * (a + alpha * b);
    });
    let c_term = Point::eval_eq(inner, &point) * interpolate(claims.point.skip, s);
    // Column `64 i + v` is `eq(point, i) s[v]`, so the pin's column is its term.
    let pin_term = columns[circuit.const_pos()];
    let port_term = ports
        .split_first()
        .map_or(Challenge::ZERO, |(&skip, region)| {
            let lagrange = lagrange_weights(skip, SKIP_SIZE);
            public
                .weights(region)
                .map(|(w, weight)| {
                    let slots = &columns[SKIP_SIZE * w..][..SKIP_SIZE];
                    weight
                        * slots
                            .iter()
                            .zip(&lagrange)
                            .map(|(&c, &l)| c * l)
                            .sum::<Challenge>()
                })
                .sum()
        });
    if running != form + alpha_sq * c_term + alpha_cube * pin_term + alpha_4 * port_term {
        transcript.abort();
        return Err(LincheckError::FinalCheck);
    }
    transcript.finish();
    Ok(slice_claim(skip, outer, &point, s))
}
