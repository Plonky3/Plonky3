//! The lincheck prover: column weights by a backward walk, the witness folded over its instances, then a product sumcheck.

use alloc::vec::Vec;

use p3_challenger::{CanObserve, CanSample};
use p3_field::PrimeCharacteristicRing;
use p3_multilinear_util::poly::Poly;
use p3_sumcheck::lagrange::extrapolate_01inf;
use tracing::{info_span, instrument};

use super::fold::partial_fold;
use super::sumcheck::ProductSumcheck;
use super::transcript::{LincheckShape, ProverTranscript};
use super::{LincheckProof, slice_claim, split_point};
use crate::point::QuirkyClaim;
use crate::skip::K_SKIP;
use crate::zerocheck::ZerocheckClaims;
use crate::{Batch, Challenge, Val, Witness};

/// Reduce the zerocheck's claims on a batch to one claim on `z`.
///
/// # Panics
///
/// - The witness or the claims are not the batch's.
#[instrument(skip_all, name = "lincheck prove", fields(log_n = batch.log_n()))]
pub fn prove<C>(
    batch: &Batch<'_>,
    witness: &Witness,
    claims: &ZerocheckClaims,
    challenger: &mut C,
) -> (LincheckProof, QuirkyClaim)
where
    C: CanObserve<Val> + CanSample<Val>,
{
    let circuit = batch.circuit();
    let z = witness.z();
    assert_eq!(
        z.len(),
        1 << (batch.log_n() - K_SKIP),
        "the witness is the batch's"
    );
    let (outer, inner) = split_point(&claims.point, batch).expect("the claims are the batch's");
    let shape = LincheckShape::new(batch);
    let mut transcript = ProverTranscript::new(challenger, &shape);

    // Phase 1: alpha batches A, B, C and the constant pin.
    let alpha = transcript.claims(&claims.point, [claims.a, claims.b, claims.c]);
    let alpha_sq = alpha.square();

    // Phase 2: the column weights of A_0 + alpha B_0 by one backward walk, then C = I and the pin.
    let eq_inner = info_span!("eq table").in_scope(|| claims.point.eq_weights(inner.len()));
    let mut comb = Challenge::zero_vec(eq_inner.len());
    info_span!("fold transpose")
        .in_scope(|| circuit.add_fold_transpose(Challenge::ONE, alpha, &eq_inner, &mut comb));
    for (c, &e) in comb.iter_mut().zip(&eq_inner) {
        *c += alpha_sq * e;
    }
    comb[circuit.const_pos()] += alpha_sq * alpha;

    // Phase 3: the witness folded at the instance coordinates.
    let folded = info_span!("partial fold").in_scope(|| {
        let eq_outer = Poly::new_from_point(outer, Challenge::ONE);
        partial_fold(
            z,
            circuit.k_log(),
            circuit.useful_bits(),
            eq_outer.as_slice(),
        )
    });

    // Phase 4: the product sumcheck over the inner variables past the skip, top first.
    let rounds = shape.rounds();
    let mut sumcheck = ProductSumcheck::new(comb, folded);
    let mut running = sumcheck.claim();
    let mut messages = Vec::with_capacity(rounds);
    let mut point = Vec::with_capacity(rounds);
    let mut next = if rounds > 0 {
        sumcheck.round()
    } else {
        (Challenge::ZERO, Challenge::ZERO)
    };
    info_span!("product sumcheck").in_scope(|| {
        for t in 0..rounds {
            // q(0) + q(1) = claim in characteristic two.
            let (q1, q_inf) = next;
            let q0 = running + q1;
            messages.push([q0, q_inf]);
            let r = transcript.round([q0, q_inf]);
            running = extrapolate_01inf(q0, q1, q_inf, r);
            point.push(r);
            if t + 1 < rounds {
                next = sumcheck.bind_and_round(r);
            } else {
                sumcheck.bind_last(r);
            }
        }
    });

    // Phase 5: the slices, then one claim at a fresh skip coordinate.
    let slices = sumcheck.into_z();
    let skip = transcript.finish(&slices);
    let claim = slice_claim(skip, outer, &point, &slices);
    (
        LincheckProof {
            rounds: messages,
            slices,
        },
        claim,
    )
}
