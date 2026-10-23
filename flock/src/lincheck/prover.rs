//! The lincheck prover: column weights by a backward walk, the witness folded over its instances, then a product sumcheck.

use alloc::vec::Vec;

use p3_binary_field::FoldedRows;
use p3_challenger::{CanObserve, CanSample};
use p3_field::PrimeCharacteristicRing;
use p3_maybe_rayon::prelude::*;
use p3_multilinear_util::poly::Poly;
use p3_sumcheck::lagrange::extrapolate_01inf;
use p3_verifier_arith::Native;
use tracing::{info_span, instrument};

use super::sumcheck::ProductSumcheck;
use super::transcript::{LincheckShape, ProverTranscript};
use super::{LincheckProof, slice_claim, split_point};
use crate::point::{QuirkyClaim, QuirkyPoint};
use crate::ports::PublicPorts;
use crate::skip::{K_SKIP, SKIP_SIZE, lagrange_weights};
use crate::zerocheck::ZerocheckClaims;
use crate::{Batch, Challenge, Val, Witness};

/// The row weights `L_v(skip) eq(inner, i)` at row `64 i + v`, a high table times a low one, so no table spans the rows.
struct RowWeights {
    /// `eq` of the high inner coordinates.
    hi: Vec<Challenge>,
    /// `eq` of the low inner coordinates, times the Lagrange weights of the skip.
    lo: Vec<Challenge>,
    /// Row bits the low table spans.
    shift: usize,
}

impl RowWeights {
    /// The weights of a point's last `vars` multilinear coordinates.
    fn new(point: &QuirkyPoint, vars: usize) -> Self {
        let inner = &point.rest[point.rest.len() - vars..];
        let (hi, lo) = inner.split_at(vars - vars / 2);
        let lagrange = lagrange_weights(&point.skip, SKIP_SIZE);
        let lo = Poly::new_from_point(lo, Challenge::ONE)
            .as_slice()
            .iter()
            .flat_map(|&e| lagrange.iter().map(move |&l| e * l))
            .collect();
        Self {
            hi: Poly::new_from_point(hi, Challenge::ONE).into_evals(),
            lo,
            shift: vars / 2 + K_SKIP,
        }
    }

    /// The weight of row `s`.
    #[inline]
    fn at(&self, s: usize) -> Challenge {
        self.hi[s >> self.shift] * self.lo[s & ((1 << self.shift) - 1)]
    }

    /// Every weight times `scale`, filled in parallel.
    fn table(&self, scale: Challenge) -> Vec<Challenge> {
        let lo_len = self.lo.len();
        let mut out = Challenge::zero_vec(self.hi.len() * lo_len);
        out.par_chunks_mut(lo_len)
            .zip(self.hi.par_iter())
            .for_each(|(chunk, &h)| {
                let h = scale * h;
                for (o, &l) in chunk.iter_mut().zip(&self.lo) {
                    *o = h * l;
                }
            });
        out
    }
}

/// Reduce the zerocheck's claims on a batch, and its public words, to one claim on `z`.
///
/// # Panics
///
/// - The witness, the claims or the public words are not the batch's.
pub fn prove<C>(
    batch: &Batch<'_>,
    witness: &Witness,
    claims: &ZerocheckClaims,
    public: &PublicPorts,
    challenger: &mut C,
) -> (LincheckProof, QuirkyClaim)
where
    C: CanObserve<Val> + CanSample<Val>,
{
    let (proof, claim, _) = prove_folded(batch, witness.z(), claims, public, challenger);
    (proof, claim)
}

/// As [`prove`] over the witness bits alone, also returning them folded over the instances.
///
/// Entry `64 i + v` of the fold is bit `v` of inner row `i`, before the sumcheck binds anything.
///
/// # Panics
///
/// - The witness, the claims or the public words are not the batch's.
#[instrument(skip_all, name = "lincheck prove", fields(log_n = batch.log_n()))]
pub fn prove_folded<'z, C>(
    batch: &Batch<'_>,
    z: &'z [u64],
    claims: &ZerocheckClaims,
    public: &PublicPorts,
    challenger: &mut C,
) -> (LincheckProof, QuirkyClaim, FoldedRows<'z>)
where
    C: CanObserve<Val> + CanSample<Val>,
{
    let circuit = batch.circuit();
    assert_eq!(
        z.len(),
        1 << (batch.log_n() - K_SKIP),
        "the witness is the batch's"
    );
    let (outer, inner) = split_point(&claims.point, batch).expect("the claims are the batch's");
    let shape = LincheckShape::new(batch, public);
    let mut transcript = ProverTranscript::new(challenger, &shape);

    // Phase 1: alpha batches A, B, C, the constant pin and the public words.
    let (ports, alpha) = transcript.claims(
        &claims.point,
        [claims.a, claims.b, claims.c],
        shape.port_coordinates(),
    );
    let alpha_sq = alpha.square();

    // Phase 2: the column weights of A_0 + alpha B_0 by one backward walk, onto C = I and the pin.
    let rows = RowWeights::new(&claims.point, inner.len());
    let mut comb = info_span!("identity weights").in_scope(|| rows.table(alpha_sq));
    info_span!("fold transpose").in_scope(|| {
        circuit.add_fold_transpose_by(Challenge::ONE, alpha, |s| rows.at(s), &mut comb);
    });
    comb[circuit.const_pos()] += alpha_sq * alpha;

    // Each public word's weight at its slots, at the port skip coordinate.
    if let Some((&skip, region)) = ports.split_first() {
        let lagrange = lagrange_weights(&skip, SKIP_SIZE);
        let alpha_4 = alpha_sq.square();
        for (w, weight) in public.weights(&mut Native::<Val, Challenge>::new(), region) {
            for (c, &l) in comb[SKIP_SIZE * w..][..SKIP_SIZE].iter_mut().zip(&lagrange) {
                *c += alpha_4 * weight * l;
            }
        }
    }

    // Phase 3: the witness folded at the instance coordinates, read from the bits until the first bind when it can be.
    let eq_outer = Poly::new_from_point(outer, Challenge::ONE);
    let folded = info_span!("partial fold").in_scope(|| {
        FoldedRows::new(
            z,
            circuit.k_log(),
            circuit.useful_bits(),
            eq_outer.as_slice(),
        )
    });

    // Phase 4: the product sumcheck over the inner variables past the skip, top first.
    let rounds = shape.rounds();
    let mut sumcheck = ProductSumcheck::folded(comb, &folded);
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
        folded,
    )
}
