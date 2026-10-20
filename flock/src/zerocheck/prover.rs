//! The zerocheck prover: round 1 by a sweep over the skip windows, then two multilinear rounds per pass.

use alloc::vec::Vec;
use core::mem;

use p3_binary_field::{PackedGf2x64, Poly64, Poly192};
use p3_challenger::{CanObserve, CanSample, GrindingChallenger};
use p3_field::{Field, PrimeCharacteristicRing};
use p3_multilinear_util::poly::Poly;
use tracing::{info_span, instrument};

use super::fixed::eq_coordinates;
use super::fold::BitFold;
use super::multilinear::{
    Bits, RoundPair, bit_pass, bit_pass_storing, round_at_zero, single_round, table_pass,
};
use super::round1::round1_message;
use super::transcript::{ProverTranscript, ZerocheckShape};
use super::{MIN_LOG_N, Padding, ZerocheckClaims, ZerocheckProof, ZerocheckRound};
use crate::point::QuirkyPoint;
use crate::skip::{K_SKIP, SKIP_SIZE, SkipLde, lagrange_weights, round1_at};
use crate::{Batch, Witness};

/// Bit passes, two rounds each, before the pass that stores the folded tables.
///
/// A bit pass re-reads the bits, so storing pays once the tables are well below them: at level 4.
const PAIR_PASSES: usize = 2;

// The storing pass sends two rounds too, so the bit passes fit the smallest cube.
const _: () = assert!(2 * PAIR_PASSES + 2 <= MIN_LOG_N - K_SKIP);

/// Smallest folded table a paired table pass takes; below it one round at a time is faster.
const PAIRED_MIN: usize = 1 << 10;

/// The folded tables, from the pass that stores them on.
///
/// The tables lag the rounds: `pending` holds the challenges sent but not yet folded in.
struct Folded {
    /// The `a`, `b` and `c` tables.
    t: [Poly<Poly192>; 3],
    /// Ping-pong scratch a pass writes its folded tables into.
    nxt: [Vec<Poly192>; 3],
    /// The challenges not yet folded in, lowest first.
    pending: Vec<Poly192>,
}

/// The multilinear rounds' prover, stepped one round at a time.
struct RoundProver<'a> {
    /// The packed bits.
    bits: Bits<'a>,
    /// Where the witness is zero.
    padding: Padding,
    /// The eq coordinates past the skip.
    r: &'a [Poly192],
    /// The skip's Lagrange weights at its challenge.
    lagrange: Vec<Poly192>,
    /// The challenges bound so far.
    chis: Vec<Poly192>,
    /// The running claim.
    claim: Poly192,
    /// The constant and the sent coefficients of the round just sent.
    message: (Poly192, [Poly192; 2]),
    /// The second round of a pass, waiting on the first's challenge.
    pair: Option<RoundPair>,
    /// The folded tables, once stored.
    folded: Option<Folded>,
}

impl<'a> RoundProver<'a> {
    /// The prover after the skip challenge, its claim round 1 at that challenge.
    fn new(
        bits: Bits<'a>,
        padding: Padding,
        r: &'a [Poly192],
        lagrange: Vec<Poly192>,
        claim: Poly192,
    ) -> Self {
        Self {
            bits,
            padding,
            r,
            lagrange,
            chis: Vec::with_capacity(r.len()),
            claim,
            message: (Poly192::ZERO, [Poly192::ZERO; 2]),
            pair: None,
            folded: None,
        }
    }

    /// The next round's sent coefficients; `G(0)` comes from `(1 + r) G(0) + r G(1) = claim`.
    fn round(&mut self) -> [Poly192; 2] {
        let j = self.chis.len();
        let r_eq = self.r[j];
        let (g0, g1, g_inf) = if let Some(pair) = self.pair.take() {
            let (g1, g_inf) = pair.second(self.chis[j - 1]);
            (None, g1, g_inf)
        } else if self.folded.is_none() {
            self.bit_round(j)
        } else {
            self.table_round(j)
        };
        let g0 = g0.unwrap_or_else(|| (self.claim + r_eq * g1) * (Poly192::ONE + r_eq).inverse());
        // G(X) = G(0) (1 + X) + G(1) X + G(inf) X (1 + X).
        let coeffs = [g1 + g0 + g_inf, g_inf];
        self.message = (g0, coeffs);
        coeffs
    }

    /// Bind the round just sent at `chi`.
    fn bind(&mut self, chi: Poly192) {
        let (c0, [c1, c2]) = self.message;
        self.claim = c0 + chi * (c1 + chi * c2);
        self.chis.push(chi);
        if let Some(folded) = &mut self.folded {
            folded.pending.push(chi);
        }
    }

    /// The first round of a bit pass; the last bit pass also stores the folded tables.
    fn bit_round(&mut self, j: usize) -> (Option<Poly192>, Poly192, Poly192) {
        let fold = BitFold::at_level(&self.lagrange, &self.chis);
        let r_eq = &self.r[j + 1..];
        let pair = if j < 2 * PAIR_PASSES {
            info_span!("bit pass", level = j)
                .in_scope(|| bit_pass(self.bits, &fold, r_eq, &self.padding))
        } else {
            let (pair, t) = info_span!("storing pass", level = j)
                .in_scope(|| bit_pass_storing(self.bits, &fold, r_eq, &self.padding));
            // The first table pass folds this pass's two challenges, a quarter of the tables out.
            let room = t[0].num_evals() / 4;
            self.folded = Some(Folded {
                t,
                nxt: core::array::from_fn(|_| Vec::with_capacity(room)),
                pending: Vec::with_capacity(2),
            });
            pair
        };
        self.pair = Some(pair);
        (None, pair.first.0, pair.first.1)
    }

    /// A round on the folded tables: the first of a paired pass, or a single round.
    fn table_round(&mut self, j: usize) -> (Option<Poly192>, Poly192, Poly192) {
        let (r, padding) = (self.r, self.padding);
        let folded = self.folded.as_mut().expect("the tables are stored");
        let n_out = folded.t[0].num_evals() >> folded.pending.len();
        let paired = j + 1 < r.len()
            && n_out >= PAIRED_MIN
            && r[j] != Poly192::ONE
            && r[j + 1] != Poly192::ONE;
        if paired {
            let outs = folded.nxt.each_mut().map(|t| {
                t.clear();
                &mut t.spare_capacity_mut()[..n_out]
            });
            // A level-`j` value covers `2^(K_SKIP + j)` witness bits.
            let pair = info_span!("table pass", level = j).in_scope(|| {
                table_pass(
                    folded.t.each_ref().map(Poly::as_slice),
                    outs,
                    &folded.pending,
                    &r[j + 1..],
                    &padding,
                    K_SKIP + j,
                )
            });
            for (t, nxt) in folded.t.iter_mut().zip(&mut folded.nxt) {
                // SAFETY: the pass wrote the first `n_out` slots of every scratch table.
                unsafe { nxt.set_len(n_out) };
                let stale = mem::replace(t, Poly::new(mem::take(nxt)));
                *nxt = stale.into_evals();
            }
            folded.pending.clear();
            self.pair = Some(pair);
            return (None, pair.first.0, pair.first.1);
        }

        // One round at a time: fold each pending challenge, then sum the round.
        for &rho in &folded.pending {
            for t in &mut folded.t {
                t.fix_suffix_var_mut(rho);
            }
        }
        folded.pending.clear();
        let tables = folded.t.each_ref().map(Poly::as_slice);
        let r_eq = &r[j + 1..];
        let (g1, g_inf) = single_round(tables, r_eq);
        // An eq coordinate of one leaves `G(0)` out of the claim's split: sum it directly.
        let g0 = (r[j] == Poly192::ONE).then(|| round_at_zero(tables, r_eq));
        (g0, g1, g_inf)
    }

    /// `a`, `b` and `c` at the point, once every round is bound.
    ///
    /// The claim fixes `c`, so on any witness the claims are the ones the transcript implies.
    fn finish(self) -> ([Poly192; 3], Vec<Poly192>) {
        let mut folded = self.folded.expect("every prover stores its tables");
        let [a, b, _] = &mut folded.t;
        for &rho in &folded.pending {
            a.fix_suffix_var_mut(rho);
            b.fix_suffix_var_mut(rho);
        }
        let (a, b) = (a.as_slice()[0], b.as_slice()[0]);
        ([a, b, self.claim + a * b], self.chis)
    }
}

/// Prove that a batch's `A z` and `B z` multiply to `z` on every bit, down to claims at one point.
///
/// The witness's `z` is never read; the challenger must already hold the commitment to the witness.
///
/// # Panics
///
/// - The witness is not the batch's.
#[instrument(name = "zerocheck", skip_all, fields(log_n = batch.log_n()))]
pub fn prove<C>(
    batch: &Batch<'_>,
    witness: &Witness,
    pow_bits: usize,
    challenger: &mut C,
) -> (ZerocheckProof, ZerocheckClaims)
where
    C: CanObserve<Poly64> + CanSample<Poly64> + GrindingChallenger<Witness = Poly64>,
{
    let shape = ZerocheckShape::new(batch, pow_bits);
    let m = shape.log_n;
    let words = 1 << (m - K_SKIP);
    assert!(
        witness.az().len() == words && witness.bz().len() == words,
        "the witness is the batch's"
    );
    let [a, b] = [witness.az(), witness.bz()]
        .map(|t| PackedGf2x64::as_bytes(PackedGf2x64::from_bits_slice(t)));
    let bits = Bits { a, b };
    let circuit = batch.circuit();
    let padding = Padding {
        k_log: circuit.k_log(),
        useful_bits: circuit.useful_bits(),
    };
    let mut transcript = ProverTranscript::new(challenger, shape);

    // Phase 1: the eq point past the skip, fixed coordinates then drawn ones.
    let r = eq_coordinates(&transcript.outer());

    // Phase 2: round 1 on the coset, and the skip challenge.
    let lde = SkipLde::new();
    let round1 = info_span!("round 1").in_scope(|| round1_message(a, b, m, &r, &lde, &padding));
    let (z, round1_pow) = transcript.round1(&round1);

    // Phase 3: the multilinear rounds, from round 1's value at the skip challenge.
    let mut prover = RoundProver::new(
        bits,
        padding,
        &r,
        lagrange_weights(z, SKIP_SIZE),
        round1_at(z, &round1),
    );
    let span = info_span!("rounds").entered();
    let rounds = (0..r.len())
        .map(|_| {
            let coeffs = prover.round();
            let (chi, pow) = transcript.round(&coeffs);
            prover.bind(chi);
            ZerocheckRound { coeffs, pow }
        })
        .collect();
    drop(span);

    // Phase 4: the evaluations, bound before whatever consumes them.
    let (evals, mut chis) = prover.finish();
    transcript.finish(&evals);
    chis.reverse();
    let [ea, eb, ec] = evals;
    let claims = ZerocheckClaims {
        point: QuirkyPoint {
            skip: z,
            rest: chis,
        },
        a: ea,
        b: eb,
        c: ec,
    };
    let proof = ZerocheckProof {
        round1,
        round1_pow,
        rounds,
        evals,
    };
    (proof, claims)
}
