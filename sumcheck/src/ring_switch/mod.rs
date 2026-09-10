//! Ring switching (IACR eprint 2024/504, Construction 3.1): a reduction from an evaluation
//! claim about a small-field multilinear to a claim about its packed extension-field
//! multilinear.
//!
//! The prover holds `t` over `F` and its packing `t'` over `EF`, and the verifier holds a claim
//! `t(r) = s`. Splitting `r = (r_high, r_low)` at the `κ = log2([EF:F])` variables one packed
//! element absorbs, the reduction runs one degree-2 sumcheck of `ℓ' = ℓ − κ` rounds and leaves
//! a claim `t'(r') = s'` about the packed polynomial alone. Discharging that claim against a
//! commitment to `t'` is the caller's business.
//!
//! The tensor element `ŝ` is the only prover message the two checks share.
//!
//! ```text
//!     read by columns  ->  tests the incoming claim
//!     read by rows     ->  gives the sumcheck its initial sum
//! ```
//!
//! The verifier derives that initial sum itself.
//!
//! It never takes the sum from the prover.
//!
//! # A reduction, not a filter
//!
//! A false input claim is not rejected outright.
//!
//! It survives as a false surviving claim `t'(r') = s'`.
//!
//! The probability of anything else is bounded by the error below.
//!
//! # Two things this does not catch
//!
//! The degenerate case `A(r') = 0` is one.
//!
//! There the final check reads `sum == 0`.
//!
//! That constrains `s'` not at all.
//!
//! A false input claim then survives as a *true* surviving claim.
//!
//! That is the one direction discharging `s'` against a commitment cannot catch.
//!
//! The `2ℓ'/|EF|` term of the bound covers it.
//!
//! The second is a prover who tampers with `ŝ`.
//!
//! Adapting the rest of the proof to the implied sum defeats the checks here.
//!
//! Only the caller catches it, by discharging `s'` against a commitment to `t'`.
//!
//! # What the row reading buys
//!
//! It buys the `κ/|EF|` term of the bound.
//!
//! A tampered `ŝ` shifts the derived initial sum away from an honest packing's value.
//!
//! It does so at the Schwartz–Zippel rate of the batching draw.
//!
//! # Soundness
//!
//! The reduction's own soundness error is `(κ + 2ℓ') / |EF|` (eprint 2024/504 §3.2, Theorem
//! 3.5), split as:
//!
//! - `κ/|EF|` from the Schwartz–Zippel argument on the `r''` batching draw, which collapses
//!   the `2^κ` row claims into one — the draw names `κ` variables, which is where the `κ`
//!   comes from; and
//! - `2ℓ'/|EF|` for the `ℓ'` rounds of degree-2 sumcheck.
//!
//! Both terms are stated for the interactive protocol. This module is Fiat–Shamir compiled and
//! offers no grinding knob — `pow_bits` is zero on both sides and a proof carrying PoW
//! witnesses is rejected — so the bound as a whole is a per-attempt probability rather than a
//! total one: the `r''` draw and every round challenge alike are functions of messages the
//! prover chooses, and a prover may resample any of them. That is out of reach at
//! `|EF| = 2^128`; a composing protocol instantiating this over a small `EF` must supply the
//! grinding itself, outside this call.
//!
//! `r` is bound by the reduction. Both sides observe it before `ŝ`, so the bound above holds
//! unconditionally rather than resting on the caller having bound `r` first. Without that
//! binding the proof would be replayable: `ŝ` is a function of `r_high` alone, so the same
//! proof verifies verbatim at every `r_low`, and the verifier's one `r_high`-dependent test —
//! `A(r')` in the final check — is multi-affine over `EF` in the coordinates of `r_high`. Move
//! a single coordinate by `δ` and the change it induces in `A(r')` is `F`-linear in `δ`, so
//! wherever that map is singular a prover can pick a nonzero `δ` from its kernel and have a
//! false input claim accepted alongside a *true* surviving claim, which is the one direction a
//! caller discharging `s'` against a commitment cannot catch.
//!
//! That figure is wired into **no** soundness estimator: nothing here touches `p3-security`,
//! and neither the reduction nor its callers subtract it from any security target. A caller
//! composing this reduction into a proof system must account for it in that system's own
//! error budget, on top of whatever the commitment scheme discharging `t'(r') = s'` costs.

use p3_challenger::fs::TranscriptField;
use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, Field};
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::data::SumcheckData;
use crate::error::SumcheckError;
use crate::product_polynomial::ProductPolynomial;
use crate::strategy::{Basis, SumcheckProver, VariableOrder};

pub mod equality;
pub mod packing;
pub mod tensor;
pub mod transcript;
pub mod weights;

pub use equality::equality_element;
use packing::compute_s_hat_with_eq;
pub use packing::{compute_s_hat, pack, packed_vars};
pub use tensor::TensorAlgebra;
pub use transcript::{RingSwitchProverTranscript, RingSwitchShape, RingSwitchVerifierTranscript};
use weights::batched_weights_with_eq;
pub use weights::{batch_rows, batched_weights};

#[cfg(test)]
pub(crate) mod test_util;

/// The messages one ring-switching reduction puts on the wire.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(
    serialize = "F: Field, EF: ExtensionField<F>",
    deserialize = "F: Field, EF: ExtensionField<F>"
))]
pub struct RingSwitchProof<F, EF> {
    /// `ŝ = Σ_w eq̃(r_high, w) ⊗ t'(w)`, read by columns and by rows.
    pub s_hat: TensorAlgebra<F, EF>,
    /// The `ℓ'` rounds of the batched degree-2 sumcheck.
    pub sumcheck: SumcheckData<F, EF>,
    /// `s' = t'(r')`, the value of the surviving claim.
    pub final_eval: EF,
}

/// Why a ring-switching proof was rejected.
#[derive(Error, Debug, PartialEq, Eq)]
pub enum RingSwitchError {
    /// `ŝ` does not carry `DIMENSION²` base coefficients, so neither of its readings is defined.
    #[error("Ring switching: s_hat carries {actual} base coefficients, expected {expected}")]
    MalformedTensor {
        /// The number of coefficients a tensor element must carry.
        expected: usize,
        /// The number the proof supplied.
        actual: usize,
    },

    /// The evaluation point does not name the coordinate count the description fixes.
    ///
    /// Neither side can then bind it as the step the description names.
    #[error("Ring switching: the evaluation point names {actual} coordinates, expected {expected}")]
    PointWidthMismatch {
        /// The coordinate count the description fixes.
        expected: usize,
        /// The number the point supplied.
        actual: usize,
    },

    /// The claimed evaluation is not the `eq̃(·, r_low)`-combination of `ŝ`'s columns.
    #[error("Ring switching: the claimed evaluation is not the column reading of s_hat")]
    ClaimMismatch,

    /// A round of the batched sumcheck failed.
    #[error(transparent)]
    Sumcheck(#[from] SumcheckError),

    /// The surviving claim does not close the sumcheck against the equality element.
    #[error("Ring switching: the surviving claim does not close the sumcheck")]
    FinalCheck,

    /// The sumcheck carries PoW witnesses this reduction never grinds for, so a proof carrying
    /// junk witnesses would otherwise verify identically to one carrying none.
    #[error("Ring switching: the sumcheck carries {actual} PoW witnesses, expected none")]
    NonEmptyPowWitnesses {
        /// The number of witnesses the proof supplied.
        actual: usize,
    },
}

/// Proves the reduction of `t(r) = s` to a claim about `packed`, the packing of `t`.
///
/// # Returns
///
/// - The proof.
/// - The sumcheck's random point `r'`.
/// - The surviving claim's value `s' = t'(r')`.
///
/// # Why only the packed polynomial
///
/// Every value that reaches the proof is computed from it.
///
/// The claim it proves is a statement about it.
///
/// Callers hold `t` and pack it themselves.
///
/// Pack the wrong `t` and the claim proves a statement about a different multilinear.
///
/// Nothing here can detect that.
///
/// The caller's commitment to `packed` is what pins it down.
///
/// # Packing
///
/// The sumcheck runs on a scalar product polynomial.
///
/// ```text
///     binary tower       ->  extension packing is the extension itself, nothing to gain
///     wider packing      ->  throughput left unclaimed
/// ```
///
/// # What the transcript binds
///
/// The evaluation point and the surviving claim are both bound here.
///
/// ```text
///     r            absorbed before s_hat
///     final_eval   absorbed after the sumcheck rounds
/// ```
///
/// A caller composing this into a larger protocol need not bind `r` itself.
///
/// # Panics
///
/// Unless `r` names at least the `κ` packed variables.
///
/// Unless `packed` has exactly the `ℓ − κ` variables that leaves.
pub fn prove_ring_switch<F, EF, Challenger>(
    packed: &Poly<EF>,
    r: &Point<EF>,
    challenger: &mut Challenger,
) -> (RingSwitchProof<F, EF>, Point<EF>, EF)
where
    F: TranscriptField,
    EF: ExtensionField<F>,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    // The description is derived from the point handed in, never from a proof.
    //
    // Every width the transcript declares follows from it and from the field pair:
    //
    //     point width      ->  the coordinates `r` names
    //     tensor width     ->  the extension degree, squared
    //     batching width   ->  log2 of the extension degree
    let shape = RingSwitchShape::new(r.num_variables());
    let ell_prime = shape.sumcheck_rounds::<F, EF>();
    assert_eq!(
        packed.num_variables(),
        ell_prime,
        "the packed polynomial must have the {ell_prime} variables the evaluation point leaves \
         once the packed ones are removed"
    );
    let (r_high, _) = r.split_at(ell_prime);

    // `ŝ` and the weight multilinear are both readings of the same `eq̃(r_high, ·)` table, so it
    // is built once, shared, and freed before the rounds rather than held across them.
    let eq = Poly::<EF>::new_from_point(r_high.as_slice(), EF::ONE);

    // `ŝ` is a function of `r_high` alone.
    //
    // It is computed before the transcript needs it.
    let s_hat = compute_s_hat_with_eq::<F, EF>(packed, &eq);

    // Seeding folds the description's fingerprint into the borrowed sponge.
    let mut transcript = RingSwitchProverTranscript::<Challenger, F, EF>::new(challenger, shape);

    // One call binds the point, binds `ŝ`'s base coefficients, and draws `r''`.
    //
    // The point goes first.
    //
    // Every later check the verifier forms from `r` it forms locally.
    //
    // An unbound `r` would leave the proof replayable at a different point.
    //
    // What is bound for `ŝ` is the coefficients that cross the wire.
    //
    // Neither derived reading is bound directly.
    //
    // Both are still pinned: both are readings of that one step.
    let r_batch = transcript.statement(r, s_hat.coefficients());

    // `ℓ'` rounds on `h(X) = A(X) · t'(X)`, whose sum over the hypercube is the batched row
    // reading of `ŝ`.
    let weights = batched_weights_with_eq::<F, EF>(&eq, &r_batch);
    drop(eq);
    let poly = ProductPolynomial::new_unpacked(VariableOrder::Prefix, packed.clone(), weights);
    let mut prover = SumcheckProver::new(poly, batch_rows::<F, EF>(&s_hat, &r_batch));
    let mut sumcheck = SumcheckData::default();

    // The rounds are a sub-protocol.
    //
    // They seed a transcript of their own from this sponge.
    //
    // The bracket records the delegation and hands the sponge over for it.
    let r_prime = transcript.batched_sumcheck(|challenger| {
        prover.compute_sumcheck_polynomials(&mut sumcheck, challenger, ell_prime, 0, None)
    });

    // After the last round the evaluation side has been folded to a single value, which is
    // `t'(r')` — the same number a fresh `packed.eval_ext(&r_prime)` would recompute in
    // another full pass over the packed evaluations.
    let final_eval = prover.evals().as_slice()[0];

    // The surviving claim is bound before the sponge goes back.
    //
    // A caller discharging it discharges the value this run produced.
    transcript.surviving_claim(final_eval);
    transcript.finish();

    (
        RingSwitchProof {
            s_hat,
            sumcheck,
            final_eval,
        },
        r_prime,
        final_eval,
    )
}

/// Verifies the reduction of `claimed_sum = t(r)`.
///
/// # Returns
///
/// The surviving claim `t'(r') = s'`, as the pair `(r', s')`.
///
/// # Where the round count comes from
///
/// It is taken from `r`, the point the verifier owns.
///
/// A proof carrying a different number of rounds is rejected by the round replay.
///
/// # What the transcript binds
///
/// ```text
///     r            absorbed before s_hat
///     final_eval   absorbed after the rounds
/// ```
///
/// Both land at the same two points the prover uses.
///
/// A proof is therefore usable only against the point it was produced for.
///
/// The claimed sum is bound indirectly, by the column check against `ŝ`.
///
/// Both structural rejections below happen before the challenger is touched.
///
/// A malformed proof cannot leave a half-advanced transcript.
///
/// # Panics
///
/// Unless `r` names at least the `κ` packed variables.
///
/// Everything read out of the proof is validated and reported as an error instead.
///
/// # Errors
///
/// - A malformed `ŝ`.
/// - A non-empty proof-of-work witness list, since this reduction never grinds.
/// - A claim that disagrees with `ŝ`'s columns.
/// - A failed sumcheck round.
/// - A final claim that does not close the sumcheck.
///
/// The point width is never among them.
///
/// The description is derived from `r` itself.
///
/// The two cannot disagree.
pub fn verify_ring_switch<F, EF, Challenger>(
    proof: &RingSwitchProof<F, EF>,
    r: &Point<EF>,
    claimed_sum: EF,
    challenger: &mut Challenger,
) -> Result<(Point<EF>, EF), RingSwitchError>
where
    F: TranscriptField,
    EF: ExtensionField<F>,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    // The description is derived from the point this side holds, never from the proof.
    //
    // Both sides therefore declare the same widths.
    //
    // A proof disagreeing with any of them is rejected.
    //
    // It is never left to desynchronise the sponge.
    let shape = RingSwitchShape::new(r.num_variables());
    let ell_prime = shape.sumcheck_rounds::<F, EF>();

    // `columns` and `rows` index a `DIMENSION × DIMENSION` matrix, so a short `ŝ` must be
    // rejected before either reading is taken.
    if !proof.s_hat.is_well_formed() {
        return Err(RingSwitchError::MalformedTensor {
            expected: RingSwitchShape::tensor_coefficients::<F, EF>(),
            actual: proof.s_hat.coefficients().len(),
        });
    }

    // This reduction never grinds, so a proof carrying PoW witnesses is carrying data that is
    // neither checked nor bound to the transcript; reject it outright instead of ignoring it.
    if !proof.sumcheck.pow_witnesses.is_empty() {
        return Err(RingSwitchError::NonEmptyPowWitnesses {
            actual: proof.sumcheck.pow_witnesses.len(),
        });
    }

    let (r_high, r_low) = r.split_at(ell_prime);

    // Seeding folds the description's fingerprint into the borrowed sponge.
    let mut transcript = RingSwitchVerifierTranscript::<Challenger, F, EF>::new(challenger, shape);

    // One call absorbs the point, absorbs `ŝ`'s base coefficients, and redraws `r''`.
    //
    // A width the description does not allow is reported here.
    //
    // The failing step releases the driver's completeness check on its way out.
    let r_batch = transcript.statement(r, proof.s_hat.coefficients())?;

    // The incoming claim must be the `eq̃(·, r_low)`-combination of `ŝ`'s columns. This is the
    // only use of `r_low`.
    let eq_low = Poly::<EF>::new_from_point(r_low.as_slice(), EF::ONE);
    let combined: EF = proof
        .s_hat
        .columns()
        .iter()
        .zip(eq_low.as_slice())
        .map(|(&column, &weight)| column * weight)
        .sum();
    if combined != claimed_sum {
        // Two described steps are still unplayed, and this rejection plays neither.
        //
        // Releasing the completeness check keeps this the only failure reported.
        transcript.abort();
        return Err(RingSwitchError::ClaimMismatch);
    }

    // The sumcheck's initial sum is derived from `ŝ`'s rows, never sent, which is what makes a
    // dishonest `ŝ` catchable: the two readings are of the same coefficients.
    let mut sum = batch_rows::<F, EF>(&proof.s_hat, &r_batch);

    // The rounds are a sub-protocol.
    //
    // They seed a transcript of their own from this sponge.
    //
    // The bracket closes whatever the delegated replay returned.
    //
    // A rejection inside it leaves only the closing step unplayed.
    let rounds = transcript.batched_sumcheck(|challenger| {
        proof
            .sumcheck
            .verify_rounds(challenger, &mut sum, ell_prime, 0, Basis::Evaluation)
    });
    let r_prime = match rounds {
        Ok(r_prime) => r_prime,
        Err(error) => {
            transcript.abort();
            return Err(error.into());
        }
    };

    // The surviving claim is absorbed where the prover absorbed it.
    //
    // The sponge is handed back in the state the prover left it in.
    transcript.surviving_claim(proof.final_eval);
    transcript.finish();

    // The batched rows of the equality element are `A(r')`, so the sumcheck closes on
    // `A(r') · t'(r')`.
    let e = equality_element::<F, EF>(&r_high, &r_prime);
    if sum != batch_rows::<F, EF>(&e, &r_batch) * proof.final_eval {
        return Err(RingSwitchError::FinalCheck);
    }

    Ok((r_prime, proof.final_eval))
}

#[cfg(test)]
mod tests {
    use p3_challenger::CanSample;
    use p3_field::PrimeCharacteristicRing;
    use p3_multilinear_util::point::Point;
    use proptest::prelude::*;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use super::test_util::{Challenger, EF, F, base_poly, challenger};
    use super::*;

    /// Prover and verifier agree, and the surviving claim is true.
    #[test]
    fn the_reduction_round_trips() {
        let ell = 7;
        let t = base_poly(ell, 13);
        let packed = pack::<F, EF>(t.clone());
        let r = Point::<EF>::rand(&mut SmallRng::seed_from_u64(14), ell);
        let s = t.eval_base(&r);

        let mut p_chal = challenger();
        let (proof, r_prime_p, s_prime_p) = prove_ring_switch::<F, EF, _>(&packed, &r, &mut p_chal);

        let mut v_chal = challenger();
        let (r_prime_v, s_prime_v) =
            verify_ring_switch::<F, EF, _>(&proof, &r, s, &mut v_chal).unwrap();

        assert_eq!(r_prime_p, r_prime_v);
        assert_eq!(s_prime_p, s_prime_v);
        // The surviving claim is the truth about the committed polynomial.
        assert_eq!(s_prime_v, packed.eval_ext::<F>(&r_prime_v));
    }

    proptest! {
        // Sixteen cases keep the suite fast.
        //
        // Each one runs a full reduction on both sides.
        #![proptest_config(ProptestConfig { cases: 16, ..ProptestConfig::default() })]

        #[test]
        fn the_reduction_round_trips_over_random_inputs(
            ell in 5usize..=8,
            poly_seed in any::<u64>(),
            point_seed in any::<u64>(),
        ) {
            // Completeness: an honest run replays, and the claim it leaves is the truth.
            //
            // Fixture state: 4 packed variables.
            //
            // `ell` between 5 and 8 then leaves 1 to 4 rounds.
            //
            //     ell = 5  ->  32 base evaluations  ->  2 packed elements  ->  1 round
            //     ell = 8  ->  256 base evaluations ->  16 packed elements ->  4 rounds
            let t = base_poly(ell, poly_seed);
            let packed = pack::<F, EF>(t.clone());

            // The point is the verifier's own input, and the claim is the truth at it.
            let r = Point::<EF>::rand(&mut SmallRng::seed_from_u64(point_seed), ell);
            let s = t.eval_base(&r);

            // Both sides start from the same fresh sponge, as a composing protocol would.
            let mut p_chal = challenger();
            let (proof, r_prime_p, s_prime_p) =
                prove_ring_switch::<F, EF, _>(&packed, &r, &mut p_chal);

            let mut v_chal = challenger();
            let (r_prime_v, s_prime_v) =
                verify_ring_switch::<F, EF, _>(&proof, &r, s, &mut v_chal)
                    .expect("an honest reduction must verify");

            // The two sides walked one description.
            //
            // They landed on one surviving claim.
            prop_assert_eq!(&r_prime_p, &r_prime_v);
            prop_assert_eq!(s_prime_p, s_prime_v);

            // The surviving claim is the truth about the committed polynomial.
            prop_assert_eq!(s_prime_v, packed.eval_ext::<F>(&r_prime_v));

            // The sponge is handed back in one state.
            //
            // The caller stays in step afterwards.
            prop_assert_eq!(
                CanSample::<F>::sample(&mut p_chal),
                CanSample::<F>::sample(&mut v_chal),
            );
        }
    }

    /// A wrong claimed value is rejected at the column check.
    #[test]
    fn a_wrong_claim_is_rejected() {
        let ell = 6;
        let t = base_poly(ell, 15);
        let packed = pack::<F, EF>(t.clone());
        let r = Point::<EF>::rand(&mut SmallRng::seed_from_u64(16), ell);
        let s = t.eval_base(&r);

        let mut p_chal = challenger();
        let (proof, _, _) = prove_ring_switch::<F, EF, _>(&packed, &r, &mut p_chal);

        let mut v_chal = challenger();
        assert_eq!(
            verify_ring_switch::<F, EF, _>(&proof, &r, s + EF::ONE, &mut v_chal),
            Err(RingSwitchError::ClaimMismatch)
        );
    }

    /// A `ŝ` perturbed after proving is rejected. The claim handed to the verifier is the
    /// perturbed element's own column reading, formed the way the verifier forms it, so the
    /// column check passes by construction; the rejection comes from the final check instead.
    ///
    /// This does not, on its own, demonstrate that the reduction catches a prover who tampers
    /// with `ŝ` and adapts the rest of the proof to it: perturbing `ŝ` after the honest sumcheck
    /// was already recorded moves the verifier's derived `r''` and every later round challenge
    /// away from the values those recorded messages were generated under, so verification fails
    /// here even in a hypothetical design where the sumcheck's initial sum were prover-supplied.
    /// An adaptive prover — one who tampers with `ŝ` first and then runs the genuine sumcheck
    /// from the sum that tampered `ŝ` implies — is not caught by this check; see the module
    /// docs for what does and does not stop that prover.
    #[test]
    fn a_perturbed_tensor_element_is_rejected() {
        let ell = 6;
        let t = base_poly(ell, 17);
        let packed = pack::<F, EF>(t);
        let r = Point::<EF>::rand(&mut SmallRng::seed_from_u64(18), ell);

        let mut p_chal = challenger();
        let (mut proof, _, _) = prove_ring_switch::<F, EF, _>(&packed, &r, &mut p_chal);
        proof.s_hat.perturb_for_test(0, F::ONE);

        let (_, r_low) = r.split_at(ell - packed_vars::<F, EF>());
        let eq_low = Poly::<EF>::new_from_point(r_low.as_slice(), EF::ONE);
        let tampered_claim: EF = proof
            .s_hat
            .columns()
            .iter()
            .zip(eq_low.as_slice())
            .map(|(&column, &weight)| column * weight)
            .sum();

        let mut v_chal = challenger();
        assert_eq!(
            verify_ring_switch::<F, EF, _>(&proof, &r, tampered_claim, &mut v_chal),
            Err(RingSwitchError::FinalCheck)
        );
    }

    /// The surviving claim is pinned by the final check: perturbing `s'` on an otherwise
    /// honest proof is rejected.
    #[test]
    fn a_perturbed_final_claim_is_rejected() {
        let ell = 6;
        let t = base_poly(ell, 33);
        let packed = pack::<F, EF>(t.clone());
        let r = Point::<EF>::rand(&mut SmallRng::seed_from_u64(34), ell);
        let s = t.eval_base(&r);

        let mut p_chal = challenger();
        let (mut proof, _, _) = prove_ring_switch::<F, EF, _>(&packed, &r, &mut p_chal);
        proof.final_eval += EF::ONE;

        let mut v_chal = challenger();
        assert_eq!(
            verify_ring_switch::<F, EF, _>(&proof, &r, s, &mut v_chal),
            Err(RingSwitchError::FinalCheck)
        );
    }

    /// Perturbing one round message desynchronises every challenge sampled after it, so the
    /// proof fails at the final check rather than being caught round-by-round.
    #[test]
    fn a_perturbed_round_message_is_rejected() {
        let ell = 6;
        let t = base_poly(ell, 35);
        let packed = pack::<F, EF>(t.clone());
        let r = Point::<EF>::rand(&mut SmallRng::seed_from_u64(36), ell);
        let s = t.eval_base(&r);

        let mut p_chal = challenger();
        let (mut proof, _, _) = prove_ring_switch::<F, EF, _>(&packed, &r, &mut p_chal);
        proof.sumcheck.polynomial_evaluations[0][0] += EF::ONE;

        let mut v_chal = challenger();
        assert_eq!(
            verify_ring_switch::<F, EF, _>(&proof, &r, s, &mut v_chal),
            Err(RingSwitchError::FinalCheck)
        );
    }

    /// A proof is bound to the evaluation point it was produced for. `ŝ` is a function of
    /// `r_high` alone, so without `r` in the transcript the same proof would verify verbatim
    /// at every `r_low`, against the honest claim at each of those points — a true statement
    /// each time, but it makes the proof a transferable object rather than one tied to a
    /// point. Observing `r` is what rejects it. The column check still passes here, since the
    /// claim supplied really is `ŝ`'s column reading at the new `r_low`; the rejection comes
    /// from the challenges the observed `r` moved.
    #[test]
    fn a_proof_does_not_verify_against_a_different_low_point() {
        let ell = 6;
        let kappa = packed_vars::<F, EF>();
        let t = base_poly(ell, 37);
        let packed = pack::<F, EF>(t.clone());
        let r = Point::<EF>::rand(&mut SmallRng::seed_from_u64(38), ell);

        let mut p_chal = challenger();
        let (proof, _, _) = prove_ring_switch::<F, EF, _>(&packed, &r, &mut p_chal);

        // Same `r_high`, one coordinate of `r_low` moved, with the honest claim there.
        let (r_high, r_low) = r.split_at(ell - kappa);
        let mut moved = r_low.as_slice().to_vec();
        moved[0] += EF::ONE;
        let mut coords = r_high.as_slice().to_vec();
        coords.extend_from_slice(&moved);
        let other_r = Point::new(coords);
        let other_s = t.eval_base(&other_r);

        let mut v_chal = challenger();
        assert_eq!(
            verify_ring_switch::<F, EF, _>(&proof, &other_r, other_s, &mut v_chal),
            Err(RingSwitchError::FinalCheck)
        );
    }

    /// The `r_high` half is bound too, and this is the arm that matters. The verifier's only
    /// `r_high`-dependent test is `A(r')` in the final check; moving one coordinate of `r_high`
    /// by `δ` changes it by an `F`-linear function of `δ`, and wherever that function is
    /// singular an unbound `r` would let a prover pick a nonzero `δ` from its kernel and have a
    /// false input claim accepted alongside a *true* surviving claim — the one direction a
    /// caller discharging `s'` against a commitment cannot catch. Here the claim is `ŝ`'s
    /// column reading at the unchanged `r_low`, so the column arm passes by construction and
    /// only the transcript can reject.
    #[test]
    fn a_proof_does_not_verify_against_a_different_high_point() {
        let ell = 6;
        let kappa = packed_vars::<F, EF>();
        let t = base_poly(ell, 45);
        let packed = pack::<F, EF>(t.clone());
        let r = Point::<EF>::rand(&mut SmallRng::seed_from_u64(46), ell);
        let s = t.eval_base(&r);

        let mut p_chal = challenger();
        let (proof, _, _) = prove_ring_switch::<F, EF, _>(&packed, &r, &mut p_chal);

        let (r_high, r_low) = r.split_at(ell - kappa);
        let mut moved = r_high.as_slice().to_vec();
        moved[0] += EF::ONE;
        moved.extend_from_slice(r_low.as_slice());
        let other_r = Point::new(moved);

        let mut v_chal = challenger();
        assert_eq!(
            verify_ring_switch::<F, EF, _>(&proof, &other_r, s, &mut v_chal),
            Err(RingSwitchError::FinalCheck)
        );
    }

    /// This reduction never grinds, so a proof carrying PoW witnesses must be rejected rather
    /// than accepted with junk data nothing checks.
    #[test]
    fn a_proof_with_pow_witnesses_is_rejected() {
        let ell = 6;
        let t = base_poly(ell, 40);
        let packed = pack::<F, EF>(t.clone());
        let r = Point::<EF>::rand(&mut SmallRng::seed_from_u64(41), ell);
        let s = t.eval_base(&r);

        let mut p_chal = challenger();
        let (mut proof, _, _) = prove_ring_switch::<F, EF, _>(&packed, &r, &mut p_chal);
        proof.sumcheck.pow_witnesses.push(F::ONE);

        let mut v_chal = challenger();
        assert_eq!(
            verify_ring_switch::<F, EF, _>(&proof, &r, s, &mut v_chal),
            Err(RingSwitchError::NonEmptyPowWitnesses { actual: 1 })
        );
    }

    /// An adaptive prover who tampers with `ŝ` before proving, then runs a genuine sumcheck from
    /// the sum that tampered `ŝ` implies, is accepted by `verify_ring_switch` — with a surviving
    /// claim that is false. This is the reduction working as documented, not a soundness gap: a
    /// false input claim survives as a false surviving claim, caught only when the caller
    /// discharges it against a commitment to the real packed polynomial, which this test does
    /// not do. Contrast with `a_perturbed_tensor_element_is_rejected`, which perturbs `ŝ` after
    /// proving and is rejected for an unrelated reason — transcript desynchronisation.
    #[test]
    fn an_adaptive_cheat_on_s_hat_is_accepted_with_a_false_surviving_claim() {
        let ell = 6;
        let t = base_poly(ell, 43);
        let packed = pack::<F, EF>(t);
        let r = Point::<EF>::rand(&mut SmallRng::seed_from_u64(44), ell);
        let kappa = packed_vars::<F, EF>();
        let ell_prime = ell - kappa;
        let (r_high, r_low) = r.split_at(ell_prime);

        // Tamper with `ŝ` before it ever reaches the transcript, unlike the post-hoc
        // perturbation above.
        let mut s_hat = compute_s_hat::<F, EF>(&packed, &r_high);
        s_hat.perturb_for_test(0, F::ONE);

        // The claim that makes the column check pass for the tampered element, formed the way
        // the verifier forms it.
        let eq_low = Poly::<EF>::new_from_point(r_low.as_slice(), EF::ONE);
        let claimed_sum: EF = s_hat
            .columns()
            .iter()
            .zip(eq_low.as_slice())
            .map(|(&column, &weight)| column * weight)
            .sum();

        // Driven through the same transcript the honest prover uses.
        //
        // The tampered element reaches the sponge where an honest one would.
        let mut chal = challenger();
        let shape = RingSwitchShape::new(ell);
        let mut transcript = RingSwitchProverTranscript::<Challenger, F, EF>::new(&mut chal, shape);
        let r_batch = transcript.statement(&r, s_hat.coefficients());

        // The genuine sumcheck's initial sum, derived from the tampered `ŝ` exactly as the
        // verifier will derive it — not the true dot product of any real polynomial.
        let shifted_sum = batch_rows::<F, EF>(&s_hat, &r_batch);

        // A product polynomial that actually dots to `shifted_sum`: nothing downstream of this
        // point checks that the evaluations are `packed`'s, only that the recorded rounds are
        // internally consistent, so perturbing one evaluation is enough.
        let weights = batched_weights::<F, EF>(&r_high, &r_batch);
        let honest_sum = batch_rows::<F, EF>(&compute_s_hat::<F, EF>(&packed, &r_high), &r_batch);
        let index = weights
            .as_slice()
            .iter()
            .position(|&w| w != EF::ZERO)
            .expect("a generic r_high, r_batch pair leaves no weight identically zero");
        let mut fake_evals = packed.as_slice().to_vec();
        fake_evals[index] += (shifted_sum - honest_sum) * weights.as_slice()[index].inverse();
        let fake_packed = Poly::new(fake_evals);

        let poly = ProductPolynomial::new_unpacked(VariableOrder::Prefix, fake_packed, weights);
        let mut prover = SumcheckProver::new(poly, shifted_sum);
        let mut sumcheck = SumcheckData::default();
        let r_prime = transcript.batched_sumcheck(|challenger| {
            prover.compute_sumcheck_polynomials(&mut sumcheck, challenger, ell_prime, 0, None)
        });
        let sum_final = prover.claimed_sum();

        // `final_eval` set to whatever the final identity needs, computed the same way the
        // verifier computes it.
        let e = equality_element::<F, EF>(&r_high, &r_prime);
        let a_r_prime = batch_rows::<F, EF>(&e, &r_batch);
        let final_eval = sum_final * a_r_prime.inverse();

        // The description is only fully played once the surviving claim is bound.
        transcript.surviving_claim(final_eval);
        transcript.finish();

        let proof = RingSwitchProof {
            s_hat,
            sumcheck,
            final_eval,
        };

        let mut v_chal = challenger();
        let (r_prime_v, s_prime_v) =
            verify_ring_switch::<F, EF, _>(&proof, &r, claimed_sum, &mut v_chal).unwrap();

        assert_eq!(r_prime_v, r_prime);
        assert_eq!(s_prime_v, final_eval);
        assert_ne!(s_prime_v, packed.eval_ext::<F>(&r_prime_v));
    }

    /// The evaluation point must name at least the packed variables, or the split that
    /// separates `r_high` from `r_low` is not defined.
    #[test]
    #[should_panic(expected = "the evaluation point must name at least the 4 packed variables")]
    fn prove_rejects_a_point_shorter_than_the_packed_variables() {
        let t = base_poly(6, 19);
        let packed = pack::<F, EF>(t);
        let r = Point::<EF>::rand(&mut SmallRng::seed_from_u64(20), 3);
        let mut chal = challenger();
        let _ = prove_ring_switch::<F, EF, _>(&packed, &r, &mut chal);
    }

    /// A packed polynomial of the wrong arity must panic rather than be zipped against a
    /// weight table of a different length.
    #[test]
    #[should_panic(expected = "the packed polynomial must have the 2 variables")]
    fn prove_rejects_a_packed_polynomial_of_the_wrong_arity() {
        let t = base_poly(7, 21);
        let packed = pack::<F, EF>(t);
        let r = Point::<EF>::rand(&mut SmallRng::seed_from_u64(22), 6);
        let mut chal = challenger();
        let _ = prove_ring_switch::<F, EF, _>(&packed, &r, &mut chal);
    }

    /// The verifier's own point is subject to the same requirement.
    #[test]
    #[should_panic(expected = "the evaluation point must name at least the 4 packed variables")]
    fn verify_rejects_a_point_shorter_than_the_packed_variables() {
        let ell = 6;
        let t = base_poly(ell, 26);
        let packed = pack::<F, EF>(t);
        let r = Point::<EF>::rand(&mut SmallRng::seed_from_u64(27), ell);

        let mut p_chal = challenger();
        let (proof, _, _) = prove_ring_switch::<F, EF, _>(&packed, &r, &mut p_chal);

        let short = Point::<EF>::rand(&mut SmallRng::seed_from_u64(28), 3);
        let mut v_chal = challenger();
        let _ = verify_ring_switch::<F, EF, _>(&proof, &short, EF::ZERO, &mut v_chal);
    }

    /// The round count is taken from `r`, so a proof carrying a different one is rejected
    /// rather than silently desynchronising the transcript.
    #[test]
    fn verify_rejects_a_proof_with_the_wrong_round_count() {
        let ell = 6;
        let t = base_poly(ell, 29);
        let packed = pack::<F, EF>(t.clone());
        let r = Point::<EF>::rand(&mut SmallRng::seed_from_u64(30), ell);
        let s = t.eval_base(&r);

        let mut p_chal = challenger();
        let (mut proof, _, _) = prove_ring_switch::<F, EF, _>(&packed, &r, &mut p_chal);
        proof.sumcheck.polynomial_evaluations.pop();

        let mut v_chal = challenger();
        assert_eq!(
            verify_ring_switch::<F, EF, _>(&proof, &r, s, &mut v_chal),
            Err(RingSwitchError::Sumcheck(
                SumcheckError::RoundCountMismatch {
                    expected: ell - 4,
                    actual: ell - 5,
                }
            ))
        );
    }

    /// A `ŝ` of the wrong size is reported rather than read out of range: both of its
    /// readings index a `DIMENSION × DIMENSION` matrix.
    #[test]
    fn verify_rejects_a_malformed_tensor() {
        let ell = 6;
        let t = base_poly(ell, 31);
        let packed = pack::<F, EF>(t.clone());
        let r = Point::<EF>::rand(&mut SmallRng::seed_from_u64(32), ell);
        let s = t.eval_base(&r);

        let mut p_chal = challenger();
        let (mut proof, _, _) = prove_ring_switch::<F, EF, _>(&packed, &r, &mut p_chal);
        proof.s_hat.truncate_for_test(1);

        let dimension = TensorAlgebra::<F, EF>::DIMENSION;
        let mut v_chal = challenger();
        assert_eq!(
            verify_ring_switch::<F, EF, _>(&proof, &r, s, &mut v_chal),
            Err(RingSwitchError::MalformedTensor {
                expected: dimension * dimension,
                actual: 1,
            })
        );
    }
}
