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
//! The tensor element `ŝ` is the only prover message the two checks share: the verifier reads
//! it by columns to test the incoming claim, and derives the sumcheck's initial sum from its
//! rows itself rather than take that sum from the prover. That makes this a reduction, not a
//! filter: a false input claim survives as a false surviving claim `t'(r') = s'` rather than
//! being rejected outright, and a prover who tampers with `ŝ` and adapts the rest of the proof
//! to the sum that tampering implies is not caught by [`verify_ring_switch`] — only by the
//! caller discharging `s'` against a commitment to `t'`. What the row reading buys is the
//! `κ/|EF|` term of the soundness bound below: a tampered `ŝ` shifts the derived initial sum
//! away from the value an honest packing would produce, at the Schwartz–Zippel rate of the
//! batching draw.
//!
//! # Soundness
//!
//! The reduction's own soundness error is `(κ + 2ℓ') / |EF|` (eprint 2024/504 §3.2, Theorem
//! 3.5), split as:
//!
//! - `κ/|EF|` from the Schwartz–Zippel argument on the `r''` batching draw, which collapses
//!   the `κ` row claims into one; and
//! - `2ℓ'/|EF|` for the `ℓ'` rounds of degree-2 sumcheck.
//!
//! That figure is wired into **no** soundness estimator: nothing here touches `p3-security`,
//! and neither the reduction nor its callers subtract it from any security target. A caller
//! composing this reduction into a proof system must account for it in that system's own
//! error budget, on top of whatever the commitment scheme discharging `t'(r') = s'` costs.

use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, Field, PrimeCharacteristicRing};
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
pub mod weights;

use equality::{equality_element, equality_element_reference};
pub use packing::{compute_s_hat, pack, packed_vars};
use tensor::TensorAlgebra;
use weights::{batch_rows, batched_weights};

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

/// `e = eq̃(φ0(r_high), φ1(r'))`, formed by whichever route the characteristic of `F` admits.
///
/// The Remark 3.4 recurrence needs `eq̃(X, Y) = Π_i (1 − X_i − Y_i)`, which holds only when
/// `2 = 0`, and costs `O(ℓ' · d)`. Elsewhere the element is summed over the hypercube from its
/// definition, at `O(2^ℓ' · d²)` — each of the `2^ℓ'` terms is a `d × d` exterior product.
/// That asymmetry is real: the second branch shows the reduction is not specific to
/// characteristic 2, not that it is ready to be run at scale over an odd-characteristic field.
fn equality_element_for_characteristic<F, EF>(
    r_high: &Point<EF>,
    r_prime: &Point<EF>,
) -> TensorAlgebra<F, EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    if F::PrimeSubfield::ONE + F::PrimeSubfield::ONE == F::PrimeSubfield::ZERO {
        equality_element(r_high, r_prime)
    } else {
        equality_element_reference(r_high, r_prime)
    }
}

/// The `κ` batching challenges `r'' ∈ EF^κ`, drawn identically by both sides.
fn sample_batching_point<F, EF, Challenger>(challenger: &mut Challenger) -> Point<EF>
where
    F: Field,
    EF: ExtensionField<F>,
    Challenger: FieldChallenger<F>,
{
    Point::new(
        (0..packed_vars::<F, EF>())
            .map(|_| challenger.sample_algebra_element())
            .collect(),
    )
}

/// Proves the reduction of `t(r) = s` to a claim about `packed`, the packing of `t`.
///
/// Returns the proof, the sumcheck's random point `r'`, and the surviving claim's value
/// `s' = t'(r')`.
///
/// Every value that reaches the proof is computed from `packed`; `t` is read only by the debug
/// assertion that `packed` is its packing.
///
/// The sumcheck runs on a scalar [`ProductPolynomial`]. For the binary tower
/// `EF::ExtensionPacking` is `EF` itself, so packing the operands would buy nothing there;
/// over an extension with a wider packing it leaves throughput unclaimed.
///
/// `(r', s')` is transcript-bound on return: `final_eval` is observed into `challenger`
/// alongside `s_hat` and the sumcheck's own rounds before this function returns. `r` itself is
/// not bound by this reduction — a caller composing it into a larger protocol must bind `r`
/// before invoking this function.
///
/// # Panics
/// Panics unless `r` names at least the `κ` packed variables and `packed` has exactly the
/// `ℓ − κ` variables that leaves. Debug builds also check that `packed` is the packing of `t`.
pub fn prove_ring_switch<F, EF, Challenger>(
    t: &Poly<F>,
    packed: &Poly<EF>,
    r: &Point<EF>,
    challenger: &mut Challenger,
) -> (RingSwitchProof<F, EF>, Point<EF>, EF)
where
    F: Field,
    EF: ExtensionField<F>,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    let kappa = packed_vars::<F, EF>();
    assert!(
        r.num_variables() >= kappa,
        "the evaluation point must name at least the {kappa} packed variables, got {}",
        r.num_variables()
    );
    let ell_prime = r.num_variables() - kappa;
    assert_eq!(
        packed.num_variables(),
        ell_prime,
        "the packed polynomial must have the {ell_prime} variables the evaluation point leaves \
         once the packed ones are removed"
    );
    debug_assert_eq!(
        pack::<F, EF>(t),
        *packed,
        "the packed polynomial must be the packing of t"
    );

    let (r_high, _) = r.split_at(ell_prime);

    // Send `ŝ`, binding the transcript to the base coefficients that cross the wire rather
    // than to either of the readings derived from them.
    let s_hat = compute_s_hat::<F, EF>(packed, &r_high);
    challenger.observe_slice(s_hat.coefficients());

    let r_batch = sample_batching_point::<F, EF, _>(challenger);

    // `ℓ'` rounds on `h(X) = A(X) · t'(X)`, whose sum over the hypercube is the batched row
    // reading of `ŝ`.
    let weights = batched_weights::<F, EF>(&r_high, &r_batch);
    let poly = ProductPolynomial::new_unpacked(VariableOrder::Prefix, packed.clone(), weights);
    let mut prover = SumcheckProver::new(poly, batch_rows::<F, EF>(&s_hat, &r_batch));
    let mut sumcheck = SumcheckData::default();
    let r_prime =
        prover.compute_sumcheck_polynomials(&mut sumcheck, challenger, ell_prime, 0, None);

    let final_eval = packed.eval_ext::<F>(&r_prime);
    challenger.observe_algebra_element(final_eval);
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

/// Verifies the reduction of `claimed_sum = t(r)` and returns the surviving claim
/// `t'(r') = s'` as the pair `(r', s')`.
///
/// The round count is taken from `r`, which the verifier owns, so a proof carrying a different
/// number of rounds is rejected by [`SumcheckData::verify_rounds`] instead of desynchronising
/// the transcript.
///
/// The characteristic of `F` selects how the tensor-algebra equality element is formed, so the
/// reduction runs over any extension; in odd characteristic that step costs `O(2^ℓ' · d²)`
/// rather than `O(ℓ' · d)`.
///
/// `(r', s')` is transcript-bound on return: this function observes `proof.final_eval` into
/// `challenger` at the same point [`prove_ring_switch`] does, before returning it as part of
/// the surviving claim. `r` is not bound by this reduction; `claimed_sum` is, indirectly,
/// through the column check against the observed `ŝ`. A caller composing this into a larger
/// protocol must bind `r` itself before invoking it.
///
/// # Panics
/// Panics unless `r` names at least the `κ` packed variables. Everything read out of `proof`
/// is validated and reported as an error.
///
/// # Errors
/// Returns [`RingSwitchError`] for a malformed `ŝ`, a non-empty `pow_witnesses` (this reduction
/// never grinds), a claim that disagrees with `ŝ`'s columns, a failed sumcheck round, or a
/// final claim that does not close the sumcheck.
pub fn verify_ring_switch<F, EF, Challenger>(
    proof: &RingSwitchProof<F, EF>,
    r: &Point<EF>,
    claimed_sum: EF,
    challenger: &mut Challenger,
) -> Result<(Point<EF>, EF), RingSwitchError>
where
    F: Field,
    EF: ExtensionField<F>,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    let kappa = packed_vars::<F, EF>();
    assert!(
        r.num_variables() >= kappa,
        "the evaluation point must name at least the {kappa} packed variables, got {}",
        r.num_variables()
    );
    let ell_prime = r.num_variables() - kappa;

    // `columns` and `rows` index a `DIMENSION × DIMENSION` matrix, so a short `ŝ` must be
    // rejected before either reading is taken.
    if !proof.s_hat.is_well_formed() {
        let dimension = TensorAlgebra::<F, EF>::DIMENSION;
        return Err(RingSwitchError::MalformedTensor {
            expected: dimension * dimension,
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
    challenger.observe_slice(proof.s_hat.coefficients());

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
        return Err(RingSwitchError::ClaimMismatch);
    }

    // The sumcheck's initial sum is derived from `ŝ`'s rows, never sent, which is what makes a
    // dishonest `ŝ` catchable: the two readings are of the same coefficients.
    let r_batch = sample_batching_point::<F, EF, _>(challenger);
    let mut sum = batch_rows::<F, EF>(&proof.s_hat, &r_batch);

    let r_prime =
        proof
            .sumcheck
            .verify_rounds(challenger, &mut sum, ell_prime, 0, Basis::Evaluation)?;
    challenger.observe_algebra_element(proof.final_eval);

    // The batched rows of the equality element are `A(r')`, so the sumcheck closes on
    // `A(r') · t'(r')`.
    let e = equality_element_for_characteristic::<F, EF>(&r_high, &r_prime);
    if sum != batch_rows::<F, EF>(&e, &r_batch) * proof.final_eval {
        return Err(RingSwitchError::FinalCheck);
    }

    Ok((r_prime, proof.final_eval))
}

#[cfg(test)]
mod tests {
    use p3_challenger::CanObserve;
    use p3_field::PrimeCharacteristicRing;
    use p3_multilinear_util::point::Point;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use super::test_util::{EF, F, base_poly, challenger};
    use super::*;

    /// Prover and verifier agree, and the surviving claim is true.
    #[test]
    fn the_reduction_round_trips() {
        let ell = 7;
        let t = base_poly(ell, 13);
        let packed = pack::<F, EF>(&t);
        let r = Point::<EF>::rand(&mut SmallRng::seed_from_u64(14), ell);
        let s = t.eval_base(&r);

        let mut p_chal = challenger();
        let (proof, r_prime_p, s_prime_p) =
            prove_ring_switch::<F, EF, _>(&t, &packed, &r, &mut p_chal);

        let mut v_chal = challenger();
        let (r_prime_v, s_prime_v) =
            verify_ring_switch::<F, EF, _>(&proof, &r, s, &mut v_chal).unwrap();

        assert_eq!(r_prime_p, r_prime_v);
        assert_eq!(s_prime_p, s_prime_v);
        // The surviving claim is the truth about the committed polynomial.
        assert_eq!(s_prime_v, packed.eval_ext::<F>(&r_prime_v));
    }

    /// A wrong claimed value is rejected at the column check.
    #[test]
    fn a_wrong_claim_is_rejected() {
        let ell = 6;
        let t = base_poly(ell, 15);
        let packed = pack::<F, EF>(&t);
        let r = Point::<EF>::rand(&mut SmallRng::seed_from_u64(16), ell);
        let s = t.eval_base(&r);

        let mut p_chal = challenger();
        let (proof, _, _) = prove_ring_switch::<F, EF, _>(&t, &packed, &r, &mut p_chal);

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
        let packed = pack::<F, EF>(&t);
        let r = Point::<EF>::rand(&mut SmallRng::seed_from_u64(18), ell);

        let mut p_chal = challenger();
        let (mut proof, _, _) = prove_ring_switch::<F, EF, _>(&t, &packed, &r, &mut p_chal);
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
        let packed = pack::<F, EF>(&t);
        let r = Point::<EF>::rand(&mut SmallRng::seed_from_u64(34), ell);
        let s = t.eval_base(&r);

        let mut p_chal = challenger();
        let (mut proof, _, _) = prove_ring_switch::<F, EF, _>(&t, &packed, &r, &mut p_chal);
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
        let packed = pack::<F, EF>(&t);
        let r = Point::<EF>::rand(&mut SmallRng::seed_from_u64(36), ell);
        let s = t.eval_base(&r);

        let mut p_chal = challenger();
        let (mut proof, _, _) = prove_ring_switch::<F, EF, _>(&t, &packed, &r, &mut p_chal);
        proof.sumcheck.polynomial_evaluations[0][0] += EF::ONE;

        let mut v_chal = challenger();
        assert_eq!(
            verify_ring_switch::<F, EF, _>(&proof, &r, s, &mut v_chal),
            Err(RingSwitchError::FinalCheck)
        );
    }

    /// A proof is bound to the evaluation point it was produced for: verifying it against a
    /// different point of the same length is rejected, even though `r` itself is never observed
    /// into the transcript — `r_high` and `r_low` both feed checks the verifier forms locally
    /// from `r`, not from anything the proof carries.
    #[test]
    fn a_proof_does_not_verify_against_a_different_point() {
        let ell = 6;
        let t = base_poly(ell, 37);
        let packed = pack::<F, EF>(&t);
        let r = Point::<EF>::rand(&mut SmallRng::seed_from_u64(38), ell);

        let mut p_chal = challenger();
        let (proof, _, _) = prove_ring_switch::<F, EF, _>(&t, &packed, &r, &mut p_chal);

        let other_r = Point::<EF>::rand(&mut SmallRng::seed_from_u64(39), ell);
        let other_s = t.eval_base(&other_r);

        let mut v_chal = challenger();
        assert_eq!(
            verify_ring_switch::<F, EF, _>(&proof, &other_r, other_s, &mut v_chal),
            Err(RingSwitchError::ClaimMismatch)
        );
    }

    /// This reduction never grinds, so a proof carrying PoW witnesses must be rejected rather
    /// than accepted with junk data nothing checks.
    #[test]
    fn a_proof_with_pow_witnesses_is_rejected() {
        let ell = 6;
        let t = base_poly(ell, 40);
        let packed = pack::<F, EF>(&t);
        let r = Point::<EF>::rand(&mut SmallRng::seed_from_u64(41), ell);
        let s = t.eval_base(&r);

        let mut p_chal = challenger();
        let (mut proof, _, _) = prove_ring_switch::<F, EF, _>(&t, &packed, &r, &mut p_chal);
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
        let packed = pack::<F, EF>(&t);
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

        let mut chal = challenger();
        chal.observe_slice(s_hat.coefficients());
        let r_batch = sample_batching_point::<F, EF, _>(&mut chal);

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
        let r_prime =
            prover.compute_sumcheck_polynomials(&mut sumcheck, &mut chal, ell_prime, 0, None);
        let sum_final = prover.claimed_sum();

        // `final_eval` set to whatever the final identity needs, computed the same way the
        // verifier computes it.
        let e = equality_element_for_characteristic::<F, EF>(&r_high, &r_prime);
        let a_r_prime = batch_rows::<F, EF>(&e, &r_batch);
        let final_eval = sum_final * a_r_prime.inverse();

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
        let packed = pack::<F, EF>(&t);
        let r = Point::<EF>::rand(&mut SmallRng::seed_from_u64(20), 3);
        let mut chal = challenger();
        let _ = prove_ring_switch::<F, EF, _>(&t, &packed, &r, &mut chal);
    }

    /// A packed polynomial of the wrong arity must panic rather than be zipped against a
    /// weight table of a different length.
    #[test]
    #[should_panic(expected = "the packed polynomial must have the 2 variables")]
    fn prove_rejects_a_packed_polynomial_of_the_wrong_arity() {
        let t = base_poly(7, 21);
        let packed = pack::<F, EF>(&t);
        let r = Point::<EF>::rand(&mut SmallRng::seed_from_u64(22), 6);
        let mut chal = challenger();
        let _ = prove_ring_switch::<F, EF, _>(&t, &packed, &r, &mut chal);
    }

    /// A packed polynomial that is not the packing of `t` would prove a claim about a
    /// different multilinear. Debug builds catch it.
    #[test]
    #[should_panic(expected = "the packed polynomial must be the packing of t")]
    fn prove_rejects_a_packed_polynomial_that_is_not_the_packing() {
        let t = base_poly(6, 23);
        let packed = pack::<F, EF>(&base_poly(6, 24));
        let r = Point::<EF>::rand(&mut SmallRng::seed_from_u64(25), 6);
        let mut chal = challenger();
        let _ = prove_ring_switch::<F, EF, _>(&t, &packed, &r, &mut chal);
    }

    /// The verifier's own point is subject to the same requirement.
    #[test]
    #[should_panic(expected = "the evaluation point must name at least the 4 packed variables")]
    fn verify_rejects_a_point_shorter_than_the_packed_variables() {
        let ell = 6;
        let t = base_poly(ell, 26);
        let packed = pack::<F, EF>(&t);
        let r = Point::<EF>::rand(&mut SmallRng::seed_from_u64(27), ell);

        let mut p_chal = challenger();
        let (proof, _, _) = prove_ring_switch::<F, EF, _>(&t, &packed, &r, &mut p_chal);

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
        let packed = pack::<F, EF>(&t);
        let r = Point::<EF>::rand(&mut SmallRng::seed_from_u64(30), ell);
        let s = t.eval_base(&r);

        let mut p_chal = challenger();
        let (mut proof, _, _) = prove_ring_switch::<F, EF, _>(&t, &packed, &r, &mut p_chal);
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
        let packed = pack::<F, EF>(&t);
        let r = Point::<EF>::rand(&mut SmallRng::seed_from_u64(32), ell);
        let s = t.eval_base(&r);

        let mut p_chal = challenger();
        let (mut proof, _, _) = prove_ring_switch::<F, EF, _>(&t, &packed, &r, &mut p_chal);
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
