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
//! it by columns to test the incoming claim, and by rows to derive the sumcheck's initial sum.
//! A prover who tampers with a coefficient to pass one reading breaks the other.

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
pub mod pack;
pub mod tensor;
pub mod weights;

use equality::{equality_element, equality_element_reference};
pub use pack::{compute_s_hat, pack, packed_vars};
use tensor::TensorAlgebra;
use weights::{batched_weights, initial_sum};

#[cfg(test)]
pub(crate) mod test_util;

/// The messages one ring-switching reduction puts on the wire.
#[derive(Clone, Debug, Serialize, Deserialize)]
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
}

/// `e = eq̃(φ0(r_high), φ1(r'))`, formed by whichever route the characteristic of `F` admits.
///
/// The Remark 3.4 recurrence needs `eq̃(X, Y) = Π_i (1 − X_i − Y_i)`, which holds only when
/// `2 = 0`, and costs `O(ℓ' · d)`. Elsewhere the element is summed over the hypercube from its
/// definition, at `O(2^ℓ')`. That asymmetry is real: the second branch shows the reduction is
/// not specific to characteristic 2, not that it is ready to be run at scale over an
/// odd-characteristic field.
fn equality_element_for_characteristic<F, EF>(
    r_high: &Point<EF>,
    r_challenge: &Point<EF>,
) -> TensorAlgebra<F, EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    if F::PrimeSubfield::ONE + F::PrimeSubfield::ONE == F::PrimeSubfield::ZERO {
        equality_element(r_high, r_challenge)
    } else {
        equality_element_reference(r_high, r_challenge)
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
    let mut prover = SumcheckProver::new(poly, initial_sum::<F, EF>(&s_hat, &r_batch));
    let mut sumcheck = SumcheckData::default();
    let r_prime =
        prover.compute_sumcheck_polynomials(&mut sumcheck, challenger, ell_prime, 0, None);

    let final_eval = packed.eval_ext::<F>(&r_prime);
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
/// reduction runs over any extension; in odd characteristic that step costs `O(2^ℓ')` rather
/// than `O(ℓ' · d)`.
///
/// # Panics
/// Panics unless `r` names at least the `κ` packed variables. Everything read out of `proof`
/// is validated and reported as an error.
///
/// # Errors
/// Returns [`RingSwitchError`] for a malformed `ŝ`, a claim that disagrees with `ŝ`'s columns,
/// a failed sumcheck round, or a final claim that does not close the sumcheck.
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
    let dimension = TensorAlgebra::<F, EF>::DIMENSION;
    let expected = dimension * dimension;
    let actual = proof.s_hat.coefficients().len();
    if actual != expected {
        return Err(RingSwitchError::MalformedTensor { expected, actual });
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
    let mut sum = initial_sum::<F, EF>(&proof.s_hat, &r_batch);

    let r_prime =
        proof
            .sumcheck
            .verify_rounds(challenger, &mut sum, ell_prime, 0, Basis::Evaluation)?;

    // The batched rows of the equality element are `A(r')`, so the sumcheck closes on
    // `A(r') · t'(r')`.
    let e = equality_element_for_characteristic::<F, EF>(&r_high, &r_prime);
    if sum != initial_sum::<F, EF>(&e, &r_batch) * proof.final_eval {
        return Err(RingSwitchError::FinalCheck);
    }

    Ok((r_prime, proof.final_eval))
}

#[cfg(test)]
mod tests {
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

    /// A tampered `ŝ` that passes the column check is still caught by the row reading. The
    /// claim handed to the verifier is the perturbed element's own column reading, formed the
    /// way the verifier forms it, so the first check passes by construction and only the row
    /// arm can reject. Both readings are of the same coefficients, which is what leaves a
    /// cheating prover nowhere to stand.
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
