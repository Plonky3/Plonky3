//! Coefficient-form sumcheck proof + transcript helpers.
//!
//! WARP runs two sumchecks: the §6.3 twin-constraint sumcheck of degree
//! `1 + max{log n + 1, log M + d}`, and the §8.2 multilinear-batching
//! sumcheck of degree 2. Both are encoded the same way:
//!
//! - Each round the prover sends the round polynomial `h_j(X)` as
//!   coefficients `[c_0, c_1, …, c_D]` (so `h_j(X) = Σ c_k · X^k`).
//! - The verifier checks the sum constraint
//!   `h_j(0) + h_j(1) == claimed_sum` and updates `claimed_sum := h_j(r)`
//!   with the freshly-sampled challenge `r`.
//!
//! Coefficient form is slightly more verbose on the wire than WHIR's
//! `[h(0), h(inf)]` evaluation form (one extra element per round), but it
//! generalises cleanly to arbitrary degree with a Horner-only verifier.

use alloc::vec::Vec;

use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, Field};
use p3_multilinear_util::point::Point;
use serde::{Deserialize, Serialize};

use crate::error::VerifierError;

/// A sumcheck proof in coefficient form: one `Vec<EF>` per round, of length
/// `degree + 1`.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(bound = "EF: Serialize + serde::de::DeserializeOwned")]
pub struct SumcheckProof<EF> {
    /// Coefficients of each round polynomial in monomial form.
    pub round_polys: Vec<Vec<EF>>,
}

impl<EF> SumcheckProof<EF> {
    /// Construct an empty proof.
    pub const fn new() -> Self {
        Self {
            round_polys: Vec::new(),
        }
    }

    /// Number of completed rounds.
    pub const fn num_rounds(&self) -> usize {
        self.round_polys.len()
    }
}

/// Prover-side helper: append a round polynomial to the proof, observe the
/// coefficients into the challenger, and sample the next round challenge.
///
/// The polynomial is given in coefficient form `[c_0, …, c_D]`.
///
/// # Returns
///
/// The freshly sampled challenge `r`.
pub fn observe_and_sample<F, EF, Ch>(
    proof: &mut SumcheckProof<EF>,
    challenger: &mut Ch,
    coeffs: Vec<EF>,
) -> EF
where
    F: Field,
    EF: ExtensionField<F>,
    Ch: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    challenger.observe_algebra_slice(&coeffs);
    proof.round_polys.push(coeffs);
    challenger.sample_algebra_element()
}

/// Verifier-side helper: replay one sumcheck round.
///
/// Reads the round polynomial from `proof.round_polys[round]`, observes it
/// into the challenger, checks `h(0) + h(1) == *claimed_sum`, samples the
/// challenge, and updates `*claimed_sum := h(challenge)`.
///
/// # Arguments
///
/// - `phase`: a string label used in error messages ("twin-constraint" or
///   "multilinear-batching").
/// - `expected_degree`: the polynomial's expected degree; the round poly
///   must have exactly `expected_degree + 1` coefficients.
///
/// # Returns
///
/// The challenge `r` sampled this round.
pub fn verify_round<F, EF, Ch>(
    proof: &SumcheckProof<EF>,
    challenger: &mut Ch,
    claimed_sum: &mut EF,
    round: usize,
    expected_degree: usize,
    phase: &'static str,
) -> Result<EF, VerifierError>
where
    F: Field,
    EF: ExtensionField<F>,
    Ch: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    let coeffs = &proof.round_polys[round];
    if coeffs.len() != expected_degree + 1 {
        return Err(VerifierError::SumcheckDegree {
            phase,
            round,
            got: coeffs.len(),
            expected: expected_degree + 1,
        });
    }
    challenger.observe_algebra_slice(coeffs);

    // h(0) + h(1) = c_0 + (c_0 + c_1 + … + c_D) = 2·c_0 + c_1 + … + c_D
    let h_0 = coeffs[0];
    let h_1: EF = coeffs.iter().copied().sum();
    if h_0 + h_1 != *claimed_sum {
        return Err(VerifierError::SumcheckConsistency { phase, round });
    }

    let r: EF = challenger.sample_algebra_element();
    // Evaluate h(r) via Horner: c_D + r·(c_{D-1} + r·(... + r·c_0)).
    let h_r = coeffs.iter().rev().fold(EF::ZERO, |acc, &c| acc * r + c);
    *claimed_sum = h_r;
    Ok(r)
}

/// Run all rounds of a sumcheck, returning the challenges and the final
/// folded claim.
pub fn verify_sumcheck<F, EF, Ch>(
    proof: &SumcheckProof<EF>,
    challenger: &mut Ch,
    initial_claim: EF,
    expected_rounds: usize,
    expected_degree: usize,
    phase: &'static str,
) -> Result<(Point<EF>, EF), VerifierError>
where
    F: Field,
    EF: ExtensionField<F>,
    Ch: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    if proof.round_polys.len() != expected_rounds {
        return Err(VerifierError::SumcheckDegree {
            phase,
            round: proof.round_polys.len(),
            got: proof.round_polys.len(),
            expected: expected_rounds,
        });
    }
    let mut claim = initial_claim;
    let mut challenges = Vec::with_capacity(expected_rounds);
    for round in 0..expected_rounds {
        let r = verify_round::<F, EF, Ch>(
            proof,
            challenger,
            &mut claim,
            round,
            expected_degree,
            phase,
        )?;
        challenges.push(r);
    }
    Ok((Point::new(challenges), claim))
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_challenger::DuplexChallenger;
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use super::*;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;
    type Perm = Poseidon2BabyBear<16>;
    type Ch = DuplexChallenger<F, Perm, 16, 8>;

    fn challenger() -> Ch {
        let mut rng = SmallRng::seed_from_u64(1);
        let perm = Perm::new_from_rng_128(&mut rng);
        Ch::new(perm)
    }

    /// Honest one-round sumcheck: prover sends h(X) = 1 + 2X + 3X² with
    /// claimed sum h(0) + h(1) = 1 + 6 = 7, verifier accepts and samples r,
    /// then checks the new claim equals h(r).
    #[test]
    fn honest_round_accepts_and_updates_claim() {
        let coeffs = vec![EF::ONE, EF::from_u64(2), EF::from_u64(3)];

        let mut p_ch = challenger();
        let mut proof = SumcheckProof::<EF>::new();
        let r = observe_and_sample::<F, EF, _>(&mut proof, &mut p_ch, coeffs.clone());
        let h_r_prover = coeffs.iter().rev().fold(EF::ZERO, |acc, &c| acc * r + c);

        let mut v_ch = challenger();
        let mut claim = EF::from_u64(7);
        let r_v = verify_round::<F, EF, _>(&proof, &mut v_ch, &mut claim, 0, 2, "test")
            .expect("verifier accepts honest round");
        assert_eq!(r, r_v);
        assert_eq!(claim, h_r_prover);
    }

    /// Tamper with the round polynomial — verifier rejects with a
    /// consistency error.
    #[test]
    fn tampered_round_poly_rejects() {
        let coeffs = vec![EF::ONE, EF::from_u64(2), EF::from_u64(3)];
        let mut p_ch = challenger();
        let mut proof = SumcheckProof::<EF>::new();
        let _ = observe_and_sample::<F, EF, _>(&mut proof, &mut p_ch, coeffs);
        // Adversary modifies the claimed h(0): now h(0) + h(1) ≠ 7.
        proof.round_polys[0][0] = EF::from_u64(99);

        let mut v_ch = challenger();
        let mut claim = EF::from_u64(7);
        let err = verify_round::<F, EF, _>(&proof, &mut v_ch, &mut claim, 0, 2, "test")
            .expect_err("verifier rejects tampered round polynomial");
        assert!(matches!(err, VerifierError::SumcheckConsistency { .. }));
    }

    /// Wrong-degree round polynomial — verifier rejects with a degree error.
    #[test]
    fn wrong_degree_rejects() {
        let coeffs = vec![EF::ONE, EF::from_u64(2)]; // degree 1
        let mut p_ch = challenger();
        let mut proof = SumcheckProof::<EF>::new();
        let _ = observe_and_sample::<F, EF, _>(&mut proof, &mut p_ch, coeffs);

        let mut v_ch = challenger();
        let mut claim = EF::from_u64(3); // h(0) + h(1) = 1 + 3 = 4, fine
        // Verifier expects degree 2, prover sent degree 1 → reject before
        // even checking sums.
        let err = verify_round::<F, EF, _>(&proof, &mut v_ch, &mut claim, 0, 2, "test")
            .expect_err("verifier rejects wrong-degree round polynomial");
        assert!(matches!(err, VerifierError::SumcheckDegree { .. }));
    }
}
