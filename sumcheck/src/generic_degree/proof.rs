//! Transcript record:
//! - produced by the generic-degree sumcheck prover,
//! - consumed by the corresponding verifier.

use alloc::vec::Vec;

use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, Field};
use p3_multilinear_util::point::Point;
use serde::{Deserialize, Serialize};

use super::error::GenericDegreeError;
use super::util::RoundPolyInterpolator;

/// Transcript record produced by the generic-degree sumcheck prover.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct GenericDegreeProof<F, EF> {
    /// Claimed value of the sum over the boolean hypercube at round zero.
    ///
    /// Carried in the proof so the verifier consumes the whole prover output.
    pub claimed_sum: EF,
    /// Transmitted round-polynomial evaluations.
    ///
    /// Length is the number of rounds; each inner vector has length `degree`.
    pub round_polys: Vec<Vec<EF>>,
    /// One PoW witness per round when grinding is enabled, otherwise empty.
    pub pow_witnesses: Vec<F>,
}

impl<F, EF> GenericDegreeProof<F, EF> {
    /// Number of rounds in this proof.
    #[inline]
    #[must_use]
    pub const fn num_rounds(&self) -> usize {
        self.round_polys.len()
    }

    /// Run the verifier side of a generic-degree sumcheck.
    ///
    /// # Arguments
    ///
    /// - `challenger`: Fiat-Shamir transcript in the same state as the prover.
    /// - `num_rounds`: number of variables expected to be bound.
    /// - `degree`: per-variable degree of the polynomial being summed.
    /// - `pow_bits`: grinding difficulty per round, or `0`.
    ///
    /// The claimed sum is read from the proof itself.
    ///
    /// # Returns
    ///
    /// - The vector of sampled challenges.
    /// - The final claimed sum after all variables are bound.
    ///
    /// # Closing the protocol
    ///
    /// Sumcheck only reduces the cube sum to one polynomial evaluation; the protocol is not yet complete.
    /// The caller must check that the returned final sum equals the polynomial at the returned challenge through:
    ///
    /// - PCS openings for committed multilinears.
    /// - Closed-form evaluation for structural multilinears (`eq`, `next`, selectors).
    ///
    /// When an outer protocol fixes the claimed sum, the caller must also check the proof's claimed sum against it.
    pub fn verify<Challenger>(
        &self,
        challenger: &mut Challenger,
        num_rounds: usize,
        degree: usize,
        pow_bits: usize,
    ) -> Result<(Point<EF>, EF), GenericDegreeError>
    where
        F: Field,
        EF: ExtensionField<F>,
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        // A degree-zero round polynomial carries no information;
        //
        // The Lagrange helper would later index out of bounds on the empty evaluation slice.
        //
        // Catch it here with a typed error rather than a panic.
        if degree == 0 {
            return Err(GenericDegreeError::InvalidDegree { degree });
        }

        // Reject up front if the proof has the wrong round count.
        if self.round_polys.len() != num_rounds {
            return Err(GenericDegreeError::RoundCountMismatch {
                expected: num_rounds,
                actual: self.round_polys.len(),
            });
        }

        // Canonical proof shape — every accepting proof has a unique form:
        // - `pow_bits == 0` requires an empty `pow_witnesses` vector,
        // - `pow_bits > 0`  requires exactly `num_rounds` witnesses.
        let expected_pow_witnesses = if pow_bits > 0 { num_rounds } else { 0 };
        if self.pow_witnesses.len() != expected_pow_witnesses {
            return Err(GenericDegreeError::PowWitnessCountMismatch {
                expected: expected_pow_witnesses,
                actual: self.pow_witnesses.len(),
            });
        }

        // Bind the transcript to the claimed sum so the challenges depend on the statement.
        challenger.observe_algebra_element(self.claimed_sum);

        // Barycentric weights for the integer domain 0, 1, …, degree are shared by every round.
        let interpolator = RoundPolyInterpolator::new(degree);

        let mut running_sum = self.claimed_sum;
        let mut challenges = Vec::with_capacity(num_rounds);

        for (round, evals) in self.round_polys.iter().enumerate() {
            // Each round polynomial must carry exactly `degree` evaluations.
            if evals.len() != degree {
                return Err(GenericDegreeError::PolyEvalCountMismatch {
                    round,
                    expected: degree,
                    actual: evals.len(),
                });
            }

            // Replay the prover's transcript writes.
            challenger.observe_algebra_slice(evals);

            if pow_bits > 0 && !challenger.check_witness(pow_bits, self.pow_witnesses[round]) {
                return Err(GenericDegreeError::InvalidPowWitness { round });
            }

            // Sample the same challenge the prover saw, then reduce the claim through it.
            let challenge: EF = challenger.sample_algebra_element();
            running_sum = interpolator.eval(evals, running_sum, challenge);
            challenges.push(challenge);
        }

        Ok((Point::new(challenges), running_sum))
    }
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

    fn fresh_challenger() -> Ch {
        // Fixed seed so prover and verifier transcripts match exactly.
        let mut rng = SmallRng::seed_from_u64(0xDEADBEEF);
        let perm = Perm::new_from_rng_128(&mut rng);
        Ch::new(perm)
    }

    #[test]
    fn verify_rejects_wrong_round_count() {
        // A 0-round proof must be rejected when two rounds are expected.
        let mut ch = fresh_challenger();
        let proof: GenericDegreeProof<F, EF> = GenericDegreeProof::default();
        let err = proof.verify(&mut ch, 2, 3, 0).unwrap_err();
        assert!(matches!(
            err,
            GenericDegreeError::RoundCountMismatch {
                expected: 2,
                actual: 0
            }
        ));
    }

    #[test]
    fn verify_rejects_zero_degree() {
        // Degree zero carries no information, so the verifier rejects it with a typed error.
        let mut ch = fresh_challenger();
        let proof: GenericDegreeProof<F, EF> = GenericDegreeProof::default();
        let err = proof.verify(&mut ch, 0, 0, 0).unwrap_err();
        assert!(matches!(
            err,
            GenericDegreeError::InvalidDegree { degree: 0 }
        ));
    }

    #[test]
    fn verify_rejects_unexpected_pow_witnesses() {
        // With pow_bits == 0 a canonical proof carries no PoW witnesses.
        // Accepting a spurious one would let two proofs verify the same statement (malleability).
        let mut ch = fresh_challenger();
        let proof = GenericDegreeProof::<F, EF> {
            claimed_sum: EF::ZERO,
            round_polys: vec![vec![EF::ZERO; 1]],
            pow_witnesses: vec![F::ZERO],
        };
        let err = proof.verify(&mut ch, 1, 1, 0).unwrap_err();
        assert!(matches!(
            err,
            GenericDegreeError::PowWitnessCountMismatch {
                expected: 0,
                actual: 1
            }
        ));
    }
}
