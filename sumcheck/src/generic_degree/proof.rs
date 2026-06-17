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
///
/// # Layout
///
/// - `claimed_sum` is the value the prover asserts for the hypercube sum.
/// - One round entry per bound variable; each holds `degree` field elements.
/// - The transmitted evaluations are `h(0), h(2), h(3), ..., h(degree)`.
/// - The value `h(1)` is recovered by the verifier as `sum - h(0)`.
/// - PoW witnesses are present only when grinding is configured.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct GenericDegreeProof<F, EF> {
    /// Claimed value of the sum over the boolean hypercube at round zero.
    ///
    /// Carried in the proof so the verifier consumes the whole prover output
    /// rather than receiving the claim through a side channel.
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
    /// When the claimed sum is fixed by an outer protocol, the caller must also
    /// check that the proof's claimed sum matches that expected value.
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

        // Bind the transcript to the claimed sum before sampling any challenge,
        // so the round challenges depend on the statement being proven.
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
        // Fixture: an empty proof, but the verifier expects 2 rounds.
        //
        // Mutation: pass num_rounds = 2 against a 0-round proof.
        //
        //     proof rounds: 0
        //     expected:     2
        //     -> RoundCountMismatch
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
        // Invariant: a degree-zero round polynomial carries no information.
        // The verifier must reject up front with a typed error.
        //
        // Fixture state: empty proof, but degree set to zero.
        //
        // Mutation: pass degree = 0.
        //
        //     proof rounds: 0
        //     num_rounds:   0
        //     degree:       0       ← rejected
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
        // Invariant: with `pow_bits == 0` the canonical proof shape requires `pow_witnesses` to be empty.
        // Without this rule two distinct proofs would verify for the same statement (malleability).
        //
        // Fixture state: a one-round proof with an extra PoW witness attached.
        //
        //     round_polys.len() = 1      → matches num_rounds = 1
        //     pow_bits = 0
        //     pow_witnesses.len() = 1    ← spurious, must be rejected
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
