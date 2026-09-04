//! Transcript record:
//! - produced by the generic-degree sumcheck prover,
//! - consumed by the corresponding verifier.

use alloc::vec::Vec;

use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, PrimeField64};
use p3_multilinear_util::point::Point;
use serde::{Deserialize, Serialize};

use super::error::GenericDegreeError;
use super::transcript::VerifierTranscript;
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
    ///
    /// # Shape checks come first
    ///
    /// Every length this proof carries is checked before any of it is absorbed.
    ///
    /// The transcript panics on a step of an undescribed length.
    /// That is right for two sides built from different descriptions.
    /// It is wrong for untrusted input, which must be rejected instead.
    ///
    /// So the checks run first.
    /// They compare against the same numbers that shape the description.
    ///
    /// # Errors
    ///
    /// - The per-variable degree is zero, which carries no information.
    /// - The round count differs from the expected one.
    /// - The witness count differs from the expected one.
    /// - A round polynomial carries the wrong number of evaluations.
    /// - A grinding witness misses the required difficulty.
    pub fn verify<Challenger>(
        &self,
        challenger: &mut Challenger,
        num_rounds: usize,
        degree: usize,
        pow_bits: usize,
    ) -> Result<(Point<EF>, EF), GenericDegreeError>
    where
        F: PrimeField64,
        EF: ExtensionField<F>,
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        // Phase 1: check the proof's shape.
        //
        // Nothing is absorbed yet, so every rejection here is a clean error.

        // A degree-zero round polynomial carries no information.
        // Interpolation would later index an empty evaluation slice.
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
        // - zero difficulty requires an empty witness vector,
        // - positive difficulty requires exactly one witness per round.
        let expected_pow_witnesses = if pow_bits > 0 { num_rounds } else { 0 };
        if self.pow_witnesses.len() != expected_pow_witnesses {
            return Err(GenericDegreeError::PowWitnessCountMismatch {
                expected: expected_pow_witnesses,
                actual: self.pow_witnesses.len(),
            });
        }

        // Each round polynomial is one transcript step of a fixed width.
        //
        //     described step:  Fixed(degree) evaluations
        //     proof carries:   evals.len()
        //     mismatch      -> reject here, before absorbing anything
        for (round, evals) in self.round_polys.iter().enumerate() {
            if evals.len() != degree {
                return Err(GenericDegreeError::PolyEvalCountMismatch {
                    round,
                    expected: degree,
                    actual: evals.len(),
                });
            }
        }

        // Phase 2: replay the transcript, now that every length is known good.

        // Seeded from the same numbers the prover seeded with.
        let mut transcript = VerifierTranscript::<Challenger, F, EF>::new(
            challenger,
            num_rounds,
            degree,
            pow_bits,
            self.claimed_sum,
        );

        // Barycentric weights for the integer domain 0, 1, …, degree are shared by every round.
        let interpolator = RoundPolyInterpolator::new(degree);

        let mut running_sum = self.claimed_sum;
        let mut challenges = Vec::with_capacity(num_rounds);

        for (round, evals) in self.round_polys.iter().enumerate() {
            // One call binds the polynomial, re-checks the grind, and draws the challenge.
            let witness = (pow_bits > 0).then(|| self.pow_witnesses[round]);
            let challenge = transcript.round(evals, witness)?;

            // Reduce the running claim through the challenge just drawn.
            running_sum = interpolator.eval(evals, running_sum, challenge);
            challenges.push(challenge);
        }

        // Require that the whole described sequence was played.
        transcript.finish();

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
    use crate::generic_degree::pattern;

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
    fn verify_rejects_wrong_evaluation_count_without_panicking() {
        // Invariant: a round polynomial of the wrong width is untrusted input.
        //
        // The transcript panics on a step of an undescribed width.
        // So the width is checked before anything is absorbed.
        //
        // Fixture state: one round, expected width 3.
        //
        // Mutation: send 2 evaluations instead.
        //
        //     described step:  Fixed(3)
        //     proof carries:   2
        //     -> structured error, no panic
        let mut ch = fresh_challenger();
        let proof = GenericDegreeProof::<F, EF> {
            claimed_sum: EF::ZERO,
            round_polys: vec![vec![EF::ZERO; 2]],
            pow_witnesses: vec![],
        };
        let err = proof.verify(&mut ch, 1, 3, 0).unwrap_err();
        assert_eq!(
            err,
            GenericDegreeError::PolyEvalCountMismatch {
                round: 0,
                expected: 3,
                actual: 2,
            }
        );
    }

    #[test]
    fn shape_numbers_are_bound_into_the_seed() {
        // Invariant: every number that shapes a run moves the shape fingerprint.
        //
        // The fingerprint enters the sponge when the transcript is seeded.
        // It is therefore what keeps two configurations off the same challenges.
        //
        // Fixture state: 4 rounds, degree 3, no grinding.
        let base = pattern::<F, EF>(4, 3, 0).pattern_hash();

        // One more round appends three more steps.
        assert_ne!(base, pattern::<F, EF>(5, 3, 0).pattern_hash());

        // A wider round polynomial changes each round step's declared width.
        assert_ne!(base, pattern::<F, EF>(4, 4, 0).pattern_hash());

        // Enabling grinding inserts a step per round.
        assert_ne!(base, pattern::<F, EF>(4, 3, 8).pattern_hash());

        // Two positive difficulties differ only inside the grinding steps.
        assert_ne!(
            pattern::<F, EF>(4, 3, 8).pattern_hash(),
            pattern::<F, EF>(4, 3, 9).pattern_hash(),
        );
    }

    #[test]
    fn claimed_sum_is_bound_before_the_first_challenge() {
        // Invariant: the claimed sum reaches the sponge before any challenge.
        //
        // Without that, a prover could pick the sum after seeing the challenge.
        //
        // Fixture state: one round, one evaluation, no grinding.
        //
        // Mutation: change only the claimed sum.
        //
        //     run A: claimed sum 1
        //     run B: claimed sum 0
        //     -> different challenge, so different reduction point
        let proof = GenericDegreeProof::<F, EF> {
            claimed_sum: EF::ONE,
            round_polys: vec![vec![EF::ONE]],
            pow_witnesses: vec![],
        };
        let tampered = GenericDegreeProof::<F, EF> {
            claimed_sum: EF::ZERO,
            ..proof.clone()
        };

        let mut ch_a = fresh_challenger();
        let (point_a, _) = proof.verify(&mut ch_a, 1, 1, 0).unwrap();

        let mut ch_b = fresh_challenger();
        let (point_b, _) = tampered.verify(&mut ch_b, 1, 1, 0).unwrap();

        assert_ne!(point_a, point_b);
    }

    #[test]
    fn round_polynomial_is_bound_before_its_challenge() {
        // Invariant: a round polynomial is bound before its own challenge.
        //
        // Without that, a prover could pick the polynomial to suit the challenge.
        //
        // Fixture state: two rounds, width 1, no grinding.
        //
        // Mutation: change the first round's only evaluation.
        //
        //     honest:   [[1], [1]]
        //     tampered: [[2], [1]]
        //     -> round 0's challenge changes, and every later one with it
        let proof = GenericDegreeProof::<F, EF> {
            claimed_sum: EF::ONE,
            round_polys: vec![vec![EF::ONE], vec![EF::ONE]],
            pow_witnesses: vec![],
        };
        let mut tampered = proof.clone();
        tampered.round_polys[0][0] = EF::TWO;

        let mut ch_a = fresh_challenger();
        let (point_a, _) = proof.verify(&mut ch_a, 2, 1, 0).unwrap();

        let mut ch_b = fresh_challenger();
        let (point_b, _) = tampered.verify(&mut ch_b, 2, 1, 0).unwrap();

        assert_ne!(point_a, point_b);
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
