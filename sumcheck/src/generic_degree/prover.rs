//! Per-round prover state and the default sumcheck driver.

use alloc::vec::Vec;

use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, Field};
use p3_multilinear_util::point::Point;

use super::proof::GenericDegreeProof;

/// Per-round callback used to drive the prover loop.
///
/// # Contract
///
/// - Carries the prover's mutable polynomial state across rounds.
/// - Exposes two operations called in fixed order by the driver.
pub trait RoundProver<EF> {
    /// Bind the most recently active variable to the verifier's challenge.
    ///
    /// - Called exactly once between consecutive round-polynomial requests.
    /// - In-place updates are recommended.
    fn fold(&mut self, r: EF);

    /// Return the current round polynomial as a list of evaluations.
    ///
    /// This reads the current state without mutating it; binding happens only in [`Self::fold`].
    ///
    /// # Returns
    ///
    /// Vector of length `degree` containing `h(0), h(2), h(3), ..., h(degree)`.
    fn round_poly(&self) -> Vec<EF>;

    /// Run the prover side of a generic-degree sumcheck driven by this state.
    ///
    /// # Arguments
    ///
    /// - `challenger`: Fiat-Shamir transcript shared with the verifier.
    /// - `num_rounds`: number of variables to bind.
    /// - `degree`: per-variable degree of the polynomial being summed.
    /// - `pow_bits`: grinding difficulty per round, or `0` to skip.
    /// - `claimed_sum`: claimed value of the sum at the start of round zero.
    ///
    /// # Returns
    ///
    /// - The transcript record, which carries the claimed sum.
    /// - The vector of challenges sampled by the verifier.
    ///
    /// # Transcript
    ///
    /// The caller must observe `claimed_sum` into the challenger before calling,
    /// so that the round challenges depend on it. This method observes only the
    /// round polynomials and any proof-of-work witnesses. The verifier observes
    /// the claimed sum at the matching point in [`GenericDegreeProof::verify`],
    /// so prover and verifier transcripts agree only when the caller mirrors that
    /// observe here.
    ///
    /// # Panics
    ///
    /// Panics if `degree` is zero.
    fn prove<F, Challenger>(
        &mut self,
        challenger: &mut Challenger,
        num_rounds: usize,
        degree: usize,
        pow_bits: usize,
        claimed_sum: EF,
    ) -> (GenericDegreeProof<F, EF>, Point<EF>)
    where
        F: Field,
        EF: ExtensionField<F>,
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        // A degree-zero polynomial would carry no information.
        assert!(degree > 0, "generic-degree sumcheck: degree must be > 0");

        let mut proof = GenericDegreeProof {
            claimed_sum,
            round_polys: Vec::with_capacity(num_rounds),
            pow_witnesses: Vec::with_capacity(if pow_bits > 0 { num_rounds } else { 0 }),
        };
        let mut challenges = Vec::with_capacity(num_rounds);

        for _r in 0..num_rounds {
            let evals = self.round_poly();
            debug_assert_eq!(
                evals.len(),
                degree,
                "round_poly returned {} evals, expected degree = {degree}",
                evals.len(),
            );

            challenger.observe_algebra_slice(&evals);

            // Optional proof-of-work; raises the cost of grinding a favorable challenge.
            if pow_bits > 0 {
                proof.pow_witnesses.push(challenger.grind(pow_bits));
            }

            let challenge: EF = challenger.sample_algebra_element();
            self.fold(challenge);

            proof.round_polys.push(evals);
            challenges.push(challenge);
        }

        (proof, Point::new(challenges))
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_challenger::DuplexChallenger;
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;
    use p3_multilinear_util::poly::Poly;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;
    use crate::generic_degree::GenericDegreeError;

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

    /// Test prover for a product of three multilinears.
    ///
    /// The summed polynomial is the pointwise product `a(x) * b(x) * c(x)` over the `log_m`-variable hypercube.
    struct TripleProductProver {
        /// Three multilinear factors, all over the same variable space.
        factors: [Poly<EF>; 3],
    }

    impl RoundProver<EF> for TripleProductProver {
        fn fold(&mut self, r: EF) {
            // Bind the most recently active variable in every factor.
            for f in &mut self.factors {
                f.fix_prefix_var_mut(r);
            }
        }

        fn round_poly(&self) -> Vec<EF> {
            // Return h(0), h(2), h(3) for the current round polynomial.
            let h0 = round_poly_product(&self.factors, EF::ZERO);
            let h2 = round_poly_product(&self.factors, EF::from_u64(2));
            let h3 = round_poly_product(&self.factors, EF::from_u64(3));
            vec![h0, h2, h3]
        }
    }

    fn round_poly_product(factors: &[Poly<EF>; 3], node: EF) -> EF {
        // Step 1: for each factor, interpolate between the two halves at X=node.
        //
        //     factor(node, x') = factor(0, x') + (factor(1, x') - factor(0, x')) * node
        let interpolated: Vec<Vec<EF>> = factors
            .iter()
            .map(|p| {
                let half = p.num_evals() / 2;
                (0..half)
                    .map(|i| {
                        let a0 = p.as_slice()[i];
                        let a1 = p.as_slice()[i + half];
                        a0 + (a1 - a0) * node
                    })
                    .collect()
            })
            .collect();

        // Step 2: sum the pointwise product across the unbound cube.
        let n_remaining = interpolated[0].len();
        let mut acc = EF::ZERO;
        for j in 0..n_remaining {
            let mut prod = EF::ONE;
            for v in &interpolated {
                prod *= v[j];
            }
            acc += prod;
        }
        acc
    }

    /// Build the three random factors and their ground-truth product sum.
    fn random_instance(rng: &mut SmallRng, m: usize) -> ([Vec<F>; 3], EF) {
        let a: Vec<F> = (0..m).map(|_| rng.random()).collect();
        let b: Vec<F> = (0..m).map(|_| rng.random()).collect();
        let c: Vec<F> = (0..m).map(|_| rng.random()).collect();
        let claimed_sum: EF = (0..m)
            .map(|i| EF::from(a[i]) * EF::from(b[i]) * EF::from(c[i]))
            .sum();
        ([a, b, c], claimed_sum)
    }

    fn prover_from(columns: &[Vec<F>; 3]) -> TripleProductProver {
        TripleProductProver {
            factors: [
                Poly::new(columns[0].iter().copied().map(EF::from).collect()),
                Poly::new(columns[1].iter().copied().map(EF::from).collect()),
                Poly::new(columns[2].iter().copied().map(EF::from).collect()),
            ],
        }
    }

    #[test]
    fn end_to_end_prove_verify_degree_3() {
        // Fixture state:
        //
        //     log_m = 6, so the hypercube has 64 points
        //     three random base-field columns a, b, c
        //     claimed sum = sum_x a(x) * b(x) * c(x)
        //
        // Invariant:
        //
        //     prover and verifier sample the same challenges
        //     the final claimed sum equals a(r) * b(r) * c(r) at the challenge point
        let mut rng = SmallRng::seed_from_u64(123);
        let log_m = 6usize;
        let (columns, claimed_sum) = random_instance(&mut rng, 1 << log_m);

        // Prover side.
        let mut prover_state = prover_from(&columns);
        let mut p_ch = fresh_challenger();
        p_ch.observe_algebra_element(claimed_sum);
        let (proof, prover_challenges) =
            prover_state.prove::<F, _>(&mut p_ch, log_m, 3, 0, claimed_sum);

        // The proof carries the claimed sum the verifier will consume.
        assert_eq!(proof.claimed_sum, claimed_sum);

        // After all rounds each factor has been bound down to a single value.
        let final_a = prover_state.factors[0].as_slice()[0];
        let final_b = prover_state.factors[1].as_slice()[0];
        let final_c = prover_state.factors[2].as_slice()[0];

        // Verifier side replays the transcript and recovers the final claim.
        let mut v_ch = fresh_challenger();
        let (verifier_challenges, final_sum) = proof.verify(&mut v_ch, log_m, 3, 0).unwrap();

        // Both sides must observe the same Fiat-Shamir transcript.
        assert_eq!(prover_challenges, verifier_challenges);
        // The final sumcheck equation is the product of multilinear evaluations.
        assert_eq!(final_sum, final_a * final_b * final_c);
    }

    #[test]
    fn end_to_end_prove_verify_with_grinding() {
        // Same product sumcheck, now with per-round proof-of-work enabled.
        // This exercises the grind / check-witness path that the degree-3 test skips.
        //
        // Fixture state: log_m = 4, pow_bits = 4 to keep grinding fast.
        let mut rng = SmallRng::seed_from_u64(321);
        let log_m = 4usize;
        let pow_bits = 4usize;
        let (columns, claimed_sum) = random_instance(&mut rng, 1 << log_m);

        let mut prover_state = prover_from(&columns);
        let mut p_ch = fresh_challenger();
        p_ch.observe_algebra_element(claimed_sum);
        let (proof, prover_challenges) =
            prover_state.prove::<F, _>(&mut p_ch, log_m, 3, pow_bits, claimed_sum);

        // Grinding emits exactly one witness per round.
        assert_eq!(proof.pow_witnesses.len(), log_m);

        let final_a = prover_state.factors[0].as_slice()[0];
        let final_b = prover_state.factors[1].as_slice()[0];
        let final_c = prover_state.factors[2].as_slice()[0];

        // The verifier re-checks each round's witness while replaying the transcript.
        let mut v_ch = fresh_challenger();
        let (verifier_challenges, final_sum) = proof.verify(&mut v_ch, log_m, 3, pow_bits).unwrap();

        assert_eq!(prover_challenges, verifier_challenges);
        assert_eq!(final_sum, final_a * final_b * final_c);
    }

    #[test]
    fn verify_rejects_tampered_pow_witness() {
        // Invariant: a corrupted PoW witness must fail the per-round grinding check.
        //
        // Fixture state: a valid grinded proof over log_m = 4 rounds.
        //
        // Mutation: corrupt the last round's witness.
        //   - rounds 0..2 keep valid witnesses
        //   - the transcript therefore stays in sync up to the final round
        //   - the final round then fails its grinding check
        let mut rng = SmallRng::seed_from_u64(99);
        let log_m = 4usize;
        let pow_bits = 4usize;
        let (columns, claimed_sum) = random_instance(&mut rng, 1 << log_m);

        let mut prover_state = prover_from(&columns);
        let mut p_ch = fresh_challenger();
        p_ch.observe_algebra_element(claimed_sum);
        let (mut proof, _) = prover_state.prove::<F, _>(&mut p_ch, log_m, 3, pow_bits, claimed_sum);

        let last_round = log_m - 1;
        let valid_witness = proof.pow_witnesses[last_round];

        // Why an offset search instead of a single `+1` tamper:
        //   - a corrupted witness still clears the grind by chance with probability 2^-pow_bits
        //   - parallel grinding may return any valid witness, not a fixed smallest one
        //
        // Walking successive offsets pins the rejection regardless of the witness grinding picked.
        let mut offset = F::ONE;
        loop {
            // Replace the honest witness with a nearby, almost-certainly-invalid value.
            proof.pow_witnesses[last_round] = valid_witness + offset;

            let mut v_ch = fresh_challenger();
            match proof.verify(&mut v_ch, log_m, 3, pow_bits) {
                // The failure is pinned to the corrupted round.
                Err(GenericDegreeError::InvalidPowWitness { round }) => {
                    assert_eq!(round, last_round);
                    break;
                }
                // Rare coincidental grind hit: advance to the next offset.
                Ok(_) => offset += F::ONE,
                Err(other) => panic!("expected an invalid pow witness error, got {other:?}"),
            }
        }
    }
}
