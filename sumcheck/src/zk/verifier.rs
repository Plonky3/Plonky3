//! HVZK verifier with affine-chain replay over the prefix-binding layout.

use alloc::vec::Vec;

use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_field::{ExtensionField, Field, HornerIter};
use p3_multilinear_util::point::Point;

use super::data::ZkSumcheckData;
use crate::error::SumcheckError;
use crate::layout::{LayoutStrategy, Verifier};
use crate::strategy::VariableOrder;
use crate::table::TableShape;

/// HVZK verifier for the prefix-binding sumcheck.
///
/// Per round, the verifier:
///
/// - reads wire `[c_0, c_2, c_3, ..., c_d]` (linear coefficient dropped),
/// - reconstructs `c_1` from `h_j(0) + h_j(1) = target`,
/// - checks the proof-of-work witness when enabled,
/// - samples `gamma_j` and sets the next target to `h_j(gamma_j)`.
pub struct ZkVerifier<F, EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    /// Plain prefix-binding verifier holding the claims that fix `mu`.
    inner: Verifier<F, EF>,
}

impl<F, EF> ZkVerifier<F, EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    /// Builds the verifier registry.
    pub fn new(table_shapes: &[TableShape]) -> Self {
        // Layout must match the prover's, otherwise claim points are lifted under the wrong selector.
        // Pinned by the drift-guard test in this module.
        Self {
            inner: Verifier::new(
                table_shapes,
                LayoutStrategy::new(true, VariableOrder::Prefix),
            ),
        }
    }

    /// Records concrete opening claims on the inner verifier.
    pub fn add_claim<Ch>(
        &mut self,
        table_idx: usize,
        polys: &[usize],
        evals: &[EF],
        challenger: &mut Ch,
    ) where
        Ch: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        // Delegate; the HVZK overlay carries no extra state at claim time.
        self.inner.add_claim(table_idx, polys, evals, challenger);
    }

    /// Records a virtual evaluation claim on the inner verifier.
    pub fn add_virtual_eval<Ch>(&mut self, eval: EF, challenger: &mut Ch)
    where
        Ch: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        // Same delegation pattern as concrete openings.
        self.inner.add_virtual_eval(eval, challenger);
    }

    /// Claim sum `mu` weighted by powers of `alpha`.
    ///
    /// Reads only the recorded claims, not any witness data.
    /// Used by the witness-free simulator to derive `mu` without re-implementing the alpha-power loop.
    pub(crate) fn sum(&self, alpha: EF) -> EF {
        self.inner.sum(alpha)
    }

    /// Replays the prover's HVZK sumcheck transcript.
    ///
    /// # Phases
    ///
    /// 1. Reject malformed shapes up front.
    /// 2. Sample alpha and derive `mu` from the recorded claims.
    /// 3. Absorb mask commits and `mu_tilde`, then sample `eps`.
    /// 4. Walk the round chain: reconstruct `c_1`, check PoW, sample `gamma_j`, advance the target by Horner evaluation.
    ///
    /// # Returns
    ///
    /// - Vector of per-round challenges `gamma_1, ..., gamma_k`.
    /// - Residual claim `target = h_k(gamma_k)`, fed to the downstream committed-sumcheck reduction.
    ///
    /// # Errors
    ///
    /// - Mismatch between the verifier-side and proof-side mask code length.
    /// - Wrong number of rounds, mask commitments, or PoW witnesses.
    /// - A per-round wire of the wrong shape.
    /// - A failing proof-of-work witness check.
    ///
    /// # Panics
    ///
    /// - Base field characteristic is `2`.
    /// - Mask code message length is below `2`.
    /// - Folding factor is `0`.
    #[allow(clippy::too_many_arguments)]
    pub fn into_sumcheck<M, Ch>(
        self,
        zk_data: &ZkSumcheckData<F, EF>,
        mask_commits: &[M::Commitment],
        ell_zk: usize,
        folding_factor: usize,
        pow_bits: usize,
        challenger: &mut Ch,
    ) -> Result<(Point<EF>, EF), SumcheckError>
    where
        M: Mmcs<EF>,
        Ch: FieldChallenger<F> + GrindingChallenger<Witness = F> + CanObserve<M::Commitment>,
    {
        // Lemma 6.4 hypotheses on the verifier-side parameters.
        assert!(F::TWO != F::ZERO, "Lemma 6.4 requires char(F) != 2");
        assert!(ell_zk >= 2, "Lemma 6.4 requires ell_zk >= 2");
        assert!(folding_factor >= 1, "sumcheck requires at least one round");

        // Phase 1: shape checks (input validation before Construction 6.3 replay).

        // ell_zk mismatch reject.
        // Closes the (2, 3) non-injectivity gap in the wire-shape check: lengths 2 and 3 share a wire layout.
        if zk_data.ell_zk != ell_zk {
            return Err(SumcheckError::EllZkMismatch {
                expected: ell_zk,
                actual: zk_data.ell_zk,
            });
        }
        // One wire entry per round.
        if zk_data.round_coefficients.len() != folding_factor {
            return Err(SumcheckError::RoundCountMismatch {
                expected: folding_factor,
                actual: zk_data.round_coefficients.len(),
            });
        }
        // One mask commitment per round.
        if mask_commits.len() != folding_factor {
            return Err(SumcheckError::MaskCommitmentCountMismatch {
                expected: folding_factor,
                actual: mask_commits.len(),
            });
        }
        // PoW witnesses are required only when grinding is enabled.
        let expected_pow = if pow_bits > 0 { folding_factor } else { 0 };
        if zk_data.pow_witnesses.len() != expected_pow {
            return Err(SumcheckError::PowWitnessCountMismatch {
                expected: expected_pow,
                actual: zk_data.pow_witnesses.len(),
            });
        }
        // Each wire carries h_size - 1 coefficients (c_1 dropped):
        //
        //     h_size    = max(ell_zk, 3)
        //     wire_size = h_size - 1
        let h_size = ell_zk.max(3);
        let wire_size = h_size - 1;
        for (idx, wire) in zk_data.round_coefficients.iter().enumerate() {
            if wire.len() != wire_size {
                return Err(SumcheckError::WireSizeMismatch {
                    round: idx + 1,
                    expected: wire_size,
                    actual: wire.len(),
                });
            }
        }

        // Phase 2: transcript prelude (matches the prover byte-for-byte; replays Construction 6.3 setup).

        // Sample alpha, then derive mu from the recorded claims.
        let alpha: EF = challenger.sample_algebra_element();
        let mu = self.inner.sum(alpha);

        // Phase 3: absorb mask commits and mu_tilde, sample eps (Construction 6.3 steps 1-3 replay).
        for commit in mask_commits {
            challenger.observe(commit.clone());
        }
        challenger.observe_algebra_element(zk_data.mu_tilde);
        let eps: EF = challenger.sample_algebra_element();

        // Phase 4: walk the round chain (Construction 6.3 step 4 verifier replay).
        //
        // Round-1 target from the initial Construction 6.3 verifier identity:
        //
        //     h_1(0) + h_1(1) = eps * mu + mu_tilde
        let mut target: EF = eps * mu + zk_data.mu_tilde;
        let mut randomness: Vec<EF> = Vec::with_capacity(folding_factor);

        for (j_idx, wire) in zk_data.round_coefficients.iter().enumerate() {
            // Wire layout (`c_1` dropped):  [ c_0, c_2, c_3, ..., c_d ].
            let c0 = wire[0];
            let high_sum: EF = wire[1..].iter().copied().sum();

            // Reconstruct c_1 from the Construction 6.3 verifier identity:
            //
            //     2 c_0 + c_1 + sum_{i>=2} c_i = target
            //     => c_1 = target - 2 c_0 - sum_{i>=2} c_i
            let c1 = target - c0.double() - high_sum;

            // Absorb the wire on the transcript.
            challenger.observe_algebra_slice(wire);

            // Optional proof-of-work check before the per-round challenge.
            if pow_bits > 0 && !challenger.check_witness(pow_bits, zk_data.pow_witnesses[j_idx]) {
                return Err(SumcheckError::InvalidPowWitness);
            }

            // Sample the per-round challenge.
            let gamma_j: EF = challenger.sample_algebra_element();

            // Next target via Horner (Construction 6.3 round chaining: target_{j+1} = h_j(gamma_j)):
            //
            //     h_j(gamma) = c_0 + gamma * (c_1 + gamma * (c_2 + ... + gamma * c_d))
            let mut coeffs_vec: Vec<EF> = Vec::with_capacity(h_size);
            coeffs_vec.push(c0);
            coeffs_vec.push(c1);
            coeffs_vec.extend_from_slice(&wire[1..]);
            let h_at_gamma_j: EF = coeffs_vec.iter().copied().horner(gamma_j);

            target = h_at_gamma_j;
            randomness.push(gamma_j);
        }

        Ok((Point::new(randomness), target))
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use p3_field::{Field, PackedValue, PrimeCharacteristicRing};
    use proptest::prelude::*;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;
    use crate::layout::TableShape;
    use crate::zk::test_helpers::{EF, F, MyChallenger, MyMmcs, build_prover_verifier, make_setup};
    use crate::zk::{ZkSumcheckData, ZkVerifier};

    #[test]
    fn forged_pow_witness_rejected() {
        // Invariant: a tampered PoW witness is rejected with the dedicated invalid-witness error.
        //
        // Fixture state:
        //
        //     n_vars = 6, folding = 2, ell_zk = 4, num_eqs = 1, seed = 0
        //     pow_bits = 16
        //
        //     |valid preimages| / |F| = 2^{32 - 16} / 2^32 = 2^{-16}
        //
        // # Why one seed is enough
        //
        // Random tampering passes the difficulty check with probability 2^{-16}.
        // Effectively zero for a single concrete seed.
        //
        // Mutation: bump the first round's PoW witness by ONE.
        //
        //     proof:    pow_witnesses = [ w_0 + 1, w_1 ]
        //     verifier: rederives difficulty, rejects on round 1
        //               -> InvalidPowWitness
        let n_vars = 6;
        let folding_factor = 2;
        let ell_zk = 4;
        let num_eqs = 1;
        let seed = 0u64;
        let pow_bits = 16;

        // Deterministic permutation, MMCS, and encoding from the seed.
        let (perm, mmcs, encoding) = make_setup(seed, ell_zk);
        let mut data_rng = SmallRng::seed_from_u64(seed.wrapping_add(1));
        let evals: Vec<F> = (0..(1usize << n_vars)).map(|_| data_rng.random()).collect();

        // Matched prover and verifier on the same witness shape.
        let (mut prover, mut verifier, _) =
            build_prover_verifier(evals, folding_factor, encoding, mmcs);

        // Synchronised claim phase.
        let mut prover_ch = MyChallenger::new(perm.clone());
        let mut verifier_ch = MyChallenger::new(perm);
        for _ in 0..num_eqs {
            let eval = prover.add_virtual_eval(&mut prover_ch);
            verifier.add_virtual_eval(eval, &mut verifier_ch);
        }

        // Honest prover sumcheck with grinding enabled.
        let mut zk_data = ZkSumcheckData::<F, EF>::default();
        let mut prover_rng = SmallRng::seed_from_u64(seed.wrapping_add(2));
        let (_residual, _rand, mask_oracles) =
            prover.into_sumcheck(&mut zk_data, pow_bits, &mut prover_ch, &mut prover_rng);

        // Sanity check before tampering: one witness per round.
        assert_eq!(zk_data.pow_witnesses.len(), folding_factor);
        // Mutation.
        zk_data.pow_witnesses[0] += F::ONE;

        // Verifier replay against the tampered proof.
        let mask_commits: Vec<_> = mask_oracles.iter().map(|(c, _)| c.clone()).collect();
        let result = verifier.into_sumcheck::<MyMmcs, _>(
            &zk_data,
            &mask_commits,
            ell_zk,
            folding_factor,
            pow_bits,
            &mut verifier_ch,
        );

        assert!(
            matches!(result, Err(SumcheckError::InvalidPowWitness)),
            "verifier accepted a forged PoW witness; got {result:?}",
        );
    }

    #[test]
    fn ell_zk_mismatch_rejected() {
        // Invariant: prover/verifier disagreement on ell_zk is rejected up front.
        //
        // # Why a dedicated check
        //
        // The wire-shape check is non-injective in {2, 3}: both produce wire_size = 2.
        // Without the dedicated mismatch error, the bug would slip past the shape check.
        //
        // Fixture state:
        //
        //     n_vars = 6, folding = 2, ell_zk = 4, num_eqs = 1, seed = 0
        //
        // Mutation:
        //
        //     proof:    ell_zk = 4
        //     verifier: ell_zk = 5
        //                  -> EllZkMismatch { expected: 5, actual: 4 }
        let n_vars = 6;
        let folding_factor = 2;
        let ell_zk = 4;
        let num_eqs = 1;
        let seed = 0u64;
        let pow_bits = 0;

        // Deterministic setup.
        let (perm, mmcs, encoding) = make_setup(seed, ell_zk);
        let mut data_rng = SmallRng::seed_from_u64(seed.wrapping_add(1));
        let evals: Vec<F> = (0..(1usize << n_vars)).map(|_| data_rng.random()).collect();

        // Matched prover and verifier.
        let (mut prover, mut verifier, _) =
            build_prover_verifier(evals, folding_factor, encoding, mmcs);

        // Synchronised claim phase.
        let mut prover_ch = MyChallenger::new(perm.clone());
        let mut verifier_ch = MyChallenger::new(perm);
        for _ in 0..num_eqs {
            let eval = prover.add_virtual_eval(&mut prover_ch);
            verifier.add_virtual_eval(eval, &mut verifier_ch);
        }

        // Honest prover run; ell_zk recorded in zk_data is 4.
        let mut zk_data = ZkSumcheckData::<F, EF>::default();
        let mut prover_rng = SmallRng::seed_from_u64(seed.wrapping_add(2));
        let (_residual, _rand, mask_oracles) =
            prover.into_sumcheck(&mut zk_data, pow_bits, &mut prover_ch, &mut prover_rng);

        // Verifier replay with the wrong ell_zk parameter.
        let wrong_ell_zk = ell_zk + 1;
        let mask_commits: Vec<_> = mask_oracles.iter().map(|(c, _)| c.clone()).collect();
        let result = verifier.into_sumcheck::<MyMmcs, _>(
            &zk_data,
            &mask_commits,
            wrong_ell_zk,
            folding_factor,
            pow_bits,
            &mut verifier_ch,
        );

        assert!(
            matches!(
                result,
                Err(SumcheckError::EllZkMismatch { expected, actual })
                    if expected == wrong_ell_zk && actual == ell_zk
            ),
            "verifier should have rejected ell_zk mismatch; got {result:?}",
        );
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(8))]

        #[test]
        fn prop_rbr_tampering_changes_verifier_output(
            n_vars in 3usize..=6,
            ell_zk in 2usize..=4,
            num_eqs in 1usize..=2,
            seed in 0u64..512,
            tamper_round_seed in 0usize..16,
            tamper_pos_seed in 0usize..8,
        ) {
            // Invariant: flipping any single coordinate of any round's wire shifts the verifier's final target away from the honest run.
            //
            // # Why local checks are not enough
            //
            // The affine reconstruction of c_1 forces the per-round identity to hold on a tampered wire.
            // The verifier therefore does not reject locally.
            // RBR soundness still needs the cheat caught.
            // This test asserts the divergence propagates through Fiat-Shamir.
            //
            // # Coverage role
            //
            // Lemma 6.5 bounds the per-round rehabilitation probability:
            //
            //     eps_j <= eps_mca + ell_zk * |Lambda|^2 / |F|
            //
            // That is a theorem about the abstract protocol.
            // What an implementation can test is conformance, which is what this proptest does.
            //
            // # Not tested here
            //
            // The quantitative empirical-rate match.
            // With gamma_j sampled from EF approx 2^124 the bound is approx 2^-110.
            // CI cannot host the trial count needed to observe it.

            // Compression step requires the folded polynomial to retain at least one full packed lane.
            let k_pack = p3_util::log2_strict_usize(<F as Field>::Packing::WIDTH);
            prop_assume!(n_vars > k_pack);
            let folding_factor = 1 + (seed as usize % (n_vars - k_pack));

            // Deterministic setup for the run.
            let (perm, mmcs, encoding) = make_setup(seed, ell_zk);

            let mut data_rng = SmallRng::seed_from_u64(seed.wrapping_add(1));
            let evals: Vec<F> = (0..(1usize << n_vars)).map(|_| data_rng.random()).collect();

            let (mut prover, mut verifier, _n_vars) =
                build_prover_verifier(evals, folding_factor, encoding, mmcs);

            // Synchronised claim phase.
            let mut prover_challenger = MyChallenger::new(perm.clone());
            let mut verifier_challenger = MyChallenger::new(perm.clone());
            for _ in 0..num_eqs {
                let eval = prover.add_virtual_eval(&mut prover_challenger);
                verifier.add_virtual_eval(eval, &mut verifier_challenger);
            }

            // Honest prover sumcheck.
            let pow_bits = 0;
            let mut zk_data = ZkSumcheckData::<F, EF>::default();
            let mut prover_rng = SmallRng::seed_from_u64(seed.wrapping_add(2));
            let (_residual_prover, _gammas, mask_oracles) = prover.into_sumcheck(
                &mut zk_data,
                pow_bits,
                &mut prover_challenger,
                &mut prover_rng,
            );
            let mask_commits: Vec<_> = mask_oracles.iter().map(|(c, _)| c.clone()).collect();

            // Mutation: pick a uniformly random (round, position) and add ONE to that wire coefficient.
            let tamper_round = tamper_round_seed % zk_data.round_coefficients.len();
            let wire_len = zk_data.round_coefficients[tamper_round].len();
            let tamper_pos = tamper_pos_seed % wire_len;
            let mut tampered_zk_data = zk_data.clone();
            tampered_zk_data.round_coefficients[tamper_round][tamper_pos] += F::ONE;

            // Honest verifier replay using the untampered proof.
            let mut honest_v_challenger = MyChallenger::new(perm.clone());
            let mut honest_verifier =
                ZkVerifier::<F, EF>::new(&[TableShape::new(n_vars, 1)]);

            // Parallel prover used only to replay the claim phase.
            //
            //     replay_prover -> evals -> honest_verifier (observes the same draws)
            //
            // The prover output is discarded; only Fiat-Shamir state matters.
            let mut prover_replay = MyChallenger::new(perm.clone());
            let mut replay_evals = Vec::with_capacity(num_eqs);
            let (mut replay_prover, _, _) = {
                let mut replay_rng = SmallRng::seed_from_u64(seed.wrapping_add(1));
                let evals: Vec<F> =
                    (0..(1usize << n_vars)).map(|_| replay_rng.random()).collect();
                let (perm2, mmcs2, encoding2) = make_setup(seed, ell_zk);
                // perm2 is unused: the outer perm is already in scope.
                let _ = perm2;
                build_prover_verifier(evals, folding_factor, encoding2, mmcs2)
            };
            for _ in 0..num_eqs {
                let e = replay_prover.add_virtual_eval(&mut prover_replay);
                honest_verifier.add_virtual_eval(e, &mut honest_v_challenger);
                replay_evals.push(e);
            }
            let honest_result = honest_verifier.into_sumcheck::<MyMmcs, _>(
                &zk_data,
                &mask_commits,
                ell_zk,
                folding_factor,
                pow_bits,
                &mut honest_v_challenger,
            );
            prop_assert!(honest_result.is_ok());
            let (_honest_rand, honest_target) = honest_result.unwrap();

            // Tampered verifier replay: same setup, mutated wires.
            let mut tampered_v_challenger = MyChallenger::new(perm);
            let mut tampered_verifier =
                ZkVerifier::<F, EF>::new(&[TableShape::new(n_vars, 1)]);
            for &e in &replay_evals {
                tampered_verifier.add_virtual_eval(e, &mut tampered_v_challenger);
            }
            let tampered_result = tampered_verifier.into_sumcheck::<MyMmcs, _>(
                &tampered_zk_data,
                &mask_commits,
                ell_zk,
                folding_factor,
                pow_bits,
                &mut tampered_v_challenger,
            );
            prop_assert!(tampered_result.is_ok());
            let (_tampered_rand, tampered_target) = tampered_result.unwrap();

            // The two targets must differ.
            // Probability of accidental coincidence is bounded by Lemma 6.5's negligible soundness error.
            prop_assert_ne!(
                honest_target, tampered_target,
                "tampering with wire coordinate ({}, {}) must change target",
                tamper_round, tamper_pos,
            );
        }
    }
}
