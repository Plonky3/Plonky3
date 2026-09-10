//! HVZK overlay for an already-derived residual sumcheck claim.

use alloc::vec::Vec;

use p3_challenger::fs::TranscriptField;
use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_field::{ExtensionField, Field, HornerIter};
use p3_matrix::Matrix;
use p3_multilinear_util::point::Point;
use p3_zk_codes::ZkEncodingWithRandomness;
use rand::Rng;

use super::common::{mask_endpoints, sample_masks};
use super::round::{PlainPiece, RoundContext, RoundState, round_poly_to_wire};
use crate::strategy::SumcheckProver;
use crate::zk::transcript::{ZkProverTranscript, ZkSumcheckShape};
use crate::zk::{ZkSumcheckData, ZkSumcheckHandoff};

impl<F, EF> SumcheckProver<F, EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    /// Runs the HVZK sumcheck overlay on an already-derived residual product
    /// polynomial.
    ///
    /// This is the post-code-switch analogue of `ZkPrefixProver::into_sumcheck`:
    /// the caller has already reduced the layout-specific opening relation to a
    /// product polynomial, and this method applies Construction 6.3's mask
    /// transcript to the next batch of sumcheck rounds.
    ///
    /// # Joint claims and the auxiliary constant
    ///
    /// The committed-sumcheck relation (Definition 5.8 of eprint 2026/391)
    /// pairs the source claim `<f, w>` with mask-oracle claims `<xi_i, u_i>`.
    ///
    /// - The mask-claim values are prover-only; their total is the
    ///   auxiliary constant.
    /// - The bound scalar is the joint claim: source claim plus that
    ///   constant.
    /// - The constant rides the affine chain with a `2^{-j}` carry per
    ///   round:
    ///
    /// ```text
    ///     h_j gains  eps * aux * 2^{-j}  on its constant slot
    ///     =>  h_j(0) + h_j(1)  gains  eps * aux * 2^{-(j-1)}
    ///     =>  the final residual gains  eps * aux * 2^{-k}
    /// ```
    ///
    /// Downstream reductions must therefore scale the carried mask covectors
    /// by `eps * 2^{-k}`.
    ///
    /// # Eval side
    ///
    /// - Only the weight side and the claim are scaled by `eps`.
    /// - The evaluation side stays the honest folded message.
    /// - An HVZK code-switch can therefore commit it verbatim.
    ///
    /// # Panics
    ///
    /// - The configuration cannot describe a masked batch.
    /// - Folding factor exceeds the residual prover's arity.
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    #[tracing::instrument(skip_all)]
    pub fn into_zk_sumcheck<Enc, M, R, Ch>(
        mut self,
        zk_data: &mut ZkSumcheckData<F, EF>,
        encoding: &Enc,
        mmcs: &M,
        folding_factor: usize,
        pow_bits: usize,
        aux_claim: EF,
        challenger: &mut Ch,
        rng: &mut R,
    ) -> ZkSumcheckHandoff<F, EF, M>
    where
        F: TranscriptField,
        Enc: ZkEncodingWithRandomness<EF>,
        Enc::Codeword: Matrix<EF>,
        M: Mmcs<EF>,
        R: Rng,
        Ch: FieldChallenger<F> + GrindingChallenger<Witness = F> + CanObserve<M::Commitment>,
    {
        let ell_zk = encoding.message_len();

        // This entry inherits its claim instead of batching recorded ones.
        let shape = ZkSumcheckShape::new_inherited(folding_factor, ell_zk, pow_bits);

        // Lemma 6.4 hypotheses, plus the one bound the transcript never sees.
        shape
            .validate::<F>()
            .expect("a prover's own configuration must describe a masked batch");
        assert!(
            folding_factor <= self.num_variables(),
            "folding_factor must be <= residual prover arity",
        );

        let mut transcript = ZkProverTranscript::<Ch, F, EF>::new(challenger, shape);

        // Unlike the layout-driven path, this entry receives a scalar claim
        // directly, so the description opens by binding it.
        //
        // The bound value is the joint claim, matching the verifier's view.
        transcript.bind_claim(self.claimed_sum() + aux_claim);

        let (masks, mask_randomness, mask_oracle) =
            sample_masks::<EF, _, _, _>(folding_factor, encoding, mmcs, rng);
        let (mu_tilde, mut sum_future_endpoints) = mask_endpoints::<EF>(&masks, folding_factor);
        zk_data.mu_tilde = mu_tilde;

        let eps: EF = transcript.masks(mask_oracle.0.clone(), mu_tilde);
        let mut rs = Vec::with_capacity(folding_factor);
        let mut mask_evals_at_gamma = Vec::with_capacity(folding_factor);
        let pow2: Vec<EF> = EF::TWO.powers().collect_n(folding_factor + 1);
        let round_ctx = RoundContext {
            k: folding_factor,
            ell_zk,
            pow2: &pow2,
            eps,
        };

        // Running `aux * 2^{-j}` carry; halved once per round.
        let half = EF::TWO.inverse();
        let mut aux_carry = aux_claim;

        for (round_idx, mask) in masks.iter().enumerate() {
            let j = round_idx + 1;
            let mask_endpoints = mask[0].double() + mask[1..].iter().copied().sum::<EF>();
            sum_future_endpoints -= mask_endpoints;
            aux_carry *= half;

            // Measure this round, absorbing whatever binding the last one left behind.
            //
            // Everything the loop does between the two is scalar work.
            // None of the steps below read the tables.
            //
            //     round polynomial   transcript   grinding   mask evaluation
            let (plain_c0, plain_c_inf) = self.measure_round();
            // The aux carry enters only the transmitted constant slot; the
            // source-side fold below keeps the raw coefficients.
            let h = round_ctx.assemble(
                RoundState {
                    j,
                    mask,
                    past_mask_evals: &mask_evals_at_gamma,
                    future_endpoints: sum_future_endpoints,
                },
                PlainPiece {
                    c0: plain_c0 + aux_carry,
                    c_inf: plain_c_inf,
                },
            );
            let wire = round_poly_to_wire(&h);

            // One call binds the wire, grinds when enabled, and draws the challenge.
            let (gamma, witness) = transcript.round(&wire);
            zk_data.round_coefficients.push(wire);
            zk_data.pow_witnesses.extend(witness);

            let mask_at_gamma = mask.iter().copied().horner(gamma);
            mask_evals_at_gamma.push(mask_at_gamma);

            // Advance the claim now; the binding waits for the next round's pass.
            self.reduce_claim_with_coefficients(plain_c0, plain_c_inf, gamma);
            self.hold(gamma);

            rs.push(gamma);
        }

        // Every described step has been played, so the sponge goes back to the caller.
        transcript.finish();

        // The last challenge has no successor to fuse with.
        //
        // The weight scaling below reads the tables, so it settles the binding on the way in.
        //
        // Its own settling step checks the claim against what it bound.
        self.scale_weights_and_claim(eps);

        ZkSumcheckHandoff {
            residual_prover: self,
            randomness: Point::new(rs),
            eps,
            mask_messages: masks,
            mask_randomness,
            mask_oracle,
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;
    use alloc::{format, vec};

    use p3_baby_bear::BabyBear;
    use p3_field::extension::BinomialExtensionField;
    use p3_field::{Field, PackedValue, PrimeCharacteristicRing, dot_product};
    use p3_matrix::dense::RowMajorMatrix;
    use p3_multilinear_util::poly::Poly;
    use p3_util::log2_strict_usize;
    use p3_zk_codes::{ZkEncoding, ZkEncodingWithRandomness};
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    use super::*;
    use crate::product_polynomial::ProductPolynomial;
    use crate::strategy::VariableOrder;
    use crate::zk::test_helpers::{MyChallenger, MyMmcs, make_setup};
    use crate::zk::{ZkVerifier, mask_residual};

    type F = BabyBear;
    type EF = BinomialExtensionField<BabyBear, 4>;

    #[derive(Clone)]
    struct SentinelEncoding {
        ell_zk: usize,
    }

    impl ZkEncoding<EF> for SentinelEncoding {
        type Codeword = RowMajorMatrix<EF>;

        fn message_len(&self) -> usize {
            self.ell_zk
        }

        fn randomness_len(&self) -> usize {
            0
        }

        fn error(&self) -> f64 {
            0.0
        }

        fn sample_message<R: Rng>(&self, _rng: &mut R) -> Vec<EF> {
            (0..self.ell_zk)
                .map(|idx| EF::from_u64(100 + idx as u64))
                .collect()
        }

        fn query_bound(&self) -> usize {
            0
        }

        fn codeword_len(&self) -> usize {
            self.ell_zk
        }

        fn encode<R: Rng>(&self, msg: &[EF], _rng: &mut R) -> Self::Codeword {
            RowMajorMatrix::new_col(msg.to_vec())
        }

        fn sample_randomness<R: Rng>(&self, _rng: &mut R) -> Vec<EF> {
            Vec::new()
        }

        fn simulate<R: Rng>(&self, query_set: &[usize], _rng: &mut R) -> Vec<EF> {
            EF::zero_vec(query_set.len())
        }
    }

    impl ZkEncodingWithRandomness<EF> for SentinelEncoding {
        fn encode_with_randomness(&self, msg: &[EF], randomness: &[EF]) -> Self::Codeword {
            assert!(randomness.is_empty());
            RowMajorMatrix::new_col(msg.to_vec())
        }
    }

    #[test]
    fn residual_prover_zk_handoff_replays_from_claim() {
        let evals = Poly::new((1..=8).map(EF::from_u64).collect::<Vec<_>>());
        let weights = Poly::new((11..=18).map(EF::from_u64).collect::<Vec<_>>());
        let claimed_sum = dot_product::<EF, _, _>(
            evals.as_slice().iter().copied(),
            weights.as_slice().iter().copied(),
        );
        let poly = ProductPolynomial::<F, EF>::new_unpacked(VariableOrder::Prefix, evals, weights);
        let prover = SumcheckProver::new(poly, claimed_sum);

        let ell_zk = 4;
        let folding_factor = 2;
        let (perm, mmcs, encoding) = make_setup(17, ell_zk);
        let mut prover_challenger = MyChallenger::new(perm.clone());
        let mut verifier_challenger = MyChallenger::new(perm);
        let mut rng = SmallRng::seed_from_u64(19);
        let mut zk_data = ZkSumcheckData::<F, EF>::default();

        let prover_handoff = prover.into_zk_sumcheck(
            &mut zk_data,
            &encoding,
            &mmcs,
            folding_factor,
            0,
            EF::ZERO,
            &mut prover_challenger,
            &mut rng,
        );
        let mask_commitment = prover_handoff.mask_oracle.0.clone();

        let verifier_handoff = ZkVerifier::<F, EF>::verify_claim::<MyMmcs, _>(
            &zk_data,
            &mask_commitment,
            ell_zk,
            folding_factor,
            0,
            claimed_sum,
            &mut verifier_challenger,
        )
        .expect("honest residual ZK handoff should verify");

        assert_eq!(verifier_handoff.randomness, prover_handoff.randomness);
        assert_eq!(verifier_handoff.eps, prover_handoff.eps);

        let gammas = prover_handoff
            .randomness
            .iter()
            .copied()
            .collect::<Vec<_>>();
        let final_mask_residual = mask_residual::<EF>(&prover_handoff.mask_messages, &gammas);
        assert_eq!(
            verifier_handoff.claimed_residual,
            prover_handoff.residual_prover.claimed_sum() + final_mask_residual,
        );
    }

    #[test]
    fn a_verifier_handed_a_different_claim_diverges() {
        // Invariant: the inherited claim is a described step, bound on both sides.
        //
        // Nothing inside this crate can compare the prover's scalar with the verifier's.
        //
        //     prover   ->  claimed_sum + aux_claim
        //     verifier ->  whatever its caller supplies
        //
        // What the step buys is that a caller which disagrees moves `eps` and every challenge
        // after it, so the residual handed back no longer matches the prover's.
        //
        // Fixture state: 3 variables, 2 rounds, mask length 4, no auxiliary claim.
        let evals = Poly::new((1..=8).map(EF::from_u64).collect::<Vec<_>>());
        let weights = Poly::new((11..=18).map(EF::from_u64).collect::<Vec<_>>());
        let claimed_sum = dot_product::<EF, _, _>(
            evals.as_slice().iter().copied(),
            weights.as_slice().iter().copied(),
        );
        let poly = ProductPolynomial::<F, EF>::new_unpacked(VariableOrder::Prefix, evals, weights);

        let ell_zk = 4;
        let folding_factor = 2;
        let (perm, mmcs, encoding) = make_setup(41, ell_zk);
        let mut prover_challenger = MyChallenger::new(perm.clone());
        let mut rng = SmallRng::seed_from_u64(43);
        let mut zk_data = ZkSumcheckData::<F, EF>::default();

        let prover_handoff = SumcheckProver::new(poly, claimed_sum).into_zk_sumcheck(
            &mut zk_data,
            &encoding,
            &mmcs,
            folding_factor,
            0,
            EF::ZERO,
            &mut prover_challenger,
            &mut rng,
        );
        let mask_commitment = prover_handoff.mask_oracle.0.clone();

        // One replay per claim, both from the same fresh sponge.
        let replay = |claim: EF| {
            let mut challenger = MyChallenger::new(perm.clone());
            ZkVerifier::<F, EF>::verify_claim::<MyMmcs, _>(
                &zk_data,
                &mask_commitment,
                ell_zk,
                folding_factor,
                0,
                claim,
                &mut challenger,
            )
            .expect("a well-shaped proof always replays")
        };

        let honest = replay(claimed_sum);
        let tampered = replay(claimed_sum + EF::ONE);

        // The claim is bound before `eps`, so the whole stream moves with it.
        assert_ne!(honest.eps, tampered.eps);
        assert_ne!(honest.randomness, tampered.randomness);
        assert_ne!(honest.claimed_residual, tampered.claimed_residual);

        // The honest replay is the one that matches the prover.
        assert_eq!(honest.randomness, prover_handoff.randomness);
        assert_eq!(honest.eps, prover_handoff.eps);
    }

    #[test]
    fn deferred_binding_drives_the_same_hiding_rounds_as_binding_on_the_spot() {
        // Invariant: holding a binding back a round changes nothing this driver produces.
        //
        // Fixture state: five shapes, both binding orders, a non-zero auxiliary claim.
        //
        //     (2, 1)   one round, so the fused branch is never taken
        //     (4, 4)   every variable bound, the terminal case
        //     (9, 3)   packed storage, then the unpacking handoff
        //     (9, 9)   packed storage bound all the way down
        //     (15, 3)  large enough to reach the threaded branch of the fused pass
        let log_width = log2_strict_usize(<F as Field>::Packing::WIDTH);

        for (n_vars, folding_factor) in [(2usize, 1usize), (4, 4), (9, 3), (9, 9), (15, 3)] {
            for order in [VariableOrder::Prefix, VariableOrder::Suffix] {
                let mut rng = SmallRng::seed_from_u64(0x5EED + n_vars as u64);
                let evals = Poly::<EF>::rand(&mut rng, n_vars);
                let weights = Poly::<EF>::rand(&mut rng, n_vars);
                let claimed_sum = dot_product::<EF, _, _>(
                    evals.as_slice().iter().copied(),
                    weights.as_slice().iter().copied(),
                );

                // A pair below one SIMD lane group has nothing to pack.
                let build = || {
                    if n_vars >= log_width {
                        ProductPolynomial::<F, EF>::new_packed(
                            order,
                            evals.pack::<F, EF>(),
                            weights.pack::<F, EF>(),
                        )
                    } else {
                        ProductPolynomial::<F, EF>::new_unpacked(
                            order,
                            evals.clone(),
                            weights.clone(),
                        )
                    }
                };

                // The auxiliary claim rides the transmitted constant slot only.
                // A non-zero one would show up here if it ever reached the binding.
                let aux_claim = EF::from_u64(7);

                let ell_zk = 4;
                let (perm, mmcs, encoding) = make_setup(31, ell_zk);
                let mut challenger = MyChallenger::new(perm);
                let mut mask_rng = SmallRng::seed_from_u64(37);
                let mut zk_data = ZkSumcheckData::<F, EF>::default();

                // Arm under test: the driver, which holds each binding back a round.
                let mut handoff = SumcheckProver::new(build(), claimed_sum).into_zk_sumcheck(
                    &mut zk_data,
                    &encoding,
                    &mmcs,
                    folding_factor,
                    0,
                    aux_claim,
                    &mut challenger,
                    &mut mask_rng,
                );

                // Reference arm: replay the same challenges, binding each on the spot.
                let mut reference = SumcheckProver::new(build(), claimed_sum);
                for &gamma in handoff.randomness.iter() {
                    let (c0, c_inf) = reference.measure_round();
                    reference.reduce_claim_with_coefficients(c0, c_inf, gamma);
                    reference.hold(gamma);
                    reference.settle();
                }
                reference.scale_weights_and_claim(handoff.eps);

                let shape = format!("{order:?}, {n_vars} variables, {folding_factor} rounds");

                // The claim chains every round message, so a drift in any of them shows here.
                assert_eq!(
                    handoff.residual_prover.claimed_sum(),
                    reference.claimed_sum(),
                    "{shape}"
                );

                // Both bound tables, entry for entry.
                assert_eq!(
                    handoff.residual_prover.evals().as_slice(),
                    reference.evals().as_slice(),
                    "{shape}"
                );
                assert_eq!(
                    handoff.residual_prover.weights().as_slice(),
                    reference.weights().as_slice(),
                    "{shape}"
                );
            }
        }
    }

    #[test]
    fn residual_zk_handoff_samples_masks_through_encoding() {
        let evals = Poly::new((1..=8).map(EF::from_u64).collect::<Vec<_>>());
        let weights = Poly::new((11..=18).map(EF::from_u64).collect::<Vec<_>>());
        let claimed_sum = dot_product::<EF, _, _>(
            evals.as_slice().iter().copied(),
            weights.as_slice().iter().copied(),
        );
        let poly = ProductPolynomial::<F, EF>::new_unpacked(VariableOrder::Prefix, evals, weights);
        let prover = SumcheckProver::new(poly, claimed_sum);

        let ell_zk = 4;
        let folding_factor = 2;
        let (perm, mmcs, _) = make_setup(23, ell_zk);
        let encoding = SentinelEncoding { ell_zk };
        let mut challenger = MyChallenger::new(perm);
        let mut rng = SmallRng::seed_from_u64(29);
        let mut zk_data = ZkSumcheckData::<F, EF>::default();

        let handoff = prover.into_zk_sumcheck(
            &mut zk_data,
            &encoding,
            &mmcs,
            folding_factor,
            0,
            EF::ZERO,
            &mut challenger,
            &mut rng,
        );

        let sentinel = encoding.sample_message(&mut rng);
        assert_eq!(handoff.mask_messages, vec![sentinel; folding_factor]);
    }
}
