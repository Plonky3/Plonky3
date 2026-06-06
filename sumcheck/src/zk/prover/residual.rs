//! HVZK overlay for an already-derived residual sumcheck claim.

use alloc::vec::Vec;

use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_field::{ExtensionField, Field, HornerIter};
use p3_matrix::Matrix;
use p3_multilinear_util::point::Point;
use p3_zk_codes::ZkEncodingWithRandomness;
use rand::Rng;

use super::common::{observe_masks_and_mu_tilde, sample_masks};
use super::round::{PlainPiece, RoundContext, RoundState, round_poly_to_wire};
use crate::strategy::SumcheckProver;
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
        Enc: ZkEncodingWithRandomness<EF>,
        Enc::Codeword: Matrix<EF>,
        M: Mmcs<EF>,
        R: Rng,
        Ch: FieldChallenger<F> + GrindingChallenger<Witness = F> + CanObserve<M::Commitment>,
    {
        assert!(F::TWO != F::ZERO, "Lemma 6.4 requires char(F) != 2");
        assert!(folding_factor >= 1, "sumcheck requires at least one round");
        assert!(
            folding_factor <= self.num_variables(),
            "folding_factor must be <= residual prover arity",
        );

        let ell_zk = encoding.message_len();
        assert!(
            ell_zk >= 3,
            "mask degree ell_zk - 1 must cover the degree-2 plain piece (ell_zk >= 3)",
        );

        // Unlike the layout-driven path, this entry receives a scalar claim
        // directly, so bind it before the masking prelude samples `eps`.
        //
        // The bound value is the joint claim, matching the verifier's view.
        challenger.observe_algebra_element(self.claimed_sum() + aux_claim);

        let (masks, mask_randomness, mask_oracle) =
            sample_masks::<EF, _, _, _, _>(folding_factor, encoding, mmcs, challenger, rng);
        let mut sum_future_endpoints = observe_masks_and_mu_tilde::<F, EF, _>(
            &masks,
            folding_factor,
            ell_zk,
            challenger,
            zk_data,
        );

        let eps: EF = challenger.sample_algebra_element();
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

            let (plain_c0, plain_c_inf) = self.round_coefficients();
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
            challenger.observe_algebra_slice(&wire);
            zk_data.round_coefficients.push(wire);

            if pow_bits > 0 {
                zk_data.pow_witnesses.push(challenger.grind(pow_bits));
            }

            let gamma: EF = challenger.sample_algebra_element();
            let mask_at_gamma = mask.iter().copied().horner(gamma);
            mask_evals_at_gamma.push(mask_at_gamma);

            self.fold_round_with_coefficients(plain_c0, plain_c_inf, gamma);
            rs.push(gamma);
        }

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
    use alloc::vec;
    use alloc::vec::Vec;

    use p3_baby_bear::BabyBear;
    use p3_field::extension::BinomialExtensionField;
    use p3_field::{PrimeCharacteristicRing, dot_product};
    use p3_matrix::dense::RowMajorMatrix;
    use p3_multilinear_util::poly::Poly;
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
