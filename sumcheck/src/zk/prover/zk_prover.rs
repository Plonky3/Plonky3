//! Honest-verifier zero-knowledge sumcheck prover, generic over the binding mode.

use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_field::{ExtensionField, Field, HornerIter, TwoAdicField, dot_product};
use p3_matrix::Matrix;
use p3_multilinear_util::point::Point;
use p3_zk_codes::{ZkEncoding, ZkEncodingWithRandomness};
use rand::Rng;

use super::common::{observe_masks_and_mu_tilde, sample_masks};
use super::layout::ZkLayout;
use super::round::{PlainPiece, RoundContext, RoundState, round_poly_to_wire};
use crate::extrapolate_01inf;
use crate::lagrange::lagrange_weights_01inf_multi;
use crate::layout::{PrefixProver, SuffixProver};
use crate::strategy::SumcheckProver;
use crate::svo::calculate_accumulators_batch;
use crate::table::{OpeningEvals, OpeningRequest};
use crate::zk::data::{ZkSumcheckData, ZkSumcheckHandoff};

/// Honest-verifier zero-knowledge sumcheck prover.
///
/// Plain accumulator math is unchanged.
/// The overlay adds mask sampling, mask commits, and the round formula below.
///
/// # Per-round polynomial
///
/// At round `j` with past challenges `gamma_1, ..., gamma_{j-1}`:
///
/// ```text
///     h_j(X) = 2^{k-j}   * s_j(X)                        (live mask)
///            + 2^{k-j}   * sum_{l<j} s_l(gamma_l)        (past masks)
///            + 2^{k-j-1} * sum_{l>j} ( s_l(0) + s_l(1) ) (future ends)
///            + eps       * plain_j(X)                    (plain piece)
/// ```
///
/// Combined degree is `max(ell_zk - 1, 2)`.
/// The mask piece is degree `ell_zk - 1`, the plain piece is degree `2`.
///
/// # `mu_tilde` closed form
///
/// ```text
///     mu_tilde = sum_{b in {0,1}^k} ( s_1(b_1) + ... + s_k(b_k) )
///              = 2^{k-1} * sum_l ( s_l(0) + s_l(1) )
/// ```
///
/// # Residual handoff
///
/// After `k` rounds the residual claim is scaled by `eps`.
/// The factor is folded into the residual polynomial during compression, so downstream consumers see the scaling automatically.
///
/// # Binding mode
///
/// The generic layout type selects the binding direction.
/// Two ready-made aliases cover the two supported modes; see [`ZkPrefixProver`] and [`ZkSuffixProver`].
pub struct ZkProver<F, EF, Enc, M, L>
where
    F: Field,
    EF: ExtensionField<F>,
    Enc: ZkEncoding<EF>,
    M: Mmcs<EF>,
{
    /// Plain stacked-layout prover.
    inner: L,

    /// Zero-knowledge code used to encode the mask polynomials.
    encoding: Enc,

    /// Merkle commitment scheme used to commit each encoded mask.
    mmcs: M,

    /// Marker tying `F`/`EF` to the storage type without inflating the runtime layout.
    _marker: PhantomData<(F, EF)>,
}

/// HVZK prover for the prefix-binding sumcheck.
pub type ZkPrefixProver<F, EF, Enc, M> = ZkProver<F, EF, Enc, M, PrefixProver<F, EF>>;

/// HVZK prover for the suffix-binding sumcheck.
pub type ZkSuffixProver<F, EF, Enc, M> = ZkProver<F, EF, Enc, M, SuffixProver<F, EF>>;

impl<F, EF, Enc, M, L> ZkProver<F, EF, Enc, M, L>
where
    F: TwoAdicField,
    EF: ExtensionField<F>,
    Enc: ZkEncodingWithRandomness<EF>,
    M: Mmcs<EF>,
    L: ZkLayout<F, EF>,
{
    /// Wraps a plain layout with the HVZK ingredients.
    pub const fn new(inner: L, encoding: Enc, mmcs: M) -> Self {
        Self {
            inner,
            encoding,
            mmcs,
            _marker: PhantomData,
        }
    }

    /// Returns the folding factor of the wrapped inner prover.
    pub fn folding(&self) -> usize {
        self.inner.folding()
    }

    /// Returns the variable count of the stacked polynomial.
    pub fn num_variables(&self) -> usize {
        self.inner.num_variables()
    }

    /// Records opening claims at the current points and at their repeat-last successor points on the inner prover.
    ///
    /// # Returns
    ///
    /// The current-point evaluations first, then the successor-point evaluations.
    pub fn eval<Ch>(
        &mut self,
        table_idx: usize,
        batch: &OpeningRequest,
        challenger: &mut Ch,
    ) -> OpeningEvals<EF>
    where
        Ch: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        // Delegate; the HVZK overlay carries no extra state at claim time.
        self.inner.eval(table_idx, batch, challenger)
    }

    /// Records a virtual opening claim on the inner prover.
    pub fn add_virtual_eval<Ch>(&mut self, challenger: &mut Ch) -> EF
    where
        Ch: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        // Same delegation pattern as concrete openings.
        self.inner.add_virtual_eval(challenger)
    }

    /// Runs the HVZK sumcheck and returns the residual claim plus the mask oracles.
    ///
    /// # Phases
    ///
    /// 1. Plain-piece preamble (alpha, accumulators, plain_sum).
    /// 2. Sample, encode, commit, observe masks (Construction 6.3 step 1).
    /// 3. Compute and observe `mu_tilde` (step 2).
    /// 4. Sample the combining challenge `eps` (step 3).
    /// 5. Per-round sumcheck (step 4).
    /// 6. Residual handoff scaled by `eps` (mode-specific compression).
    ///
    /// # Returns
    ///
    /// - Residual sumcheck prover over the mode-specific product polynomial, scaled by `eps`.
    /// - Vector of per-round challenges `gamma_1, ..., gamma_k`.
    /// - Combining challenge `eps`, made explicit for code-switch composition.
    /// - Plain mask messages and one mask oracle per round, in round order.
    ///
    /// # Panics
    ///
    /// - Base field characteristic is `2` (violates Lemma 6.4).
    /// - Mask code message length is below `3` (mask must cover the degree-2 plain piece).
    /// - Folding factor is `0` or exceeds the polynomial's arity.
    #[allow(clippy::too_many_lines)]
    #[tracing::instrument(skip_all)]
    pub fn into_sumcheck<R, Ch>(
        self,
        zk_data: &mut ZkSumcheckData<F, EF>,
        pow_bits: usize,
        challenger: &mut Ch,
        rng: &mut R,
    ) -> ZkSumcheckHandoff<F, EF, M>
    where
        EF: TwoAdicField,
        Enc::Codeword: Matrix<EF>,
        R: Rng,
        Ch: FieldChallenger<F> + GrindingChallenger<Witness = F> + CanObserve<M::Commitment>,
    {
        // Protocol shape resolved from the inner prover + mask encoding.
        let k = self.inner.folding();
        let ell_zk = self.encoding.message_len();
        let n_vars = self.inner.num_variables();

        // Lemma 6.4 hypotheses + sanity bounds on the folding factor.
        assert!(F::TWO != F::ZERO, "Lemma 6.4 requires char(F) != 2");
        assert!(
            ell_zk >= 3,
            "mask degree ell_zk - 1 must cover the degree-2 plain piece (ell_zk >= 3)",
        );
        assert!(k >= 1, "sumcheck requires at least one round");
        assert!(
            k <= n_vars,
            "folding_factor must be <= poly.num_variables()",
        );

        // Phase 1: plain-piece preamble (setup, precedes Construction 6.3).

        // `alpha` is the per-claim batching base: powers a^0, a^1, ... weight the claim accumulators below.
        let alpha: EF = challenger.sample_algebra_element();

        // Materialise every alpha power in one batched pass.
        //
        // Layout:
        //
        //     [ a^0, ..., a^{n_concrete - 1} | a^{n_concrete}, ..., a^{N - 1} ]
        //      \____ concrete-claim block __/  \___ virtual-claim block ___/
        let n_concrete: usize = self.inner.concrete_claims().map(|claim| claim.len()).sum();
        let n_virtual = self.inner.virtual_claims().len();
        let all_alphas: Vec<EF> = alpha.powers().collect_n(n_concrete + n_virtual);
        let (concrete_alphas, virtual_alphas) = all_alphas.split_at(n_concrete);

        // One accumulator per concrete opening, sliced into its alpha block.
        let mut offset = 0;
        let accumulators: Vec<_> = self
            .inner
            .concrete_claims()
            .map(|claim| {
                let slice = &concrete_alphas[offset..offset + claim.len()];
                offset += claim.len();
                calculate_accumulators_batch(claim, slice)
            })
            .collect();

        // Plain sumcheck claim `mu`, batched by the alphas.
        let mut plain_sum = self.inner.batched_sum(alpha);

        // Phase 2: sample, encode, commit, observe masks (Construction 6.3 step 1).
        //
        // The encoder draws zero-knowledge padding randomness from the same rng.
        let (masks, mask_randomness, mask_oracle) =
            sample_masks::<EF, _, _, _, _>(k, &self.encoding, &self.mmcs, challenger, rng);

        // Phase 3: mu_tilde via the closed form (Construction 6.3 step 2).
        //
        // The helper also seeds `zk_data` and returns the running future-mask
        // endpoint budget for the per-round loop.
        let sum_endpoints_init =
            observe_masks_and_mu_tilde::<F, EF, _>(&masks, k, ell_zk, challenger, zk_data);

        // Phase 4: combining challenge `eps` (Construction 6.3 step 3).
        //
        // The construction is instantiated over `EF`: the masks, `eps`, and the
        // round polynomials all live in `EF`, so Lemma 6.4 applies with `F := EF`
        // and the per-round polynomial is uniform over the full extension field.
        let eps: EF = challenger.sample_algebra_element();

        // Phase 5: per-round sumcheck (Construction 6.3 step 4).

        // Per-round challenges.
        let mut rs: Vec<EF> = Vec::with_capacity(k);

        // Cache of `s_j(gamma_j)` values used as the past-mask term in later rounds.
        let mut mask_evals_at_gamma: Vec<EF> = Vec::with_capacity(k);

        // Running `sum_{l >= j} ( s_l(0) + s_l(1) )`.
        // The first round decrement drops `s_1`, leaving `sum_{l > 1}`.
        let mut sum_future_endpoints = sum_endpoints_init;

        // Powers-of-two table for the per-round multipliers:
        //
        //     mult_live   = pow2[k - j]
        //     mult_past   = pow2[k - j + 1]
        //     mult_future = pow2[k - j - 1]
        let pow2: Vec<EF> = EF::TWO.powers().collect_n(k + 1);

        // Round-invariant context shared by every per-round assembly call.
        let round_ctx = RoundContext {
            k,
            ell_zk,
            pow2: &pow2,
            eps,
        };

        for round_idx in 0..k {
            // 1-indexed round used by the formulas.
            let j = round_idx + 1;
            let s_j = &masks[round_idx];

            // Update the running future-endpoint sum: drop s_j's contribution so the round-j formula reads only sum_{l > j}.
            let s_j_endpoints = s_j[0].double() + s_j[1..].iter().copied().sum::<EF>();
            sum_future_endpoints -= s_j_endpoints;

            // Lagrange weights at `(gamma_1, ..., gamma_{j-1})`, used by every accumulator dot product below.
            let weights_lag = lagrange_weights_01inf_multi(&rs);

            // Plain `(c_0, c_inf)`: same formula the plain inner prover computes, summed across every recorded claim.
            let dot = |row: &[EF]| {
                dot_product::<EF, _, _>(row.iter().copied(), weights_lag.iter().copied())
            };

            // Concrete-claim branch.
            let mut plain_c0: EF = accumulators.iter().map(|a| dot(&a[round_idx][0])).sum();
            let mut plain_c_inf: EF = accumulators.iter().map(|a| dot(&a[round_idx][1])).sum();

            // Virtual-claim branch.
            for (vc, alpha_i) in self
                .inner
                .virtual_claims()
                .iter()
                .zip(virtual_alphas.iter().copied())
            {
                plain_c0 += alpha_i * dot(&vc.data[round_idx][0]);
                plain_c_inf += alpha_i * dot(&vc.data[round_idx][1]);
            }

            // Assemble h_j; see the round module for the formula and the
            // in-place affine-consistency cross-check.
            //
            // The linear coefficient is not passed: it is dropped from the wire
            // and reconstructed by the verifier from the affine identity.
            let h = round_ctx.assemble(
                RoundState {
                    j,
                    mask: s_j,
                    past_mask_evals: &mask_evals_at_gamma,
                    future_endpoints: sum_future_endpoints,
                },
                PlainPiece {
                    c0: plain_c0,
                    c_inf: plain_c_inf,
                },
            );

            // Wire format: drop the linear coefficient — the verifier
            // reconstructs it from the affine identity.
            let wire = round_poly_to_wire(&h);

            // Absorb the wire on the transcript and stash it.
            challenger.observe_algebra_slice(&wire);
            zk_data.round_coefficients.push(wire);

            // Optional grind before the per-round challenge.
            if pow_bits > 0 {
                zk_data.pow_witnesses.push(challenger.grind(pow_bits));
            }

            // Sample the per-round challenge gamma_j.
            let gamma_j: EF = challenger.sample_algebra_element();

            // Cache s_j(gamma_j) via Horner for the past-mask term in future rounds.
            let s_j_at_gamma_j: EF = s_j.iter().copied().horner(gamma_j);
            mask_evals_at_gamma.push(s_j_at_gamma_j);

            // Advance the plain sumcheck claim via quadratic extrapolation through (0, 1, inf).
            plain_sum = extrapolate_01inf(plain_c0, plain_sum - plain_c0, plain_c_inf, gamma_j);
            rs.push(gamma_j);
        }

        // Phase 6: residual handoff scaled by eps.
        //
        // The mode-specific compression lives behind the layout trait.
        // The shared invariant is `prod_poly.dot_product() == eps * plain_residual_sum`.
        let rs = Point::new(rs);
        let prod_poly = self.inner.zk_residual_handoff(&rs, alpha, eps);

        let residual_sum = eps * plain_sum;
        debug_assert_eq!(
            prod_poly.dot_product(),
            residual_sum,
            "residual product polynomial dot product must equal eps * plain_residual_sum",
        );

        ZkSumcheckHandoff {
            residual_prover: SumcheckProver::new(prod_poly, residual_sum),
            randomness: rs,
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

    use p3_field::{Field, PackedValue};
    use p3_util::log2_strict_usize;
    use proptest::prelude::*;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use crate::layout::PrefixProver;
    use crate::strategy::VariableOrder;
    use crate::table::OpeningBatch;
    use crate::zk::ZkSumcheckData;
    use crate::zk::test_helpers::{
        EF, F, MyChallenger, MyMmcs, build_prover_verifier, make_setup, run_roundtrip,
    };

    #[test]
    fn prover_verifier_roundtrip_prefix() {
        // Invariant: honest prover-verifier roundtrip on the prefix path with both claim kinds present accepts.
        //
        // Fixture state:
        //
        //     n_vars       = 8       (witness has 2^8 = 256 evaluations)
        //     folding      = 3       (three sumcheck rounds)
        //     ell_zk       = 4       (mask polynomial degree 3)
        //     num_concrete = 1       (drives the inner accumulator branch)
        //     num_virtual  = 1       (drives the virtual accumulator branch)
        //
        // Both claim kinds force the alpha-power split to do real work in both branches.
        // This catches a wrong skip count in the per-round accumulator assembly.
        run_roundtrip(VariableOrder::Prefix, 8, 3, 4, 1, 1, 0)
            .expect("honest roundtrip should accept");
    }

    #[test]
    fn prover_verifier_roundtrip_suffix() {
        // Invariant: honest prover-verifier roundtrip on the suffix path with both claim kinds present accepts.
        //
        // Fixture mirrors the prefix-mode pin so a regression in one mode surfaces with the same fingerprint as the other.
        // The residual handoff routes the combining challenge through suffix-specific compression that the prefix overlay does not exercise.
        run_roundtrip(VariableOrder::Suffix, 8, 3, 4, 1, 1, 0)
            .expect("honest roundtrip should accept");
    }

    #[test]
    fn long_mask_horner_path_prefix() {
        // Invariant: per-round Horner handles long masks identically to short ones on the prefix path.
        //
        // Fixture state: ell_zk = 32, matching a soundness-realistic mask setup.
        // The proptests below cap ell_zk at 5 to keep runtime small.
        // This case pins the long-mask arithmetic path the proptests never reach.
        run_roundtrip(VariableOrder::Prefix, 8, 3, 32, 1, 1, 0)
            .expect("honest roundtrip should accept");
    }

    #[test]
    fn long_mask_horner_path_suffix() {
        // Invariant: per-round Horner handles long masks identically to short ones on the suffix path.
        //
        // Mirrors the prefix long-mask pin.
        // A regression that mishandles the combining factor on a long mask passes the prefix pin and trips here.
        run_roundtrip(VariableOrder::Suffix, 8, 3, 32, 1, 1, 0)
            .expect("honest roundtrip should accept");
    }

    #[test]
    fn prover_verifier_roundtrip_mixed_current_next_batch() {
        // Invariant: the zero-knowledge prefix wrapper accepts a batch that mixes
        // a current opening point with its repeat-last successor point.
        // Acceptance means the prover and verifier draw identical randomness.

        // Fixture state: 8 variables, fold 3 at a time, mask width 4, no grinding.
        let n_vars = 8;
        let folding_factor = 3;
        let ell_zk = 4;
        let seed = 7u64;
        let pow_bits = 0;

        // Shared permutation, Merkle scheme, and encoding for both parties.
        let (perm, mmcs, encoding) = make_setup(seed, ell_zk);
        // Draw a random multilinear over the boolean cube of 2^8 = 256 points.
        let mut data_rng = SmallRng::seed_from_u64(seed.wrapping_add(1));
        let evals: Vec<F> = (0..(1usize << n_vars)).map(|_| data_rng.random()).collect();
        let (mut prover, mut verifier, _) =
            build_prover_verifier::<PrefixProver<F, EF>>(evals, folding_factor, encoding, mmcs);

        // Two challengers seeded identically so the transcripts stay in lockstep.
        let mut prover_challenger = MyChallenger::new(perm.clone());
        let mut verifier_challenger = MyChallenger::new(perm);

        // Open column 0 at one current point and at its successor point.
        let batch = OpeningBatch::new(vec![0], vec![0]);
        // Prover records the claims and returns one evaluation per requested point.
        let evals = prover.eval(0, &batch, &mut prover_challenger);
        assert_eq!(evals.len(), batch.len());
        // Verifier mirrors the same draws against the returned evaluations.
        verifier
            .add_claim(0, &batch, &evals, &mut verifier_challenger)
            .unwrap();

        // Add the masking claim that hides the witness in the final sumcheck.
        let virtual_eval = prover.add_virtual_eval(&mut prover_challenger);
        verifier.add_virtual_eval(virtual_eval, &mut verifier_challenger);

        // Prover commits to the mask and hands off the sumcheck transcript.
        let mut zk_data = ZkSumcheckData::<F, EF>::default();
        let mut prover_rng = SmallRng::seed_from_u64(seed.wrapping_add(2));
        let prover_handoff = prover.into_sumcheck(
            &mut zk_data,
            pow_bits,
            &mut prover_challenger,
            &mut prover_rng,
        );
        // The mask commitment is the only extra public input the verifier needs.
        let mask_commitment = prover_handoff.mask_oracle.0.clone();

        // Verifier replays the sumcheck and must accept the mixed batch.
        let verifier_handoff = verifier
            .into_sumcheck::<MyMmcs, _>(
                &zk_data,
                &mask_commitment,
                ell_zk,
                folding_factor,
                pow_bits,
                &mut verifier_challenger,
            )
            .expect("verifier should accept mixed current/Next batch");

        // Both parties must have folded along the same challenge sequence.
        assert_eq!(
            prover_handoff
                .randomness
                .iter()
                .copied()
                .collect::<Vec<_>>(),
            verifier_handoff
                .randomness
                .iter()
                .copied()
                .collect::<Vec<_>>(),
        );
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(16))]

        #[test]
        fn prop_completeness_prefix(
            n_vars in 3usize..=8,
            ell_zk in 3usize..=5,
            num_concrete in 0usize..=2,
            num_virtual in 0usize..=2,
            seed in 0u64..1024,
        ) {
            // Invariant: every honest prover output verifies across the parameter cube on the prefix path.
            //
            // Sweeping concrete and virtual independently exercises both accumulator branches and the alpha-power split between them.

            // The protocol needs at least one claim for a meaningful mu.
            prop_assume!(num_concrete + num_virtual >= 1);

            // The residual compression step requires the folded polynomial to retain at least one full packed lane.
            // Packing width varies per ISA (NEON 4, AVX2 8, AVX512 16).
            let k_pack = log2_strict_usize(<F as Field>::Packing::WIDTH);
            prop_assume!(n_vars > k_pack);

            // Pick folding factor inside the legal window.
            let folding_factor = 1 + (seed as usize % (n_vars - k_pack));

            prop_assert!(
                run_roundtrip(VariableOrder::Prefix, n_vars, folding_factor, ell_zk, num_concrete, num_virtual, seed)
                    .is_ok()
            );
        }

        #[test]
        fn prop_completeness_suffix(
            n_vars in 3usize..=8,
            ell_zk in 3usize..=5,
            num_concrete in 0usize..=2,
            num_virtual in 0usize..=2,
            seed in 0u64..1024,
        ) {
            // Invariant: every honest prover output verifies across the parameter cube on the suffix path.

            // The protocol needs at least one claim for a meaningful mu.
            prop_assume!(num_concrete + num_virtual >= 1);

            // Suffix mode has no packed-lane reservation, so folding goes up to n_vars - 1.
            // One variable stays for the residual sumcheck.
            let folding_factor = 1 + (seed as usize % (n_vars - 1));

            prop_assert!(
                run_roundtrip(VariableOrder::Suffix, n_vars, folding_factor, ell_zk, num_concrete, num_virtual, seed)
                    .is_ok()
            );
        }
    }
}
