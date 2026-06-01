//! HVZK prover overlay on top of the prefix-binding sumcheck.

use alloc::vec::Vec;

use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_field::{ExtensionField, Field, HornerIter, TwoAdicField, dot_product};
use p3_matrix::Matrix;
use p3_multilinear_util::point::Point;
use p3_zk_codes::ZkEncoding;
use rand::distr::{Distribution, StandardUniform};
use rand::{Rng, RngExt};

use super::data::{MaskOracle, ZkSumcheckData};
use crate::extrapolate_01inf;
use crate::lagrange::lagrange_weights_01inf_multi;
use crate::layout::{Layout, PrefixProver};
use crate::product_polynomial::ProductPolynomial;
use crate::strategy::{SumcheckProver, VariableOrder};
use crate::svo::calculate_accumulators_batch;

/// HVZK prover for the prefix-binding sumcheck.
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
/// Combined degree is `max(ell_zk - 1, 2)`: mask piece is degree `ell_zk - 1`, plain piece is degree `2`.
///
/// # `mu_tilde` closed form
///
/// ```text
///     mu_tilde = sum_{b in {0,1}^k} ( s_1(b_1) + ... + s_k(b_k) )
///              = 2^{k-1} * sum_l ( s_l(0) + s_l(1) )
/// ```
///
/// For `s(X) = c_0 + c_1 X + ... + c_{ell_zk - 1} X^{ell_zk - 1}`:
///
/// ```text
///     s(0) + s(1) = 2 c_0 + sum_{i>=1} c_i
///                 = mask[0].double() + sum(mask[1..])
/// ```
///
/// # Residual handoff
///
/// After `k` rounds the residual claim is scaled by `eps`:
///
/// ```text
///     residual_sum = eps * plain_residual_sum
/// ```
///
/// The factor is folded into the base polynomial via the compression step, so downstream consumers see the scaling automatically.
pub struct ZkPrefixProver<F, EF, Enc, M>
where
    F: Field,
    EF: ExtensionField<F>,
    Enc: ZkEncoding<F>,
    M: Mmcs<F>,
{
    /// Plain prefix-binding prover that supplies the unmasked per-round
    /// arithmetic and the residual product polynomial.
    inner: PrefixProver<F, EF>,

    /// Zero-knowledge code used to encode the mask polynomials.
    encoding: Enc,

    /// Merkle commitment scheme used to commit each encoded mask.
    mmcs: M,
}

impl<F, EF, Enc, M> ZkPrefixProver<F, EF, Enc, M>
where
    F: Field,
    EF: ExtensionField<F>,
    Enc: ZkEncoding<F>,
    M: Mmcs<F>,
{
    /// Wraps a plain prefix-binding prover with the HVZK ingredients.
    pub const fn new(inner: PrefixProver<F, EF>, encoding: Enc, mmcs: M) -> Self {
        Self {
            inner,
            encoding,
            mmcs,
        }
    }

    /// Returns the folding factor of the wrapped inner prover.
    pub const fn folding(&self) -> usize {
        self.inner.folding
    }

    /// Returns the variable count of the stacked polynomial.
    pub const fn num_variables(&self) -> usize {
        self.inner.num_variables
    }

    /// Records concrete opening claims on the inner prover.
    pub fn eval<Ch>(&mut self, table_idx: usize, polys: &[usize], challenger: &mut Ch) -> Vec<EF>
    where
        F: TwoAdicField,
        Ch: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        // Delegate; the HVZK overlay carries no extra state at claim time.
        self.inner.eval(table_idx, polys, challenger)
    }

    /// Records a virtual opening claim on the inner prover.
    pub fn add_virtual_eval<Ch>(&mut self, challenger: &mut Ch) -> EF
    where
        F: TwoAdicField,
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
    /// 6. Residual handoff scaled by `eps`.
    ///
    /// # Returns
    ///
    /// - Residual sumcheck prover over the packed product polynomial, scaled by `eps`.
    /// - Vector of per-round challenges `gamma_1, ..., gamma_k`.
    /// - One mask oracle per round, in round order.
    ///
    /// # Panics
    ///
    /// - Base field characteristic is `2` (violates Lemma 6.4).
    /// - Mask code message length is below `2` (Lemma 6.4 floor).
    /// - Folding factor is `0` or exceeds the polynomial's arity.
    #[allow(clippy::too_many_lines, clippy::type_complexity)]
    #[tracing::instrument(skip_all)]
    pub fn into_sumcheck<R, Ch>(
        self,
        zk_data: &mut ZkSumcheckData<F, EF>,
        pow_bits: usize,
        challenger: &mut Ch,
        rng: &mut R,
    ) -> (SumcheckProver<F, EF>, Point<EF>, Vec<MaskOracle<F, Enc, M>>)
    where
        F: TwoAdicField,
        EF: ExtensionField<F> + TwoAdicField,
        Enc::Codeword: Matrix<F>,
        R: Rng,
        StandardUniform: Distribution<F>,
        Ch: FieldChallenger<F> + GrindingChallenger<Witness = F> + CanObserve<M::Commitment>,
    {
        // Protocol shape resolved from the inner prover + mask encoding.
        let k = self.inner.folding;
        let ell_zk = self.encoding.message_len();
        let n_vars = self.inner.num_variables;

        // Lemma 6.4 hypotheses + sanity bounds on the folding factor.
        assert!(F::TWO != F::ZERO, "Lemma 6.4 requires char(F) != 2");
        assert!(ell_zk >= 2, "Lemma 6.4 requires ell_zk >= 2");
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
        let n_concrete: usize = self
            .inner
            .placements
            .iter()
            .flat_map(|placement| self.inner.claim_map[placement.idx()].iter())
            .map(|claim| claim.len())
            .sum();
        let n_virtual = self.inner.virtual_claims.len();
        let all_alphas: Vec<EF> = alpha.powers().collect_n(n_concrete + n_virtual);
        let (concrete_alphas, virtual_alphas) = all_alphas.split_at(n_concrete);

        // One accumulator per concrete opening, sliced into its alpha block.
        let mut offset = 0;
        let accumulators: Vec<_> = self
            .inner
            .placements
            .iter()
            .flat_map(|placement| self.inner.claim_map[placement.idx()].iter())
            .map(|claim| {
                let slice = &concrete_alphas[offset..offset + claim.len()];
                offset += claim.len();
                calculate_accumulators_batch(claim, slice)
            })
            .collect();

        // Plain sumcheck claim `mu`, batched by the alphas.
        let mut plain_sum = self.inner.sum(alpha);

        // Phase 2: sample, encode, commit, observe masks (Construction 6.3 step 1).
        //
        //     s_j(X) = c_0 + c_1 X + ... + c_{ell_zk - 1} X^{ell_zk - 1}
        //
        // The encoder draws zero-knowledge padding randomness, so it needs the mutable rng.
        let masks: Vec<Vec<F>> = (0..k)
            .map(|_| (0..ell_zk).map(|_| rng.random()).collect())
            .collect();
        let mask_oracles: Vec<MaskOracle<F, Enc, M>> = masks
            .iter()
            .map(|mask| {
                let codeword = self.encoding.encode(mask, rng);
                let (commit, prover_data) = self.mmcs.commit_matrix(codeword);
                challenger.observe(commit.clone());
                (commit, prover_data)
            })
            .collect();

        // Phase 3: mu_tilde via the closed form (Construction 6.3 step 2).
        //
        //     s(0) + s(1) = 2 c_0 + sum_{i >= 1} c_i
        //                 = mask[0].double() + sum(mask[1..])
        //
        //     mu_tilde    = 2^{k - 1} * sum_l ( s_l(0) + s_l(1) )
        let sum_endpoints_init: F = masks
            .iter()
            .map(|mask| mask[0].double() + mask[1..].iter().copied().sum::<F>())
            .sum();
        let two_to_k_minus_1 = F::TWO.exp_u64((k - 1) as u64);
        let mu_tilde: F = two_to_k_minus_1 * sum_endpoints_init;

        // Cross-check the closed form against the naive 2^k-term sum.
        #[cfg(debug_assertions)]
        {
            let mut naive = F::ZERO;
            for bits in 0..(1u64 << k) {
                for (l, mask) in masks.iter().enumerate() {
                    let b_l = (bits >> l) & 1;
                    // Separable mask:
                    // - s_l(0) = c_0;
                    // - s_l(1) = c_0 + c_1 + ... + c_{ell_zk - 1}.
                    let s_l_eval = if b_l == 0 {
                        mask[0]
                    } else {
                        mask.iter().copied().sum::<F>()
                    };
                    naive += s_l_eval;
                }
            }
            debug_assert_eq!(
                mu_tilde, naive,
                "mu_tilde closed form does not match naive sum over {{0,1}}^k",
            );
        }

        // Observe mu_tilde (lifted to EF) and stash on the transcript record.
        challenger.observe_algebra_element(EF::from(mu_tilde));
        zk_data.mu_tilde = mu_tilde;
        zk_data.ell_zk = ell_zk;

        // Phase 4: combining challenge `eps` (Construction 6.3 step 3).
        //
        // `eps` lives in EF; the paper samples in F.
        // Masks stay in F to preserve sublinear proof size.
        // The resulting hybrid F/EF `h_j` is handled by the simulator's F-subspace stratification.
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
        let pow2: Vec<F> = F::TWO.powers().collect_n(k + 1);

        for round_idx in 0..k {
            // 1-indexed round used by the formulas.
            let j = round_idx + 1;
            let s_j = &masks[round_idx];

            // Update the running future-endpoint sum: drop s_j's contribution so the round-j formula reads only sum_{l > j}.
            let s_j_endpoints = s_j[0].double() + s_j[1..].iter().copied().sum::<F>();
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
                .virtual_claims
                .iter()
                .zip(virtual_alphas.iter().copied())
            {
                plain_c0 += alpha_i * dot(&vc.data[round_idx][0]);
                plain_c_inf += alpha_i * dot(&vc.data[round_idx][1]);
            }

            // Recover c_1 from the plain affine identity:
            //
            //     plain_h(0) + plain_h(1) = plain_sum
            //     => c_1 = plain_sum - 2 c_0 - c_inf
            let plain_c1 = plain_sum - plain_c0.double() - plain_c_inf;

            // Assemble h_j of length max(ell_zk, 3) (Construction 6.3 step 4, round-j polynomial):
            //
            //     h[0..ell_zk] += 2^{k-j}     * s_j[i]
            //     h[0]         += 2^{k-j}     * sum_{l<j} s_l(gamma_l)
            //     h[0]         += 2^{k-j-1}   * sum_{l>j} ( s_l(0)+s_l(1) )
            //     h[0]         += eps * c_0
            //     h[1]         += eps * c_1
            //     h[2]         += eps * c_inf
            let h_size = ell_zk.max(3);
            let mut h: Vec<EF> = EF::zero_vec(h_size);

            // Live-mask contribution at every slot the mask occupies.
            let mult_live = pow2[k - j];
            for (i, &c) in s_j.iter().enumerate() {
                h[i] += mult_live * c;
            }

            // Past-mask contribution: scalar landing on the constant slot.
            let past_mask_sum: EF = mask_evals_at_gamma.iter().copied().sum();
            h[0] += past_mask_sum * mult_live;

            // Future-mask contribution: zero in the last round, present otherwise.
            if j < k {
                let mult_future = pow2[k - j - 1];
                h[0] += mult_future * sum_future_endpoints;
            }

            // Plain piece, scaled by the combining challenge.
            h[0] += eps * plain_c0;
            h[1] += eps * plain_c1;
            h[2] += eps * plain_c_inf;

            // Affine consistency invariant (Construction 6.3 verifier identity, sanity-checked on the prover side).
            //
            //     h(0) + h(1) = 2 h[0] + sum_{i >= 1} h[i]
            //
            // Per-term contribution to that sum:
            //
            //     live   : 2^{k-j} * ( s_j(0) + s_j(1) )    = 2^{k-j} * s_j_endpoints
            //     past   : 2^{k-j+1} * past_mask_sum        (h[0] only)
            //     future : 2^{k-j} * sum_future             (h[0] only, zero at j=k)
            //     plain  : eps * ( 2 c_0 + c_1 + c_inf )    = eps * plain_sum
            //
            // Anchors:
            //
            //     j = 1 -> mu_tilde + eps * mu     (round-1 target)
            //     j > 1 -> h_{j - 1}(gamma_{j - 1}) (by induction)
            #[cfg(debug_assertions)]
            {
                let mult_past = pow2[k - j + 1];
                let mut expected: EF =
                    eps * plain_sum + past_mask_sum * mult_past + mult_live * s_j_endpoints;
                if j < k {
                    expected += mult_live * sum_future_endpoints;
                }
                debug_assert_eq!(
                    h[0].double() + h[1..].iter().copied().sum::<EF>(),
                    expected,
                    "h_j affine consistency check failed at round {j}",
                );
            }

            // Wire format: send ( c_0, c_2, c_3, ..., c_d ), skipping c_1.
            // The verifier reconstructs c_1 from the affine identity.
            let mut wire: Vec<EF> = Vec::with_capacity(h_size - 1);
            wire.push(h[0]);
            wire.extend_from_slice(&h[2..]);

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

        // Phase 6: residual handoff scaled by eps (post-Construction 6.3 reduction; feeds the next protocol step).

        // Wrap the per-round challenges into a structured point.
        let rs = Point::new(rs);

        // Fold the base polynomial along the sumcheck challenges and absorb eps as the scaling factor:
        //
        //     prod_poly.dot_product() = eps * plain_residual_sum
        let compressed = tracing::info_span!("compress_prefix_to_packed")
            .in_scope(|| self.inner.poly.compress_prefix_to_packed(&rs, eps));

        // Equality weights for the residual product polynomial.
        // No eps scaling here; the factor lives entirely on the base side.
        let weights = self.inner.combine_eqs(&rs, alpha).pack::<F, EF>();
        let prod_poly =
            ProductPolynomial::<F, EF>::new_packed(VariableOrder::Prefix, compressed, weights);

        // Cross-check residual sum against the directly-evaluated dot product.
        // Catches scaling bugs in compress_prefix_to_packed.
        let residual_sum = eps * plain_sum;
        debug_assert_eq!(
            prod_poly.dot_product(),
            residual_sum,
            "residual product polynomial dot product must equal eps * plain_residual_sum",
        );

        (
            SumcheckProver::new(prod_poly, residual_sum),
            rs,
            mask_oracles,
        )
    }
}

#[cfg(test)]
mod tests {
    use p3_field::{Field, PackedValue};
    use p3_util::log2_strict_usize;
    use proptest::prelude::*;

    use crate::zk::test_helpers::{F, run_roundtrip};

    #[test]
    fn prover_verifier_roundtrip_classic_unpacked() {
        // Invariant: honest prover-verifier roundtrip with both claim kinds present accepts and produces matching challenge sequences.
        //
        // Fixture state:
        //
        //     n_vars       = 8       (witness has 2^8 = 256 evaluations)
        //     folding      = 3       (three sumcheck rounds)
        //     ell_zk       = 4       (mask polynomial degree 3)
        //     num_concrete = 1       (drives the inner accumulator branch)
        //     num_virtual  = 1       (drives the virtual accumulator branch)
        //
        // Why both kinds: the alpha-power split must do real work in both branches.
        // This catches a wrong skip count in the per-round accumulator assembly.
        run_roundtrip(8, 3, 4, 1, 1, 0).expect("honest roundtrip should accept");
    }

    #[test]
    fn long_mask_horner_path() {
        // Invariant: per-round Horner handles long masks identically to short ones.
        //
        // Fixture state:
        //
        //     ell_zk = 32 (matches an HVZK-WHIR soundness-realistic setup)
        //
        // The proptest below caps ell_zk at 5 to keep runtime small.
        // This case pins the long-mask arithmetic path the proptest never reaches.
        run_roundtrip(8, 3, 32, 1, 1, 0).expect("honest roundtrip should accept");
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(16))]

        #[test]
        fn prop_completeness_classic_unpacked(
            n_vars in 3usize..=8,
            ell_zk in 2usize..=5,
            num_concrete in 0usize..=2,
            num_virtual in 0usize..=2,
            seed in 0u64..1024,
        ) {
            // Invariant: every honest prover output verifies across the (n_vars, folding, ell_zk, num_concrete, num_virtual) cube.
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
                run_roundtrip(n_vars, folding_factor, ell_zk, num_concrete, num_virtual, seed)
                    .is_ok()
            );
        }
    }
}
