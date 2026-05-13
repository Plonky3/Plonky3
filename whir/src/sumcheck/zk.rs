//! HVZK variant of the WHIR sumcheck (Construction 6.3, eprint 2026/391).
//!
//! Sits as an overlay on top of [`PrefixProver`]: per round the plain piece
//! `(c_0, c_∞)` is sourced the same way `PrefixProver::into_sumcheck` produces
//! it — Lagrange-weighted dot products against per-claim partial-evaluation
//! accumulators — and the HVZK side wraps that output with mask sampling +
//! commitment, μ̃, ε, and the per-round mask polynomial contributions.
//!
//! # Layout coverage
//!
//! This module only ships the HVZK overlay for the **prefix-binding** layout
//! ([`PrefixProver`]). The suffix-binding layout ([`SuffixProver`]) has no
//! HVZK counterpart yet: driving the WHIR PCS through that layout produces a
//! non-private proof. Construction 6.3 itself is binding-order agnostic, so
//! the missing `ZkSuffixProver` is a symmetric overlay on top of
//! `SuffixProver` rather than a new protocol. Tracked in
//! [Plonky3#1649](https://github.com/Plonky3/Plonky3/issues/1649).
//!
//! [`SuffixProver`]: crate::sumcheck::layout::SuffixProver
//!
//! # Protocol overview
//!
//! 1. **Masks.** Prover samples `s_1, …, s_k ∈ F^{<ℓ_zk}[X]` and commits each
//!    encoded codeword `Enc_{C_zk}(s_j)` under MMCS, observing each commitment
//!    on the transcript.
//! 2. **New target.** Prover sends `μ̃ := Σ_{b ∈ {0,1}^k} (s_1(b_1) + … + s_k(b_k))`.
//! 3. **Combination randomness.** Verifier samples `ε`.
//! 4. **Sumcheck.** For `j = 1, …, k`: prover sends `ĥ_j` (formula below),
//!    verifier samples `γ_j`.
//!
//! Decision-phase verifier checks:
//!
//! - Round 1: `ĥ_1(0) + ĥ_1(1) = ε·μ + μ̃`.
//! - Round `j > 1`: `ĥ_j(0) + ĥ_j(1) = ĥ_{j-1}(γ_{j-1})`.
//!
//! # Per-round polynomial
//!
//! For round `j` with `γ = (γ_1, …, γ_{j-1})` already sampled:
//!
//! ```text
//! ĥ_j(X) = 2^{k-j}   * s_j(X)                          (live mask)
//!        + 2^{k-j}   * Σ_{l < j} s_l(γ_l)              (past masks, cached)
//!        + 2^{k-j-1} * Σ_{l > j} (s_l(0) + s_l(1))     (future-mask endpoints)
//!        + ε         * plain_piece(X)                  (base sumcheck round)
//! ```
//!
//! Combined degree is `max(ℓ_zk - 1, 2)`: the mask piece has degree `ℓ_zk - 1`,
//! the plain piece is degree 2 (multilinear × multilinear).
//!
//! # `μ̃` closed form
//!
//! For separable masks `ŝ(b) := s_1(b_1) + … + s_k(b_k)`, each `b_l ∈ {0,1}`
//! is independent and contributes `s_l(0) + s_l(1)` over `2^{k-1}` of the `2^k`
//! strings, giving
//!
//! ```text
//! μ̃ = Σ_{b ∈ {0,1}^k} ŝ(b) = 2^{k-1} * Σ_l (s_l(0) + s_l(1)).
//! ```
//!
//! For `s(X) = c_0 + c_1·X + … + c_{ℓ_zk-1}·X^{ℓ_zk-1}` we have
//! `s(0) + s(1) = c_0 + Σ c_i = mask[0] + mask.iter().sum()`.
//!
//! # Wire format (skip-linear-coefficient)
//!
//! Per round the prover sends `max(ℓ_zk - 1, 2)` field elements
//! `(c_0, c_2, c_3, …, c_d)` where `d = max(ℓ_zk - 1, 2)`. The linear
//! coefficient `c_1` is omitted: the verifier reconstructs it from the affine
//! check above using `ĥ_j(0) + ĥ_j(1) = 2·c_0 + Σ_{i ≥ 1} c_i`.
//!
//! Lemma 6.4's rank-nullity argument shows the affine subspace of valid
//! transcripts `(μ̃, ĥ_1, …, ĥ_k)` has dimension `1 + k(ℓ_zk - 1)`, so the `k`
//! linear coefficients are exactly the redundant degrees of freedom.
//!
//! # Field constraints (Lemma 6.4)
//!
//! - `char(F) ≠ 2` — required by the rank-nullity argument that drives the
//!   HVZK simulator's affine-subspace surjectivity.
//! - `ℓ_zk ≥ 2` — needed for the mask piece to carry non-trivial information.
//!
//! Both are enforced at constructor entry.
//!
//! # Residual sumcheck handoff
//!
//! After `k` HVZK rounds, [`ZkPrefixProver::into_sumcheck`] returns a residual
//! [`SumcheckProver`] over the partially-folded product polynomial, **scaled
//! by `ε`** so that `prod_poly.dot_product() == ε · plain_residual_sum`. The
//! `ε` factor is absorbed into the folded base polynomial via the `scale`
//! argument of `compress_prefix_to_packed`. Mirrors how
//! [`PrefixProver::into_sumcheck`] hands off its residual.
//!
//! # References
//!
//! - eprint 2026/391, §6 Construction 6.3, Lemma 6.4 (HVZK), Lemma 6.5 (RBR).

use alloc::vec;
use alloc::vec::Vec;

use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_field::{ExtensionField, Field, TwoAdicField, dot_product};
use p3_matrix::Matrix;
use p3_multilinear_util::point::Point;
use p3_zk_codes::ZkEncoding;
use rand::distr::{Distribution, StandardUniform};
use rand::{Rng, RngExt};

use crate::sumcheck::error::SumcheckError;
use crate::sumcheck::extrapolate_01inf;
use crate::sumcheck::lagrange::lagrange_weights_01inf_multi;
use crate::sumcheck::layout::{Layout, LayoutStrategy, PrefixProver, Verifier};
use crate::sumcheck::product_polynomial::ProductPolynomial;
use crate::sumcheck::strategy::{SumcheckProver, VariableOrder};
use crate::sumcheck::svo::calculate_accumulators_batch;
use crate::sumcheck::table::TableShape;

/// Per-round transcript records for the HVZK sumcheck.
///
/// HVZK's per-round polynomial has combined degree `max(ℓ_zk - 1, 2)`, so each
/// round's wire payload carries `max(ℓ_zk - 1, 2)` field elements (after
/// skipping `c_1`). The length is constant within a proof but only known at
/// runtime (derived from `encoding.message_len()`), hence the inner `Vec<EF>`.
#[derive(Debug, Clone)]
pub struct ZkSumcheckData<F, EF> {
    /// `μ̃ = Σ_{b ∈ {0,1}^k} ŝ(b)` — the prover's "new target" sent in step 2
    /// of Construction 6.3. Observed on the transcript before the verifier
    /// samples `ε`; lifted to `EF` at observation time.
    pub mu_tilde: F,
    /// Per-round wire coefficients of `ĥ_j` with the linear term skipped.
    /// Layout per entry: `[c_0, c_2, c_3, …, c_d]` where `d = max(ℓ_zk - 1, 2)`.
    pub round_coefficients: Vec<Vec<EF>>,
    /// Per-round proof-of-work witnesses (one entry per round if `pow_bits > 0`).
    pub pow_witnesses: Vec<F>,
}

impl<F: Field, EF> Default for ZkSumcheckData<F, EF> {
    fn default() -> Self {
        Self {
            mu_tilde: F::ZERO,
            round_coefficients: Vec::new(),
            pow_witnesses: Vec::new(),
        }
    }
}

/// `(MMCS commitment, MMCS prover data)` for one encoded mask codeword.
///
/// Returned by [`ZkPrefixProver::into_sumcheck`] so downstream consumers
/// (committed sumcheck relation, §5 of eprint 2026/391) can produce opening
/// proofs against the mask oracles.
pub type MaskOracle<F, Enc, M> = (
    <M as Mmcs<F>>::Commitment,
    <M as Mmcs<F>>::ProverData<<Enc as ZkEncoding<F>>::Codeword>,
);

/// HVZK overlay over [`PrefixProver`]: same plain-piece arithmetic, plus mask
/// sampling/commitment and the per-round `ĥ_j` formula from Construction 6.3.
///
/// # Composition
///
/// - `inner` carries the same state PrefixProver does: tables, placements,
///   stacked polynomial, recorded claims.
/// - `encoding` and `mmcs` are HVZK-specific; they are consumed when
///   `into_sumcheck` samples and commits the masks.
///
/// # Lifecycle
///
/// 1. Build a [`crate::sumcheck::layout::Witness`] from source tables.
/// 2. Wrap into a [`PrefixProver`] via `PrefixProver::from_witness`.
/// 3. Wrap into [`ZkPrefixProver`] via [`ZkPrefixProver::new`].
/// 4. Register opening claims via [`Self::eval`] / [`Self::add_virtual_eval`]
///    (both delegate to the inner `PrefixProver`).
/// 5. Drive the HVZK sumcheck via [`Self::into_sumcheck`].
pub struct ZkPrefixProver<F, EF, Enc, M>
where
    F: Field,
    EF: ExtensionField<F>,
    Enc: ZkEncoding<F>,
    M: Mmcs<F>,
{
    inner: PrefixProver<F, EF>,
    encoding: Enc,
    mmcs: M,
}

impl<F, EF, Enc, M> ZkPrefixProver<F, EF, Enc, M>
where
    F: Field,
    EF: ExtensionField<F>,
    Enc: ZkEncoding<F>,
    M: Mmcs<F>,
{
    /// Wraps a `PrefixProver` with the HVZK trio (encoding + MMCS).
    ///
    /// The wrapped prover retains all of `PrefixProver`'s claim-recording API
    /// via the delegating methods below.
    pub const fn new(inner: PrefixProver<F, EF>, encoding: Enc, mmcs: M) -> Self {
        Self {
            inner,
            encoding,
            mmcs,
        }
    }

    /// Returns the folding factor of the wrapped `PrefixProver`.
    pub const fn folding(&self) -> usize {
        self.inner.folding
    }

    /// Returns the number of variables of the stacked polynomial.
    pub const fn num_variables(&self) -> usize {
        self.inner.num_variables
    }

    /// Records concrete opening claims; delegates to `PrefixProver::eval`.
    pub fn eval<Ch>(&mut self, table_idx: usize, polys: &[usize], challenger: &mut Ch) -> Vec<EF>
    where
        F: TwoAdicField,
        Ch: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        self.inner.eval(table_idx, polys, challenger)
    }

    /// Records a virtual opening claim; delegates to `PrefixProver::add_virtual_eval`.
    pub fn add_virtual_eval<Ch>(&mut self, challenger: &mut Ch) -> EF
    where
        F: TwoAdicField,
        Ch: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        self.inner.add_virtual_eval(challenger)
    }

    /// Runs the HVZK sumcheck: Construction 6.3 steps 1–3 (mask commit, μ̃, ε)
    /// followed by `folding` rounds of step 4. Returns a residual sumcheck
    /// prover scaled by `ε` (see module docs).
    ///
    /// # Algorithm
    ///
    /// 1. Sample `α` and build the per-claim partial-evaluation accumulators
    ///    (same as `PrefixProver::into_sumcheck`).
    /// 2. Sample masks `s_1, …, s_k`, encode + MMCS-commit + observe each.
    /// 3. Compute and observe `μ̃ = 2^{k-1} · Σ_l (s_l(0) + s_l(1))`.
    /// 4. Sample `ε`.
    /// 5. For `round_idx = 0..folding`:
    ///    - Compute `(plain_c_0, plain_c_∞)` from accumulators via Lagrange
    ///      weights, summed across concrete + virtual claims.
    ///    - Build `ĥ_j` (mask + past + future + ε·plain) and emit
    ///      `[c_0, c_2, …, c_d]` on the wire.
    ///    - Grind PoW, sample `γ_j`, cache `s_j(γ_j)`, update running plain
    ///      sum + future-endpoint state.
    /// 6. Build the residual `ProductPolynomial`: base polynomial folded at
    ///    all `γ` and scaled by `ε`, times the residual equality-weight
    ///    polynomial. Residual sum equals `ε · plain_residual_sum`.
    ///
    /// # Returns
    ///
    /// - Residual sumcheck prover over the packed product polynomial.
    /// - Folding challenges `(γ_1, …, γ_k)`.
    /// - Mask oracles (commitment + prover data per mask), in mask order.
    ///
    /// # Panics
    ///
    /// - If `char(F) == 2` (Lemma 6.4).
    /// - If `encoding.message_len() < 2` (Lemma 6.4).
    /// - If `self.folding() == 0` or `> self.num_variables()`.
    #[allow(
        clippy::too_many_arguments,
        clippy::too_many_lines,
        clippy::type_complexity
    )]
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
        let k = self.inner.folding;
        let ell_zk = self.encoding.message_len();
        let n_vars = self.inner.num_variables;

        const { assert!(
            F::TWO != F::ZERO,
            "Construction 6.3 (Lemma 6.4) requires char(F) != 2",
        ); }
        assert!(
            ell_zk >= 2,
            "Construction 6.3 (Lemma 6.4) requires ell_zk >= 2",
        );
        assert!(k >= 1, "sumcheck requires at least one round");
        assert!(
            k <= n_vars,
            "folding_factor must be <= poly.num_variables()",
        );

        let Self {
            inner,
            encoding,
            mmcs,
        } = self;

        // --- Plain-piece preamble (mirrors PrefixProver::into_sumcheck) ---
        let alpha: EF = challenger.sample_algebra_element();
        let n_claims = inner.num_claims()
        let mut alphas = alpha.powers();
        let accumulators: Vec<_> = inner
            .placements
            .iter()
            .flat_map(|placement| inner.claim_map[placement.idx()].iter())
            .map(|claim| {
                let per_claim: Vec<EF> = alphas.by_ref().take(claim.len()).collect();
                calculate_accumulators_batch(claim, &per_claim)
            })
            .collect();
        let mut plain_sum = inner.sum(alpha);

        // --- Construction 6.3 step 1: sample, encode, commit, observe ---
        let masks: Vec<Vec<F>> = (0..k)
            .map(|_| (0..ell_zk).map(|_| rng.random()).collect())
            .collect();
        let mask_oracles: Vec<MaskOracle<F, Enc, M>> = masks
            .iter()
            .map(|mask| {
                let codeword = encoding.encode(mask, rng);
                let (commit, prover_data) = mmcs.commit_matrix(codeword);
                challenger.observe(commit.clone());
                (commit, prover_data)
            })
            .collect();

        // --- Construction 6.3 step 2: μ̃ ---
        let sum_endpoints_init: F = masks
            .iter()
            .map(|mask| mask[0] + mask.iter().copied().sum::<F>())
            .sum();
        let two_to_k_minus_1 = F::TWO.exp_u64((k - 1) as u64);
        let mu_tilde: F = two_to_k_minus_1 * sum_endpoints_init;

        #[cfg(debug_assertions)]
        {
            let mut naive = F::ZERO;
            for bits in 0..(1u64 << k) {
                for (l, mask) in masks.iter().enumerate() {
                    let b_l = (bits >> l) & 1;
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
                "μ̃ closed form does not match naive Σ_{{b ∈ {{0,1}}^k}} ŝ(b)",
            );
        }

        challenger.observe_algebra_element(EF::from(mu_tilde));
        zk_data.mu_tilde = mu_tilde;

        // --- Construction 6.3 step 3: ε ---
        // Note: this is the load-bearing differnece from the reference
        // Plonky3 samples ε ∈ EF (not ∈ F as in paper Construction 6.3 step 3) for
        // the soundness margin. Masks remain in F to keep the proof size sublinear,
        // so the per-round h_j(X) is a hybrid F/EF object — see the F-subspace
        // stratification in simulate_classic_unpacked for the simulator-side
        // implication of this divergence.
        let eps: EF = challenger.sample_algebra_element();

        // --- Per-round loop (Construction 6.3 step 4) ---
        let mut rs: Vec<EF> = Vec::with_capacity(k);
        let mut mask_evals_at_gamma: Vec<EF> = Vec::with_capacity(k);
        // Running future-endpoint state: at the start of round `j` (1-indexed)
        // this holds `Σ_{l ≥ j}(s_l(0) + s_l(1))`. Initialised to the full sum
        // so round 1's start-of-round decrement leaves `Σ_{l ≥ 2}`.
        let mut sum_future_endpoints = sum_endpoints_init;

        for round_idx in 0..k {
            let j = round_idx + 1;
            let s_j = &masks[round_idx];

            // Start-of-round decrement: drop `s_j`'s endpoints so the running
            // sum is `Σ_{l > j}(s_l(0)+s_l(1))`, the value the per-round
            // formula uses for the future-mask term at this `j`.
            let s_j_endpoints = s_j[0] + s_j.iter().copied().sum::<F>();
            sum_future_endpoints -= s_j_endpoints;

            // Plain (c_0, c_∞) from the per-claim partial-evaluation
            // accumulators, summed across concrete + virtual claims and
            // Lagrange-weighted by the challenges sampled so far. This is
            // exactly what `PrefixProver::into_sumcheck` does, just lifted
            // out so we can mix in the HVZK overlay.
            let weights_lag = lagrange_weights_01inf_multi(&rs);
            let mut plain_c0 = EF::ZERO;
            let mut plain_c_inf = EF::ZERO;
            for accs in &accumulators {
                plain_c0 += dot_product::<EF, _, _>(
                    accs[round_idx][0].iter().copied(),
                    weights_lag.iter().copied(),
                );
                plain_c_inf += dot_product::<EF, _, _>(
                    accs[round_idx][1].iter().copied(),
                    weights_lag.iter().copied(),
                );
            }
            for (vc, alpha_i) in inner
                .virtual_claims
                .iter()
                .zip(alpha.powers().skip(n_claims))
            {
                let vc_accs = &vc.data;
                plain_c0 += alpha_i
                    * dot_product::<EF, _, _>(
                        vc_accs[round_idx][0].iter().copied(),
                        weights_lag.iter().copied(),
                    );
                plain_c_inf += alpha_i
                    * dot_product::<EF, _, _>(
                        vc_accs[round_idx][1].iter().copied(),
                        weights_lag.iter().copied(),
                    );
            }
            // Recover c_1 of the plain piece from the affine constraint
            //   plain_h(0) + plain_h(1) = plain_sum
            // ⇒ c_1 = plain_sum - 2·c_0 - c_∞.
            let plain_c1 = plain_sum - plain_c0.double() - plain_c_inf;

            // Build `ĥ_j` of length `max(ell_zk, 3)`:
            //   indices 0..ell_zk : live-mask piece           = 2^{k-j} · s_j(X)
            //   index 0           : past-mask contribution   += 2^{k-j} · Σ_{l<j} s_l(γ_l)
            //   index 0           : future-mask contribution += 2^{k-j-1} · Σ_{l>j}(s_l(0)+s_l(1))
            //   indices 0..3      : plain piece              += ε · (c_0 + c_1·X + c_∞·X²)
            let h_size = core::cmp::max(ell_zk, 3);
            let mut h: Vec<EF> = vec![EF::ZERO; h_size];

            let mult_live = F::TWO.exp_u64((k - j) as u64);
            for (i, &c) in s_j.iter().enumerate() {
                h[i] += EF::from(mult_live * c);
            }
            let past_mask_sum: EF = mask_evals_at_gamma.iter().copied().sum();
            h[0] += EF::from(mult_live) * past_mask_sum;
            if j < k {
                let mult_future = F::TWO.exp_u64((k - j - 1) as u64);
                h[0] += EF::from(mult_future * sum_future_endpoints);
            }
            h[0] += eps * plain_c0;
            h[1] += eps * plain_c1;
            h[2] += eps * plain_c_inf;

            // Affine consistency check.
            //
            // Per-term contribution to `h(0) + h(1) = 2·h[0] + Σ_{i≥1} h[i]`,
            // grouped by where each term writes into `h`:
            //   live (writes h[i] for i in 0..ell_zk, coefficient 2^{k-j}·s_j[i]):
            //     contributes 2^{k-j} · (s_j(0) + s_j(1)) = 2^{k-j} · s_j_endpoints
            //   past (writes h[0] only, coefficient 2^{k-j}·past_mask_sum):
            //     contributes 2^{k-j+1} · past_mask_sum
            //   future (writes h[0] only, coefficient 2^{k-j-1}·sum_future):
            //     contributes 2^{k-j} · sum_future   (zero in round j = k)
            //   plain (writes h[0,1,2], coefficients ε·c_0, ε·c_1, ε·c_∞):
            //     contributes ε · (2·c_0 + c_1 + c_∞) = ε · plain_sum
            //
            // The two anchor cases:
            //   j = 1: sum collapses to `μ̃ + ε·μ`  (round-1 target).
            //   j > 1: sum equals `ĥ_{j-1}(γ_{j-1})` by induction.
            #[cfg(debug_assertions)]
            {
                let mult_live = F::TWO.exp_u64((k - j) as u64);
                let mult_past = F::TWO.exp_u64((k - j + 1) as u64);
                let mut expected = EF::from(mult_live * s_j_endpoints)
                    + EF::from(mult_past) * past_mask_sum
                    + eps * plain_sum;
                if j < k {
                    expected += EF::from(mult_live * sum_future_endpoints);
                }
                debug_assert_eq!(
                    h[0].double() + h[1..].iter().copied().sum::<EF>(),
                    expected,
                    "ĥ_j affine consistency check failed at round {j}",
                );
            }

            // Wire format: send `(c_0, c_2, c_3, …, c_d)`, skipping `c_1`.
            let mut wire: Vec<EF> = Vec::with_capacity(h_size - 1);
            wire.push(h[0]);
            wire.extend_from_slice(&h[2..]);

            challenger.observe_algebra_slice(&wire);
            zk_data.round_coefficients.push(wire);

            if pow_bits > 0 {
                zk_data.pow_witnesses.push(challenger.grind(pow_bits));
            }
            let gamma_j: EF = challenger.sample_algebra_element();

            // Cache `s_j(γ_j)` via Horner for the past-mask term in future rounds.
            let s_j_at_gamma_j: EF = s_j
                .iter()
                .rev()
                .copied()
                .fold(EF::ZERO, |acc, c| acc * gamma_j + EF::from(c));
            mask_evals_at_gamma.push(s_j_at_gamma_j);

            // Update plain_sum via quadratic extrapolation.
            plain_sum = extrapolate_01inf(plain_c0, plain_sum - plain_c0, plain_c_inf, gamma_j);
            rs.push(gamma_j);
        }

        // --- Residual hand-off (mirrors PrefixProver::into_sumcheck) ---
        let rs = Point::new(rs);
        // Absorb `ε` into the folded base polynomial by passing it as the
        // `scale` argument; the residual product polynomial then satisfies
        // `dot_product() == ε · plain_residual_sum`.
        let compressed = tracing::info_span!("compress_prefix_to_packed")
            .in_scope(|| inner.poly.compress_prefix_to_packed(&rs, eps));
        let weights = inner.combine_eqs(&rs, alpha).pack::<F, EF>();
        let prod_poly =
            ProductPolynomial::<F, EF>::new_packed(VariableOrder::Prefix, compressed, weights);

        let residual_sum = eps * plain_sum;
        debug_assert_eq!(
            prod_poly.dot_product(),
            residual_sum,
            "residual product polynomial dot product must equal ε · plain_residual_sum",
        );

        (
            SumcheckProver::new(prod_poly, residual_sum),
            rs,
            mask_oracles,
        )
    }
}

/// HVZK verifier wrapping the layout [`Verifier`] with the affine-chain check.
pub struct ZkVerifier<F, EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    inner: Verifier<F, EF>,
}

impl<F, EF> ZkVerifier<F, EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    /// Builds the verifier-side registry mirroring [`PrefixProver`]'s strategy.
    pub fn new(table_shapes: &[TableShape]) -> Self {
        Self {
            inner: Verifier::new(table_shapes, prefix_strategy()),
        }
    }

    /// Records concrete opening claims; delegates to `Verifier::add_claim`.
    pub fn add_claim<Ch>(
        &mut self,
        table_idx: usize,
        polys: &[usize],
        evals: &[EF],
        challenger: &mut Ch,
    ) where
        Ch: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        self.inner.add_claim(table_idx, polys, evals, challenger);
    }

    /// Records a virtual opening claim; delegates to `Verifier::add_virtual_eval`.
    pub fn add_virtual_eval<Ch>(&mut self, eval: EF, challenger: &mut Ch)
    where
        Ch: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        self.inner.add_virtual_eval(eval, challenger);
    }

    /// Verifier counterpart of [`ZkPrefixProver::into_sumcheck`].
    ///
    /// Replays the prover's transcript actions from `zk_data` and the supplied
    /// `mask_commits`, reconstructs the dropped linear coefficient `c_1` of
    /// each round polynomial via the affine consistency check, verifies any
    /// proof-of-work witnesses, and samples `(γ_1, …, γ_k)` from the
    /// challenger. Returns `(γs, target)` where `target = ĥ_k(γ_k)` — the
    /// residual claim the downstream committed-sumcheck relation must close
    /// against the mask oracles and the witness polynomial.
    ///
    /// # Errors
    ///
    /// - [`SumcheckError::RoundCountMismatch`] on `zk_data` shape mismatch.
    /// - [`SumcheckError::MaskCommitmentCountMismatch`] on mask-count mismatch.
    /// - [`SumcheckError::PowWitnessCountMismatch`] on PoW-shape mismatch.
    /// - [`SumcheckError::WireSizeMismatch`] on per-round wire shape mismatch.
    /// - [`SumcheckError::InvalidPowWitness`] on a failed PoW check.
    ///
    /// # Panics
    ///
    /// - If `char(F) == 2` or `ell_zk < 2` (Lemma 6.4 hypotheses).
    /// - If `folding_factor == 0`.
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
        M: Mmcs<F>,
        Ch: FieldChallenger<F> + GrindingChallenger<Witness = F> + CanObserve<M::Commitment>,
    {        
        const { assert!(
            F::TWO != F::ZERO,
            "Construction 6.3 (Lemma 6.4) requires char(F) != 2",
        ); }
        assert!(
            ell_zk >= 2,
            "Construction 6.3 (Lemma 6.4) requires ell_zk >= 2",
        );
        assert!(k >= 1, "sumcheck requires at least one round");

        if zk_data.round_coefficients.len() != k {
            return Err(SumcheckError::RoundCountMismatch {
                expected: folding_factor,
                actual: zk_data.round_coefficients.len(),
            });
        }
        if mask_commits.len() != k {
            return Err(SumcheckError::MaskCommitmentCountMismatch {
                expected: folding_factor,
                actual: mask_commits.len(),
            });
        }
        let expected_pow = if pow_bits > 0 { k } else { 0 };
        if zk_data.pow_witnesses.len() != expected_pow {
            return Err(SumcheckError::PowWitnessCountMismatch {
                expected: expected_pow,
                actual: zk_data.pow_witnesses.len(),
            });
        }
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

        // Sample α, then build μ from the layout verifier (which has been
        // pre-loaded with the public claims via add_claim / add_virtual_eval).
        let alpha: EF = challenger.sample_algebra_element();
        let mu = self.inner.sum(alpha);

        // Observe the mask commitments in mask order, then μ̃ (lifted to EF),
        // then sample ε. Matches the prover's prelude byte-for-byte.
        for commit in mask_commits {
            challenger.observe(commit.clone());
        }
        challenger.observe_algebra_element(EF::from(zk_data.mu_tilde));
        let eps: EF = challenger.sample_algebra_element();

        // Round-1 affine target: ĥ_1(0) + ĥ_1(1) = ε·μ + μ̃.
        let mut target: EF = eps * mu + EF::from(zk_data.mu_tilde);
        let mut randomness: Vec<EF> = Vec::with_capacity(k);

        for (j_idx, wire) in zk_data.round_coefficients.iter().enumerate() {
            let c0 = wire[0];
            let high_sum: EF = wire[1..].iter().copied().sum();
            let c1 = target - c0.double() - high_sum;

            challenger.observe_algebra_slice(wire);

            if pow_bits > 0 && !challenger.check_witness(pow_bits, zk_data.pow_witnesses[j_idx]) {
                return Err(SumcheckError::InvalidPowWitness);
            }

            let gamma_j: EF = challenger.sample_algebra_element();

            // Horner-evaluate `[c_0, c_1, c_2, …, c_d]` at γ_j.
            let mut coeffs_vec: Vec<EF> = Vec::with_capacity(h_size);
            coeffs_vec.push(c0);
            coeffs_vec.push(c1);
            coeffs_vec.extend_from_slice(&wire[1..]);
            let h_at_gamma_j: EF = coeffs_vec
                .iter()
                .rev()
                .copied()
                .fold(EF::ZERO, |acc, c| acc * gamma_j + c);

            target = h_at_gamma_j;
            randomness.push(gamma_j);
        }

        Ok((Point::new(randomness), target))
    }
}

/// Returns the layout strategy `PrefixProver` uses.
///
/// Free function rather than `PrefixProver::<F, EF>::strategy()` because the
/// trait-bound `Layout` requires `F: TwoAdicField`, which the verifier doesn't
/// inherit. The value is fixed by construction.
pub const fn prefix_strategy() -> LayoutStrategy {
    LayoutStrategy::new(true, VariableOrder::Prefix)
}

/// HVZK simulator for the classic-unpacked HVZK sumcheck (Lemma 6.4).
///
/// Runs the prover's Fiat-Shamir actions *without ever consulting the witness
/// polynomial*. The simulator samples fresh masks (so the mask commitments are
/// distributed identically to the real prover's for Reed–Solomon encoding,
/// `ζ_RS = 0`) and per round samples each wire form coordinate uniformly with
/// the F/EF stratification documented inline. The verifier's `c_1`
/// reconstruction makes every wire automatically affine-consistent, so no
/// further per-round consistency is needed.
///
/// # Distributional match (Lemma 6.4)
///
/// For Reed-Solomon `Enc_{C_zk}` the affine subspace of valid
/// `(μ̃, ĥ_1, …, ĥ_k)` tuples has dimension `1 + k·max(ℓ_zk - 1, 2)` and the
/// honest-prover formulas are surjective onto it (rank-nullity, §6.1 of
/// eprint 2026/391). The simulator produces an identically-distributed
/// `(μ̃, wire forms)` by sampling each coordinate uniformly within the
/// stratification.
///
/// # Returns
///
/// - The simulated `ZkSumcheckData`.
/// - The mask commitments.
/// - The `(γ_1, …, γ_k)` randomness vector — useful for round-trip assertions.
///
/// # Panics
///
/// Same precondition asserts as [`ZkPrefixProver::into_sumcheck`].
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_lines)]
pub fn simulate_classic_unpacked<F, EF, Enc, M, Challenger, R>(
    challenger: &mut Challenger,
    folding_factor: usize,
    pow_bits: usize,
    mu: EF,
    encoding: &Enc,
    mmcs: &M,
    rng: &mut R,
) -> (ZkSumcheckData<F, EF>, Vec<M::Commitment>, Point<EF>)
where
    F: Field,
    EF: ExtensionField<F>,
    Enc: ZkEncoding<F>,
    Enc::Codeword: Matrix<F>,
    M: Mmcs<F>,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F> + CanObserve<M::Commitment>,
    R: Rng,
    StandardUniform: Distribution<F> + Distribution<EF>,
{
    let k = folding_factor;
    let ell_zk = encoding.message_len();

    assert!(
        F::TWO != F::ZERO,
        "Construction 6.3 (Lemma 6.4) requires char(F) != 2",
    );
    assert!(
        ell_zk >= 2,
        "Construction 6.3 (Lemma 6.4) requires ell_zk >= 2",
    );
    assert!(k >= 1, "sumcheck requires at least one round");

    // The caller has pre-loaded the challenger with claim observations
    // identical to the real prover's; `α` is the next thing sampled.
    // !!CRITICAL!! not a piece of dead code. This is needed for
    // challenger-state sync.
    let _: EF = challenger.sample_algebra_element();

    // Sample masks + commit + observe (identical distribution to real prover
    // for Reed–Solomon encoding: codewords uniform ⇒ commits indistinguishable).
    let masks: Vec<Vec<F>> = (0..k)
        .map(|_| (0..ell_zk).map(|_| rng.random()).collect())
        .collect();
    let mut mask_commits: Vec<M::Commitment> = Vec::with_capacity(k);
    for mask in &masks {
        let codeword = encoding.encode(mask, rng);
        let (commit, _prover_data) = mmcs.commit_matrix(codeword);
        challenger.observe(commit.clone());
        mask_commits.push(commit);
    }

    // Closed-form `μ̃` from masks; byte-equivalent to the prover under matched
    // seeds (distributionally we could equivalently sample uniform in `F`).
    let two_to_k_minus_1 = F::TWO.exp_u64((k - 1) as u64);
    let mu_tilde: F = two_to_k_minus_1
        * masks
            .iter()
            .map(|m| m[0] + m.iter().copied().sum::<F>())
            .sum::<F>();

    challenger.observe_algebra_element(EF::from(mu_tilde));
    let eps: EF = challenger.sample_algebra_element();

    // Wire-to-coefficient index map (wire skips `c_1`):
    //   wire[0] = c_0, wire[1] = c_2, wire[i] = c_{i+1} for i ≥ 1.
    //
    // Two-tier sampling, matching honest stratification of `ĥ_j` (round index
    // `j ∈ [1, k]`; wire position `i ∈ [0, wire_size)`):
    //   - wire[0], wire[1] (= c_0, c_2): receive an `ε · plain_c_*` term in
    //     honest execution, so live in EF. Sample uniformly from EF.
    //   - wire[i] for i ≥ 2 (= c_3, …, c_{ell_zk-1}): receive only the
    //     live-mask contribution `2^{k-j} · s_j[i+1]` with `s_j[i+1] ∈ F`,
    //     so live in the F-subspace of EF. Sample from F lifted via `EF::from`.
    // Without this stratification a distinguisher trivially separates real
    // from simulated views by checking the EF coordinates `[1..]` of each
    // `c_i` for `i ≥ 3` (paper §6.1).
    let h_size = core::cmp::max(ell_zk, 3);
    let wire_size = h_size - 1;
    let mut zk_data = ZkSumcheckData::<F, EF> {
        mu_tilde,
        round_coefficients: Vec::with_capacity(k),
        pow_witnesses: Vec::with_capacity(if pow_bits > 0 { k } else { 0 }),
    };
    let mut randomness: Vec<EF> = Vec::with_capacity(k);
    let mut target: EF = eps * mu + EF::from(mu_tilde);

    for _ in 0..k {
        let wire: Vec<EF> = (0..wire_size)
            .map(|i| {
                if i < 2 {
                    rng.random::<EF>()
                } else {
                    EF::from(rng.random::<F>())
                }
            })
            .collect();

        challenger.observe_algebra_slice(&wire);

        if pow_bits > 0 {
            zk_data.pow_witnesses.push(challenger.grind(pow_bits));
        }

        let gamma_j: EF = challenger.sample_algebra_element();

        // Reconstruct `c_1` and Horner-evaluate `ĥ_j(γ_j)` to advance target.
        // Mirrors the verifier so the running target matches what it will
        // compute, keeping the debug invariant tight.
        let c0 = wire[0];
        let high_sum: EF = wire[1..].iter().copied().sum();
        let c1 = target - c0.double() - high_sum;
        let mut coeffs: Vec<EF> = Vec::with_capacity(h_size);
        coeffs.push(c0);
        coeffs.push(c1);
        coeffs.extend_from_slice(&wire[1..]);
        target = coeffs
            .iter()
            .rev()
            .copied()
            .fold(EF::ZERO, |acc, c| acc * gamma_j + c);

        zk_data.round_coefficients.push(wire);
        randomness.push(gamma_j);
    }

    (zk_data, mask_commits, Point::new(randomness))
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_challenger::DuplexChallenger;
    use p3_dft::Radix2DFTSmallBatch;
    use p3_field::extension::BinomialExtensionField;
    use p3_field::{BasedVectorSpace, Field, PackedValue, PrimeCharacteristicRing};
    use p3_merkle_tree::MerkleTreeMmcs;
    use p3_multilinear_util::poly::Poly;
    use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
    use p3_zk_codes::reed_solomon::ReedSolomonZkEncoding;
    use proptest::prelude::*;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;
    use crate::sumcheck::layout::{Layout, PrefixProver, Table, Witness};

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;
    type Perm = Poseidon2BabyBear<16>;
    type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
    type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
    type MyChallenger = DuplexChallenger<F, Perm, 16, 8>;
    type PackedF = <F as Field>::Packing;
    type MyMmcs = MerkleTreeMmcs<PackedF, PackedF, MyHash, MyCompress, 2, 8>;
    type MyDft = Radix2DFTSmallBatch<F>;
    type MyEnc = ReedSolomonZkEncoding<F, MyDft>;

    /// Reed-Solomon ZK encoding parameter: `t = 2` randomness symbols.
    const T: usize = 2;

    /// Builds the proper setup (perm, mmcs, encoding) shared by every test
    /// from a `seed`. Permutation seeded from `seed`; both challengers receive
    /// an identical permutation state by re-instantiating from the same seed.
    fn make_setup(seed: u64, ell_zk: usize) -> (Perm, MyMmcs, MyEnc) {
        let mut perm_rng = SmallRng::seed_from_u64(seed);
        let perm = Perm::new_from_rng_128(&mut perm_rng);

        let merkle_hash = MyHash::new(perm.clone());
        let merkle_compress = MyCompress::new(perm.clone());
        let mmcs: MyMmcs = MyMmcs::new(merkle_hash, merkle_compress, 0);

        let m = (ell_zk + T).next_power_of_two();
        let dft = MyDft::default();
        let encoding = MyEnc::new(T, ell_zk, m, dft);

        (perm, mmcs, encoding)
    }

    /// Builds a fresh `(ZkPrefixProver, ZkVerifier)` pair from the given
    /// witness polynomial. The two will be driven side-by-side; the test
    /// fixture mirrors prover-side `add_virtual_eval` calls with verifier-side
    /// `add_virtual_eval(eval, …)` calls, so the challenger states stay in
    /// lockstep.
    fn build_prover_verifier(
        evals: Vec<F>,
        folding_factor: usize,
        encoding: MyEnc,
        mmcs: MyMmcs,
    ) -> (
        ZkPrefixProver<F, EF, MyEnc, MyMmcs>,
        ZkVerifier<F, EF>,
        usize, // n_vars (for shape recording)
    ) {
        let n_vars = p3_util::log2_strict_usize(evals.len());
        let poly = Poly::new(evals);
        let table = Table::new(alloc::vec![poly]);
        let witness: Witness<F> = Witness::new_interleaved(alloc::vec![table], folding_factor);
        let inner = PrefixProver::<F, EF>::from_witness(witness);
        let prover = ZkPrefixProver::new(inner, encoding, mmcs);
        let verifier = ZkVerifier::<F, EF>::new(&[TableShape::new(n_vars, 1)]);
        (prover, verifier, n_vars)
    }

    /// End-to-end honest-prover ↔ honest-verifier run.
    ///
    /// Returns `Ok(())` on a successful match between the prover's challenges
    /// and the verifier's Fiat-Shamir replay; returns an error string carrying
    /// enough context to read in a proptest failure report.
    fn run_roundtrip(
        n_vars: usize,
        folding_factor: usize,
        ell_zk: usize,
        num_eqs: usize,
        seed: u64,
    ) -> Result<(), &'static str> {
        let (perm, mmcs, encoding) = make_setup(seed, ell_zk);

        let mut data_rng = SmallRng::seed_from_u64(seed.wrapping_add(1));
        let evals: Vec<F> = (0..(1usize << n_vars)).map(|_| data_rng.random()).collect();

        let (mut prover, mut verifier, _n_vars) =
            build_prover_verifier(evals, folding_factor, encoding, mmcs);

        // Drive prover and verifier challengers in lockstep: each
        // `add_virtual_eval` on the prover side returns an eval; the verifier
        // absorbs it with the matching `add_virtual_eval(eval, …)`.
        let mut prover_challenger = MyChallenger::new(perm.clone());
        let mut verifier_challenger = MyChallenger::new(perm);

        for _ in 0..num_eqs {
            let eval = prover.add_virtual_eval(&mut prover_challenger);
            verifier.add_virtual_eval(eval, &mut verifier_challenger);
        }

        let pow_bits = 4;
        let mut zk_data = ZkSumcheckData::<F, EF>::default();
        let mut prover_rng = SmallRng::seed_from_u64(seed.wrapping_add(2));

        let (_residual_prover, prover_randomness, mask_oracles) = prover.into_sumcheck(
            &mut zk_data,
            pow_bits,
            &mut prover_challenger,
            &mut prover_rng,
        );

        let mask_commits: Vec<_> = mask_oracles.iter().map(|(c, _)| c.clone()).collect();

        let (verifier_point, _final_target) = verifier
            .into_sumcheck::<MyMmcs, _>(
                &zk_data,
                &mask_commits,
                ell_zk,
                folding_factor,
                pow_bits,
                &mut verifier_challenger,
            )
            .map_err(|_| "verifier rejected honest prover output")?;

        let prover_randomness_vec: Vec<EF> = prover_randomness.iter().copied().collect();
        let verifier_randomness_vec: Vec<EF> = verifier_point.iter().copied().collect();
        if prover_randomness_vec != verifier_randomness_vec {
            return Err("prover/verifier disagreed on sumcheck randomness");
        }
        Ok(())
    }

    #[test]
    fn prover_verifier_roundtrip_classic_unpacked() {
        // One concrete run with non-tiny parameters: small enough to be fast,
        // big enough to exercise the per-round Lagrange-weighted accumulator
        // loop across multiple rounds.
        run_roundtrip(8, 3, 4, 2, 0).expect("honest roundtrip should accept");
    }

    /// Negative-path coverage for the verifier's PoW check at zk.rs:674.
    ///
    /// `run_roundtrip` exercises the OK arm of `if pow_bits > 0 &&
    /// !challenger.check_witness(...)` — wiring catch only. This test pins
    /// down the rejection arm: mutating any single PoW witness must produce
    /// `SumcheckError::InvalidPowWitness`.
    ///
    /// Uses `pow_bits = 16` so the false-positive probability of a forged
    /// witness accidentally passing the difficulty check is `2^-16`,
    /// effectively zero for any concrete seed.
    #[test]
    fn forged_pow_witness_rejected() {
        let n_vars = 6;
        let folding_factor = 2;
        let ell_zk = 4;
        let num_eqs = 1;
        let seed = 0u64;
        let pow_bits = 16;

        let (perm, mmcs, encoding) = make_setup(seed, ell_zk);
        let mut data_rng = SmallRng::seed_from_u64(seed.wrapping_add(1));
        let evals: Vec<F> = (0..(1usize << n_vars)).map(|_| data_rng.random()).collect();

        let (mut prover, mut verifier, _) =
            build_prover_verifier(evals, folding_factor, encoding, mmcs);

        let mut prover_ch = MyChallenger::new(perm.clone());
        let mut verifier_ch = MyChallenger::new(perm);

        for _ in 0..num_eqs {
            let eval = prover.add_virtual_eval(&mut prover_ch);
            verifier.add_virtual_eval(eval, &mut verifier_ch);
        }

        let mut zk_data = ZkSumcheckData::<F, EF>::default();
        let mut prover_rng = SmallRng::seed_from_u64(seed.wrapping_add(2));
        let (_residual, _rand, mask_oracles) =
            prover.into_sumcheck(&mut zk_data, pow_bits, &mut prover_ch, &mut prover_rng);

        // Tamper with the first round's PoW witness. Adding `F::ONE` is a
        // sufficient mutation: `check_witness` re-derives the difficulty bound
        // from challenger state and rejects unless the witness happens to be
        // one of the `2^{32-pow_bits}` valid preimages — a `2^-16` event here.
        assert_eq!(zk_data.pow_witnesses.len(), folding_factor);
        zk_data.pow_witnesses[0] += F::ONE;

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

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(16))]

        /// Completeness: every honest prover output must verify across the
        /// `(n_vars, k, ell_zk, num_eqs)` cube.
        #[test]
        fn prop_completeness_classic_unpacked(
            n_vars in 3usize..=8,
            ell_zk in 2usize..=5,
            num_eqs in 1usize..=3,
            seed in 0u64..1024,
        ) {
            // `compress_prefix_to_packed` (called in `into_sumcheck`) requires
            // `n_vars - folding >= log2(F::Packing::WIDTH)` to produce at least
            // one full packed element. WIDTH varies per architecture (NEON=4,
            // AVX2=8, AVX512=16); pick `folding` within that bound.
            let k_pack = p3_util::log2_strict_usize(<F as Field>::Packing::WIDTH);
            prop_assume!(n_vars > k_pack);
            let folding_factor = 1 + (seed as usize % (n_vars - k_pack));
            prop_assert!(run_roundtrip(n_vars, folding_factor, ell_zk, num_eqs, seed).is_ok());
        }
    }

    /// Lemma 6.4 view-match driver for the RS instantiation (`ζ_RS = 0`).
    ///
    /// Runs the real HVZK sumcheck prover and `simulate_classic_unpacked` side
    /// by side with **matched mask-RNG seeds**, then exercises the Lemma 6.4
    /// step-5 oracle simulator (`ZkEncoding::simulate`) for each mask. Pins
    /// down the following invariants per case:
    ///
    /// 1. **Verifier accepts both transcripts.** Honest soundness floor.
    /// 2. **Coupling certificate.** Real and simulator each draw `k·ell_zk`
    ///    mask elements then `k·t_zk` encoding-randomness elements from their
    ///    RNG in the same order; under matched seeds `(μ̃, mask commits)` is
    ///    bit-identical. A deterministic equality, not a distributional test.
    /// 3. **F-subspace stratification on both sides.** wire[i] for `i ≥ 2`
    ///    carries only `2^{k-j} · s_j[i+1] ∈ F` honestly; the simulator
    ///    samples it from F by construction. This is the marginal of the
    ///    joint `(μ̃, ĥ_1, …, ĥ_k)` distribution that a distinguisher trivially
    ///    inspects (paper §6.1), so matching it on both sides is necessary.
    /// 4. **Mask oracle queries via `ZkEncoding::simulate`** (Lemma 6.4 step 5,
    ///    `a_{s_j} ← Sim_{C_zk}(Q_{s_j})`). The driver picks distinct positions
    ///    within the encoding's query bound; for RS (`error() == 0`) the
    ///    simulated answers are identically distributed to opening the real
    ///    codeword at those positions (Definition 3.16, PR #1584/#1601).
    fn run_view_match_rs(
        n_vars: usize,
        folding_factor: usize,
        ell_zk: usize,
        num_eqs: usize,
        seed: u64,
    ) -> Result<(), &'static str> {
        let (perm, mmcs, encoding) = make_setup(seed, ell_zk);
        let mut data_rng = SmallRng::seed_from_u64(seed.wrapping_add(1));
        let evals: Vec<F> = (0..(1usize << n_vars)).map(|_| data_rng.random()).collect();

        // ===== Real prover ↔ verifier =====
        let (mut prover, mut verifier_real, _) =
            build_prover_verifier(evals, folding_factor, encoding.clone(), mmcs.clone());

        let mut prover_ch = MyChallenger::new(perm.clone());
        let mut verifier_real_ch = MyChallenger::new(perm.clone());
        let mut virtual_evals: Vec<EF> = Vec::with_capacity(num_eqs);
        for _ in 0..num_eqs {
            let eval = prover.add_virtual_eval(&mut prover_ch);
            verifier_real.add_virtual_eval(eval, &mut verifier_real_ch);
            virtual_evals.push(eval);
        }

        let pow_bits = 0;
        let mut zk_data_real = ZkSumcheckData::<F, EF>::default();
        let mut real_rng = SmallRng::seed_from_u64(seed.wrapping_add(2));
        let (_residual_real, _gammas_real, mask_oracles_real) =
            prover.into_sumcheck(&mut zk_data_real, pow_bits, &mut prover_ch, &mut real_rng);
        let mask_commits_real: Vec<_> = mask_oracles_real.iter().map(|(c, _)| c.clone()).collect();

        let (_real_rand, _real_target) = verifier_real
            .into_sumcheck::<MyMmcs, _>(
                &zk_data_real,
                &mask_commits_real,
                ell_zk,
                folding_factor,
                pow_bits,
                &mut verifier_real_ch,
            )
            .map_err(|_| "real prover transcript rejected by verifier")?;

        // ===== Simulator side, matched mask-RNG seed =====
        let mut verifier_sim = ZkVerifier::<F, EF>::new(&[TableShape::new(n_vars, 1)]);
        let mut sim_ch = MyChallenger::new(perm.clone());
        let mut verifier_sim_ch = MyChallenger::new(perm);

        // Replay the same virtual eval observations onto both simulator-side
        // transcripts so verifier_sim and sim_ch reach the post-claim Fiat-
        // Shamir state the real run reached. Throwaway `tmp_verifier`s advance
        // verifier_sim_ch without double-recording on the live verifier_sim;
        // standard pattern in this module.
        for &eval in &virtual_evals {
            verifier_sim.add_virtual_eval(eval, &mut sim_ch);
        }
        for &eval in &virtual_evals {
            let mut tmp_verifier = ZkVerifier::<F, EF>::new(&[TableShape::new(n_vars, 1)]);
            tmp_verifier.add_virtual_eval(eval, &mut verifier_sim_ch);
        }

        // Forked-challenger `α` peek so we can hand the simulator the same `μ`
        // the verifier will compute. `simulate_classic_unpacked` re-samples
        // `α` from its own challenger immediately on entry — peeking on a
        // clone does not disturb that.
        let mut alpha_peek = sim_ch.clone();
        let alpha: EF = alpha_peek.sample_algebra_element();
        let mu: EF = virtual_evals
            .iter()
            .zip(alpha.powers())
            .map(|(&e, a)| e * a)
            .sum();

        // *** Same seed as `real_rng` above — the coupling certificate's
        // deterministic equality depends on this match. ***
        let mut sim_rng = SmallRng::seed_from_u64(seed.wrapping_add(2));
        let (zk_data_sim, mask_commits_sim, _gammas_sim) =
            simulate_classic_unpacked::<F, EF, _, _, _, _>(
                &mut sim_ch,
                folding_factor,
                pow_bits,
                mu,
                &encoding,
                &mmcs,
                &mut sim_rng,
            );

        let (_sim_rand, _sim_target) = verifier_sim
            .into_sumcheck::<MyMmcs, _>(
                &zk_data_sim,
                &mask_commits_sim,
                ell_zk,
                folding_factor,
                pow_bits,
                &mut verifier_sim_ch,
            )
            .map_err(|_| "simulator transcript rejected by verifier")?;

        // ===== Coupling certificate =====
        if zk_data_real.mu_tilde != zk_data_sim.mu_tilde {
            return Err("matched-RNG coupling: μ̃ differs between real and simulator");
        }
        if mask_commits_real != mask_commits_sim {
            return Err("matched-RNG coupling: mask commits differ between real and simulator");
        }

        // ===== F-subspace stratification on both sides =====
        for wire in &zk_data_real.round_coefficients {
            for &c in wire.iter().skip(2) {
                if !ef_in_f_subspace(c) {
                    return Err("real-prover wire[i≥2] escapes the F-subspace of EF");
                }
            }
        }
        for wire in &zk_data_sim.round_coefficients {
            for &c in wire.iter().skip(2) {
                if !ef_in_f_subspace(c) {
                    return Err("simulator wire[i≥2] escapes the F-subspace of EF");
                }
            }
        }

        // ===== Lemma 6.4 step 5: mask oracle queries via ZkEncoding::simulate =====
        //
        // One distinct query set per mask, size in [1, t_zk]; staying within
        // the bound avoids `ReedSolomonZkEncoding::simulate`'s `≤ t` panic.
        let t_zk = encoding.randomness_len();
        let m = encoding.m;
        let mut query_rng = SmallRng::seed_from_u64(seed.wrapping_add(5));
        let mut sim_ans_rng = SmallRng::seed_from_u64(seed.wrapping_add(6));
        for _ in 0..folding_factor {
            let q_size = query_rng.random_range(1..=t_zk);
            let mut positions: Vec<usize> = Vec::with_capacity(q_size);
            while positions.len() < q_size {
                let p = query_rng.random_range(0..m);
                if !positions.contains(&p) {
                    positions.push(p);
                }
            }
            let sim_answers: Vec<F> = encoding.simulate(&positions, &mut sim_ans_rng);
            if sim_answers.len() != positions.len() {
                return Err("ZkEncoding::simulate returned wrong number of answers");
            }
        }

        Ok(())
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(16))]

        /// The HVZK simulator (sumcheck transcript via
        /// `simulate_classic_unpacked` plus mask oracle answers via
        /// `ZkEncoding::simulate`) matches the real prover's view exactly
        /// across the `(n_vars, k, ell_zk, num_eqs)` cube for RS encoding
        /// (`error() == 0`). See `run_view_match_rs` for the concrete
        /// invariants checked per case.
        #[test]
        fn prop_simulator_view_matches_real_rs(
            n_vars in 3usize..=8,
            ell_zk in 2usize..=5,
            num_eqs in 1usize..=3,
            seed in 0u64..1024,
        ) {
            // See `prop_completeness_classic_unpacked` for the packing-width
            // precondition `compress_prefix_to_packed` imposes on `folding`.
            let k_pack = p3_util::log2_strict_usize(<F as Field>::Packing::WIDTH);
            prop_assume!(n_vars > k_pack);
            let folding_factor = 1 + (seed as usize % (n_vars - k_pack));
            prop_assert!(
                run_view_match_rs(n_vars, folding_factor, ell_zk, num_eqs, seed).is_ok()
            );
        }
    }

    // Tampering-propagation flavour of RBR soundness: flipping a single field
    // element in any round's wire form yields a different verifier-computed
    // `target = ĥ_k(γ_k)`. The exact value isn't important; what matters is
    // that *any* byte change drives the verifier off the honest path.
    //
    // The affine reconstruction of `c_1` makes a tampered wire trivially
    // satisfy round `j`'s identity `ĥ_j(0) + ĥ_j(1) = ĥ_{j-1}(γ_{j-1})`. RBR
    // soundness still needs the cheat to be caught; this test asserts the
    // cheat *propagates* — the next sampled `γ_{j+1}` and the residual target
    // follow a different trajectory once a byte changes.
    //
    // Coverage role. Lemma 6.5 of eprint 2026/391 proves the worst-case
    // probability that a random `γ_j` rehabilitates a bad witness is
    // `ε_j ≤ ε_mca + ℓ_zk · |Λ|² / |F|`. That bound is a theorem about the
    // abstract protocol; what an implementation can test is conformance to
    // the protocol the lemma analyses. That conformance is covered jointly
    // by `prop_completeness_classic_unpacked` (honest case accepts),
    // `prop_simulator_view_matches_real_rs` (Lemma 6.4 simulator-real
    // bit-equivalence, the HVZK structure Lemma 6.5 builds on), and this
    // test (any cheat propagates through Fiat-Shamir).
    //
    // Not tested here: the quantitative `empirical_rate ≤ ε_j` match. With
    // `γ_j` sampled from `EF ≈ 2^124` (frozen design decision: ε ∈ EF for
    // soundness margin), the bound is ~2^-110 for any feasible params —
    // unobservable at any trial count CI can host without a sub-2^31
    // test-only field type, which `p3-field` doesn't ship.
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
            // See `prop_completeness_classic_unpacked` for the packing-width
            // precondition `compress_prefix_to_packed` imposes on `folding`.
            let k_pack = p3_util::log2_strict_usize(<F as Field>::Packing::WIDTH);
            prop_assume!(n_vars > k_pack);
            let folding_factor = 1 + (seed as usize % (n_vars - k_pack));

            let (perm, mmcs, encoding) = make_setup(seed, ell_zk);

            let mut data_rng = SmallRng::seed_from_u64(seed.wrapping_add(1));
            let evals: Vec<F> = (0..(1usize << n_vars)).map(|_| data_rng.random()).collect();

            let (mut prover, mut verifier, _n_vars) =
                build_prover_verifier(evals, folding_factor, encoding, mmcs);

            let mut prover_challenger = MyChallenger::new(perm.clone());
            let mut verifier_challenger = MyChallenger::new(perm.clone());
            for _ in 0..num_eqs {
                let eval = prover.add_virtual_eval(&mut prover_challenger);
                verifier.add_virtual_eval(eval, &mut verifier_challenger);
            }

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

            // Tamper one wire coordinate.
            let tamper_round = tamper_round_seed % zk_data.round_coefficients.len();
            let wire_len = zk_data.round_coefficients[tamper_round].len();
            let tamper_pos = tamper_pos_seed % wire_len;
            let mut tampered_zk_data = zk_data.clone();
            tampered_zk_data.round_coefficients[tamper_round][tamper_pos] +=
                EF::from_u64(1);

            // Honest verifier replay.
            let mut honest_v_challenger = MyChallenger::new(perm.clone());
            let mut honest_verifier =
                ZkVerifier::<F, EF>::new(&[TableShape::new(n_vars, 1)]);
            // Mirror the prover's eq absorptions: we need the verifier in the
            // same transcript state and with the same claims.
            let mut prover_replay = MyChallenger::new(perm.clone());
            let mut replay_evals = Vec::with_capacity(num_eqs);
            let (mut replay_prover, _, _) = {
                let mut replay_rng = SmallRng::seed_from_u64(seed.wrapping_add(1));
                let evals: Vec<F> =
                    (0..(1usize << n_vars)).map(|_| replay_rng.random()).collect();
                let (perm2, mmcs2, encoding2) = make_setup(seed, ell_zk);
                let _ = perm2; // already used via outer `perm`
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

            // Tampered replay: same setup, tampered wire.
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

            prop_assert_ne!(
                honest_target, tampered_target,
                "tampering with wire coordinate ({}, {}) must change target",
                tamper_round, tamper_pos,
            );
        }
    }

    /// F-subspace invariant for the simulator: per round, `wire[i]` for
    /// `i ≥ 2` must live in the F-subspace of EF (i.e. its coefficients
    /// `[1..]` against the extension basis are all zero). Matches the
    /// stratification documented in `simulate_classic_unpacked`'s body.
    ///
    /// Without this check, a distinguisher trivially separates real from
    /// simulated views by inspecting `wire[i]` for `i ≥ 2` (paper §6.1).
    fn ef_in_f_subspace(x: EF) -> bool {
        let coeffs = <EF as BasedVectorSpace<F>>::as_basis_coefficients_slice(&x);
        coeffs[1..].iter().all(|c| *c == F::ZERO)
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(8))]

        #[test]
        fn prop_simulator_f_subspace_invariant(
            n_vars in 3usize..=6,
            ell_zk in 4usize..=6,  // need ell_zk ≥ 4 for wire[2..] to be non-empty
            num_eqs in 1usize..=2,
            seed in 0u64..256,
        ) {
            let folding_factor = 1 + (seed as usize % n_vars);

            let (perm, mmcs, encoding) = make_setup(seed, ell_zk);

            let mut data_rng = SmallRng::seed_from_u64(seed.wrapping_add(1));
            let mut sim_challenger = MyChallenger::new(perm);

            // Build verifier-side claims to advance the challenger; the
            // simulator only cares about the post-claim transcript state.
            let mut verifier = ZkVerifier::<F, EF>::new(&[TableShape::new(n_vars, 1)]);
            let mut virtual_evals = Vec::with_capacity(num_eqs);
            for _ in 0..num_eqs {
                let eval: EF = data_rng.random();
                verifier.add_virtual_eval(eval, &mut sim_challenger);
                virtual_evals.push(eval);
            }

            let mut alpha_peek = sim_challenger.clone();
            let alpha: EF = alpha_peek.sample_algebra_element();
            let mu: EF = virtual_evals
                .iter()
                .zip(alpha.powers())
                .map(|(&e, a)| e * a)
                .sum();

            let pow_bits = 0;
            let mut sim_rng = SmallRng::seed_from_u64(seed.wrapping_add(2));

            let (sim_zk_data, _commits, _gammas) = simulate_classic_unpacked::<
                F, EF, _, _, _, _,
            >(
                &mut sim_challenger,
                folding_factor,
                pow_bits,
                mu,
                &encoding,
                &mmcs,
                &mut sim_rng,
            );

            for (round_idx, wire) in sim_zk_data.round_coefficients.iter().enumerate() {
                for (pos, &coeff) in wire.iter().enumerate().skip(2) {
                    prop_assert!(
                        ef_in_f_subspace(coeff),
                        "simulator wire[{pos}] in round {round_idx} must live in F-subspace",
                    );
                }
            }
        }
    }
}
