//! HVZK variant of the WHIR sumcheck (Construction 6.3, eprint 2026/391).
//!
//! Companion to [`super::single`]: same sumcheck reduction, but with `k` random
//! univariate masks committed under a ZK encoding so that the prover's `k`
//! round-polynomials no longer leak linear functions of the secret message.
//!
//! # Protocol overview
//!
//! 1. **Masks.** Prover samples `s_1, …, s_k ∈ F^{<ℓ_zk}[X]` and commits
//!    each encoded codeword `Enc_{C_zk}(s_j)` under MMCS, observing each
//!    commitment on the transcript.
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
//! Mirrors the same convention used by [`super::single`] for the plain
//! (degree-2) round polynomial.
//!
//! # Field constraints (Lemma 6.4)
//!
//! - `char(F) ≠ 2` — required by the rank-nullity argument that drives the
//!   HVZK simulator's affine-subspace surjectivity.
//! - `ℓ_zk ≥ 2` — needed for the mask piece to carry non-trivial information.
//!
//! Both are enforced at constructor entry.
//!
//! # References
//!
//! - eprint 2026/391, §6 Construction 6.3, Lemma 6.4 (HVZK), Lemma 6.5 (RBR).

use alloc::vec;
use alloc::vec::Vec;

use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_field::{ExtensionField, Field};
use p3_matrix::Matrix;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use p3_zk_codes::ZkEncoding;
use rand::distr::{Distribution, StandardUniform};
use rand::{Rng, RngExt};

use crate::constraints::statement::EqStatement;
use crate::sumcheck::error::SumcheckError;
use crate::sumcheck::extrapolate_01inf;
use crate::sumcheck::strategy::VariableOrder;

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

/// Namespace for the HVZK variant of the WHIR sumcheck.
///
/// Mirrors [`super::single::SingleSumcheck`]: a unit struct hosting the static
/// constructors for each sumcheck strategy.
pub struct ZkSumcheck;

/// Stateful prover for the HVZK sumcheck (Construction 6.3).
///
/// Carries the plain-piece polynomial pair (`evals`, `weights`) and its running
/// sum — folded at each `γ_j` exactly like the non-ZK path — plus the mask
/// bookkeeping required to build `ĥ_j` per the per-round formula in the module
/// docs.
#[allow(dead_code)]
pub struct ZkSumcheckProver<F, EF, Enc, M>
where
    F: Field,
    EF: ExtensionField<F>,
    Enc: ZkEncoding<F>,
    Enc::Codeword: Matrix<F>,
    M: Mmcs<F>,
{
    /// Folded evaluations of the witness polynomial — the first factor of the
    /// `Ĝ(X_1, …, X_k)` plain piece in Construction 6.3 step 4. Promoted to EF
    /// after round 1's fold and refolded at each `γ_j`.
    evals: Poly<EF>,
    /// Folded weights polynomial — the second factor of `Ĝ`, derived from the
    /// `EqStatement` plus the `α` batching challenge. Refolded in lockstep with
    /// `evals`.
    weights: Poly<EF>,
    /// Plain-piece sum: `Σ_{x ∈ {0,1}^{k-rounds_done}} evals(x) · weights(x)`,
    /// the residual hypercube sum tracked by the standard sumcheck invariant
    /// (matches `SumcheckProver::sum` in `single.rs`). Updated to
    /// `plain_h_{j-1}(γ_{j-1})` after each round-{j-1} fold.
    plain_sum: EF,
    /// ZK encoding `Enc_{C_zk}` used for the masks (Theorem 6.2 ingredient `C_zk`).
    encoding: Enc,
    /// The `k` mask polynomials `s_1, …, s_k ∈ F^{<ℓ_zk}[X]` as coefficient
    /// vectors of length `ℓ_zk` (Construction 6.3 step 1).
    masks: Vec<Vec<F>>,
    /// MMCS commitment + prover data for each encoded mask codeword.
    ///
    /// - The commitment is observed on the challenger, binding the masks to
    ///   the `ε` challenge sampled later.
    /// - The prover data is kept so downstream consumers (committed sumcheck
    ///   relation, §2.4 / §5 of the paper) can produce opening proofs for
    ///   queries to the mask oracles.
    mask_oracles: Vec<(M::Commitment, M::ProverData<Enc::Codeword>)>,
    /// Combination challenge `ε` sampled after `μ̃`; multiplies the plain
    /// piece in every round polynomial. Widened to the extension field for
    /// soundness margin (the paper writes `ε ← F`).
    eps: EF,
    /// Running future-mask endpoint sum.
    ///
    /// At the start of round `j`, before that round's start-of-round
    /// decrement, this field holds
    ///
    /// ```text
    /// Σ_{l ≥ j} (s_l(0) + s_l(1)).
    /// ```
    ///
    /// Each round subtracts `s_j(0) + s_j(1)` first, leaving `Σ_{l > j}`,
    /// which is the future-mask term in `ĥ_j`'s formula. The same quantity at
    /// `j = 1` drives the closed-form `μ̃ = 2^{k-1} · Σ_{l=1}^k (s_l(0) + s_l(1))`.
    sum_future_endpoints: F,
    /// `s_l(γ_l)` for `l < current_round`, accumulated as rounds progress.
    /// Drives the past-mask term `Σ_{l < j} s_l(γ_l)` of `ĥ_j`.
    mask_evals_at_gamma: Vec<EF>,
    /// Number of rounds remaining; decremented per `round()` call.
    rounds_left: usize,
}

impl ZkSumcheck {
    /// HVZK sumcheck via the classic unpacked (scalar) strategy.
    ///
    /// Mirrors [`super::single::SingleSumcheck::new_classic_unpacked`] in
    /// shape. Runs Construction 6.3 steps 1–3 (sample and commit masks, send
    /// `μ̃`, sample `ε`) plus round 1 of step 4 (build `ĥ_1`, sample `γ_1`,
    /// fold the base polynomial).
    ///
    /// # Algorithm
    ///
    /// 1. Sample a batching challenge `α` and combine multiple `EqStatement`
    ///    constraints into a single weight polynomial.
    /// 2. Sample masks `s_1, …, s_k ∈ F^{<ℓ_zk}[X]`; for each, encode under
    ///    `Enc_{C_zk}`, MMCS-commit the codeword, and observe the commitment.
    /// 3. Compute and observe `μ̃ = 2^{k-1} · Σ_l (s_l(0) + s_l(1))`.
    /// 4. Sample `ε`.
    /// 5. Build `ĥ_1` per the per-round formula; observe its non-linear
    ///    coefficients on the transcript; grind; sample `γ_1`.
    /// 6. Cache `s_1(γ_1)` and fold the base polynomial at `γ_1`.
    ///
    /// # Returns
    ///
    /// - The HVZK prover state, ready for rounds 2..=k via [`ZkSumcheckProver::round`].
    /// - The first verifier challenge `γ_1`.
    ///
    /// # Panics
    ///
    /// - If `char(F) == 2` (Lemma 6.4 requires `char(F) ≠ 2`).
    /// - If `encoding.message_len() < 2` (Lemma 6.4 requires `ℓ_zk ≥ 2`).
    /// - If `folding_factor` is 0 or exceeds `poly.num_variables()`.
    #[allow(clippy::too_many_arguments)]
    pub fn new_classic_unpacked<F, EF, Enc, M, Challenger, R>(
        poly: &Poly<F>,
        zk_data: &mut ZkSumcheckData<F, EF>,
        challenger: &mut Challenger,
        folding_factor: usize,
        pow_bits: usize,
        statement: &EqStatement<EF>,
        encoding: &Enc,
        mmcs: &M,
        rng: &mut R,
    ) -> (ZkSumcheckProver<F, EF, Enc, M>, Point<EF>)
    where
        F: Field,
        EF: ExtensionField<F>,
        Enc: ZkEncoding<F> + Clone,
        Enc::Codeword: Matrix<F>,
        M: Mmcs<F>,
        Challenger:
            FieldChallenger<F> + GrindingChallenger<Witness = F> + CanObserve<M::Commitment>,
        R: Rng,
        StandardUniform: Distribution<F>,
    {
        let k = folding_factor;
        let ell_zk = encoding.message_len();
        let n_vars = poly.num_variables();

        assert!(
            F::TWO != F::ZERO,
            "Construction 6.3 (Lemma 6.4) requires char(F) != 2",
        );
        assert!(
            ell_zk >= 2,
            "Construction 6.3 (Lemma 6.4) requires ell_zk >= 2",
        );
        assert!(k >= 1, "sumcheck requires at least one round");
        assert!(
            k <= n_vars,
            "folding_factor must be <= poly.num_variables()",
        );

        // Sample a batching challenge for combining multiple equality
        // constraints into a single weight polynomial. Construction 6.3
        // assumes a single-claim input (relation `R_{C, C_zk, sl}`,
        // Definition 5.8); we collapse here before proceeding.
        let alpha: EF = challenger.sample_algebra_element();
        let mut weights = Poly::zero(n_vars);
        let mut sum = EF::ZERO;
        statement.combine_hypercube::<F, false>(&mut weights, &mut sum, alpha);

        // --- Construction 6.3 step 1: sample, encode, commit, observe ---
        // Sample masks `s_1, …, s_k ∈ F^{<ell_zk}[X]` as coefficient vectors;
        // for each, encode the codeword under `Enc_{C_zk}`, MMCS-commit, and
        // observe the commitment. The commitment is what binds the masks to
        // the `ε` challenge sampled later. Encoding randomness is consumed
        // inside `Enc::encode` and not stored.
        let masks: Vec<Vec<F>> = (0..k)
            .map(|_| (0..ell_zk).map(|_| rng.random()).collect())
            .collect();
        let mask_oracles: Vec<(M::Commitment, M::ProverData<Enc::Codeword>)> = masks
            .iter()
            .map(|mask| {
                let codeword = encoding.encode(mask, rng);
                let (commit, prover_data) = mmcs.commit_matrix(codeword);
                challenger.observe(commit.clone());
                (commit, prover_data)
            })
            .collect();

        // --- Construction 6.3 step 2: send μ̃ ---
        // μ̃ = Σ_{b ∈ {0,1}^k} ŝ(b) = 2^{k-1} · Σ_l (s_l(0) + s_l(1))
        // where s_l(0) + s_l(1) = c_0 + Σ c_i = mask[0] + Σ mask
        // for s_l(X) = c_0 + c_1·X + … + c_{ell_zk-1}·X^{ell_zk-1}.
        let sum_future_endpoints: F = masks
            .iter()
            .map(|mask| mask[0] + mask.iter().copied().sum::<F>())
            .sum();
        let two_to_k_minus_1 = F::TWO.exp_u64((k - 1) as u64);
        let mu_tilde: F = two_to_k_minus_1 * sum_future_endpoints;

        // Cross-check the closed form against the naive 2^k-term sum.
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

        // Lift `μ̃` to EF for transcript observation (codebase convention).
        challenger.observe_algebra_element(EF::from(mu_tilde));
        // Record `μ̃` in the proof so the verifier can re-observe it in the
        // same position before sampling `ε`.
        zk_data.mu_tilde = mu_tilde;

        // --- Construction 6.3 step 3: sample ε ---
        let eps: EF = challenger.sample_algebra_element();

        // --- Construction 6.3 step 4, round 1: build ĥ_1, fold base ---

        // Start-of-round decrement: subtract `s_1`'s endpoints so the running
        // future-mask sum holds `Σ_{l > 1} (s_l(0) + s_l(1))`, the value the
        // per-round formula uses at `j = 1`.
        let s_1_endpoints = masks[0][0] + masks[0].iter().copied().sum::<F>();
        let sum_future_endpoints_state = sum_future_endpoints - s_1_endpoints;

        // Plain piece (degree-2): returns (c_0, c_∞); derive c_1 from the
        // affine constraint h(0) + h(1) = sum, i.e. c_1 = sum - 2·c_0 - c_∞.
        let (plain_c0, plain_c_inf) =
            VariableOrder::Prefix.sumcheck_coefficients(poly.as_slice(), weights.as_slice());
        let plain_c1 = sum - plain_c0.double() - plain_c_inf;

        // Build `ĥ_1` of length `max(ell_zk, 3)`:
        //   indices 0..ell_zk : live-mask piece           = 2^{k-1} · s_1(X)
        //   index 0           : future-mask contribution += 2^{k-2} · Σ_{l>1}(s_l(0)+s_l(1))
        //   indices 0..3      : plain piece              += ε · (c_0 + c_1·X + c_∞·X²)
        // The future-mask term is only present when `k ≥ 2`.
        let h1_size = core::cmp::max(ell_zk, 3);
        let mut h1: Vec<EF> = vec![EF::ZERO; h1_size];

        let two_pow_k_minus_1 = F::TWO.exp_u64((k - 1) as u64);
        for (i, &c) in masks[0].iter().enumerate() {
            h1[i] += EF::from(two_pow_k_minus_1 * c);
        }
        if k >= 2 {
            let two_pow_k_minus_2 = F::TWO.exp_u64((k - 2) as u64);
            h1[0] += EF::from(two_pow_k_minus_2 * sum_future_endpoints_state);
        }

        h1[0] += eps * plain_c0;
        h1[1] += eps * plain_c1;
        h1[2] += eps * plain_c_inf;

        // Round-1 affine consistency check:
        //   h(0) + h(1) = c_0 + (c_0 + c_1 + … + c_d) = 2·c_0 + Σ_{i ≥ 1} c_i,
        // which must equal μ̃ + ε·μ.
        debug_assert_eq!(
            h1[0].double() + h1[1..].iter().copied().sum::<EF>(),
            EF::from(mu_tilde) + eps * sum,
            "ĥ_1 should satisfy h(0) + h(1) = μ̃ + ε·μ",
        );

        // Wire format: send (c_0, c_2, c_3, …, c_d), skipping c_1; verifier
        // reconstructs c_1 from the affine consistency check above.
        let mut h1_wire: Vec<EF> = Vec::with_capacity(h1_size - 1);
        h1_wire.push(h1[0]);
        for i in 2..h1_size {
            h1_wire.push(h1[i]);
        }

        challenger.observe_algebra_slice(&h1_wire);
        zk_data.round_coefficients.push(h1_wire);

        // Proof-of-work grind, then sample γ_1.
        if pow_bits > 0 {
            zk_data.pow_witnesses.push(challenger.grind(pow_bits));
        }
        let gamma_1: EF = challenger.sample_algebra_element();

        // Cache `s_1(γ_1)` via Horner for the past-mask term in future rounds.
        let s1_at_gamma1: EF = masks[0]
            .iter()
            .rev()
            .copied()
            .fold(EF::ZERO, |acc, c| acc * gamma_1 + EF::from(c));
        let mask_evals_at_gamma: Vec<EF> = vec![s1_at_gamma1];

        // Fold base polynomial and weights at γ_1; update plain sum to
        // plain_h(γ_1) via quadratic extrapolation. `plain_sum` tracks the
        // plain-piece sum only — mask-side bookkeeping lives in this struct's
        // other fields, multiplied by `ε` when assembled into `ĥ_j`.
        weights.fix_prefix_var_mut(gamma_1);
        let folded_poly = poly.fix_prefix_var(gamma_1);
        let new_sum = extrapolate_01inf(plain_c0, sum - plain_c0, plain_c_inf, gamma_1);

        // Sanity: the folded pair's hypercube dot product equals the updated
        // plain sum (mirrors `ProductPolynomial::dot_product` ↔ `sum` invariant
        // in `single.rs`). Catches fold/extrapolate mismatches before round 2.
        debug_assert_eq!(
            folded_poly
                .iter()
                .zip(weights.iter())
                .map(|(&e, &w)| e * w)
                .sum::<EF>(),
            new_sum,
            "round-1 fold should preserve plain sumcheck invariant",
        );

        let prover = ZkSumcheckProver {
            evals: folded_poly,
            weights,
            plain_sum: new_sum,
            encoding: encoding.clone(),
            masks,
            mask_oracles,
            eps,
            // After round 1's start-decrement, this holds Σ_{l ≥ 2}, which is
            // the state round 2 expects at its start (before its own decrement).
            sum_future_endpoints: sum_future_endpoints_state,
            mask_evals_at_gamma,
            rounds_left: k - 1,
        };

        (prover, Point::new(vec![gamma_1]))
    }

    /// Verifier counterpart of [`Self::new_classic_unpacked`].
    ///
    /// Replays the prover's transcript actions from `zk_data` and the supplied
    /// `mask_commits`, reconstructs the dropped linear coefficient `c_1` of
    /// each round polynomial via the affine consistency check, verifies any
    /// proof-of-work witnesses, and samples `(γ_1, …, γ_k)` from the
    /// challenger. Returns the challenge point and the residual claim
    /// `ĥ_k(γ_k)` that downstream protocols (e.g. the committed sumcheck
    /// relation, §5 of eprint 2026/391) must check against the mask oracles +
    /// the witness polynomial.
    ///
    /// # Inputs
    ///
    /// - `zk_data` — the prover's `ZkSumcheckData` proof artefact (μ̃, per-round
    ///   wire forms, PoW witnesses).
    /// - `challenger` — Fiat-Shamir transcript, must be in the same state the
    ///   prover's transcript was in at the start of `new_classic_unpacked`.
    /// - `folding_factor` — `k`.
    /// - `pow_bits` — proof-of-work difficulty (must match the prover).
    /// - `statement` — same `EqStatement` the prover used; the verifier uses
    ///   it to derive `μ` from the claimed evaluations and `α` (without ever
    ///   building the weight polynomial).
    /// - `mask_commits` — the `k` MMCS mask commitments, in mask order; the
    ///   verifier observes them on the transcript exactly as the prover did.
    /// - `ell_zk` — `encoding.message_len()`. Used to re-derive the wire size
    ///   `max(ℓ_zk - 1, 2)`.
    ///
    /// # Errors
    ///
    /// - [`SumcheckError::RoundCountMismatch`] if `zk_data.round_coefficients`
    ///   does not have exactly `folding_factor` entries.
    /// - [`SumcheckError::MaskCommitmentCountMismatch`] if `mask_commits.len()
    ///   != folding_factor`.
    /// - [`SumcheckError::WireSizeMismatch`] if any round's wire form has the
    ///   wrong length.
    /// - [`SumcheckError::InvalidPowWitness`] on a failed PoW check.
    ///
    /// # Panics
    ///
    /// - If `char(F) == 2` or `ell_zk < 2` (Lemma 6.4 hypotheses).
    /// - If `folding_factor == 0`.
    #[allow(clippy::too_many_arguments)]
    pub fn verify_classic_unpacked<F, EF, M, Challenger>(
        zk_data: &ZkSumcheckData<F, EF>,
        challenger: &mut Challenger,
        folding_factor: usize,
        pow_bits: usize,
        statement: &EqStatement<EF>,
        mask_commits: &[M::Commitment],
        ell_zk: usize,
    ) -> Result<(Point<EF>, EF), SumcheckError>
    where
        F: Field,
        EF: ExtensionField<F>,
        M: Mmcs<F>,
        Challenger:
            FieldChallenger<F> + GrindingChallenger<Witness = F> + CanObserve<M::Commitment>,
    {
        let k = folding_factor;

        assert!(
            F::TWO != F::ZERO,
            "Construction 6.3 (Lemma 6.4) requires char(F) != 2",
        );
        assert!(
            ell_zk >= 2,
            "Construction 6.3 (Lemma 6.4) requires ell_zk >= 2",
        );
        assert!(k >= 1, "sumcheck requires at least one round");

        // Proof-shape checks: each must hold or the proof is malformed,
        // independent of any field arithmetic.
        if zk_data.round_coefficients.len() != k {
            return Err(SumcheckError::RoundCountMismatch {
                expected: k,
                actual: zk_data.round_coefficients.len(),
            });
        }
        if mask_commits.len() != k {
            return Err(SumcheckError::MaskCommitmentCountMismatch {
                expected: k,
                actual: mask_commits.len(),
            });
        }
        // Wire form is `[c_0, c_2, …, c_d]` of length `max(ℓ_zk, 3) - 1` =
        // `max(ℓ_zk - 1, 2)`.
        let h_size = core::cmp::max(ell_zk, 3);
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

        // === Prelude: mirror the prover's transcript actions before round 1. ===

        // Sample α and accumulate μ from the EqStatement directly. Equivalent
        // to the prover's `combine_hypercube` accumulation but without
        // materialising the weight polynomial.
        let alpha: EF = challenger.sample_algebra_element();
        let mut mu = EF::ZERO;
        statement.combine_evals(&mut mu, alpha);

        // Observe each mask commitment in the same order the prover did.
        for commit in mask_commits {
            challenger.observe(commit.clone());
        }

        // Observe μ̃ (lifted to EF — codebase convention; matches prover).
        challenger.observe_algebra_element(EF::from(zk_data.mu_tilde));

        // Sample ε.
        let eps: EF = challenger.sample_algebra_element();

        // === Sumcheck rounds ===

        // Round-1 affine target: ĥ_1(0) + ĥ_1(1) = ε·μ + μ̃.
        // For round j ≥ 2 the target gets overwritten with ĥ_{j-1}(γ_{j-1}).
        let mut target: EF = eps * mu + EF::from(zk_data.mu_tilde);
        let mut randomness: Vec<EF> = Vec::with_capacity(k);

        for (j_idx, wire) in zk_data.round_coefficients.iter().enumerate() {
            // Reconstruct the dropped `c_1` from
            //   target = ĥ_j(0) + ĥ_j(1) = 2·c_0 + c_1 + Σ_{i ≥ 2} c_i.
            let c0 = wire[0];
            let high_sum: EF = wire[1..].iter().copied().sum();
            let c1 = target - c0.double() - high_sum;

            // Observe the wire form on the transcript (same bytes the prover
            // pushed). Subsequent grind / sample reproduces the prover's flow.
            challenger.observe_algebra_slice(wire);

            // Verify PoW (only when prover grinded).
            if pow_bits > 0
                && !challenger.check_witness(pow_bits, zk_data.pow_witnesses[j_idx])
            {
                return Err(SumcheckError::InvalidPowWitness);
            }

            let gamma_j: EF = challenger.sample_algebra_element();

            // Reassemble [c_0, c_1, c_2, …, c_d] and Horner-evaluate at γ_j.
            // Iterate via chained iterators to avoid the allocation.
            let coeffs = core::iter::once(c0)
                .chain(core::iter::once(c1))
                .chain(wire[1..].iter().copied());
            // Horner needs highest-degree first; collect into a small stack
            // buffer via Vec then iterate reversed (bounded length, ≤ ell_zk).
            let mut coeffs_vec: Vec<EF> = Vec::with_capacity(h_size);
            coeffs_vec.extend(coeffs);
            let h_at_gamma_j: EF = coeffs_vec
                .iter()
                .rev()
                .copied()
                .fold(EF::ZERO, |acc, c| acc * gamma_j + c);

            target = h_at_gamma_j;
            randomness.push(gamma_j);
        }

        // After the round-k iteration `target = ĥ_k(γ_k)`, the residual claim
        // the next protocol layer (committed sumcheck relation) must verify.
        Ok((Point::new(randomness), target))
    }
}

impl<F, EF, Enc, M> ZkSumcheckProver<F, EF, Enc, M>
where
    F: Field,
    EF: ExtensionField<F>,
    Enc: ZkEncoding<F>,
    Enc::Codeword: Matrix<F>,
    M: Mmcs<F>,
{
    /// Runs one masked sumcheck round for `j ∈ 2..=k`.
    ///
    /// Computes `ĥ_j` per the per-round formula in the module docs, observes
    /// its `max(ell_zk - 1, 2)` non-linear coefficients on the transcript,
    /// grinds, samples `γ_j`, folds `(evals, weights)`, and updates the running
    /// mask bookkeeping. Returns `γ_j`.
    ///
    /// # Panics
    ///
    /// - If no rounds remain (caller must invoke at most `k-1` times after
    ///   construction).
    pub fn round<Challenger>(
        &mut self,
        zk_data: &mut ZkSumcheckData<F, EF>,
        challenger: &mut Challenger,
        pow_bits: usize,
    ) -> EF
    where
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        assert!(self.rounds_left > 0, "HVZK sumcheck has no rounds left");

        let k = self.masks.len();
        // 1-indexed round number: at the start of `round()` we have
        // `rounds_left = k - j + 1`, so `j = k - rounds_left + 1`.
        let j = k - self.rounds_left + 1;
        // 0-indexed mask slot for s_j.
        let s_j = &self.masks[j - 1];
        let ell_zk = s_j.len();

        // Snapshot ĥ_{j-1}(γ_{j-1}) for the affine-consistency debug check.
        // At this point `mask_evals_at_gamma` has length `j-1` and contains
        // `s_1(γ_1), …, s_{j-1}(γ_{j-1})`; `sum_future_endpoints` is in its
        // pre-decrement state, holding `Σ_{l ≥ j}(s_l(0)+s_l(1))`. Then
        //
        //   ĥ_{j-1}(γ_{j-1})
        //     = 2^{k-(j-1)} · (live + past mask sum at γ_<j)        // = Σ mask_evals_at_gamma
        //     + 2^{k-j}     · Σ_{l > j-1}(s_l(0)+s_l(1))            // = sum_future_endpoints
        //     + ε           · plain_h_{j-1}(γ_{j-1}).               // = plain_sum
        //
        // The future-mask term is non-empty for all `j ∈ 2..=k` because there
        // is always at least one mask with index `≥ j` (namely `s_j`).
        #[cfg(debug_assertions)]
        let prev_h_at_gamma_prev: EF = {
            let mult_live_past = F::TWO.exp_u64(self.rounds_left as u64);
            let mult_future = F::TWO.exp_u64((self.rounds_left - 1) as u64);
            let past_mask_sum: EF = self.mask_evals_at_gamma.iter().copied().sum();
            EF::from(mult_live_past) * past_mask_sum
                + EF::from(mult_future * self.sum_future_endpoints)
                + self.eps * self.plain_sum
        };

        // Start-of-round decrement: subtract `s_j`'s endpoints so the running
        // sum holds `Σ_{l > j}(s_l(0)+s_l(1))` — the future-mask term the
        // per-round formula expects. Mirrors the same step inlined for round 1
        // in `new_classic_unpacked`.
        let s_j_endpoints = s_j[0] + s_j.iter().copied().sum::<F>();
        self.sum_future_endpoints -= s_j_endpoints;

        // Plain piece (degree-2): returns (c_0, c_∞); derive c_1 from the
        // affine constraint h(0) + h(1) = plain_sum.
        let (plain_c0, plain_c_inf) = VariableOrder::Prefix
            .sumcheck_coefficients(self.evals.as_slice(), self.weights.as_slice());
        let plain_c1 = self.plain_sum - plain_c0.double() - plain_c_inf;

        // Build `ĥ_j` of length `max(ell_zk, 3)`:
        //   indices 0..ell_zk : live-mask piece           = 2^{k-j} · s_j(X)
        //   index 0           : past-mask contribution   += 2^{k-j} · Σ_{l<j} s_l(γ_l)
        //   index 0           : future-mask contribution += 2^{k-j-1} · Σ_{l>j}(s_l(0)+s_l(1))
        //   indices 0..3      : plain piece              += ε · (c_0 + c_1·X + c_∞·X²)
        // The future-mask term is only present when `j < k`.
        let h_size = core::cmp::max(ell_zk, 3);
        let mut h: Vec<EF> = vec![EF::ZERO; h_size];

        // Live mask: 2^{k-j} = 2^{rounds_left - 1}.
        let mult_live = F::TWO.exp_u64((self.rounds_left - 1) as u64);
        for (i, &c) in s_j.iter().enumerate() {
            h[i] += EF::from(mult_live * c);
        }

        // Past masks: same multiplier as live; constant in X (added to c_0).
        let past_mask_sum: EF = self.mask_evals_at_gamma.iter().copied().sum();
        h[0] += EF::from(mult_live) * past_mask_sum;

        // Future masks: 2^{k-j-1} = 2^{rounds_left - 2}; only when `j < k`.
        if j < k {
            let mult_future = F::TWO.exp_u64((self.rounds_left - 2) as u64);
            h[0] += EF::from(mult_future * self.sum_future_endpoints);
        }

        // Plain piece: ε · (c_0 + c_1·X + c_∞·X²).
        h[0] += self.eps * plain_c0;
        h[1] += self.eps * plain_c1;
        h[2] += self.eps * plain_c_inf;

        // Affine consistency check (cheap: O(ell_zk) sum):
        //   h(0) + h(1) = 2·c_0 + Σ_{i ≥ 1} c_i  must equal  ĥ_{j-1}(γ_{j-1}).
        debug_assert_eq!(
            h[0].double() + h[1..].iter().copied().sum::<EF>(),
            prev_h_at_gamma_prev,
            "ĥ_j should satisfy h(0) + h(1) = ĥ_{{j-1}}(γ_{{j-1}})",
        );

        // Wire format: send (c_0, c_2, c_3, …, c_d), skipping c_1; verifier
        // reconstructs c_1 from the affine consistency check above.
        let mut h_wire: Vec<EF> = Vec::with_capacity(h_size - 1);
        h_wire.push(h[0]);
        for i in 2..h_size {
            h_wire.push(h[i]);
        }

        challenger.observe_algebra_slice(&h_wire);
        zk_data.round_coefficients.push(h_wire);

        // Proof-of-work grind, then sample γ_j.
        if pow_bits > 0 {
            zk_data.pow_witnesses.push(challenger.grind(pow_bits));
        }
        let gamma_j: EF = challenger.sample_algebra_element();

        // Cache `s_j(γ_j)` via Horner for the past-mask term in future rounds.
        let sj_at_gamma_j: EF = s_j
            .iter()
            .rev()
            .copied()
            .fold(EF::ZERO, |acc, c| acc * gamma_j + EF::from(c));
        self.mask_evals_at_gamma.push(sj_at_gamma_j);

        // Fold both polynomials at γ_j and update the plain sum to
        // plain_h_j(γ_j) via quadratic extrapolation.
        self.evals.fix_prefix_var_mut(gamma_j);
        self.weights.fix_prefix_var_mut(gamma_j);
        self.plain_sum =
            extrapolate_01inf(plain_c0, self.plain_sum - plain_c0, plain_c_inf, gamma_j);

        // Sanity: hypercube dot-product still matches the running plain sum.
        debug_assert_eq!(
            self.evals
                .iter()
                .zip(self.weights.iter())
                .map(|(&e, &w)| e * w)
                .sum::<EF>(),
            self.plain_sum,
            "fold should preserve plain sumcheck invariant after round j",
        );

        self.rounds_left -= 1;
        gamma_j
    }

    /// Read-only access to the encoded mask oracles for downstream protocols
    /// (committed sumcheck relation; §2.4 / §5 of eprint 2026/391).
    ///
    /// Returns `(MMCS commitment, prover data)` per mask. Callers produce
    /// opening proofs by passing the prover data back into the same MMCS
    /// instance.
    pub fn mask_oracles(&self) -> &[(M::Commitment, M::ProverData<Enc::Codeword>)] {
        &self.mask_oracles
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_challenger::DuplexChallenger;
    use p3_dft::Radix2DFTSmallBatch;
    use p3_field::Field;
    use p3_field::extension::BinomialExtensionField;
    use p3_merkle_tree::MerkleTreeMmcs;
    use p3_multilinear_util::point::Point;
    use p3_multilinear_util::poly::Poly;
    use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
    use p3_zk_codes::reed_solomon::ReedSolomonZkEncoding;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;
    use crate::constraints::statement::EqStatement;

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

    /// Prover/verifier roundtrip: the verifier's Fiat-Shamir replay must agree
    /// with the prover on every `γ_j`. Any divergence in observed bytes (mask
    /// commits, μ̃, per-round wire form, PoW witness, sample order) breaks the
    /// γ-equality at the offending round and onward.
    #[test]
    fn prover_verifier_roundtrip_classic_unpacked() {
        let mut perm_rng = SmallRng::seed_from_u64(42);
        let perm = Perm::new_from_rng_128(&mut perm_rng);

        // MMCS for mask commitments (matches `pcs/tests.rs` setup).
        let merkle_hash = MyHash::new(perm.clone());
        let merkle_compress = MyCompress::new(perm.clone());
        let mmcs: MyMmcs = MyMmcs::new(merkle_hash, merkle_compress, 0);

        // Reed-Solomon ZK encoding: msg_len = ell_zk = 2 (the smallest valid
        // value per Lemma 6.4); m = 8, t = 2 keeps message + randomness ≤ m.
        let dft = MyDft::default();
        let ell_zk = 2;
        let encoding = MyEnc::new(2, ell_zk, 8, dft);

        // Random multilinear polynomial with 4 variables (16 evaluations).
        let n_vars = 4;
        let mut data_rng = SmallRng::seed_from_u64(7);
        let evals: Vec<F> = (0..(1 << n_vars)).map(|_| data_rng.random()).collect();
        let poly = Poly::new(evals);

        // EqStatement with one random point; the eval is computed against the
        // poly so the claim is consistent (otherwise μ would not match the
        // hypercube sum the prover computes internally).
        let mut eq_statement = EqStatement::initialize(n_vars);
        let eq_point: Point<EF> = Point::new(
            (0..n_vars)
                .map(|_| data_rng.random::<EF>())
                .collect::<Vec<EF>>(),
        );
        let eq_eval = poly.eval_base::<EF>(&eq_point);
        eq_statement.add_evaluated_constraint(eq_point, eq_eval);

        let folding_factor = 3;
        let pow_bits = 0;

        // === Prover ===
        // Fresh challenger seeded identically to the verifier's via cloned
        // permutation state.
        let mut prover_challenger = MyChallenger::new(perm.clone());
        let mut zk_data = ZkSumcheckData::<F, EF>::default();
        let mut prover_rng = SmallRng::seed_from_u64(11);

        let (mut prover, gamma_1_point) = ZkSumcheck::new_classic_unpacked::<F, EF, _, _, _, _>(
            &poly,
            &mut zk_data,
            &mut prover_challenger,
            folding_factor,
            pow_bits,
            &eq_statement,
            &encoding,
            &mmcs,
            &mut prover_rng,
        );

        // Collect the prover's challenges (γ_1 from the constructor, γ_2..γ_k
        // from `round()`).
        let mut prover_randomness: Vec<EF> = gamma_1_point.iter().copied().collect();
        for _ in 1..folding_factor {
            let gamma_j = prover.round(&mut zk_data, &mut prover_challenger, pow_bits);
            prover_randomness.push(gamma_j);
        }

        // Snapshot mask commitments (before the prover state goes out of scope
        // — `mask_oracles` holds the prover data which the verifier doesn't
        // need, only the commitment half).
        let mask_commits: Vec<_> = prover
            .mask_oracles()
            .iter()
            .map(|(c, _)| c.clone())
            .collect();

        // === Verifier ===
        let mut verifier_challenger = MyChallenger::new(perm);
        let (verifier_point, _final_target) =
            ZkSumcheck::verify_classic_unpacked::<F, EF, MyMmcs, _>(
                &zk_data,
                &mut verifier_challenger,
                folding_factor,
                pow_bits,
                &eq_statement,
                &mask_commits,
                ell_zk,
            )
            .expect("HVZK verifier should accept honest proof");

        // Equality of randomness vectors implies byte-level FS agreement: the
        // verifier observed the same prelude bytes (α, mask commits, μ̃, ε)
        // and the same per-round wire bytes the prover pushed.
        assert_eq!(
            prover_randomness,
            verifier_point.iter().copied().collect::<Vec<EF>>(),
            "prover and verifier disagreed on the sumcheck randomness",
        );
    }
}
