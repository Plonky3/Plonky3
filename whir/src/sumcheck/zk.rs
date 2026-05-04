//! HVZK variant of the WHIR sumcheck (Construction 6.3 of eprint 2026/391).
//!
//! Companion to [`super::single`]: same reduction, but with `k` random
//! univariate masks committed under a ZK encoding so that the prover's k
//! round-polynomials no longer leak linear functions of the secret message.
//!
//! ⚠️ This file is currently a SCAFFOLD. The implementation work is the body
//! of issue #1586 (HVZK-WHIR 3/6).
//!
//! # What this module does (once implemented)
//!
//! WHIR's plain sumcheck reduces a claim about a 2^k-interleaved codeword to a
//! claim about the base codeword by sending k univariate round-polynomials
//! `ĥ_1, …, ĥ_k`. Each `ĥ_j` is a linear function of the secret message `f`,
//! so it leaks information.
//!
//! Construction 6.3 fixes this by:
//!
//! 1. Sampling k random univariate masks `s_j ∈ F^{<ℓ_zk}[X]`, encoded under a
//!    ZK encoding (`p3_zk_codes::ZkEncoding`, see #1584 / #1601), and sending
//!    each `s_j` as an oracle in round 1.
//! 2. Sending `μ̃ := Σ_{b ∈ {0,1}^k} (ŝ_1(b_1) + … + ŝ_k(b_k))` as the new
//!    "noise target".
//! 3. The verifier picks one combination challenge `ε ← F`.
//! 4. Running k sumcheck rounds on the masked claim
//!    `Σ_b (ŝ(b) + ε · Ĝ(b)) = μ̃ + ε · μ`, where `Ĝ` is the original
//!    sumcheck target. Each `ĥ_j` is now (mask piece from `s_j`) plus
//!    `ε`-times the original sumcheck contribution.
//!
//! Total prover-to-verifier overhead is `Õ(k · λ)` field elements,
//! independent of the witness size — this is what gives the paper its
//! `1 + o(1)` zero-knowledge overhead.
//!
//! # Why the simulator works (Lemma 6.4)
//!
//! Fix verifier randomness `(ε, γ)`. The verifier's k+1 affine consistency
//! checks
//!
//! ```text
//! ĥ_1(0) + ĥ_1(1) = ε · μ + μ̃,
//! ĥ_j(0) + ĥ_j(1) = ĥ_{j-1}(γ_{j-1})    for 2 ≤ j ≤ k,
//! ```
//!
//! cut out an affine subspace `T ⊆ F^{1 + k·ℓ_zk}` of dimension
//! `1 + k·(ℓ_zk - 1)`. The honest-prover formulas define an affine map
//! `A : (F^{<ℓ_zk}[X])^k → F^{1 + k·ℓ_zk}` whose image is exactly `T`
//! (rank-nullity argument; needs `char(F) ≠ 2`). Hence uniform masks induce
//! the uniform distribution over `T`, which the simulator samples directly
//! without ever touching `f`.
//!
//! # Constraints
//!
//! - `char(F) ≠ 2` (Lemma 6.4 proof). KoalaBear, BabyBear satisfy this.
//! - `ℓ_zk ≥ 2` (Lemma 6.4).
//! - The non-ZK path in [`super::single`] must remain byte-identical when this
//!   variant is gated off. Achieve via a const-generic flag on the existing
//!   sumcheck types.
//!
//! # API guideline (from the issue, not binding)
//!
//! Naming note: issue #1586 spells the embedded prover as `WhirSumcheckProver<F, EF>`,
//! which is a naming slip — that type doesn't exist. The actual stateful prover
//! returned by `SingleSumcheck::new_*` is [`super::strategy::SumcheckProver`],
//! and that is what we compose here.
//!
//! ```ignore
//! pub struct ZkSumcheckProver<F, EF, Enc>
//! where
//!     F: Field,
//!     EF: ExtensionField<F>,
//!     Enc: p3_zk_codes::ZkEncoding<F>,
//! {
//!     base: super::strategy::SumcheckProver<F, EF>, // poly + claimed sum
//!     encoding: Enc,
//!     masks: Vec<UnivariatePolynomial<F>>,          // s_1, …, s_k
//!     mask_oracles: Vec<Enc::Codeword>,             // Enc_{C_zk}(s_j)
//! }
//!
//! impl<F, EF, Enc> ZkSumcheckProver<F, EF, Enc>
//! where
//!     F: Field,
//!     EF: ExtensionField<F>,
//!     Enc: p3_zk_codes::ZkEncoding<F>,
//! {
//!     /// Step 1 of Construction 6.3: sample k masks and commit them.
//!     pub fn commit_masks<R: rand::Rng>(&mut self, rng: &mut R);
//!
//!     /// Step 2: compute and send μ̃ = Σ_b ŝ(b).
//!     pub fn send_target_mu_tilde(&self) -> EF;
//!
//!     /// Step 4 (one round): produce ĥ_j after receiving (γ_{j-1}, ε).
//!     pub fn round(&mut self, gamma: EF, eps: EF) -> /* UnivariatePoly */ ();
//! }
//! ```
//!
//! Merging `commit_masks` and `send_target_mu_tilde` into one call is fine
//! if it makes the transcript wiring cleaner — flag it in the PR.
//!
//! # Implementation TODOs
//!
//! - [x] Add `p3-zk-codes` to `whir/Cargo.toml`.
//! - [ ] Decide const-generic gating on `super::single` types so non-ZK
//!       transcript stays byte-identical (regression test required).
//! - [ ] Add a `static_assert!` (or const evaluator) enforcing
//!       `char(F) ≠ 2` and `ℓ_zk ≥ 2` at type-construction time.
//! - [ ] Implement Step 1 (sample + encode masks). Mask coefficient layout:
//!       `s_j` as `Vec<F>` of length `ℓ_zk`. Encoding via `Enc::encode`.
//! - [ ] Implement Step 2 (compute `μ̃`). Closed form: for separable masks
//!       `ŝ(b) = Σ_j ŝ_j(b_j)`, `Σ_{b ∈ {0,1}^k} ŝ(b) = 2^{k-1} · Σ_j (ŝ_j(0) + ŝ_j(1))`.
//! - [ ] Implement Step 3 (sample ε from challenger after observing masks
//!       and `μ̃`).
//! - [ ] Implement Step 4 (k masked sumcheck rounds). For each round j,
//!       construct `ĥ_j(X) ∈ F^{<max{2, ℓ_zk}}[X]` as defined in the
//!       paper. Reuse `super::single` machinery for the `Ĝ`-piece.
//! - [ ] Verifier path: check `ĥ_1(0) + ĥ_1(1) = ε·μ + μ̃` and
//!       `ĥ_j(0) + ĥ_j(1) = ĥ_{j-1}(γ_{j-1})`.
//! - [ ] Output: same as `super::single` plus the mask oracles for
//!       downstream protocols (committed sumcheck relation; see §2.4 / §5
//!       of the paper).
//! - [ ] Simulator (Lemma 6.4): sample `(μ̃, ĥ_1, …, ĥ_k)` uniformly
//!       subject to the verifier's k+1 affine constraints. For RS
//!       encoding the encoding error is 0, so the simulator must match
//!       the real view *exactly* — that's what the proptest checks.
//!
//! # Acceptance criteria (from the issue)
//!
//! - proptest: completeness across many `(k, ℓ_zk)` configurations.
//! - proptest: HVZK simulator (Lemma 6.4) matches the real view exactly for
//!   RS-based encoding (error 0). Use `Enc::simulate` from #1584 for
//!   queries to mask oracles.
//! - proptest: empirical RBR soundness (Lemma 6.5) over a small field.
//! - regression test: with ZK gated off, transcript is byte-identical to
//!   today's sumcheck.
//! - criterion bench: ZK overhead `≤ 1.05×` for `(2^20 rows, k = 4, λ = 100)`.
//!
//! # References
//!
//! - eprint 2026/391, §6 Construction 6.3, Lemma 6.4 (HVZK), Lemma 6.5 (RBR).
//! - Local digest: `paper-db-2026-391/21-section-6-zk-sumcheck.md`,
//!   `22-section-6.1-zk.md`, `23-section-6.2-rbr.md`.
//! - Issue: Plonky3/Plonky3#1586 (HVZK-WHIR 3/6).
//! - Depends on: #1584 (ZK encoding traits, now in `p3-zk-codes`).
//! - Tracked under: #1590 (HVZK-WHIR effort).
