//! HVZK variant of the WHIR sumcheck (Construction 6.3 of eprint 2026/391).
//!
//! Companion to [`super::single`]: same reduction, but with `k` random
//! univariate masks committed under a ZK encoding so that the prover's `k`
//! round-polynomials no longer leak linear functions of the secret message.
//!
//! ⚠️ This file is a **scaffold**. Type shape, decision block, and constructor
//! signatures are stable; method bodies are stubbed with `unimplemented!()`
//! and land in subsequent commits on PR #1605.
//!
//! See `paper-db-2026-391/21-section-6-zk-sumcheck.md` for the protocol spec
//! and `22-section-6.1-zk.md` for the HVZK proof sketch.
//!
//! # Decision block (frozen for this PR)
//!
//! 1. **Architecture** — `pub struct ZkSumcheck;` (namespace, mirrors
//!    [`super::single::SingleSumcheck`]) plus
//!    [`ZkSumcheckProver`]`<F, EF, Enc>` (state, mirrors
//!    [`super::strategy::SumcheckProver`] plus mask bookkeeping). Considered
//!    alternative: WizardOfMenlo/whir#241 used a runtime `mask_length: usize`
//!    flag on the existing `Config` and branched at runtime; we picked the
//!    type-split to match Plonky3's existing namespace+state pattern and keep
//!    `single.rs` byte-frozen.
//!
//! 2. **MVP scope** — only [`ZkSumcheck::new_classic_unpacked`] in this PR.
//!    SIMD-packed (`new_classic_packed`), SVO (`new_svo`), and the dispatcher
//!    (`new`) are follow-up PRs that hang off the same `ZkSumcheck` namespace
//!    and reuse `ZkSumcheckProver`. SVO is structurally compatible: only the
//!    `(c0, c_inf)` computation of the plain piece differs.
//!
//! 3. **Transcript ordering** — observe each mask oracle in turn; observe
//!    `μ̃`; sample `ε`; per round j: observe `ĥ_j` (`ℓ_zk - 1` field
//!    elements; see #4); grind; sample `γ_j`. The protocol pins the relative
//!    order of prover-sends and verifier-samples (Construction 6.3, with
//!    HVZK forcing `μ̃` before `ε` and RBR forcing `ĥ_j` before `γ_j`); the
//!    per-message observe granularity is a Fiat-Shamir realization choice
//!    and we pick granular for debuggability.
//!
//! 4. **Wire format — skip the linear coefficient.** Per round, send
//!    `(c0, c2, c3, …, c_{ℓ_zk-1})`; verifier derives `c1` from the affine
//!    consistency check `ĥ_j(0) + ĥ_j(1) = ε·μ + μ̃` (round 1) or
//!    `= ĥ_{j-1}(γ_{j-1})` (subsequent rounds). Matches Lemma 6.4's affine
//!    subspace dimension `1 + k·(ℓ_zk - 1)` and mirrors the existing
//!    plain-path convention in [`super::single`].
//!
//! 5. **Field constraints** — `char(F) ≠ 2` (Lemma 6.4 rank-nullity argument)
//!    and `ℓ_zk ≥ 2`. Enforced via runtime `assert!` in the constructor for
//!    now; promote to compile-time when/if `Field` exposes characteristic as
//!    a `const`.
//!
//! 6. **Round-loop sharing** — the per-round arithmetic is duplicated from
//!    `super::single` in this PR (≈20 LoC). Refactor into a shared
//!    `sumcheck::core` helper in a follow-up PR. Rationale: keeps `single.rs`
//!    byte-frozen for free, narrows review surface, defers the cross-cutting
//!    refactor to when both paths are stable.
//!
//! # Per-round formula (Construction 6.3 step 4(a))
//!
//! For round `j` (1-indexed), with `γ = (γ_1, …, γ_{j-1})` already sampled:
//!
//! ```text
//! ĥ_j(X) = 2^{k-j}   · s_j(X)                                  (live mask)
//!        + 2^{k-j}   · Σ_{l < j} s_l(γ_l)                      (past masks, cached)
//!        + 2^{k-j-1} · sum_future_endpoints                    (future masks, running)
//!        + ε         · plain_piece(X)                          (base sumcheck round)
//! ```
//!
//! After observing `ĥ_j` (minus c1) and sampling `γ_j` the prover:
//! - pushes `s_j(γ_j)` onto `mask_evals_at_gamma`,
//! - decrements `sum_future_endpoints -= s_{j+1}(0) + s_{j+1}(1)` (if a next round exists),
//! - calls `base.fix_prefix_var_mut(γ_j)`.
//!
//! The closed-form `μ̃ = 2^{k-1} · sum_future_endpoints_initial` is checked
//! against the naive `Σ_{b ∈ {0,1}^k} ŝ(b)` form via `debug_assert!` in the
//! constructor (catches the multiplicity bug class that the reference impl
//! shipped at first; see `_search_log.md`).
//!
//! # References
//!
//! - eprint 2026/391, §6 Construction 6.3, Lemma 6.4 (HVZK), Lemma 6.5 (RBR).
//! - Local digest: `paper-db-2026-391/21-section-6-zk-sumcheck.md`,
//!   `22-section-6.1-zk.md`, `23-section-6.2-rbr.md`.
//! - Reference impl: WizardOfMenlo/whir#241 (merged 2026-03-31).
//! - Issue: Plonky3/Plonky3#1586 (HVZK-WHIR 3/6).
//! - Depends on: #1584/#1601 (ZK encoding traits in `p3-zk-codes`).
//! - Tracked under: #1590 (HVZK-WHIR effort).

use alloc::vec::Vec;

use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, Field};
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use p3_zk_codes::ZkEncoding;
use rand::Rng;

use crate::constraints::statement::EqStatement;
use crate::sumcheck::SumcheckData;
use crate::sumcheck::strategy::SumcheckProver;

/// Namespace for the HVZK variant of the WHIR sumcheck.
///
/// Mirrors [`super::single::SingleSumcheck`]: a unit struct hosting the static
/// constructors for each sumcheck strategy. In this PR only
/// [`Self::new_classic_unpacked`] is exposed; the SIMD-packed and SVO
/// variants land in follow-up PRs and will hang off the same namespace.
pub struct ZkSumcheck;

/// Stateful prover for the HVZK sumcheck (Construction 6.3).
///
/// Mirrors [`SumcheckProver`] (the plain prover state) and adds the mask
/// bookkeeping required by the construction. The plain prover handles the
/// witness-side polynomial fold exactly as in the non-ZK path; everything
/// mask-related lives here.
//
// Fields are populated by stubs that aren't implemented yet; the
// `dead_code` allow lifts in commit 2/3 of PR #1605 once the prelude and
// round logic actually read them.
#[allow(dead_code)]
pub struct ZkSumcheckProver<F, EF, Enc>
where
    F: Field,
    EF: ExtensionField<F>,
    Enc: ZkEncoding<F>,
{
    /// Plain sumcheck state (poly + claimed sum). Folded at each `γ_j`
    /// exactly like the non-ZK path — fold logic is unchanged.
    base: SumcheckProver<F, EF>,
    /// ZK encoding used for the masks.
    encoding: Enc,
    /// `k` mask polynomials `s_1, …, s_k` as coefficient vectors of length
    /// `ell_zk`.
    masks: Vec<Vec<F>>,
    /// Encoded mask oracles. Kept in state and exposed via
    /// [`Self::mask_oracles`] so downstream protocols (committed sumcheck
    /// relation, §2.4 / §5 of the paper) can consume them.
    mask_oracles: Vec<Enc::Codeword>,
    /// Combination challenge `ε` sampled after observing `μ̃`. Used in every
    /// subsequent round to scale the plain piece.
    eps: EF,
    /// Running scalar `Σ_{l > current_round} (s_l(0) + s_l(1))`.
    /// Precomputed at construction (`μ̃ = 2^{k-1} · sum_future_endpoints`
    /// initially); decremented per round as masks transition from "future"
    /// to "past".
    sum_future_endpoints: F,
    /// `s_l(γ_l)` for `l < current_round`, accumulated as rounds progress.
    mask_evals_at_gamma: Vec<F>,
    /// Number of rounds remaining (`k` initially, decremented per round).
    rounds_left: usize,
    /// Mask coefficient length `ℓ_zk`. Wire-format degree of `ĥ_j` is
    /// `ℓ_zk - 1` (we send `ℓ_zk - 1` coefficients per round, skipping the
    /// linear one — see decision block item 4).
    ell_zk: usize,
}

impl ZkSumcheck {
    /// HVZK sumcheck via the classic unpacked (scalar) strategy.
    ///
    /// Mirrors [`super::single::SingleSumcheck::new_classic_unpacked`] in
    /// shape. Runs steps 1–3 of Construction 6.3 (sample masks, observe mask
    /// oracles, send `μ̃`, sample `ε`) plus the first masked sumcheck round,
    /// returning the prover state and the first verifier challenge `γ_1`.
    ///
    /// # Panics
    ///
    /// - If `char(F) == 2` (Lemma 6.4 requires `char(F) ≠ 2`).
    /// - If `encoding.message_len() < 2` (Lemma 6.4 requires `ℓ_zk ≥ 2`).
    /// - All assertions inherited from
    ///   [`super::single::SingleSumcheck::new_classic_unpacked`].
    #[allow(clippy::too_many_arguments)]
    pub fn new_classic_unpacked<F, EF, Enc, Challenger, R>(
        _poly: &Poly<F>,
        _sumcheck_data: &mut SumcheckData<F, EF>,
        _challenger: &mut Challenger,
        _folding_factor: usize,
        _pow_bits: usize,
        _statement: &EqStatement<EF>,
        _encoding: &Enc,
        _rng: &mut R,
    ) -> (ZkSumcheckProver<F, EF, Enc>, Point<EF>)
    where
        F: Field,
        EF: ExtensionField<F>,
        Enc: ZkEncoding<F>,
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
        R: Rng,
    {
        // TODO(commit 2): runtime asserts (char(F) != 2, ell_zk >= 2).
        // TODO(commit 2): Step 1 — sample k masks, encode each via encoding.encode,
        //                  observe each mask oracle on the challenger.
        // TODO(commit 2): Step 2 — compute μ̃ via closed form
        //                  μ̃ = 2^{k-1} · Σ_l (s_l(0) + s_l(1));
        //                  debug_assert against naive Σ_{b ∈ {0,1}^k} ŝ(b);
        //                  observe μ̃.
        // TODO(commit 2): Step 3 — sample ε from challenger.
        // TODO(commit 3): Step 4 round 1 — build ĥ_1 (mask piece + ε·plain_piece);
        //                  serialize as (c0, c2, …, c_{ℓ_zk-1}); observe; grind;
        //                  sample γ_1; fold base; update bookkeeping.
        unimplemented!("commits 2–3 of PR #1605: prelude + first masked round")
    }

    // TODO(follow-up commits): verifier counterpart (`verify_classic_unpacked`)
    // and HVZK simulator (`simulate_classic_unpacked`).
}

impl<F, EF, Enc> ZkSumcheckProver<F, EF, Enc>
where
    F: Field,
    EF: ExtensionField<F>,
    Enc: ZkEncoding<F>,
{
    /// Runs one masked sumcheck round (rounds `2..=k`).
    ///
    /// Computes `ĥ_j` per the per-round formula in this module's docstring,
    /// sends the `ell_zk - 1` non-linear coefficients on the transcript,
    /// samples `γ_j`, folds the base prover, and updates the running mask
    /// bookkeeping.
    pub fn round<Challenger>(
        &mut self,
        _sumcheck_data: &mut SumcheckData<F, EF>,
        _challenger: &mut Challenger,
        _sum: &mut EF,
        _pow_bits: usize,
    ) -> EF
    where
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        // TODO(commit 3): per-round arithmetic + bookkeeping update.
        unimplemented!("commit 3 of PR #1605: round() impl")
    }

    /// Read-only access to the encoded mask oracles, for downstream protocols
    /// (committed sumcheck relation; see §2.4 / §5 of eprint 2026/391).
    pub fn mask_oracles(&self) -> &[Enc::Codeword] {
        &self.mask_oracles
    }
}
