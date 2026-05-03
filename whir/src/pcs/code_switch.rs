//! HVZK code-switching round (Construction 9.7, eprint 2026/391 Section 9.4).
//!
//! Reduces a proximity claim about oracle `f` w.r.t. source code `C` to a
//! proximity claim about oracle `g` w.r.t. a smaller target code `C'`.
//!
//! The ZK variant adds:
//! 1. A fresh mask oracle `s = Enc_{C_zk}((r, s_pad), r'')`.
//! 2. A private-zero-evader OOD answer `y = ze_ood(rho) · (f, r, s_pad)^T`.
//!
//! The type scaffolding and math identity tests use `p3-zk-codes`
//! (`LinearZkEncoding`) and `whir::utils` (zero-evader helpers) which
//! landed via #1584/#1585. Full prover/verifier wiring awaits #1586
//! (HVZK sumcheck with sublinear masks).

use alloc::vec::Vec;

use p3_commit::Mmcs;
use serde::{Deserialize, Serialize};

/// ZK-specific per-round configuration, derived from protocol parameters.
///
/// Nested inside `RoundConfig::zk` as `Option<RoundZkConfig<F>>`.
/// When `None`, the round uses the non-ZK path with no transcript changes.
///
/// The issue asks for `zk: bool`, but a boolean alone cannot drive the
/// simulator query bounds or proof-size accounting. This struct carries
/// all derived dimensions so they stay explicit.
#[derive(Debug, Clone)]
pub struct RoundZkConfig<F> {
    /// Number of target-oracle queries the simulator may make.
    pub target_query_budget: usize,
    /// Number of mask-oracle queries the simulator may make.
    pub mask_query_budget: usize,
    /// Message length of the mask code (`ell_zk`).
    pub mask_message_len: usize,
    /// Randomness length of the mask code.
    pub mask_randomness_len: usize,
    /// Number of private OOD samples (`t_ood`).
    pub ood_samples: usize,
    /// Evaluation domain size for the mask code.
    pub mask_domain_size: usize,
    /// Interleaving width of the mask code (`iota_zk`).
    pub mask_width: usize,
    /// Generator of the folded mask evaluation domain.
    pub folded_mask_domain_gen: F,
}

/// Per-round ZK mask coefficient carrier.
///
/// Produced by the batching step (both prover and verifier) and consumed
/// by the next ZK sumcheck relation. Prevents the covector/weight mismatch
/// bug found in WizardOfMenlo/whir#249.
#[derive(Debug, Clone)]
pub struct ZkMaskClaim<EF> {
    /// Coefficient on the inherited source claim (`nu_1`).
    pub base_claim_coeff: EF,
    /// Batching coefficients for OOD answers (`nu_{1+i}` for `i in [t_ood]`).
    pub ood_coeffs: Vec<EF>,
    /// Batching coefficients for in-domain openings.
    pub in_domain_coeffs: Vec<EF>,
    /// `G^$_C` rows (source randomness part) weighted by batching coefficients.
    pub source_randomness_weights: Vec<EF>,
    /// Zero-padded mask weights for `s_pad` portion.
    pub pad_weights: Vec<EF>,
}

/// Additional per-round proof data for the ZK code-switching path.
///
/// Intended to be held as `Option<WhirRoundZkProof<...>>` inside
/// `WhirRoundProof` once the prover/verifier round flow is wired.
///
/// Keeping this separate from `WhirRoundProof` for now lets the standalone
/// math tests define the ZK proof shape without changing non-ZK serialization.
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(bound(
    serialize = "EF: Serialize, MT::Commitment: Serialize, MT::Proof: Serialize",
    deserialize = "EF: Deserialize<'de>, MT::Commitment: Deserialize<'de>, MT::Proof: Deserialize<'de>"
))]
pub struct WhirRoundZkProof<F: Send + Sync + Clone, EF, MT: Mmcs<F>> {
    /// Mask oracle commitment (Merkle root for `s = Enc_{C_zk}((r, s_pad), r'')`).
    pub mask_commitment: MT::Commitment,
    /// Private OOD answers: `y = ze_ood(rho_ood) * [f; r; s_pad]`.
    ///
    /// Distinct from `WhirRoundProof::ood_answers` which are public
    /// folded-polynomial evaluations.
    pub private_ood_answers: Vec<EF>,
}

#[cfg(test)]
mod tests;
