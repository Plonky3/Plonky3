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
//! landed via #1584/#1585.
//!
//! # Sumcheck handoff
//!
//! The HVZK sumcheck from #1586/#1605 currently exposes the prefix-binding
//! overlay (`ZkPrefixProver` / `ZkVerifier`). Its residual claim is already
//! scaled by the sumcheck challenge `eps`; the code-switch relation must carry
//! that scale on the inherited source claim when composing Construction 9.7
//! after Construction 6.3.

use alloc::vec::Vec;

use p3_commit::Mmcs;
use p3_field::Field;
use serde::{Deserialize, Serialize};

use crate::utils::padded_ood_t1;

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
    ///
    /// Must match `ZkSumcheckData::ell_zk`; the #1605 verifier rejects
    /// mismatches before replaying the round transcript.
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
    /// Scale already applied to the inherited residual sumcheck claim.
    ///
    /// For the #1605 HVZK sumcheck this is the sampled `eps`. The value is
    /// tracked separately from `base_claim_coeff` so the output relation can
    /// distinguish Construction 9.7 batching randomness from Construction 6.3
    /// masking randomness.
    pub residual_sumcheck_scale: EF,
    /// Batching coefficients for OOD answers (`nu_{1+i}` for `i in [t_ood]`).
    pub ood_coeffs: Vec<EF>,
    /// Batching coefficients for in-domain openings.
    pub in_domain_coeffs: Vec<EF>,
    /// `G^$_C` rows (source randomness part) weighted by batching coefficients.
    pub source_randomness_weights: Vec<EF>,
    /// Zero-padded mask weights for `s_pad` portion.
    pub pad_weights: Vec<EF>,
}

/// Returns `(1, rho, rho^2, ..., rho^{dim - 1})`.
///
/// Construction 9.7 uses these coefficients to batch the inherited claim,
/// private OOD answers, and in-domain source openings into a single output
/// relation.
#[inline]
#[must_use]
#[allow(dead_code)]
pub(crate) fn batching_coefficients<EF: Field>(rho: EF, dim: usize) -> Vec<EF> {
    let mut coeffs = Vec::with_capacity(dim);
    let mut power = EF::ONE;
    for _ in 0..dim {
        coeffs.push(power);
        power *= rho;
    }
    coeffs
}

/// Computes the private OOD answer from Construction 9.7.
///
/// This is `ze_ood(rho) · (f, r, s_pad)^T`, represented as a padded
/// zero-evader evaluation of `(source_message || mask_message)`.
#[inline]
#[allow(dead_code)]
pub(crate) fn private_ood_answer<EF: Field>(
    rho: EF,
    source_message: &[EF],
    mask_message: &[EF],
) -> EF {
    padded_ood_t1(rho, source_message, mask_message)
}

/// Computes the verifier-side batched claim `mu'`.
///
/// The inherited claim is passed in before the #1605 sumcheck scale is
/// applied. `claim.residual_sumcheck_scale` is therefore part of the first
/// term:
///
/// ```text
/// mu' = nu_1 * eps * mu
///     + sum_i nu_{1+i} * y_i
///     + sum_j nu_{1+t_ood+j} * f(x_j)
/// ```
///
/// The caller is responsible for using the same coefficient order when
/// constructing the output covectors for `f`, previous mask messages, and the
/// fresh code-switch mask.
#[inline]
#[allow(dead_code)]
pub(crate) fn batched_claim<EF: Field>(
    inherited_claim: EF,
    private_ood_answers: &[EF],
    source_openings: &[EF],
    claim: &ZkMaskClaim<EF>,
) -> EF {
    assert_eq!(
        private_ood_answers.len(),
        claim.ood_coeffs.len(),
        "private OOD answer count must match batching coefficients",
    );
    assert_eq!(
        source_openings.len(),
        claim.in_domain_coeffs.len(),
        "source opening count must match batching coefficients",
    );

    let ood_sum: EF = claim
        .ood_coeffs
        .iter()
        .zip(private_ood_answers)
        .map(|(&coeff, &answer)| coeff * answer)
        .sum();
    let in_domain_sum: EF = claim
        .in_domain_coeffs
        .iter()
        .zip(source_openings)
        .map(|(&coeff, &opening)| coeff * opening)
        .sum();

    claim.base_claim_coeff * claim.residual_sumcheck_scale * inherited_claim
        + ood_sum
        + in_domain_sum
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
