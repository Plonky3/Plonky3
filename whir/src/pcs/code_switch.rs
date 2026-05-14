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
//!
//! The suffix-binding WHIR path must remain non-ZK until #1649 lands a
//! corresponding `ZkSuffixProver`; silently routing a ZK proof through suffix
//! mode would produce a non-private transcript.

use alloc::vec;
use alloc::vec::Vec;

use p3_commit::Mmcs;
use p3_field::Field;
use p3_zk_codes::LinearZkEncoding;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::utils::padded_ood_t1;

/// Errors in the standalone Construction 9.7 code-switching reduction.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum CodeSwitchError {
    /// The number of private OOD answers does not match the OOD batching coefficients.
    #[error("private OOD answer count mismatch: expected {expected}, got {actual}")]
    PrivateOodAnswerCountMismatch { expected: usize, actual: usize },

    /// The number of source openings does not match the in-domain batching coefficients.
    #[error("source opening count mismatch: expected {expected}, got {actual}")]
    SourceOpeningCountMismatch { expected: usize, actual: usize },

    /// The number of OOD points does not match the OOD batching coefficients.
    #[error("OOD point count mismatch: expected {expected}, got {actual}")]
    OodPointCountMismatch { expected: usize, actual: usize },

    /// The number of queried source positions does not match the in-domain batching coefficients.
    #[error("source query position count mismatch: expected {expected}, got {actual}")]
    QueryPositionCountMismatch { expected: usize, actual: usize },

    /// The inherited source covector has the wrong length.
    #[error("source covector length mismatch: expected {expected}, got {actual}")]
    SourceCovectorLengthMismatch { expected: usize, actual: usize },

    /// The supplied source message has the wrong length for an output relation.
    #[error("source message length mismatch: expected {expected}, got {actual}")]
    SourceMessageLengthMismatch { expected: usize, actual: usize },

    /// The fresh mask message `(r, s_pad)` has the wrong length.
    #[error("mask message length mismatch: expected {expected}, got {actual}")]
    MaskMessageLengthMismatch { expected: usize, actual: usize },

    /// The number of auxiliary witnesses does not match the carried auxiliary covectors.
    #[error("auxiliary witness count mismatch: expected {expected}, got {actual}")]
    AuxiliaryWitnessCountMismatch { expected: usize, actual: usize },

    /// An auxiliary witness has the wrong length for its carried covector.
    #[error("auxiliary witness {index} length mismatch: expected {expected}, got {actual}")]
    AuxiliaryWitnessLengthMismatch {
        index: usize,
        expected: usize,
        actual: usize,
    },

    /// A generator row from the source code has the wrong message length.
    #[error("source-code message row length mismatch: expected {expected}, got {actual}")]
    SourceMessageRowLengthMismatch { expected: usize, actual: usize },

    /// A generator row from the source code has the wrong randomness length.
    #[error("source-code randomness row length mismatch: expected {expected}, got {actual}")]
    SourceRandomnessRowLengthMismatch { expected: usize, actual: usize },
}

/// ZK-specific per-round configuration, derived from protocol parameters.
///
/// Nested inside `RoundConfig::zk` as `Option<RoundZkConfig<F>>`.
/// When `None`, the round uses the non-ZK path with no transcript changes.
///
/// The issue asks for `zk: bool`, but a boolean alone cannot drive the
/// simulator query bounds or proof-size accounting. This struct carries
/// all derived dimensions so they stay explicit.
///
/// This is intentionally configuration-only for now. The non-ZK proof shape
/// must stay byte-compatible with today's WHIR proofs; ZK-only transcript data
/// belongs behind `Option` fields on the round proof once the round flow is
/// wired.
#[derive(Debug, Clone)]
pub struct RoundZkConfig<F> {
    /// Number of target-oracle queries the simulator may make.
    pub target_query_budget: usize,
    /// Number of mask-oracle queries the simulator may make.
    ///
    /// This is a composed-protocol budget, not a benchmark placeholder. It
    /// must cover every opening the next IOR makes to the fresh mask oracle.
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

/// Output linear relation produced by Construction 9.7.
///
/// The relation is evaluated as:
///
/// ```text
/// <f, source_covector>
///   + sum_i <aux_i, auxiliary_covectors_i>
///   + <(r, s_pad), mask_covector>
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodeSwitchOutputRelation<EF> {
    /// Covector for the source message `f`.
    pub source_covector: Vec<EF>,
    /// Carried covectors for auxiliary mask witnesses from prior rounds.
    pub auxiliary_covectors: Vec<Vec<EF>>,
    /// Covector for the fresh mask message `(r, s_pad)`.
    pub mask_covector: Vec<EF>,
}

impl<EF: Field> CodeSwitchOutputRelation<EF> {
    /// Evaluate the output relation on concrete witnesses.
    pub fn evaluate(
        &self,
        source_message: &[EF],
        auxiliary_witnesses: &[&[EF]],
        mask_message: &[EF],
    ) -> Result<EF, CodeSwitchError> {
        if source_message.len() != self.source_covector.len() {
            return Err(CodeSwitchError::SourceMessageLengthMismatch {
                expected: self.source_covector.len(),
                actual: source_message.len(),
            });
        }
        if auxiliary_witnesses.len() != self.auxiliary_covectors.len() {
            return Err(CodeSwitchError::AuxiliaryWitnessCountMismatch {
                expected: self.auxiliary_covectors.len(),
                actual: auxiliary_witnesses.len(),
            });
        }
        if mask_message.len() != self.mask_covector.len() {
            return Err(CodeSwitchError::MaskMessageLengthMismatch {
                expected: self.mask_covector.len(),
                actual: mask_message.len(),
            });
        }

        let mut value = inner_product(source_message, &self.source_covector);
        for (index, (witness, covector)) in auxiliary_witnesses
            .iter()
            .zip(&self.auxiliary_covectors)
            .enumerate()
        {
            if witness.len() != covector.len() {
                return Err(CodeSwitchError::AuxiliaryWitnessLengthMismatch {
                    index,
                    expected: covector.len(),
                    actual: witness.len(),
                });
            }
            value += inner_product(witness, covector);
        }
        value += inner_product(mask_message, &self.mask_covector);

        Ok(value)
    }
}

#[inline]
fn inner_product<EF: Field>(a: &[EF], b: &[EF]) -> EF {
    a.iter().zip(b).map(|(&x, &y)| x * y).sum()
}

/// Returns `(1, rho, rho^2, ..., rho^{dim - 1})`.
///
/// Construction 9.7 uses these coefficients to batch the inherited claim,
/// private OOD answers, and in-domain source openings into a single output
/// relation.
#[must_use]
pub fn batching_coefficients<EF: Field>(rho: EF, dim: usize) -> Vec<EF> {
    let mut coeffs = Vec::with_capacity(dim);
    let mut power = EF::ONE;
    for _ in 0..dim {
        coeffs.push(power);
        power *= rho;
    }
    coeffs
}

/// Computes one private OOD answer from Construction 9.7.
///
/// This is `ze_ood(rho) · (f, r, s_pad)^T`, represented as a padded
/// zero-evader evaluation of `(source_message || mask_message)`.
#[inline]
pub fn private_ood_answer<EF: Field>(rho: EF, source_message: &[EF], mask_message: &[EF]) -> EF {
    padded_ood_t1(rho, source_message, mask_message)
}

/// Computes all private OOD answers for one code-switching round.
#[must_use]
pub fn private_ood_answers<EF: Field>(
    rho_ood_points: &[EF],
    source_message: &[EF],
    mask_message: &[EF],
) -> Vec<EF> {
    rho_ood_points
        .iter()
        .map(|&rho| private_ood_answer(rho, source_message, mask_message))
        .collect()
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
pub fn batched_claim<EF: Field>(
    inherited_claim: EF,
    private_ood_answers: &[EF],
    source_openings: &[EF],
    claim: &ZkMaskClaim<EF>,
) -> Result<EF, CodeSwitchError> {
    if private_ood_answers.len() != claim.ood_coeffs.len() {
        return Err(CodeSwitchError::PrivateOodAnswerCountMismatch {
            expected: claim.ood_coeffs.len(),
            actual: private_ood_answers.len(),
        });
    }
    if source_openings.len() != claim.in_domain_coeffs.len() {
        return Err(CodeSwitchError::SourceOpeningCountMismatch {
            expected: claim.in_domain_coeffs.len(),
            actual: source_openings.len(),
        });
    }

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

    Ok(
        claim.base_claim_coeff * claim.residual_sumcheck_scale * inherited_claim
            + ood_sum
            + in_domain_sum,
    )
}

/// Builds the output relation covectors for Construction 9.7.
///
/// `query_positions` are flattened codeword positions. For interleaving depth
/// `iota`, callers should pass `iota * x_i + limb` for every queried limb.
#[allow(clippy::too_many_arguments)]
pub fn output_relation<F, EF, Enc>(
    source_encoding: &Enc,
    source_covector: &[EF],
    auxiliary_covectors: &[&[EF]],
    source_randomness_len: usize,
    pad_len: usize,
    rho_ood_points: &[EF],
    query_positions: &[usize],
    claim: &ZkMaskClaim<EF>,
) -> Result<CodeSwitchOutputRelation<EF>, CodeSwitchError>
where
    F: Field,
    EF: Field + From<F>,
    Enc: LinearZkEncoding<F>,
{
    if rho_ood_points.len() != claim.ood_coeffs.len() {
        return Err(CodeSwitchError::OodPointCountMismatch {
            expected: claim.ood_coeffs.len(),
            actual: rho_ood_points.len(),
        });
    }
    if query_positions.len() != claim.in_domain_coeffs.len() {
        return Err(CodeSwitchError::QueryPositionCountMismatch {
            expected: claim.in_domain_coeffs.len(),
            actual: query_positions.len(),
        });
    }

    let source_len = source_encoding.message_len();
    if source_covector.len() != source_len {
        return Err(CodeSwitchError::SourceCovectorLengthMismatch {
            expected: source_len,
            actual: source_covector.len(),
        });
    }

    let mask_len = source_randomness_len + pad_len;
    let inherited_scale = claim.base_claim_coeff * claim.residual_sumcheck_scale;

    let mut next_source_covector: Vec<EF> = source_covector
        .iter()
        .map(|&x| inherited_scale * x)
        .collect();
    let next_auxiliary_covectors: Vec<Vec<EF>> = auxiliary_covectors
        .iter()
        .map(|covector| covector.iter().map(|&x| inherited_scale * x).collect())
        .collect();
    let mut mask_covector = vec![EF::ZERO; mask_len];

    for (&rho, &coeff) in rho_ood_points.iter().zip(&claim.ood_coeffs) {
        let mut power = EF::ONE;
        for dst in &mut next_source_covector {
            *dst += coeff * power;
            power *= rho;
        }
        for dst in &mut mask_covector {
            *dst += coeff * power;
            power *= rho;
        }
    }

    for (&position, &coeff) in query_positions.iter().zip(&claim.in_domain_coeffs) {
        let message_row = source_encoding.message_row(position);
        if message_row.len() != source_len {
            return Err(CodeSwitchError::SourceMessageRowLengthMismatch {
                expected: source_len,
                actual: message_row.len(),
            });
        }
        for (dst, row) in next_source_covector.iter_mut().zip(message_row) {
            *dst += coeff * EF::from(row);
        }

        let randomness_row = source_encoding.randomness_row(position);
        if randomness_row.len() != source_randomness_len {
            return Err(CodeSwitchError::SourceRandomnessRowLengthMismatch {
                expected: source_randomness_len,
                actual: randomness_row.len(),
            });
        }
        for (dst, row) in mask_covector
            .iter_mut()
            .take(source_randomness_len)
            .zip(randomness_row)
        {
            *dst += coeff * EF::from(row);
        }
    }

    Ok(CodeSwitchOutputRelation {
        source_covector: next_source_covector,
        auxiliary_covectors: next_auxiliary_covectors,
        mask_covector,
    })
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
    ///
    /// The verifier must absorb this commitment before sampling private OOD
    /// points and must verify any later mask openings against this root. Merely
    /// storing the root is not enough for Construction 9.7.
    pub mask_commitment: MT::Commitment,
    /// Private OOD answers: `y = ze_ood(rho_ood) * [f; r; s_pad]`.
    ///
    /// Distinct from `WhirRoundProof::ood_answers` which are public
    /// folded-polynomial evaluations.
    pub private_ood_answers: Vec<EF>,
}

#[cfg(test)]
mod tests;
