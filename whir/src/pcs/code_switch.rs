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
//!
//! # Simulator boundary
//!
//! Lemma 9.8's simulator has randomized components supplied by the previous
//! and next IORs: private-zero-evader OOD samples, source-oracle query answers,
//! and mask/source codeword openings. This module owns the deterministic
//! Construction 9.7 reduction from those sampled answers to the verifier's
//! `mu'` claim and output relation. Full PCS-level simulator composition,
//! including target and mask opening simulation, belongs to the composed WHIR
//! prover/verifier path.

use alloc::vec;
use alloc::vec::Vec;

use p3_field::Field;
use p3_zk_codes::LinearZkEncoding;
use thiserror::Error;

pub use crate::parameters::RoundZkConfig;
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

/// Per-round ZK mask coefficient carrier.
///
/// Produced by the batching step (both prover and verifier) and consumed
/// by the next ZK sumcheck relation. Prevents the covector/weight mismatch
/// bug found in WizardOfMenlo/whir#249.
#[derive(Debug, Clone)]
pub struct ZkMaskClaim<EF> {
    /// Coefficient on the inherited source claim (`nu_1`).
    pub base_claim_coeff: EF,
    /// Scale applied by the HVZK sumcheck to the source residual.
    ///
    /// For the #1605 HVZK sumcheck this is the sampled `eps`. The incoming
    /// scalar claim must already include this scale on the source part. This
    /// value is kept separately so the output relation can apply `eps` only to
    /// the source covector, not to carried auxiliary mask covectors.
    pub residual_sumcheck_scale: EF,
    /// Batching coefficients for OOD answers (`nu_{1+i}` for `i in [t_ood]`).
    pub ood_coeffs: Vec<EF>,
    /// Batching coefficients for in-domain openings.
    pub in_domain_coeffs: Vec<EF>,
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

/// Deterministic verifier view induced by one code-switching round.
///
/// This intentionally does not sample randomness itself. The caller supplies
/// the values generated by the private zero-evader simulator and by simulated
/// oracle openings; this helper checks the transcript dimensions and derives
/// the exact `mu'` and output linear relation that the composed verifier sees.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodeSwitchVerifierView<EF> {
    /// Simulated private OOD answers.
    pub private_ood_answers: Vec<EF>,
    /// Simulated source-code openings used by the batching challenge.
    pub source_openings: Vec<EF>,
    /// Batched Construction 9.7 claim.
    pub mu_prime: EF,
    /// Output relation passed to the next IOR.
    pub output_relation: CodeSwitchOutputRelation<EF>,
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
/// The inherited claim is the scalar already handed off by the previous IOR.
/// In the #1605 HVZK sumcheck composition that means its source residual has
/// already been scaled by `eps`, while carried auxiliary mask claims remain in
/// the relation with their own coefficients. Consequently this function does
/// not apply `claim.residual_sumcheck_scale`; that scale is used only when
/// constructing the output source covector.
///
/// ```text
/// mu' = nu_1 * mu
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

    Ok(claim.base_claim_coeff * inherited_claim + ood_sum + in_domain_sum)
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
    let source_inherited_scale = claim.base_claim_coeff * claim.residual_sumcheck_scale;
    let auxiliary_inherited_scale = claim.base_claim_coeff;

    let mut next_source_covector: Vec<EF> = source_covector
        .iter()
        .map(|&x| source_inherited_scale * x)
        .collect();
    let next_auxiliary_covectors: Vec<Vec<EF>> = auxiliary_covectors
        .iter()
        .map(|covector| {
            covector
                .iter()
                .map(|&x| auxiliary_inherited_scale * x)
                .collect()
        })
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

/// Builds the deterministic part of the Lemma 9.8 verifier view.
///
/// The private OOD answers and source openings may be honestly computed or
/// sampled by simulators for the adjacent IORs. Construction 9.7 only needs
/// them as batched transcript values, then derives `mu'` and the output
/// relation from the same challenges.
#[allow(clippy::too_many_arguments)]
pub fn simulated_verifier_view<F, EF, Enc>(
    source_encoding: &Enc,
    inherited_claim: EF,
    source_covector: &[EF],
    auxiliary_covectors: &[&[EF]],
    source_randomness_len: usize,
    pad_len: usize,
    rho_ood_points: &[EF],
    query_positions: &[usize],
    simulated_private_ood_answers: &[EF],
    simulated_source_openings: &[EF],
    claim: &ZkMaskClaim<EF>,
) -> Result<CodeSwitchVerifierView<EF>, CodeSwitchError>
where
    F: Field,
    EF: Field + From<F>,
    Enc: LinearZkEncoding<F>,
{
    let mu_prime = batched_claim(
        inherited_claim,
        simulated_private_ood_answers,
        simulated_source_openings,
        claim,
    )?;
    let output_relation = output_relation(
        source_encoding,
        source_covector,
        auxiliary_covectors,
        source_randomness_len,
        pad_len,
        rho_ood_points,
        query_positions,
        claim,
    )?;

    Ok(CodeSwitchVerifierView {
        private_ood_answers: simulated_private_ood_answers.to_vec(),
        source_openings: simulated_source_openings.to_vec(),
        mu_prime,
        output_relation,
    })
}

#[cfg(test)]
mod tests;
