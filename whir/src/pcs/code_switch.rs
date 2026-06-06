//! HVZK code-switching round (Construction 9.7, eprint 2026/391 Section 9.4).
//!
//! - Reduces a proximity claim about a source oracle to one about a smaller
//!   target oracle, in zero knowledge.
//! - This module owns the deterministic relation algebra only.
//! - Proof payloads and the round loop land with the WHIR adapter (#1587).
//!
//! # Round shape
//!
//! ```text
//! prover  : sends fresh mask oracle encoding (r || s_pad)
//!           answers OOD points  y_i = ze*(rho_i) * (f || r || s_pad)^T
//! verifier: opens f at x_1..x_t, samples batching coefficients nu
//!           batches             mu' = nu_1*mu + sum nu*y_i + sum nu*f(x_j)
//! output  : linear relation over (f, carried masks, (r || s_pad))
//! ```
//!
//! - `r`: the source encoding randomness.
//! - `s_pad`: fresh, and what hides the OOD answers.
//!
//! # Sumcheck handoff (Construction 6.3)
//!
//! The incoming residual claim carries a challenge `eps` on its source part:
//!
//! - source covector: scaled by `eps` here,
//! - auxiliary covectors: `eps`-opaque, any scale is already baked in.
//!
//! # Privacy preconditions (not enforced here)
//!
//! - `pad_len >= t_ood`: one fresh pad coordinate per OOD answer.
//! - OOD points pairwise distinct and nonzero.
//! - Otherwise the answers leak a linear functional of the committed data.
//!
//! # Simulator boundary (Lemma 9.8)
//!
//! ```text
//! adjacent simulators: sample OOD answers, query answers, openings
//! this module        : derives the batched claim and output relation
//! ```

use alloc::vec::Vec;
use core::ops::Mul;

use p3_field::{Field, dot_product};
#[cfg(test)]
use p3_zk_codes::LinearZkEncoding;
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

    /// The number of source randomness rows does not match the in-domain batching coefficients.
    #[error("source randomness row count mismatch: expected {expected}, got {actual}")]
    SourceRandomnessRowCountMismatch { expected: usize, actual: usize },

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
/// - Produced by the batching step (both prover and verifier).
/// - Consumed by the next ZK sumcheck relation.
///
/// TODO(#1587): construct this from the round transcript when wiring the
/// round-0 WHIR adapter path.
#[derive(Debug, Clone)]
pub struct ZkMaskClaim<EF> {
    /// Coefficient on the inherited source claim (`nu_1`).
    pub base_claim_coeff: EF,
    /// Scale applied by the HVZK sumcheck to the source residual.
    ///
    /// - For the HVZK sumcheck handoff this is the sampled `eps`.
    /// - The incoming scalar claim already includes it on its source part.
    /// - The output relation applies it to the source covector only.
    /// - Auxiliary covectors are `eps`-opaque:
    ///   a claim that already carries `eps` on an auxiliary part must arrive
    ///   with the correspondingly scaled auxiliary covector.
    pub residual_sumcheck_scale: EF,
    /// Batching coefficients for OOD answers (`nu_{1+i}` for `i in [t_ood]`).
    pub ood_coeffs: Vec<EF>,
    /// Batching coefficients for in-domain openings.
    pub in_domain_coeffs: Vec<EF>,
}

impl<EF: Field> ZkMaskClaim<EF> {
    /// Computes the verifier-side batched claim `mu'`.
    ///
    /// ```text
    /// mu' = nu_1 * mu
    ///     + sum_i nu_{1+i} * y_i
    ///     + sum_j nu_{1+t_ood+j} * f(x_j)
    /// ```
    ///
    /// - `mu`: the scalar handed off by the previous reduction.
    /// - Its source residual already carries the `eps` scale.
    /// - `eps` is therefore not applied here;
    ///   it enters only the output source covector.
    pub fn batched_claim(
        &self,
        inherited_claim: EF,
        private_ood_answers: &[EF],
        source_openings: &[EF],
    ) -> Result<EF, CodeSwitchError> {
        // One batching coefficient per out-of-domain answer.
        if private_ood_answers.len() != self.ood_coeffs.len() {
            return Err(CodeSwitchError::PrivateOodAnswerCountMismatch {
                expected: self.ood_coeffs.len(),
                actual: private_ood_answers.len(),
            });
        }
        // One batching coefficient per in-domain opening.
        if source_openings.len() != self.in_domain_coeffs.len() {
            return Err(CodeSwitchError::SourceOpeningCountMismatch {
                expected: self.in_domain_coeffs.len(),
                actual: source_openings.len(),
            });
        }

        // sum_i nu_{1+i} * y_i over the out-of-domain answers.
        let ood_sum = dot_product::<EF, _, _>(
            self.ood_coeffs.iter().copied(),
            private_ood_answers.iter().copied(),
        );
        // sum_j nu_{1+t_ood+j} * f(x_j) over the in-domain openings.
        let in_domain_sum = dot_product::<EF, _, _>(
            self.in_domain_coeffs.iter().copied(),
            source_openings.iter().copied(),
        );

        // nu_1 * mu plus both batched transcript contributions.
        Ok(self.base_claim_coeff * inherited_claim + ood_sum + in_domain_sum)
    }

    /// Builds the output relation covectors for Construction 9.7.
    ///
    /// - Convenience form over a linear zero-knowledge encoding.
    /// - Used by the algebra and simulator-boundary tests.
    /// - `query_positions` are flattened codeword positions:
    ///   interleaving depth `iota` maps limb queries to `iota * x_i + limb`.
    ///
    /// TODO(#1587): the adapter wiring should call the row-level builder with
    /// rows supplied by the committed WHIR source oracle.
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn output_relation<F, Enc>(
        &self,
        source_encoding: &Enc,
        source_covector: &[EF],
        auxiliary_covectors: &[&[EF]],
        source_randomness_len: usize,
        pad_len: usize,
        rho_ood_points: &[EF],
        query_positions: &[usize],
    ) -> Result<CodeSwitchOutputRelation<EF>, CodeSwitchError>
    where
        F: Field,
        EF: Mul<F, Output = EF>,
        Enc: LinearZkEncoding<F>,
    {
        // One batching coefficient per out-of-domain point.
        if rho_ood_points.len() != self.ood_coeffs.len() {
            return Err(CodeSwitchError::OodPointCountMismatch {
                expected: self.ood_coeffs.len(),
                actual: rho_ood_points.len(),
            });
        }
        // One batching coefficient per flattened query position.
        if query_positions.len() != self.in_domain_coeffs.len() {
            return Err(CodeSwitchError::QueryPositionCountMismatch {
                expected: self.in_domain_coeffs.len(),
                actual: query_positions.len(),
            });
        }
        // The inherited covector addresses the full source message.
        let source_len = source_encoding.message_len();
        if source_covector.len() != source_len {
            return Err(CodeSwitchError::SourceCovectorLengthMismatch {
                expected: source_len,
                actual: source_covector.len(),
            });
        }

        // Fetch one `G^#` row per flattened query position.
        let source_rows = query_positions
            .iter()
            .map(|&position| source_encoding.message_row(position))
            .collect::<Vec<_>>();
        // Fetch the matching `G^$` row for the same positions.
        let source_randomness_rows = query_positions
            .iter()
            .map(|&position| source_encoding.randomness_row(position))
            .collect::<Vec<_>>();

        // Delegate to the row-level form.
        // Owned rows are borrowed through their slice view: no ref vector.
        self.output_relation_from_rows(
            source_len,
            source_covector,
            auxiliary_covectors,
            source_randomness_len,
            pad_len,
            rho_ood_points,
            &source_rows,
            &source_randomness_rows,
        )
    }

    /// Builds the output relation covectors from explicit source-code rows.
    ///
    /// - Row-level form of Construction 9.7, intended for the WHIR adapter.
    /// - There the rows come from the folded WHIR target oracle.
    ///
    /// # Covector assembly
    ///
    /// Each transcript event contributes one covector layer:
    ///
    /// ```text
    ///                  source slot j          mask slot j (j < r_len)   pad slot
    /// inherited        nu_1 * eps * sl[j]     0                         0
    /// OOD point rho    nu * rho^j             nu * rho^{ell + j}        nu * rho^{...}
    /// query x          nu * G^#[x][j]         nu * G^$[x][j]            0
    /// ```
    ///
    /// - OOD powers run over the concatenation `(f || r || s_pad)`.
    /// - Auxiliary covectors are carried with scale `nu_1` only.
    /// - The fresh pad never appears in openings.
    ///
    /// TODO(#1587): plug this into the round-0 adapter path after the proof
    /// payload and round configuration slice land.
    #[allow(clippy::too_many_arguments)]
    pub fn output_relation_from_rows<Row, S>(
        &self,
        source_message_len: usize,
        source_covector: &[EF],
        auxiliary_covectors: &[&[EF]],
        source_randomness_len: usize,
        pad_len: usize,
        rho_ood_points: &[EF],
        source_rows: &[S],
        source_randomness_rows: &[S],
    ) -> Result<CodeSwitchOutputRelation<EF>, CodeSwitchError>
    where
        Row: Copy,
        EF: Mul<Row, Output = EF>,
        S: AsRef<[Row]>,
    {
        // Phase 0: reject every dimension mismatch before touching a buffer.
        //
        // One batching coefficient per out-of-domain point.
        if rho_ood_points.len() != self.ood_coeffs.len() {
            return Err(CodeSwitchError::OodPointCountMismatch {
                expected: self.ood_coeffs.len(),
                actual: rho_ood_points.len(),
            });
        }
        // One batching coefficient per opened position.
        if source_rows.len() != self.in_domain_coeffs.len() {
            return Err(CodeSwitchError::QueryPositionCountMismatch {
                expected: self.in_domain_coeffs.len(),
                actual: source_rows.len(),
            });
        }
        // Message and randomness rows are indexed by the same query list.
        if source_randomness_rows.len() != self.in_domain_coeffs.len() {
            return Err(CodeSwitchError::SourceRandomnessRowCountMismatch {
                expected: self.in_domain_coeffs.len(),
                actual: source_randomness_rows.len(),
            });
        }
        // The inherited covector addresses the full source message.
        if source_covector.len() != source_message_len {
            return Err(CodeSwitchError::SourceCovectorLengthMismatch {
                expected: source_message_len,
                actual: source_covector.len(),
            });
        }
        // Every `G^#` row must address the declared message width.
        for message_row in source_rows {
            let row_len = message_row.as_ref().len();
            if row_len != source_message_len {
                return Err(CodeSwitchError::SourceMessageRowLengthMismatch {
                    expected: source_message_len,
                    actual: row_len,
                });
            }
        }
        // Every `G^$` row must address the declared randomness width.
        for randomness_row in source_randomness_rows {
            let row_len = randomness_row.as_ref().len();
            if row_len != source_randomness_len {
                return Err(CodeSwitchError::SourceRandomnessRowLengthMismatch {
                    expected: source_randomness_len,
                    actual: row_len,
                });
            }
        }

        // Phase 1: inherited contributions.
        //
        // The mask message is the source randomness followed by the fresh pad.
        let mask_len = source_randomness_len + pad_len;
        // Only the source residual carries the sumcheck scale `eps`.
        let source_inherited_scale = self.base_claim_coeff * self.residual_sumcheck_scale;
        // Carried auxiliary covectors are `eps`-opaque: scale by `nu_1` alone.
        let auxiliary_inherited_scale = self.base_claim_coeff;

        // Source covector starts as `nu_1 * eps * sl`.
        let mut next_source_covector: Vec<EF> = source_covector
            .iter()
            .map(|&x| source_inherited_scale * x)
            .collect();
        // Each carried auxiliary covector becomes `nu_1 * u_i`.
        let next_auxiliary_covectors: Vec<Vec<EF>> = auxiliary_covectors
            .iter()
            .map(|covector| {
                covector
                    .iter()
                    .map(|&x| auxiliary_inherited_scale * x)
                    .collect()
            })
            .collect();
        // Mask slots receive contributions only from the phases below.
        let mut mask_covector = EF::zero_vec(mask_len);

        // Phase 2: out-of-domain contributions.
        //
        //     source slot j : += coeff * rho^j          (j = 0..ell)
        //     mask   slot j : += coeff * rho^{ell + j}  (powers keep running)
        //
        // The running term matches the prover-side concatenated evaluation.
        for (&rho, &coeff) in rho_ood_points.iter().zip(&self.ood_coeffs) {
            // Fold coeff into the running power: one multiplication per slot.
            let mut term = coeff;
            // Source slots take coeff * rho^0 .. coeff * rho^{ell-1}.
            for dst in &mut next_source_covector {
                *dst += term;
                term *= rho;
            }
            // Mask slots take coeff * rho^ell onwards.
            for dst in &mut mask_covector {
                *dst += term;
                term *= rho;
            }
        }

        // Phase 3: in-domain contributions.
        //
        //     opening x : f(x) = <f, G^#[x]> + <r, G^$[x]>
        //
        // Rows stay in their own (typically base) field:
        // extension * base picks the cheap scalar product.
        for ((message_row, randomness_row), &coeff) in source_rows
            .iter()
            .zip(source_randomness_rows)
            .zip(&self.in_domain_coeffs)
        {
            // Message part: nu * G^#[x] onto the source slots.
            for (dst, &row) in next_source_covector.iter_mut().zip(message_row.as_ref()) {
                *dst += coeff * row;
            }
            // Randomness part: nu * G^$[x] onto the first r_len mask slots.
            // The fresh pad never appears in openings, so its slots stay zero.
            for (dst, &row) in mask_covector
                .iter_mut()
                .take(source_randomness_len)
                .zip(randomness_row.as_ref())
            {
                *dst += coeff * row;
            }
        }

        Ok(CodeSwitchOutputRelation {
            source_covector: next_source_covector,
            auxiliary_covectors: next_auxiliary_covectors,
            mask_covector,
        })
    }
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
/// - Samples no randomness itself.
/// - Inputs come from the private zero-evader simulator and from simulated
///   oracle openings.
/// - Holds the exact `mu'` and output relation the composed verifier sees.
///
/// TODO(#1587): deterministic Lemma 9.8 boundary used by tests until the
/// full PCS-level simulator is wired through the WHIR adapter.
#[cfg(test)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CodeSwitchVerifierView<EF> {
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
        // The source witness must address the full source covector.
        if source_message.len() != self.source_covector.len() {
            return Err(CodeSwitchError::SourceMessageLengthMismatch {
                expected: self.source_covector.len(),
                actual: source_message.len(),
            });
        }
        // One witness per carried auxiliary covector.
        if auxiliary_witnesses.len() != self.auxiliary_covectors.len() {
            return Err(CodeSwitchError::AuxiliaryWitnessCountMismatch {
                expected: self.auxiliary_covectors.len(),
                actual: auxiliary_witnesses.len(),
            });
        }
        // The mask witness must address the full mask covector.
        if mask_message.len() != self.mask_covector.len() {
            return Err(CodeSwitchError::MaskMessageLengthMismatch {
                expected: self.mask_covector.len(),
                actual: mask_message.len(),
            });
        }

        // <f, source covector>: the source message contribution.
        let mut value = dot_product::<EF, _, _>(
            source_message.iter().copied(),
            self.source_covector.iter().copied(),
        );
        // Carried auxiliary contributions, one inner product per round mask.
        for (index, (witness, covector)) in auxiliary_witnesses
            .iter()
            .zip(&self.auxiliary_covectors)
            .enumerate()
        {
            // Auxiliary widths vary per round, so each pair is checked here.
            if witness.len() != covector.len() {
                return Err(CodeSwitchError::AuxiliaryWitnessLengthMismatch {
                    index,
                    expected: covector.len(),
                    actual: witness.len(),
                });
            }
            value += dot_product::<EF, _, _>(witness.iter().copied(), covector.iter().copied());
        }
        // <(r || s_pad), mask covector>: the fresh mask contribution.
        value += dot_product::<EF, _, _>(
            mask_message.iter().copied(),
            self.mask_covector.iter().copied(),
        );

        Ok(value)
    }
}

/// Computes all private OOD answers for one code-switching round.
///
/// ```text
/// y_i = ze*(rho_i) * (f || r || s_pad)^T
///     = sum_j f_j * rho_i^j  +  rho_i^ell * sum_j mask_j * rho_i^j
/// ```
///
/// # Privacy precondition
///
/// - Fresh entropy: only the trailing pad coordinates of the mask.
/// - The rest is correlated with committed data through the openings.
/// - Required: pad length at least the number of points.
/// - Required: points pairwise distinct and nonzero.
/// - Why it works: the fresh-pad block is then a full-rank scaled
///   Vandermonde, so uniform pads make the answers jointly uniform.
///
/// # Why the bound is tight
///
/// Two answers, one fresh pad coordinate `s`:
///
/// ```text
/// y_1 = <(f || r), ze*(rho_1)> + s * rho_1^{ell+r_len}
/// y_2 = <(f || r), ze*(rho_2)> + s * rho_2^{ell+r_len}
///
/// rho_2^{ell+r_len} * y_1 - rho_1^{ell+r_len} * y_2    // s cancels
/// -> a public linear functional of the committed (f || r)
/// -> one leaked query against the source ZK budget
/// ```
#[must_use]
pub fn private_ood_answers<EF: Field>(
    rho_ood_points: &[EF],
    source_message: &[EF],
    mask_message: &[EF],
) -> Vec<EF> {
    rho_ood_points
        .iter()
        // Every answer evaluates the same concatenated witness at its own point.
        .map(|&rho| padded_ood_t1(rho, source_message, mask_message))
        .collect()
}

/// Builds the deterministic part of the Lemma 9.8 verifier view.
///
/// Not a full transcript simulator:
///
/// - inputs: transcript values, honestly computed or sampled by the
///   simulators of the adjacent reductions,
/// - outputs: the Construction 9.7 claim and relation derived from the same
///   public challenges.
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(crate) fn simulated_verifier_view<F, EF, Enc>(
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
    EF: Field + Mul<F, Output = EF>,
    Enc: LinearZkEncoding<F>,
{
    // Batch the simulated transcript scalars into the verifier claim.
    let mu_prime = claim.batched_claim(
        inherited_claim,
        simulated_private_ood_answers,
        simulated_source_openings,
    )?;
    // Assemble the output covectors from the same public challenges.
    let output_relation = claim.output_relation(
        source_encoding,
        source_covector,
        auxiliary_covectors,
        source_randomness_len,
        pad_len,
        rho_ood_points,
        query_positions,
    )?;

    Ok(CodeSwitchVerifierView {
        private_ood_answers: simulated_private_ood_answers.to_vec(),
        source_openings: simulated_source_openings.to_vec(),
        mu_prime,
        output_relation,
    })
}

#[cfg(test)]
mod tests {
    //! Invariant tests for Construction 9.7 over BabyBear.

    use alloc::vec;
    use alloc::vec::Vec;

    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_challenger::DuplexChallenger;
    use p3_commit::{ExtensionMmcs, Mmcs};
    use p3_dft::{Radix2DFTSmallBatch, Radix2Dit};
    use p3_field::extension::BinomialExtensionField;
    use p3_field::{Field, PrimeCharacteristicRing};
    use p3_merkle_tree::MerkleTreeMmcs;
    use p3_sumcheck::zk::{ZkVerifier, simulate_classic_unpacked};
    use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
    use p3_zk_codes::{
        LinearZkEncoding, ReedSolomonZkEncoding, ZkEncoding, ZkEncodingWithRandomness,
    };
    use proptest::prelude::*;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::{
        CodeSwitchError, CodeSwitchOutputRelation, ZkMaskClaim, private_ood_answers,
        simulated_verifier_view,
    };
    use crate::utils::{eval_ze_star_n, padded_ood_t1};

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;
    type Perm = Poseidon2BabyBear<16>;
    type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
    type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
    type MyChallenger = DuplexChallenger<F, Perm, 16, 8>;
    type PackedF = <F as Field>::Packing;
    type BaseMmcs = MerkleTreeMmcs<PackedF, PackedF, MyHash, MyCompress, 2, 8>;
    type MyMmcs = ExtensionMmcs<F, EF, BaseMmcs>;
    type MyDft = Radix2DFTSmallBatch<EF>;
    type MyEnc = ReedSolomonZkEncoding<EF, MyDft>;

    /// Builds a permutation, Merkle commitment scheme, and mask encoding from a seed.
    fn make_setup(seed: u64, ell_zk: usize) -> (Perm, MyMmcs, MyEnc) {
        // Deterministic permutation so simulator and replay share parameters.
        let mut perm_rng = SmallRng::seed_from_u64(seed);
        let perm = Perm::new_from_rng_128(&mut perm_rng);
        // Merkle hashing and 2-to-1 compression over the same permutation.
        let merkle_hash = MyHash::new(perm.clone());
        let merkle_compress = MyCompress::new(perm.clone());
        let base_mmcs = BaseMmcs::new(merkle_hash, merkle_compress, 0);
        let mmcs = ExtensionMmcs::new(base_mmcs);
        // Mask encoding: 2 randomness coefficients, message width `ell_zk`,
        // domain just large enough to hold message plus randomness.
        let encoding = MyEnc::new(
            2,
            ell_zk,
            (ell_zk + 2).next_power_of_two(),
            MyDft::default(),
        );

        (perm, mmcs, encoding)
    }

    /// Lifts a small integer into the extension field.
    fn ef(v: u64) -> EF {
        EF::from(F::from_u64(v))
    }

    /// Inner product of two equal-length extension-field slices.
    fn inner_product(a: &[EF], b: &[EF]) -> EF {
        a.iter().zip(b.iter()).map(|(x, y)| *x * *y).sum()
    }

    /// Builds a base-field Reed-Solomon zero-knowledge encoding.
    fn make_rs_encoding(
        msg_len: usize,
        t: usize,
        m: usize,
    ) -> ReedSolomonZkEncoding<F, Radix2Dit<F>> {
        ReedSolomonZkEncoding::new(t, msg_len, m, Radix2Dit::default())
    }

    /// Draws a vector of uniformly random extension-field elements.
    fn draw_ef_vec(rng: &mut SmallRng, len: usize) -> Vec<EF> {
        (0..len).map(|_| rng.random()).collect()
    }

    /// Solves for the 2-coefficient encoding randomness that explains two openings.
    ///
    /// # Why this is solvable
    ///
    /// - Each opening decomposes as `cw[x] = <msg, G^#[x]> + <rand, G^$[x]>`.
    /// - With two openings and two randomness coefficients this is a
    ///   2 x 2 linear system in the randomness.
    /// - For Reed-Solomon rows at distinct nonzero points the system matrix is
    ///   a scaled Vandermonde block, hence invertible.
    fn solve_two_randomness_for_openings(
        encoding: &ReedSolomonZkEncoding<EF, Radix2Dit<EF>>,
        message: &[EF],
        positions: &[usize; 2],
        openings: &[EF],
    ) -> Vec<EF> {
        assert_eq!(openings.len(), 2);

        // For each opening, isolate the randomness contribution:
        //     <rand, G^$[x]> = cw[x] - <msg, G^#[x]>
        let delta = |index: usize| {
            let position = positions[index];
            let message_row = encoding.message_row(position);
            let message_value = inner_product(message, &message_row);
            let randomness_row = encoding.randomness_row(position);
            assert_eq!(randomness_row.len(), 2);
            (randomness_row, openings[index] - message_value)
        };
        let (row0, rhs0) = delta(0);
        let (row1, rhs1) = delta(1);

        // Cramer's rule on the 2 x 2 system
        //     [a b] [r_0]   [rhs0]
        //     [c d] [r_1] = [rhs1]
        let a = row0[0];
        let b = row0[1];
        let c = row1[0];
        let d = row1[1];
        let det = a * d - b * c;
        assert!(!det.is_zero(), "RS query rows should be independent");

        vec![(rhs0 * d - b * rhs1) / det, (a * rhs1 - rhs0 * c) / det]
    }

    /// Solves for the single fresh pad coordinate hitting a target OOD answer.
    ///
    /// # Why this is solvable
    ///
    /// - The answer is linear in the pad with coefficient
    ///   `rho^{ell + r_len}`, which is nonzero whenever `rho != 0`.
    /// - This is the Lemma 9.3 programmability used by the simulator boundary.
    fn solve_pad_for_private_ood(
        rho: EF,
        message: &[EF],
        source_randomness: &[EF],
        target: EF,
    ) -> EF {
        assert!(!rho.is_zero(), "programming formula requires nonzero rho");

        // Fixed contributions of the committed message and source randomness.
        let message_eval = eval_ze_star_n(rho, message);
        let source_randomness_eval = eval_ze_star_n(rho, source_randomness);
        // Power shift placing the randomness block after the message block.
        let shift = rho.exp_u64(message.len() as u64);
        // Coefficient multiplying the pad coordinate in the answer.
        let pad_scale = shift * rho.exp_u64(source_randomness.len() as u64);
        assert!(
            !pad_scale.is_zero(),
            "programming formula requires a nonzero pad coefficient",
        );

        // Invert the affine map: pad = (target - fixed parts) / coefficient.
        (target - message_eval - shift * source_randomness_eval) / pad_scale
    }

    /// Lifts a base-field row into extension-field elements for inner products.
    fn lift_row(row: &[F]) -> Vec<EF> {
        row.iter().map(|&x| EF::from(x)).collect()
    }

    #[test]
    fn test_construction_9_7_mu_prime_identity_n0() {
        // Invariant: mu' (verifier batching) == relation(witness), n = 0.
        //
        // NOTE: 2 OOD answers with 1 fresh pad coordinate is completeness-only.
        // It violates `pad_len >= t_ood`; see the leak test further below.
        let ell = 4;
        let r_len = 2;
        let s_pad_len = 1;
        let t_ood = 2;
        let t = 3;
        let iota = 1;
        let m = 8;

        let source_enc = make_rs_encoding(ell, r_len, m);

        // Honest witness: source message, encoding randomness, fresh pad.
        let f: Vec<EF> = (1..=ell as u64).map(ef).collect();
        let r: Vec<EF> = (10..10 + r_len as u64).map(ef).collect();
        let s_pad: Vec<EF> = (20..20 + s_pad_len as u64).map(ef).collect();

        // Real RS codeword for the source oracle, lifted to the extension.
        let f_base: Vec<F> = (1..=ell as u64).map(F::from_u64).collect();
        let r_base: Vec<F> = (10..10 + r_len as u64).map(F::from_u64).collect();
        let cw = source_enc.encode_with_randomness(&f_base, &r_base);
        let f_codeword: Vec<EF> = cw.values.iter().map(|&v| EF::from(v)).collect();

        // Inherited linear claim mu = <f, sl> (no eps scale: scale is 1).
        let sl: Vec<EF> = (1..=ell as u64).map(|i| ef(i * 100)).collect();
        let mu = inner_product(&f, &sl);

        // Honest OOD answers: ze*(rho_i) over (f || r || s_pad).
        let rho_ood_points: Vec<EF> = (0..t_ood).map(|i| ef(50 + i as u64)).collect();
        let mut f_r_s: Vec<EF> = Vec::with_capacity(ell + r_len + s_pad_len);
        f_r_s.extend_from_slice(&f);
        f_r_s.extend_from_slice(&r);
        f_r_s.extend_from_slice(&s_pad);
        let y: Vec<EF> = rho_ood_points
            .iter()
            .map(|&rho| eval_ze_star_n(rho, &f_r_s))
            .collect();
        assert_eq!(y.len(), t_ood);

        // Honest in-domain openings at three codeword positions.
        let query_positions: Vec<usize> = vec![0, 2, 4];
        assert_eq!(query_positions.len(), t);
        let source_openings: Vec<EF> = query_positions.iter().map(|&pos| f_codeword[pos]).collect();

        // Batching coefficients nu = (1, rho, rho^2, ...) as ze*(rho_batch).
        let nu_dim = 1 + t_ood + t * iota;
        let rho_batch = ef(77);
        let nu = rho_batch.powers().collect_n(nu_dim);
        let claim = ZkMaskClaim {
            base_claim_coeff: nu[0],
            residual_sumcheck_scale: EF::ONE,
            ood_coeffs: nu[1..1 + t_ood].to_vec(),
            in_domain_coeffs: nu[1 + t_ood..].to_vec(),
        };

        // Verifier side: batch the transcript scalars.
        let mu_prime = claim.batched_claim(mu, &y, &source_openings).unwrap();
        // Relation side: assemble the output covectors from the challenges.
        let relation = claim
            .output_relation::<F, _>(
                &source_enc,
                &sl,
                &[],
                r_len,
                s_pad_len,
                &rho_ood_points,
                &query_positions,
            )
            .unwrap();

        // Evaluate the relation on the honest witness (f, (r || s_pad)).
        let mut r_s_pad = r;
        r_s_pad.extend_from_slice(&s_pad);
        let mu_prime_from_relation = relation.evaluate(&f, &[], &r_s_pad).unwrap();

        // Completeness: both sides compute the same scalar.
        assert_eq!(
            mu_prime, mu_prime_from_relation,
            "Construction 9.7 mu' identity failed: verifier-computed mu' != output relation linear form"
        );
    }

    #[test]
    fn test_construction_9_7_mu_prime_identity_n2() {
        // Invariant: mu' (verifier batching) == relation(witness), n = 2.
        //
        // Fixture state: 2 prior auxiliary mask oracles, scale 1.
        let ell = 3;
        let r_len = 2;
        let s_pad_len = 1;
        let t_ood = 1;
        let t = 2;
        let iota = 1;
        let n = 2;
        let m = 8;

        let source_enc = make_rs_encoding(ell, r_len, m);

        // Honest witness: source message, encoding randomness, fresh pad.
        let f: Vec<EF> = (1..=ell as u64).map(|i| ef(i * 3)).collect();
        let r: Vec<EF> = (10..10 + r_len as u64).map(ef).collect();
        let s_pad: Vec<EF> = vec![ef(42)];

        // Real RS codeword for the source oracle, lifted to the extension.
        let f_base: Vec<F> = (1..=ell as u64).map(|i| F::from_u64(i * 3)).collect();
        let r_base: Vec<F> = (10..10 + r_len as u64).map(F::from_u64).collect();
        let cw = source_enc.encode_with_randomness(&f_base, &r_base);
        let f_codeword: Vec<EF> = cw.values.iter().map(|&v| EF::from(v)).collect();

        let sl: Vec<EF> = (1..=ell as u64).map(|i| ef(i * 7)).collect();

        // Two carried auxiliary masks with distinct witnesses and covectors.
        let mask_msg_len_zk = 2;
        let xi: Vec<Vec<EF>> = (0..n)
            .map(|k| {
                (0..mask_msg_len_zk)
                    .map(|j| ef((k * 10 + j + 30) as u64))
                    .collect()
            })
            .collect();
        let sl_aux: Vec<Vec<EF>> = (0..n)
            .map(|k| {
                (0..mask_msg_len_zk)
                    .map(|j| ef((k * 5 + j + 60) as u64))
                    .collect()
            })
            .collect();

        // Inherited claim: source part plus both auxiliary parts.
        let mut mu = inner_product(&f, &sl);
        for i in 0..n {
            mu += inner_product(&xi[i], &sl_aux[i]);
        }

        // Honest OOD answer over the concatenation (f || r || s_pad).
        let rho_ood_points = [ef(99)];
        let mut f_r_s: Vec<EF> = Vec::new();
        f_r_s.extend_from_slice(&f);
        f_r_s.extend_from_slice(&r);
        f_r_s.extend_from_slice(&s_pad);
        let y: Vec<EF> = rho_ood_points
            .iter()
            .map(|&rho| eval_ze_star_n(rho, &f_r_s))
            .collect();

        // Honest in-domain openings at two codeword positions.
        let query_positions: Vec<usize> = vec![1, 3];
        let source_openings: Vec<EF> = query_positions.iter().map(|&p| f_codeword[p]).collect();

        // Batching coefficients as a power sequence.
        let nu_dim = 1 + t_ood + t * iota;
        let rho_batch = ef(55);
        let nu = rho_batch.powers().collect_n(nu_dim);
        let claim = ZkMaskClaim {
            base_claim_coeff: nu[0],
            residual_sumcheck_scale: EF::ONE,
            ood_coeffs: nu[1..1 + t_ood].to_vec(),
            in_domain_coeffs: nu[1 + t_ood..].to_vec(),
        };

        // Verifier side vs relation side, with auxiliary covectors carried.
        let mu_prime = claim.batched_claim(mu, &y, &source_openings).unwrap();
        let aux_refs: Vec<&[EF]> = sl_aux.iter().map(Vec::as_slice).collect();
        let relation = claim
            .output_relation::<F, _>(
                &source_enc,
                &sl,
                &aux_refs,
                r_len,
                s_pad_len,
                &rho_ood_points,
                &query_positions,
            )
            .unwrap();

        // Evaluate on the honest witness, including both auxiliary witnesses.
        let mut r_s_pad = r.clone();
        r_s_pad.extend_from_slice(&s_pad);
        let xi_refs: Vec<&[EF]> = xi.iter().map(Vec::as_slice).collect();
        let mu_prime_from_relation = relation.evaluate(&f, &xi_refs, &r_s_pad).unwrap();

        assert_eq!(
            mu_prime, mu_prime_from_relation,
            "Construction 9.7 mu' identity failed with n=2 auxiliary masks"
        );
    }

    #[test]
    fn test_construction_9_7_mu_prime_identity_iota2() {
        // Invariant: the mu' identity holds under interleaving depth 2.
        //
        //     symbol x -> 2 limbs -> flattened rows (2x, 2x + 1)
        //
        // Fixture state: domain m = 16 so that msg_len + t = 6 < 16.
        let ell = 4;
        let r_len = 2;
        let s_pad_len = 2;
        let t_ood = 2;
        let t = 2;
        let iota = 2;
        let m = 16;

        let source_enc = make_rs_encoding(ell, r_len, m);

        // Honest witness: source message, encoding randomness, fresh pad.
        let f: Vec<EF> = (1..=ell as u64).map(|i| ef(i * 2)).collect();
        let r: Vec<EF> = (0..r_len as u64).map(|i| ef(30 + i)).collect();
        let s_pad: Vec<EF> = (0..s_pad_len as u64).map(|i| ef(40 + i)).collect();

        // Real RS codeword for the source oracle, lifted to the extension.
        let f_base: Vec<F> = (1..=ell as u64).map(|i| F::from_u64(i * 2)).collect();
        let r_base: Vec<F> = (0..r_len as u64).map(|i| F::from_u64(30 + i)).collect();
        let cw = source_enc.encode_with_randomness(&f_base, &r_base);
        let f_codeword: Vec<EF> = cw.values.iter().map(|&v| EF::from(v)).collect();

        // Inherited linear claim with scale 1.
        let sl: Vec<EF> = (0..ell as u64).map(|i| ef(7 + i * 3)).collect();
        let mu = inner_product(&f, &sl);

        // Honest OOD answers over the concatenation (f || r || s_pad).
        let rho_ood_points = [ef(6), ef(8)];
        let mut f_r_s = Vec::with_capacity(ell + r_len + s_pad_len);
        f_r_s.extend_from_slice(&f);
        f_r_s.extend_from_slice(&r);
        f_r_s.extend_from_slice(&s_pad);
        let y: Vec<EF> = rho_ood_points
            .iter()
            .map(|&rho| eval_ze_star_n(rho, &f_r_s))
            .collect();

        // Open both limbs of each queried symbol: 2 symbols * 2 limbs = 4 openings.
        let query_symbols = [0_usize, 2_usize];
        let mut source_openings = Vec::with_capacity(t * iota);
        for &symbol in &query_symbols {
            for limb in 0..iota {
                source_openings.push(f_codeword[symbol * iota + limb]);
            }
        }

        // Batching coefficients: one per limb, so nu has 1 + t_ood + t*iota entries.
        let nu_dim = 1 + t_ood + t * iota;
        let nu = ef(13).powers().collect_n(nu_dim);
        let claim = ZkMaskClaim {
            base_claim_coeff: nu[0],
            residual_sumcheck_scale: EF::ONE,
            ood_coeffs: nu[1..1 + t_ood].to_vec(),
            in_domain_coeffs: nu[1 + t_ood..].to_vec(),
        };

        // Verifier side: batch over the flattened openings.
        let mu_prime = claim.batched_claim(mu, &y, &source_openings).unwrap();
        // Relation side: flatten symbol queries into per-limb row positions.
        let query_positions: Vec<usize> = query_symbols
            .iter()
            .flat_map(|&symbol| (0..iota).map(move |limb| symbol * iota + limb))
            .collect();
        let relation = claim
            .output_relation::<F, _>(
                &source_enc,
                &sl,
                &[],
                r_len,
                s_pad_len,
                &rho_ood_points,
                &query_positions,
            )
            .unwrap();

        // Evaluate the relation on the honest witness.
        let mut r_s_pad = r.clone();
        r_s_pad.extend_from_slice(&s_pad);
        let mu_prime_from_relation = relation.evaluate(&f, &[], &r_s_pad).unwrap();

        assert_eq!(
            mu_prime, mu_prime_from_relation,
            "Construction 9.7 mu' identity failed for iota=2 flattened row indexing"
        );
    }

    #[test]
    fn test_construction_9_7_mu_prime_identity_eps_scaled_handoff() {
        // Invariant: eps scales ONLY the inherited source covector.
        // OOD and in-domain layers batch independently of eps.
        // The covectors are rebuilt by hand to pin the exact layering.
        let ell = 3;
        let r_len = 1;
        let s_pad_len = 1;
        let t_ood = 1;
        let t = 1;
        let iota = 1;
        let m = 8;

        let source_enc = make_rs_encoding(ell, r_len, m);

        // Honest witness: source message, encoding randomness, fresh pad.
        let f: Vec<EF> = vec![ef(2), ef(5), ef(9)];
        let r: Vec<EF> = vec![ef(12)];
        let s_pad: Vec<EF> = vec![ef(20)];

        // Real RS codeword for the source oracle, lifted to the extension.
        let f_base: Vec<F> = [2, 5, 9].into_iter().map(F::from_u64).collect();
        let r_base = vec![F::from_u64(12)];
        let cw = source_enc.encode_with_randomness(&f_base, &r_base);
        let f_codeword: Vec<EF> = cw.values.iter().map(|&v| EF::from(v)).collect();

        // Inherited claim folds eps into the source residual: mu = eps * <f, sl>.
        let eps = ef(19);
        let sl: Vec<EF> = vec![ef(4), ef(7), ef(11)];
        let mu = eps * inner_product(&f, &sl);

        // Honest OOD answer over (f || r || s_pad).
        let rho_ood_points = [ef(31)];
        let y = [padded_ood_t1(
            rho_ood_points[0],
            &f,
            &[r.clone(), s_pad.clone()].concat(),
        )];

        // One in-domain opening.
        let query_position = 2;
        let source_opening = f_codeword[query_position];

        // Batching coefficients (nu_1, nu_2, nu_3).
        let nu = ef(17).powers().collect_n(1 + t_ood + t * iota);

        let claim = ZkMaskClaim {
            base_claim_coeff: nu[0],
            residual_sumcheck_scale: eps,
            ood_coeffs: vec![nu[1]],
            in_domain_coeffs: vec![nu[2]],
        };
        let mu_prime = claim.batched_claim(mu, &y, &[source_opening]).unwrap();

        // Hand-built source covector, layer by layer:
        //
        //     inherited:  eps * nu_1 * sl[j]
        //     OOD:        nu_2 * rho^j          (j = 0..ell)
        //     query:      nu_3 * G^#[x][j]
        let mut sl_prime = vec![EF::ZERO; ell];
        for (sp, s) in sl_prime.iter_mut().zip(&sl) {
            *sp += eps * nu[0] * *s;
        }
        let mut power = EF::ONE;
        for sp in sl_prime.iter_mut() {
            *sp += nu[1] * power;
            power *= rho_ood_points[0];
        }
        let g_sharp = lift_row(&source_enc.message_row(query_position));
        for (sp, gs) in sl_prime.iter_mut().zip(&g_sharp) {
            *sp += nu[2] * *gs;
        }

        // Hand-built mask covector:
        //
        //     OOD:        nu_2 * rho^{ell + j}  (powers continue past the source)
        //     query:      nu_3 * G^$[x][j]      (randomness slot only)
        let mask_msg_len = r_len + s_pad_len;
        let mut sl_mask = vec![EF::ZERO; mask_msg_len];
        let mut power = EF::ONE;
        for _ in 0..ell {
            power *= rho_ood_points[0];
        }
        for sm in sl_mask.iter_mut() {
            *sm += nu[1] * power;
            power *= rho_ood_points[0];
        }
        let g_dollar = lift_row(&source_enc.randomness_row(query_position));
        for (sm, gd) in sl_mask.iter_mut().zip(&g_dollar) {
            *sm += nu[2] * *gd;
        }

        // Evaluate the hand-built relation on the honest witness.
        let mut r_s_pad = r;
        r_s_pad.extend_from_slice(&s_pad);
        let mu_prime_from_relation =
            inner_product(&f, &sl_prime) + inner_product(&r_s_pad, &sl_mask);

        assert_eq!(
            mu_prime, mu_prime_from_relation,
            "Construction 9.7 must preserve the eps-scaled residual handoff"
        );
    }

    #[test]
    fn test_eps_scaled_handoff_does_not_scale_auxiliary_covectors() {
        // Invariant: an eps-free auxiliary covector is batched by nu_1 alone,
        // never by nu_1 * eps.
        // This models a fresh sumcheck mask, whose covector is eps-free.
        let ell = 3;
        let r_len = 1;
        let s_pad_len = 1;
        let t_ood = 1;
        let t = 1;
        let iota = 1;
        let m = 8;

        let source_enc = make_rs_encoding(ell, r_len, m);

        // Honest witness: source message, encoding randomness, fresh pad.
        let f: Vec<EF> = vec![ef(3), ef(8), ef(13)];
        let r: Vec<EF> = vec![ef(21)];
        let s_pad: Vec<EF> = vec![ef(34)];

        // Real RS codeword for the source oracle, lifted to the extension.
        let f_base: Vec<F> = [3, 8, 13].into_iter().map(F::from_u64).collect();
        let r_base = vec![F::from_u64(21)];
        let cw = source_enc.encode_with_randomness(&f_base, &r_base);
        let f_codeword: Vec<EF> = cw.values.iter().map(|&v| EF::from(v)).collect();

        // Inherited claim: eps on the source part, no eps on the aux part.
        //
        //     mu = eps * <f, sl>  +  <aux_w, aux_cov>
        let source_covector = vec![ef(5), ef(7), ef(11)];
        let auxiliary_witness = vec![ef(17), ef(19)];
        let auxiliary_covector = vec![ef(23), ef(29)];
        let eps = ef(31);
        let source_claim = inner_product(&f, &source_covector);
        let auxiliary_claim = inner_product(&auxiliary_witness, &auxiliary_covector);
        let inherited_claim = eps * source_claim + auxiliary_claim;

        // Honest OOD answer over (f || r || s_pad).
        let rho_ood_points = [ef(37)];
        let mut mask_message = r;
        mask_message.extend_from_slice(&s_pad);
        let y = private_ood_answers(&rho_ood_points, &f, &mask_message);

        // One in-domain opening and the batching coefficients.
        let query_position = 4;
        let source_opening = f_codeword[query_position];
        let nu = ef(41).powers().collect_n(1 + t_ood + t * iota);
        let claim = ZkMaskClaim {
            base_claim_coeff: nu[0],
            residual_sumcheck_scale: eps,
            ood_coeffs: vec![nu[1]],
            in_domain_coeffs: vec![nu[2]],
        };

        // Verifier side vs relation side.
        let mu_prime = claim
            .batched_claim(inherited_claim, &y, &[source_opening])
            .unwrap();
        let relation = claim
            .output_relation::<F, _>(
                &source_enc,
                &source_covector,
                &[&auxiliary_covector],
                r_len,
                s_pad_len,
                &rho_ood_points,
                &[query_position],
            )
            .unwrap();
        let mu_prime_from_relation = relation
            .evaluate(&f, &[&auxiliary_witness], &mask_message)
            .unwrap();

        // The carried auxiliary covector is exactly nu_1 * aux_cov: no eps.
        let expected_auxiliary_covector: Vec<EF> =
            auxiliary_covector.iter().map(|&x| nu[0] * x).collect();

        assert_eq!(
            relation.auxiliary_covectors[0], expected_auxiliary_covector,
            "auxiliary covectors must not inherit the HVZK sumcheck eps scale"
        );
        assert_eq!(
            mu_prime, mu_prime_from_relation,
            "eps-scaled source handoff must compose with unscaled auxiliary masks"
        );
    }

    #[test]
    fn test_eps_scaled_auxiliary_covector_is_not_scaled_again() {
        // Invariant: a covector already carrying eps is scaled by nu_1 once,
        // never by eps again.
        //
        // Fixture:
        //     covector passed : eps * u
        //     inherited claim : eps * <f, sl> + <aux_w, eps * u>
        let source_message = vec![ef(3), ef(5)];
        let source_covector = vec![ef(7), ef(11)];
        let auxiliary_witness = vec![ef(13), ef(17)];
        let auxiliary_covector = [ef(19), ef(23)];
        let eps = ef(29);
        let nu_1 = ef(31);

        // Pre-scale the auxiliary covector by eps, as the sumcheck output does.
        let scaled_auxiliary_covector: Vec<EF> =
            auxiliary_covector.iter().map(|&x| eps * x).collect();
        let inherited_claim = eps * inner_product(&source_message, &source_covector)
            + inner_product(&auxiliary_witness, &scaled_auxiliary_covector);
        // No OOD points and no openings: the inherited layer is isolated.
        let claim = ZkMaskClaim {
            base_claim_coeff: nu_1,
            residual_sumcheck_scale: eps,
            ood_coeffs: Vec::new(),
            in_domain_coeffs: Vec::new(),
        };

        let mu_prime = claim.batched_claim(inherited_claim, &[], &[]).unwrap();
        let relation = claim
            .output_relation_from_rows::<F, &[F]>(
                source_message.len(),
                &source_covector,
                &[&scaled_auxiliary_covector],
                0,
                0,
                &[],
                &[],
                &[],
            )
            .unwrap();

        // Output auxiliary covector is nu_1 * (eps * u), not nu_1 * eps^2 * u.
        assert_eq!(
            relation.auxiliary_covectors[0],
            scaled_auxiliary_covector
                .iter()
                .map(|&x| nu_1 * x)
                .collect::<Vec<_>>(),
            "auxiliary covectors already carrying eps must not be scaled by eps again",
        );
        assert_eq!(
            relation
                .evaluate(&source_message, &[&auxiliary_witness], &[])
                .unwrap(),
            mu_prime,
        );
    }

    #[test]
    fn test_private_ood_answer_consistency() {
        // Invariant: the padded two-block evaluation equals the plain
        // power-basis evaluation on the concatenated vector.
        //
        //     padded(rho, f, r || s_pad) == ze*(rho) * (f || r || s_pad)^T
        let f = vec![ef(3), ef(7), ef(11)];
        let r = vec![ef(5), ef(9)];
        let s_pad = vec![ef(13)];

        // Concatenate the witness exactly as the OOD answer sees it.
        let mut concat = Vec::new();
        concat.extend_from_slice(&f);
        concat.extend_from_slice(&r);
        concat.extend_from_slice(&s_pad);

        let rho = ef(17);

        let from_concat = eval_ze_star_n(rho, &concat);
        let from_padded = padded_ood_t1(rho, &f, &[r.clone(), s_pad.clone()].concat());

        assert_eq!(
            from_concat, from_padded,
            "padded_ood_t1 vs eval_ze_star_n mismatch"
        );

        // Cross-check against a fully unrolled degree-5 evaluation.
        let r17 = rho;
        let expected = concat[0]
            + concat[1] * r17
            + concat[2] * r17 * r17
            + concat[3] * r17 * r17 * r17
            + concat[4] * r17 * r17 * r17 * r17
            + concat[5] * r17 * r17 * r17 * r17 * r17;

        assert_eq!(from_concat, expected, "OOD univariate evaluation mismatch");
    }

    #[test]
    fn test_output_relation_handles_empty_mask_and_zero_rho() {
        // Invariant: the algebra degenerates cleanly at two boundary shapes.
        //
        //     mask message : empty -> empty mask covector
        //     rho = 0      : ze*(0) = (1, 0, 0, ...) -> y = f[0]
        //
        // y = f[0] is also why rho = 0 costs 1/|F| of HVZK simulation error.
        let source_message = vec![ef(3), ef(5), ef(7)];
        let source_covector = vec![ef(11), ef(13), ef(17)];
        let source_row = vec![F::from_u64(2), F::ZERO, F::ONE];
        let empty_randomness_row: Vec<F> = Vec::new();
        let rho_ood_points = [EF::ZERO];
        let claim = ZkMaskClaim {
            base_claim_coeff: ef(19),
            residual_sumcheck_scale: EF::ONE,
            ood_coeffs: vec![ef(23)],
            in_domain_coeffs: vec![ef(29)],
        };

        // Honest transcript values for the degenerate shape.
        let inherited_claim = inner_product(&source_message, &source_covector);
        let private_ood = private_ood_answers(&rho_ood_points, &source_message, &[]);
        let source_opening = inner_product(&source_message, &lift_row(&source_row));
        let mu_prime = claim
            .batched_claim(inherited_claim, &private_ood, &[source_opening])
            .unwrap();
        // No randomness, no pad: the mask covector must come out empty.
        let relation = claim
            .output_relation_from_rows(
                source_message.len(),
                &source_covector,
                &[],
                0,
                0,
                &rho_ood_points,
                &[&source_row],
                &[&empty_randomness_row],
            )
            .unwrap();

        assert!(relation.mask_covector.is_empty());
        // rho = 0 selects the constant coefficient: a bare witness value.
        assert_eq!(
            private_ood[0], source_message[0],
            "rho = 0 should select the constant source coefficient",
        );
        assert_eq!(
            relation.evaluate(&source_message, &[], &[]).unwrap(),
            mu_prime,
        );
    }

    #[test]
    fn test_zero_evader_powers() {
        // Invariant: the batching zero-evader is the power sequence
        // (1, rho, rho^2, rho^3, ...).
        let rho = ef(5);
        let nu = rho.powers().collect_n(4);

        assert_eq!(nu[0], EF::ONE);
        assert_eq!(nu[1], rho);
        assert_eq!(nu[2], rho * rho);
        assert_eq!(nu[3], rho * rho * rho);
    }

    #[test]
    fn test_private_ood_answers_matches_padded_ood_t1() {
        // Invariant: the multi-point helper is exactly one padded evaluation
        // per OOD point, all over the same witness vector.
        let f = vec![ef(2), ef(4)];
        let mask = vec![ef(8), ef(16), ef(32)];
        let points = [ef(3), ef(5), ef(7)];

        let answers = private_ood_answers(&points, &f, &mask);
        let expected: Vec<EF> = points
            .iter()
            .map(|&rho| padded_ood_t1(rho, &f, &mask))
            .collect();

        assert_eq!(answers, expected);
    }

    #[test]
    fn test_ood_answers_leak_committed_data_without_enough_pad() {
        // Invariant: joint privacy of t_ood answers needs pad_len >= t_ood.
        //
        // Fixture state: 2 answers share a SINGLE fresh pad coordinate s.
        //
        //     y_1 = <(f || r), ze*(rho_1)> + s * rho_1^{ell+r_len}
        //     y_2 = <(f || r), ze*(rho_2)> + s * rho_2^{ell+r_len}
        //
        //     rho_2^{ell+r_len} * y_1 - rho_1^{ell+r_len} * y_2   // s cancels
        //     -> public linear functional of the committed (f || r)
        //     -> one leaked query against the source ZK budget
        let f = vec![ef(3), ef(5), ef(7)];
        let r = vec![ef(11), ef(13)];
        let s_pad = vec![ef(17)];
        let mask = [r.clone(), s_pad].concat();
        let rho = [ef(19), ef(23)];

        // Honest prover answers for the under-padded shape.
        let y = private_ood_answers(&rho, &f, &mask);

        // The pad coefficient in answer i is rho_i^{ell + r_len}.
        let committed = [f, r].concat();
        let shift = committed.len() as u64;
        let c1 = rho[0].exp_u64(shift);
        let c2 = rho[1].exp_u64(shift);

        // The pad contributions c1 * s and c2 * s cancel in the combination,
        // leaving only committed data.
        let eliminated = c2 * y[0] - c1 * y[1];
        let predicted =
            c2 * eval_ze_star_n(rho[0], &committed) - c1 * eval_ze_star_n(rho[1], &committed);

        assert_eq!(
            eliminated, predicted,
            "under-padded OOD answers must reveal this committed functional",
        );
    }

    #[test]
    fn test_simulated_verifier_view_matches_code_switch_relation() {
        // Invariant: fed honest transcript values, the simulator boundary
        // derives the same claim and relation as the honest path.
        let ell = 3;
        let r_len = 2;
        let s_pad_len = 1;
        let t_ood = 2;
        let t = 2;
        let iota = 1;
        let m = 8;

        let source_enc = make_rs_encoding(ell, r_len, m);

        // Honest witness: source message, encoding randomness, fresh pad.
        let f: Vec<EF> = vec![ef(4), ef(6), ef(10)];
        let r: Vec<EF> = vec![ef(13), ef(17)];
        let s_pad: Vec<EF> = vec![ef(19)];
        let mut mask_message = r;
        mask_message.extend_from_slice(&s_pad);

        // Real RS codeword for the source oracle, lifted to the extension.
        let f_base: Vec<F> = [4, 6, 10].into_iter().map(F::from_u64).collect();
        let r_base: Vec<F> = [13, 17].into_iter().map(F::from_u64).collect();
        let cw = source_enc.encode_with_randomness(&f_base, &r_base);
        let f_codeword: Vec<EF> = cw.values.iter().map(|&v| EF::from(v)).collect();

        // Inherited claim carries eps on the source part.
        let source_covector = vec![ef(23), ef(29), ef(31)];
        let eps = ef(47);
        let inherited_claim = eps * inner_product(&f, &source_covector);

        // Honest OOD answers and in-domain openings.
        let rho_ood_points = [ef(37), ef(41)];
        let private_ood = private_ood_answers(&rho_ood_points, &f, &mask_message);
        let query_positions = [1_usize, 4_usize];
        let source_openings: Vec<EF> = query_positions
            .iter()
            .map(|&position| f_codeword[position])
            .collect();

        // Batching coefficients as a power sequence.
        let nu = ef(43).powers().collect_n(1 + t_ood + t * iota);
        let claim = ZkMaskClaim {
            base_claim_coeff: nu[0],
            residual_sumcheck_scale: eps,
            ood_coeffs: nu[1..1 + t_ood].to_vec(),
            in_domain_coeffs: nu[1 + t_ood..].to_vec(),
        };

        // Feed the honest transcript values through the simulator boundary.
        let view = simulated_verifier_view::<F, EF, _>(
            &source_enc,
            inherited_claim,
            &source_covector,
            &[],
            r_len,
            s_pad_len,
            &rho_ood_points,
            &query_positions,
            &private_ood,
            &source_openings,
            &claim,
        )
        .unwrap();
        // Reference: the honest batching of the same scalars.
        let expected_mu = claim
            .batched_claim(inherited_claim, &private_ood, &source_openings)
            .expect("valid simulated transcript dimensions");
        let relation_value = view
            .output_relation
            .evaluate(&f, &[], &mask_message)
            .unwrap();

        // The view echoes its inputs and reproduces both sides of the identity.
        assert_eq!(view.private_ood_answers, private_ood);
        assert_eq!(view.source_openings, source_openings);
        assert_eq!(view.mu_prime, expected_mu);
        assert_eq!(
            relation_value, view.mu_prime,
            "deterministic simulator view must derive the same relation as the honest code-switch path"
        );
    }

    #[test]
    fn test_zk_sumcheck_simulator_eps_handoff_to_code_switch_view() {
        // Invariant: the real #1732 sumcheck handoff composes into a
        // satisfiable code-switch view, with eps applied exactly once.
        //
        //     simulator: samples transcript, fixes (eps, claimed_residual)
        //     replay   : verifies the transcript, recovers the same handoff
        let ell_zk = 4;
        let folding_factor = 2;
        let (perm, mmcs, sumcheck_encoding) = make_setup(91, ell_zk);
        let mut simulator_challenger = MyChallenger::new(perm.clone());
        let mut replay_challenger = MyChallenger::new(perm);
        let simulator_verifier = ZkVerifier::<F, EF>::new_prefix(&[]);
        let mut simulator_rng = SmallRng::seed_from_u64(92);
        // Run the HVZK sumcheck simulator end to end.
        let (zk_data, mask_commits, simulator_randomness) = simulate_classic_unpacked(
            &mut simulator_challenger,
            &simulator_verifier,
            folding_factor,
            0,
            &sumcheck_encoding,
            &mmcs,
            &mut simulator_rng,
        );
        // Replay the simulated transcript through the real verifier.
        let verifier_handoff = ZkVerifier::<F, EF>::new_prefix(&[])
            .into_sumcheck::<MyMmcs, _>(
                &zk_data,
                &mask_commits,
                ell_zk,
                folding_factor,
                0,
                &mut replay_challenger,
            )
            .expect("simulated HVZK sumcheck transcript should verify");
        assert_eq!(verifier_handoff.randomness, simulator_randomness);
        assert!(
            !verifier_handoff.eps.is_zero(),
            "fixture seed should produce a nonzero eps for the code-switch source scale",
        );

        // Source-side fixture: a small RS source oracle over the extension.
        let ell = 3;
        let source_randomness_len = 2;
        let pad_len = 1;
        let source_enc = ReedSolomonZkEncoding::<EF, Radix2Dit<EF>>::new(
            source_randomness_len,
            ell,
            8,
            Radix2Dit::default(),
        );
        let source_message = vec![ef(4), ef(6), ef(10)];
        let source_randomness = vec![ef(13), ef(17)];
        let s_pad = vec![ef(19)];
        let mut mask_message = source_randomness.clone();
        mask_message.extend_from_slice(&s_pad);
        let source_codeword =
            source_enc.encode_with_randomness(&source_message, &source_randomness);
        let query_positions = [1_usize, 4_usize];
        let source_openings = query_positions
            .iter()
            .map(|&position| source_codeword.values[position])
            .collect::<Vec<_>>();

        // Program a one-hot source covector so the inherited identity holds:
        //
        //     eps * <f, sl> = claimed_residual
        //     => sl[pivot] = claimed_residual / eps / f[pivot]
        let mut source_covector = EF::zero_vec(source_message.len());
        let (pivot, &pivot_value) = source_message
            .iter()
            .enumerate()
            .find(|(_, value)| !value.is_zero())
            .expect("fixture source message should contain a nonzero entry");
        source_covector[pivot] =
            verifier_handoff.claimed_residual / verifier_handoff.eps / pivot_value;

        // Honest OOD answers and batching coefficients.
        let rho_ood_points = [ef(37), ef(41)];
        let private_ood = private_ood_answers(&rho_ood_points, &source_message, &mask_message);
        let nu = ef(43)
            .powers()
            .collect_n(1 + rho_ood_points.len() + source_openings.len());
        let claim = ZkMaskClaim {
            base_claim_coeff: nu[0],
            residual_sumcheck_scale: verifier_handoff.eps,
            ood_coeffs: nu[1..1 + rho_ood_points.len()].to_vec(),
            in_domain_coeffs: nu[1 + rho_ood_points.len()..].to_vec(),
        };

        // The handoff composes into a satisfiable code-switch view.
        let view = simulated_verifier_view::<EF, EF, _>(
            &source_enc,
            verifier_handoff.claimed_residual,
            &source_covector,
            &[],
            source_randomness_len,
            pad_len,
            &rho_ood_points,
            &query_positions,
            &private_ood,
            &source_openings,
            &claim,
        )
        .expect("programmed code-switch view should have valid dimensions");
        let expected_mu = claim
            .batched_claim(
                verifier_handoff.claimed_residual,
                &private_ood,
                &source_openings,
            )
            .expect("valid batched claim dimensions");
        let relation_value = view
            .output_relation
            .evaluate(&source_message, &[], &mask_message)
            .expect("programmed code-switch relation should evaluate");

        assert_eq!(view.mu_prime, expected_mu);
        assert_eq!(relation_value, view.mu_prime);
        assert_eq!(view.private_ood_answers.len(), rho_ood_points.len());
        assert_eq!(view.source_openings.len(), query_positions.len());
    }

    #[test]
    fn test_programmable_simulator_components_for_zk_source_view() {
        // This fixed seed is chosen to make the #1732 simulator sample nonzero eps;
        // the proptest below skips the rare zero-eps cases explicitly.
        assert!(
            assert_programmable_simulator_components_for_zk_source_view(101),
            "fixture seed must produce nonzero eps; choose another seed if simulator sampling changes"
        );
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(8))]

        #[test]
        fn prop_programmable_simulator_components_for_zk_source_view(seed in 0_u64..64) {
            prop_assume!(assert_programmable_simulator_components_for_zk_source_view(
                seed.wrapping_add(101),
            ));
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn prop_construction_9_7_mu_prime_identity(seed in any::<u64>()) {
            // Invariant: for every parameter shape and field values,
            //
            //     nu_1*mu + sum nu*y_i + sum nu*f(x_j)
            //   = <f, sl'> + sum <aux_i, u'_i> + <(r || s_pad), mask'>
            //
            // Dimensions derive from the seed.
            // Values come from a seeded RNG, so failures replay deterministically.
            let mut rng = SmallRng::seed_from_u64(seed);
            // Source message length: 1..=5.
            let ell = 1 + (seed % 5) as usize;
            // Source randomness length: 0..=2.
            let r_len = ((seed / 5) % 3) as usize;
            // Fresh pad length: 0..=2.
            let pad_len = ((seed / 15) % 3) as usize;
            // Out-of-domain answer count: 0..=2.
            let t_ood = ((seed / 45) % 3) as usize;
            // In-domain opening count: 0..=2.
            let t = ((seed / 135) % 3) as usize;
            // Carried auxiliary mask count: 0..=2.
            let n_aux = ((seed / 405) % 3) as usize;

            // Honest witness: source message, source randomness, fresh pad.
            let f = draw_ef_vec(&mut rng, ell);
            let r = draw_ef_vec(&mut rng, r_len);
            let s_pad = draw_ef_vec(&mut rng, pad_len);
            let mask_message = [r.clone(), s_pad].concat();

            // Inherited relation: eps-scaled source part plus auxiliary parts.
            //
            //     mu = eps * <f, sl> + sum_i <aux_w_i, aux_cov_i>
            let sl = draw_ef_vec(&mut rng, ell);
            let eps: EF = rng.random();
            let aux_witnesses: Vec<Vec<EF>> =
                (0..n_aux).map(|_| draw_ef_vec(&mut rng, 2)).collect();
            let aux_covectors: Vec<Vec<EF>> =
                (0..n_aux).map(|_| draw_ef_vec(&mut rng, 2)).collect();
            let mut inherited = eps * inner_product(&f, &sl);
            for (w, c) in aux_witnesses.iter().zip(&aux_covectors) {
                inherited += inner_product(w, c);
            }

            // Synthetic generator rows: the identity is row-agnostic, so any
            // rows of the right widths exercise the full algebra.
            let message_rows: Vec<Vec<EF>> =
                (0..t).map(|_| draw_ef_vec(&mut rng, ell)).collect();
            let randomness_rows: Vec<Vec<EF>> =
                (0..t).map(|_| draw_ef_vec(&mut rng, r_len)).collect();
            // Honest openings: f(x) = <f, G^#[x]> + <r, G^$[x]>.
            let openings: Vec<EF> = message_rows
                .iter()
                .zip(&randomness_rows)
                .map(|(mr, rr)| inner_product(&f, mr) + inner_product(&r, rr))
                .collect();

            // Honest OOD answers over (f || r || s_pad), arbitrary points.
            let rho_points = draw_ef_vec(&mut rng, t_ood);
            let y = private_ood_answers(&rho_points, &f, &mask_message);

            // Arbitrary batching coefficients, as sampled by the verifier.
            let nu = draw_ef_vec(&mut rng, 1 + t_ood + t);
            let claim = ZkMaskClaim {
                base_claim_coeff: nu[0],
                residual_sumcheck_scale: eps,
                ood_coeffs: nu[1..1 + t_ood].to_vec(),
                in_domain_coeffs: nu[1 + t_ood..].to_vec(),
            };

            // Verifier side: batch the transcript scalars.
            let mu_prime = claim.batched_claim(inherited, &y, &openings).unwrap();

            // Relation side: assemble covectors and evaluate the witness.
            // Owned rows are borrowed through their slice view directly.
            let aux_cov_refs: Vec<&[EF]> =
                aux_covectors.iter().map(Vec::as_slice).collect();
            let relation = claim
                .output_relation_from_rows(
                    ell,
                    &sl,
                    &aux_cov_refs,
                    r_len,
                    pad_len,
                    &rho_points,
                    &message_rows,
                    &randomness_rows,
                )
                .unwrap();
            let aux_wit_refs: Vec<&[EF]> =
                aux_witnesses.iter().map(Vec::as_slice).collect();
            let value = relation.evaluate(&f, &aux_wit_refs, &mask_message).unwrap();

            prop_assert_eq!(mu_prime, value);
        }
    }

    /// Runs the programmable simulator-boundary scenario for one seed.
    ///
    /// # Returns
    ///
    /// `false` only when the sampled eps is zero (the caller skips the case);
    /// every other deviation panics through the internal assertions.
    fn assert_programmable_simulator_components_for_zk_source_view(seed: u64) -> bool {
        // Stage 1: real HVZK sumcheck simulator plus verifier replay.
        let ell_zk = 4;
        let folding_factor = 2;
        let (perm, mmcs, sumcheck_encoding) = make_setup(seed, ell_zk);
        let mut simulator_challenger = MyChallenger::new(perm.clone());
        let mut replay_challenger = MyChallenger::new(perm);
        let simulator_verifier = ZkVerifier::<F, EF>::new_prefix(&[]);
        let mut simulator_rng = SmallRng::seed_from_u64(seed.wrapping_add(1));
        let (zk_data, mask_commits, simulator_randomness) = simulate_classic_unpacked(
            &mut simulator_challenger,
            &simulator_verifier,
            folding_factor,
            0,
            &sumcheck_encoding,
            &mmcs,
            &mut simulator_rng,
        );
        let verifier_handoff = ZkVerifier::<F, EF>::new_prefix(&[])
            .into_sumcheck::<MyMmcs, _>(
                &zk_data,
                &mask_commits,
                ell_zk,
                folding_factor,
                0,
                &mut replay_challenger,
            )
            .expect("simulated HVZK sumcheck transcript should verify");
        assert_eq!(verifier_handoff.randomness, simulator_randomness);
        // The pivot programming below divides by eps; skip the measure-zero case.
        if verifier_handoff.eps.is_zero() {
            return false;
        }

        // Stage 2: source-side fixture.
        let ell = 3;
        let source_randomness_len = 2;
        let pad_len = 1;
        let source_enc = ReedSolomonZkEncoding::<EF, Radix2Dit<EF>>::new(
            source_randomness_len,
            ell,
            8,
            Radix2Dit::default(),
        );
        let source_message = vec![ef(4), ef(6), ef(10)];

        // Program a one-hot source covector hitting the inherited residual:
        //
        //     eps * <f, sl> = claimed_residual
        let mut source_covector = EF::zero_vec(source_message.len());
        let (pivot, &pivot_value) = source_message
            .iter()
            .enumerate()
            .find(|(_, value)| !value.is_zero())
            .expect("fixture source message should contain a nonzero entry");
        source_covector[pivot] =
            verifier_handoff.claimed_residual / verifier_handoff.eps / pivot_value;

        // Stage 3: programmability of the source-code openings.
        //
        //     simulator: samples 2 openings uniformly
        //     witness  : solve the 2x2 system for the encoding randomness
        //                that explains both openings honestly
        let query_positions = [1_usize, 4_usize];
        let mut code_sim_rng = SmallRng::seed_from_u64(seed.wrapping_add(2));
        let simulated_source_openings = source_enc.simulate(&query_positions, &mut code_sim_rng);
        let source_randomness = solve_two_randomness_for_openings(
            &source_enc,
            &source_message,
            &query_positions,
            &simulated_source_openings,
        );
        let source_codeword =
            source_enc.encode_with_randomness(&source_message, &source_randomness);
        let honest_source_openings = query_positions
            .iter()
            .map(|&position| source_codeword.values[position])
            .collect::<Vec<_>>();
        assert_eq!(
            honest_source_openings, simulated_source_openings,
            "Sim_C' openings must be explainable by an honest RS randomness witness",
        );

        // Stage 4: programmability of the private OOD answer.
        //
        //     simulator: samples y uniformly
        //     witness  : solve the single pad coordinate hitting y
        let rho_ood_points = [ef(37)];
        let mut ze_sim_rng = SmallRng::seed_from_u64(seed.wrapping_add(3));
        let simulated_private_ood = vec![ze_sim_rng.random::<EF>()];
        let s_pad = vec![solve_pad_for_private_ood(
            rho_ood_points[0],
            &source_message,
            &source_randomness,
            simulated_private_ood[0],
        )];
        let mut mask_message = source_randomness;
        mask_message.extend_from_slice(&s_pad);
        let honest_private_ood =
            private_ood_answers(&rho_ood_points, &source_message, &mask_message);
        assert_eq!(
            honest_private_ood, simulated_private_ood,
            "S_ze_ood output must be explainable by the Lemma 9.3 programmed pad witness",
        );

        // Stage 5: programmability of the fresh mask-oracle openings.
        //
        //     simulator: samples 2 mask openings uniformly
        //     witness  : solve the mask encoding randomness explaining them
        let mask_enc = ReedSolomonZkEncoding::<EF, Radix2Dit<EF>>::new(
            source_randomness_len,
            mask_message.len(),
            8,
            Radix2Dit::default(),
        );
        let simulated_mask_openings = mask_enc.simulate(&query_positions, &mut code_sim_rng);
        assert_eq!(
            simulated_mask_openings.len(),
            query_positions.len(),
            "Sim_C_zk must produce one opening per mask query",
        );
        let mask_randomness = solve_two_randomness_for_openings(
            &mask_enc,
            &mask_message,
            &query_positions,
            &simulated_mask_openings,
        );
        let mask_codeword = mask_enc.encode_with_randomness(&mask_message, &mask_randomness);
        let honest_mask_openings = query_positions
            .iter()
            .map(|&position| mask_codeword.values[position])
            .collect::<Vec<_>>();
        assert_eq!(
            honest_mask_openings, simulated_mask_openings,
            "Sim_C_zk openings must be explainable by an honest RS randomness witness",
        );

        // Stage 6: witness-independent coupling certificate.
        //
        //     fresh round mask : sampled independently of the witness
        //     -> matched RNG streams give byte-equal mask commitments
        //
        // Witness-dependent parts (source commitment, openings, OOD answers)
        // are covered by the programmability stages above instead.
        let round_mask_encoding = ReedSolomonZkEncoding::<EF, Radix2Dit<EF>>::new(
            source_randomness_len,
            mask_message.len(),
            8,
            Radix2Dit::default(),
        );
        let mut real_mask_rng = SmallRng::seed_from_u64(seed.wrapping_add(4));
        let mut sim_mask_rng = SmallRng::seed_from_u64(seed.wrapping_add(4));
        let real_round_mask = (0..mask_message.len())
            .map(|_| real_mask_rng.random::<EF>())
            .collect::<Vec<_>>();
        let sim_round_mask = (0..mask_message.len())
            .map(|_| sim_mask_rng.random::<EF>())
            .collect::<Vec<_>>();
        let real_mask_codeword = round_mask_encoding.encode(&real_round_mask, &mut real_mask_rng);
        let sim_mask_codeword = round_mask_encoding.encode(&sim_round_mask, &mut sim_mask_rng);
        let (real_mask_commitment, _) = mmcs.commit_matrix(real_mask_codeword);
        let (sim_mask_commitment, _) = mmcs.commit_matrix(sim_mask_codeword);
        assert_eq!(real_round_mask, sim_round_mask);
        assert_eq!(
            real_mask_commitment, sim_mask_commitment,
            "witness-independent code-switch masks should couple under matched RNG",
        );

        // Stage 7: the programmed components compose into a satisfiable view.
        let nu = ef(43)
            .powers()
            .collect_n(1 + rho_ood_points.len() + query_positions.len());
        let claim = ZkMaskClaim {
            base_claim_coeff: nu[0],
            residual_sumcheck_scale: verifier_handoff.eps,
            ood_coeffs: nu[1..1 + rho_ood_points.len()].to_vec(),
            in_domain_coeffs: nu[1 + rho_ood_points.len()..].to_vec(),
        };

        let simulated_view = simulated_verifier_view::<EF, EF, _>(
            &source_enc,
            verifier_handoff.claimed_residual,
            &source_covector,
            &[],
            source_randomness_len,
            pad_len,
            &rho_ood_points,
            &query_positions,
            &simulated_private_ood,
            &simulated_source_openings,
            &claim,
        )
        .expect("programmed code-switch view should have valid dimensions");

        // The programmed witness satisfies the derived output relation.
        assert_eq!(
            simulated_view
                .output_relation
                .evaluate(&source_message, &[], &mask_message)
                .expect("programmed witness should satisfy the output relation"),
            simulated_view.mu_prime,
        );

        true
    }

    #[test]
    fn test_round0_zero_randomness_source_view_is_deterministic() {
        // Invariant: a round-0 source oracle has no encoding randomness.
        //
        //     openings     : deterministic (no randomness term)
        //     programmable : only the private OOD answer, via the fresh pad
        let source_randomness_len = 0;
        let pad_len = 1;
        let source_enc = ReedSolomonZkEncoding::<EF, Radix2Dit<EF>>::new(
            source_randomness_len,
            3,
            8,
            Radix2Dit::default(),
        );
        let source_message = vec![ef(4), ef(6), ef(10)];
        let source_randomness = Vec::new();
        // Deterministic openings: no randomness term in the decomposition.
        let source_codeword =
            source_enc.encode_with_randomness(&source_message, &source_randomness);
        let query_positions = [1_usize, 4_usize];
        let source_openings = query_positions
            .iter()
            .map(|&position| source_codeword.values[position])
            .collect::<Vec<_>>();

        let source_covector = vec![ef(7), ef(11), ef(13)];
        let inherited_claim = inner_product(&source_message, &source_covector);
        // Program the single pad coordinate to hit an arbitrary target answer.
        let rho_ood_points = [ef(37)];
        let target_ood = ef(91);
        let s_pad = vec![solve_pad_for_private_ood(
            rho_ood_points[0],
            &source_message,
            &source_randomness,
            target_ood,
        )];
        let mask_message = s_pad;
        assert_eq!(mask_message.len(), pad_len);
        let private_ood = private_ood_answers(&rho_ood_points, &source_message, &mask_message);
        assert_eq!(private_ood, vec![target_ood]);

        // Batching coefficients as a power sequence.
        let nu = ef(43)
            .powers()
            .collect_n(1 + rho_ood_points.len() + query_positions.len());
        let claim = ZkMaskClaim {
            base_claim_coeff: nu[0],
            residual_sumcheck_scale: EF::ONE,
            ood_coeffs: nu[1..1 + rho_ood_points.len()].to_vec(),
            in_domain_coeffs: nu[1 + rho_ood_points.len()..].to_vec(),
        };

        // The programmed round-0 view is satisfiable by the honest witness.
        let view = simulated_verifier_view::<EF, EF, _>(
            &source_enc,
            inherited_claim,
            &source_covector,
            &[],
            source_randomness_len,
            pad_len,
            &rho_ood_points,
            &query_positions,
            &private_ood,
            &source_openings,
            &claim,
        )
        .expect("round-0 zero-randomness view should have valid dimensions");

        assert_eq!(
            view.output_relation
                .evaluate(&source_message, &[], &mask_message)
                .expect("round-0 witness should satisfy the output relation"),
            view.mu_prime,
        );
    }

    #[test]
    fn test_simulated_verifier_view_rejects_private_ood_count_mismatch() {
        // Fixture state: 2 OOD coefficients but only 1 simulated answer.
        //
        //     ood_coeffs: [nu_2, nu_3]
        //     answers   : [y_1]          -> 1 != 2 -> reject
        let enc = make_rs_encoding(3, 2, 8);
        let claim = ZkMaskClaim {
            base_claim_coeff: ef(1),
            residual_sumcheck_scale: ef(1),
            ood_coeffs: vec![ef(2), ef(3)],
            in_domain_coeffs: vec![ef(4)],
        };

        let err = simulated_verifier_view::<F, EF, _>(
            &enc,
            ef(9),
            &[ef(1), ef(2), ef(3)],
            &[],
            2,
            1,
            &[ef(7), ef(8)],
            &[0],
            &[ef(10)],
            &[ef(11)],
            &claim,
        )
        .unwrap_err();

        assert_eq!(
            err,
            CodeSwitchError::PrivateOodAnswerCountMismatch {
                expected: 2,
                actual: 1
            }
        );
    }

    #[test]
    fn test_batched_claim_rejects_ood_count_mismatch() {
        // Fixture state: 2 OOD coefficients but only 1 answer.
        //
        //     ood_coeffs: [nu_2, nu_3]
        //     answers   : [y_1]          -> 1 != 2 -> reject
        let claim = ZkMaskClaim {
            base_claim_coeff: ef(1),
            residual_sumcheck_scale: ef(1),
            ood_coeffs: vec![ef(2), ef(3)],
            in_domain_coeffs: vec![ef(4)],
        };

        let err = claim
            .batched_claim(ef(9), &[ef(10)], &[ef(11)])
            .unwrap_err();

        assert_eq!(
            err,
            CodeSwitchError::PrivateOodAnswerCountMismatch {
                expected: 2,
                actual: 1
            }
        );
    }

    #[test]
    fn test_batched_claim_rejects_source_opening_count_mismatch() {
        // Fixture state: 2 in-domain coefficients but only 1 opening.
        //
        //     in_domain_coeffs: [nu_3, nu_4]
        //     openings        : [f(x_1)]     -> 1 != 2 -> reject
        let claim = ZkMaskClaim {
            base_claim_coeff: ef(1),
            residual_sumcheck_scale: ef(1),
            ood_coeffs: vec![ef(2)],
            in_domain_coeffs: vec![ef(3), ef(4)],
        };

        let err = claim
            .batched_claim(ef(9), &[ef(10)], &[ef(11)])
            .unwrap_err();

        assert_eq!(
            err,
            CodeSwitchError::SourceOpeningCountMismatch {
                expected: 2,
                actual: 1
            }
        );
    }

    #[test]
    fn test_output_relation_rejects_query_count_mismatch() {
        // Fixture state: 2 in-domain coefficients but only 1 query position.
        //
        //     in_domain_coeffs: [nu_2, nu_3]
        //     positions       : [x_1]        -> 1 != 2 -> reject
        let enc = make_rs_encoding(3, 2, 8);
        let claim = ZkMaskClaim {
            base_claim_coeff: ef(1),
            residual_sumcheck_scale: ef(1),
            ood_coeffs: vec![ef(2)],
            in_domain_coeffs: vec![ef(3), ef(4)],
        };

        let err = claim
            .output_relation::<F, _>(&enc, &[ef(1), ef(2), ef(3)], &[], 2, 1, &[ef(9)], &[0])
            .unwrap_err();

        assert_eq!(
            err,
            CodeSwitchError::QueryPositionCountMismatch {
                expected: 2,
                actual: 1
            }
        );
    }

    #[test]
    fn test_output_relation_rejects_source_randomness_row_count_mismatch() {
        // Fixture state: 1 message row but 0 randomness rows.
        //
        // Both row lists are indexed by the same query list, so their counts
        // must agree with the in-domain coefficients independently.
        let claim = ZkMaskClaim {
            base_claim_coeff: ef(1),
            residual_sumcheck_scale: ef(1),
            ood_coeffs: vec![ef(2)],
            in_domain_coeffs: vec![ef(3)],
        };
        let source_row = vec![F::ONE, F::ZERO, F::ZERO];
        let err = claim
            .output_relation_from_rows(
                3,
                &[ef(1), ef(2), ef(3)],
                &[],
                2,
                1,
                &[ef(9)],
                &[&source_row],
                &[],
            )
            .unwrap_err();

        assert_eq!(
            err,
            CodeSwitchError::SourceRandomnessRowCountMismatch {
                expected: 1,
                actual: 0,
            }
        );
    }

    #[test]
    fn test_output_relation_rejects_source_message_row_length_mismatch() {
        // Fixture state: declared source width 3, but a G^# row of width 2.
        //
        //     row: [1, 0]    declared width: 3 -> 2 != 3 -> reject
        let claim = ZkMaskClaim {
            base_claim_coeff: ef(1),
            residual_sumcheck_scale: ef(1),
            ood_coeffs: Vec::new(),
            in_domain_coeffs: vec![ef(3)],
        };
        let short_message_row = vec![F::ONE, F::ZERO];
        let randomness_row = vec![F::ONE, F::ZERO];
        let err = claim
            .output_relation_from_rows(
                3,
                &[ef(1), ef(2), ef(3)],
                &[],
                2,
                1,
                &[],
                &[&short_message_row],
                &[&randomness_row],
            )
            .unwrap_err();

        assert_eq!(
            err,
            CodeSwitchError::SourceMessageRowLengthMismatch {
                expected: 3,
                actual: 2,
            }
        );
    }

    #[test]
    fn test_output_relation_rejects_source_randomness_row_length_mismatch() {
        // Fixture state: declared randomness width 2, but a G^$ row of width 1.
        //
        //     row: [1]    declared width: 2 -> 1 != 2 -> reject
        let claim = ZkMaskClaim {
            base_claim_coeff: ef(1),
            residual_sumcheck_scale: ef(1),
            ood_coeffs: Vec::new(),
            in_domain_coeffs: vec![ef(3)],
        };
        let message_row = vec![F::ONE, F::ZERO, F::ZERO];
        let short_randomness_row = vec![F::ONE];
        let err = claim
            .output_relation_from_rows(
                3,
                &[ef(1), ef(2), ef(3)],
                &[],
                2,
                1,
                &[],
                &[&message_row],
                &[&short_randomness_row],
            )
            .unwrap_err();

        assert_eq!(
            err,
            CodeSwitchError::SourceRandomnessRowLengthMismatch {
                expected: 2,
                actual: 1,
            }
        );
    }

    #[test]
    fn test_output_relation_rejects_source_covector_length_mismatch() {
        // Fixture state: encoding message width 3, but a covector of width 2.
        //
        //     covector: [a, b]    message width: 3 -> 2 != 3 -> reject
        let enc = make_rs_encoding(3, 2, 8);
        let claim = ZkMaskClaim {
            base_claim_coeff: ef(1),
            residual_sumcheck_scale: ef(1),
            ood_coeffs: vec![ef(2)],
            in_domain_coeffs: vec![ef(3)],
        };

        let err = claim
            .output_relation::<F, _>(&enc, &[ef(1), ef(2)], &[], 2, 1, &[ef(9)], &[0])
            .unwrap_err();

        assert_eq!(
            err,
            CodeSwitchError::SourceCovectorLengthMismatch {
                expected: 3,
                actual: 2
            }
        );
    }

    #[test]
    fn test_evaluate_rejects_dimension_mismatches() {
        // Fixture relation: source width 2, one auxiliary of width 2, mask width 1.
        let relation = CodeSwitchOutputRelation {
            source_covector: vec![ef(1), ef(2)],
            auxiliary_covectors: vec![vec![ef(3), ef(4)]],
            mask_covector: vec![ef(5)],
        };
        let source = [ef(6), ef(7)];
        let aux = [ef(8), ef(9)];
        let mask = [ef(10)];

        // Source witness of width 1 against a covector of width 2.
        assert_eq!(
            relation.evaluate(&[ef(6)], &[&aux], &mask).unwrap_err(),
            CodeSwitchError::SourceMessageLengthMismatch {
                expected: 2,
                actual: 1,
            }
        );
        // Zero auxiliary witnesses against one carried covector.
        assert_eq!(
            relation.evaluate(&source, &[], &mask).unwrap_err(),
            CodeSwitchError::AuxiliaryWitnessCountMismatch {
                expected: 1,
                actual: 0,
            }
        );
        // Auxiliary witness of width 1 against a covector of width 2.
        assert_eq!(
            relation.evaluate(&source, &[&aux[..1]], &mask).unwrap_err(),
            CodeSwitchError::AuxiliaryWitnessLengthMismatch {
                index: 0,
                expected: 2,
                actual: 1,
            }
        );
        // Empty mask witness against a mask covector of width 1.
        assert_eq!(
            relation.evaluate(&source, &[&aux], &[]).unwrap_err(),
            CodeSwitchError::MaskMessageLengthMismatch {
                expected: 1,
                actual: 0,
            }
        );
    }

    #[test]
    fn test_rs_row_decomposition_matches_encoding() {
        // Invariant: every codeword symbol decomposes through the generator
        // rows as cw[x] = <msg, G^#[x]> + <rand, G^$[x]>.
        //
        // This is the Definition 3.17 split the code-switch relies on.
        let msg_len = 4;
        let t = 2;
        let m = 8;
        let enc = make_rs_encoding(msg_len, t, m);

        let msg: Vec<F> = (1..=msg_len as u64).map(F::from_u64).collect();
        let rand: Vec<F> = (10..10 + t as u64).map(F::from_u64).collect();
        let cw = enc.encode_with_randomness(&msg, &rand);

        // Check the decomposition at every domain position.
        for i in 0..m {
            let m_dot: F = enc
                .message_row(i)
                .iter()
                .zip(&msg)
                .map(|(a, b)| *a * *b)
                .sum();
            let r_dot: F = enc
                .randomness_row(i)
                .iter()
                .zip(&rand)
                .map(|(a, b)| *a * *b)
                .sum();
            assert_eq!(
                cw.values[i],
                m_dot + r_dot,
                "Row decomposition failed at position {i}"
            );
        }
    }
}
