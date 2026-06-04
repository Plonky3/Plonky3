//! HVZK code-switching round (Construction 9.7, eprint 2026/391 Section 9.4).
//!
//! This module provides the Construction 9.7 algebra that the next #1587 WHIR
//! adapter PR will consume. It intentionally does not yet add proof payloads or
//! change the PCS round loop.
//!
//! Reduces a proximity claim about oracle `f` w.r.t. source code `C` to a
//! proximity claim about oracle `g` w.r.t. a smaller target code `C'`.
//!
//! The ZK variant adds:
//! 1. A fresh mask oracle `s = Enc_{C_zk}((r, s_pad), r'')`.
//! 2. A private-zero-evader OOD answer `y = ze_ood(rho) · (f, r, s_pad)^T`.
//!
//! The algebra and identity tests use `p3-zk-codes` (`LinearZkEncoding`)
//! and `whir::utils` (zero-evader helpers) directly.
//!
//! # Sumcheck handoff
//!
//! The HVZK sumcheck handoff from #1732 carries an `eps`-scaled residual
//! claim. The code-switch relation keeps that scale explicit so composing
//! Construction 9.7 after Construction 6.3 scales only the inherited source
//! covector, not carried auxiliary mask covectors.
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
/// Produced by the batching step (both prover and verifier) and consumed by
/// the next ZK sumcheck relation.
///
/// TODO(#1587): construct this from the round transcript when wiring the
/// round-0 WHIR adapter path.
#[derive(Debug, Clone)]
pub struct ZkMaskClaim<EF> {
    /// Coefficient on the inherited source claim (`nu_1`).
    pub base_claim_coeff: EF,
    /// Scale applied by the HVZK sumcheck to the source residual.
    ///
    /// For the HVZK sumcheck handoff this is the sampled `eps`. The incoming
    /// scalar claim must already include this scale on the source part. This
    /// value is kept separately so the output relation can apply `eps` only to
    /// the source covector, not to carried auxiliary mask covectors.
    ///
    /// Auxiliary covectors are `eps`-opaque to this builder: if an inherited
    /// auxiliary claim already contains an `eps` scale, the caller must pass the
    /// correspondingly scaled auxiliary covector.
    pub residual_sumcheck_scale: EF,
    /// Batching coefficients for OOD answers (`nu_{1+i}` for `i in [t_ood]`).
    pub ood_coeffs: Vec<EF>,
    /// Batching coefficients for in-domain openings.
    pub in_domain_coeffs: Vec<EF>,
}

impl<EF: Field> ZkMaskClaim<EF> {
    /// Computes the verifier-side batched claim `mu'`.
    ///
    /// The inherited claim is the scalar already handed off by the previous
    /// IOR. In the HVZK sumcheck composition that means its source residual has
    /// already been scaled by `eps`, while carried auxiliary mask claims remain
    /// in the relation with their own coefficients.
    /// Consequently this function does not apply
    /// [`Self::residual_sumcheck_scale`]; that scale is used only when
    /// constructing the output source covector.
    ///
    /// ```text
    /// mu' = nu_1 * mu
    ///     + sum_i nu_{1+i} * y_i
    ///     + sum_j nu_{1+t_ood+j} * f(x_j)
    /// ```
    pub fn batched_claim(
        &self,
        inherited_claim: EF,
        private_ood_answers: &[EF],
        source_openings: &[EF],
    ) -> Result<EF, CodeSwitchError> {
        if private_ood_answers.len() != self.ood_coeffs.len() {
            return Err(CodeSwitchError::PrivateOodAnswerCountMismatch {
                expected: self.ood_coeffs.len(),
                actual: private_ood_answers.len(),
            });
        }
        if source_openings.len() != self.in_domain_coeffs.len() {
            return Err(CodeSwitchError::SourceOpeningCountMismatch {
                expected: self.in_domain_coeffs.len(),
                actual: source_openings.len(),
            });
        }

        let ood_sum: EF = self
            .ood_coeffs
            .iter()
            .zip(private_ood_answers)
            .map(|(&coeff, &answer)| coeff * answer)
            .sum();
        let in_domain_sum: EF = self
            .in_domain_coeffs
            .iter()
            .zip(source_openings)
            .map(|(&coeff, &opening)| coeff * opening)
            .sum();

        Ok(self.base_claim_coeff * inherited_claim + ood_sum + in_domain_sum)
    }

    /// Builds the output relation covectors for Construction 9.7.
    ///
    /// TODO(#1587): this `LinearZkEncoding` convenience form is currently used
    /// by algebra and simulator-boundary tests. The adapter wiring should call
    /// [`Self::output_relation_from_rows`] with rows supplied by the committed
    /// WHIR source oracle.
    ///
    /// `query_positions` are flattened codeword positions. For interleaving
    /// depth `iota`, callers should pass `iota * x_i + limb` for every queried
    /// limb.
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
        EF: From<F>,
        Enc: LinearZkEncoding<F>,
    {
        if rho_ood_points.len() != self.ood_coeffs.len() {
            return Err(CodeSwitchError::OodPointCountMismatch {
                expected: self.ood_coeffs.len(),
                actual: rho_ood_points.len(),
            });
        }
        if query_positions.len() != self.in_domain_coeffs.len() {
            return Err(CodeSwitchError::QueryPositionCountMismatch {
                expected: self.in_domain_coeffs.len(),
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

        let source_rows = query_positions
            .iter()
            .map(|&position| source_encoding.message_row(position))
            .collect::<Vec<_>>();
        let source_randomness_rows = query_positions
            .iter()
            .map(|&position| source_encoding.randomness_row(position))
            .collect::<Vec<_>>();
        let source_row_refs = source_rows.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let source_randomness_row_refs = source_randomness_rows
            .iter()
            .map(Vec::as_slice)
            .collect::<Vec<_>>();

        self.output_relation_from_rows(
            source_len,
            source_covector,
            auxiliary_covectors,
            source_randomness_len,
            pad_len,
            rho_ood_points,
            &source_row_refs,
            &source_randomness_row_refs,
        )
    }

    /// Builds the output relation covectors from explicit source-code rows.
    ///
    /// This is the row-level form of Construction 9.7 intended for the WHIR
    /// adapter, where the source rows come from the folded WHIR target oracle
    /// rather than a standalone `LinearZkEncoding`.
    ///
    /// TODO(#1587): plug this into the round-0 adapter path after the proof
    /// payload and `RoundZkConfig` slice land.
    #[allow(clippy::too_many_arguments)]
    pub fn output_relation_from_rows<Row>(
        &self,
        source_message_len: usize,
        source_covector: &[EF],
        auxiliary_covectors: &[&[EF]],
        source_randomness_len: usize,
        pad_len: usize,
        rho_ood_points: &[EF],
        source_rows: &[&[Row]],
        source_randomness_rows: &[&[Row]],
    ) -> Result<CodeSwitchOutputRelation<EF>, CodeSwitchError>
    where
        Row: Copy,
        EF: From<Row>,
    {
        if rho_ood_points.len() != self.ood_coeffs.len() {
            return Err(CodeSwitchError::OodPointCountMismatch {
                expected: self.ood_coeffs.len(),
                actual: rho_ood_points.len(),
            });
        }
        if source_rows.len() != self.in_domain_coeffs.len() {
            return Err(CodeSwitchError::QueryPositionCountMismatch {
                expected: self.in_domain_coeffs.len(),
                actual: source_rows.len(),
            });
        }
        if source_randomness_rows.len() != self.in_domain_coeffs.len() {
            return Err(CodeSwitchError::SourceRandomnessRowCountMismatch {
                expected: self.in_domain_coeffs.len(),
                actual: source_randomness_rows.len(),
            });
        }
        if source_covector.len() != source_message_len {
            return Err(CodeSwitchError::SourceCovectorLengthMismatch {
                expected: source_message_len,
                actual: source_covector.len(),
            });
        }

        let mask_len = source_randomness_len + pad_len;
        let source_inherited_scale = self.base_claim_coeff * self.residual_sumcheck_scale;
        let auxiliary_inherited_scale = self.base_claim_coeff;

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

        for (&rho, &coeff) in rho_ood_points.iter().zip(&self.ood_coeffs) {
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

        for ((message_row, randomness_row), &coeff) in source_rows
            .iter()
            .zip(source_randomness_rows)
            .zip(&self.in_domain_coeffs)
        {
            if message_row.len() != source_message_len {
                return Err(CodeSwitchError::SourceMessageRowLengthMismatch {
                    expected: source_message_len,
                    actual: message_row.len(),
                });
            }
            for (dst, &row) in next_source_covector.iter_mut().zip(*message_row) {
                *dst += coeff * EF::from(row);
            }

            if randomness_row.len() != source_randomness_len {
                return Err(CodeSwitchError::SourceRandomnessRowLengthMismatch {
                    expected: source_randomness_len,
                    actual: randomness_row.len(),
                });
            }
            for (dst, row) in mask_covector
                .iter_mut()
                .take(source_randomness_len)
                .zip(*randomness_row)
            {
                *dst += coeff * EF::from(*row);
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
/// TODO(#1587): this is the deterministic Lemma 9.8 boundary used by tests
/// until the full PCS-level simulator is wired through the WHIR adapter.
///
/// This intentionally does not sample randomness itself. The caller supplies
/// the values generated by the private zero-evader simulator and by simulated
/// oracle openings; this helper checks the transcript dimensions and derives
/// the exact `mu'` and output linear relation that the composed verifier sees.
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

        let mut value = dot_product::<EF, _, _>(
            source_message.iter().copied(),
            self.source_covector.iter().copied(),
        );
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
            value += dot_product::<EF, _, _>(witness.iter().copied(), covector.iter().copied());
        }
        value += dot_product::<EF, _, _>(
            mask_message.iter().copied(),
            self.mask_covector.iter().copied(),
        );

        Ok(value)
    }
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
        .map(|&rho| padded_ood_t1(rho, source_message, mask_message))
        .collect()
}

/// Builds the deterministic part of the Lemma 9.8 verifier view.
///
/// This is not a full transcript simulator. It receives the values sampled by
/// adjacent simulators and derives the Construction 9.7 claim/relation that a
/// verifier would see.
///
/// The private OOD answers and source openings may be honestly computed or
/// sampled by simulators for the adjacent IORs. Construction 9.7 only needs
/// them as batched transcript values, then derives `mu'` and the output
/// relation from the same challenges.
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
    EF: Field + From<F>,
    Enc: LinearZkEncoding<F>,
{
    let mu_prime = claim.batched_claim(
        inherited_claim,
        simulated_private_ood_answers,
        simulated_source_openings,
    )?;
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
    //! Tests for the HVZK code-switching round (Construction 9.7).
    //!
    //! These tests validate the mathematical invariants of Construction 9.7
    //! from eprint 2026/391 using hand-constructed data over BabyBear.
    //!
    //! These tests use the real `p3-zk-codes` (`LinearZkEncoding`) and
    //! `whir::utils` zero-evader APIs instead of hand-rolled stubs.

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

    use super::{CodeSwitchError, ZkMaskClaim, private_ood_answers, simulated_verifier_view};
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

    fn make_setup(seed: u64, ell_zk: usize) -> (Perm, MyMmcs, MyEnc) {
        let mut perm_rng = SmallRng::seed_from_u64(seed);
        let perm = Perm::new_from_rng_128(&mut perm_rng);
        let merkle_hash = MyHash::new(perm.clone());
        let merkle_compress = MyCompress::new(perm.clone());
        let base_mmcs = BaseMmcs::new(merkle_hash, merkle_compress, 0);
        let mmcs = ExtensionMmcs::new(base_mmcs);
        let encoding = MyEnc::new(
            2,
            ell_zk,
            (ell_zk + 2).next_power_of_two(),
            MyDft::default(),
        );

        (perm, mmcs, encoding)
    }

    fn ef(v: u64) -> EF {
        EF::from(F::from_u64(v))
    }

    fn inner_product(a: &[EF], b: &[EF]) -> EF {
        a.iter().zip(b.iter()).map(|(x, y)| *x * *y).sum()
    }

    fn make_rs_encoding(
        msg_len: usize,
        t: usize,
        m: usize,
    ) -> ReedSolomonZkEncoding<F, Radix2Dit<F>> {
        ReedSolomonZkEncoding::new(t, msg_len, m, Radix2Dit::default())
    }

    fn solve_two_randomness_for_openings(
        encoding: &ReedSolomonZkEncoding<EF, Radix2Dit<EF>>,
        message: &[EF],
        positions: &[usize; 2],
        openings: &[EF],
    ) -> Vec<EF> {
        assert_eq!(openings.len(), 2);

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

        let a = row0[0];
        let b = row0[1];
        let c = row1[0];
        let d = row1[1];
        let det = a * d - b * c;
        assert!(!det.is_zero(), "RS query rows should be independent");

        vec![(rhs0 * d - b * rhs1) / det, (a * rhs1 - rhs0 * c) / det]
    }

    fn solve_pad_for_private_ood(
        rho: EF,
        message: &[EF],
        source_randomness: &[EF],
        target: EF,
    ) -> EF {
        assert!(!rho.is_zero(), "programming formula requires nonzero rho");

        let message_eval = eval_ze_star_n(rho, message);
        let source_randomness_eval = eval_ze_star_n(rho, source_randomness);
        let shift = rho.exp_u64(message.len() as u64);
        let pad_scale = shift * rho.exp_u64(source_randomness.len() as u64);
        assert!(
            !pad_scale.is_zero(),
            "programming formula requires a nonzero pad coefficient",
        );

        (target - message_eval - shift * source_randomness_eval) / pad_scale
    }

    /// Lift a base-field row into extension-field elements for inner products.
    fn lift_row(row: &[F]) -> Vec<EF> {
        row.iter().map(|&x| EF::from(x)).collect()
    }

    /// Construction 9.7 `mu'` identity test with `n = 0` (no prior auxiliary masks).
    ///
    /// Uses `ReedSolomonZkEncoding` with `msg_len=4, t=2, m=8` and validates
    /// the completeness equation using `LinearZkEncoding::message_row` /
    /// `randomness_row` for `G^#` / `G^$` and `eval_ze_star_n` for OOD answers.
    #[test]
    fn test_construction_9_7_mu_prime_identity_n0() {
        let ell = 4;
        let r_len = 2;
        let s_pad_len = 1;
        let t_ood = 2;
        let t = 3;
        let iota = 1;
        let m = 8;

        let source_enc = make_rs_encoding(ell, r_len, m);

        let f: Vec<EF> = (1..=ell as u64).map(ef).collect();
        let r: Vec<EF> = (10..10 + r_len as u64).map(ef).collect();
        let s_pad: Vec<EF> = (20..20 + s_pad_len as u64).map(ef).collect();

        let f_base: Vec<F> = (1..=ell as u64).map(F::from_u64).collect();
        let r_base: Vec<F> = (10..10 + r_len as u64).map(F::from_u64).collect();
        let cw = source_enc.encode_with_randomness(&f_base, &r_base);
        let f_codeword: Vec<EF> = cw.values.iter().map(|&v| EF::from(v)).collect();

        let sl: Vec<EF> = (1..=ell as u64).map(|i| ef(i * 100)).collect();
        let mu = inner_product(&f, &sl);

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

        let query_positions: Vec<usize> = vec![0, 2, 4];
        assert_eq!(query_positions.len(), t);

        let source_openings: Vec<EF> = query_positions.iter().map(|&pos| f_codeword[pos]).collect();

        let nu_dim = 1 + t_ood + t * iota;
        let rho_batch = ef(77);
        let nu = rho_batch.powers().collect_n(nu_dim);
        let claim = ZkMaskClaim {
            base_claim_coeff: nu[0],
            residual_sumcheck_scale: EF::ONE,
            ood_coeffs: nu[1..1 + t_ood].to_vec(),
            in_domain_coeffs: nu[1 + t_ood..].to_vec(),
        };

        let mu_prime = claim.batched_claim(mu, &y, &source_openings).unwrap();
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

        let mut r_s_pad = r;
        r_s_pad.extend_from_slice(&s_pad);

        let mu_prime_from_relation = relation.evaluate(&f, &[], &r_s_pad).unwrap();

        assert_eq!(
            mu_prime, mu_prime_from_relation,
            "Construction 9.7 mu' identity failed: verifier-computed mu' != output relation linear form"
        );
    }

    /// Same identity test with `n = 2` (two prior auxiliary mask oracles).
    #[test]
    fn test_construction_9_7_mu_prime_identity_n2() {
        let ell = 3;
        let r_len = 2;
        let s_pad_len = 1;
        let t_ood = 1;
        let t = 2;
        let iota = 1;
        let n = 2;
        let m = 8;

        let source_enc = make_rs_encoding(ell, r_len, m);

        let f: Vec<EF> = (1..=ell as u64).map(|i| ef(i * 3)).collect();
        let r: Vec<EF> = (10..10 + r_len as u64).map(ef).collect();
        let s_pad: Vec<EF> = vec![ef(42)];

        let f_base: Vec<F> = (1..=ell as u64).map(|i| F::from_u64(i * 3)).collect();
        let r_base: Vec<F> = (10..10 + r_len as u64).map(F::from_u64).collect();
        let cw = source_enc.encode_with_randomness(&f_base, &r_base);
        let f_codeword: Vec<EF> = cw.values.iter().map(|&v| EF::from(v)).collect();

        let sl: Vec<EF> = (1..=ell as u64).map(|i| ef(i * 7)).collect();

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

        let mut mu = inner_product(&f, &sl);
        for i in 0..n {
            mu += inner_product(&xi[i], &sl_aux[i]);
        }

        let rho_ood_points = [ef(99)];
        let mut f_r_s: Vec<EF> = Vec::new();
        f_r_s.extend_from_slice(&f);
        f_r_s.extend_from_slice(&r);
        f_r_s.extend_from_slice(&s_pad);
        let y: Vec<EF> = rho_ood_points
            .iter()
            .map(|&rho| eval_ze_star_n(rho, &f_r_s))
            .collect();

        let query_positions: Vec<usize> = vec![1, 3];
        let source_openings: Vec<EF> = query_positions.iter().map(|&p| f_codeword[p]).collect();

        let nu_dim = 1 + t_ood + t * iota;
        let rho_batch = ef(55);
        let nu = rho_batch.powers().collect_n(nu_dim);
        let claim = ZkMaskClaim {
            base_claim_coeff: nu[0],
            residual_sumcheck_scale: EF::ONE,
            ood_coeffs: nu[1..1 + t_ood].to_vec(),
            in_domain_coeffs: nu[1 + t_ood..].to_vec(),
        };

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

        let mut r_s_pad = r.clone();
        r_s_pad.extend_from_slice(&s_pad);
        let xi_refs: Vec<&[EF]> = xi.iter().map(Vec::as_slice).collect();

        let mu_prime_from_relation = relation.evaluate(&f, &xi_refs, &r_s_pad).unwrap();

        assert_eq!(
            mu_prime, mu_prime_from_relation,
            "Construction 9.7 mu' identity failed with n=2 auxiliary masks"
        );
    }

    /// Same identity with `iota = 2`, so queried source symbols expand to multiple
    /// flattened generator rows `x_{i,l} = iota * x_i + l`.
    ///
    /// Uses a larger domain (`m = 16`) to accommodate `msg_len + t = 6 < 16`.
    #[test]
    fn test_construction_9_7_mu_prime_identity_iota2() {
        let ell = 4;
        let r_len = 2;
        let s_pad_len = 2;
        let t_ood = 2;
        let t = 2;
        let iota = 2;
        let m = 16;

        let source_enc = make_rs_encoding(ell, r_len, m);

        let f: Vec<EF> = (1..=ell as u64).map(|i| ef(i * 2)).collect();
        let r: Vec<EF> = (0..r_len as u64).map(|i| ef(30 + i)).collect();
        let s_pad: Vec<EF> = (0..s_pad_len as u64).map(|i| ef(40 + i)).collect();

        let f_base: Vec<F> = (1..=ell as u64).map(|i| F::from_u64(i * 2)).collect();
        let r_base: Vec<F> = (0..r_len as u64).map(|i| F::from_u64(30 + i)).collect();
        let cw = source_enc.encode_with_randomness(&f_base, &r_base);
        let f_codeword: Vec<EF> = cw.values.iter().map(|&v| EF::from(v)).collect();

        let sl: Vec<EF> = (0..ell as u64).map(|i| ef(7 + i * 3)).collect();
        let mu = inner_product(&f, &sl);

        let rho_ood_points = [ef(6), ef(8)];
        let mut f_r_s = Vec::with_capacity(ell + r_len + s_pad_len);
        f_r_s.extend_from_slice(&f);
        f_r_s.extend_from_slice(&r);
        f_r_s.extend_from_slice(&s_pad);
        let y: Vec<EF> = rho_ood_points
            .iter()
            .map(|&rho| eval_ze_star_n(rho, &f_r_s))
            .collect();

        let query_symbols = [0_usize, 2_usize];
        let mut source_openings = Vec::with_capacity(t * iota);
        for &symbol in &query_symbols {
            for limb in 0..iota {
                source_openings.push(f_codeword[symbol * iota + limb]);
            }
        }

        let nu_dim = 1 + t_ood + t * iota;
        let nu = ef(13).powers().collect_n(nu_dim);
        let claim = ZkMaskClaim {
            base_claim_coeff: nu[0],
            residual_sumcheck_scale: EF::ONE,
            ood_coeffs: nu[1..1 + t_ood].to_vec(),
            in_domain_coeffs: nu[1 + t_ood..].to_vec(),
        };

        let mu_prime = claim.batched_claim(mu, &y, &source_openings).unwrap();
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

        let mut r_s_pad = r.clone();
        r_s_pad.extend_from_slice(&s_pad);
        let mu_prime_from_relation = relation.evaluate(&f, &[], &r_s_pad).unwrap();

        assert_eq!(
            mu_prime, mu_prime_from_relation,
            "Construction 9.7 mu' identity failed for iota=2 flattened row indexing"
        );
    }

    /// Same identity when the inherited sumcheck claim has already been scaled
    /// by the HVZK sumcheck challenge `eps`.
    ///
    /// The #1732 handoff folds `eps` into its residual claim, so
    /// the code-switch output relation must scale only the inherited source
    /// covector. The fresh OOD and in-domain terms are batched by Construction 9.7
    /// independently.
    #[test]
    fn test_construction_9_7_mu_prime_identity_eps_scaled_handoff() {
        let ell = 3;
        let r_len = 1;
        let s_pad_len = 1;
        let t_ood = 1;
        let t = 1;
        let iota = 1;
        let m = 8;

        let source_enc = make_rs_encoding(ell, r_len, m);

        let f: Vec<EF> = vec![ef(2), ef(5), ef(9)];
        let r: Vec<EF> = vec![ef(12)];
        let s_pad: Vec<EF> = vec![ef(20)];

        let f_base: Vec<F> = [2, 5, 9].into_iter().map(F::from_u64).collect();
        let r_base = vec![F::from_u64(12)];
        let cw = source_enc.encode_with_randomness(&f_base, &r_base);
        let f_codeword: Vec<EF> = cw.values.iter().map(|&v| EF::from(v)).collect();

        let eps = ef(19);
        let sl: Vec<EF> = vec![ef(4), ef(7), ef(11)];
        let mu = eps * inner_product(&f, &sl);

        let rho_ood_points = [ef(31)];
        let y = [padded_ood_t1(
            rho_ood_points[0],
            &f,
            &[r.clone(), s_pad.clone()].concat(),
        )];

        let query_position = 2;
        let source_opening = f_codeword[query_position];

        let nu = ef(17).powers().collect_n(1 + t_ood + t * iota);

        let claim = ZkMaskClaim {
            base_claim_coeff: nu[0],
            residual_sumcheck_scale: eps,
            ood_coeffs: vec![nu[1]],
            in_domain_coeffs: vec![nu[2]],
        };
        let mu_prime = claim.batched_claim(mu, &y, &[source_opening]).unwrap();

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

        let mut r_s_pad = r;
        r_s_pad.extend_from_slice(&s_pad);

        let mu_prime_from_relation =
            inner_product(&f, &sl_prime) + inner_product(&r_s_pad, &sl_mask);

        assert_eq!(
            mu_prime, mu_prime_from_relation,
            "Construction 9.7 must preserve the eps-scaled residual handoff"
        );
    }

    /// Same handoff as above, but with a prior auxiliary mask oracle.
    ///
    /// The HVZK sumcheck `eps` scale belongs to the source residual only. Carried mask
    /// auxiliary covectors must be batched by `nu_1`, not by `nu_1 * eps`.
    #[test]
    fn test_eps_scaled_handoff_does_not_scale_auxiliary_covectors() {
        let ell = 3;
        let r_len = 1;
        let s_pad_len = 1;
        let t_ood = 1;
        let t = 1;
        let iota = 1;
        let m = 8;

        let source_enc = make_rs_encoding(ell, r_len, m);

        let f: Vec<EF> = vec![ef(3), ef(8), ef(13)];
        let r: Vec<EF> = vec![ef(21)];
        let s_pad: Vec<EF> = vec![ef(34)];

        let f_base: Vec<F> = [3, 8, 13].into_iter().map(F::from_u64).collect();
        let r_base = vec![F::from_u64(21)];
        let cw = source_enc.encode_with_randomness(&f_base, &r_base);
        let f_codeword: Vec<EF> = cw.values.iter().map(|&v| EF::from(v)).collect();

        let source_covector = vec![ef(5), ef(7), ef(11)];
        let auxiliary_witness = vec![ef(17), ef(19)];
        let auxiliary_covector = vec![ef(23), ef(29)];

        let eps = ef(31);
        let source_claim = inner_product(&f, &source_covector);
        let auxiliary_claim = inner_product(&auxiliary_witness, &auxiliary_covector);
        let inherited_claim = eps * source_claim + auxiliary_claim;

        let rho_ood_points = [ef(37)];
        let mut mask_message = r;
        mask_message.extend_from_slice(&s_pad);
        let y = private_ood_answers(&rho_ood_points, &f, &mask_message);

        let query_position = 4;
        let source_opening = f_codeword[query_position];
        let nu = ef(41).powers().collect_n(1 + t_ood + t * iota);
        let claim = ZkMaskClaim {
            base_claim_coeff: nu[0],
            residual_sumcheck_scale: eps,
            ood_coeffs: vec![nu[1]],
            in_domain_coeffs: vec![nu[2]],
        };

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

    /// Same convention as above, but for an original mask covector that already
    /// carries the HVZK sumcheck `eps` scale in the inherited auxiliary claim.
    #[test]
    fn test_eps_scaled_auxiliary_covector_is_not_scaled_again() {
        let source_message = vec![ef(3), ef(5)];
        let source_covector = vec![ef(7), ef(11)];
        let auxiliary_witness = vec![ef(13), ef(17)];
        let auxiliary_covector = [ef(19), ef(23)];
        let eps = ef(29);
        let nu_1 = ef(31);

        let scaled_auxiliary_covector: Vec<EF> =
            auxiliary_covector.iter().map(|&x| eps * x).collect();
        let inherited_claim = eps * inner_product(&source_message, &source_covector)
            + inner_product(&auxiliary_witness, &scaled_auxiliary_covector);
        let claim = ZkMaskClaim {
            base_claim_coeff: nu_1,
            residual_sumcheck_scale: eps,
            ood_coeffs: Vec::new(),
            in_domain_coeffs: Vec::new(),
        };

        let mu_prime = claim.batched_claim(inherited_claim, &[], &[]).unwrap();
        let relation = claim
            .output_relation_from_rows::<F>(
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

    /// Verify private zero-evader OOD answer via `padded_ood_t1` matches
    /// the manual `eval_ze_star_n` on the concatenated vector.
    #[test]
    fn test_private_ood_answer_consistency() {
        let f = vec![ef(3), ef(7), ef(11)];
        let r = vec![ef(5), ef(9)];
        let s_pad = vec![ef(13)];

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

        let inherited_claim = inner_product(&source_message, &source_covector);
        let private_ood = private_ood_answers(&rho_ood_points, &source_message, &[]);
        let source_opening = inner_product(&source_message, &lift_row(&source_row));
        let mu_prime = claim
            .batched_claim(inherited_claim, &private_ood, &[source_opening])
            .unwrap();
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
        assert_eq!(
            private_ood[0], source_message[0],
            "rho = 0 should select the constant source coefficient",
        );
        assert_eq!(
            relation.evaluate(&source_message, &[], &[]).unwrap(),
            mu_prime,
        );
    }

    /// Verify batching zero-evader produces correct powers.
    #[test]
    fn test_zero_evader_powers() {
        let rho = ef(5);
        let nu = rho.powers().collect_n(4);

        assert_eq!(nu[0], EF::ONE);
        assert_eq!(nu[1], rho);
        assert_eq!(nu[2], rho * rho);
        assert_eq!(nu[3], rho * rho * rho);
    }

    #[test]
    fn test_private_ood_answers_matches_padded_ood_t1() {
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
    fn test_simulated_verifier_view_matches_code_switch_relation() {
        let ell = 3;
        let r_len = 2;
        let s_pad_len = 1;
        let t_ood = 2;
        let t = 2;
        let iota = 1;
        let m = 8;

        let source_enc = make_rs_encoding(ell, r_len, m);

        let f: Vec<EF> = vec![ef(4), ef(6), ef(10)];
        let r: Vec<EF> = vec![ef(13), ef(17)];
        let s_pad: Vec<EF> = vec![ef(19)];
        let mut mask_message = r;
        mask_message.extend_from_slice(&s_pad);

        let f_base: Vec<F> = [4, 6, 10].into_iter().map(F::from_u64).collect();
        let r_base: Vec<F> = [13, 17].into_iter().map(F::from_u64).collect();
        let cw = source_enc.encode_with_randomness(&f_base, &r_base);
        let f_codeword: Vec<EF> = cw.values.iter().map(|&v| EF::from(v)).collect();

        let source_covector = vec![ef(23), ef(29), ef(31)];
        let eps = ef(47);
        let inherited_claim = eps * inner_product(&f, &source_covector);

        let rho_ood_points = [ef(37), ef(41)];
        let private_ood = private_ood_answers(&rho_ood_points, &f, &mask_message);

        let query_positions = [1_usize, 4_usize];
        let source_openings: Vec<EF> = query_positions
            .iter()
            .map(|&position| f_codeword[position])
            .collect();

        let nu = ef(43).powers().collect_n(1 + t_ood + t * iota);
        let claim = ZkMaskClaim {
            base_claim_coeff: nu[0],
            residual_sumcheck_scale: eps,
            ood_coeffs: nu[1..1 + t_ood].to_vec(),
            in_domain_coeffs: nu[1 + t_ood..].to_vec(),
        };

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
        let expected_mu = claim
            .batched_claim(inherited_claim, &private_ood, &source_openings)
            .expect("valid simulated transcript dimensions");
        let relation_value = view
            .output_relation
            .evaluate(&f, &[], &mask_message)
            .unwrap();

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
        let ell_zk = 4;
        let folding_factor = 2;
        let (perm, mmcs, sumcheck_encoding) = make_setup(91, ell_zk);
        let mut simulator_challenger = MyChallenger::new(perm.clone());
        let mut replay_challenger = MyChallenger::new(perm);
        let simulator_verifier = ZkVerifier::<F, EF>::new_prefix(&[]);
        let mut simulator_rng = SmallRng::seed_from_u64(92);
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
        assert!(
            !verifier_handoff.eps.is_zero(),
            "fixture seed should produce a nonzero eps for the code-switch source scale",
        );

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

        let mut source_covector = EF::zero_vec(source_message.len());
        let (pivot, &pivot_value) = source_message
            .iter()
            .enumerate()
            .find(|(_, value)| !value.is_zero())
            .expect("fixture source message should contain a nonzero entry");
        source_covector[pivot] =
            verifier_handoff.claimed_residual / verifier_handoff.eps / pivot_value;

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

    fn assert_programmable_simulator_components_for_zk_source_view(seed: u64) -> bool {
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
        if verifier_handoff.eps.is_zero() {
            return false;
        }

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
        let mut source_covector = EF::zero_vec(source_message.len());
        let (pivot, &pivot_value) = source_message
            .iter()
            .enumerate()
            .find(|(_, value)| !value.is_zero())
            .expect("fixture source message should contain a nonzero entry");
        source_covector[pivot] =
            verifier_handoff.claimed_residual / verifier_handoff.eps / pivot_value;

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

        // Witness-independent coupling certificate, mirroring the HVZK sumcheck
        // matched-RNG simulator test where byte equality is meaningful. Construction 9.7's
        // fresh round mask is sampled independently of the source witness, so equal
        // RNG streams produce equal mask commitments. Source commitments, source
        // openings, and private OOD answers are witness-dependent; those are covered
        // by the programmability checks above instead of a fake byte-equality claim.
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
        let source_codeword =
            source_enc.encode_with_randomness(&source_message, &source_randomness);
        let query_positions = [1_usize, 4_usize];
        let source_openings = query_positions
            .iter()
            .map(|&position| source_codeword.values[position])
            .collect::<Vec<_>>();

        let source_covector = vec![ef(7), ef(11), ef(13)];
        let inherited_claim = inner_product(&source_message, &source_covector);
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

        let nu = ef(43)
            .powers()
            .collect_n(1 + rho_ood_points.len() + query_positions.len());
        let claim = ZkMaskClaim {
            base_claim_coeff: nu[0],
            residual_sumcheck_scale: EF::ONE,
            ood_coeffs: nu[1..1 + rho_ood_points.len()].to_vec(),
            in_domain_coeffs: nu[1 + rho_ood_points.len()..].to_vec(),
        };

        // This mirrors the actual round-0 source shape: the source oracle has no
        // encoding randomness, so source openings are deterministic rather than
        // simulated via `ZkEncoding::simulate`. The only programmable randomized
        // part at this boundary is the private OOD mask contribution.
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
    fn test_output_relation_rejects_query_count_mismatch() {
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
    fn test_output_relation_rejects_source_covector_length_mismatch() {
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

    /// Verify that `ReedSolomonZkEncoding::message_row` / `randomness_row`
    /// decompose the codeword correctly: `cw[i] = <msg, G^#[i]> + <rand, G^$[i]>`.
    #[test]
    fn test_rs_row_decomposition_matches_encoding() {
        let msg_len = 4;
        let t = 2;
        let m = 8;
        let enc = make_rs_encoding(msg_len, t, m);

        let msg: Vec<F> = (1..=msg_len as u64).map(F::from_u64).collect();
        let rand: Vec<F> = (10..10 + t as u64).map(F::from_u64).collect();
        let cw = enc.encode_with_randomness(&msg, &rand);

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
