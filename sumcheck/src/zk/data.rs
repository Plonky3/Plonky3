//! Transcript schema and oracle handle for the HVZK sumcheck.

use alloc::vec::Vec;

use p3_commit::Mmcs;
use p3_field::{ExtensionField, Field, HornerIter};
use p3_matrix::dense::RowMajorMatrix;
use p3_multilinear_util::point::Point;
use serde::{Deserialize, Serialize};

use crate::strategy::SumcheckProver;

/// Per-round prover output of the HVZK sumcheck protocol.
///
/// - Prover writes;
/// - Verifier reads back during Fiat-Shamir replay.
///
/// One instance covers a full run of `k` rounds.
///
/// # Wire format
///
/// Per round, the polynomial has coefficient layout
///
/// ```text
///     [ c_0, c_1, c_2, ..., c_d ]    with  d = max(ell_zk - 1, 2)
/// ```
///
/// The linear coefficient `c_1` is dropped on the wire.
///
/// The verifier reconstructs `c_1` from the affine identity
///
/// ```text
///     h_j(0) + h_j(1) = 2 * c_0 + sum_{i >= 1} c_i = target
/// ```
///
/// applied to the previous round's target.
///
/// # Soundness link to Lemma 6.4
///
/// Valid transcripts form an affine subspace of dimension `1 + k * (ell_zk - 1)`.
/// The `k` dropped linear coefficients are exactly the redundant degrees of freedom of the rank-nullity argument.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZkSumcheckData<F, EF> {
    /// Sum of all mask polynomial evaluations across the boolean hypercube `{0,1}^k`.
    ///
    /// Observed on the transcript before the verifier samples the combining challenge.
    /// Lives in the extension field because the mask coefficients do.
    pub mu_tilde: EF,

    /// Message length of the zero-knowledge mask code.
    ///
    /// The verifier rejects up front if its own expected value disagrees with this.
    /// Pinning this in the transcript closes a non-injectivity gap in the wire-length check: lengths `2` and `3` share a wire layout.
    pub ell_zk: usize,

    /// Per-round wire payload with the linear coefficient dropped.
    ///
    /// One entry per sumcheck round.
    /// Layout per entry: `[c_0, c_2, c_3, ..., c_d]` with `d = max(ell_zk - 1, 2)`.
    pub round_coefficients: Vec<Vec<EF>>,

    /// Per-round proof-of-work witnesses.
    ///
    /// Length equals the number of rounds when grinding is enabled.
    /// Empty when `pow_bits == 0`.
    pub pow_witnesses: Vec<F>,
}

impl<F, EF: Field> Default for ZkSumcheckData<F, EF> {
    fn default() -> Self {
        Self {
            // Real runs overwrite this in step 2 once the prover has summed the masks.
            mu_tilde: EF::ZERO,
            // Sentinel: honest runs set this to the encoding's message length; the verifier rejects 0.
            ell_zk: 0,
            // Filled with one wire entry per sumcheck round.
            round_coefficients: Vec::new(),
            // Filled only when grinding is enabled.
            pow_witnesses: Vec::new(),
        }
    }
}

/// Handle to one committed batch of interleaved mask codewords.
///
/// - Pairs the public Merkle root with the prover-side data needed to open
///   the batch at requested positions.
/// - Row `z` of the committed matrix holds position `z` of every mask in
///   the batch.
/// - One Merkle path therefore authenticates all of them.
pub type MaskOracle<EF, M> = (
    <M as Mmcs<EF>>::Commitment,
    <M as Mmcs<EF>>::ProverData<RowMajorMatrix<EF>>,
);

/// Typed prover handoff produced by the HVZK sumcheck.
///
/// - Downstream code-switching needs both the residual prover and the
///   sampled `eps` scale.
/// - A named type makes the Construction 6.3 to Construction 9.7 boundary
///   explicit.
pub struct ZkSumcheckHandoff<F, EF, M>
where
    F: Field,
    EF: ExtensionField<F>,
    M: Mmcs<EF>,
{
    /// Residual sumcheck prover whose claim is scaled by `eps`.
    pub residual_prover: SumcheckProver<F, EF>,
    /// Per-round sumcheck challenges.
    pub randomness: Point<EF>,
    /// Construction 6.3 combining challenge.
    pub eps: EF,
    /// Plain mask messages sampled by the prover, in round order.
    ///
    /// These are prover-only witnesses. Code-switch composition uses them to
    /// carry the verifier-visible masked residual as auxiliary linear claims.
    pub mask_messages: Vec<Vec<EF>>,
    /// Encoding randomness used for each mask, in round order.
    ///
    /// Prover-only. The HVZK base case reveals blinded combinations
    /// `r* = r' + gamma * r`, which requires the raw values.
    pub mask_randomness: Vec<Vec<EF>>,
    /// The batch's interleaved mask oracle: one commitment, `k` columns.
    pub mask_oracle: MaskOracle<EF, M>,
}

/// Typed verifier handoff produced by replaying an HVZK sumcheck transcript.
///
/// This mirrors [`ZkSumcheckHandoff`] without prover-only mask data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ZkVerifierHandoff<EF> {
    /// Per-round sumcheck challenges.
    pub randomness: Point<EF>,
    /// Residual claim after replay.
    pub claimed_residual: EF,
    /// Construction 6.3 combining challenge.
    pub eps: EF,
}

/// Evaluates the final verifier-visible mask residual after all HVZK sumcheck rounds.
///
/// For masks `s_j(X)` and verifier challenges `gamma_j`, the mask part of the
/// final Construction 6.3 target is:
///
/// ```text
///     sum_j s_j(gamma_j)
/// ```
///
/// This is the closed form of the live/past/future mask recurrence used while
/// assembling the round polynomials.
#[must_use]
pub fn mask_residual<EF>(masks: &[Vec<EF>], gammas: &[EF]) -> EF
where
    EF: Field,
{
    assert_eq!(masks.len(), gammas.len());
    masks
        .iter()
        .zip(gammas)
        .map(|(mask, &gamma)| mask.iter().copied().horner(gamma))
        .sum()
}

/// Linear covectors whose dot products with the masks equal [`mask_residual`].
#[must_use]
pub fn mask_residual_covectors<EF>(masks: &[Vec<EF>], gammas: &[EF]) -> Vec<Vec<EF>>
where
    EF: Field,
{
    assert!(
        masks
            .iter()
            .all(|mask| mask.len() == masks.first().map_or(0, Vec::len))
    );
    mask_residual_covectors_from_shape(masks.len(), masks.first().map_or(0, Vec::len), gammas)
}

/// Linear covectors for masks with a known rectangular shape.
///
/// The covector for mask `s_j` is `[1, gamma_j, gamma_j^2, ...]`.
/// Code-switch composition carries these as the fresh sumcheck-mask claims.
#[must_use]
pub fn mask_residual_covectors_from_shape<EF: Field>(
    mask_count: usize,
    mask_len: usize,
    gammas: &[EF],
) -> Vec<Vec<EF>> {
    assert_eq!(mask_count, gammas.len());
    gammas
        .iter()
        .map(|gamma| gamma.powers().collect_n(mask_len))
        .collect()
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use p3_baby_bear::BabyBear;
    use p3_field::extension::BinomialExtensionField;
    use p3_field::{Field, PrimeCharacteristicRing, dot_product};

    use super::{mask_residual, mask_residual_covectors};

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;

    fn reference_mask_recurrence<EF>(masks: &[Vec<EF>], gammas: &[EF]) -> EF
    where
        EF: Field,
    {
        assert_eq!(masks.len(), gammas.len());
        let k = masks.len();
        if k == 0 {
            return EF::ZERO;
        }

        let pow2: Vec<EF> = EF::TWO.powers().collect_n(k + 1);
        let mut mask_evals_at_gamma = Vec::with_capacity(k);
        let mut sum_future_endpoints: EF = masks
            .iter()
            .map(|mask| mask[0].double() + mask[1..].iter().copied().sum::<EF>())
            .sum();
        let mut target = EF::ZERO;

        for (round_idx, (s_j, &gamma_j)) in masks.iter().zip(gammas).enumerate() {
            let j = round_idx + 1;
            let s_j_endpoints = s_j[0].double() + s_j[1..].iter().copied().sum::<EF>();
            sum_future_endpoints -= s_j_endpoints;

            let h_size = s_j.len().max(3);
            let mut h = EF::zero_vec(h_size);
            let mult_live = pow2[k - j];
            for (i, &c) in s_j.iter().enumerate() {
                h[i] += mult_live * c;
            }

            let past_mask_sum: EF = mask_evals_at_gamma.iter().copied().sum();
            h[0] += past_mask_sum * mult_live;
            if j < k {
                h[0] += pow2[k - j - 1] * sum_future_endpoints;
            }

            target = h
                .iter()
                .rev()
                .copied()
                .fold(EF::ZERO, |acc, coeff| acc * gamma_j + coeff);

            let s_j_at_gamma = s_j
                .iter()
                .rev()
                .copied()
                .fold(EF::ZERO, |acc, coeff| acc * gamma_j + coeff);
            mask_evals_at_gamma.push(s_j_at_gamma);
        }

        target
    }

    #[test]
    fn mask_residual_closed_form_matches_round_recurrence() {
        let masks = vec![
            vec![
                EF::from_u64(3),
                EF::from_u64(5),
                EF::from_u64(7),
                EF::from_u64(11),
            ],
            vec![
                EF::from_u64(13),
                EF::from_u64(17),
                EF::from_u64(19),
                EF::from_u64(23),
            ],
            vec![
                EF::from_u64(29),
                EF::from_u64(31),
                EF::from_u64(37),
                EF::from_u64(41),
            ],
        ];
        let gammas = vec![EF::from_u64(43), EF::from_u64(47), EF::from_u64(53)];

        assert_eq!(
            mask_residual::<EF>(&masks, &gammas),
            reference_mask_recurrence::<EF>(&masks, &gammas),
        );
    }

    #[test]
    fn mask_residual_covectors_evaluate_closed_form() {
        let masks = vec![
            vec![EF::from_u64(2), EF::from_u64(3), EF::from_u64(5)],
            vec![EF::from_u64(7), EF::from_u64(11), EF::from_u64(13)],
        ];
        let gammas = vec![EF::from_u64(17), EF::from_u64(19)];
        let covectors = mask_residual_covectors::<EF>(&masks, &gammas);
        let by_covectors = masks
            .iter()
            .zip(&covectors)
            .map(|(mask, covector)| {
                dot_product::<EF, _, _>(mask.iter().copied(), covector.iter().copied())
            })
            .sum::<EF>();

        assert_eq!(by_covectors, mask_residual::<EF>(&masks, &gammas));
    }
}
