//! Mask bookkeeping for the masked sumcheck batches.

use alloc::vec::Vec;
use core::mem::take;

use p3_commit::{ExtensionMmcs, Mmcs};
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use p3_sumcheck::strategy::SumcheckProver;
use p3_sumcheck::zk::{ZkSumcheckData, ZkSumcheckHandoff, mask_residual_covectors_from_shape};

use crate::pcs::zk::base_case::MaskProverData;
use crate::pcs::zk::constraint::MaskClaims;

/// All mask-oracle state the prover carries to the base case.
///
/// - Messages, randomness, and covectors are flat in chronological order.
/// - The group list tiles them, one Merkle handle per committed batch.
pub(super) struct ProverMasks<F, EF, MT>
where
    F: TwoAdicField,
    EF: ExtensionField<F>,
    MT: Mmcs<F>,
{
    /// Mask messages, chronological.
    pub(super) messages: Vec<Vec<EF>>,
    /// Mask encoding randomness, matching order.
    pub(super) randomness: Vec<Vec<EF>>,
    /// Group widths and Merkle prover data, in commit order.
    pub(super) groups: Vec<(usize, MaskProverData<F, EF, MT>)>,
    /// Dense covectors with their accumulated scales.
    pub(super) claims: MaskClaims<EF>,
}

impl<F, EF, MT> ProverMasks<F, EF, MT>
where
    F: TwoAdicField,
    EF: ExtensionField<F>,
    MT: Mmcs<F>,
{
    pub(super) const fn new() -> Self {
        Self {
            messages: Vec::new(),
            randomness: Vec::new(),
            groups: Vec::new(),
            claims: MaskClaims::new(),
        }
    }
}

/// Carried state after one masked sumcheck batch.
pub(super) struct BatchState<F: Field, EF: ExtensionField<F>> {
    /// Residual prover over the folded message with eps-scaled weights.
    pub(super) residual_prover: SumcheckProver<F, EF>,
    /// The batch's folding randomness.
    pub(super) randomness: Point<EF>,
}

/// Stores one masked sumcheck batch in the proof and the mask bookkeeping.
pub(super) fn record_sumcheck_batch<F, EF, MT>(
    sumchecks: &mut Vec<ZkSumcheckData<F, EF>>,
    mask_commitments: &mut Vec<MT::Commitment>,
    masks: &mut ProverMasks<F, EF, MT>,
    mut handoff: ZkSumcheckHandoff<F, EF, ExtensionMmcs<F, EF, MT>>,
    zk_data: &mut ZkSumcheckData<F, EF>,
    ell_zk: usize,
) -> BatchState<F, EF>
where
    F: TwoAdicField + Send + Sync,
    EF: ExtensionField<F>,
    MT: Mmcs<F>,
{
    let folding = handoff.randomness.num_variables();
    sumchecks.push(take(zk_data));
    mask_commitments.push(handoff.mask_oracle.0.clone());

    // Carried covectors absorb eps * 2^{-k}; the fresh sumcheck masks enter
    // at scale one with power covectors at their round challenges.
    masks.claims.absorb_sumcheck(handoff.eps, folding);
    let gammas: Vec<EF> = handoff.randomness.iter().copied().collect();
    for covector in mask_residual_covectors_from_shape(folding, ell_zk, &gammas) {
        masks.claims.push(covector);
    }
    masks.messages.append(&mut handoff.mask_messages);
    masks.randomness.append(&mut handoff.mask_randomness);
    masks.groups.push((folding, handoff.mask_oracle.1));

    BatchState {
        residual_prover: handoff.residual_prover,
        randomness: handoff.randomness,
    }
}

/// Folds a limb-major chunked vector by the eq table at `gamma`.
///
/// Chunk `b` belongs to limb `b`.
/// The output `sum_b eq(b, gamma) * chunk_b` matches the leaf fold
/// orientation.
pub(super) fn fold_limb_chunks<F, EF>(values: &[F], chunk: usize, gamma: &Point<EF>) -> Vec<EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    assert_eq!(values.len(), chunk << gamma.num_variables());
    let eq_table = Poly::new_from_point(gamma.as_slice(), EF::ONE);
    let mut out = EF::zero_vec(chunk);
    for (b, &weight) in eq_table.as_slice().iter().enumerate() {
        for (dst, &src) in out.iter_mut().zip(&values[b * chunk..(b + 1) * chunk]) {
            *dst += weight * src;
        }
    }
    out
}
