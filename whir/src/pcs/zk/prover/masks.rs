//! Mask bookkeeping for the masked sumcheck batches.
//!
//! Every batch of `k` sumcheck rounds leaves two trails:
//!
//! ```text
//!  proof side  ->  the wire transcript and one interleaved mask commitment go into the proof
//!  claim side  ->  the k masks join the carried relation as <xi_i, u_i> terms, settled at the base case
//! ```

use alloc::vec::Vec;
use core::mem::take;

use p3_commit::{ExtensionMmcs, Mmcs};
use p3_field::{ExtensionField, Field, TwoAdicField, dot_product};
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use p3_sumcheck::strategy::SumcheckProver;
use p3_sumcheck::zk::{ZkSumcheckData, ZkSumcheckHandoff, mask_residual_covectors_from_shape};

use crate::pcs::zk::base_case::MaskProverData;
use crate::pcs::zk::constraint::MaskClaims;

/// All mask-oracle state the prover carries to the base case.
///
/// Flat lists in chronological mask order, tiled by the groups:
///
/// ```text
///     messages   : [ m1 m2 m3 | m4 | m5 m6 m7 | ... ]
///     randomness : [ r1 r2 r3 | r4 | r5 r6 r7 | ... ]
///     covectors  : [ u1 u2 u3 | u4 | u5 u6 u7 | ... ]
///     groups     : [ (3, tree)  (1, tree)  (3, tree)  ... ]
/// ```
///
/// One Merkle handle per group answers the base-case spot checks.
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
    /// Running mask-claim total `sum_i <xi_i, u_i>` at the current scales.
    ///
    /// Maintained by the mutating methods so callers read it in O(1)
    /// instead of re-evaluating every covector each round.
    pub(super) aux: EF,
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
            aux: EF::ZERO,
        }
    }

    /// Stores one masked sumcheck batch.
    ///
    /// ```text
    ///     proof side  ->  wire transcript, interleaved mask commitment
    ///     claim side  ->  rescale the carried covectors,
    ///                     append the batch's k fresh masks
    /// ```
    pub(super) fn record_batch(
        &mut self,
        sumchecks: &mut Vec<ZkSumcheckData<F, EF>>,
        mask_commitments: &mut Vec<MT::Commitment>,
        mut handoff: ZkSumcheckHandoff<F, EF, ExtensionMmcs<F, EF, MT>>,
        zk_data: &mut ZkSumcheckData<F, EF>,
        ell_zk: usize,
    ) -> BatchState<F, EF>
    where
        F: Send + Sync,
    {
        let folding = handoff.randomness.num_variables();
        // Proof side: the batch's wire transcript and mask commitment.
        sumchecks.push(take(zk_data));
        mask_commitments.push(handoff.mask_oracle.0.clone());

        // Claim side, carried masks: every covector absorbs eps * 2^{-k},
        // so the running total absorbs the same scale.
        self.aux *= self.claims.absorb_sumcheck(handoff.eps, folding);
        // Claim side, fresh masks: mask j enters at scale one with the
        // power covector pow(gamma_j), its residual at the round challenge.
        let gammas: Vec<EF> = handoff.randomness.iter().copied().collect();
        for (covector, message) in mask_residual_covectors_from_shape(folding, ell_zk, &gammas)
            .into_iter()
            .zip(&handoff.mask_messages)
        {
            self.aux += dot_product::<EF, _, _>(covector.iter().copied(), message.iter().copied());
            self.claims.push(covector);
        }
        // Retain the secrets behind the new oracle for the base-case
        // reveals, plus its Merkle handle for the spot-check openings.
        self.messages.append(&mut handoff.mask_messages);
        self.randomness.append(&mut handoff.mask_randomness);
        self.groups.push((folding, handoff.mask_oracle.1));

        BatchState {
            residual_prover: handoff.residual_prover,
            randomness: handoff.randomness,
        }
    }

    /// Records the fresh code-switch mask of one round as its own group.
    pub(super) fn push_switch_mask(
        &mut self,
        covector: Vec<EF>,
        message: Vec<EF>,
        randomness: Vec<EF>,
        data: MaskProverData<F, EF, MT>,
    ) {
        self.aux += dot_product::<EF, _, _>(covector.iter().copied(), message.iter().copied());
        self.claims.push(covector);
        self.messages.push(message);
        self.randomness.push(randomness);
        self.groups.push((1, data));
    }
}

/// Carried state after one masked sumcheck batch.
pub(super) struct BatchState<F: Field, EF: ExtensionField<F>> {
    /// Residual prover over the folded message with eps-scaled weights.
    pub(super) residual_prover: SumcheckProver<F, EF>,
    /// The batch's folding randomness.
    pub(super) randomness: Point<EF>,
}

/// Folds a limb-major chunked vector by the eq table at `gamma`.
///
/// Chunk `b` holds the slice belonging to limb `b` of the committed oracle:
///
/// ```text
///     values = [ chunk_0 | chunk_1 | ... | chunk_{2^k - 1} ]
///     out    = sum_b eq(b, gamma) * chunk_b
/// ```
///
/// The orientation matches the verifier's leaf fold.
pub(super) fn fold_limb_chunks<F, EF>(values: &[F], chunk: usize, gamma: &Point<EF>) -> Vec<EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    // One chunk per vertex of the folded hypercube.
    assert_eq!(values.len(), chunk << gamma.num_variables());
    // Fold weights: eq(b, gamma) for every prefix b.
    let eq_table = Poly::new_from_point(gamma.as_slice(), EF::ONE);
    let mut out = EF::zero_vec(chunk);
    for (b, &weight) in eq_table.as_slice().iter().enumerate() {
        // Accumulate chunk b, scaled by its weight.
        //
        // Mixed-field step: extension weight times base-field entry.
        for (dst, &src) in out.iter_mut().zip(&values[b * chunk..(b + 1) * chunk]) {
            *dst += weight * src;
        }
    }
    out
}
