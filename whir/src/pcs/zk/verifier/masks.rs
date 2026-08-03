//! Mask-claim bookkeeping for the HVZK-WHIR verifier replay.
//!
//! The symbolic counterpart of the prover's `ProverMasks`: the verifier
//! reconstructs the same mask covectors, group shapes, and commitments round
//! by round, then hands them to the base case.

use alloc::vec::Vec;

use p3_commit::Mmcs;
use p3_field::{ExtensionField, Field};
use p3_multilinear_util::point::Point;
use p3_sumcheck::zk::mask_residual_covectors_from_shape;

use crate::pcs::zk::constraint::MaskClaims;
use crate::pcs::zk::mask::{MaskCodeShape, MaskGroupShape};

/// Verifier-side mask state carried to the base case.
///
/// One covector, group shape, and commitment per mask oracle, in commit order.
pub(super) struct VerifierMasks<F, EF, MT>
where
    F: Field,
    EF: ExtensionField<F>,
    MT: Mmcs<F>,
{
    /// Dense covectors with their accumulated scales.
    pub(super) claims: MaskClaims<EF>,
    /// Group widths and codes, in commit order.
    pub(super) groups: Vec<MaskGroupShape>,
    /// One commitment per mask group, matching `groups`.
    pub(super) commitments: Vec<MT::Commitment>,
}

impl<F, EF, MT> VerifierMasks<F, EF, MT>
where
    F: Field,
    EF: ExtensionField<F>,
    MT: Mmcs<F>,
{
    pub(super) const fn new() -> Self {
        Self {
            claims: MaskClaims::new(),
            groups: Vec::new(),
            commitments: Vec::new(),
        }
    }

    /// Records one masked sumcheck batch.
    ///
    /// The carried covectors absorb `eps * 2^{-k}`; the batch's `k` fresh
    /// sumcheck masks enter at scale one as power covectors of the round
    /// randomness.
    pub(super) fn record_sumcheck_batch(
        &mut self,
        eps: EF,
        folding: usize,
        ell_zk: usize,
        randomness: &Point<EF>,
        shape: MaskCodeShape,
        commitment: MT::Commitment,
    ) {
        self.claims.absorb_sumcheck(eps, folding);
        let gammas: Vec<EF> = randomness.iter().copied().collect();
        for covector in mask_residual_covectors_from_shape(folding, ell_zk, &gammas) {
            self.claims.push(covector);
        }
        self.groups.push(MaskGroupShape {
            shape,
            width: folding,
        });
        self.commitments.push(commitment);
    }

    /// Records one code-switch round's fresh mask as a width-one group.
    pub(super) fn push_switch_mask(
        &mut self,
        covector: Vec<EF>,
        shape: MaskCodeShape,
        commitment: MT::Commitment,
    ) {
        self.claims.push(covector);
        self.groups.push(MaskGroupShape { shape, width: 1 });
        self.commitments.push(commitment);
    }
}
