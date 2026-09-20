//! Zero-knowledge extension of the stacked sumcheck layout.
//!
//! Captures the data the masking layer needs from any layout, and the one
//! piece of arithmetic that genuinely branches on the binding direction.

use p3_field::{ExtensionField, TwoAdicField};
use p3_multilinear_util::point::Point;

use crate::layout::{Layout, PrefixProver, ProverMultiClaim, ProverVirtualClaim, SuffixProver};
use crate::product_polynomial::ProductPolynomial;
use crate::strategy::VariableOrder;

/// Per-mode hooks consumed by the zero-knowledge prover.
///
/// Every implementor exposes the data the masking layer reads.
/// The residual handoff is the only operation that branches on the binding direction.
pub trait ZkLayout<F, EF>: Layout<F, EF>
where
    F: TwoAdicField,
    EF: ExtensionField<F>,
{
    /// Walks concrete claims in placement order.
    #[inline]
    fn concrete_claims(&self) -> impl Iterator<Item = &ProverMultiClaim<F, EF>> {
        self.claims().concrete_claims()
    }

    /// Returns the virtual-claim slice.
    #[inline]
    fn virtual_claims(&self) -> &[ProverVirtualClaim<EF>] {
        &self.claims().virtual_claims
    }

    /// Returns the alpha-batched plain sum.
    #[inline]
    fn batched_sum(&self, alpha: EF) -> EF {
        self.claims().sum(alpha)
    }

    /// Builds the residual product polynomial, scaled by the combining challenge.
    ///
    /// Consumes the layout; the residual factor is its last consumer.
    fn zk_residual_handoff(self, rs: &Point<EF>, alpha: EF, eps: EF) -> ProductPolynomial<F, EF>
    where
        EF: TwoAdicField;
}

impl<F, EF> ZkLayout<F, EF> for PrefixProver<F, EF>
where
    F: TwoAdicField,
    EF: ExtensionField<F>,
{
    fn zk_residual_handoff(self, rs: &Point<EF>, alpha: EF, eps: EF) -> ProductPolynomial<F, EF>
    where
        EF: TwoAdicField,
    {
        // Scale the compression by the masking challenge.
        // Narrow residuals use scalar storage.
        self.residual_product(rs, alpha, eps)
    }
}

impl<F, EF> ZkLayout<F, EF> for SuffixProver<F, EF>
where
    F: TwoAdicField,
    EF: ExtensionField<F>,
{
    fn zk_residual_handoff(self, rs: &Point<EF>, alpha: EF, eps: EF) -> ProductPolynomial<F, EF>
    where
        EF: TwoAdicField,
    {
        // Reverse the challenges to match the suffix-binding frame.
        let reversed = rs.reversed();
        // Walk per-table slots; the combining challenge rides on the slot compression.
        let compressed = tracing::info_span!("compress_stacked_with_eps")
            .in_scope(|| self.compress_stacked_scaled(&reversed, eps));
        // The SVO preprocessing covers what packing would help; no packing here.
        let weights = self.combine_weights(&reversed, alpha);
        ProductPolynomial::new_unpacked(VariableOrder::Suffix, compressed, weights)
    }
}
