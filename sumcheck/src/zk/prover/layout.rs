//! Zero-knowledge extension of the stacked sumcheck layout.
//!
//! Captures the data the masking layer needs from any layout, and the one
//! piece of arithmetic that genuinely branches on the binding direction.

use p3_field::{ExtensionField, Field, PackedValue, TwoAdicField};
use p3_multilinear_util::point::Point;
use p3_util::log2_strict_usize;

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
    fn concrete_claims(&self) -> impl Iterator<Item = &ProverMultiClaim<F, EF>>;

    /// Returns the virtual-claim slice.
    fn virtual_claims(&self) -> &[ProverVirtualClaim<EF>];

    /// Returns the alpha-batched plain sum.
    fn batched_sum(&self, alpha: EF) -> EF;

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
    fn concrete_claims(&self) -> impl Iterator<Item = &ProverMultiClaim<F, EF>> {
        // Flatten placement order across all per-table claim lists.
        self.placements
            .iter()
            .flat_map(|placement| self.claim_map[placement.idx()].iter())
    }

    fn virtual_claims(&self) -> &[ProverVirtualClaim<EF>] {
        &self.virtual_claims
    }

    fn batched_sum(&self, alpha: EF) -> EF {
        self.sum(alpha)
    }

    fn zk_residual_handoff(self, rs: &Point<EF>, alpha: EF, eps: EF) -> ProductPolynomial<F, EF>
    where
        EF: TwoAdicField,
    {
        // Prefix packs the residual weights: one full SIMD lane must survive the fold.
        // `Poly::pack` requires `num_variables - k >= k_pack`, else it panics.
        // Suffix is unpacked and unconstrained, so this guard is prefix-only.
        // Phrased as `k + k_pack <= num_variables` to avoid `usize` underflow.
        let k_pack = log2_strict_usize(<F as Field>::Packing::WIDTH);
        assert!(
            rs.num_variables() + k_pack <= self.num_variables(),
            "prefix packed residual needs num_variables - folding >= k_pack",
        );

        // Fold the stacked polynomial low-to-high.
        // The combining challenge is baked into the compression scale.
        let compressed = tracing::info_span!("compress_prefix_to_packed")
            .in_scope(|| self.poly.compress_prefix_to_packed(rs, eps));
        // Pack the equality weights for the SIMD-friendly residual rounds.
        let weights = self.combine_eqs(rs, alpha).pack::<F, EF>();
        ProductPolynomial::new_packed(VariableOrder::Prefix, compressed, weights)
    }
}

impl<F, EF> ZkLayout<F, EF> for SuffixProver<F, EF>
where
    F: TwoAdicField,
    EF: ExtensionField<F>,
{
    fn concrete_claims(&self) -> impl Iterator<Item = &ProverMultiClaim<F, EF>> {
        // Flatten placement order across all per-table claim lists.
        self.placements
            .iter()
            .flat_map(|placement| self.claim_map[placement.idx()].iter())
    }

    fn virtual_claims(&self) -> &[ProverVirtualClaim<EF>] {
        &self.virtual_claims
    }

    fn batched_sum(&self, alpha: EF) -> EF {
        self.sum(alpha)
    }

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
        let weights = self.combine_eqs(&reversed, alpha);
        ProductPolynomial::new_unpacked(VariableOrder::Suffix, compressed, weights)
    }
}
