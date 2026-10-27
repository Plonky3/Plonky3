//! Soundness terms for a batched product-tree GKR reduction.
//!
//! The numerators are `b`, `5 * R`, `L * (T - 1)`, and `h`.
//! They charge the batch point, sumcheck rounds, tree batching, and child collapse.
//! Here `b` counts batch coordinates, `R` degree-five rounds, and `L`, `T`, `h` describe the product trees.

use alloc::vec::Vec;

use crate::{ErrorBits, SecurityTerm};

/// Label for the point comparing a batched root table.
pub const PRODUCT_GKR_BATCH_POINT_LABEL: &str = "product-gkr-batch-point";

/// Label for degree-five product-reduction sumchecks.
pub const PRODUCT_GKR_SUMCHECK_LABEL: &str = "bus-product-gkr-sumcheck";

/// Label for random linear combinations of product trees.
pub const PRODUCT_GKR_BATCHING_LABEL: &str = "bus-product-gkr-tree-batching";

/// Label for random coordinates that collapse child claims.
pub const PRODUCT_GKR_COLLAPSE_LABEL: &str = "bus-product-gkr-child-collapse";

/// Caller-supplied security counts for one product-GKR schedule.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProductGkrSecurityProfile {
    /// Variables in each logical product tree.
    log_height: usize,
    /// Product trees reduced under one batching challenge per layer.
    tree_count: usize,
    /// Degree-five sumcheck rounds across every reduction layer.
    sumcheck_rounds: usize,
    /// Root-to-leaf reduction layers.
    layer_count: usize,
    /// Random coordinates used to collapse child claims.
    collapse_challenges: usize,
    /// Coordinates of the point comparing a batched root table, zero for separate trees.
    batch_variables: usize,
}

impl ProductGkrSecurityProfile {
    /// Checks representability and the relations encoded directly by these fields.
    ///
    /// The remaining counts are trusted inputs to the numeric model.
    /// Protocol code should derive them from its executed schedule.
    /// The binary bus does so through `BusPlan::security_term` in `p3-bus`.
    #[must_use]
    pub const fn new(
        log_height: usize,
        tree_count: usize,
        sumcheck_rounds: usize,
        layer_count: usize,
        collapse_challenges: usize,
    ) -> Option<Self> {
        if log_height >= usize::BITS as usize
            || tree_count == 0
            || tree_count > usize::MAX / 4
            || collapse_challenges != log_height
            || (log_height == 0) != (layer_count == 0)
        {
            return None;
        }
        Some(Self {
            log_height,
            tree_count,
            sumcheck_rounds,
            layer_count,
            collapse_challenges,
            batch_variables: 0,
        })
    }

    /// The same counts for `2^batch_variables` trees held in one table and compared at a random point.
    #[must_use]
    pub const fn with_batch_variables(mut self, batch_variables: usize) -> Self {
        self.batch_variables = batch_variables;
        self
    }

    /// Variables in each logical product tree.
    #[must_use]
    pub const fn log_height(&self) -> usize {
        self.log_height
    }

    /// Return each algebraic error source of the reduction over a `field_bits`-bit challenge field.
    #[must_use]
    pub fn components(&self, field_bits: usize) -> Vec<SecurityTerm> {
        let mut terms = Vec::with_capacity(4);
        // Two distinct root tables agree at a random point of their extensions only at a root.
        push_nonzero_term(
            &mut terms,
            PRODUCT_GKR_BATCH_POINT_LABEL,
            field_bits,
            self.batch_variables as u128,
        );
        // Each count comes from the concrete product schedule.
        push_nonzero_term(
            &mut terms,
            PRODUCT_GKR_SUMCHECK_LABEL,
            field_bits,
            5 * self.sumcheck_rounds as u128,
        );
        push_nonzero_term(
            &mut terms,
            PRODUCT_GKR_BATCHING_LABEL,
            field_bits,
            self.layer_count as u128 * self.tree_count.saturating_sub(1) as u128,
        );
        push_nonzero_term(
            &mut terms,
            PRODUCT_GKR_COLLAPSE_LABEL,
            field_bits,
            self.collapse_challenges as u128,
        );
        terms
    }
}

/// Append one union-bound term when its event can occur.
fn push_nonzero_term(
    terms: &mut Vec<SecurityTerm>,
    label: &'static str,
    field_bits: usize,
    numerator: u128,
) {
    // A degree-zero check has no Schwartz--Zippel failure event.
    if numerator != 0 {
        terms.push(SecurityTerm::new(
            label,
            ErrorBits::from_numerator(field_bits, numerator),
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_batched_table_charges_its_point_and_no_tree_batching() {
        // Height 4 over 2^3 trees: rounds 3 + 5 = 8, one tree, four collapse coordinates.
        let profile = ProductGkrSecurityProfile::new(4, 1, 8, 2, 4)
            .unwrap()
            .with_batch_variables(3);
        let terms = profile.components(192);
        let labels: Vec<_> = terms.iter().map(|term| term.label).collect();
        assert_eq!(
            labels,
            [
                PRODUCT_GKR_BATCH_POINT_LABEL,
                PRODUCT_GKR_SUMCHECK_LABEL,
                PRODUCT_GKR_COLLAPSE_LABEL
            ]
        );
        assert_eq!(terms[0].bits.bits(), 192.0 - libm::log2(3.0));
        assert_eq!(terms[1].bits.bits(), 192.0 - libm::log2(40.0));
    }
}
