//! Soundness terms for a batched product-tree GKR reduction.
//!
//! The three numerators are `5 * R`, `L * (T - 1)`, and `h`.
//! They charge sumcheck rounds, tree batching, and child collapse.
//! Here `R` counts degree-five rounds while `L`, `T`, and `h` describe the product trees.

use alloc::vec::Vec;

use crate::{ErrorBits, SecurityTerm};

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
        })
    }

    /// Variables in each logical product tree.
    #[must_use]
    pub const fn log_height(&self) -> usize {
        self.log_height
    }

    /// Return each algebraic error source of the reduction over a `field_bits`-bit challenge field.
    #[must_use]
    pub fn components(&self, field_bits: usize) -> Vec<SecurityTerm> {
        let mut terms = Vec::with_capacity(3);
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
            error_from_numerator(field_bits, numerator),
        ));
    }
}

/// Convert an exact numerator over the challenge-field order into bits.
pub(crate) fn error_from_numerator(field_bits: usize, numerator: u128) -> ErrorBits {
    // Round upward before taking the logarithm.
    // This keeps the reported error at least as large up to f64 rounding.
    let mut rounded = numerator as f64;
    if (rounded as u128) < numerator {
        rounded = rounded.next_up();
    }
    ErrorBits::from_log2((field_bits as f64 - libm::log2(rounded)).max(0.0))
}
