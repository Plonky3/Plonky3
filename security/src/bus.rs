//! Soundness terms for a binary-native multiset bus.
//!
//! Challenges are uniform field elements sampled after the tuple columns are committed.
//!
//! The four numerators are `max(1, s) * N`, `5 * R`, `L * (T - 1)`, and `h`.
//! They charge tuple collisions, sumcheck rounds, tree batching, and child collapse.
//! Here `R` counts radix-four rounds while `L`, `T`, and `h` describe the product trees.
//!
//! Commitment binding and terminal-claim authentication are composed by the caller.

use alloc::vec::Vec;

use crate::{ErrorBits, SecurityTerm};

/// Label for collisions in tuple compression and the product offset.
pub const BUS_FINGERPRINT_LABEL: &str = "bus-tuple-fingerprint";

/// Label for degree-five product-reduction sumchecks.
pub const PRODUCT_GKR_SUMCHECK_LABEL: &str = "bus-product-gkr-sumcheck";

/// Label for random linear combinations of product trees.
pub const PRODUCT_GKR_BATCHING_LABEL: &str = "bus-product-gkr-tree-batching";

/// Label for random coordinates that collapse child claims.
pub const PRODUCT_GKR_COLLAPSE_LABEL: &str = "bus-product-gkr-child-collapse";

/// Validated dimensions of one binary-native bus argument.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BusSecurityModel {
    /// Lower bound on the base-two logarithm of the challenge-field order.
    field_bits: usize,
    /// Variables in the multilinear tuple compressor.
    tuple_variables: usize,
    /// Declared factor positions on the push and pull sides.
    non_padding_leaf_counts: [usize; 2],
    /// Variables in each identity-padded product tree.
    product_log_height: usize,
    /// Product trees reduced under one batching challenge per layer.
    product_tree_count: usize,
}

impl BusSecurityModel {
    /// Validate the challenge field and every security-relevant dimension.
    ///
    /// Each declared factor count must fit in the logical product tree.
    /// An absent bus or an unrepresentable product shape returns no model.
    #[must_use]
    pub const fn new(
        field_bits: usize,
        tuple_variables: usize,
        non_padding_leaf_counts: [usize; 2],
        product_log_height: usize,
        product_tree_count: usize,
    ) -> Option<Self> {
        // The implementation addresses a logical tree with one machine word.
        if field_bits == 0
            || tuple_variables >= usize::BITS as usize
            || product_log_height >= usize::BITS as usize
            || product_tree_count == 0
            || product_tree_count > usize::MAX / 4
        {
            return None;
        }

        // Every non-padding factor must lie inside the authenticated tree domain.
        let capacity = 1usize << product_log_height;
        if non_padding_leaf_counts[0] > capacity
            || non_padding_leaf_counts[1] > capacity
            || (non_padding_leaf_counts[0] == 0 && non_padding_leaf_counts[1] == 0)
        {
            return None;
        }

        Some(Self {
            field_bits,
            tuple_variables,
            non_padding_leaf_counts,
            product_log_height,
            product_tree_count,
        })
    }

    /// Return each algebraic error source as a separate report term.
    ///
    /// The result excludes commitment binding and authentication of terminal leaf claims.
    #[must_use]
    pub fn terms(&self) -> Vec<SecurityTerm> {
        // A product difference has total degree at most max(1, s) * N.
        // Here s is the tuple dimension and N is the larger multiset capacity in use.
        let factors = self.non_padding_leaf_counts[0].max(self.non_padding_leaf_counts[1]);
        let fingerprint_numerator = self.tuple_variables.max(1) as u128 * factors as u128;
        let mut terms = alloc::vec![SecurityTerm::new(
            BUS_FINGERPRINT_LABEL,
            error_from_numerator(self.field_bits, fingerprint_numerator),
        )];

        // Radix four contracts two multiplication levels after any leading binary layer.
        // A layer at point length r runs r degree-five sumcheck rounds.
        let mut remaining = self.product_log_height;
        let mut point_len = 0usize;
        let mut layer_count = 0u128;
        let mut radix_four_rounds = 0u128;
        while remaining > 0 {
            let branch_count = if remaining == self.product_log_height && remaining % 2 == 1 {
                1
            } else {
                radix_four_rounds += point_len as u128;
                2
            };
            point_len += branch_count;
            remaining -= branch_count;
            layer_count += 1;
        }

        // A zero-height product has no reduction transcript and contributes no GKR term.
        push_nonzero_term(
            &mut terms,
            PRODUCT_GKR_SUMCHECK_LABEL,
            self.field_bits,
            5 * radix_four_rounds,
        );
        push_nonzero_term(
            &mut terms,
            PRODUCT_GKR_BATCHING_LABEL,
            self.field_bits,
            layer_count * self.product_tree_count.saturating_sub(1) as u128,
        );
        push_nonzero_term(
            &mut terms,
            PRODUCT_GKR_COLLAPSE_LABEL,
            self.field_bits,
            self.product_log_height as u128,
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
fn error_from_numerator(field_bits: usize, numerator: u128) -> ErrorBits {
    // Round upward before taking the logarithm.
    // This keeps the reported error at least as large as the exact rational bound.
    let mut rounded = numerator as f64;
    if (rounded as u128) < numerator {
        rounded = rounded.next_up();
    }
    ErrorBits::from_log2((field_bits as f64 - libm::log2(rounded)).max(0.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bits(terms: &[SecurityTerm], label: &str) -> Option<f64> {
        // Labels are unique because each random experiment is reported once.
        terms
            .iter()
            .find(|term| term.label == label)
            .map(|term| term.bits.bits())
    }

    #[test]
    fn exact_terms_follow_a_twenty_variable_bus() {
        // Fixture state:
        //
        //     tuple fingerprint  4 * 2^20 roots
        //     radix-four rounds  0 + 2 + ... + 18 = 90
        //     tree batching      10 layers * (2 - 1) roots
        //     child collapse     20 coordinates
        let model = BusSecurityModel::new(128, 4, [1 << 20, 1 << 18], 20, 2)
            .expect("the dimensions fit the product tree");
        let terms = model.terms();

        assert_eq!(
            bits(&terms, BUS_FINGERPRINT_LABEL),
            Some(128.0 - libm::log2((4u64 << 20) as f64))
        );
        assert_eq!(
            bits(&terms, PRODUCT_GKR_SUMCHECK_LABEL),
            Some(128.0 - libm::log2(450.0))
        );
        assert_eq!(
            bits(&terms, PRODUCT_GKR_BATCHING_LABEL),
            Some(128.0 - libm::log2(10.0))
        );
        assert_eq!(
            bits(&terms, PRODUCT_GKR_COLLAPSE_LABEL),
            Some(128.0 - libm::log2(20.0))
        );
    }

    #[test]
    fn zero_height_omits_every_product_reduction_term() {
        // A one-leaf tree compares its roots directly.
        // Only tuple compression can hide an unequal multiset.
        let terms = BusSecurityModel::new(128, 0, [1, 1], 0, 2)
            .expect("one factor fits a zero-height tree")
            .terms();

        assert_eq!(terms.len(), 1);
        assert_eq!(terms[0].label, BUS_FINGERPRINT_LABEL);
        assert_eq!(terms[0].bits.bits(), 128.0);
    }

    #[test]
    fn odd_height_charges_the_leading_binary_collapse() {
        // A height-three tree starts with one binary level.
        // Its remaining radix-four layer runs one degree-five round.
        let terms = BusSecurityModel::new(128, 1, [8, 8], 3, 2)
            .expect("eight factors fit a height-three tree")
            .terms();

        assert_eq!(
            bits(&terms, PRODUCT_GKR_SUMCHECK_LABEL),
            Some(128.0 - libm::log2(5.0))
        );
        assert_eq!(bits(&terms, PRODUCT_GKR_BATCHING_LABEL), Some(127.0));
        assert_eq!(
            bits(&terms, PRODUCT_GKR_COLLAPSE_LABEL),
            Some(128.0 - libm::log2(3.0))
        );
    }

    #[test]
    fn malformed_or_oversized_shapes_have_no_model() {
        // Mutation matrix:
        //
        //     zero field width       no challenge entropy
        //     zero trees             no product statement
        //     oversized tree count   child-message length can overflow
        //     oversized height       logical capacity cannot be shifted
        //     two leaves at height 0 factor lies outside the tree
        //     no factors             no bus statement
        assert!(BusSecurityModel::new(0, 1, [1, 1], 0, 2).is_none());
        assert!(BusSecurityModel::new(128, 1, [1, 1], 0, 0).is_none());
        assert!(BusSecurityModel::new(128, 1, [1, 1], 0, usize::MAX).is_none());
        assert!(BusSecurityModel::new(128, usize::BITS as usize, [1, 1], 0, 2).is_none());
        assert!(BusSecurityModel::new(128, 1, [1, 1], usize::BITS as usize, 2).is_none());
        assert!(BusSecurityModel::new(128, 1, [2, 1], 0, 2).is_none());
        assert!(BusSecurityModel::new(128, 1, [0, 0], 0, 2).is_none());
    }

    #[test]
    fn a_192_bit_field_adds_sixty_four_bits_to_every_term() {
        // Both models have identical algebraic numerators.
        // Only the challenge-field denominator differs.
        let narrow = BusSecurityModel::new(128, 4, [1 << 20, 1 << 18], 20, 2)
            .expect("the 128-bit model is valid")
            .terms();
        let wide = BusSecurityModel::new(192, 4, [1 << 20, 1 << 18], 20, 2)
            .expect("the 192-bit model is valid")
            .terms();

        assert_eq!(narrow.len(), wide.len());
        for (narrow_term, wide_term) in narrow.iter().zip(wide) {
            assert_eq!(narrow_term.label, wide_term.label);
            assert!((wide_term.bits.bits() - narrow_term.bits.bits() - 64.0).abs() < 1e-12);
        }
    }
}
