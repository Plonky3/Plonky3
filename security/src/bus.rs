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

/// Label for the union of every binary-bus algebraic error event.
pub const BINARY_BUS_LABEL: &str = "binary-bus";

/// Label for collisions in tuple compression and the product offset.
pub const BUS_FINGERPRINT_LABEL: &str = "bus-tuple-fingerprint";

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
}

/// Validated dimensions of one binary-native bus argument.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BusSecurityModel {
    /// Lower bound on the base-two logarithm of the challenge-field order.
    field_bits: usize,
    /// Variables in the multilinear tuple compressor.
    tuple_variables: usize,
    /// Declared factor positions on the push and pull sides.
    non_padding_leaf_counts: [usize; 2],
    /// Counts derived from the concrete product schedule.
    product: ProductGkrSecurityProfile,
}

impl BusSecurityModel {
    /// Checks representability and compatibility with the supplied product profile.
    ///
    /// Each declared factor count must fit in the logical product tree.
    /// An absent bus or an unrepresentable product shape returns no model.
    /// Other counts are trusted inputs to the numeric model.
    /// The binary bus derives checked counts through `BusPlan::security_term` in `p3-bus`.
    #[must_use]
    pub const fn new(
        field_bits: usize,
        tuple_variables: usize,
        non_padding_leaf_counts: [usize; 2],
        product: ProductGkrSecurityProfile,
    ) -> Option<Self> {
        // The implementation addresses a logical tree with one machine word.
        if field_bits == 0 || tuple_variables >= usize::BITS as usize {
            return None;
        }

        // Every non-padding factor must lie inside the authenticated tree domain.
        let capacity = 1usize << product.log_height;
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
            product,
        })
    }

    /// Return each algebraic error source for diagnostic reporting.
    ///
    /// The result excludes commitment binding and authentication of terminal leaf claims.
    /// These components must not be passed separately as protocol extras.
    #[must_use]
    pub fn components(&self) -> Vec<SecurityTerm> {
        // A product difference has total degree at most max(1, s) * N.
        // Here s is the tuple dimension and N is the larger multiset capacity in use.
        let factors = self.non_padding_leaf_counts[0].max(self.non_padding_leaf_counts[1]);
        let fingerprint_numerator = self.tuple_variables.max(1) as u128 * factors as u128;
        let mut terms = alloc::vec![SecurityTerm::new(
            BUS_FINGERPRINT_LABEL,
            error_from_numerator(self.field_bits, fingerprint_numerator),
        )];

        // The protocol adapter supplies counts from the concrete product schedule.
        push_nonzero_term(
            &mut terms,
            PRODUCT_GKR_SUMCHECK_LABEL,
            self.field_bits,
            5 * self.product.sumcheck_rounds as u128,
        );
        push_nonzero_term(
            &mut terms,
            PRODUCT_GKR_BATCHING_LABEL,
            self.field_bits,
            self.product.layer_count as u128 * self.product.tree_count.saturating_sub(1) as u128,
        );
        push_nonzero_term(
            &mut terms,
            PRODUCT_GKR_COLLAPSE_LABEL,
            self.field_bits,
            self.product.collapse_challenges as u128,
        );

        terms
    }

    /// Return the union bound that composes as one protocol extra.
    #[must_use]
    pub fn combined_term(&self) -> SecurityTerm {
        let components = self.components();
        let errors = components
            .iter()
            .map(|component| component.bits)
            .collect::<Vec<_>>();
        SecurityTerm::new(BINARY_BUS_LABEL, ErrorBits::sum(&errors))
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
    // This keeps the reported error at least as large up to f64 rounding.
    let mut rounded = numerator as f64;
    if (rounded as u128) < numerator {
        rounded = rounded.next_up();
    }
    ErrorBits::from_log2((field_bits as f64 - libm::log2(rounded)).max(0.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fri::FriRegime;
    use crate::grinding::GrindingSites;
    use crate::shape::{InstanceShape, StarkAirParams};
    use crate::stark::proven_security_report;

    fn profile(
        log_height: usize,
        tree_count: usize,
        sumcheck_rounds: usize,
        layer_count: usize,
    ) -> ProductGkrSecurityProfile {
        ProductGkrSecurityProfile::new(
            log_height,
            tree_count,
            sumcheck_rounds,
            layer_count,
            log_height,
        )
        .unwrap()
    }

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
        let model = BusSecurityModel::new(128, 4, [1 << 20, 1 << 18], profile(20, 2, 90, 10))
            .expect("the dimensions fit the product tree");
        let terms = model.components();

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
        let terms = BusSecurityModel::new(128, 0, [1, 1], profile(0, 2, 0, 0))
            .expect("one factor fits a zero-height tree")
            .components();

        assert_eq!(terms.len(), 1);
        assert_eq!(terms[0].label, BUS_FINGERPRINT_LABEL);
        assert_eq!(terms[0].bits.bits(), 128.0);
    }

    #[test]
    fn odd_height_charges_the_leading_binary_collapse() {
        // A height-three tree starts with one binary level.
        // Its remaining radix-four layer runs one degree-five round.
        let terms = BusSecurityModel::new(128, 1, [8, 8], profile(3, 2, 1, 2))
            .expect("eight factors fit a height-three tree")
            .components();

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
        assert!(BusSecurityModel::new(0, 1, [1, 1], profile(0, 2, 0, 0)).is_none());
        assert!(ProductGkrSecurityProfile::new(0, 0, 0, 0, 0).is_none());
        assert!(ProductGkrSecurityProfile::new(0, usize::MAX, 0, 0, 0).is_none());
        assert!(
            BusSecurityModel::new(128, usize::BITS as usize, [1, 1], profile(0, 2, 0, 0),)
                .is_none()
        );
        assert!(ProductGkrSecurityProfile::new(usize::BITS as usize, 2, 0, 0, 0).is_none());
        assert!(BusSecurityModel::new(128, 1, [2, 1], profile(0, 2, 0, 0)).is_none());
        assert!(BusSecurityModel::new(128, 1, [0, 0], profile(0, 2, 0, 0)).is_none());
        assert!(ProductGkrSecurityProfile::new(1, 2, 0, 0, 0).is_none());
    }

    #[test]
    fn a_192_bit_field_adds_sixty_four_bits_to_every_term() {
        // Both models have identical algebraic numerators.
        // Only the challenge-field denominator differs.
        let narrow = BusSecurityModel::new(128, 4, [1 << 20, 1 << 18], profile(20, 2, 90, 10))
            .expect("the 128-bit model is valid")
            .components();
        let wide = BusSecurityModel::new(192, 4, [1 << 20, 1 << 18], profile(20, 2, 90, 10))
            .expect("the 192-bit model is valid")
            .components();

        assert_eq!(narrow.len(), wide.len());
        for (narrow_term, wide_term) in narrow.iter().zip(wide) {
            assert_eq!(narrow_term.label, wide_term.label);
            assert!((wide_term.bits.bits() - narrow_term.bits.bits() - 64.0).abs() < 1e-12);
        }
    }

    #[test]
    fn combined_term_sums_all_component_probabilities() {
        // Height three has four nonzero error sources with numerators 8, 5, 2, and 3.
        let model = BusSecurityModel::new(128, 1, [8, 8], profile(3, 2, 1, 2)).unwrap();
        let components = model.components();
        let expected = ErrorBits::sum(
            &components
                .iter()
                .map(|component| component.bits)
                .collect::<Vec<_>>(),
        );
        assert_eq!(model.combined_term().bits, expected);
        assert_eq!(model.combined_term().label, BINARY_BUS_LABEL);
    }

    #[test]
    fn combined_term_composes_as_one_protocol_extra() {
        // Use a small field so the bus union bound is visible in the final report.
        let regime = FriRegime {
            log_blowup: 1,
            num_queries: 100,
            log_final_poly_len: 0,
            max_log_arity: 3,
            commit_pow_bits: 0,
            query_pow_bits: 16,
        };
        let air = StarkAirParams {
            num_constraints: 1,
            max_constraint_degree: 2,
            num_quotient_chunks: 1,
            max_combo: 2,
        };
        let shape = InstanceShape {
            log_trace_length: 20,
            modulus_bits: 64,
            collision_resistance: 128,
            num_batched_functions: 1,
        };
        let model =
            BusSecurityModel::new(64, 4, [1 << 20, 1 << 18], profile(20, 2, 90, 10)).unwrap();
        let term = model.combined_term();

        // The report contains one composed event rather than four independent minima.
        let report = proven_security_report(&regime, &air, &shape, &[term], &GrindingSites::NONE);
        assert!(
            report
                .udr
                .terms()
                .iter()
                .any(|candidate| candidate.label == BINARY_BUS_LABEL)
        );
        assert!(!report.udr.terms().iter().any(|candidate| {
            [
                BUS_FINGERPRINT_LABEL,
                PRODUCT_GKR_SUMCHECK_LABEL,
                PRODUCT_GKR_BATCHING_LABEL,
                PRODUCT_GKR_COLLAPSE_LABEL,
            ]
            .contains(&candidate.label)
        }));
    }

    #[test]
    fn large_numerator_rounding_branch_is_exercised() {
        // The fingerprint numerator exceeds the exact integer range of f64.
        let model = BusSecurityModel::new(128, 3, [(1usize << 62) + 1, 1], profile(63, 2, 961, 32))
            .unwrap();
        assert!(model.components()[0].bits.bits().is_finite());
    }
}
