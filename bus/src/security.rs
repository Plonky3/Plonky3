//! Security accounting derived from a checked bus layout.

use alloc::vec::Vec;
use core::num::NonZeroUsize;

use p3_security::SecurityTerm;
use p3_security::bus::{BusSecurityModel, ProductGkrSecurityProfile};

use crate::BusPlan;

impl BusPlan {
    /// Builds the union-bound term consumed by a protocol security report.
    ///
    /// `field_bits` is a lower bound on `log2 |EF|`.
    /// Every bus challenge must be sampled from that same extension field.
    /// This includes the fingerprint, offset, and product-GKR challenges.
    ///
    /// The result excludes commitment binding and authentication of terminal leaf claims.
    #[must_use]
    pub fn security_term(&self, field_bits: NonZeroUsize) -> SecurityTerm {
        self.security_model(field_bits).combined_term()
    }

    /// Builds separately labelled terms for diagnostic reporting.
    ///
    /// These components must not be passed separately as protocol extras.
    /// Their probability sum is represented by the single composable term.
    #[must_use]
    pub fn security_components(&self, field_bits: NonZeroUsize) -> Vec<SecurityTerm> {
        self.security_model(field_bits).components()
    }

    /// Derives exact soundness dimensions from the concrete product schedule.
    fn security_model(&self, field_bits: NonZeroUsize) -> BusSecurityModel {
        let shape = self.product_shape();
        let layers = shape.layers();
        let sumcheck_rounds = layers.iter().map(|(_, rounds)| rounds).sum();
        let collapse_challenges = layers
            .iter()
            .map(|(arity, _)| arity.trailing_zeros() as usize)
            .sum();
        let geometry = self.security_geometry();
        let profile = ProductGkrSecurityProfile::new(
            shape.log_height(),
            shape.num_trees(),
            sumcheck_rounds,
            layers.len(),
            collapse_challenges,
        )
        .expect("a checked product shape has valid security dimensions");

        // A checked plan supplies dimensions accepted by the numeric security model.
        BusSecurityModel::new(
            field_bits.get(),
            geometry.tuple_variables(),
            geometry.non_padding_leaf_counts(),
            profile,
        )
        .expect("a checked bus plan has valid security dimensions")
    }
}

#[cfg(test)]
mod tests {
    use alloc::string::ToString;
    use alloc::vec;

    use p3_air::symbolic::{BaseEntry, SymbolicVariable};
    use p3_baby_bear::BabyBear;
    use p3_security::ErrorBits;
    use p3_security::bus::{
        BINARY_BUS_LABEL, BUS_FINGERPRINT_LABEL, PRODUCT_GKR_BATCHING_LABEL,
        PRODUCT_GKR_COLLAPSE_LABEL, PRODUCT_GKR_SUMCHECK_LABEL,
    };

    use super::*;
    use crate::{
        BusActivation, BusDirection, BusPlanInput, ProductGkrRootShape, ProductGkrShape,
        SymbolicBusInteraction,
    };

    /// Build one balanced bus at a chosen trace height and tuple width.
    fn plan(log_height: usize, width: usize) -> BusPlan {
        let fields = (0..width)
            .map(|index| SymbolicVariable::new(BaseEntry::Main { offset: 0 }, index).into())
            .collect::<Vec<_>>();
        let interactions = BusDirection::ALL.map(|direction| SymbolicBusInteraction::<BabyBear> {
            bus_name: "bus".to_string(),
            direction,
            fields: fields.clone(),
            activation: BusActivation::Always,
        });
        BusPlan::build(&[BusPlanInput {
            log_height,
            interactions: &interactions,
        }])
        .unwrap()
        .unwrap()
    }

    #[test]
    fn adapter_preserves_components_and_returns_their_union() {
        // Height four has two radix-four layers with zero and two rounds.
        let plan = plan(4, 5);
        let field_bits = NonZeroUsize::new(128).unwrap();
        let components = plan.security_components(field_bits);
        let labels = components
            .iter()
            .map(|component| component.label)
            .collect::<Vec<_>>();
        assert_eq!(
            labels,
            vec![
                BUS_FINGERPRINT_LABEL,
                PRODUCT_GKR_SUMCHECK_LABEL,
                PRODUCT_GKR_BATCHING_LABEL,
                PRODUCT_GKR_COLLAPSE_LABEL,
            ]
        );

        // Three tuple variables compress eight slots across sixteen active rows per side.
        let fingerprint = components
            .iter()
            .find(|component| component.label == BUS_FINGERPRINT_LABEL)
            .unwrap();
        assert_eq!(fingerprint.bits.bits(), 128.0 - 48.0_f64.log2());

        // The composable term charges the probability sum rather than its largest component.
        let expected = ErrorBits::sum(
            &components
                .iter()
                .map(|component| component.bits)
                .collect::<Vec<_>>(),
        );
        let combined = plan.security_term(field_bits);
        assert_eq!(combined.label, BINARY_BUS_LABEL);
        assert_eq!(combined.bits, expected);
        assert_eq!(combined.bits.bits(), 122.0);
    }

    #[test]
    fn every_supported_height_uses_the_concrete_product_schedule() {
        // Compare the adapter against the schedule executed by product GKR.
        for log_height in 0..=40 {
            let plan = plan(log_height, 1);
            let shape =
                ProductGkrShape::new(log_height, 2, ProductGkrRootShape::FirstTwoShared).unwrap();
            let layers = shape.layers();
            assert_eq!(plan.product_shape(), shape);

            let components = plan.security_components(NonZeroUsize::new(128).unwrap());
            let sumcheck_rounds = layers.iter().map(|(_, rounds)| rounds).sum::<usize>();
            let collapse_challenges = layers
                .iter()
                .map(|(arity, _)| arity.trailing_zeros() as usize)
                .sum::<usize>();
            let component = |label| components.iter().find(|term| term.label == label);

            if sumcheck_rounds == 0 {
                assert!(component(PRODUCT_GKR_SUMCHECK_LABEL).is_none());
            } else {
                let expected = 128.0 - ((5 * sumcheck_rounds) as f64).log2();
                assert_eq!(
                    component(PRODUCT_GKR_SUMCHECK_LABEL).unwrap().bits.bits(),
                    expected
                );
            }
            if layers.is_empty() {
                assert!(component(PRODUCT_GKR_BATCHING_LABEL).is_none());
            } else {
                let expected = 128.0 - (layers.len() as f64).log2();
                assert_eq!(
                    component(PRODUCT_GKR_BATCHING_LABEL).unwrap().bits.bits(),
                    expected
                );
            }
            if collapse_challenges == 0 {
                assert!(component(PRODUCT_GKR_COLLAPSE_LABEL).is_none());
            } else {
                let expected = 128.0 - (collapse_challenges as f64).log2();
                assert_eq!(
                    component(PRODUCT_GKR_COLLAPSE_LABEL).unwrap().bits.bits(),
                    expected
                );
            }
        }
    }
}
