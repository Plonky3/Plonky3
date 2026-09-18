//! Security accounting derived from a checked bus layout.

use alloc::vec::Vec;

use p3_security::SecurityTerm;
use p3_security::bus::BusSecurityModel;

use crate::{BusSecurityGeometry, ProductGkrRootShape, ProductGkrShape};

/// Build separately labelled soundness terms from verifier-owned dimensions.
///
/// Returns no terms if the dimensions are inconsistent or the field width is zero.
/// The result excludes commitment binding and authentication of terminal leaf claims.
#[must_use]
pub fn bus_security_terms(
    geometry: BusSecurityGeometry,
    field_bits: usize,
) -> Option<Vec<SecurityTerm>> {
    // The redundant capacity fields make accidental geometry drift detectable here.
    let shift = u32::try_from(geometry.log_logical_leaf_count).ok()?;
    let logical_leaf_count = 1usize.checked_shl(shift)?;
    let product_shape = ProductGkrShape::new(
        geometry.log_logical_leaf_count,
        geometry.tree_count,
        ProductGkrRootShape::FirstTwoShared,
    )
    .ok()?;
    if geometry.logical_leaf_count != logical_leaf_count
        || geometry.layer_count != product_shape.layers().len()
    {
        return None;
    }

    // The generic model validates factor counts and the product-message dimensions.
    BusSecurityModel::new(
        field_bits,
        geometry.tuple_variables,
        geometry.non_padding_leaf_counts,
        geometry.log_logical_leaf_count,
        geometry.tree_count,
    )
    .map(|model| model.terms())
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_security::bus::{
        BUS_FINGERPRINT_LABEL, PRODUCT_GKR_BATCHING_LABEL, PRODUCT_GKR_COLLAPSE_LABEL,
        PRODUCT_GKR_SUMCHECK_LABEL,
    };

    use super::*;

    // A small checked layout has two radix-four reduction layers.
    const GEOMETRY: BusSecurityGeometry = BusSecurityGeometry {
        tuple_variables: 3,
        non_padding_leaf_counts: [16, 8],
        logical_leaf_count: 16,
        log_logical_leaf_count: 4,
        tree_count: 2,
        layer_count: 2,
    };

    #[test]
    fn adapter_preserves_every_separate_error_source() {
        // Fixture state:
        //
        //     fingerprint      3 * 16 roots
        //     sumcheck         5 * 2 roots
        //     tree batching    2 * (2 - 1) roots
        //     child collapse   4 roots
        let terms = bus_security_terms(GEOMETRY, 128).expect("the checked geometry is valid");
        let labels = terms.iter().map(|term| term.label).collect::<Vec<_>>();

        assert_eq!(
            labels,
            vec![
                BUS_FINGERPRINT_LABEL,
                PRODUCT_GKR_SUMCHECK_LABEL,
                PRODUCT_GKR_BATCHING_LABEL,
                PRODUCT_GKR_COLLAPSE_LABEL,
            ]
        );
        assert_eq!(terms[0].bits.bits(), 128.0 - 48f64.log2());
        assert_eq!(terms[1].bits.bits(), 128.0 - 10f64.log2());
        assert_eq!(terms[2].bits.bits(), 127.0);
        assert_eq!(terms[3].bits.bits(), 126.0);
    }

    #[test]
    fn adapter_rejects_forged_redundant_geometry() {
        // A caller cannot lower the reported layer count while retaining a height-four tree.
        let wrong_layers = BusSecurityGeometry {
            layer_count: 1,
            ..GEOMETRY
        };
        assert!(bus_security_terms(wrong_layers, 128).is_none());

        // A caller cannot claim an eight-leaf capacity for a height-four tree.
        let wrong_capacity = BusSecurityGeometry {
            logical_leaf_count: 8,
            ..GEOMETRY
        };
        assert!(bus_security_terms(wrong_capacity, 128).is_none());

        // A shared push/pull root requires both direction-specific product trees.
        let one_tree = BusSecurityGeometry {
            tree_count: 1,
            ..GEOMETRY
        };
        assert!(bus_security_terms(one_tree, 128).is_none());

        // A zero-width challenge space cannot support algebraic soundness.
        assert!(bus_security_terms(GEOMETRY, 0).is_none());
    }

    #[test]
    fn adapter_rejects_shift_overflow_without_panicking() {
        // The untrusted copy of the geometry requests an impossible machine-word shift.
        let oversized = BusSecurityGeometry {
            logical_leaf_count: 1,
            log_logical_leaf_count: usize::BITS as usize,
            layer_count: (usize::BITS as usize).div_ceil(2),
            ..GEOMETRY
        };
        assert!(bus_security_terms(oversized, 128).is_none());
    }
}
