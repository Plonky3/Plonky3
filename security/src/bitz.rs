//! Soundness terms of a BitZ reduction: the grand product in the exponent and the inner-product sumcheck.
//!
//! Round 2 and Phase 3 of Construction 4.4 lose nothing once every fold is below `ord(g)`, which the parameters check.
//! The grand product is charged as Lemma 6.2 through its own schedule, and the inner product `2 r1 / |K|`.

use alloc::vec::Vec;

use crate::gkr::{ProductGkrSecurityProfile, error_from_numerator};
use crate::{ErrorBits, SecurityTerm};

/// Label for the quadratic sumcheck reducing the leaf claim to the committed bits.
pub const BITZ_INNER_PRODUCT_LABEL: &str = "bitz-inner-product-sumcheck";

/// Counts of one BitZ reduction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BitzSecurityModel {
    /// Bits of the challenge field.
    field_bits: usize,
    /// Quadratic rounds of the inner product, one per row variable.
    inner_rounds: usize,
    /// Grinding before every inner-product challenge.
    inner_pow_bits: usize,
    /// The grand product's schedule.
    product: ProductGkrSecurityProfile,
}

impl BitzSecurityModel {
    /// The model of a reduction with this schedule.
    #[must_use]
    pub const fn new(
        field_bits: usize,
        inner_rounds: usize,
        inner_pow_bits: usize,
        product: ProductGkrSecurityProfile,
    ) -> Self {
        Self {
            field_bits,
            inner_rounds,
            inner_pow_bits,
            product,
        }
    }

    /// Every algebraic error source, for a union bound with the opening's.
    #[must_use]
    pub fn components(&self) -> Vec<SecurityTerm> {
        let mut terms = self.product.components(self.field_bits);
        if self.inner_rounds > 0 {
            // Each round is a degree-two check; grinding multiplies the cost of every retry.
            let bits = error_from_numerator(self.field_bits, 2 * self.inner_rounds as u128).bits()
                + self.inner_pow_bits as f64;
            terms.push(SecurityTerm::new(
                BITZ_INNER_PRODUCT_LABEL,
                ErrorBits::from_log2(bits),
            ));
        }
        terms
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_terms_are_negligible_over_a_192_bit_field() {
        // A shape of 2^16 rows, 2^12 columns and one claim.
        let profile = ProductGkrSecurityProfile::new(16, 1, 8 * 13 + 4 * 16, 8, 16)
            .unwrap()
            .with_batch_variables(12);
        let model = BitzSecurityModel::new(192, 16, 0, profile);
        let terms = model.components();
        assert!(terms.iter().any(|t| t.label == BITZ_INNER_PRODUCT_LABEL));
        assert!(terms.iter().all(|t| t.bits.bits() > 175.0), "{terms:?}");
        // Grinding adds its bits to the inner-product term.
        let ground = BitzSecurityModel::new(192, 16, 10, profile).components();
        let inner = |terms: &[SecurityTerm]| {
            terms
                .iter()
                .find(|t| t.label == BITZ_INNER_PRODUCT_LABEL)
                .unwrap()
                .bits
                .bits()
        };
        assert_eq!(inner(&ground), inner(&terms) + 10.0);
    }
}
