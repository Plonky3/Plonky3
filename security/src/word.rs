//! Soundness terms for the word shift reduction.

use alloc::vec::Vec;

use crate::{ErrorBits, SecurityTerm};

/// Label for the complete word shift reduction error.
pub const WORD_SHIFT_LABEL: &str = "word-shift-reduction";

/// Label for batching relation families and operand positions.
pub const WORD_SHIFT_BATCHING_LABEL: &str = "word-shift-batching";

/// Label for the two quadratic sumchecks.
pub const WORD_SHIFT_SUMCHECK_LABEL: &str = "word-shift-sumcheck";

/// Checked dimensions of one word shift reduction.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WordShiftSecurityModel {
    /// Lower bound on the base-two logarithm of the challenge-field order.
    field_bits: usize,
    /// Variables selecting a relation family or operand position.
    batching_variables: usize,
    /// Quadratic rounds over bit and word indices.
    sumcheck_rounds: usize,
}

impl WordShiftSecurityModel {
    /// Creates a model when the challenge field has a nonzero security width.
    #[must_use]
    pub const fn new(
        field_bits: usize,
        batching_variables: usize,
        sumcheck_rounds: usize,
    ) -> Option<Self> {
        // A zero-bit challenge field cannot support a statistical reduction.
        if field_bits == 0 {
            return None;
        }

        Some(Self {
            field_bits,
            batching_variables,
            sumcheck_rounds,
        })
    }

    /// Returns each algebraic error source for diagnostic reporting.
    #[must_use]
    pub fn components(self) -> Vec<SecurityTerm> {
        // Evaluating the batching multilinear at a random point costs its total degree.
        let mut terms = Vec::with_capacity(2);
        if self.batching_variables != 0 {
            terms.push(SecurityTerm::new(
                WORD_SHIFT_BATCHING_LABEL,
                error_from_numerator(self.field_bits, self.batching_variables as u128),
            ));
        }

        // Each quadratic sumcheck round contributes at most two roots.
        if self.sumcheck_rounds != 0 {
            terms.push(SecurityTerm::new(
                WORD_SHIFT_SUMCHECK_LABEL,
                error_from_numerator(self.field_bits, 2 * self.sumcheck_rounds as u128),
            ));
        }

        terms
    }

    /// Returns the union bound used by an enclosing proof system.
    #[must_use]
    pub fn combined_term(self) -> SecurityTerm {
        // Compose the independently labelled failure events by probability addition.
        let errors = self
            .components()
            .iter()
            .map(|component| component.bits)
            .collect::<Vec<_>>();
        SecurityTerm::new(WORD_SHIFT_LABEL, ErrorBits::sum(&errors))
    }
}

/// Converts an exact numerator over the challenge-field order into bits.
fn error_from_numerator(field_bits: usize, numerator: u128) -> ErrorBits {
    // Round upward so floating-point conversion cannot understate the error.
    let mut rounded = numerator as f64;
    if rounded as u128 != numerator {
        rounded = rounded.next_up();
    }
    ErrorBits::from_log2((field_bits as f64 - libm::log2(rounded)).max(0.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn word64_profile_charges_both_random_experiments() {
        // Fixture state:
        //
        //     batching point   2 operation bits + 2 operand bits
        //     bit sumcheck     6 quadratic rounds
        //     word sumcheck   20 quadratic rounds
        let model = WordShiftSecurityModel::new(128, 4, 26).unwrap();
        let components = model.components();

        // Four batching variables give the numerator four.
        assert_eq!(components[0].label, WORD_SHIFT_BATCHING_LABEL);
        assert_eq!(components[0].bits.bits(), 126.0);

        // Twenty-six quadratic rounds give the numerator fifty-two.
        assert_eq!(components[1].label, WORD_SHIFT_SUMCHECK_LABEL);
        assert_eq!(components[1].bits.bits(), 128.0 - libm::log2(52.0));

        // The composable term is the union of both events.
        let expected = ErrorBits::sum(&[components[0].bits, components[1].bits]);
        assert_eq!(model.combined_term().bits, expected);
    }

    #[test]
    fn deterministic_shape_omits_empty_terms() {
        // No sampled variable means no statistical failure event.
        let model = WordShiftSecurityModel::new(128, 0, 0).unwrap();
        assert!(model.components().is_empty());

        // A challenge field must expose at least one bit of entropy.
        assert_eq!(WordShiftSecurityModel::new(0, 4, 6), None);
    }
}
