//! Soundness terms for the word-level relation proof.

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

/// Label for the complete word-level relation proof error.
pub const WORD_PROOF_LABEL: &str = "word-proof";

/// Label for separating the relation families of one statement.
pub const WORD_RELATION_BATCHING_LABEL: &str = "word-relation-batching";

/// Label for the point the batched relations are required to vanish at.
pub const WORD_ZEROCHECK_POINT_LABEL: &str = "word-relation-zerocheck-point";

/// Label for the rounds of the batched relation vanishing check.
pub const WORD_ZEROCHECK_ROUNDS_LABEL: &str = "word-relation-zerocheck-rounds";

/// Checked dimensions of one complete word-level relation proof.
#[derive(Clone, Debug, PartialEq)]
pub struct WordProofSecurityModel {
    /// Lower bound on the base-two logarithm of the challenge-field order.
    field_bits: usize,
    /// Relation families combined under one batching coefficient.
    relation_families: usize,
    /// Constraint and within-word variables the vanishing check binds.
    zerocheck_variables: usize,
    /// Per-variable degree of the batched relation polynomial.
    zerocheck_degree: usize,
    /// Dimensions of the shift reduction that follows.
    shift: WordShiftSecurityModel,
    /// Labelled errors the commitment charges for discharging the one surviving claim.
    commitment: Vec<SecurityTerm>,
}

impl WordProofSecurityModel {
    /// Creates a model when every sampled experiment has a well-formed shape.
    ///
    /// The width argument is a lower bound on the base-two logarithm of the challenge-field order.
    ///
    /// Every batching, vanishing, and reduction challenge must be drawn from that same field.
    ///
    /// The last argument holds the terms the commitment charges, priced by the commitment itself.
    #[must_use]
    pub fn new(
        field_bits: usize,
        relation_families: usize,
        zerocheck_variables: usize,
        zerocheck_degree: usize,
        shift: WordShiftSecurityModel,
        commitment: Vec<SecurityTerm>,
    ) -> Option<Self> {
        // A degree-zero composition carries no round polynomial to separate against.
        if field_bits == 0 || zerocheck_degree == 0 {
            return None;
        }

        Some(Self {
            field_bits,
            relation_families,
            zerocheck_variables,
            zerocheck_degree,
            shift,
            commitment,
        })
    }

    /// Returns each algebraic error source for diagnostic reporting.
    #[must_use]
    pub fn components(&self) -> Vec<SecurityTerm> {
        let mut terms = Vec::new();

        // One coefficient separates the families, so its degree is one below their count.
        let separated = self.relation_families.saturating_sub(1);
        if separated != 0 {
            terms.push(SecurityTerm::new(
                WORD_RELATION_BATCHING_LABEL,
                error_from_numerator(self.field_bits, separated as u128),
            ));
        }

        // A relation that fails somewhere on the cube survives only at a root of its extension.
        if self.zerocheck_variables != 0 {
            terms.push(SecurityTerm::new(
                WORD_ZEROCHECK_POINT_LABEL,
                error_from_numerator(self.field_bits, self.zerocheck_variables as u128),
            ));
            terms.push(SecurityTerm::new(
                WORD_ZEROCHECK_ROUNDS_LABEL,
                error_from_numerator(
                    self.field_bits,
                    self.zerocheck_degree as u128 * self.zerocheck_variables as u128,
                ),
            ));
        }

        // The shift reduction publishes its own separately labelled experiments.
        terms.extend(self.shift.components());

        // The commitment prices the ring switch and its own opening from its own schedule.
        terms.extend(self.commitment.iter().copied());

        terms
    }

    /// Returns the union bound over every experiment the proof runs.
    #[must_use]
    pub fn combined_term(&self) -> SecurityTerm {
        // Compose the independently labelled failure events by probability addition.
        let errors = self
            .components()
            .iter()
            .map(|component| component.bits)
            .collect::<Vec<_>>();
        SecurityTerm::new(WORD_PROOF_LABEL, ErrorBits::sum(&errors))
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
    use alloc::vec;

    use super::*;
    use crate::binary::BinaryPcsRegime;
    use crate::multilinear::bit_ring_switch_term;

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
    fn the_complete_proof_charges_every_stage_once() {
        // Fixture state:
        //
        // - two relation families, so one separating coefficient;
        // - nine vanishing variables bound at per-variable degree three;
        // - four shift batching variables and twenty-six quadratic rounds;
        // - a commitment charging a ring switch beside its own single opening.
        let shift = WordShiftSecurityModel::new(128, 4, 26).unwrap();
        let pcs = BinaryPcsRegime::new(128, 6, 2, 1, 40, 0).unwrap();
        let commitment = vec![bit_ring_switch_term(1, 7, 6, 128), pcs.opening_term(1)];
        let model = WordProofSecurityModel::new(128, 2, 9, 3, shift, commitment.clone()).unwrap();
        let components = model.components();

        // One coefficient separates two families, so its numerator is one.
        assert_eq!(components[0].label, WORD_RELATION_BATCHING_LABEL);
        assert_eq!(components[0].bits.bits(), 128.0);

        // Nine variables give the vanishing point the numerator nine.
        assert_eq!(components[1].label, WORD_ZEROCHECK_POINT_LABEL);
        assert_eq!(components[1].bits.bits(), 128.0 - libm::log2(9.0));

        // Nine cubic rounds give the numerator twenty-seven.
        assert_eq!(components[2].label, WORD_ZEROCHECK_ROUNDS_LABEL);
        assert_eq!(components[2].bits.bits(), 128.0 - libm::log2(27.0));

        // Four shift batching variables give the numerator four.
        assert_eq!(components[3].label, WORD_SHIFT_BATCHING_LABEL);
        assert_eq!(components[3].bits.bits(), 126.0);

        // Twenty-six quadratic shift rounds give the numerator fifty-two.
        assert_eq!(components[4].label, WORD_SHIFT_SUMCHECK_LABEL);
        assert_eq!(components[4].bits.bits(), 128.0 - libm::log2(52.0));

        // The commitment's own terms close the list, in the order it supplied them.
        assert_eq!(components[5..], commitment);
        assert_eq!(components.len(), 7);

        // The composable term is the union of every event.
        let errors = components
            .iter()
            .map(|component| component.bits)
            .collect::<Vec<_>>();
        assert_eq!(model.combined_term().bits, ErrorBits::sum(&errors));
    }

    #[test]
    fn a_proof_model_rejects_a_shape_it_cannot_charge_honestly() {
        let shift = WordShiftSecurityModel::new(128, 4, 26).unwrap();

        // A challenge field must expose at least one bit of entropy.
        assert!(WordProofSecurityModel::new(0, 2, 9, 3, shift, Vec::new()).is_none());

        // A degree-zero composition carries no round polynomial to separate against.
        assert!(WordProofSecurityModel::new(128, 2, 9, 0, shift, Vec::new()).is_none());
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
