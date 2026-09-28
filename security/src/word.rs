//! Soundness terms for the word-level relation proof.

use alloc::vec::Vec;

use crate::{CandidateSet, ErrorBits, SecurityTerm};

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

/// Label for the complete full-width unsigned multiplication reduction error.
pub const WORD_INTEGER_MUL_LABEL: &str = "word-integer-mul";

/// Label for the row point the two exponent lifts are compared at.
pub const WORD_INTEGER_MUL_POINT_LABEL: &str = "word-integer-mul-point";

/// Label for the layer, line, and leaf checks of both product trees.
pub const WORD_INTEGER_MUL_PRODUCT_LABEL: &str = "word-integer-mul-product-check";

/// Per-variable degree of every sumcheck the multiplication reduction runs.
const INTEGER_MUL_DEGREE: u128 = 3;

/// Checked dimensions of one full-width unsigned multiplication reduction.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WordIntegerMulSecurityModel {
    /// Lower bound on the base-two logarithm of the challenge-field order.
    field_bits: usize,
    /// Variables selecting one padded multiplication row.
    row_variables: usize,
    /// Variables selecting one bit within a word.
    bit_variables: usize,
}

impl WordIntegerMulSecurityModel {
    /// Creates a model when the challenge field has a nonzero security width.
    #[must_use]
    pub const fn new(
        field_bits: usize,
        row_variables: usize,
        bit_variables: usize,
    ) -> Option<Self> {
        // A zero-bit challenge field cannot support a statistical reduction.
        if field_bits == 0 {
            return None;
        }

        Some(Self {
            field_bits,
            row_variables,
            bit_variables,
        })
    }

    /// Returns the error numerator of one product tree of the given depth.
    ///
    /// ```text
    ///     layer d      cubic sumcheck over m + d variables, then one line draw
    ///     leaf check   cubic sumcheck over m + depth variables
    ///
    ///     numerator = sum_{d < depth} (3 * (m + d) + 1) + 3 * (m + depth)
    /// ```
    const fn tree_numerator(&self, depth: usize) -> u128 {
        let rows = self.row_variables as u128;
        let depth = depth as u128;

        // The layer sum of 3 * (m + d) is 3 * (m * depth + depth * (depth - 1) / 2).
        let layers = INTEGER_MUL_DEGREE * (rows * depth + depth * depth.saturating_sub(1) / 2);
        layers + depth + INTEGER_MUL_DEGREE * (rows + depth)
    }

    /// Returns each algebraic error source for diagnostic reporting.
    #[must_use]
    pub fn components(self) -> Vec<SecurityTerm> {
        let mut terms = Vec::with_capacity(2);

        // Two lifts differing on some row agree at a random row point only at a root.
        if self.row_variables != 0 {
            terms.push(SecurityTerm::new(
                WORD_INTEGER_MUL_POINT_LABEL,
                error_from_numerator(self.field_bits, self.row_variables as u128),
            ));
        }

        // The factor tree spans two bit indices and the result tree one index over twice the width.
        let numerator = self.tree_numerator(2 * self.bit_variables)
            + self.tree_numerator(self.bit_variables + 1);
        terms.push(SecurityTerm::new(
            WORD_INTEGER_MUL_PRODUCT_LABEL,
            error_from_numerator(self.field_bits, numerator),
        ));

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
        SecurityTerm::new(WORD_INTEGER_MUL_LABEL, ErrorBits::sum(&errors))
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
    /// Relation terms separated by the powers of one batching coefficient.
    relation_families: usize,
    /// Constraint and within-word variables the vanishing check binds.
    zerocheck_variables: usize,
    /// Per-variable degree of the batched relation polynomial.
    zerocheck_degree: usize,
    /// Dimensions of the multiplication reduction that precedes the vanishing check, if any.
    integer_mul: Option<WordIntegerMulSecurityModel>,
    /// Dimensions of the shift reduction that follows.
    shift: WordShiftSecurityModel,
    /// Labelled errors the commitment charges for discharging the one surviving claim.
    commitment: Vec<SecurityTerm>,
    /// The candidates the commitment still leaves open while this proof draws.
    ///
    /// Every draw below lands after the commitment and before the opening names one.
    ///
    /// Each is charged over this set, and the set itself comes through untouched.
    candidates: CandidateSet,
}

impl WordProofSecurityModel {
    /// Creates a model when every sampled experiment has a well-formed shape.
    ///
    /// The width argument is a lower bound on the base-two logarithm of the challenge-field order.
    ///
    /// Every batching, vanishing, and reduction challenge must be drawn from that same field.
    ///
    /// The last two arguments both come from the commitment.
    ///
    /// They are what it charges, and how many candidates it still leaves open.
    ///
    /// A statement without multiplication relations passes no multiplication model.
    #[must_use]
    #[allow(
        clippy::too_many_arguments,
        reason = "each argument is one independently derived stage of the schedule"
    )]
    pub fn new(
        field_bits: usize,
        relation_families: usize,
        zerocheck_variables: usize,
        zerocheck_degree: usize,
        integer_mul: Option<WordIntegerMulSecurityModel>,
        shift: WordShiftSecurityModel,
        commitment: Vec<SecurityTerm>,
        log2_candidates: f64,
    ) -> Option<Self> {
        // A degree-zero composition carries no round polynomial to separate against.
        if field_bits == 0 || zerocheck_degree == 0 {
            return None;
        }

        // A count that is not a real size prices nothing, so no number is reported at all.
        let candidates = CandidateSet::from_log2(log2_candidates)?;

        Some(Self {
            field_bits,
            relation_families,
            zerocheck_variables,
            zerocheck_degree,
            integer_mul,
            shift,
            commitment,
            candidates,
        })
    }

    /// Returns each algebraic error source for diagnostic reporting.
    #[must_use]
    pub fn components(&self) -> Vec<SecurityTerm> {
        let mut terms = Vec::new();

        // The multiplication reduction runs first and publishes its own labelled experiments.
        if let Some(integer_mul) = self.integer_mul {
            terms.extend(integer_mul.components());
        }

        // Powers of one coefficient separate the terms, so its degree is one below their count.
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

        // Every draw above lands after the commitment and before the opening names one.
        //
        // A prover may therefore choose which candidate it is after seeing them.
        for term in &mut terms {
            *term = term.over_candidates(self.candidates);
        }

        // The commitment prices the ring switch and its own opening from its own schedule.
        //
        // Those terms arrive already charged by the layer that drew them.
        //
        // So this proof carries them through rather than charging the same set twice.
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
        // ```text
        //     batching point   2 operation bits + 2 operand bits
        //     bit sumcheck     6 quadratic rounds
        //     word sumcheck   20 quadratic rounds
        // ```
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
        let model = WordProofSecurityModel::new(128, 2, 9, 3, None, shift, commitment.clone(), 0.0)
            .unwrap();
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
        assert!(WordProofSecurityModel::new(0, 2, 9, 3, None, shift, Vec::new(), 0.0).is_none());

        // A degree-zero composition carries no round polynomial to separate against.
        assert!(WordProofSecurityModel::new(128, 2, 9, 0, None, shift, Vec::new(), 0.0).is_none());

        // A candidate count that names no real set size prices nothing.
        for count in [f64::NAN, f64::INFINITY, -1.0] {
            assert!(
                WordProofSecurityModel::new(128, 2, 9, 3, None, shift, Vec::new(), count).is_none()
            );
        }
    }

    #[test]
    fn candidates_left_open_are_charged_to_every_draw_that_precedes_them() {
        // Fixture state: the same shape, priced against a commitment leaving 2^5 candidates.
        let shift = WordShiftSecurityModel::new(128, 4, 26).unwrap();
        let pcs = BinaryPcsRegime::new(128, 6, 2, 1, 40, 0).unwrap();
        let commitment = vec![bit_ring_switch_term(1, 7, 6, 128), pcs.opening_term(1)];
        let settled =
            WordProofSecurityModel::new(128, 2, 9, 3, None, shift, commitment.clone(), 0.0)
                .unwrap();
        let open = WordProofSecurityModel::new(128, 2, 9, 3, None, shift, commitment.clone(), 5.0)
            .unwrap();

        // Each of the proof's own five draws loses exactly the candidate bound.
        let before = settled.components();
        let after = open.components();
        for (settled, open) in before.iter().zip(&after).take(5) {
            assert_eq!(open.label, settled.label);
            assert_eq!(open.bits.bits(), settled.bits.bits() - 5.0);
        }

        // The commitment already charged its own reductions, so its terms do not move.
        assert_eq!(after[5..], commitment);
        assert_eq!(before[5..], commitment);
    }

    #[test]
    fn a_multiplication_reduction_charges_its_point_and_both_trees() {
        // Fixture state: eight padded rows of 64-bit products.
        //
        //     factor tree   depth 12, layer d over 3 + d variables, leaf over 15
        //     result tree   depth 7,  layer d over 3 + d variables, leaf over 10
        let model = WordIntegerMulSecurityModel::new(128, 3, 6).unwrap();
        let components = model.components();

        // Three row variables give the comparison point the numerator three.
        assert_eq!(components[0].label, WORD_INTEGER_MUL_POINT_LABEL);
        assert_eq!(components[0].bits.bits(), 128.0 - libm::log2(3.0));

        // Factor tree: 3 * (36 + 66) + 12 + 45 = 363.
        // Result tree: 3 * (21 + 21) + 7 + 30 = 163.
        assert_eq!(components[1].label, WORD_INTEGER_MUL_PRODUCT_LABEL);
        assert_eq!(components[1].bits.bits(), 128.0 - libm::log2(526.0));

        // A single row needs no comparison point, but both trees still run.
        let single = WordIntegerMulSecurityModel::new(128, 0, 6).unwrap();
        let components = single.components();
        assert_eq!(components.len(), 1);
        assert_eq!(components[0].label, WORD_INTEGER_MUL_PRODUCT_LABEL);

        // A challenge field must expose at least one bit of entropy.
        assert_eq!(WordIntegerMulSecurityModel::new(0, 3, 6), None);
    }

    #[test]
    fn the_multiplication_terms_precede_the_vanishing_check() {
        // Fixture state: the complete proof above, now also proving 64-bit products.
        let shift = WordShiftSecurityModel::new(128, 4, 26).unwrap();
        let integer_mul = WordIntegerMulSecurityModel::new(128, 3, 6).unwrap();
        let model =
            WordProofSecurityModel::new(128, 7, 9, 3, Some(integer_mul), shift, Vec::new(), 2.0)
                .unwrap();
        let components = model.components();

        // The transcript runs the products first, so their terms lead the list.
        let labels = components.iter().map(|term| term.label).collect::<Vec<_>>();
        assert_eq!(
            labels,
            [
                WORD_INTEGER_MUL_POINT_LABEL,
                WORD_INTEGER_MUL_PRODUCT_LABEL,
                WORD_RELATION_BATCHING_LABEL,
                WORD_ZEROCHECK_POINT_LABEL,
                WORD_ZEROCHECK_ROUNDS_LABEL,
                WORD_SHIFT_BATCHING_LABEL,
                WORD_SHIFT_SUMCHECK_LABEL,
            ]
        );

        // Seven batched terms under powers of one coefficient give the numerator six.
        assert_eq!(components[2].bits.bits(), 128.0 - libm::log2(6.0) - 2.0);

        // The products are drawn before the opening too, so they pay the candidate bound.
        let expected = integer_mul.components();
        assert_eq!(components[0].bits.bits(), expected[0].bits.bits() - 2.0);
        assert_eq!(components[1].bits.bits(), expected[1].bits.bits() - 2.0);
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
