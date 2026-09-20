//! The reusable key a statement is proved and verified against.

use alloc::vec;
use alloc::vec::Vec;
use core::num::NonZeroUsize;

use p3_field::Field;
use p3_multilinear_util::point::Point;
use p3_security::SecurityTerm;
use p3_security::binary::BinaryPcsRegime;
use p3_security::word::WordProofSecurityModel;
use p3_word::{ConstraintSystem, Word};

use super::error::WordProofError;
use super::record::OPERAND_EVALUATIONS;
use super::relation::{BATCHED_FAMILIES, ZEROCHECK_DEGREE};
use super::transcript::TranscriptShape;
use crate::{KeyCompileError, Packed, PackedWitness, PackedWord, ShiftClaim, ShiftReductionKey};

/// A checked word system compiled into a complete proving key.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WordProofKey<W: Word> {
    /// Reduction from shifted operand claims to one committed-trace claim.
    pub(super) shift: ShiftReductionKey<W>,
}

impl<W: Word> WordProofKey<W> {
    /// Compiles a checked relation system into a reusable proving key.
    ///
    /// # Errors
    ///
    /// Returns an error when the system is too large for the compact key representation.
    pub fn new(system: ConstraintSystem<W>) -> Result<Self, KeyCompileError> {
        // Key compilation fixes every sparse reference before proving begins.
        Ok(Self {
            shift: ShiftReductionKey::new(system)?,
        })
    }

    /// Returns the checked relation system represented by this key.
    #[inline]
    pub const fn system(&self) -> &ConstraintSystem<W> {
        // Owning the system stops a compiled layout being paired with another statement.
        self.shift.system()
    }

    /// Returns the number of variables the padded committed bit trace spans.
    ///
    /// The commitment must cover exactly that many variables.
    #[must_use]
    pub fn trace_variables(&self) -> usize {
        // Word coordinates lead and within-word coordinates trail.
        self.shift.word_variables() + Self::bit_variables()
    }

    /// Returns the exact algebraic soundness term for the complete proof.
    ///
    /// The width argument is a lower bound on the base-two logarithm of the challenge-field order.
    ///
    /// Every challenge of every sub-reduction must be drawn from that same extension field.
    ///
    /// The absorbed argument is the base-two logarithm of the bits one committed element holds.
    #[must_use]
    pub fn security_term(
        &self,
        field_bits: NonZeroUsize,
        absorbed_log: usize,
        pcs: &BinaryPcsRegime,
    ) -> Option<SecurityTerm> {
        Some(
            self.security_model(field_bits, absorbed_log, pcs)?
                .combined_term(),
        )
    }

    /// Returns separately labelled terms for security-report diagnostics.
    ///
    /// The terms cover batching, vanishing, shift, ring-switch, and commitment experiments.
    #[must_use]
    pub fn security_components(
        &self,
        field_bits: NonZeroUsize,
        absorbed_log: usize,
        pcs: &BinaryPcsRegime,
    ) -> Option<Vec<SecurityTerm>> {
        Some(
            self.security_model(field_bits, absorbed_log, pcs)?
                .components(),
        )
    }

    /// Builds the numeric soundness model from the executed schedule.
    fn security_model(
        &self,
        field_bits: NonZeroUsize,
        absorbed_log: usize,
        pcs: &BinaryPcsRegime,
    ) -> Option<WordProofSecurityModel> {
        // Every count is derived from the owned system rather than supplied by a caller.
        WordProofSecurityModel::new(
            field_bits.get(),
            BATCHED_FAMILIES,
            self.zerocheck_variables(),
            ZEROCHECK_DEGREE,
            self.shift.security_model(field_bits),
            absorbed_log,
            self.trace_variables(),
            *pcs,
        )
    }

    /// Number of within-word variables of the selected word width.
    pub(super) const fn bit_variables() -> usize {
        // Both supported widths are powers of two.
        W::BITS.ilog2() as usize
    }

    /// Number of variables the batched vanishing check binds.
    pub(super) const fn zerocheck_variables(&self) -> usize {
        // Constraint rows are the high coordinates and within-word bits the low ones.
        self.shift.constraint_variables() + Self::bit_variables()
    }

    /// Dimensions bound into the relation transcript before either challenge.
    pub(super) fn transcript_shape(&self) -> TranscriptShape {
        let system = self.system();
        TranscriptShape::new(
            self.shift.constraint_variables(),
            Self::bit_variables(),
            self.trace_variables(),
            system.public_len(),
            system.witness_len(),
            [
                system.zero_constraints().len(),
                system.and_constraints().len(),
                system.integer_mul_constraints().len(),
            ],
        )
    }

    /// Rejects a statement this protocol does not prove end to end.
    pub(super) fn validate_statement<E>(&self) -> Result<(), WordProofError<E>> {
        // Absorbing an unproved family would let it pass under a protocol that never reads it.
        let count = self.system().integer_mul_constraints().len();
        if count != 0 {
            return Err(WordProofError::UnprovedRelation { count });
        }
        Ok(())
    }

    /// Rejects a commitment that covers a different hypercube from the padded trace.
    pub(super) fn validate_arity<E>(&self, num_variables: usize) -> Result<(), WordProofError<E>> {
        let expected = self.trace_variables();
        if num_variables != expected {
            return Err(WordProofError::TraceShape {
                expected,
                actual: num_variables,
            });
        }
        Ok(())
    }

    /// Builds the shifted-operand claim the reduction consumes.
    pub(super) fn operand_claim<EF: Field>(
        &self,
        point: &Point<EF>,
        operands: &[EF; OPERAND_EVALUATIONS],
    ) -> ShiftClaim<EF> {
        // Constraint coordinates lead the point and within-word coordinates trail it.
        let (constraint_point, bit_point) = point.split_at(self.shift.constraint_variables());
        let [linear, left, right, output] = *operands;
        ShiftClaim::new(
            constraint_point.as_slice().to_vec(),
            bit_point.as_slice().to_vec(),
            [linear],
            [left, right, output],
            [EF::ZERO; 4],
        )
    }
}

impl<W: PackedWord> WordProofKey<W> {
    /// Lays the committed words out as the padded bit trace the commitment reads.
    pub(super) fn trace_bits(&self, values: &PackedWitness<W>) -> Vec<Packed<W>> {
        // Words above the declared segment are zero, so they move no opening.
        let mut trace = vec![W::pack(W::ZERO); 1 << self.shift.word_variables()];
        trace[..values.witness().len()].copy_from_slice(values.witness());
        trace
    }
}

#[cfg(test)]
mod tests {
    use p3_security::binary::BinaryPcsRegime;
    use p3_word::{
        IntegerMulConstraint, Operand, ShiftedValue, ValueIndex, Word64, ZeroConstraint,
    };

    use super::*;

    // Base-two logarithm of the bits one committed element holds.
    const ABSORBED: usize = 7;

    // A statement wide enough that the padded trace covers more than one element.
    fn key() -> WordProofKey<Word64> {
        let value = ValueIndex::witness(0).expect("test position fits");
        let linear = ZeroConstraint::new(Operand::single(ShiftedValue::plain(value)));
        let system = ConstraintSystem::new(0, 8, vec![linear], vec![], vec![])
            .expect("the fixture addresses only declared words");
        WordProofKey::new(system).expect("the fixture compiles")
    }

    #[test]
    fn the_padded_trace_spans_the_word_and_bit_coordinates() {
        // Eight committed words need three word coordinates beside six within-word ones.
        assert_eq!(key().trace_variables(), 9);

        // One relation fits the single padded row, so only the bit axis is bound.
        assert_eq!(key().zerocheck_variables(), 6);
    }

    #[test]
    fn an_unproved_relation_family_is_refused_before_anything_is_absorbed() {
        let operand = |position| {
            Operand::single(ShiftedValue::plain(
                ValueIndex::witness(position).expect("test position fits"),
            ))
        };
        let product = IntegerMulConstraint::new(operand(0), operand(1), operand(2), operand(3));
        let system = ConstraintSystem::<Word64>::new(0, 8, vec![], vec![], vec![product])
            .expect("the fixture addresses only declared words");
        let key = WordProofKey::new(system).expect("the fixture compiles");

        assert_eq!(
            key.validate_statement::<()>(),
            Err(WordProofError::UnprovedRelation { count: 1 })
        );
    }

    #[test]
    fn a_commitment_over_another_hypercube_is_refused_before_anything_is_absorbed() {
        // The padded trace spans nine variables, so any other width is a mismatch.
        assert_eq!(key().validate_arity::<()>(9), Ok(()));
        assert_eq!(
            key().validate_arity::<()>(10),
            Err(WordProofError::TraceShape {
                expected: 9,
                actual: 10,
            })
        );
        assert_eq!(
            key().validate_arity::<()>(8),
            Err(WordProofError::TraceShape {
                expected: 9,
                actual: 8,
            })
        );
    }

    #[test]
    fn the_security_model_charges_every_reduction_stage() {
        let key = key();
        let regime = BinaryPcsRegime::new(128, key.trace_variables() - ABSORBED, 2, 1, 40, 0)
            .expect("the fixture schedule is valid");
        let field_bits = NonZeroUsize::new(128).expect("a nonzero width");
        let components = key
            .security_components(field_bits, ABSORBED, &regime)
            .expect("the fixture shape is chargeable");

        // Batching, both vanishing draws, both shift draws, ring switch, and commitment.
        let labels = components
            .iter()
            .map(|component| component.label)
            .collect::<Vec<_>>();
        assert_eq!(
            labels,
            [
                p3_security::word::WORD_RELATION_BATCHING_LABEL,
                p3_security::word::WORD_ZEROCHECK_POINT_LABEL,
                p3_security::word::WORD_ZEROCHECK_ROUNDS_LABEL,
                p3_security::word::WORD_SHIFT_BATCHING_LABEL,
                p3_security::word::WORD_SHIFT_SUMCHECK_LABEL,
                p3_security::BIT_RING_SWITCH_LABEL,
                p3_security::binary::BINARY_PCS_OPENING_LABEL,
            ]
        );

        // One separating coefficient, and four shift batching variables.
        assert_eq!(components[0].bits.bits(), 128.0);
        assert_eq!(components[3].bits.bits(), 126.0);

        // The union of every event is no stronger than its weakest component.
        let combined = key
            .security_term(field_bits, ABSORBED, &regime)
            .expect("the fixture shape is chargeable");
        assert!(
            components
                .iter()
                .all(|component| combined.bits.bits() <= component.bits.bits())
        );
    }
}
