//! The reusable key a statement is proved and verified against.

use alloc::vec;
use alloc::vec::Vec;
use core::num::NonZeroUsize;

use p3_field::Field;
use p3_multilinear_util::point::Point;
use p3_security::SecurityTerm;
use p3_security::word::WordProofSecurityModel;
use p3_sumcheck::PrescribedOpeningSecurity;
use p3_word::{ConstraintSystem, Word};

use super::error::WordProofError;
use super::relation::{BATCHED_FAMILIES, OPERAND_EVALUATIONS, ZEROCHECK_DEGREE};
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
    ///
    /// Returns an error when the system declares a relation family this protocol cannot prove.
    pub fn new(system: ConstraintSystem<W>) -> Result<Self, KeyCompileError> {
        // A key that could never be proved must not exist, so nothing can price one either.
        let count = system.integer_mul_constraints().len();
        if count != 0 {
            return Err(KeyCompileError::UnprovedRelation { count });
        }

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
    /// The commitment must cover at least that many variables.
    ///
    /// A wider commitment is read at its leading coordinates fixed to zero.
    #[must_use]
    pub fn trace_variables(&self) -> usize {
        // Word coordinates lead and within-word coordinates trail.
        self.shift.word_variables() + Self::bit_variables()
    }

    /// Returns the exact algebraic soundness term for the complete proof.
    ///
    /// The challenge field is the one every sub-reduction draws from.
    ///
    /// The argument is what the commitment charges for one opening, priced by itself.
    ///
    /// # Returns
    ///
    /// Nothing when the commitment names a candidate count that is not a real set size.
    #[must_use]
    pub fn security_term<EF: Field>(
        &self,
        commitment: &PrescribedOpeningSecurity,
    ) -> Option<SecurityTerm> {
        Some(self.security_model::<EF>(commitment)?.combined_term())
    }

    /// Returns separately labelled terms for security-report diagnostics.
    ///
    /// The terms cover batching, vanishing, shift, and whatever the commitment charges.
    ///
    /// # Returns
    ///
    /// Nothing when the commitment names a candidate count that is not a real set size.
    #[must_use]
    pub fn security_components<EF: Field>(
        &self,
        commitment: &PrescribedOpeningSecurity,
    ) -> Option<Vec<SecurityTerm>> {
        Some(self.security_model::<EF>(commitment)?.components())
    }

    /// Builds the numeric soundness model from the executed schedule.
    fn security_model<EF: Field>(
        &self,
        commitment: &PrescribedOpeningSecurity,
    ) -> Option<WordProofSecurityModel> {
        // The counts are the key's own and both commitment inputs are the commitment's own.
        let field_bits = NonZeroUsize::new(EF::bits()).expect("a field has at least one element");
        WordProofSecurityModel::new(
            field_bits.get(),
            BATCHED_FAMILIES,
            self.zerocheck_variables(),
            ZEROCHECK_DEGREE,
            self.shift.security_model(field_bits),
            commitment.terms.clone(),
            commitment.log2_max_candidates,
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
    pub(super) fn transcript_shape(&self, commitment_variables: usize) -> TranscriptShape {
        let system = self.system();
        TranscriptShape::new(
            self.shift.constraint_variables(),
            Self::bit_variables(),
            commitment_variables,
            system.public_len(),
            system.witness_len(),
            [
                system.zero_constraints().len(),
                system.and_constraints().len(),
                system.integer_mul_constraints().len(),
            ],
        )
    }

    /// Rejects a commitment too narrow to hold the padded trace.
    pub(super) fn validate_arity<E>(&self, num_variables: usize) -> Result<(), WordProofError<E>> {
        let expected = self.trace_variables();
        if num_variables < expected {
            return Err(WordProofError::TraceShape {
                expected,
                actual: num_variables,
            });
        }
        Ok(())
    }

    /// Lifts a reduced trace point to the commitment's own hypercube.
    pub(super) fn commitment_point<EF: Field>(
        &self,
        point: &[EF],
        commitment_variables: usize,
    ) -> Point<EF> {
        // Leading coordinates address the padding words, which are zero.
        //
        // Fixing them therefore reads only the declared words.
        let padding = commitment_variables - self.trace_variables();
        Point::new(
            vec![EF::ZERO; padding]
                .into_iter()
                .chain(point.iter().copied())
                .collect(),
        )
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
    pub(super) fn trace_bits(
        &self,
        values: &PackedWitness<W>,
        commitment_variables: usize,
    ) -> Vec<Packed<W>> {
        // Words above the declared segment are zero, so they move no opening.
        let words = commitment_variables - Self::bit_variables();
        let mut trace = vec![W::pack(W::ZERO); 1 << words];
        trace[..values.witness().len()].copy_from_slice(values.witness());
        trace
    }
}

#[cfg(test)]
mod tests {
    use p3_binary_field::BinaryField128;
    use p3_binary_pcs::{BinaryPcsConfig, BinaryPcsParams, BooleanPcs};
    use p3_field::PrimeCharacteristicRing;
    use p3_keccak::Keccak256Hash;
    use p3_merkle_tree::MerkleTreeMmcs;
    use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher};
    use p3_word::{
        IntegerMulConstraint, Operand, ShiftedValue, ValueIndex, Word64, ZeroConstraint,
    };

    use super::*;

    type EF = BinaryField128;
    type MyHash = SerializingHasher<Keccak256Hash>;
    type MyCompress = CompressionFunctionFromHasher<Keccak256Hash, 2, 32>;
    type MyMmcs = MerkleTreeMmcs<EF, u8, MyHash, MyCompress, 2, 32>;
    type Scheme = BooleanPcs<EF, MyMmcs, MyMmcs>;

    // Base-two logarithm of the bits one committed element holds.
    const ABSORBED: usize = 7;

    // Draws this proof makes itself, before the opening names one candidate.
    const OWN_DRAWS: usize = 5;

    // A candidate count a proximity argument in the list-decoding regime would report.
    const CANDIDATES: f64 = 5.321_928_094_887_363;

    // A statement wide enough that the padded trace covers more than one element.
    fn key() -> WordProofKey<Word64> {
        let value = ValueIndex::witness(0).expect("test position fits");
        let linear = ZeroConstraint::new(Operand::single(ShiftedValue::plain(value)));
        let system = ConstraintSystem::new(0, 8, vec![linear], vec![], vec![])
            .expect("the fixture addresses only declared words");
        WordProofKey::new(system).expect("the fixture compiles")
    }

    fn commitment_scheme(trace_variables: usize) -> Scheme {
        let mmcs = MyMmcs::new(
            MyHash::new(Keccak256Hash),
            MyCompress::new(Keccak256Hash),
            0,
        );
        let params = BinaryPcsParams {
            log_inv_rate: 2,
            pow_bits: 0,
            security_level: 40,
        };
        let config =
            BinaryPcsConfig::try_new_with_folding::<EF, EF>(trace_variables - ABSORBED, params, 1)
                .expect("the fixture arity supports one folding round");
        Scheme::new(config, mmcs.clone(), mmcs, trace_variables)
            .expect("the fixture arity is valid")
    }

    #[test]
    fn the_padded_trace_spans_the_word_and_bit_coordinates() {
        // Eight committed words need three word coordinates beside six within-word ones.
        assert_eq!(key().trace_variables(), 9);

        // One relation fits the single padded row, so only the bit axis is bound.
        assert_eq!(key().zerocheck_variables(), 6);
    }

    #[test]
    fn an_unproved_relation_family_leaves_no_key_to_price() {
        let operand = |position| {
            Operand::single(ShiftedValue::plain(
                ValueIndex::witness(position).expect("test position fits"),
            ))
        };
        let product = IntegerMulConstraint::new(operand(0), operand(1), operand(2), operand(3));
        let system = ConstraintSystem::<Word64>::new(0, 8, vec![], vec![], vec![product])
            .expect("the fixture addresses only declared words");

        assert_eq!(
            WordProofKey::new(system),
            Err(KeyCompileError::UnprovedRelation { count: 1 })
        );
    }

    #[test]
    fn a_commitment_too_narrow_for_the_trace_is_refused_before_anything_is_absorbed() {
        // The padded trace spans nine variables, so anything narrower is a mismatch.
        assert_eq!(key().validate_arity::<()>(9), Ok(()));
        assert_eq!(key().validate_arity::<()>(10), Ok(()));
        assert_eq!(
            key().validate_arity::<()>(8),
            Err(WordProofError::TraceShape {
                expected: 9,
                actual: 8,
            })
        );
    }

    #[test]
    fn a_wider_commitment_is_read_at_its_leading_coordinates_fixed_to_zero() {
        // Fixture state: a nine-variable trace inside an eleven-variable commitment.
        let point = [EF::ONE; 9];
        let lifted = key().commitment_point(&point, 11);

        assert_eq!(lifted.num_variables(), 11);
        assert_eq!(lifted.as_slice()[..2], [EF::ZERO; 2]);
        assert_eq!(lifted.as_slice()[2..], point);
    }

    #[test]
    fn a_commitment_leaving_candidates_open_charges_every_draw_before_it() {
        // Fixture state: the same commitment, once settled and once leaving 2^5.32 open.
        //
        // That count is what a Johnson-regime proximity argument reports at rate one quarter.
        let key = key();
        let scheme = commitment_scheme(key.trace_variables());
        let settled = scheme.opening_security(1);
        assert_eq!(settled.log2_max_candidates, 0.0);
        let open = PrescribedOpeningSecurity {
            terms: settled.terms.clone(),
            log2_max_candidates: CANDIDATES,
        };

        // Each draw this proof makes between the commitment and the opening loses that much.
        let before = key.security_components::<EF>(&settled).unwrap();
        let after = key.security_components::<EF>(&open).unwrap();
        for (before, after) in before.iter().zip(&after).take(OWN_DRAWS) {
            assert_eq!(after.label, before.label);
            assert_eq!(after.bits.bits(), before.bits.bits() - CANDIDATES);
        }

        // The commitment charged its own reductions already, so its terms are carried intact.
        assert_eq!(after[OWN_DRAWS..], settled.terms);

        // A union bound over the whole set costs at most the set itself.
        let settled_level = key.security_term::<EF>(&settled).unwrap().bits.bits();
        let open_level = key.security_term::<EF>(&open).unwrap().bits.bits();
        assert!(open_level < settled_level);
        assert!(settled_level - open_level <= CANDIDATES);
    }

    #[test]
    fn a_candidate_count_that_names_no_set_is_refused() {
        let key = key();
        let scheme = commitment_scheme(key.trace_variables());
        let terms = scheme.opening_security(1).terms;

        // Nothing is reported rather than a number resting on an unusable count.
        for count in [f64::NAN, f64::INFINITY, -1.0] {
            let unusable = PrescribedOpeningSecurity {
                terms: terms.clone(),
                log2_max_candidates: count,
            };
            assert_eq!(key.security_term::<EF>(&unusable), None);
            assert_eq!(key.security_components::<EF>(&unusable), None);
        }
    }

    #[test]
    fn the_security_model_charges_every_reduction_stage() {
        let key = key();
        let scheme = commitment_scheme(key.trace_variables());
        let commitment = scheme.opening_security(1);
        let components = key.security_components::<EF>(&commitment).unwrap();

        // Batching, both vanishing draws, both shift draws, then whatever the commitment charges.
        let labels = components
            .iter()
            .map(|component| component.label)
            .collect::<Vec<_>>();
        assert_eq!(
            labels[..5],
            [
                p3_security::word::WORD_RELATION_BATCHING_LABEL,
                p3_security::word::WORD_ZEROCHECK_POINT_LABEL,
                p3_security::word::WORD_ZEROCHECK_ROUNDS_LABEL,
                p3_security::word::WORD_SHIFT_BATCHING_LABEL,
                p3_security::word::WORD_SHIFT_SUMCHECK_LABEL,
            ]
        );

        // One separating coefficient, and four shift batching variables.
        assert_eq!(components[0].bits.bits(), 128.0);
        assert_eq!(components[3].bits.bits(), 126.0);

        // The commitment's own terms are carried through untouched.
        assert_eq!(components[5..], commitment.terms);

        // The union of every event is no stronger than its weakest component.
        let combined = key.security_term::<EF>(&commitment).unwrap();
        assert!(
            components
                .iter()
                .all(|component| combined.bits.bits() <= component.bits.bits())
        );
    }
}
