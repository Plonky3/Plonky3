//! Sumcheck reduction from shifted operands to one committed-word opening.
//!
//! The reduction proves one identity in two stages:
//!
//! ```text
//! batched operand claim - public share
//!     = sum_(sequence, bit) sparse_witness(sequence, bit) * shift(sequence, bit)
//!     = sum_word committed_word(word, r_bit) * wiring(word, r_bit).
//! ```
//!
//! The first sumcheck binds the 32-bit or 64-bit coordinate.
//! The second binds the committed word index.
//! Source words remain bit-packed until the first point has been sampled.
//!
//! Shift spellings are verifier-fixed metadata, so they are summed exactly instead of opened as
//! prover polynomials. Each two-slot operator is evaluated as `inner^T(outer^T(output))`, which
//! preserves composition order without adding sampled shift-code variables.
//!
//! A false operand claim either survives as a false final opening or hits a batching or sumcheck
//! collision. Before the PCS opening, that algebraic error is
//! `(4 + 2 * (log2(word_bits) + log2(padded_words))) / |EF|`.

mod claim;
mod error;
mod polynomial;
mod transcript;
mod wiring;

use alloc::vec::Vec;
use core::num::NonZeroUsize;

use p3_binary_field::Gf2;
use p3_challenger::fs::TranscriptField;
use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{Algebra, ExtensionField};
use p3_security::SecurityTerm;
use p3_security::word::WordShiftSecurityModel;
use p3_sumcheck::generic_degree::{GenericDegreeProof, RoundProver};
use p3_word::{ConstraintSystem, Segment, Word};
use serde::{Deserialize, Serialize};

use crate::{CompiledKeyLayout, KeyCompileError, PackedWitness, PackedWord};

pub use claim::{ShiftClaim, ShiftOpeningClaim};
pub use error::ShiftReductionError;
use transcript::{ShiftProverTranscript, ShiftVerifierTranscript, TranscriptShape};
use wiring::{
    PreparedWeights, bit_prover, public_contribution, shift_evaluations, wiring_evaluation,
    word_prover,
};

/// Transcript record for the two quadratic sumchecks.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShiftReductionProof<F, EF> {
    /// Rounds binding the within-word bit coordinate.
    bit_sumcheck: GenericDegreeProof<F, EF>,
    /// Rounds binding the committed word coordinate.
    word_sumcheck: GenericDegreeProof<F, EF>,
    /// Evaluation claimed for the committed bit trace.
    witness_evaluation: EF,
}

/// A checked word system together with its compiled shift wiring.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShiftReductionKey<W: Word> {
    /// Constraint system defining the statement.
    system: ConstraintSystem<W>,
    /// Sparse word-to-relation metadata used by both proving phases.
    layout: CompiledKeyLayout<W>,
    /// Variables spanning the widest padded relation family.
    constraint_variables: usize,
}

impl<W: Word> ShiftReductionKey<W> {
    /// Compiles a checked constraint system into a reusable reduction key.
    pub fn new(system: ConstraintSystem<W>) -> Result<Self, KeyCompileError> {
        // Key compilation fixes every sparse reference before proving begins.
        let layout = CompiledKeyLayout::new(&system)?;
        let constraint_count = system
            .zero_constraints()
            .len()
            .max(system.and_constraints().len())
            .max(system.integer_mul_constraints().len())
            .max(1);
        let constraint_variables = constraint_count.next_power_of_two().ilog2() as usize;
        Ok(Self {
            system,
            layout,
            constraint_variables,
        })
    }

    /// Returns the checked relation system represented by this key.
    #[inline]
    pub const fn system(&self) -> &ConstraintSystem<W> {
        // Owning the system prevents a compiled layout from being paired with another statement.
        &self.system
    }

    /// Returns the exact algebraic soundness term for this reduction.
    #[must_use]
    pub fn security_term(&self, field_bits: NonZeroUsize) -> SecurityTerm {
        // Four batching variables precede both quadratic sumchecks.
        self.security_model(field_bits).combined_term()
    }

    /// Returns separately labelled terms for security-report diagnostics.
    #[must_use]
    pub fn security_components(&self, field_bits: NonZeroUsize) -> Vec<SecurityTerm> {
        // Components must be unioned before composition with another protocol.
        self.security_model(field_bits).components()
    }

    /// Proves that all supplied operand claims read the committed word trace.
    ///
    /// The caller must bind the trace commitment before invoking this reduction.
    /// The returned evaluation remains unauthenticated until the Boolean PCS opens it.
    ///
    /// # Errors
    ///
    /// Returns an error when a statement point or witness segment has the wrong shape.
    pub fn prove<F, EF, Challenger>(
        &self,
        values: &PackedWitness<W>,
        claim: &ShiftClaim<EF>,
        challenger: &mut Challenger,
    ) -> Result<(ShiftReductionProof<F, EF>, ShiftOpeningClaim<EF>), ShiftReductionError>
    where
        W: PackedWord,
        F: TranscriptField,
        EF: ExtensionField<F> + Algebra<Gf2>,
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        // Reject shape mismatches before the transcript absorbs any statement value.
        self.validate_claim(claim)?;
        values
            .check_shape(&self.system)
            .map_err(|error| ShiftReductionError::SegmentLength {
                segment: error.segment,
                expected: error.expected,
                actual: error.actual,
            })?;

        // Batch relation families first and operand positions second.
        let public_words = values
            .public()
            .iter()
            .copied()
            .map(W::unpack)
            .collect::<Vec<_>>();
        let shape = self.transcript_shape();
        let mut transcript =
            ShiftProverTranscript::<_, F, EF>::new(challenger, shape, claim, &public_words);
        let batch = transcript.batching();
        let prepared = PreparedWeights::new(claim, batch);
        let public = public_contribution(
            self.layout.public(),
            &public_words,
            &prepared,
            claim.bit_point(),
        );
        let witness_claim = prepared.claim(claim) - public;

        // Phase 1 binds the within-word bit without expanding the committed words.
        let bit_rounds = W::BITS.ilog2() as usize;
        let mut bit = bit_prover(&self.layout, values, &prepared, claim.bit_point());
        let (bit_sumcheck, bit_point) = transcript.bit_sumcheck(|challenger| {
            bit.prove::<F, _>(challenger, bit_rounds, 2, 0, witness_claim)
        });
        let intermediate_claim = bit.terminal_sum();
        let mut shifts = bit.right_constants();
        shifts.truncate(self.layout.witness().shift_sequences().len());
        // The word phase can dominate memory, so release both fixed-width tables first.
        drop(bit);

        // Phase 2 binds the committed word index and exposes one trace opening.
        let word_rounds = self.word_variables();
        let mut word = word_prover(
            &self.layout,
            values,
            &prepared,
            bit_point.as_slice(),
            &shifts,
        );
        let (word_sumcheck, word_point) = transcript.word_sumcheck(|challenger| {
            word.prove::<F, _>(challenger, word_rounds, 2, 0, intermediate_claim)
        });
        let [witness_evaluation, wiring_evaluation] = word.first_constants();
        debug_assert_eq!(word.terminal_sum(), witness_evaluation * wiring_evaluation);

        // Boolean PCS layout places the within-word coordinates after the word coordinates.
        let opening = ShiftOpeningClaim::new(
            [word_point.as_slice(), bit_point.as_slice()].concat(),
            witness_evaluation,
        );
        transcript.finish();
        Ok((
            ShiftReductionProof {
                bit_sumcheck,
                word_sumcheck,
                witness_evaluation,
            },
            opening,
        ))
    }

    /// Verifies a shift reduction and returns its unauthenticated trace opening.
    ///
    /// The caller must discharge the result through the Boolean PCS.
    ///
    /// # Errors
    ///
    /// Returns an error for malformed proofs, inconsistent rounds, or a failed closing equation.
    pub fn verify<F, EF, Challenger>(
        &self,
        public: &[W],
        claim: &ShiftClaim<EF>,
        proof: &ShiftReductionProof<F, EF>,
        challenger: &mut Challenger,
    ) -> Result<ShiftOpeningClaim<EF>, ShiftReductionError>
    where
        F: TranscriptField,
        EF: ExtensionField<F> + Algebra<Gf2>,
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        // Statement dimensions are checked before any transcript replay.
        self.validate_claim(claim)?;
        if public.len() != self.system.public_len() {
            return Err(ShiftReductionError::SegmentLength {
                segment: Segment::Public,
                expected: self.system.public_len(),
                actual: public.len(),
            });
        }

        // Reconstruct the exact batched claim the prover started from.
        let shape = self.transcript_shape();
        let mut transcript =
            ShiftVerifierTranscript::<_, F, EF>::new(challenger, shape, claim, public);
        let batch = transcript.batching();
        let prepared = PreparedWeights::new(claim, batch);
        let public =
            public_contribution(self.layout.public(), public, &prepared, claim.bit_point());
        let expected_claim = prepared.claim(claim) - public;
        if proof.bit_sumcheck.claimed_sum != expected_claim {
            transcript.abort();
            return Err(ShiftReductionError::IntermediateClaim);
        }

        // The two sumchecks meet at the evaluation left by the bit-index rounds.
        let bit_result = transcript.bit_sumcheck(|challenger| {
            proof
                .bit_sumcheck
                .verify(challenger, W::BITS.ilog2() as usize, 2, 0)
        });
        let (bit_point, intermediate_claim) = match bit_result {
            Ok(result) => result,
            Err(error) => {
                transcript.abort();
                return Err(error.into());
            }
        };
        if proof.word_sumcheck.claimed_sum != intermediate_claim {
            transcript.abort();
            return Err(ShiftReductionError::IntermediateClaim);
        }
        let word_result = transcript.word_sumcheck(|challenger| {
            proof
                .word_sumcheck
                .verify(challenger, self.word_variables(), 2, 0)
        });
        let (word_point, final_claim) = match word_result {
            Ok(result) => result,
            Err(error) => {
                transcript.abort();
                return Err(error.into());
            }
        };

        // The wiring factor is public and is evaluated directly from the compiled relation graph.
        let shifts = shift_evaluations(&self.layout, claim.bit_point(), bit_point.as_slice());
        let wiring = wiring_evaluation(&self.layout, &prepared, &shifts, word_point.as_slice());
        transcript.finish();
        if final_claim != proof.witness_evaluation * wiring {
            return Err(ShiftReductionError::FinalClaim);
        }

        // The caller authenticates this value against the committed Boolean trace.
        Ok(ShiftOpeningClaim::new(
            [word_point.as_slice(), bit_point.as_slice()].concat(),
            proof.witness_evaluation,
        ))
    }

    /// Checks the two public point dimensions.
    fn validate_claim<F>(&self, claim: &ShiftClaim<F>) -> Result<(), ShiftReductionError> {
        // The constraint axis spans the widest zero-padded family.
        if claim.constraint_point().len() != self.constraint_variables {
            return Err(ShiftReductionError::PointLength {
                axis: "constraint",
                expected: self.constraint_variables,
                actual: claim.constraint_point().len(),
            });
        }

        // The bit axis spans every bit of one selected word width.
        let bit_variables = W::BITS.ilog2() as usize;
        if claim.bit_point().len() != bit_variables {
            return Err(ShiftReductionError::PointLength {
                axis: "bit",
                expected: bit_variables,
                actual: claim.bit_point().len(),
            });
        }
        Ok(())
    }

    /// Returns dimensions bound into the batching transcript.
    const fn transcript_shape(&self) -> TranscriptShape {
        // Every count is verifier-derived from the owned system.
        TranscriptShape::new(
            self.constraint_variables,
            W::BITS.ilog2() as usize,
            self.system.public_len(),
            self.system.witness_len(),
        )
    }

    /// Returns the number of variables covering the committed word segment.
    fn word_variables(&self) -> usize {
        // An empty segment is represented by one zero padding slot.
        self.system.witness_len().max(1).next_power_of_two().ilog2() as usize
    }

    /// Builds the numeric soundness model from the executed schedule.
    fn security_model(&self, field_bits: NonZeroUsize) -> WordShiftSecurityModel {
        // Two two-variable batching axes and both quadratic round counts are exact.
        WordShiftSecurityModel::new(
            field_bits.get(),
            4,
            W::BITS.ilog2() as usize + self.word_variables(),
        )
        .expect("a nonzero field width gives a valid shift-reduction model")
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_binary_field::{BinaryChallenger, BinaryField64, BinaryField128};
    use p3_challenger::HashChallenger;
    use p3_field::PrimeCharacteristicRing;
    use p3_keccak::Keccak256Hash;
    use p3_word::{
        AndConstraint, IntegerMulConstraint, Operand, Shift, ShiftKind, ShiftedValue, ValueIndex,
        Word32, Word64, ZeroConstraint,
    };
    use proptest::prelude::*;

    use super::*;
    use crate::shift::wiring::evaluate_claim;

    type F = BinaryField128;
    type Challenger = BinaryChallenger<F, HashChallenger<u8, Keccak256Hash, 32>>;
    type BaseChallenger = BinaryChallenger<BinaryField64, HashChallenger<u8, Keccak256Hash, 32>>;

    fn challenger() -> Challenger {
        // Prover and verifier begin from the same empty transcript.
        Challenger::from_hasher(Vec::new(), Keccak256Hash)
    }

    fn base_challenger() -> BaseChallenger {
        // This challenger exercises extension-field claims over a narrower transcript field.
        BaseChallenger::from_hasher(Vec::new(), Keccak256Hash)
    }

    fn fixture(
        public_value: u64,
        witness_values: [u64; 3],
    ) -> (
        ShiftReductionKey<Word64>,
        PackedWitness<Word64>,
        Vec<Word64>,
        ShiftClaim<F>,
    ) {
        // The fixture reaches full-width, lane-local, arithmetic, and composed shifts.
        let public_index = ValueIndex::public(0).unwrap();
        let witness = [
            ValueIndex::witness(0).unwrap(),
            ValueIndex::witness(1).unwrap(),
            ValueIndex::witness(2).unwrap(),
        ];
        let rotate = Shift::new(ShiftKind::RotateRight, 13).unwrap();
        let arithmetic = Shift::new(ShiftKind::ArithmeticRight, 7).unwrap();
        let lane = Shift::new(ShiftKind::Lane32RotateRight, 5).unwrap();
        let inner = Shift::new(ShiftKind::LogicalLeft, 9).unwrap();
        let outer = Shift::new(ShiftKind::RotateRight, 17).unwrap();

        let left = Operand::new(vec![
            ShiftedValue::single(witness[0], rotate),
            ShiftedValue::single(public_index, lane),
        ]);
        let right = Operand::new(vec![
            ShiftedValue::single(witness[1], arithmetic),
            ShiftedValue::pair(witness[2], inner, outer).unwrap(),
        ]);
        let output = Operand::new(vec![
            ShiftedValue::plain(witness[0]),
            ShiftedValue::plain(witness[0]),
            ShiftedValue::plain(witness[2]),
        ]);
        let system = ConstraintSystem::new(
            1,
            3,
            vec![ZeroConstraint::new(Operand::new(vec![]))],
            vec![AndConstraint::new(
                left.clone(),
                right.clone(),
                output.clone(),
            )],
            vec![IntegerMulConstraint::new(left.clone(), right, output, left)],
        )
        .unwrap();
        let key = ShiftReductionKey::new(system).unwrap();
        let public = vec![Word64::new(public_value)];
        let witness = witness_values.map(Word64::new).to_vec();
        let packed = PackedWitness::new(key.system(), &public, &witness).unwrap();
        let constraint_point = vec![];
        let bit_point = (0..6)
            .map(|index| F::from_u64((index + 2) as u64))
            .collect();
        let claim = evaluate_claim(key.system(), &packed, constraint_point, bit_point);
        (key, packed, public, claim)
    }

    #[test]
    fn reduction_round_trips_and_returns_the_direct_trace_evaluation() {
        // Fixture state: one public word and three committed words under five shift spellings.
        let (key, witness, public, claim) = fixture(
            0x0123_4567_89ab_cdef,
            [
                0xfedc_ba98_7654_3210,
                0x8000_0001_7fff_ffff,
                0x0f0f_f0f0_a5a5_5a5a,
            ],
        );
        let mut prover = challenger();
        let (proof, prover_opening) = key
            .prove(&witness, &claim, &mut prover)
            .expect("the honest witness has the checked shape");

        // Verifier replay must recover the identical point and value.
        let mut verifier = challenger();
        let verifier_opening = key
            .verify(&public, &claim, &proof, &mut verifier)
            .expect("the honest reduction closes");
        assert_eq!(prover_opening, verifier_opening);

        // Direct evaluation uses the same word-major, bit-minor Boolean trace layout.
        let (word_point, bit_point) = prover_opening.point().split_at(key.word_variables());
        let word_weights = transcript::equality_weights(word_point);
        let bit_weights = transcript::equality_weights(bit_point);
        let expected = witness
            .witness()
            .iter()
            .enumerate()
            .map(|(word, &packed)| {
                word_weights[word] * wiring::evaluate_word(Word64::unpack(packed), &bit_weights)
            })
            .sum::<F>();
        assert_eq!(prover_opening.value(), expected);
    }

    #[test]
    fn tampering_each_sumcheck_or_opening_is_rejected() {
        // Start from one valid proof so every mutation targets one protocol boundary.
        let (key, witness, public, claim) = fixture(3, [5, 7, 11]);
        let mut prover = challenger();
        let (proof, _) = key.prove(&witness, &claim, &mut prover).unwrap();

        // Mutation 1: the first bit-round message no longer sums to the entering claim.
        let mut bit = proof.clone();
        bit.bit_sumcheck.round_polys[0][0] += F::ONE;
        assert!(
            key.verify(&public, &claim, &bit, &mut challenger())
                .is_err()
        );

        // Mutation 2: the word phase claims a different intermediate value.
        let mut bridge = proof.clone();
        bridge.word_sumcheck.claimed_sum += F::ONE;
        assert_eq!(
            key.verify(&public, &claim, &bridge, &mut challenger()),
            Err(ShiftReductionError::IntermediateClaim)
        );

        // Mutation 3: public words are part of the statement bound before batching.
        let changed_public = [Word64::new(public[0].get() ^ 1)];
        assert!(
            key.verify(&changed_public, &claim, &proof, &mut challenger())
                .is_err()
        );

        // Mutation 4: the final PCS opening no longer closes the word sumcheck.
        let mut opening = proof;
        opening.witness_evaluation += F::ONE;
        assert_eq!(
            key.verify(&public, &claim, &opening, &mut challenger()),
            Err(ShiftReductionError::FinalClaim)
        );
    }

    #[test]
    fn changing_any_public_operand_claim_is_rejected() {
        // Every semantic role is public input and precedes both batching challenges.
        let (key, witness, public, claim) = fixture(29, [31, 37, 41]);
        let mut prover = challenger();
        let (proof, _) = key.prove(&witness, &claim, &mut prover).unwrap();

        for changed in 0..8 {
            let mut zero = *claim.zero();
            let mut bitwise_and = *claim.bitwise_and();
            let mut integer_mul = *claim.integer_mul();
            match changed {
                0 => zero[0] += F::ONE,
                1..=3 => bitwise_and[changed - 1] += F::ONE,
                4..=7 => integer_mul[changed - 4] += F::ONE,
                _ => unreachable!(),
            }
            let changed_claim = ShiftClaim::new(
                claim.constraint_point().to_vec(),
                claim.bit_point().to_vec(),
                zero,
                bitwise_and,
                integer_mul,
            );

            assert!(
                key.verify(&public, &changed_claim, &proof, &mut challenger())
                    .is_err(),
                "claim position {changed} was not bound",
            );
        }
    }

    #[test]
    fn changing_constraint_order_changes_the_verifier_wiring() {
        // Two rows make the constraint coordinate and its compiled reference order observable.
        let first = ValueIndex::witness(0).unwrap();
        let second = ValueIndex::witness(1).unwrap();
        let left = Shift::new(ShiftKind::LogicalLeft, 7).unwrap();
        let rotate = Shift::new(ShiftKind::RotateRight, 11).unwrap();
        let first_relation =
            ZeroConstraint::new(Operand::single(ShiftedValue::single(first, left)));
        let second_relation =
            ZeroConstraint::new(Operand::single(ShiftedValue::single(second, rotate)));
        let system = ConstraintSystem::new(
            0,
            2,
            vec![first_relation.clone(), second_relation.clone()],
            vec![],
            vec![],
        )
        .unwrap();
        let reordered =
            ConstraintSystem::new(0, 2, vec![second_relation, first_relation], vec![], vec![])
                .unwrap();
        let key = ShiftReductionKey::new(system).unwrap();
        let reordered_key = ShiftReductionKey::new(reordered).unwrap();
        let words = [Word64::new(0x0123), Word64::new(0x4567)];
        let witness = PackedWitness::new(key.system(), &[], &words).unwrap();
        let claim = evaluate_claim(
            key.system(),
            &witness,
            vec![F::from_u64(7)],
            vec![F::from_u64(3); 6],
        );
        let mut prover = challenger();
        let (proof, _) = key.prove(&witness, &claim, &mut prover).unwrap();

        // The shape and transcript inputs agree, but the verifier-fixed row wiring does not.
        assert!(
            reordered_key
                .verify(&[], &claim, &proof, &mut challenger())
                .is_err()
        );
    }

    #[test]
    fn challenge_field_can_extend_the_transcript_field() {
        // The reduction must not force commitments and challenges into one field type.
        let (key, witness, public, claim) = fixture(13, [17, 19, 23]);
        let mut prover = base_challenger();
        let (proof, expected) = key.prove(&witness, &claim, &mut prover).unwrap();
        let mut verifier = base_challenger();
        let actual = key.verify(&public, &claim, &proof, &mut verifier).unwrap();

        assert_eq!(actual, expected);
    }

    #[test]
    fn malformed_statement_points_are_rejected_before_replay() {
        // Word32 needs five bit coordinates and this statement has no constraint coordinate.
        let system = ConstraintSystem::<Word32>::new(0, 0, vec![], vec![], vec![]).unwrap();
        let key = ShiftReductionKey::new(system).unwrap();
        let claim = ShiftClaim::new(
            vec![F::ZERO],
            vec![F::ZERO; 4],
            [F::ZERO],
            [F::ZERO; 3],
            [F::ZERO; 4],
        );
        let proof = ShiftReductionProof {
            bit_sumcheck: GenericDegreeProof::default(),
            word_sumcheck: GenericDegreeProof::default(),
            witness_evaluation: F::ZERO,
        };

        // Constraint length is checked first and reports both dimensions.
        assert_eq!(
            key.verify(&[], &claim, &proof, &mut challenger()),
            Err(ShiftReductionError::PointLength {
                axis: "constraint",
                expected: 0,
                actual: 1,
            })
        );
    }

    #[test]
    fn empty_system_reduces_to_a_zero_opening() {
        // An empty committed segment still has one implicit zero padding word.
        let system = ConstraintSystem::<Word32>::new(0, 0, vec![], vec![], vec![]).unwrap();
        let key = ShiftReductionKey::new(system).unwrap();
        let witness = PackedWitness::new(key.system(), &[], &[]).unwrap();
        let claim = ShiftClaim::new(
            vec![],
            vec![F::from_u64(2); 5],
            [F::ZERO],
            [F::ZERO; 3],
            [F::ZERO; 4],
        );
        let mut prover = challenger();
        let (proof, expected) = key.prove(&witness, &claim, &mut prover).unwrap();
        let mut verifier = challenger();
        let actual = key.verify(&[], &claim, &proof, &mut verifier).unwrap();

        assert_eq!(actual, expected);
        assert_eq!(actual.value(), F::ZERO);
    }

    proptest! {
        #[test]
        fn random_words_match_the_scalar_reference(
            public in any::<u64>(),
            witness in prop::array::uniform3(any::<u64>()),
        ) {
            // Random bit patterns exercise sign extension and both 32-bit lanes.
            let (key, values, public_words, claim) = fixture(public, witness);
            let mut prover = challenger();
            let (proof, expected) = key.prove(&values, &claim, &mut prover).unwrap();
            let mut verifier = challenger();
            let actual = key.verify(&public_words, &claim, &proof, &mut verifier).unwrap();

            prop_assert_eq!(actual, expected);
        }
    }
}
