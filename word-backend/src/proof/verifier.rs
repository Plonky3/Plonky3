//! Verifying side of the ordered sub-reductions.
//!
//! Every step replays the proving order exactly, slot for slot.

use p3_binary_field::Gf2;
use p3_binary_pcs::BooleanMultilinearPcs;
use p3_challenger::fs::TranscriptField;
use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{Algebra, ExtensionField};
use p3_multilinear_util::point::Point;
use p3_word::{Segment, Word};

use super::error::WordProofError;
use super::key::WordProofKey;
use super::record::WordProof;
use super::relation::{ZEROCHECK_DEGREE, closing_value};
use super::transcript::ProofVerifierTranscript;

impl<W: Word> WordProofKey<W> {
    /// Verifies every relation against the committed trace.
    ///
    /// # Errors
    ///
    /// Returns an error when any of the following holds.
    ///
    /// - The statement declares a relation family this protocol does not prove.
    /// - The statement or the commitment has the wrong shape.
    /// - The sampled batching coefficient vanishes.
    /// - A reduction or the final opening does not close.
    pub fn verify<F, EF, Pcs, Challenger>(
        &self,
        pcs: &Pcs,
        commitment: &Pcs::Commitment,
        public: &[W],
        proof: &WordProof<F, EF, Pcs::Proof>,
        challenger: &mut Challenger,
    ) -> Result<(), WordProofError<Pcs::Error>>
    where
        F: TranscriptField,
        EF: ExtensionField<F> + Algebra<Gf2>,
        Pcs: BooleanMultilinearPcs<EF, Challenger>,
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        // Statement dimensions are checked before any transcript replay.
        self.validate_statement()?;
        self.validate_arity(pcs.num_variables())?;
        if public.len() != self.system().public_len() {
            return Err(WordProofError::SegmentLength {
                segment: Segment::Public,
                expected: self.system().public_len(),
                actual: public.len(),
            });
        }

        // The verifier reaches the prover's sponge state from the same commitment.
        pcs.observe_commitment(commitment, challenger);
        let variables = self.zerocheck_variables();
        let mut transcript =
            ProofVerifierTranscript::<_, F, EF>::new(challenger, self.transcript_shape(), public);
        let (vanishing_point, batching) = transcript.challenges(variables);
        if batching.is_zero() {
            return Err(WordProofError::DegenerateBatching);
        }

        // A relation set that fails anywhere on the cube cannot sum to zero here.
        if !proof.zerocheck.claimed_sum.is_zero() {
            transcript.abort();
            return Err(WordProofError::RelationClaim);
        }
        let replay = transcript.zerocheck(|challenger| {
            proof
                .zerocheck
                .verify(challenger, variables, ZEROCHECK_DEGREE, 0)
        });
        let (point, final_claim) = match replay {
            Ok(result) => result,
            Err(error) => {
                transcript.abort();
                return Err(error.into());
            }
        };
        transcript.finish();

        // The equality factor is public and is evaluated from the sampled point directly.
        let equality = Point::eval_eq(&vanishing_point, point.as_slice());
        if final_claim != closing_value(equality, &proof.operands, batching) {
            return Err(WordProofError::RelationClaim);
        }

        // Each operand claim is tied to the committed words by the shift reduction.
        let opening = self.shift.verify::<F, EF, _>(
            public,
            &self.operand_claim(&point, &proof.operands),
            &proof.shift,
            challenger,
        )?;

        // One authenticated opening closes the whole protocol.
        pcs.verify_at_points(
            commitment,
            &[Point::new(opening.point().to_vec())],
            &[opening.value()],
            &proof.opening,
            challenger,
        )
        .map_err(WordProofError::Commitment)
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use p3_binary_field::{BinaryChallenger, BinaryField128};
    use p3_binary_pcs::{BinaryPcsConfig, BinaryPcsParams, BooleanMultilinearPcs, BooleanPcs};
    use p3_challenger::HashChallenger;
    use p3_field::PrimeCharacteristicRing;
    use p3_keccak::Keccak256Hash;
    use p3_merkle_tree::MerkleTreeMmcs;
    use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher};
    use p3_word::{
        AndConstraint, ConstraintSystem, IntegerMulConstraint, Operand, Shift, ShiftKind,
        ShiftedValue, ValueIndex, Word32, Word64, ZeroConstraint,
    };

    use super::*;
    use crate::proof::record::OPERAND_EVALUATIONS;
    use crate::proof::relation::ZEROCHECK_DEGREE;
    use crate::proof::transcript::ProofVerifierTranscript;
    use crate::shift::transcript::equality_weights;
    use crate::{PackedWitness, PackedWord, ShiftReductionError, WordProofKey};

    type EF = BinaryField128;
    type MyHash = SerializingHasher<Keccak256Hash>;
    type MyCompress = CompressionFunctionFromHasher<Keccak256Hash, 2, 32>;
    type MyMmcs = MerkleTreeMmcs<EF, u8, MyHash, MyCompress, 2, 32>;
    type Scheme = BooleanPcs<EF, MyMmcs, MyMmcs>;
    type Challenger = BinaryChallenger<EF, HashChallenger<u8, Keccak256Hash, 32>>;
    type SchemeProof = <Scheme as BooleanMultilinearPcs<EF, Challenger>>::Proof;
    type SchemeError = <Scheme as BooleanMultilinearPcs<EF, Challenger>>::Error;
    type Commitment = <Scheme as BooleanMultilinearPcs<EF, Challenger>>::Commitment;
    type Record = WordProof<EF, EF, SchemeProof>;

    /// Base-two logarithm of the bits one committed element holds.
    const ABSORBED: usize = 7;

    fn challenger() -> Challenger {
        // Prover and verifier begin from the same empty transcript.
        Challenger::from_hasher(Vec::new(), Keccak256Hash)
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

    fn public(position: usize) -> ValueIndex {
        ValueIndex::public(position).expect("test position fits")
    }

    fn committed(position: usize) -> ValueIndex {
        ValueIndex::witness(position).expect("test position fits")
    }

    fn shifted<W: Word>(index: ValueIndex, kind: ShiftKind, amount: usize) -> ShiftedValue<W> {
        ShiftedValue::single(
            index,
            Shift::new(kind, amount).expect("test shift is in range"),
        )
    }

    fn run<W: PackedWord>(
        key: &WordProofKey<W>,
        scheme: &Scheme,
        values: &PackedWitness<W>,
    ) -> (Commitment, Record) {
        key.prove::<EF, EF, _, _>(scheme, values, &mut challenger())
            .expect("a transcript is produced whether or not the statement holds")
    }

    fn check<W: PackedWord>(
        key: &WordProofKey<W>,
        scheme: &Scheme,
        commitment: &Commitment,
        public_words: &[W],
        proof: &Record,
    ) -> Result<(), WordProofError<SchemeError>> {
        key.verify::<EF, EF, _, _>(scheme, commitment, public_words, proof, &mut challenger())
    }

    /// A 64-bit statement exercising both proved families and four shift spellings.
    fn word64_statement(seed: u64) -> (ConstraintSystem<Word64>, [Word64; 2], [Word64; 8]) {
        // A rotation and a public word pin the first committed word.
        let linear = ZeroConstraint::new(Operand::new(vec![
            ShiftedValue::plain(committed(0)),
            shifted(committed(1), ShiftKind::RotateRight, 13),
            ShiftedValue::plain(public(0)),
        ]));

        // A two-term left input meets a plain right input.
        let first_and = AndConstraint::new(
            Operand::new(vec![
                ShiftedValue::plain(committed(2)),
                shifted(committed(3), ShiftKind::LogicalLeft, 5),
            ]),
            Operand::single(ShiftedValue::plain(committed(4))),
            Operand::single(ShiftedValue::plain(committed(5))),
        );

        // A sign-extending lane shift meets the public mask.
        let second_and = AndConstraint::new(
            Operand::single(shifted(committed(6), ShiftKind::Lane32ArithmeticRight, 7)),
            Operand::single(ShiftedValue::plain(public(1))),
            Operand::single(ShiftedValue::plain(committed(7))),
        );

        let system = ConstraintSystem::new(2, 8, vec![linear], vec![first_and, second_and], vec![])
            .expect("the fixture addresses only declared words");

        // Free words are spread apart so no two relations share a value.
        let free = |step: u64| seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(step);
        let public_words = [Word64::new(free(1)), Word64::new(free(2))];
        let (w1, w2, w3, w4, w6) = (free(3), free(4), free(5), free(6), free(7));

        // Native integer operations derive every dependent word independently.
        let w0 = w1.rotate_right(13) ^ public_words[0].get();
        let w5 = (w2 ^ (w3 << 5)) & w4;
        let low = (((w6 as u32) as i32) >> 7) as u32;
        let high = (((w6 >> 32) as u32 as i32) >> 7) as u32;
        let w7 = (u64::from(low) | (u64::from(high) << 32)) & public_words[1].get();

        let words = [w0, w1, w2, w3, w4, w5, w6, w7].map(Word64::new);
        (system, public_words, words)
    }

    /// A 32-bit statement exercising both proved families and three shift spellings.
    fn word32_statement(seed: u32) -> (ConstraintSystem<Word32>, [Word32; 1], [Word32; 16]) {
        // An arithmetic shift and a public word pin the first committed word.
        let linear = ZeroConstraint::new(Operand::new(vec![
            ShiftedValue::plain(committed(0)),
            shifted(committed(1), ShiftKind::ArithmeticRight, 9),
            ShiftedValue::plain(public(0)),
        ]));

        // A repeated term cancels, so only the rotation survives the left input.
        let repeated = ShiftedValue::plain(committed(2));
        let first_and = AndConstraint::new(
            Operand::new(vec![
                repeated,
                repeated,
                shifted(committed(3), ShiftKind::RotateRight, 11),
            ]),
            Operand::single(ShiftedValue::plain(committed(4))),
            Operand::single(ShiftedValue::plain(committed(5))),
        );

        // A logical right shift meets a plain committed word.
        let second_and = AndConstraint::new(
            Operand::single(shifted(committed(6), ShiftKind::LogicalRight, 3)),
            Operand::single(ShiftedValue::plain(committed(7))),
            Operand::single(ShiftedValue::plain(committed(8))),
        );

        let system =
            ConstraintSystem::new(1, 16, vec![linear], vec![first_and, second_and], vec![])
                .expect("the fixture addresses only declared words");

        let free = |step: u32| seed.wrapping_mul(0x9E37_79B9).wrapping_add(step);
        let public_words = [Word32::new(free(1))];
        let (w1, w3, w4, w6, w7) = (free(2), free(4), free(5), free(6), free(7));

        // Native integer operations derive every dependent word independently.
        let w0 = (((w1 as i32) >> 9) as u32) ^ public_words[0].get();
        let w5 = w3.rotate_right(11) & w4;
        let w8 = (w6 >> 3) & w7;

        // Unused committed words carry distinct padding so a mis-addressed term shows up.
        let mut words = core::array::from_fn::<u32, 16, _>(|slot| free(0x100 + slot as u32));
        words[0] = w0;
        words[1] = w1;
        words[3] = w3;
        words[4] = w4;
        words[5] = w5;
        words[6] = w6;
        words[7] = w7;
        words[8] = w8;
        (system, public_words, words.map(Word32::new))
    }

    #[test]
    fn a_word64_statement_proves_and_verifies() {
        // The scalar reference is the independently derived witness, checked before proving.
        let (system, public_words, words) = word64_statement(0xA1B2_C3D4_E5F6_0718);
        assert_eq!(system.verify(&public_words, &words), Ok(()));

        let key = WordProofKey::new(system.clone()).unwrap();
        let scheme = commitment_scheme(key.trace_variables());
        let values = PackedWitness::new(&system, &public_words, &words).unwrap();
        let (commitment, proof) = run(&key, &scheme, &values);

        check(&key, &scheme, &commitment, &public_words, &proof).unwrap();
    }

    #[test]
    fn a_word32_statement_proves_and_verifies() {
        // The scalar reference is the independently derived witness, checked before proving.
        let (system, public_words, words) = word32_statement(0x1357_9BDF);
        assert_eq!(system.verify(&public_words, &words), Ok(()));

        let key = WordProofKey::new(system.clone()).unwrap();
        let scheme = commitment_scheme(key.trace_variables());
        let values = PackedWitness::new(&system, &public_words, &words).unwrap();
        let (commitment, proof) = run(&key, &scheme, &values);

        check(&key, &scheme, &commitment, &public_words, &proof).unwrap();
    }

    #[test]
    fn the_operand_claims_match_an_independent_evaluation() {
        // The four evaluations must be the operand columns at the point the check ended on.
        let (system, public_words, words) = word64_statement(0x0BAD_C0DE_0BAD_C0DE);
        let key = WordProofKey::new(system.clone()).unwrap();
        let scheme = commitment_scheme(key.trace_variables());
        let values = PackedWitness::new(&system, &public_words, &words).unwrap();
        let (_, proof) = run(&key, &scheme, &values);

        // Replay the transcript far enough to recover the point the prover ended on.
        let mut replay = challenger();
        let _ = scheme
            .commit_bits(&key.trace_bits(&values), &mut replay)
            .unwrap();
        let variables = key.zerocheck_variables();
        let mut transcript = ProofVerifierTranscript::<_, EF, EF>::new(
            &mut replay,
            key.transcript_shape(),
            &public_words,
        );
        let _ = transcript.challenges(variables);
        let (point, _) = transcript
            .zerocheck(|challenger| {
                proof
                    .zerocheck
                    .verify(challenger, variables, ZEROCHECK_DEGREE, 0)
            })
            .unwrap();
        transcript.finish();

        // Evaluate each operand column directly from the words the statement declares.
        let (constraint_point, bit_point) = point.split_at(key.shift.constraint_variables());
        let rows = equality_weights(constraint_point.as_slice());
        let bits = equality_weights(bit_point.as_slice());
        let evaluate = |operand: &Operand<Word64>, row: usize| {
            let word = operand.evaluate(&public_words, &words).unwrap().get();
            rows[row]
                * (0..64)
                    .filter(|bit| (word >> bit) & 1 == 1)
                    .map(|bit| bits[bit])
                    .sum::<EF>()
        };
        let linear: EF = system
            .zero_constraints()
            .iter()
            .enumerate()
            .map(|(row, relation)| evaluate(relation.value(), row))
            .sum();
        let mut bitwise = [EF::ZERO; 3];
        for (row, relation) in system.and_constraints().iter().enumerate() {
            bitwise[0] += evaluate(relation.left(), row);
            bitwise[1] += evaluate(relation.right(), row);
            bitwise[2] += evaluate(relation.output(), row);
        }

        assert_eq!(proof.operands, [linear, bitwise[0], bitwise[1], bitwise[2]]);
    }

    #[test]
    fn a_broken_linear_relation_is_rejected() {
        // Mutation: flip one bit of the word the vanishing relation pins.
        let (system, public_words, mut words) = word64_statement(0x0F1E_2D3C_4B5A_6978);
        words[0] = Word64::new(words[0].get() ^ 1);
        assert!(system.verify(&public_words, &words).is_err());

        let key = WordProofKey::new(system.clone()).unwrap();
        let scheme = commitment_scheme(key.trace_variables());
        let values = PackedWitness::new(&system, &public_words, &words).unwrap();
        let (commitment, proof) = run(&key, &scheme, &values);

        assert!(check(&key, &scheme, &commitment, &public_words, &proof).is_err());
    }

    #[test]
    fn a_broken_bitwise_relation_is_rejected() {
        // Mutation: flip the highest bit of the second bitwise output.
        let (system, public_words, mut words) = word64_statement(0x2222_3333_4444_5555);
        words[7] = Word64::new(words[7].get() ^ (1 << 63));
        assert!(system.verify(&public_words, &words).is_err());

        let key = WordProofKey::new(system.clone()).unwrap();
        let scheme = commitment_scheme(key.trace_variables());
        let values = PackedWitness::new(&system, &public_words, &words).unwrap();
        let (commitment, proof) = run(&key, &scheme, &values);

        assert!(check(&key, &scheme, &commitment, &public_words, &proof).is_err());
    }

    #[test]
    fn a_word32_bitwise_output_is_pinned_bit_by_bit() {
        // Mutation: single-bit changes at the ends and the middle of a bitwise output.
        let (system, public_words, words) = word32_statement(0x2468_ACE0);
        let key = WordProofKey::new(system.clone()).unwrap();
        let scheme = commitment_scheme(key.trace_variables());

        for bit in [0, 7, 16, 31] {
            let mut broken = words;
            broken[5] = Word32::new(broken[5].get() ^ (1 << bit));
            let values = PackedWitness::new(&system, &public_words, &broken).unwrap();
            let (commitment, proof) = run(&key, &scheme, &values);
            assert!(check(&key, &scheme, &commitment, &public_words, &proof).is_err());
        }
    }

    #[test]
    fn a_word32_linear_relation_is_pinned_bit_by_bit() {
        // Mutation: single-bit changes to the word the vanishing relation pins.
        let (system, public_words, words) = word32_statement(0x0F0F_1E1E);
        let key = WordProofKey::new(system.clone()).unwrap();
        let scheme = commitment_scheme(key.trace_variables());

        for bit in [0, 15, 31] {
            let mut broken = words;
            broken[0] = Word32::new(broken[0].get() ^ (1 << bit));
            let values = PackedWitness::new(&system, &public_words, &broken).unwrap();
            let (commitment, proof) = run(&key, &scheme, &values);
            assert!(check(&key, &scheme, &commitment, &public_words, &proof).is_err());
        }
    }

    #[test]
    fn every_operand_evaluation_is_load_bearing() {
        // Mutation: perturb each of the four evaluations the proof carries.
        let (system, public_words, words) = word64_statement(0x5555_6666_7777_8888);
        let key = WordProofKey::new(system.clone()).unwrap();
        let scheme = commitment_scheme(key.trace_variables());
        let values = PackedWitness::new(&system, &public_words, &words).unwrap();
        let (commitment, proof) = run(&key, &scheme, &values);

        for slot in 0..OPERAND_EVALUATIONS {
            let mut tampered = proof.clone();
            tampered.operands[slot] += EF::ONE;
            assert!(check(&key, &scheme, &commitment, &public_words, &tampered).is_err());
        }
    }

    #[test]
    fn every_vanishing_round_is_load_bearing() {
        // Mutation: perturb one transmitted evaluation of every round in turn.
        let (system, public_words, words) = word64_statement(0x9999_AAAA_BBBB_CCCC);
        let key = WordProofKey::new(system.clone()).unwrap();
        let scheme = commitment_scheme(key.trace_variables());
        let values = PackedWitness::new(&system, &public_words, &words).unwrap();
        let (commitment, proof) = run(&key, &scheme, &values);

        for round in 0..proof.zerocheck.round_polys.len() {
            let mut tampered = proof.clone();
            tampered.zerocheck.round_polys[round][0] += EF::ONE;
            assert!(check(&key, &scheme, &commitment, &public_words, &tampered).is_err());
        }
    }

    #[test]
    fn a_nonzero_claimed_sum_is_rejected() {
        // Mutation: claim a nonzero sum for a check whose whole point is that it vanishes.
        let (system, public_words, words) = word64_statement(0xDDDD_EEEE_FFFF_0000);
        let key = WordProofKey::new(system.clone()).unwrap();
        let scheme = commitment_scheme(key.trace_variables());
        let values = PackedWitness::new(&system, &public_words, &words).unwrap();
        let (commitment, proof) = run(&key, &scheme, &values);

        let mut tampered = proof;
        tampered.zerocheck.claimed_sum += EF::ONE;
        assert!(matches!(
            check(&key, &scheme, &commitment, &public_words, &tampered),
            Err(WordProofError::RelationClaim)
        ));
    }

    #[test]
    fn swapping_two_bitwise_inputs_is_rejected() {
        // Mutation: exchange the left and right claims, which leaves their product unmoved.
        //
        // The vanishing check therefore still closes, and only the wiring separates them.
        let (system, public_words, words) = word64_statement(0x4A4A_5B5B_6C6C_7D7D);
        let key = WordProofKey::new(system.clone()).unwrap();
        let scheme = commitment_scheme(key.trace_variables());
        let values = PackedWitness::new(&system, &public_words, &words).unwrap();
        let (commitment, proof) = run(&key, &scheme, &values);

        let mut tampered = proof.clone();
        tampered.operands.swap(1, 2);
        assert_ne!(tampered.operands, proof.operands);
        assert!(matches!(
            check(&key, &scheme, &commitment, &public_words, &tampered),
            Err(WordProofError::Shift(
                ShiftReductionError::IntermediateClaim
            ))
        ));
    }

    #[test]
    fn a_substituted_public_word_is_rejected() {
        // Mutation: verify against a public word the prover never bound.
        let (system, public_words, words) = word64_statement(0x1111_2222_3333_4444);
        let key = WordProofKey::new(system.clone()).unwrap();
        let scheme = commitment_scheme(key.trace_variables());
        let values = PackedWitness::new(&system, &public_words, &words).unwrap();
        let (commitment, proof) = run(&key, &scheme, &values);

        let substituted = [Word64::new(public_words[0].get() ^ 1), public_words[1]];
        assert!(check(&key, &scheme, &commitment, &substituted, &proof).is_err());
    }

    #[test]
    fn a_commitment_to_another_witness_is_rejected() {
        // Mutation: keep the proof and swap in the commitment of a different trace.
        let (system, public_words, words) = word64_statement(0x6161_6262_6363_6464);
        let key = WordProofKey::new(system.clone()).unwrap();
        let scheme = commitment_scheme(key.trace_variables());
        let values = PackedWitness::new(&system, &public_words, &words).unwrap();
        let (_, proof) = run(&key, &scheme, &values);

        let (other_system, other_public, other_words) = word64_statement(0x7171_7272_7373_7474);
        let other = PackedWitness::new(&other_system, &other_public, &other_words).unwrap();
        let (other_commitment, _) = run(&key, &scheme, &other);

        assert!(check(&key, &scheme, &other_commitment, &public_words, &proof).is_err());
    }

    #[test]
    fn an_unproved_relation_family_is_refused() {
        // An unsigned product has no reduction here, so the statement must be refused.
        let operand = |position| Operand::single(ShiftedValue::plain(committed(position)));
        let product = IntegerMulConstraint::new(operand(0), operand(1), operand(2), operand(3));
        let system = ConstraintSystem::<Word64>::new(0, 8, vec![], vec![], vec![product]).unwrap();
        let key = WordProofKey::new(system.clone()).unwrap();
        let scheme = commitment_scheme(key.trace_variables());
        let words = [Word64::new(0); 8];
        let values = PackedWitness::new(&system, &[], &words).unwrap();

        assert!(matches!(
            key.prove::<EF, EF, _, _>(&scheme, &values, &mut challenger()),
            Err(WordProofError::UnprovedRelation { count: 1 })
        ));

        // A proof of another statement is refused before the transcript is replayed.
        let (sound, public_words, sound_words) = word64_statement(0x0123_4567_89AB_CDEF);
        let sound_key = WordProofKey::new(sound.clone()).unwrap();
        let sound_scheme = commitment_scheme(sound_key.trace_variables());
        let sound_values = PackedWitness::new(&sound, &public_words, &sound_words).unwrap();
        let (commitment, proof) = run(&sound_key, &sound_scheme, &sound_values);
        assert!(matches!(
            check(&key, &scheme, &commitment, &[], &proof),
            Err(WordProofError::UnprovedRelation { count: 1 })
        ));
    }

    #[test]
    fn a_commitment_over_another_hypercube_is_refused() {
        // The padded trace and the commitment must span exactly the same variables.
        let (system, public_words, words) = word64_statement(0x3141_5926_5358_9793);
        let key = WordProofKey::new(system.clone()).unwrap();
        let wider = commitment_scheme(key.trace_variables() + 1);
        let values = PackedWitness::new(&system, &public_words, &words).unwrap();

        let expected = key.trace_variables();
        assert!(matches!(
            key.prove::<EF, EF, _, _>(&wider, &values, &mut challenger()),
            Err(WordProofError::TraceShape { expected: reported, actual })
                if reported == expected && actual == expected + 1
        ));

        // The verifier reads the same dimension, so it refuses the same mismatch.
        let narrow = commitment_scheme(expected);
        let (commitment, proof) = run(&key, &narrow, &values);
        assert!(matches!(
            check(&key, &wider, &commitment, &public_words, &proof),
            Err(WordProofError::TraceShape { expected: reported, actual })
                if reported == expected && actual == expected + 1
        ));
    }
}
