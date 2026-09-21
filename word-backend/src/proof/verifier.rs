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
    /// - The statement has the wrong shape, or the commitment is too narrow for it.
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
        let commitment_variables = pcs.num_variables();
        self.validate_arity(commitment_variables)?;
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
        let mut transcript = ProofVerifierTranscript::<_, F, EF>::new(
            challenger,
            self.transcript_shape(commitment_variables),
            public,
        );
        let (vanishing_point, batching) = transcript
            .challenges(variables)
            .ok_or(WordProofError::DegenerateBatching)?;

        // A relation set that fails anywhere on the cube cannot sum to zero here.
        if !proof.zerocheck.claimed_sum.is_zero() {
            transcript.abort();
            return Err(WordProofError::RelationSum);
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
            &[self.commitment_point(opening.point(), commitment_variables)],
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
    use p3_binary_pcs::{
        BinaryPcsConfig, BinaryPcsParams, BooleanPcs, BooleanPcsError, BooleanProof,
    };
    use p3_challenger::HashChallenger;
    use p3_commit::Mmcs;
    use p3_field::PrimeCharacteristicRing;
    use p3_keccak::Keccak256Hash;
    use p3_merkle_tree::MerkleTreeMmcs;
    use p3_sumcheck::generic_degree::RoundProver;
    use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher};
    use p3_word::{
        AndConstraint, ConstraintKind, ConstraintSystem, Operand, Shift, ShiftKind, ShiftedValue,
        ValueIndex, VerificationError, Word32, Word64, ZeroConstraint,
    };

    use super::*;
    use crate::proof::prover::bit_table;
    use crate::proof::relation::{OPERAND_EVALUATIONS, RelationZerocheck, ZEROCHECK_DEGREE};
    use crate::proof::transcript::{ProofProverTranscript, ProofVerifierTranscript};
    use crate::shift::transcript::equality_weights;
    use crate::{OperationColumns, PackedWitness, PackedWord, ShiftReductionError, WordProofKey};

    type EF = BinaryField128;
    type MyHash = SerializingHasher<Keccak256Hash>;
    type MyCompress = CompressionFunctionFromHasher<Keccak256Hash, 2, 32>;
    type MyMmcs = MerkleTreeMmcs<EF, u8, MyHash, MyCompress, 2, 32>;
    type Scheme = BooleanPcs<EF, MyMmcs, MyMmcs>;
    type Challenger = BinaryChallenger<EF, HashChallenger<u8, Keccak256Hash, 32>>;
    type SchemeProof = BooleanProof<EF, MyMmcs, MyMmcs>;
    type SchemeError = BooleanPcsError<EF, <MyMmcs as Mmcs<EF>>::Error>;
    type Commitment = <MyMmcs as Mmcs<EF>>::Commitment;
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
        let arity = key.trace_variables();
        let _ = scheme
            .commit_bits(&key.trace_bits(&values, arity), &mut replay)
            .unwrap();
        let variables = key.zerocheck_variables();
        let mut transcript = ProofVerifierTranscript::<_, EF, EF>::new(
            &mut replay,
            key.transcript_shape(arity),
            &public_words,
        );
        let _ = transcript.challenges(variables).unwrap();
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
        assert_eq!(
            system.verify(&public_words, &words),
            Err(VerificationError::Unsatisfied {
                kind: ConstraintKind::Zero,
                constraint: 0,
            })
        );

        let key = WordProofKey::new(system.clone()).unwrap();
        let scheme = commitment_scheme(key.trace_variables());
        let values = PackedWitness::new(&system, &public_words, &words).unwrap();
        let (commitment, proof) = run(&key, &scheme, &values);

        let rejection = check(&key, &scheme, &commitment, &public_words, &proof);
        assert!(
            matches!(&rejection, Err(WordProofError::RelationClaim)),
            "the vanishing check must reject, got {rejection:?}"
        );
    }

    #[test]
    fn a_broken_bitwise_relation_is_rejected() {
        // Mutation: flip the highest bit of the second bitwise output.
        let (system, public_words, mut words) = word64_statement(0x2222_3333_4444_5555);
        words[7] = Word64::new(words[7].get() ^ (1 << 63));
        assert_eq!(
            system.verify(&public_words, &words),
            Err(VerificationError::Unsatisfied {
                kind: ConstraintKind::And,
                constraint: 1,
            })
        );

        let key = WordProofKey::new(system.clone()).unwrap();
        let scheme = commitment_scheme(key.trace_variables());
        let values = PackedWitness::new(&system, &public_words, &words).unwrap();
        let (commitment, proof) = run(&key, &scheme, &values);

        let rejection = check(&key, &scheme, &commitment, &public_words, &proof);
        assert!(
            matches!(&rejection, Err(WordProofError::RelationClaim)),
            "the vanishing check must reject, got {rejection:?}"
        );
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
            let rejection = check(&key, &scheme, &commitment, &public_words, &proof);
            assert!(
                matches!(&rejection, Err(WordProofError::RelationClaim)),
                "bit {bit} must be rejected, got {rejection:?}"
            );
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
            let rejection = check(&key, &scheme, &commitment, &public_words, &proof);
            assert!(
                matches!(&rejection, Err(WordProofError::RelationClaim)),
                "bit {bit} must be rejected, got {rejection:?}"
            );
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
            let rejection = check(&key, &scheme, &commitment, &public_words, &tampered);
            assert!(
                matches!(&rejection, Err(WordProofError::RelationClaim)),
                "operand {slot} must be rejected, got {rejection:?}"
            );
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
            let rejection = check(&key, &scheme, &commitment, &public_words, &tampered);
            assert!(
                matches!(&rejection, Err(WordProofError::RelationClaim)),
                "round {round} must be rejected, got {rejection:?}"
            );
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
            Err(WordProofError::RelationSum)
        ));
    }

    #[test]
    fn a_consistent_proof_of_a_false_statement_only_fails_the_zero_sum_check() {
        // Mutation: break the statement, then prove its true nonzero sum honestly.
        let (system, public_words, mut words) = word64_statement(0x0C0F_FEE0_0C0F_FEE0);
        words[0] = Word64::new(words[0].get() ^ 1);
        assert_eq!(
            system.verify(&public_words, &words),
            Err(VerificationError::Unsatisfied {
                kind: ConstraintKind::Zero,
                constraint: 0,
            })
        );

        let key = WordProofKey::new(system.clone()).unwrap();
        let arity = key.trace_variables();
        let scheme = commitment_scheme(arity);
        let values = PackedWitness::new(&system, &public_words, &words).unwrap();

        let mut sponge = challenger();
        let (commitment, prover_data) = scheme
            .commit_bits(&key.trace_bits(&values, arity), &mut sponge)
            .unwrap();
        let variables = key.zerocheck_variables();
        let mut transcript = ProofProverTranscript::<_, EF, EF>::new(
            &mut sponge,
            key.transcript_shape(arity),
            &public_words,
        );
        let (vanishing_point, batching) = transcript.challenges(variables).unwrap();

        // Sum the batched relation over the cube directly, from the relation definition.
        let columns = OperationColumns::new(key.system(), &values).unwrap();
        let rows = 1 << key.shift.constraint_variables();
        let equality = equality_weights(&vanishing_point);
        let linear = bit_table::<Word64, EF>(columns.zero(), rows);
        let bitwise = columns
            .bitwise_and()
            .each_ref()
            .map(|column| bit_table::<Word64, EF>(column, rows));
        let claimed = (0..equality.len())
            .map(|cell| {
                let product = bitwise[0][cell] * bitwise[1][cell] - bitwise[2][cell];
                equality[cell] * (linear[cell] + batching * product)
            })
            .sum::<EF>();
        assert_ne!(claimed, EF::ZERO, "a false statement has a nonzero sum");

        // Everything after the claimed sum is an honest run over that sum.
        let mut prover = RelationZerocheck::new(equality, linear, bitwise, batching);
        let (zerocheck, point) = transcript.zerocheck(|challenger| {
            prover.prove::<EF, _>(challenger, variables, ZEROCHECK_DEGREE, 0, claimed)
        });
        let operands = prover.terminal_operands();
        transcript.finish();
        let (shift, opening) = key
            .shift
            .prove::<EF, EF, _>(&values, &key.operand_claim(&point, &operands), &mut sponge)
            .unwrap();
        let (_, opening_proof) = scheme
            .open_at_points(
                prover_data,
                &[key.commitment_point(opening.point(), arity)],
                &mut sponge,
            )
            .unwrap();
        let forged = WordProof {
            zerocheck,
            operands,
            shift,
            opening: opening_proof,
        };

        // Replay far enough to show the closing check accepts this proof unaided.
        let mut replay = challenger();
        scheme.observe_commitment(&commitment, &mut replay);
        let mut verifier = ProofVerifierTranscript::<_, EF, EF>::new(
            &mut replay,
            key.transcript_shape(arity),
            &public_words,
        );
        let (replayed_point, replayed_batching) = verifier.challenges(variables).unwrap();
        let (end, final_claim) = verifier
            .zerocheck(|challenger| {
                forged
                    .zerocheck
                    .verify(challenger, variables, ZEROCHECK_DEGREE, 0)
            })
            .unwrap();
        verifier.finish();
        assert_eq!(
            final_claim,
            closing_value(
                Point::eval_eq(&replayed_point, end.as_slice()),
                &forged.operands,
                replayed_batching,
            )
        );

        // Only the zero-sum refusal stands between this proof and acceptance.
        let rejection = check(&key, &scheme, &commitment, &public_words, &forged);
        assert!(
            matches!(&rejection, Err(WordProofError::RelationSum)),
            "the zero-sum check must reject, got {rejection:?}"
        );
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
        let rejection = check(&key, &scheme, &commitment, &substituted, &proof);
        assert!(
            matches!(&rejection, Err(WordProofError::RelationClaim)),
            "a substituted public word moves the challenges, got {rejection:?}"
        );
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

        let rejection = check(&key, &scheme, &other_commitment, &public_words, &proof);
        assert!(
            matches!(&rejection, Err(WordProofError::RelationClaim)),
            "the commitment moves the vanishing point, got {rejection:?}"
        );
    }

    #[test]
    fn a_foreign_trace_opening_is_rejected() {
        // Mutation: keep every reduction and swap only the commitment's opening proof.
        //
        // Nothing before the final discharge changes, so only the commitment can catch it.
        let (system, public_words, words) = word64_statement(0x7E57_0BEC_7E57_0BEC);
        let key = WordProofKey::new(system.clone()).unwrap();
        let scheme = commitment_scheme(key.trace_variables());
        let values = PackedWitness::new(&system, &public_words, &words).unwrap();
        let (commitment, proof) = run(&key, &scheme, &values);

        let (_, other_words) = {
            let (other, _, other_words) = word64_statement(0x1D1D_2E2E_3F3F_4040);
            (other, other_words)
        };
        let other_values = PackedWitness::new(&system, &public_words, &other_words).unwrap();
        let (_, other_proof) = run(&key, &scheme, &other_values);

        let mut tampered = proof;
        tampered.opening = other_proof.opening;
        // The commitment's own reduction error is opaque here, so only it is left open.
        let rejection = check(&key, &scheme, &commitment, &public_words, &tampered);
        assert!(
            matches!(
                &rejection,
                Err(WordProofError::Commitment(BooleanPcsError::ReductionProof(
                    _
                )))
            ),
            "the commitment must refuse a foreign opening, got {rejection:?}"
        );
    }

    #[test]
    fn a_commitment_too_narrow_for_the_trace_is_refused() {
        // Eight committed 64-bit words need three word coordinates beside six bit ones.
        let (system, public_words, words) = word64_statement(0x3141_5926_5358_9793);
        let key = WordProofKey::new(system.clone()).unwrap();
        let narrow = commitment_scheme(key.trace_variables() - 1);
        let values = PackedWitness::new(&system, &public_words, &words).unwrap();

        let refusal = key
            .prove::<EF, EF, _, _>(&narrow, &values, &mut challenger())
            .err();
        assert!(
            matches!(
                &refusal,
                Some(WordProofError::TraceShape {
                    expected: 9,
                    actual: 8,
                })
            ),
            "an eight-variable commitment must be refused, got {refusal:?}"
        );

        // The verifier reads the same dimension, so it refuses the same mismatch.
        let exact = commitment_scheme(9);
        let (commitment, proof) = run(&key, &exact, &values);
        let refusal = check(&key, &narrow, &commitment, &public_words, &proof);
        assert!(
            matches!(
                &refusal,
                Err(WordProofError::TraceShape {
                    expected: 9,
                    actual: 8,
                })
            ),
            "an eight-variable commitment must be refused, got {refusal:?}"
        );
    }

    #[test]
    fn a_statement_narrower_than_one_committed_element_still_proves() {
        // A single committed 64-bit word spans six variables, one short of the minimum.
        //
        // The word axis is therefore padded up to the commitment's own arity.
        let value = ValueIndex::witness(0).expect("test position fits");
        let linear = ZeroConstraint::new(Operand::single(ShiftedValue::plain(value)));
        let system = ConstraintSystem::<Word64>::new(0, 1, vec![linear], vec![], vec![]).unwrap();
        let words = [Word64::new(0)];
        assert_eq!(system.verify(&[], &words), Ok(()));

        let key = WordProofKey::new(system.clone()).unwrap();
        assert_eq!(key.trace_variables(), 6);
        let scheme = commitment_scheme(ABSORBED + 1);
        let values = PackedWitness::new(&system, &[], &words).unwrap();
        let (commitment, proof) = run(&key, &scheme, &values);

        check(&key, &scheme, &commitment, &[], &proof).unwrap();

        // The padding must not absorb a violation: a nonzero word has to be rejected.
        let broken = PackedWitness::new(&system, &[], &[Word64::new(1)]).unwrap();
        let (broken_commitment, broken_proof) = run(&key, &scheme, &broken);
        let rejection = check(&key, &scheme, &broken_commitment, &[], &broken_proof);
        assert!(
            matches!(&rejection, Err(WordProofError::RelationClaim)),
            "a violated relation must be rejected, got {rejection:?}"
        );
    }
}
