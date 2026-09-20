//! Proving side of the ordered sub-reductions.
//!
//! # Sub-reduction order
//!
//! ```text
//!     commitment -> unsigned product -> binary product -> bitwise -> linear
//!                -> shift -> public -> commitment opening
//! ```
//!
//! The order is load-bearing and the verifier must not diverge from it.
//!
//! The two product slots read nothing from the transcript here.
//!
//! A statement declaring unsigned products is refused rather than proved.
//!
//! The relation language carries no binary-field product family.
//!
//! # Why the order is sound
//!
//! Every slot binds its claims before the challenge that consumes them.
//!
//! - The commitment is bound before the vanishing point, so the trace cannot follow it.
//! - Public words and every dimension are bound before both relation challenges.
//! - The four operand evaluations are bound before the shift reduction draws its batching axes.
//! - The reduced trace value is bound before the commitment draws its opening challenges.
//!
//! Public words enter the sponge as fixed statement data, which is the public check.
//!
//! The shift reduction then subtracts their exact contribution instead of opening them.

use alloc::vec::Vec;

use p3_binary_field::Gf2;
use p3_binary_pcs::BooleanMultilinearPcs;
use p3_challenger::fs::TranscriptField;
use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{Algebra, ExtensionField, Field};
use p3_multilinear_util::point::Point;
use p3_sumcheck::generic_degree::RoundProver;

use super::error::WordProofError;
use super::key::WordProofKey;
use super::record::{ProvedStatement, WordProof};
use super::relation::{RelationZerocheck, ZEROCHECK_DEGREE};
use super::transcript::ProofProverTranscript;
use crate::shift::transcript::equality_weights;
use crate::{OperationColumns, Packed, PackedWitness, PackedWord};

impl<W: PackedWord> WordProofKey<W> {
    /// Proves every declared relation and discharges the result through the commitment.
    ///
    /// The commitment is produced here so that it precedes every sampled challenge.
    ///
    /// # Errors
    ///
    /// Returns an error when any of the following holds.
    ///
    /// - The statement declares a relation family this protocol does not prove.
    /// - The witness or the commitment has the wrong shape.
    /// - The sampled batching coefficient vanishes.
    /// - The commitment refuses the trace or its opening.
    pub fn prove<F, EF, Pcs, Challenger>(
        &self,
        pcs: &Pcs,
        values: &PackedWitness<W>,
        challenger: &mut Challenger,
    ) -> Result<ProvedStatement<F, EF, Pcs, Challenger>, WordProofError<Pcs::Error>>
    where
        F: TranscriptField,
        EF: ExtensionField<F> + Algebra<Gf2>,
        Pcs: BooleanMultilinearPcs<EF, Challenger>,
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        // Reject every shape before the transcript absorbs a single statement value.
        self.validate_statement()?;
        self.validate_arity(pcs.num_variables())?;
        values
            .check_shape(self.system())
            .map_err(|error| WordProofError::SegmentLength {
                segment: error.segment,
                expected: error.expected,
                actual: error.actual,
            })?;

        // Binding the commitment first stops the trace being chosen after the challenges.
        let (commitment, prover_data) = pcs
            .commit_bits(&self.trace_bits(values), challenger)
            .map_err(WordProofError::Commitment)?;

        // Both relation challenges follow the complete statement.
        let public_words = values
            .public()
            .iter()
            .copied()
            .map(W::unpack)
            .collect::<Vec<_>>();
        let variables = self.zerocheck_variables();
        let mut transcript = ProofProverTranscript::<_, F, EF>::new(
            challenger,
            self.transcript_shape(),
            &public_words,
        );
        let (vanishing_point, batching) = transcript.challenges(variables);
        if batching.is_zero() {
            return Err(WordProofError::DegenerateBatching);
        }

        // The batched relation polynomial vanishes on the whole padded cube.
        let columns = OperationColumns::new(self.system(), values).map_err(|error| {
            WordProofError::SegmentLength {
                segment: error.segment,
                expected: error.expected,
                actual: error.actual,
            }
        })?;
        let rows = 1 << self.shift.constraint_variables();
        let mut prover = RelationZerocheck::new(
            equality_weights(&vanishing_point),
            bit_table::<W, EF>(columns.zero(), rows),
            columns
                .bitwise_and()
                .each_ref()
                .map(|column| bit_table::<W, EF>(column, rows)),
            batching,
        );
        let (zerocheck, point) = transcript.zerocheck(|challenger| {
            prover.prove::<F, _>(challenger, variables, ZEROCHECK_DEGREE, 0, EF::ZERO)
        });
        let operands = prover.terminal_operands();
        transcript.finish();

        // Every operand claim shares the point the vanishing check ended on.
        let (shift, opening) = self.shift.prove::<F, EF, _>(
            values,
            &self.operand_claim(&point, &operands),
            challenger,
        )?;

        // One authenticated opening closes the whole protocol.
        let (opened, opening_proof) = pcs
            .open_at_points(
                prover_data,
                &[Point::new(opening.point().to_vec())],
                challenger,
            )
            .map_err(WordProofError::Commitment)?;
        if opened.first() != Some(&opening.value()) {
            return Err(WordProofError::OpeningValue);
        }

        Ok((
            commitment,
            WordProof {
                zerocheck,
                operands,
                shift,
                opening: opening_proof,
            },
        ))
    }
}

/// Expands one packed column into its bit multilinear over the padded cube.
fn bit_table<W: PackedWord, EF: Field>(column: &[Packed<W>], rows: usize) -> Vec<EF> {
    // Row index leads the flat address and within-word bit index trails it.
    let width = W::BITS as usize;
    let mut table = EF::zero_vec(rows * width);
    for (row, &packed) in column.iter().enumerate() {
        let mut remaining = W::unpack(packed).to_u64();
        while remaining != 0 {
            // Visit only set lanes, which halves the scatter on random words.
            let bit = remaining.trailing_zeros() as usize;
            table[row * width + bit] = EF::ONE;
            remaining &= remaining - 1;
        }
    }
    table
}

#[cfg(test)]
mod tests {
    use p3_binary_field::BinaryField128;
    use p3_field::PrimeCharacteristicRing;
    use p3_word::{Word32, Word64};

    use super::*;

    type EF = BinaryField128;

    #[test]
    fn a_packed_column_expands_to_its_set_bits() {
        // Fixture state: one 32-bit word with its lowest and highest bits set.
        let column = [Word32::pack(Word32::new(0x8000_0001))];
        let table = bit_table::<Word32, EF>(&column, 2);

        // Set lanes are one, every other lane of both rows is zero.
        assert_eq!(table.len(), 64);
        for (index, value) in table.iter().enumerate() {
            let expected = if index == 0 || index == 31 {
                EF::ONE
            } else {
                EF::ZERO
            };
            assert_eq!(*value, expected, "lane {index}");
        }
    }

    #[test]
    fn a_padded_row_contributes_no_bits() {
        // Fixture state: one 64-bit row inside a two-row padded cube.
        let column = [Word64::pack(Word64::new(u64::MAX))];
        let table = bit_table::<Word64, EF>(&column, 2);

        assert!(table[..64].iter().all(|value| *value == EF::ONE));
        assert!(table[64..].iter().all(|value| *value == EF::ZERO));
    }
}
