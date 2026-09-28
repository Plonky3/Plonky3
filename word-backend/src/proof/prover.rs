//! Proving side of the ordered sub-reductions.
//!
//! # Step order
//!
//! ```text
//!     commitment -> statement
//!                -> integer multiplication: row point, both product trees, product claims
//!                -> vanishing point -> batching coefficient
//!                -> batched zerocheck: zero, AND, low bit, product claims -> operand claims
//!                -> shift batching, public share subtracted
//!                -> bit sumcheck -> word sumcheck -> trace evaluation
//!                -> commitment opening
//! ```
//!
//! The order is load-bearing and the verifier must not diverge from it.
//!
//! One zerocheck covers every local family and the product claims under one coefficient.
//!
//! Binary-field multiplication is not a relation of this language, so no step proves it.
//!
//! # Why the order is sound
//!
//! Every step binds its claims before the challenge that consumes them.
//!
//! - The commitment is bound before the vanishing point, so the trace cannot follow it.
//! - Public words and every dimension are bound before the multiplication row point.
//! - Every product claim is bound before the vanishing point and coefficient that weight it.
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
use p3_field::{Algebra, ExtensionField};
use p3_multilinear_util::point::Point;
use p3_sumcheck::generic_degree::RoundProver;

use super::error::WordProofError;
use super::key::WordProofKey;
use super::record::WordProof;
use super::relation::{ProductTables, RelationZerocheck, ZEROCHECK_DEGREE, claimed_sum};
use super::transcript::ProofProverTranscript;
use crate::columns::bit_table;
use crate::{OperationColumns, PackedWitness, PackedWord};

impl<W: PackedWord> WordProofKey<W> {
    /// Proves every declared relation and discharges the result through the commitment.
    ///
    /// The commitment is produced here so that it precedes every sampled challenge.
    ///
    /// # Errors
    ///
    /// Returns an error when any of the following holds.
    ///
    /// - The witness has the wrong shape, or the commitment is too narrow for it.
    /// - The statement declares products the challenge field is too small to lift.
    /// - The sampled batching coefficient vanishes.
    /// - The commitment refuses the trace or its opening.
    #[allow(
        clippy::type_complexity,
        reason = "an alias would need bounds it cannot carry"
    )]
    pub fn prove<F, EF, Pcs, Challenger>(
        &self,
        pcs: &Pcs,
        values: &PackedWitness<W>,
        challenger: &mut Challenger,
    ) -> Result<(Pcs::Commitment, WordProof<F, EF, Pcs::Proof>), WordProofError<Pcs::Error>>
    where
        F: TranscriptField,
        EF: ExtensionField<F> + Algebra<Gf2>,
        Pcs: BooleanMultilinearPcs<EF, Challenger>,
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        // Reject every shape before the transcript absorbs a single statement value.
        let commitment_variables = pcs.num_variables();
        self.validate_arity(commitment_variables)?;
        let columns = OperationColumns::new(self.statement(), values).map_err(|error| {
            WordProofError::SegmentLength {
                segment: error.segment,
                expected: error.expected,
                actual: error.actual,
            }
        })?;
        if let Some(reduction) = &self.integer_mul {
            reduction.check_field::<EF>()?;
        }

        // Binding the commitment first stops the trace being chosen after the challenges.
        let (commitment, prover_data) = pcs
            .commit_bits(&self.trace_bits(values, commitment_variables), challenger)
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
            self.transcript_shape(commitment_variables),
            &public_words,
        );

        // Every product reduces to claims on its four columns before either relation draw.
        let product = transcript.integer_mul(|challenger| {
            self.integer_mul
                .map(|reduction| reduction.prove::<F, EF, W, _>(columns.integer_mul(), challenger))
        });
        let (vanishing_point, batching) = transcript
            .challenges(variables)
            .ok_or(WordProofError::DegenerateBatching)?;

        // The local terms vanish on the padded cube and each claim term sums to its claim.
        let rows = 1 << self.shift.constraint_variables();
        let (integer_mul, tables, sum) = match product {
            Some((proof, claims)) => {
                let tables = ProductTables {
                    columns: columns
                        .integer_mul()
                        .each_ref()
                        .map(|column| bit_table::<W, EF>(column, rows)),
                    weights: self.product_weights(&vanishing_point, &claims),
                };
                let sum = claimed_sum(&claims.each_ref().map(|claim| claim.value), batching);
                (Some(proof), Some(tables), sum)
            }
            None => (None, None, EF::ZERO),
        };
        let mut prover = RelationZerocheck::new(
            Point::new(vanishing_point.as_slice()).equality_weights_msb(),
            bit_table::<W, EF>(columns.zero(), rows),
            columns
                .bitwise_and()
                .each_ref()
                .map(|column| bit_table::<W, EF>(column, rows)),
            tables,
            batching,
        );
        let (zerocheck, point) = transcript.zerocheck(|challenger| {
            prover.prove::<F, _>(challenger, variables, ZEROCHECK_DEGREE, 0, sum)
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
                &[self.commitment_point(opening.point(), commitment_variables)],
                challenger,
            )
            .map_err(WordProofError::Commitment)?;
        if opened.first() != Some(&opening.value()) {
            return Err(WordProofError::OpeningValue);
        }

        Ok((
            commitment,
            WordProof {
                integer_mul,
                zerocheck,
                operands,
                shift,
                opening: opening_proof,
            },
        ))
    }
}
