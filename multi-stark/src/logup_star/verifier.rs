//! Check that every reader pulled what the table holds at the entry it named.

use p3_challenger::fs::TranscriptField;
use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::ExtensionField;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;

use super::error::LogupStarError;
use super::plan::{BlockRole, LogupStarPlan};
use super::proof::{LogupStarOutput, LogupStarProof, TableOutput};
use super::prover::PRODUCT_DEGREE;
use super::transcript::{LogupStarShape, LogupStarVerifierTranscript};
use super::{TableLookup, position, product, statement_values};
use crate::fractional_gkr::verify_fractional_gkr;

impl<F, EF> LogupStarProof<F, EF>
where
    F: TranscriptField,
    EF: ExtensionField<F>,
{
    /// Check this reduction against the statement it claims to prove.
    ///
    /// # Arguments
    ///
    /// - `lookups`: the statement, one entry per table.
    /// - `challenger`: sponge of the surrounding protocol, borrowed for the run.
    ///
    /// # Returns
    ///
    /// The evaluation claims the caller must discharge against its commitments.
    ///
    /// Nothing returned here is authenticated.
    ///
    /// What is proved is that those claims and the statement's own stand or fall together.
    ///
    /// # Errors
    ///
    /// Returns an error when the proof is malformed.
    ///
    /// Returns an error when the reduction fails its own checks.
    ///
    /// Returns an error when a lookup identity does not hold.
    ///
    /// # Panics
    ///
    /// Panics if the statement does not describe a reduction.
    ///
    /// That is a caller error rather than a malformed proof.
    pub fn verify<Challenger>(
        &self,
        lookups: &[TableLookup<'_, EF>],
        challenger: &mut Challenger,
    ) -> Result<LogupStarOutput<EF>, LogupStarError>
    where
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        // Both sides derive the layout from the statement alone, so none of it travels.
        let plan = LogupStarPlan::new(lookups);

        // Phase 1: measure the proof against the shape.
        //
        // Nothing is absorbed yet, so every rejection here is a clean error.
        //
        // Doing it first is also what keeps the replay below from failing on a length.
        //
        // The transcript is then playable to its end on every path out of this function.
        self.check_shape(&plan)?;

        // Phase 2: replay, from the reader weights down to the column claims.
        //
        // Every step is played even once a check has failed.
        //
        // The sponge the caller gets back is then in the state the prover left it in.
        let shape = LogupStarShape::new(&plan);
        let mut transcript =
            LogupStarVerifierTranscript::<Challenger, F, EF>::new(challenger, &shape);

        transcript.statement(&statement_values(lookups));
        let reader_batching = transcript.reader_batching();
        for pushforward in &self.pushforwards {
            transcript.pushforward(pushforward);
        }
        let entry_challenges = transcript.entry_challenges(plan.num_tables());

        let reduction = transcript.fraction_reduction(|challenger| {
            verify_fractional_gkr::<F, EF, _>(&self.fraction_gkr, plan.num_variables, challenger)
        });

        transcript.position_claims(&self.position_claims);
        let column_batching = transcript.column_batching();

        let product = transcript.product_sumcheck(|challenger| {
            self.product
                .verify(challenger, plan.max_table_variables, PRODUCT_DEGREE, 0)
        });

        transcript.column_claims(&self.column_claims.concat());
        transcript.finish();

        // Phase 3: check what was replayed.
        let reduction = reduction?;

        // The reduction opened the padded fractions at one point.
        //
        // Rebuilding that opening from the statement is what ties it to this lookup.
        //
        // Every weight on the numerator side is public.
        //
        // So is the denominator side, apart from the position values just bound.
        let (numerator, denominator) = self.rebuild_opening(
            &plan,
            lookups,
            &reduction.point,
            reader_batching,
            &entry_challenges,
        );
        if numerator != reduction.numerator {
            return Err(LogupStarError::LeafNumerator);
        }
        if denominator != reduction.denominator {
            return Err(LogupStarError::LeafDenominator);
        }

        // The sumcheck must start from the sum the statement asks for, not one of its own.
        if self.product.claimed_sum
            != product::claimed_sum(lookups, reader_batching, column_batching)
        {
            return Err(LogupStarError::ProductClaimedSum);
        }

        let (table_point, final_value) = product?;
        if final_value
            != product::final_value::<F, EF>(
                &table_point,
                &self.pushforwards,
                &self.column_claims,
                column_batching,
            )
        {
            return Err(LogupStarError::ProductFinalValue);
        }

        Ok(LogupStarOutput {
            position_point: reduction.point.get_subpoint_over_range(
                plan.num_variables - plan.max_reader_variables..plan.num_variables,
            ),
            table_point,
            tables: plan
                .tables
                .iter()
                .enumerate()
                .map(|(table, shape)| {
                    let first = plan.reader_offset(table);
                    TableOutput {
                        column_claims: self.column_claims[table].clone(),
                        position_claims: self.position_claims[first..first + shape.readers.len()]
                            .to_vec(),
                    }
                })
                .collect(),
        })
    }

    /// Reject a proof whose counts the statement never describes.
    ///
    /// Every count here decides how the replay is walked.
    ///
    /// All of them are therefore settled before the sponge is touched.
    fn check_shape(&self, plan: &LogupStarPlan) -> Result<(), LogupStarError> {
        if self.pushforwards.len() != plan.num_tables() {
            return Err(LogupStarError::PushforwardCount {
                expected: plan.num_tables(),
                actual: self.pushforwards.len(),
            });
        }
        for (table, (shape, pushforward)) in plan.tables.iter().zip(&self.pushforwards).enumerate()
        {
            let expected = 1 << shape.num_variables;
            if pushforward.len() != expected {
                return Err(LogupStarError::PushforwardWidth {
                    table,
                    expected,
                    actual: pushforward.len(),
                });
            }
        }

        if self.position_claims.len() != plan.num_readers() {
            return Err(LogupStarError::PositionClaimCount {
                expected: plan.num_readers(),
                actual: self.position_claims.len(),
            });
        }

        if self.column_claims.len() != plan.num_tables() {
            return Err(LogupStarError::ColumnClaimShape {
                expected: plan.num_tables(),
                actual: self.column_claims.len(),
            });
        }
        for (table, (shape, claims)) in plan.tables.iter().zip(&self.column_claims).enumerate() {
            if claims.len() != shape.width {
                return Err(LogupStarError::ColumnClaimCount {
                    table,
                    expected: shape.width,
                    actual: claims.len(),
                });
            }
        }

        Ok(())
    }

    /// Rebuild the numerator and denominator the reduction should have opened.
    ///
    /// Each block contributes its own value, weighted by the equality factor selecting it:
    ///
    /// ```text
    ///     reader block   weight * scale * eq(claim point, own coordinates)
    ///                    weight * (challenge - claimed position value)
    ///
    ///     table  block   weight * pushforward at the own coordinates
    ///                    weight * (position embedding at those coordinates - challenge)
    /// ```
    ///
    /// Padding carries a zero over a one.
    ///
    /// Block weights over a whole cover of the cube sum to one.
    ///
    /// So the padding's share is whatever the real blocks leave behind, and none is listed.
    fn rebuild_opening(
        &self,
        plan: &LogupStarPlan,
        lookups: &[TableLookup<'_, EF>],
        point: &Point<EF>,
        reader_batching: EF,
        entry_challenges: &[EF],
    ) -> (EF, EF) {
        let mut numerator = EF::ZERO;
        let mut denominator = EF::ZERO;
        let mut covered = EF::ZERO;

        for block in &plan.blocks {
            let weight = block.weight::<F, EF>(point, plan.num_variables);
            let own = block.subpoint(point, plan.num_variables);
            covered += weight;

            match block.role {
                BlockRole::Reader { index } => {
                    // A reader's weights are the equality tensor at its own claim point.
                    //
                    // Its position in the table earns one power of the batching challenge.
                    let reader = &lookups[block.table].readers[index];
                    let scale = reader_batching.exp_u64(index as u64);
                    numerator +=
                        weight * scale * Point::eval_eq(reader.point.as_slice(), own.as_slice());

                    let claim = self.position_claims[plan.reader_offset(block.table) + index];
                    denominator += weight * (entry_challenges[block.table] - claim);
                }
                BlockRole::Table => {
                    // The pushforward travels in the clear, so this side is entirely public.
                    let pushforward = Poly::new(self.pushforwards[block.table].as_slice());
                    numerator += weight * pushforward.eval_ext::<F>(&own);

                    // Negated, which is what makes an honest statement sum to zero.
                    denominator +=
                        weight * (position::eval::<F, EF>(&own) - entry_challenges[block.table]);
                }
            }
        }

        (numerator, denominator + (EF::ONE - covered))
    }
}
