//! Prove that every reader pulled what the table holds at the entry it named.

use alloc::vec;
use alloc::vec::Vec;

use p3_challenger::fs::TranscriptField;
use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, PackedValue};
use p3_multilinear_util::poly::{Poly, PolyMaybePacked};
use p3_multilinear_util::split_eq::SplitEq;
use p3_sumcheck::generic_degree::RoundProver;
use p3_util::log2_strict_usize;

use super::plan::{BlockRole, LogupStarPlan};
use super::product::{self, ProductProver};
use super::proof::{LogupStarOutput, LogupStarProof, TableOutput};
use super::transcript::{LogupStarProverTranscript, LogupStarShape};
use super::witness::Weights;
use super::{TableLookup, TableWitness, position};
use crate::fractional_gkr::{Fraction, LeafNumerator, prove_fractional_gkr};

impl<F, EF> LogupStarProof<F, EF>
where
    F: TranscriptField,
    EF: ExtensionField<F>,
{
    /// Prove one indexed-lookup reduction.
    ///
    /// # Arguments
    ///
    /// - `lookups`: the statement, one entry per table.
    /// - `witness`: the tables and the entry each reader row names, in the same order.
    /// - `challenger`: sponge of the surrounding protocol, borrowed for the run.
    ///
    /// # Returns
    ///
    /// The proof, and the evaluation claims the caller must discharge against its commitments.
    ///
    /// # Panics
    ///
    /// Panics if the statement and the witness disagree on how many tables or readers there are.
    ///
    /// Panics if a row names an entry its table does not have.
    ///
    /// Panics if the claims the statement carries are not the ones the witness produces.
    ///
    /// That means the caller was asked to prove something false.
    #[tracing::instrument(skip_all, name = "prove logup*")]
    pub fn prove<Challenger>(
        lookups: &[TableLookup<'_, EF>],
        witness: &[TableWitness<'_, F>],
        challenger: &mut Challenger,
    ) -> (Self, LogupStarOutput<EF>)
    where
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        assert_eq!(
            lookups.len(),
            witness.len(),
            "the statement and the witness must describe the same tables"
        );
        for (lookup, table) in lookups.iter().zip(witness) {
            assert_eq!(
                lookup.readers.len(),
                table.readers.len(),
                "the statement and the witness must describe the same readers"
            );
            assert_eq!(
                lookup.width(),
                table.columns.len(),
                "the statement and the witness must describe the same columns"
            );
        }

        // Both sides derive the layout from the statement alone, so none of it travels.
        let plan = LogupStarPlan::new(lookups);
        let shape = LogupStarShape::new(&plan);
        let mut transcript =
            LogupStarProverTranscript::<Challenger, F, EF>::new(challenger, &shape);

        // Phase 1: bind what is being proved, then weigh the readers of each table.
        //
        // Every challenge below is a function of the claims.
        //
        // None of them can be seen before the claims are fixed.
        //
        // The pushforward depends on the batching challenge, so it is drawn next.
        transcript.statement(&TableLookup::statement(lookups));
        let reader_batching = transcript.reader_batching();
        let weights = Weights::build(&plan, lookups, witness, reader_batching);
        for pushforward in &weights.pushforwards {
            transcript.pushforward(pushforward);
        }

        // Phase 2: one challenge per table, drawn only now that every pushforward is bound.
        //
        // Sharing one challenge between tables would let two of them cancel each other's errors.
        let entry_challenges = transcript.entry_challenges(plan.num_tables());

        // Phase 3: prove the fractions sum to zero.
        //
        // The tables are packed when the field has lanes to fill.
        //
        // The cube also has to be wide enough to address those lanes.
        //
        // That is what lets the reduction use its SIMD kernel.
        let (numerator, denominator) = weights.leaf_tables(&plan, witness, &entry_challenges);
        // A field with one lane per element would only pay for the copy.
        let lanes = log2_strict_usize(F::Packing::WIDTH);
        let (numerator, denominator) = if lanes > 0 && plan.num_variables >= lanes {
            (
                PolyMaybePacked::Packed(numerator.pack::<F, EF>()),
                PolyMaybePacked::Packed(denominator.pack::<F, EF>()),
            )
        } else {
            (
                PolyMaybePacked::Scalar(numerator),
                PolyMaybePacked::Scalar(denominator),
            )
        };
        let (fraction_gkr, gkr_output) = transcript.fraction_reduction(|challenger| {
            prove_fractional_gkr(
                Fraction {
                    n: LeafNumerator::Ext(&numerator),
                    d: &denominator,
                },
                challenger,
            )
        });

        // Phase 4: open each reader's position column at the point the reduction reached.
        //
        // The column is the reader's rows under the table-position embedding.
        //
        // That is what the verifier rebuilds the denominator side from.
        let mut position_claims = vec![EF::ZERO; plan.num_readers()];
        for block in &plan.blocks {
            let BlockRole::Reader { index } = block.role else {
                continue;
            };
            let own = block.subpoint(&gkr_output.point, plan.num_variables);
            let column = witness[block.table].readers[index]
                .positions
                .iter()
                .map(|&entry| position::embed::<F>(entry))
                .collect::<Vec<_>>();
            position_claims[plan.reader_offset(block.table) + index] =
                SplitEq::<F, EF>::new_packed(&own, EF::ONE).eval_base(Poly::new(column.as_slice()));
        }
        transcript.position_claims(&position_claims);

        // Phase 5: bind every table to its pushforward.
        //
        // One challenge separates one column claim from the next.
        //
        // It is drawn once the reduction above has fixed everything it could be adapted to.
        let column_batching = transcript.column_batching();
        let columns = witness
            .iter()
            .map(|table| table.columns)
            .collect::<Vec<_>>();
        let (mut product_prover, claimed_sum) = ProductProver::new::<F>(
            plan.max_table_variables,
            &weights.pushforwards,
            &columns,
            column_batching,
        );

        // The statement says what that sum has to be, and the witness says what it is.
        //
        // A caller proving something false is caught here.
        //
        // It is not left to build a proof no verifier would take.
        //
        // The cost is one pass over the claims, so this stays a hard assertion.
        assert_eq!(
            claimed_sum,
            product::claimed_sum(lookups, reader_batching, column_batching),
            "the claims the statement carries are not the ones the witness produces"
        );

        let (product, product_point) = transcript.product_sumcheck(|challenger| {
            product_prover.prove::<F, Challenger>(
                challenger,
                plan.max_table_variables,
                PRODUCT_DEGREE,
                0,
                claimed_sum,
            )
        });

        // Phase 6: open every table column at the point the sumcheck landed on.
        let column_claims = plan
            .tables
            .iter()
            .zip(witness)
            .map(|(shape, table)| {
                let own = product_point.get_subpoint_over_range(
                    plan.max_table_variables - shape.num_variables..plan.max_table_variables,
                );
                table
                    .columns
                    .iter()
                    .map(|column| {
                        SplitEq::<F, EF>::new_packed(&own, EF::ONE).eval_base(Poly::new(*column))
                    })
                    .collect()
            })
            .collect::<Vec<Vec<EF>>>();
        transcript.column_claims(&column_claims.concat());
        transcript.finish();

        let output = LogupStarOutput {
            position_point: gkr_output.point.get_subpoint_over_range(
                plan.num_variables - plan.max_reader_variables..plan.num_variables,
            ),
            table_point: product_point,
            tables: plan
                .tables
                .iter()
                .enumerate()
                .map(|(table, shape)| {
                    let first = plan.reader_offset(table);
                    TableOutput {
                        column_claims: column_claims[table].clone(),
                        position_claims: position_claims[first..first + shape.readers.len()]
                            .to_vec(),
                    }
                })
                .collect(),
        };

        (
            Self {
                pushforwards: weights.pushforwards,
                fraction_gkr,
                position_claims,
                product,
                column_claims,
            },
            output,
        )
    }
}

/// Per-variable degree of the product sumcheck.
///
/// Its summand is a pushforward times a column, both multilinear.
pub(crate) const PRODUCT_DEGREE: usize = 2;
