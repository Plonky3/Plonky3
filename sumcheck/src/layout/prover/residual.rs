//! Residual sumcheck prover handed over by the suffix layout.

use alloc::vec::Vec;

use p3_challenger::fs::TranscriptField;
use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{Algebra, ExtensionField, Field};
use p3_maybe_rayon::prelude::*;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;

use super::banked::BankedColumn;
use super::suffix::weighted_sum;
use crate::SumcheckData;
use crate::layout::witness::{Table, TablePlacement, column_slots};
use crate::strategy::{Basis, IntoTranscriptField, ReprSumcheckProver, VariableOrder};
use crate::transcript::{ProverTranscript, SumcheckShape};

/// Rows per block of the column aggregate.
///
/// Blocks split a tall table's rows. When they are too few to occupy every thread, each block
/// also sums its columns in parallel.
const AGGREGATE_BLOCK: usize = 1 << 12;

/// Residual sumcheck prover over the stacked polynomial, with its tables in `R`.
///
/// # Routes
///
/// ```text
///     dense       one product over the whole stacked space
///
///     row-first   W(c, x) = s_c * T(x)  for every column c and row x
///                 row rounds:     sum_x  A(x) * T(x),   A(x) = sum_c  s_c * P_c(x)
///                 column rounds:  sum_c  P_c(r) * s_c * T(r)
///
///     banked      one column P under equality claims, W(x) = sum_c  gamma_c * eq(z_c, x)
///                 stage rounds:   over the bank sums of P, 2^h entries per claim
///                 between stages: P bound once at the stage's challenges
/// ```
///
/// Every route measures the same round polynomials, so all of them play the same transcript.
///
/// Suffix binding fixes the row variables first. The row-first route therefore measures its
/// first `log2(height)` rounds on tables one column long, not stacked-space long. Once every row
/// variable is bound, each column is evaluated at the row challenges and the remaining rounds
/// run over one entry per column slot.
#[derive(Debug, Clone)]
pub struct SuffixResidualProver<F: Field, EF: ExtensionField<F>, R: Field> {
    /// The rounds currently being played.
    prover: ReprSumcheckProver<F, EF, R>,
    /// The column slots still to come, present while the row-first route binds rows.
    columns: Option<ColumnHandoff<F, EF, R>>,
    /// The column the banked route plays its stages over, present on that route alone.
    banked: Option<BankedColumn<F, R>>,
}

/// The column slots a row-first prover continues over once every row variable is bound.
#[derive(Debug, Clone)]
struct ColumnHandoff<F: Field, EF, R> {
    /// Source tables, all of one height.
    tables: Vec<Table<F>>,
    /// Slot of every source column inside the stacked polynomial.
    placements: Vec<TablePlacement>,
    /// Per source table and column: the scale on the shared row weights, zero if unopened.
    scales: Vec<Vec<R>>,
    /// Arity of the stacked polynomial.
    num_variables: usize,
    /// Row challenges sampled so far, in round order.
    challenges: Vec<EF>,
}

impl<F, EF, R> SuffixResidualProver<F, EF, R>
where
    F: Field,
    EF: ExtensionField<F>,
    R: IntoTranscriptField<EF> + Algebra<F>,
{
    /// Wraps a prover over the whole stacked space.
    pub(crate) const fn dense(prover: ReprSumcheckProver<F, EF, R>) -> Self {
        Self {
            prover,
            columns: None,
            banked: None,
        }
    }

    /// Builds the banked route over one dense column filling the stacked space.
    ///
    /// # Arguments
    ///
    /// - `table` — the lone source table, one column.
    /// - `claims` — every claim's point and batching coefficient, each opening the column directly.
    /// - `sum` — the batched claim.
    /// - `stage_rounds` — rounds a full stage plays over bank sums before the column is bound.
    #[tracing::instrument(skip_all)]
    pub(crate) fn banked(
        table: Table<F>,
        claims: &[(Point<EF>, EF)],
        sum: EF,
        stage_rounds: usize,
    ) -> Self {
        let (prover, column) = BankedColumn::new(table, claims, sum, stage_rounds);
        Self {
            prover,
            columns: None,
            banked: Some(column),
        }
    }

    /// Builds the row-first route from the shared row weights and one scale per column.
    ///
    /// # Arguments
    ///
    /// - `tables` — source tables, all of one height.
    /// - `placements` — slot of every source column inside the stacked polynomial.
    /// - `scales` — per source table and column, the scale on `row_weights`; zero if unopened.
    /// - `row_weights` — the row table every column's weights are a multiple of.
    /// - `sum` — the batched claim.
    /// - `num_variables` — arity of the stacked polynomial.
    #[tracing::instrument(skip_all)]
    pub(crate) fn row_first(
        tables: Vec<Table<F>>,
        placements: Vec<TablePlacement>,
        scales: Vec<Vec<EF>>,
        row_weights: Vec<R>,
        sum: EF,
        num_variables: usize,
    ) -> Self {
        // Every product with a source cell is taken in `R`, which multiplies by `F` directly.
        let scales: Vec<Vec<R>> = scales
            .into_iter()
            .map(|scales| scales.into_iter().map(R::from).collect())
            .collect();
        let aggregate = aggregate_columns(&tables, &scales, row_weights.len());
        let prover = ReprSumcheckProver::from_repr_tables(
            VariableOrder::Suffix,
            Poly::new(aggregate),
            Poly::new(row_weights),
            sum,
        );
        Self {
            prover,
            columns: Some(ColumnHandoff {
                tables,
                placements,
                scales,
                num_variables,
                challenges: Vec::new(),
            }),
            banked: None,
        }
    }

    /// Whether the row-first route is still binding rows.
    #[cfg(test)]
    pub(crate) const fn is_row_first(&self) -> bool {
        self.columns.is_some()
    }

    /// Whether the banked route plays the rounds.
    #[cfg(test)]
    pub(crate) const fn is_banked(&self) -> bool {
        self.banked.is_some()
    }

    /// Returns the current claimed sum over the remaining unbound variables.
    pub fn claimed_sum(&self) -> EF {
        self.prover.claimed_sum()
    }

    /// Returns the number of remaining (unbound) variables.
    pub fn num_variables(&self) -> usize {
        let slot_variables = self
            .columns
            .as_ref()
            .map_or(0, ColumnHandoff::num_slot_variables);
        let played = self.banked.as_ref().map_or_else(
            || self.prover.num_variables(),
            |column| column.num_variables(&self.prover),
        );
        played + slot_variables
    }

    /// Applies an outstanding binding, so the tables are current with the claim.
    ///
    /// See [`ReprSumcheckProver::settle`].
    pub fn settle(&mut self) {
        self.prover.settle();
    }

    /// The stacked polynomial bound at every challenge sampled so far, in `R`.
    ///
    /// Only the banked route holds that polynomial, and only once a stage is fully played.
    /// The column is then bound here rather than when the next round opens a stage.
    ///
    /// `None`, applying nothing, on any other route or state.
    pub fn bound_column(&mut self) -> Option<&[R]> {
        match &mut self.banked {
            Some(column) if column.stage_played(&self.prover) => {
                Some(column.bound(&mut self.prover))
            }
            _ => None,
        }
    }

    /// Runs `folding_factor` sumcheck rounds.
    ///
    /// Plays the same transcript as [`ReprSumcheckProver::compute_sumcheck_polynomials`],
    /// including the challenge left outstanding on return.
    ///
    /// # Returns
    ///
    /// The verifier challenges sampled during this batch.
    ///
    /// # Panics
    ///
    /// - Folding factor must not exceed the current number of remaining variables.
    #[tracing::instrument(skip_all, level = "debug")]
    pub fn compute_sumcheck_polynomials<Challenger>(
        &mut self,
        sumcheck_data: &mut SumcheckData<F, EF>,
        challenger: &mut Challenger,
        folding_factor: usize,
        pow_bits: usize,
    ) -> Point<EF>
    where
        F: TranscriptField,
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        let shape = SumcheckShape::new(folding_factor, pow_bits, Basis::Evaluation);
        let mut transcript = ProverTranscript::<Challenger, F, EF>::new(challenger, shape);

        let mut challenges = Vec::with_capacity(folding_factor);
        for _ in 0..folding_factor {
            // Once every row variable is bound, the column slots take over.
            if self.prover.num_variables() == 0
                && let Some(columns) = self.columns.take()
            {
                self.prover = columns.into_prover(&mut self.prover);
            }

            // Once a stage is played, the next one opens over the column bound at its challenges.
            if let Some(column) = &mut self.banked
                && column.stage_played(&self.prover)
            {
                self.prover = column.next_stage(&mut self.prover);
            }

            let r = self.prover.round(sumcheck_data, &mut transcript);
            if let Some(columns) = &mut self.columns {
                columns.challenges.push(r);
            }
            if let Some(column) = &mut self.banked {
                column.push_challenge(R::from(r));
            }
            challenges.push(r);
        }

        // Require that every described step was played.
        transcript.finish();

        Point::new(challenges)
    }
}

impl<F: Field, EF: ExtensionField<F>, R> ColumnHandoff<F, EF, R> {
    /// Number of variables addressing a column slot.
    fn num_slot_variables(&self) -> usize {
        self.num_variables - self.tables[0].num_variables()
    }

    /// Builds the prover over the column slots from the fully bound row prover.
    ///
    /// ```text
    ///     evals[slot(c)]   = P_c(r)
    ///     weights[slot(c)] = s_c * T(r)
    /// ```
    #[tracing::instrument(skip_all)]
    fn into_prover(self, rows: &mut ReprSumcheckProver<F, EF, R>) -> ReprSumcheckProver<F, EF, R>
    where
        R: IntoTranscriptField<EF> + Algebra<F>,
    {
        let row_variables = self.tables[0].num_variables();
        let num_slot_variables = self.num_slot_variables();

        // Applying the last row binding leaves the shared row weights at the row point.
        let row_weight = rows.weights().as_slice()[0];

        // Suffix rounds bind the last row variable first.
        let point = Point::new(self.challenges).reversed();
        let eq = R::from_table(Poly::new_from_point(point.as_slice(), EF::ONE).into_evals());

        // Every column slot holds its column at the row point.
        let mut evals = Poly::<R>::zero(num_slot_variables);
        column_slots(
            &self.placements,
            &self.tables,
            row_variables,
            evals.as_mut_slice(),
        )
        .into_par_iter()
        .for_each(|(slot, table_idx, poly_idx)| {
            let column = self.tables[table_idx].poly(poly_idx);
            slot[0] = weighted_sum(&eq, column.as_slice());
        });

        // Every column slot weighs in with its scale on the shared row weights.
        let mut weights = Poly::<R>::zero(num_slot_variables);
        for (slot, table_idx, poly_idx) in column_slots(
            &self.placements,
            &self.tables,
            row_variables,
            weights.as_mut_slice(),
        ) {
            slot[0] = self.scales[table_idx][poly_idx] * row_weight;
        }

        ReprSumcheckProver::from_repr_tables(
            VariableOrder::Suffix,
            evals,
            weights,
            rows.claimed_sum(),
        )
    }
}

/// Sums every column times its scale, row by row.
///
/// ```text
///     A(x) = sum_c  scale_c * P_c(x)
/// ```
///
/// # Performance
///
/// - A column with a zero scale is skipped.
/// - A zero row costs nothing and a one row adds the scale, so a column of bits costs no product.
fn aggregate_columns<F: Field, R: Field + Algebra<F>>(
    tables: &[Table<F>],
    scales: &[Vec<R>],
    len: usize,
) -> Vec<R> {
    let columns: Vec<(&[F], R)> = tables
        .iter()
        .zip(scales)
        .flat_map(|(table, scales)| table.iter_polys().zip(scales.iter().copied()))
        .filter(|&(_, scale)| scale != R::ZERO)
        .collect();

    let mut aggregate = R::zero_vec(len);

    // Enough row blocks to occupy every thread: each block adds its columns in place.
    if len.div_ceil(AGGREGATE_BLOCK) >= current_num_threads() {
        aggregate
            .par_chunks_mut(AGGREGATE_BLOCK)
            .enumerate()
            .for_each(|(block, out)| {
                let rows = block * AGGREGATE_BLOCK..block * AGGREGATE_BLOCK + out.len();
                for &(column, scale) in &columns {
                    add_scaled_column(out, &column[rows.clone()], scale);
                }
            });
        return aggregate;
    }

    // Fewer blocks than threads: each block also sums its columns in parallel.
    aggregate
        .par_chunks_mut(AGGREGATE_BLOCK)
        .enumerate()
        .for_each(|(block, out)| {
            let len = out.len();
            let rows = block * AGGREGATE_BLOCK..block * AGGREGATE_BLOCK + len;
            let sum = columns.par_iter().par_fold_reduce(
                || R::zero_vec(len),
                |mut acc, &(column, scale)| {
                    add_scaled_column(&mut acc, &column[rows.clone()], scale);
                    acc
                },
                |mut acc, other| {
                    acc.iter_mut()
                        .zip(other)
                        .for_each(|(acc, other)| *acc += other);
                    acc
                },
            );
            out.copy_from_slice(&sum);
        });
    aggregate
}

/// Adds `scale * column` into `acc`, adding the scale directly on a one row.
fn add_scaled_column<F: Field, R: Field + Algebra<F>>(acc: &mut [R], column: &[F], scale: R) {
    for (acc, &value) in acc.iter_mut().zip(column) {
        if value == F::ONE {
            *acc += scale;
        } else if value != F::ZERO {
            *acc += scale * value;
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use p3_baby_bear::BabyBear;
    use p3_binary_field::{BinaryField128, Ghash128};
    use p3_field::extension::BinomialExtensionField;
    use p3_matrix::dense::RowMajorMatrix;
    use rand::distr::{Distribution, StandardUniform};
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;

    /// Checks the aggregate against a row-by-row sum over two tables of mixed bit and arbitrary
    /// cells, with the first column of each table left unopened.
    ///
    /// Heights fall on both sides of the block count that occupies every thread, so both the
    /// in-place and the column-parallel block sums are exercised.
    fn assert_aggregate_matches_row_sums<F, R>(seed: u64)
    where
        F: Field,
        R: Field + Algebra<F>,
        StandardUniform: Distribution<F> + Distribution<R>,
    {
        let mut rng = SmallRng::seed_from_u64(seed);
        for (row_variables, width) in [(0, 3), (3, 5), (13, 1), (13, 6), (16, 2)] {
            let len = 1 << row_variables;
            let tables: Vec<Table<F>> = (0..2)
                .map(|_| {
                    let values = (0..width * len)
                        .map(|cell| {
                            if cell % 3 == 0 {
                                rng.random()
                            } else {
                                F::from_bool(rng.random_bool(0.5))
                            }
                        })
                        .collect();
                    Table::new(RowMajorMatrix::new(values, len))
                })
                .collect();
            let scales: Vec<Vec<R>> = (0..2)
                .map(|_| {
                    (0..width)
                        .map(|column| if column == 0 { R::ZERO } else { rng.random() })
                        .collect()
                })
                .collect();

            let expected: Vec<R> = (0..len)
                .map(|row| {
                    tables
                        .iter()
                        .zip(&scales)
                        .flat_map(|(table, scales)| {
                            table
                                .iter_polys()
                                .zip(scales)
                                .map(move |(column, &scale)| scale * column[row])
                        })
                        .sum()
                })
                .collect();
            assert_eq!(
                aggregate_columns(&tables, &scales, len),
                expected,
                "row_variables={row_variables}, width={width}"
            );
        }
    }

    #[test]
    fn aggregate_matches_row_sums_over_binary_field() {
        assert_aggregate_matches_row_sums::<BinaryField128, Ghash128>(1);
    }

    #[test]
    fn aggregate_matches_row_sums_over_extension_of_prime_field() {
        assert_aggregate_matches_row_sums::<BabyBear, BinomialExtensionField<BabyBear, 4>>(2);
    }
}
