//! Rounds of a lone column's residual sumcheck, played over bank sums of the column.
//!
//! # Overview
//!
//! One column filling the whole stacked space, weighed by a batch of equality claims:
//!
//! ```text
//!     W(x) = sum_c  gamma_c * eq(z_c, x)
//! ```
//!
//! Suffix binding fixes the lowest index bits first, and the rounds run in stages of `h`.
//! A stage splits every index into its high part `u` and its lowest `h` bits `b`, and each point
//! to match:
//!
//! ```text
//!     eq(z_c, (u, b)) = eq(hi_c, u) * eq(lo_c, b)
//!     S_c[b]          = sum_u  eq(hi_c, u) * P(u, b)
//! ```
//!
//! Binding is linear and never touches `u`. So each round of a stage measures the same message
//! over the banks `S_c`, weighed by `gamma_c * eq(lo_c, .)`, as over the whole column weighed by
//! `W`. The rounds therefore run over `2^h` entries per claim.
//!
//! Once a stage is played, one pass binds the column at its challenges `r`, and every claim keeps
//! its high coordinates under a new coefficient:
//!
//! ```text
//!     P'(u)    = sum_b  eq(r, b) * P(u, b)
//!     gamma'_c = gamma_c * eq(lo_c, r)
//! ```
//!
//! The next stage runs over `P'`, which is `2^h` times shorter and already in the round
//! representation. No weight table is ever materialized.

use alloc::borrow::Cow;
use alloc::vec::Vec;
use core::array;
use core::ops::Range;

use p3_field::{ExtensionField, Field, PackedField, PackedValue, PrimeCharacteristicRing};
use p3_maybe_rayon::prelude::*;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use p3_util::log2_strict_usize;

use crate::layout::witness::Table;
use crate::strategy::{FromTable, IntoTranscriptField, ReprSumcheckProver, VariableOrder};

/// Rounds one stage plays over bank sums before the column is bound.
pub(super) const STAGE_ROUNDS: usize = 4;

/// Rows of the column one task reads, and converts into the round representation, at a time.
const BLOCK_ROWS: usize = 1 << 7;

/// The column a banked prover plays its rounds over, and every claim's unbound coordinates.
#[derive(Debug, Clone)]
pub(super) struct BankedColumn<F: Field, R> {
    /// The lone source table, one column over the whole stacked space, until the first bind.
    table: Option<Table<F>>,
    /// The column bound at every challenge of the stages bound so far, once one is.
    bound: Option<Vec<R>>,
    /// Per claim, the coordinates of its point that no stage has reached yet.
    points: Vec<Point<R>>,
    /// Per claim, its batching coefficient times its equality weight at every bound challenge.
    scales: Vec<R>,
    /// Variables the stage prover carries to address a claim.
    claim_variables: usize,
    /// Rounds a full stage plays.
    stage_rounds: usize,
    /// Challenges of the stage being played, in round order.
    challenges: Vec<R>,
}

impl<F: Field, R: Field> BankedColumn<F, R> {
    /// Starts the banked rounds, returning the first stage's prover and the column behind it.
    ///
    /// # Arguments
    ///
    /// - `table` — one dense column filling the stacked space, of at least one variable.
    /// - `claims` — every claim's point and batching coefficient, at least one.
    /// - `sum` — the batched claim.
    /// - `stage_rounds` — rounds a full stage plays, at least one.
    ///
    /// # Panics
    ///
    /// On a table other than one dense column of at least one variable, on no claim, or on a
    /// stage of no round.
    pub(super) fn new<EF>(
        table: Table<F>,
        claims: &[(Point<EF>, EF)],
        sum: EF,
        stage_rounds: usize,
    ) -> (ReprSumcheckProver<F, EF, R>, Self)
    where
        EF: ExtensionField<F>,
        R: IntoTranscriptField<EF>,
    {
        assert_eq!(table.num_polys(), 1, "a banked prover weighs one column");
        assert!(
            table.num_variables() > 0,
            "a banked column has a round to play"
        );
        assert!(!claims.is_empty(), "a banked column is weighed by a claim");
        assert!(stage_rounds > 0, "a stage plays a round");

        let (points, scales) = claims
            .iter()
            .map(|(point, scale)| {
                let point = Point::new(point.iter().copied().map(R::from).collect());
                (point, R::from(*scale))
            })
            .unzip();
        let mut column = Self {
            table: Some(table),
            bound: None,
            points,
            scales,
            claim_variables: log2_strict_usize(claims.len().next_power_of_two()),
            stage_rounds,
            challenges: Vec::with_capacity(stage_rounds),
        };
        let prover = column.stage(sum);
        (prover, column)
    }

    /// Variables still to bind, given the stage prover currently being played.
    pub(super) fn num_variables<EF>(&self, stage: &ReprSumcheckProver<F, EF, R>) -> usize
    where
        EF: ExtensionField<F>,
        R: IntoTranscriptField<EF>,
    {
        // The claim variables are summed over, never bound.
        stage.num_variables() - self.claim_variables + self.points[0].num_variables()
    }

    /// Whether every round of the current stage has been played.
    pub(super) fn stage_played<EF>(&self, stage: &ReprSumcheckProver<F, EF, R>) -> bool
    where
        EF: ExtensionField<F>,
        R: IntoTranscriptField<EF>,
    {
        stage.num_variables() == self.claim_variables
    }

    /// Records a challenge of the current stage.
    pub(super) fn push_challenge(&mut self, challenge: R) {
        self.challenges.push(challenge);
    }

    /// Binds the column at the played stage's challenges, then opens the next stage over it.
    ///
    /// # Panics
    ///
    /// The current stage is not fully played, or no variable is left for another stage.
    pub(super) fn next_stage<EF>(
        &mut self,
        stage: &mut ReprSumcheckProver<F, EF, R>,
    ) -> ReprSumcheckProver<F, EF, R>
    where
        EF: ExtensionField<F>,
        R: IntoTranscriptField<EF>,
    {
        self.bind(stage);
        self.stage(stage.claimed_sum())
    }

    /// The column bound at every challenge played so far, binding the played stage's first.
    ///
    /// # Panics
    ///
    /// The current stage is not fully played.
    pub(super) fn bound<EF>(&mut self, stage: &mut ReprSumcheckProver<F, EF, R>) -> &[R]
    where
        EF: ExtensionField<F>,
        R: IntoTranscriptField<EF>,
    {
        self.bind(stage);
        self.bound
            .as_deref()
            .expect("a played stage leaves a bound column")
    }

    /// Binds the column at the played stage's challenges, unless it already is.
    ///
    /// Each claim's coefficient takes the stage prover's settled weight: its equality weight
    /// over the stage's variables, at their challenges.
    #[tracing::instrument(skip_all, level = "debug")]
    fn bind<EF>(&mut self, stage: &mut ReprSumcheckProver<F, EF, R>)
    where
        EF: ExtensionField<F>,
        R: IntoTranscriptField<EF>,
    {
        assert!(self.stage_played(stage), "a stage binds once it is played");
        if self.challenges.is_empty() {
            return;
        }

        let weights = stage.weights();
        let num_claims = self.scales.len();
        self.scales
            .copy_from_slice(&weights.as_slice()[..num_claims]);

        // Suffix rounds bind the lowest bit first, so the first challenge names it.
        let point = Point::new(core::mem::take(&mut self.challenges)).reversed();
        let eq = Poly::new_from_point(point.as_slice(), R::ONE);
        let bound = self.bound.as_deref().map_or_else(
            || {
                // No stage reads the source column once it is bound, so it is released here.
                let table = self
                    .table
                    .take()
                    .expect("the source column binds only once");
                bind_column::<F, EF, R>(dense_column(&table), eq.as_slice())
            },
            |bound| bind_repr(bound, eq.as_slice()),
        );
        self.bound = Some(bound);
    }

    /// Opens a stage over the column as currently bound, with `sum` the claim it carries.
    #[tracing::instrument(skip_all, level = "debug")]
    fn stage<EF>(&mut self, sum: EF) -> ReprSumcheckProver<F, EF, R>
    where
        EF: ExtensionField<F>,
        R: IntoTranscriptField<EF>,
    {
        let unbound = self.points[0].num_variables();
        let rounds = self.stage_rounds.min(unbound);
        assert!(
            rounds > 0,
            "a stage opens only over a variable left to bind"
        );

        // Each point splits where the stage starts: the high part weighs the rows of a bank.
        let (points, low): (Vec<_>, Vec<_>) = self
            .points
            .iter()
            .map(|point| point.split_at(unbound - rounds))
            .unzip();
        self.points = points;
        let banks = self.bound.as_deref().map_or_else(
            || {
                let table = self
                    .table
                    .as_ref()
                    .expect("an unbound column is still held");
                bank_sums::<F, EF, R>(dense_column(table), &self.points, rounds)
            },
            |bound| bank_sums_repr(bound, &self.points, rounds),
        );

        // Claim `c` owns entries `c * 2^rounds ..`, so suffix rounds bind its bank index first.
        let len = 1usize << (rounds + self.claim_variables);
        let mut evals = R::zero_vec(len);
        evals[..banks.len()].copy_from_slice(&banks);
        let mut weights = R::zero_vec(len);
        for ((low, &scale), slot) in low
            .iter()
            .zip(&self.scales)
            .zip(weights.chunks_exact_mut(1 << rounds))
        {
            slot.copy_from_slice(Poly::new_from_point(low.as_slice(), scale).as_slice());
        }

        ReprSumcheckProver::from_repr_tables(
            VariableOrder::Suffix,
            Poly::new(evals),
            Poly::new(weights),
            sum,
        )
    }
}

/// The lone column of a banked table.
fn dense_column<F: Field>(table: &Table<F>) -> &[F] {
    table
        .column(0)
        .as_dense()
        .expect("a banked column is held densely")
}

/// A run of source values, crossed into `R` in one conversion.
fn to_repr<F, EF, R>(values: &[F]) -> Vec<R>
where
    F: Field,
    EF: ExtensionField<F>,
    R: FromTable<EF>,
{
    R::from_table(values.iter().map(|&value| EF::from(value)).collect())
}

/// Every claim's bank sums, `2^banks` per claim, claim after claim.
///
/// ```text
///     S_c[b] = sum_u  eq(hi_c, u) * P(u, b)
/// ```
///
/// # Performance
///
/// - One pass over the column, a block of rows at a time.
/// - Each block crosses into `R` in one conversion, then weighs every claim.
/// - Each claim's row weights split at the block, so no table spans the rows.
fn bank_sums<F, EF, R>(column: &[F], high_points: &[Point<R>], banks: usize) -> Vec<R>
where
    F: Field,
    EF: ExtensionField<F>,
    R: Field + FromTable<EF>,
{
    bank_sums_over(column.len(), high_points, banks, |rows| {
        Cow::Owned(to_repr::<F, EF, R>(&column[rows]))
    })
}

/// [`bank_sums`] over a column already in `R`, read in place.
fn bank_sums_repr<R: Field>(column: &[R], high_points: &[Point<R>], banks: usize) -> Vec<R> {
    bank_sums_over(column.len(), high_points, banks, |rows| {
        Cow::Borrowed(&column[rows])
    })
}

/// [`bank_sums`] over a column of `len` entries, whose rows `block` hands out in `R`.
fn bank_sums_over<'a, R: Field>(
    len: usize,
    high_points: &[Point<R>],
    banks: usize,
    block: impl Fn(Range<usize>) -> Cow<'a, [R]> + Sync,
) -> Vec<R> {
    // A row of banks narrower than the packing is weighed one lane at a time.
    if (1usize << banks).is_multiple_of(R::Packing::WIDTH) {
        bank_sums_in::<R, R::Packing>(len, high_points, banks, block)
    } else {
        bank_sums_in::<R, R>(len, high_points, banks, block)
    }
}

/// [`bank_sums_over`] over lanes `P`, whose width divides `2^banks`.
fn bank_sums_in<'a, R, P>(
    len: usize,
    high_points: &[Point<R>],
    banks: usize,
    block: impl Fn(Range<usize>) -> Cow<'a, [R]> + Sync,
) -> Vec<R>
where
    R: Field,
    P: PackedField<Scalar = R>,
{
    let groups = (1usize << banks) / P::WIDTH;
    let block_rows = BLOCK_ROWS.min(len >> banks);
    let block_variables = log2_strict_usize(block_rows);
    let block_len = block_rows << banks;

    // Each claim's row weight splits into one factor per block and one per row inside it.
    let weights: Vec<(Vec<R>, Vec<P>)> = high_points
        .iter()
        .map(|point| {
            let (outer, inner) = point.split_at(point.num_variables() - block_variables);
            let outer = Poly::new_from_point(outer.as_slice(), R::ONE).into_evals();
            let inner = Poly::new_from_point(inner.as_slice(), R::ONE)
                .iter()
                .map(|&weight| P::from(weight))
                .collect();
            (outer, inner)
        })
        .collect();

    let sums = (0..len / block_len).into_par_iter().par_fold_reduce(
        || P::zero_vec(high_points.len() * groups),
        |mut sums, index| {
            let values = block(index * block_len..(index + 1) * block_len);
            let rows = P::pack_slice(&values);
            for ((outer, inner), sums) in weights.iter().zip(sums.chunks_exact_mut(groups)) {
                let scale = P::from(outer[index]);
                for (group, sum) in sums.iter_mut().enumerate() {
                    *sum += scale * dot::<_, 8>(inner, |row| rows[row * groups + group]);
                }
            }
            sums
        },
        |mut sums, other| {
            sums.iter_mut()
                .zip(other)
                .for_each(|(sum, other)| *sum += other);
            sums
        },
    );
    P::unpack_slice(&sums).to_vec()
}

/// The column bound at a stage's challenges, one entry per row of banks.
///
/// ```text
///     out[u] = sum_b  eq[b] * P(u, b)
/// ```
fn bind_column<F, EF, R>(column: &[F], eq: &[R]) -> Vec<R>
where
    F: Field,
    EF: ExtensionField<F>,
    R: Field + FromTable<EF>,
{
    bind_column_over(column.len(), eq, |rows| {
        Cow::Owned(to_repr::<F, EF, R>(&column[rows]))
    })
}

/// [`bind_column`] over a column already in `R`, read in place.
fn bind_repr<R: Field>(column: &[R], eq: &[R]) -> Vec<R> {
    bind_column_over(column.len(), eq, |rows| Cow::Borrowed(&column[rows]))
}

/// [`bind_column`] over a column of `len` entries, whose rows `block` hands out in `R`.
fn bind_column_over<'a, R: Field>(
    len: usize,
    eq: &[R],
    block: impl Fn(Range<usize>) -> Cow<'a, [R]> + Sync,
) -> Vec<R> {
    // A row of banks narrower than the packing is bound one lane at a time.
    if eq.len().is_multiple_of(R::Packing::WIDTH) {
        bind_column_in::<R, R::Packing>(len, eq, block)
    } else {
        bind_column_in::<R, R>(len, eq, block)
    }
}

/// [`bind_column_over`] over lanes `P`, whose width divides the length of `eq`.
fn bind_column_in<'a, R, P>(
    len: usize,
    eq: &[R],
    block: impl Fn(Range<usize>) -> Cow<'a, [R]> + Sync,
) -> Vec<R>
where
    R: Field,
    P: PackedField<Scalar = R>,
{
    let banks = eq.len();
    let groups = banks / P::WIDTH;
    let eq = P::pack_slice(eq);

    let mut bound = R::zero_vec(len / banks);
    bound
        .par_chunks_mut(BLOCK_ROWS)
        .enumerate()
        .for_each(|(index, bound)| {
            let first = index * BLOCK_ROWS * banks;
            let values = block(first..first + bound.len() * banks);
            let rows = P::pack_slice(&values).chunks_exact(groups);
            for (slot, row) in bound.iter_mut().zip(rows) {
                *slot = dot::<_, 4>(eq, |group| row[group])
                    .as_slice()
                    .iter()
                    .copied()
                    .sum();
            }
        });
    bound
}

/// `sum_i weights[i] * value(i)`, reducing once per `N` products.
#[inline]
fn dot<P, const N: usize>(weights: &[P], value: impl Fn(usize) -> P) -> P
where
    P: PrimeCharacteristicRing + Copy,
{
    let (chunks, rest) = weights.as_chunks::<N>();
    let mut sum = P::ZERO;
    for (chunk_idx, chunk) in chunks.iter().enumerate() {
        let values = array::from_fn(|i| value(chunk_idx * N + i));
        sum += P::dot_product(chunk, &values);
    }
    let offset = chunks.len() * N;
    for (i, &weight) in rest.iter().enumerate() {
        sum += weight * value(offset + i);
    }
    sum
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use p3_baby_bear::BabyBear;
    use p3_binary_field::{BinaryField128, Ghash128};
    use p3_field::extension::BinomialExtensionField;
    use p3_field::{ExtensionField, Field};
    use p3_multilinear_util::point::Point;
    use p3_multilinear_util::poly::Poly;
    use rand::distr::{Distribution, StandardUniform};
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::{BLOCK_ROWS, bank_sums, bank_sums_repr, bind_column, bind_repr};
    use crate::strategy::FromTable;

    /// Degree-4 binomial extension of BabyBear.
    type BabyBearExt4 = BinomialExtensionField<BabyBear, 4>;

    /// Column arities around the block size: one partial block, one block, several blocks.
    const ARITIES: [usize; 5] = [2, 4, 7, 11, 13];

    /// Stage depths from one bank pair, narrower than any wide packing, to a full stage.
    const DEPTHS: [usize; 4] = [1, 2, 3, 4];

    /// Checks both kernels against a row-by-row reference over random columns and points.
    ///
    /// The reference sums each bank directly over its rows, and binds the column one suffix
    /// variable at a time, the way the dense route does.
    ///
    /// Each kernel runs twice: over the source column, converting it block by block, and over
    /// the same column already in `R`, read in place as every later stage reads it.
    fn assert_kernels_match_reference<F, EF, R>(seed: u64)
    where
        F: Field,
        EF: ExtensionField<F>,
        R: Field + FromTable<EF>,
        StandardUniform: Distribution<F> + Distribution<R>,
    {
        let mut rng = SmallRng::seed_from_u64(seed);
        for num_variables in ARITIES {
            for depth in DEPTHS.into_iter().filter(|&depth| depth <= num_variables) {
                let banks = 1usize << depth;
                let column: Vec<F> = (0..1 << num_variables).map(|_| rng.random()).collect();
                let lifted: Vec<R> = R::from_table(column.iter().map(|&v| EF::from(v)).collect());
                let shape = alloc::format!("num_variables={num_variables}, depth={depth}");

                // Three claims, so the per-claim sums sit side by side.
                let points: Vec<Point<R>> = (0..3)
                    .map(|_| Point::rand(&mut rng, num_variables - depth))
                    .collect();
                let expected: Vec<R> = points
                    .iter()
                    .flat_map(|point| {
                        let eq = Poly::new_from_point(point.as_slice(), R::ONE);
                        let lifted = &lifted;
                        (0..banks).map(move |bank| {
                            eq.iter()
                                .enumerate()
                                .map(|(row, &weight)| weight * lifted[row * banks + bank])
                                .sum::<R>()
                        })
                    })
                    .collect();
                assert_eq!(
                    bank_sums::<F, EF, R>(&column, &points, depth),
                    expected,
                    "{shape}"
                );
                assert_eq!(
                    bank_sums_repr(&lifted, &points, depth),
                    expected,
                    "{shape}, in place"
                );

                // Bind the lowest variable first, one challenge at a time.
                let challenges: Vec<R> = (0..depth).map(|_| rng.random()).collect();
                let mut reference = Poly::new(lifted.clone());
                for &challenge in &challenges {
                    reference.fix_suffix_var_mut(challenge);
                }
                let point = Point::new(challenges).reversed();
                let eq = Poly::new_from_point(point.as_slice(), R::ONE);
                assert_eq!(
                    bind_column::<F, EF, R>(&column, eq.as_slice()),
                    reference.as_slice(),
                    "{shape}"
                );
                assert_eq!(
                    bind_repr(&lifted, eq.as_slice()),
                    reference.as_slice(),
                    "{shape}, in place"
                );
            }
        }
        // The sweep reaches past one block, so more than one task holds rows.
        assert!(1 << (ARITIES[ARITIES.len() - 1] - DEPTHS[DEPTHS.len() - 1]) > BLOCK_ROWS);
    }

    #[test]
    fn kernels_match_reference_over_binary_field() {
        assert_kernels_match_reference::<BinaryField128, BinaryField128, Ghash128>(1);
    }

    #[test]
    fn kernels_match_reference_over_extension_of_prime_field() {
        assert_kernels_match_reference::<BabyBear, BabyBearExt4, BabyBearExt4>(2);
    }
}
