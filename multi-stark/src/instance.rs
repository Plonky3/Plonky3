//! Public instance wrappers for batched multi-STARK proving and verification.
//!
//! Setup takes only AIR references, since public values and main trace tables are proof-time data.
//! A proving instance adds the committed main trace table, the shared proving key, and public values.
//! A verifying instance adds the shared verifying key, the trace height, and public values.

use alloc::vec::Vec;
use core::ops::Deref;

use p3_air::BaseAir;
use p3_multilinear_util::point::Point;
use p3_sumcheck::layout::Table;
use p3_sumcheck::{OpeningBatch, OpeningProtocol, TableShape, TableSpec};

use crate::config::MultiStarkConfig;
pub use crate::keys::{ProvingKey, VerifyingKey, setup};
pub use crate::proof::MultiStarkProof;
pub use crate::prover::prove;
pub use crate::verifier::{VerificationError, verify};

/// What one committed batch of tables opens, and at which point.
///
/// A scheme takes two lists: the columns of every batch, and one point per batch.
///
/// Nothing in it ties an entry of one list to the entry beside it in the other.
///
/// Lists built apart can fall out of step.
///
/// A claim checked at another table's point still verifies.
///
/// So both come out of one pass here.
pub(super) struct OpeningSchedule<EF> {
    /// Shape agreement handed to the commitment scheme.
    protocol: OpeningProtocol,
    /// One point per batch, in the protocol's own opening order.
    points: Vec<Point<EF>>,
    /// Position of the table owning each batch, in the same order.
    owners: Vec<usize>,
}

impl<EF> OpeningSchedule<EF> {
    /// Lay out one entry per committed table, tables in commitment order.
    ///
    /// # Arguments
    ///
    /// - `tables`: each table's committed shape, and the batches it is opened in paired with
    ///   the point each is taken at.
    ///
    /// # Panics
    ///
    /// Panics if a table schedules no opening, which would commit columns nothing reads.
    pub(super) fn new<I>(tables: I) -> Self
    where
        I: IntoIterator<Item = (TableShape, Vec<(OpeningBatch<usize>, Point<EF>)>)>,
    {
        let mut specs = Vec::new();
        let mut points = Vec::new();
        let mut owners = Vec::new();

        // One walk over the tables fills all three lists.
        //
        // Their order then agrees by construction.
        //
        // Two functions keeping a convention is what it replaces.
        for (table, (shape, openings)) in tables.into_iter().enumerate() {
            assert!(
                !openings.is_empty(),
                "a committed table must be opened at least once"
            );

            let mut batches = Vec::with_capacity(openings.len());
            for (batch, point) in openings {
                batches.push(batch);
                points.push(point);
                owners.push(table);
            }
            specs.push(TableSpec::new(shape, batches));
        }

        Self {
            protocol: OpeningProtocol::new(specs),
            points,
            owners,
        }
    }

    /// Shape agreement to hand the commitment scheme.
    pub(super) const fn protocol(&self) -> &OpeningProtocol {
        &self.protocol
    }

    /// One point per batch, in the order the scheme walks the batches.
    pub(super) fn points(&self) -> &[Point<EF>] {
        &self.points
    }

    /// Open one batch per table: every column, at that table's own point.
    ///
    /// This is the shape every committed table has had since before a second point existed.
    ///
    /// # Arguments
    ///
    /// - `tables`: each table's row-count logarithm, width, successor-view columns, and point.
    pub(super) fn whole_tables<I>(tables: I) -> Self
    where
        I: IntoIterator<Item = (usize, usize, Vec<usize>, Point<EF>)>,
    {
        Self::new(
            tables
                .into_iter()
                .map(|(log_height, width, next_columns, point)| {
                    (
                        TableShape::new(log_height, width),
                        alloc::vec![(Self::whole_table_batch(width, next_columns), point)],
                    )
                }),
        )
    }

    /// The shape agreement of a whole-table opening, before any point is bound.
    ///
    /// Security assessment runs ahead of the proof and reads the shapes alone.
    ///
    /// It sits here so the two builders cannot disagree on what a table opens.
    ///
    /// # Arguments
    ///
    /// - `tables`: each table's row-count logarithm, width, and successor-view columns.
    pub(super) fn whole_table_protocol<I>(tables: I) -> OpeningProtocol
    where
        I: IntoIterator<Item = (usize, usize, Vec<usize>)>,
    {
        OpeningProtocol::new(
            tables
                .into_iter()
                .map(|(log_height, width, next_columns)| {
                    TableSpec::new(
                        TableShape::new(log_height, width),
                        alloc::vec![Self::whole_table_batch(width, next_columns)],
                    )
                })
                .collect(),
        )
    }

    /// The batch reading a table's whole width, plus its successor-view columns.
    fn whole_table_batch(width: usize, next_columns: Vec<usize>) -> OpeningBatch<usize> {
        OpeningBatch::new((0..width).collect::<Vec<_>>(), next_columns)
    }

    /// Where each table's own columns land among the per-batch results, in table order.
    ///
    /// A table's columns are opened in the first batch it owns.
    pub(super) fn first_batch_per_table(&self) -> Vec<usize> {
        let mut first = Vec::new();

        // Owners are non-decreasing.
        //
        // A batch starts a table when its owner is the first one not yet recorded.
        for (batch, &owner) in self.owners.iter().enumerate() {
            if first.len() == owner {
                first.push(batch);
            }
        }
        first
    }
}

/// One AIR statement proved inside a batched committed proof.
pub struct ProverInstance<'a, C, A>
where
    C: MultiStarkConfig,
{
    /// AIR whose constraints are proved.
    air: &'a A,
    /// Already-transposed trace table: one row per AIR column.
    table: Table<C::Val>,
    /// Proving key carrying optional reusable preprocessed data.
    proving_key: &'a ProvingKey<C>,
    /// Public values forwarded to the AIR.
    public_values: &'a [C::Val],
}

/// Collection of prover-side AIR instances proved with one shared proof.
///
/// All contained instances must use the same proving key. Main trace tables are
/// committed in this collection's order.
pub struct ProverInstances<'a, C, A>(Vec<ProverInstance<'a, C, A>>)
where
    C: MultiStarkConfig;

/// One AIR statement checked inside a batched proof.
pub struct VerifierInstance<'a, C, A>
where
    C: MultiStarkConfig,
{
    /// AIR whose constraints are checked.
    air: &'a A,
    /// Verifying key carrying optional reusable preprocessed data.
    verifying_key: &'a VerifyingKey<C>,
    /// Base-two logarithm of this instance's main trace height.
    num_variables: usize,
    /// Public values forwarded to the AIR.
    public_values: &'a [C::Val],
}

/// Collection of verifier-side AIR instances checked against one shared proof.
///
/// All contained instances must use the same verifying key. The order must match
/// the prover-side instance order used to create the proof.
pub struct VerifierInstances<'a, C, A>(Vec<VerifierInstance<'a, C, A>>)
where
    C: MultiStarkConfig;

pub(super) struct Instance<'a, C, A>
where
    C: MultiStarkConfig,
{
    /// AIR whose constraints are proved.
    pub(super) air: &'a A,
    /// Public values forwarded to the AIR.
    pub(super) public_values: &'a [C::Val],
    /// Base-two logarithm of this instance's main trace height.
    num_variables: usize,
}

pub(super) struct Instances<'a, C, A>(Vec<Instance<'a, C, A>>)
where
    C: MultiStarkConfig;

/// A prover-side batch split into the parts the proving flow consumes.
pub(super) struct ProverParts<'a, C, A>
where
    C: MultiStarkConfig,
{
    /// Proving key shared by every instance in the batch.
    pub(super) proving_key: &'a ProvingKey<C>,
    /// Owned main trace tables, in batch order, ready to commit.
    pub(super) tables: Vec<Table<C::Val>>,
    /// Per-instance metadata retained for the zerocheck and openings.
    pub(super) instances: Instances<'a, C, A>,
}

impl<'a, C, A> ProverInstance<'a, C, A>
where
    C: MultiStarkConfig,
{
    /// Create a prover-side AIR instance from proof-time data.
    ///
    /// The table must already be transposed into the multilinear layout: one row
    /// per AIR column.
    pub const fn new(
        air: &'a A,
        table: Table<C::Val>,
        proving_key: &'a ProvingKey<C>,
        public_values: &'a [C::Val],
    ) -> Self {
        Self {
            air,
            table,
            proving_key,
            public_values,
        }
    }
}

impl<'a, C, A> VerifierInstance<'a, C, A>
where
    C: MultiStarkConfig,
{
    /// Create a verifier-side AIR instance from verification-time data.
    ///
    /// `num_variables` is the base-two logarithm of this instance's main trace height.
    pub const fn new(
        air: &'a A,
        verifying_key: &'a VerifyingKey<C>,
        num_variables: usize,
        public_values: &'a [C::Val],
    ) -> Self {
        Self {
            air,
            verifying_key,
            num_variables,
            public_values,
        }
    }
}

impl<'a, C, A> Deref for ProverInstances<'a, C, A>
where
    C: MultiStarkConfig,
{
    type Target = [ProverInstance<'a, C, A>];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'a, C, A> Deref for VerifierInstances<'a, C, A>
where
    C: MultiStarkConfig,
{
    type Target = [VerifierInstance<'a, C, A>];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'a, C, A> Deref for Instances<'a, C, A>
where
    C: MultiStarkConfig,
{
    type Target = [Instance<'a, C, A>];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'a, C, A> ProverInstances<'a, C, A>
where
    C: MultiStarkConfig,
{
    pub(super) fn statement(&self) -> Instances<'a, C, A> {
        Instances(
            self.0
                .iter()
                .map(|instance| Instance {
                    air: instance.air,
                    public_values: instance.public_values,
                    num_variables: instance.table.num_variables(),
                })
                .collect(),
        )
    }

    /// Create a prover-side batch in proof order.
    ///
    /// The order must match the AIR order used at setup.
    /// Main and preprocessed tables are committed and opened in this order.
    pub const fn new(instances: Vec<ProverInstance<'a, C, A>>) -> Self {
        Self(instances)
    }

    pub(super) fn proving_key(&self) -> &'a ProvingKey<C> {
        // Every instance must reference one shared proving key.
        // Adjacent-pair equality is transitively all-equal.
        assert!(
            self.0
                .windows(2)
                .all(|pair| core::ptr::eq(pair[0].proving_key, pair[1].proving_key)),
            "all prover instances must use the same proving key"
        );
        self.0
            .first()
            .expect("prover instances cannot be empty")
            .proving_key
    }

    pub(super) fn into_parts(self) -> ProverParts<'a, C, A> {
        let proving_key = self.proving_key();

        // One pass over the owned instances.
        // Read the borrowed metadata, then move the table out.
        let mut instances = Vec::with_capacity(self.0.len());
        let mut tables = Vec::with_capacity(self.0.len());
        for instance in self.0 {
            instances.push(Instance {
                air: instance.air,
                public_values: instance.public_values,
                num_variables: instance.table.num_variables(),
            });
            tables.push(instance.table);
        }

        ProverParts {
            proving_key,
            tables,
            instances: Instances(instances),
        }
    }
}

impl<'a, C, A> VerifierInstances<'a, C, A>
where
    C: MultiStarkConfig,
{
    pub(super) fn statement(&self) -> Instances<'a, C, A> {
        Instances(
            self.0
                .iter()
                .map(|instance| Instance {
                    air: instance.air,
                    public_values: instance.public_values,
                    num_variables: instance.num_variables,
                })
                .collect(),
        )
    }

    /// Create a verifier-side batch in proof order.
    ///
    /// The order must match the prover-side batch order and the AIR order used at setup.
    /// Openings are replayed in this order.
    pub const fn new(instances: Vec<VerifierInstance<'a, C, A>>) -> Self {
        Self(instances)
    }

    pub(super) fn verifying_key(&self) -> &'a VerifyingKey<C> {
        // Every instance must reference one shared verifying key.
        // Adjacent-pair equality is transitively all-equal.
        assert!(
            self.0
                .windows(2)
                .all(|pair| core::ptr::eq(pair[0].verifying_key, pair[1].verifying_key)),
            "all verifier instances must use the same verifying key"
        );
        self.0
            .first()
            .expect("verifier instances cannot be empty")
            .verifying_key
    }

    pub(super) fn into_parts(self) -> (&'a VerifyingKey<C>, Instances<'a, C, A>) {
        let verifying_key = self.verifying_key();
        let instances = self
            .0
            .into_iter()
            .map(|instance| Instance {
                air: instance.air,
                public_values: instance.public_values,
                num_variables: instance.num_variables,
            })
            .collect();

        (verifying_key, Instances(instances))
    }
}

impl<'a, C, A> Instances<'a, C, A>
where
    C: MultiStarkConfig,
    A: BaseAir<C::Val>,
{
    pub(super) fn num_variables(&self) -> Vec<usize> {
        self.0
            .iter()
            .map(|instance| instance.num_variables)
            .collect()
    }

    pub(super) fn widths(&self) -> Vec<usize> {
        self.0.iter().map(|instance| instance.air.width()).collect()
    }

    pub(super) fn next_columns(&self) -> Vec<Vec<usize>> {
        self.0
            .iter()
            .map(|instance| instance.air.main_next_row_columns())
            .collect()
    }

    pub(super) fn airs(&self) -> Vec<&A> {
        self.0
            .iter()
            .map(|instance| instance.air)
            .collect::<Vec<_>>()
    }

    pub(super) fn public_values(&self) -> Vec<&[C::Val]> {
        self.0
            .iter()
            .map(|instance| instance.public_values)
            .collect()
    }

    /// Shape agreement for the main trace opening, without the points.
    pub(super) fn opening_protocol(&self) -> OpeningProtocol {
        OpeningSchedule::<C::Challenge>::whole_table_protocol(
            self.num_variables()
                .iter()
                .zip(self.widths().iter())
                .zip(self.next_columns())
                .map(|((&log_height, &width), next_columns)| (log_height, width, next_columns)),
        )
    }

    /// Shape agreement for the preprocessed trace opening, without the points.
    pub(super) fn preprocessed_opening_protocol(&self) -> OpeningProtocol {
        OpeningSchedule::<C::Challenge>::whole_table_protocol(
            self.iter()
                .filter(|instance| instance.air.preprocessed_width() != 0)
                .map(|instance| {
                    (
                        instance.num_variables,
                        instance.air.preprocessed_width(),
                        instance.air.preprocessed_next_row_columns(),
                    )
                }),
        )
    }

    /// Schedule the main trace opening at the bound point.
    ///
    /// # Panics
    ///
    /// Panics if the bound point does not cover the tallest trace.
    pub(super) fn main_schedule(
        &self,
        point: &Point<C::Challenge>,
    ) -> OpeningSchedule<C::Challenge> {
        OpeningSchedule::whole_tables(
            self.num_variables()
                .iter()
                .zip(self.widths().iter())
                .zip(self.next_columns())
                .map(|((&log_height, &width), next_columns)| {
                    (
                        log_height,
                        width,
                        next_columns,
                        self.trace_suffix(point, log_height),
                    )
                }),
        )
    }

    /// Schedule the preprocessed trace opening at the bound point.
    ///
    /// AIRs declaring no preprocessed columns commit nothing and are skipped.
    ///
    /// # Panics
    ///
    /// Panics if the bound point does not cover the tallest trace.
    pub(super) fn preprocessed_schedule(
        &self,
        point: &Point<C::Challenge>,
    ) -> OpeningSchedule<C::Challenge> {
        OpeningSchedule::whole_tables(
            self.iter()
                .filter(|instance| instance.air.preprocessed_width() != 0)
                .map(|instance| {
                    (
                        instance.num_variables,
                        instance.air.preprocessed_width(),
                        instance.air.preprocessed_next_row_columns(),
                        self.trace_suffix(point, instance.num_variables),
                    )
                }),
        )
    }

    pub(super) fn preprocessed_next_columns(&self) -> Vec<Vec<usize>> {
        self.iter()
            .filter(|instance| instance.air.preprocessed_width() != 0)
            .map(|instance| instance.air.preprocessed_next_row_columns())
            .collect()
    }

    pub(super) fn max_num_variables(&self) -> usize {
        self.num_variables().iter().cloned().max().unwrap()
    }

    /// Cut the row coordinates of one table out of a bound point.
    ///
    /// The lookup reduction may add leading block-selector coordinates.
    ///
    /// A short table also carries fewer row coordinates than a tall one.
    ///
    /// Only the trailing coordinates addressing this table's rows are opened.
    ///
    /// # Panics
    ///
    /// Panics if the bound point does not cover the tallest trace.
    fn trace_suffix(
        &self,
        point: &Point<C::Challenge>,
        num_variables: usize,
    ) -> Point<C::Challenge> {
        assert!(
            point.num_variables() >= self.max_num_variables(),
            "the bound point must cover the tallest trace"
        );
        point.split_at(point.num_variables() - num_variables).1
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;

    use super::*;

    type F = BabyBear;

    /// A point whose single coordinate labels it, so a mismatch is visible in an assertion.
    fn labelled(label: u64) -> Point<F> {
        Point::new(vec![F::from_u64(label)])
    }

    /// Three tables owning one, two and one batch, each batch carrying its own label.
    ///
    ///     table 0:  batch 0 -> point 10
    ///     table 1:  batch 1 -> point 20, batch 2 -> point 21
    ///     table 2:  batch 3 -> point 30
    fn uneven_schedule() -> OpeningSchedule<F> {
        OpeningSchedule::new(vec![
            (
                TableShape::new(3, 2),
                vec![(OpeningBatch::new(vec![0, 1], Vec::new()), labelled(10))],
            ),
            (
                TableShape::new(4, 3),
                vec![
                    (OpeningBatch::new(vec![0, 1, 2], Vec::new()), labelled(20)),
                    (OpeningBatch::new(vec![2], Vec::new()), labelled(21)),
                ],
            ),
            (
                TableShape::new(2, 1),
                vec![(OpeningBatch::new(vec![0], Vec::new()), labelled(30))],
            ),
        ])
    }

    #[test]
    fn every_batch_keeps_the_point_it_was_built_with() {
        // The scheme walks the batches in one list and the points in another.
        //
        // Nothing downstream checks that entry k of one belongs with entry k of the other.
        //
        // A claim checked at a neighbour's point would still verify.
        //
        //     batches:  t0/b0   t1/b0   t1/b1   t2/b0
        //     points:    10      20      21      30
        //
        // The labels make the pairing observable.
        //
        // Batch k carries the point it was declared with, not the one beside it.
        let schedule = uneven_schedule();
        let expected = [10, 20, 21, 30].map(labelled);

        assert_eq!(schedule.points().len(), schedule.protocol().num_openings());
        assert_eq!(schedule.points(), expected);

        // The owning table of each batch follows the same order.
        let owners = schedule
            .protocol()
            .iter_openings()
            .map(|(table, _)| table)
            .collect::<Vec<_>>();
        assert_eq!(owners, vec![0, 1, 1, 2]);
    }

    #[test]
    fn a_table_is_read_from_the_first_batch_it_owns() {
        // An AIR's own columns live in the first batch of its table.
        //
        //     table 0 -> batch 0
        //     table 1 -> batch 1    its second batch, 2, answers a different claim
        //     table 2 -> batch 3
        //
        // Reading the per-batch results in order would hand table 2's values to table 1.
        assert_eq!(uneven_schedule().first_batch_per_table(), vec![0, 1, 3]);
    }

    #[test]
    fn one_batch_per_table_numbers_the_batches_like_the_tables() {
        // The shape every committed table has today, where the two orders coincide.
        let schedule = OpeningSchedule::whole_tables([
            (3, 2, vec![], labelled(1)),
            (2, 1, vec![0], labelled(2)),
        ]);

        assert_eq!(schedule.first_batch_per_table(), vec![0, 1]);
        assert_eq!(schedule.points(), [labelled(1), labelled(2)]);
    }

    #[test]
    #[should_panic(expected = "a committed table must be opened at least once")]
    fn a_committed_table_cannot_go_unopened() {
        // Committing columns nothing opens leaves them bound by no claim.
        let _ = OpeningSchedule::<F>::new(vec![(TableShape::new(3, 2), Vec::new())]);
    }
}
