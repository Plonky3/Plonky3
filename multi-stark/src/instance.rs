//! Public instance wrappers for batched multi-STARK proving and verification.
//!
//! Setup takes only AIR references, since public values and main trace tables are proof-time data.
//! A proving instance adds the committed main trace table, the shared proving key, and public values.
//! A verifying instance adds the shared verifying key, the trace height, and public values.

use alloc::vec::Vec;
use core::ops::Deref;

use p3_air::BaseAir;
use p3_field::Field;
use p3_lookup::TraceWindow;
use p3_multilinear_util::point::Point;
use p3_sumcheck::layout::Table;
use p3_sumcheck::{OpeningBatch, OpeningProtocol, TableShape, TableSpec};

use crate::config::MultiStarkConfig;
use crate::indexed::IndexedPlan;
pub use crate::keys::{ProvingKey, VerifyingKey, setup};
pub use crate::proof::MultiStarkProof;
pub use crate::prover::prove;
pub use crate::verifier::{VerificationError, verify};

/// Which point a schedule resolves one opening batch against.
///
/// A batch names columns, and its point is settled only when the run reaches it.
///
/// Naming the point rather than carrying it lets one description serve two readers.
///
/// The security assessment runs before any point exists, and the run itself needs them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum BatchPoint {
    /// The point the zerocheck bound.
    Bound,
    /// The point the indexed reduction left its position claims at.
    Position,
    /// The point the indexed reduction left its table claims at.
    Table,
}

/// The points one run reached, for a layout to resolve its batches against.
pub(super) struct BoundPoints<'a, EF> {
    /// Where the zerocheck landed.
    pub(super) bound: &'a Point<EF>,
    /// Where the indexed reduction left its position claims, when a batch asks for it.
    pub(super) position: Option<&'a Point<EF>>,
    /// Where the indexed reduction left its table claims, when a batch asks for it.
    pub(super) table: Option<&'a Point<EF>>,
}

impl<EF> BoundPoints<'_, EF> {
    /// The point a batch names.
    ///
    /// # Panics
    ///
    /// Panics if a batch names a point this run never reached.
    const fn resolve(&self, which: BatchPoint) -> &Point<EF> {
        match which {
            BatchPoint::Bound => self.bound,
            BatchPoint::Position => self
                .position
                .expect("a position batch needs the reduction's position point"),
            BatchPoint::Table => self
                .table
                .expect("a table batch needs the reduction's table point"),
        }
    }
}

/// One table's committed shape, and every batch it is opened in.
type TableBatches = (TableShape, Vec<(OpeningBatch<usize>, BatchPoint)>);

/// What every committed table of one commitment opens, before the points are known.
///
/// The security assessment and the run must agree on this exactly.
///
/// An assessment covering fewer batches than the run reports a bound the proof misses.
///
/// So both read this one description rather than building their own.
pub(super) struct OpeningLayout {
    /// One entry per committed table, in commitment order.
    tables: Vec<TableBatches>,
}

impl OpeningLayout {
    /// Lay out one entry per committed table, tables in commitment order.
    pub(super) fn new<I>(tables: I) -> Self
    where
        I: IntoIterator<Item = TableBatches>,
    {
        Self {
            tables: tables.into_iter().collect(),
        }
    }

    /// The shape agreement, which no point enters.
    pub(super) fn protocol(&self) -> OpeningProtocol {
        OpeningProtocol::new(
            self.tables
                .iter()
                .map(|(shape, batches)| {
                    TableSpec::new(
                        *shape,
                        batches.iter().map(|(batch, _)| batch.clone()).collect(),
                    )
                })
                .collect(),
        )
    }

    /// Resolve every batch against the points this run reached.
    ///
    /// Each table is opened at the trailing coordinates addressing its own rows.
    ///
    /// # Panics
    ///
    /// Panics if a batch names a point this run never reached.
    ///
    /// Panics if a point does not cover the table it is resolved for.
    pub(super) fn schedule<EF: Field>(&self, points: &BoundPoints<'_, EF>) -> OpeningSchedule<EF> {
        OpeningSchedule::new(self.tables.iter().map(|(shape, batches)| {
            let openings = batches
                .iter()
                .map(|(batch, which)| {
                    let point = points.resolve(*which);
                    let rows = shape.num_variables();
                    assert!(
                        point.num_variables() >= rows,
                        "an opening point must cover the table it is taken for"
                    );
                    (
                        batch.clone(),
                        point.split_at(point.num_variables() - rows).1,
                    )
                })
                .collect();
            (*shape, openings)
        }))
    }
}

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
    /// What the main commitment opens, before any point is bound.
    ///
    /// Every table contributes its whole width at the point the zerocheck binds.
    ///
    /// A table an indexed lookup reaches contributes further batches.
    ///
    /// Those are taken at the points that reduction closes on.
    ///
    /// # Arguments
    ///
    /// - `indexed`: the indexed-lookup plan, when the batch declares one.
    pub(super) fn main_layout(&self, indexed: Option<&IndexedPlan>) -> OpeningLayout {
        let mut tables = self
            .num_variables()
            .iter()
            .zip(self.widths().iter())
            .zip(self.next_columns())
            .map(|((&log_height, &width), next_columns)| {
                (
                    TableShape::new(log_height, width),
                    alloc::vec![(
                        OpeningSchedule::<C::Challenge>::whole_table_batch(width, next_columns),
                        BatchPoint::Bound,
                    )],
                )
            })
            .collect::<Vec<_>>();

        // The reduction closes on two points, and both land on this commitment.
        //
        // Both live in the main trace, so both land on this commitment.
        if let Some(plan) = indexed {
            for table in plan.tables() {
                for reader in &table.readers {
                    tables[reader.air].1.push((
                        OpeningBatch::new(alloc::vec![reader.position], Vec::new()),
                        BatchPoint::Position,
                    ));
                }
                if table.table.window == TraceWindow::Main {
                    tables[table.table.air].1.push((
                        OpeningBatch::new(table.table.columns.clone(), Vec::new()),
                        BatchPoint::Table,
                    ));
                }
            }
        }

        OpeningLayout::new(tables)
    }

    /// What the preprocessed commitment opens, before any point is bound.
    ///
    /// AIRs declaring no preprocessed columns commit nothing and are skipped.
    ///
    /// A table an indexed lookup reads out of this window contributes one further batch.
    ///
    /// # Arguments
    ///
    /// - `indexed`: the indexed-lookup plan, when the batch declares one.
    pub(super) fn preprocessed_layout(&self, indexed: Option<&IndexedPlan>) -> OpeningLayout {
        // Only AIRs with preprocessed columns are committed, so the two orders differ.
        let committed = self
            .iter()
            .enumerate()
            .filter(|(_, instance)| instance.air.preprocessed_width() != 0)
            .map(|(air, _)| air)
            .collect::<Vec<_>>();

        let mut tables = committed
            .iter()
            .map(|&air| {
                let instance = &self.0[air];
                let width = instance.air.preprocessed_width();
                (
                    TableShape::new(instance.num_variables, width),
                    alloc::vec![(
                        OpeningSchedule::<C::Challenge>::whole_table_batch(
                            width,
                            instance.air.preprocessed_next_row_columns(),
                        ),
                        BatchPoint::Bound,
                    )],
                )
            })
            .collect::<Vec<_>>();

        if let Some(plan) = indexed {
            for table in plan.tables() {
                if table.table.window != TraceWindow::Preprocessed {
                    continue;
                }
                let slot = committed
                    .iter()
                    .position(|&air| air == table.table.air)
                    .expect("a preprocessed table's AIR commits preprocessed columns");
                tables[slot].1.push((
                    OpeningBatch::new(table.table.columns.clone(), Vec::new()),
                    BatchPoint::Table,
                ));
            }
        }

        OpeningLayout::new(tables)
    }

    /// Shape agreement for the main trace opening, without the points.
    pub(super) fn opening_protocol(&self, indexed: Option<&IndexedPlan>) -> OpeningProtocol {
        self.main_layout(indexed).protocol()
    }

    /// Shape agreement for the preprocessed trace opening, without the points.
    pub(super) fn preprocessed_opening_protocol(
        &self,
        indexed: Option<&IndexedPlan>,
    ) -> OpeningProtocol {
        self.preprocessed_layout(indexed).protocol()
    }

    pub(super) fn preprocessed_next_columns(&self) -> Vec<Vec<usize>> {
        self.iter()
            .filter(|instance| instance.air.preprocessed_width() != 0)
            .map(|instance| instance.air.preprocessed_next_row_columns())
            .collect()
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
    fn a_layout_resolves_every_batch_against_the_point_it_names() {
        // One table opened twice: its whole width at the bound point.
        //
        // Its second batch takes one column at the reduction's table point.
        //
        //     batch 0 -> Bound     batch 1 -> Table
        //
        // The layout carries the names, and the run supplies the points.
        let layout = OpeningLayout::new(vec![(
            TableShape::new(1, 2),
            vec![
                (OpeningBatch::new(vec![0, 1], Vec::new()), BatchPoint::Bound),
                (OpeningBatch::new(vec![1], Vec::new()), BatchPoint::Table),
            ],
        )]);

        // The shape agreement is the same whether or not a point exists yet.
        assert_eq!(layout.protocol().num_openings(), 2);

        let bound = labelled(7);
        let table = labelled(9);
        let schedule = layout.schedule(&BoundPoints {
            bound: &bound,
            position: None,
            table: Some(&table),
        });

        assert_eq!(schedule.points(), [bound, table]);
        assert_eq!(schedule.first_batch_per_table(), vec![0]);
    }

    #[test]
    #[should_panic(expected = "a table batch needs the reduction's table point")]
    fn a_layout_cannot_resolve_a_point_the_run_never_reached() {
        // A run with no reduction cannot answer a batch naming the reduction's point.
        let layout = OpeningLayout::new(vec![(
            TableShape::new(1, 1),
            vec![(OpeningBatch::new(vec![0], Vec::new()), BatchPoint::Table)],
        )]);

        let bound = labelled(1);
        let _ = layout.schedule(&BoundPoints {
            bound: &bound,
            position: None,
            table: None,
        });
    }

    #[test]
    #[should_panic(expected = "a committed table must be opened at least once")]
    fn a_committed_table_cannot_go_unopened() {
        // Committing columns nothing opens leaves them bound by no claim.
        let _ = OpeningSchedule::<F>::new(vec![(TableShape::new(3, 2), Vec::new())]);
    }
}
