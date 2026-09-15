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

/// Cut the row coordinates of one table out of a bound point.
///
/// The lookup reduction may add leading block-selector coordinates.
///
/// A short table also carries fewer row coordinates than a tall one.
///
/// Only the trailing coordinates addressing this table's rows are opened.
pub(super) fn trace_suffix<EF: Field>(point: &Point<EF>, num_variables: usize) -> Point<EF> {
    point.split_at(point.num_variables() - num_variables).1
}

/// What one opening batch answers.
///
/// A batch is found by what it answers rather than by its position.
///
/// A caller reading a claim back never re-derives the order they were laid down in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum BatchRole {
    /// An AIR's own columns, at the point the zerocheck bound.
    Air,
    /// One reader's position column, at the point the reduction left its position claims.
    Position {
        /// Position of the table in plan order.
        table: usize,
        /// Position of the reader within that table.
        reader: usize,
    },
    /// One table's own columns, at the point the reduction left its table claims.
    TableColumns {
        /// Position of the table in plan order.
        table: usize,
    },
}

/// One batch's role, and what it is opened against.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Opening<P> {
    /// What this batch answers.
    pub(super) role: BatchRole,
    /// The point on the proving path, and nothing on the assessment path.
    pub(super) against: P,
}

/// What one committed batch of tables opens, and what each batch is opened against.
///
/// A scheme takes two lists: the columns of every batch, and one point per batch.
///
/// Nothing in it ties an entry of one list to the entry beside it in the other.
///
/// Lists built apart can fall out of step.
///
/// A claim checked at another table's point still verifies.
///
/// Today's shapes make that loud rather than silent.
///
/// Two tables of different heights fail the arity check.
///
/// Two of the same height share a suffix, so nothing moves.
///
/// It turns silent once one commitment carries two distinct points of one arity.
///
/// A table opened more than once is what introduces that.
///
/// So the batches and what they are opened against come out of one walk.
///
/// The payload is a point on the proving path.
///
/// On the assessment path, which runs before any point exists, it is nothing at all.
pub(super) struct OpeningSchedule<P> {
    /// Shape agreement handed to the commitment scheme.
    protocol: OpeningProtocol,
    /// What each batch is opened against, in the protocol's own opening order.
    payloads: Vec<P>,
}

impl<P> OpeningSchedule<P> {
    /// Lay out one entry per committed table, tables in commitment order.
    ///
    /// # Arguments
    ///
    /// - `tables`: each table's committed shape, and the batches it is opened in paired with
    ///   what each is opened against.
    ///
    /// # Panics
    ///
    /// Panics if a table schedules no opening, which would commit columns nothing reads.
    pub(super) fn new<I>(tables: I) -> Self
    where
        I: IntoIterator<Item = (TableShape, Vec<(OpeningBatch<usize>, P)>)>,
    {
        let mut specs = Vec::new();
        let mut payloads = Vec::new();

        // One walk over the tables fills both lists, so their order agrees by construction.
        for (shape, openings) in tables {
            assert!(
                !openings.is_empty(),
                "a committed table must be opened at least once"
            );

            let mut batches = Vec::with_capacity(openings.len());
            for (batch, payload) in openings {
                batches.push(batch);
                payloads.push(payload);
            }
            specs.push(TableSpec::new(shape, batches));
        }

        Self {
            protocol: OpeningProtocol::new(specs),
            payloads,
        }
    }

    /// Shape agreement to hand the commitment scheme.
    pub(super) const fn protocol(&self) -> &OpeningProtocol {
        &self.protocol
    }

    /// What each batch is opened against, in the order the scheme walks the batches.
    ///
    /// The run reads this through the point-taking accessor below.
    #[cfg(test)]
    pub(super) fn payloads(&self) -> &[P] {
        &self.payloads
    }

    /// The shape agreement alone, for a caller that never resolves the payloads.
    pub(super) fn into_protocol(self) -> OpeningProtocol {
        self.protocol
    }

    /// Where each table's own columns land among the per-batch results, in table order.
    ///
    /// A table's columns are opened in the first batch it owns.
    ///
    /// The batches of one table are consecutive.
    ///
    /// This reads that off the shape agreement the scheme itself walks.
    ///
    /// A list kept beside it would be one more thing to hold in step.
    pub(super) fn first_batch_per_table(&self) -> Vec<usize> {
        let mut first = Vec::new();

        // Owners are non-decreasing.
        //
        // A batch starts a table when its owner is the first one not yet recorded.
        for (batch, (table, _)) in self.protocol.iter_openings().enumerate() {
            if first.len() == table {
                first.push(batch);
            }
        }
        first
    }
}

impl<P: Clone> OpeningSchedule<Opening<P>> {
    /// What each batch is opened against, in the order the scheme walks the batches.
    pub(super) fn against(&self) -> Vec<P> {
        self.payloads
            .iter()
            .map(|opening| opening.against.clone())
            .collect()
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

    /// Schedule the main trace opening, in the order the scheme walks the batches.
    ///
    /// Every table opens its whole width at the point the zerocheck bound.
    ///
    /// A table the indexed reduction reaches opens further batches.
    ///
    /// Those are taken at the points that reduction closes on.
    ///
    /// The security assessment walks this same schedule, so the two cannot diverge.
    ///
    /// # Arguments
    ///
    /// - `indexed`: the indexed-lookup plan, when the batch declares one.
    /// - `against`: what to open a batch of this role, over this many row variables, against.
    pub(super) fn main_schedule<P>(
        &self,
        indexed: Option<&IndexedPlan>,
        against: impl Fn(BatchRole, usize) -> P,
    ) -> OpeningSchedule<Opening<P>> {
        let mut tables = self
            .num_variables()
            .iter()
            .zip(self.widths().iter())
            .zip(self.next_columns())
            .map(|((&log_height, &width), next_columns)| {
                (
                    TableShape::new(log_height, width),
                    alloc::vec![(
                        OpeningBatch::new((0..width).collect::<Vec<_>>(), next_columns),
                        Opening {
                            role: BatchRole::Air,
                            against: against(BatchRole::Air, log_height),
                        },
                    )],
                )
            })
            .collect::<Vec<_>>();

        // A reader's position column and a main-window table's columns both live here.
        if let Some(plan) = indexed {
            for (table, planned) in plan.tables().iter().enumerate() {
                for (reader, placement) in planned.readers.iter().enumerate() {
                    let role = BatchRole::Position { table, reader };
                    tables[placement.air].1.push((
                        OpeningBatch::new(alloc::vec![placement.position], Vec::new()),
                        Opening {
                            role,
                            against: against(role, placement.num_variables),
                        },
                    ));
                }
                if planned.table.window == TraceWindow::Main {
                    let role = BatchRole::TableColumns { table };
                    tables[planned.table.air].1.push((
                        OpeningBatch::new(planned.table.columns.clone(), Vec::new()),
                        Opening {
                            role,
                            against: against(role, planned.table.num_variables),
                        },
                    ));
                }
            }
        }

        OpeningSchedule::new(tables)
    }

    /// Schedule the preprocessed trace opening, in the order the scheme walks the batches.
    ///
    /// AIRs declaring no preprocessed columns commit nothing and are skipped.
    ///
    /// A table the indexed reduction reads out of this window opens one further batch.
    ///
    /// # Arguments
    ///
    /// - `indexed`: the indexed-lookup plan, when the batch declares one.
    /// - `against`: what to open a batch of this role, over this many row variables, against.
    pub(super) fn preprocessed_schedule<P>(
        &self,
        indexed: Option<&IndexedPlan>,
        against: impl Fn(BatchRole, usize) -> P,
    ) -> OpeningSchedule<Opening<P>> {
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
                        OpeningBatch::new(
                            (0..width).collect::<Vec<_>>(),
                            instance.air.preprocessed_next_row_columns(),
                        ),
                        Opening {
                            role: BatchRole::Air,
                            against: against(BatchRole::Air, instance.num_variables),
                        },
                    )],
                )
            })
            .collect::<Vec<_>>();

        if let Some(plan) = indexed {
            for (table, planned) in plan.tables().iter().enumerate() {
                if planned.table.window != TraceWindow::Preprocessed {
                    continue;
                }
                let slot = committed
                    .iter()
                    .position(|&air| air == planned.table.air)
                    .expect("a preprocessed table's AIR commits preprocessed columns");
                let role = BatchRole::TableColumns { table };
                tables[slot].1.push((
                    OpeningBatch::new(planned.table.columns.clone(), Vec::new()),
                    Opening {
                        role,
                        against: against(role, planned.table.num_variables),
                    },
                ));
            }
        }

        OpeningSchedule::new(tables)
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
    fn uneven_schedule() -> OpeningSchedule<Point<F>> {
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

        assert_eq!(
            schedule.payloads().len(),
            schedule.protocol().num_openings()
        );
        assert_eq!(schedule.payloads(), expected);

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
        let schedule = OpeningSchedule::new(vec![
            (
                TableShape::new(3, 2),
                vec![(OpeningBatch::new(vec![0, 1], Vec::new()), labelled(1))],
            ),
            (
                TableShape::new(2, 1),
                vec![(OpeningBatch::new(vec![0], vec![0]), labelled(2))],
            ),
        ]);

        assert_eq!(schedule.first_batch_per_table(), vec![0, 1]);
        assert_eq!(schedule.payloads(), [labelled(1), labelled(2)]);
    }

    #[test]
    #[should_panic(expected = "a committed table must be opened at least once")]
    fn a_committed_table_cannot_go_unopened() {
        // Committing columns nothing opens leaves them bound by no claim.
        let _ = OpeningSchedule::<F>::new(vec![(TableShape::new(3, 2), Vec::new())]);
    }
}
