//! Planning of the batch's indexed lookups, from the AIRs and their trace heights alone.
//!
//! An AIR declares that it reads a named table, or that it provides one.
//!
//! This module resolves those names against each other and fixes the order both sides walk.
//!
//! ```text
//!     air 0  reads  "range8" at column 3, pulling columns 4, 5
//!     air 2  provides "range8" from preprocessed columns 0, 1
//!
//!         ->  one table of 2^h entries, two columns, one reader
//! ```
//!
//! Nothing here comes out of a proof.
//!
//! Both sides run the same pass over the same AIRs and reach the same plan.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

use p3_air::{Air, BaseAir};
use p3_field::{ExtensionField, Field};
use p3_lookup::indexed::{IndexedLookupError, IndexedLookups, TraceWindow};
use p3_lookup::symbolic::InteractionSymbolicBuilder;
use p3_multilinear_util::point::Point;

use crate::logup_star::{Reader, TableLookup, position};
use crate::opening::TableOpening;

/// Where one table lives and how big it is.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TablePlacement {
    /// Position of the providing AIR in caller order.
    pub air: usize,
    /// Which committed window the columns address.
    pub window: TraceWindow,
    /// Columns an entry carries, in the order readers pull them.
    pub columns: Vec<usize>,
    /// Base-two logarithm of the providing AIR's trace height.
    pub num_variables: usize,
}

/// Where one reader lives and how big it is.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReaderPlacement {
    /// Position of the reading AIR in caller order.
    pub air: usize,
    /// Main-trace column holding the entry each row names.
    pub position: usize,
    /// Main-trace columns holding the values pulled, in table-column order.
    pub payload: Vec<usize>,
    /// Base-two logarithm of the reading AIR's trace height.
    pub num_variables: usize,
}

/// One table together with every reader that pulls from it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IndexedTablePlan {
    /// Name both sides resolve this table by.
    pub name: String,
    /// Where the table itself lives.
    pub table: TablePlacement,
    /// Every reader of this table, in caller order then emission order.
    pub readers: Vec<ReaderPlacement>,
}

/// Every indexed lookup in one batch, in the order both sides walk it.
///
/// Tables are ordered by name rather than by the position of the AIR providing them.
///
/// That gives the table list one canonical order, independent of how the batch is arranged.
///
/// Readers are not: they follow caller order, and each carries its AIR's position.
///
/// Swapping two reader AIRs of one table therefore changes the statement.
///
/// Both sides read the same instance order and the seed binds it, so they still agree.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IndexedPlan {
    /// One entry per table, in name order.
    tables: Vec<IndexedTablePlan>,
}

impl IndexedPlan {
    /// Resolve every declaration in the batch against every other.
    ///
    /// Returns nothing when no AIR in the batch declares an indexed lookup.
    ///
    /// That is the signal that this proof carries no indexed-lookup section at all.
    ///
    /// # Arguments
    ///
    /// - `airs`: every AIR in the batch, in caller order.
    /// - `num_variables`: base-two logarithm of each AIR's trace height, in the same order.
    ///
    /// # Errors
    ///
    /// Returns an error when a declaration names a column its AIR does not have.
    ///
    /// Returns an error when two AIRs provide one table, or when nobody provides one that is read.
    ///
    /// Returns an error when a table is provided but never read.
    ///
    /// Returns an error when a reader pulls a different number of columns than the table carries.
    ///
    /// Returns an error when a table holds more entries than the base field embeds injectively.
    ///
    /// # Panics
    ///
    /// Panics when the two slices disagree on length.
    pub fn build<F, EF, A>(
        airs: &[&A],
        num_variables: &[usize],
    ) -> Result<Option<Self>, IndexedLookupError>
    where
        F: Field,
        EF: ExtensionField<F>,
        A: BaseAir<F> + Air<InteractionSymbolicBuilder<F, EF>>,
    {
        assert_eq!(
            airs.len(),
            num_variables.len(),
            "every AIR must carry a trace height"
        );

        // One symbolic pass per AIR, checking each declaration against that AIR's own widths.
        let declared = airs
            .iter()
            .map(|&air| IndexedLookups::from_air::<F, EF, _>(air))
            .collect::<Result<Vec<_>, _>>()?;

        if declared.iter().all(IndexedLookups::is_empty) {
            return Ok(None);
        }

        // Collect the providers first, since a reader is only valid against one.
        //
        // Tables are keyed by name, not by the position of the AIR providing them.
        //
        // The name sort is what gives the table list one canonical order.
        let mut tables: BTreeMap<String, IndexedTablePlan> = BTreeMap::new();
        for (air, lookups) in declared.iter().enumerate() {
            for table in lookups.tables() {
                let placement = TablePlacement {
                    air,
                    window: table.window,
                    columns: table.columns.clone(),
                    num_variables: num_variables[air],
                };

                // A table needs a variable for the reduction to split on.
                if placement.num_variables == 0 {
                    return Err(IndexedLookupError::TableWithoutEntries {
                        table: table.name.clone(),
                    });
                }

                // Two entries sharing an embedding share a pole.
                //
                // Weight could then move between them unseen.
                if !position::fits::<F>(placement.num_variables) {
                    return Err(IndexedLookupError::TableTooLarge {
                        table: table.name.clone(),
                        num_variables: placement.num_variables,
                        bits: F::bits(),
                    });
                }

                if let Some(existing) = tables.get(&table.name) {
                    return Err(IndexedLookupError::TableProvidedTwice {
                        table: table.name.clone(),
                        first: existing.table.air,
                        second: air,
                    });
                }

                tables.insert(
                    table.name.clone(),
                    IndexedTablePlan {
                        name: table.name.clone(),
                        table: placement,
                        readers: Vec::new(),
                    },
                );
            }
        }

        // Attach each reader to its table, walking AIRs in caller order.
        //
        // Within one AIR the emission order is kept, so a reader's place is fixed by the AIR.
        for (air, lookups) in declared.iter().enumerate() {
            for read in lookups.reads() {
                let Some(plan) = tables.get_mut(&read.table) else {
                    return Err(IndexedLookupError::TableNotProvided {
                        table: read.table.clone(),
                        air,
                    });
                };

                // Every reader pulls the whole entry, so one width serves the table.
                if read.payload.len() != plan.table.columns.len() {
                    return Err(IndexedLookupError::WidthMismatch {
                        table: read.table.clone(),
                        air,
                        pulled: read.payload.len(),
                        carried: plan.table.columns.len(),
                    });
                }

                plan.readers.push(ReaderPlacement {
                    air,
                    position: read.position,
                    payload: read.payload.clone(),
                    num_variables: num_variables[air],
                });
            }
        }

        // A table nobody reads leaves the reduction with nothing to prove about it.
        for plan in tables.values() {
            if plan.readers.is_empty() {
                return Err(IndexedLookupError::TableNotRead {
                    table: plan.name.clone(),
                    air: plan.table.air,
                });
            }
        }

        Ok(Some(Self {
            tables: tables.into_values().collect(),
        }))
    }

    /// Every table in the batch, in the order both sides walk them.
    pub fn tables(&self) -> &[IndexedTablePlan] {
        &self.tables
    }

    /// Number of readers across every table.
    pub fn num_readers(&self) -> usize {
        self.tables.iter().map(|table| table.readers.len()).sum()
    }

    /// Read the statement off the point the zerocheck bound and each AIR's opened columns.
    ///
    /// A reader's claim point is the suffix of the bound point matching its own trace height.
    ///
    /// That is the convention every other opening in the batch already follows.
    ///
    /// # Arguments
    ///
    /// - `point`: the point the zerocheck bound, covering the tallest trace.
    /// - `openings`: each AIR's opened main-trace columns, in caller order.
    ///
    /// # Panics
    ///
    /// Panics when the bound point is shorter than a reader's own trace.
    ///
    /// Panics when an AIR's opening does not carry a column a reader pulls.
    ///
    /// Panics when the opening list is shorter than the AIRs taking part.
    pub fn statement<EF: Field>(
        &self,
        point: &Point<EF>,
        openings: &[TableOpening<'_, EF>],
    ) -> IndexedStatement<EF> {
        let mut points = Vec::with_capacity(self.num_readers());
        let mut claims = Vec::with_capacity(self.num_readers());

        for table in &self.tables {
            for reader in &table.readers {
                assert!(
                    point.num_variables() >= reader.num_variables,
                    "the bound point must cover every reader's trace"
                );

                // The reader's own coordinates are the trailing ones, as everywhere else.
                let own = point
                    .split_at(point.num_variables() - reader.num_variables)
                    .1;

                let opened = openings
                    .get(reader.air)
                    .expect("every AIR taking part carries an opening")
                    .local;
                let pulled = reader
                    .payload
                    .iter()
                    .map(|&column| {
                        *opened
                            .get(column)
                            .expect("a reader pulls a column its AIR opened")
                    })
                    .collect::<Vec<_>>();

                points.push(own);
                claims.push(pulled);
            }
        }

        IndexedStatement {
            points,
            claims,
            readers_per_table: self
                .tables
                .iter()
                .map(|table| table.readers.len())
                .collect(),
            table_variables: self
                .tables
                .iter()
                .map(|table| table.table.num_variables)
                .collect(),
        }
    }
}

/// The claims one batch's indexed lookups are about, owned so the statement can borrow them.
///
/// # Soundness
///
/// Every point here is a suffix of a challenge the surrounding protocol drew.
///
/// That challenge is drawn after the position columns and the tables are committed.
///
/// A point chosen before those commitments costs the argument its free range check.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IndexedStatement<EF> {
    /// One claim point per reader, in plan order.
    points: Vec<Point<EF>>,
    /// One claim per pulled column per reader, in plan order.
    claims: Vec<Vec<EF>>,
    /// Reader count of each table, in plan order.
    readers_per_table: Vec<usize>,
    /// Entry count of each table, as a base-two logarithm, in plan order.
    table_variables: Vec<usize>,
}

impl<EF: Field> IndexedStatement<EF> {
    /// Borrowing view of every reader, in plan order.
    ///
    /// The result is kept by the caller, because the statement below borrows from it.
    pub fn readers(&self) -> Vec<Reader<'_, EF>> {
        self.points
            .iter()
            .zip(&self.claims)
            .map(|(point, claims)| Reader {
                point,
                claims: claims.as_slice(),
            })
            .collect()
    }

    /// Group the readers into one entry per table, in plan order.
    ///
    /// # Panics
    ///
    /// Panics when the reader list does not match the one this statement describes.
    pub fn lookups<'a>(&self, readers: &'a [Reader<'a, EF>]) -> Vec<TableLookup<'a, EF>> {
        assert_eq!(
            readers.len(),
            self.readers_per_table.iter().sum::<usize>(),
            "the reader list must be the one this statement describes"
        );

        let mut first = 0;
        self.readers_per_table
            .iter()
            .zip(&self.table_variables)
            .map(|(&count, &num_variables)| {
                let lookup = TableLookup {
                    num_variables,
                    readers: &readers[first..first + count],
                };
                first += count;
                lookup
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;
    use p3_lookup::IndexedLookupBuilder;

    use super::*;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;

    /// One table declaration: its name, the window it lives in, and its columns.
    type TableScript = (&'static str, TraceWindow, Vec<usize>);

    /// One read declaration: the table's name, the position column, and the pulled columns.
    type ReadScript = (&'static str, usize, Vec<usize>);

    /// An AIR that emits a fixed script of indexed declarations and nothing else.
    struct Declaring {
        /// Width the AIR reports for its main trace.
        width: usize,
        /// Width the AIR reports for its preprocessed trace.
        preprocessed_width: usize,
        /// Reads emitted, in order.
        reads: Vec<ReadScript>,
        /// Tables provided, in order.
        tables: Vec<TableScript>,
    }

    impl Declaring {
        /// An AIR of the given main width that declares nothing.
        fn bare(width: usize) -> Self {
            Self {
                width,
                preprocessed_width: 0,
                reads: Vec::new(),
                tables: Vec::new(),
            }
        }

        /// Add one read of the named table.
        fn reading(mut self, table: &'static str, position: usize, payload: Vec<usize>) -> Self {
            self.reads.push((table, position, payload));
            self
        }

        /// Add one provided table in the main trace.
        fn providing(mut self, name: &'static str, columns: Vec<usize>) -> Self {
            self.tables.push((name, TraceWindow::Main, columns));
            self
        }

        /// Add one provided table in the preprocessed trace, widening that trace to hold it.
        fn providing_preprocessed(mut self, name: &'static str, columns: Vec<usize>) -> Self {
            let widest = columns.iter().copied().max().map_or(0, |column| column + 1);
            self.preprocessed_width = self.preprocessed_width.max(widest);
            self.tables.push((name, TraceWindow::Preprocessed, columns));
            self
        }
    }

    impl BaseAir<F> for Declaring {
        fn width(&self) -> usize {
            self.width
        }

        fn preprocessed_width(&self) -> usize {
            self.preprocessed_width
        }
    }

    impl<AB: IndexedLookupBuilder<F = F>> Air<AB> for Declaring {
        fn eval(&self, builder: &mut AB) {
            for (table, position, payload) in &self.reads {
                builder.push_indexed_read(table, *position, payload.iter().copied());
            }
            for (name, window, columns) in &self.tables {
                builder.push_indexed_table(name, *window, columns.iter().copied());
            }
        }
    }

    /// Plan a batch of AIRs at the given heights.
    fn plan(
        airs: &[&Declaring],
        num_variables: &[usize],
    ) -> Result<Option<IndexedPlan>, IndexedLookupError> {
        IndexedPlan::build::<F, EF, _>(airs, num_variables)
    }

    #[test]
    fn a_batch_declaring_nothing_plans_nothing() {
        // Most batches take no part in an indexed lookup.
        //
        // The absent plan is what tells the proof it carries no indexed section at all.
        let air = Declaring::bare(4);

        assert_eq!(plan(&[&air, &air], &[3, 3]), Ok(None));
    }

    #[test]
    fn a_table_keeps_the_window_its_air_declared() {
        // The window decides which commitment the table's claims are opened against.
        //
        //     Main          ->  the batch's main commitment
        //     Preprocessed  ->  the preprocessed commitment in the keys
        //
        // A preprocessed range table checked against main columns is one the prover picks.
        //
        // The verifying key is what should fix it.
        //
        // So the window has to survive planning rather than defaulting to the main trace.
        let reader = Declaring::bare(2).reading("range", 0, vec![1]);
        let fixed = Declaring::bare(1).providing_preprocessed("range", vec![0]);

        let preprocessed = plan(&[&reader, &fixed], &[3, 2]).unwrap().unwrap();
        let table = &preprocessed.tables()[0];

        assert_eq!(table.table.window, TraceWindow::Preprocessed);
        assert_eq!(table.table.air, 1);
        assert_eq!(table.table.columns, vec![0]);

        // A main-window provider of the same shape must not land in the same place.
        let live = Declaring::bare(1).providing("range", vec![0]);
        let other = plan(&[&reader, &live], &[3, 2]).unwrap().unwrap();

        assert_eq!(other.tables()[0].table.window, TraceWindow::Main);
        assert_ne!(other.tables()[0].table.window, table.table.window);
    }

    #[test]
    fn a_reader_and_its_table_resolve_by_name() {
        // Fixture state: air 0 reads "t", air 1 provides it.
        //
        //     air 0 (2^3 rows, 4 cols):  position col 3, pulls cols 1, 2
        //     air 1 (2^2 rows, 2 cols):  provides "t" from cols 0, 1
        let reader = Declaring::bare(4).reading("t", 3, vec![1, 2]);
        let table = Declaring::bare(2).providing("t", vec![0, 1]);

        let plan = plan(&[&reader, &table], &[3, 2])
            .expect("the batch describes a reduction")
            .expect("some AIR declares an indexed lookup");

        assert_eq!(plan.tables().len(), 1);
        assert_eq!(plan.num_readers(), 1);

        let entry = &plan.tables()[0];
        assert_eq!(entry.name, "t");
        assert_eq!(
            entry.table,
            TablePlacement {
                air: 1,
                window: TraceWindow::Main,
                columns: vec![0, 1],
                num_variables: 2,
            }
        );
        assert_eq!(
            entry.readers,
            vec![ReaderPlacement {
                air: 0,
                position: 3,
                payload: vec![1, 2],
                num_variables: 3,
            }]
        );
    }

    #[test]
    fn both_sides_plan_one_batch_identically() {
        // The verifier holds no witness, so it must reach the plan from the AIRs alone.
        //
        // Running the same pass twice is what the two sides do, one each.
        let reader = Declaring::bare(4).reading("t", 3, vec![1, 2]);
        let table = Declaring::bare(2).providing("t", vec![0, 1]);

        let prover = plan(&[&reader, &table], &[3, 2]);
        let verifier = plan(&[&reader, &table], &[3, 2]);

        assert_eq!(prover, verifier);
    }

    #[test]
    fn table_order_follows_the_name_and_not_the_air() {
        // Two batches describing the same tables, with the providing AIRs swapped.
        //
        //     batch A:  air 1 provides "a",  air 2 provides "b"
        //     batch B:  air 1 provides "b",  air 2 provides "a"
        //
        // An AIR order would list the tables differently in the two batches.
        //
        // A name order lists "a" first either way, which is all this compares.
        //
        // The placements still differ: "a" is provided by a different AIR in each.
        let reader = Declaring::bare(2)
            .reading("a", 0, vec![0])
            .reading("b", 1, vec![0]);
        let first = Declaring::bare(1).providing("a", vec![0]);
        let second = Declaring::bare(1).providing("b", vec![0]);

        let forward = plan(&[&reader, &first, &second], &[2, 1, 1])
            .unwrap()
            .unwrap();
        let backward = plan(&[&reader, &second, &first], &[2, 1, 1])
            .unwrap()
            .unwrap();

        let names = |plan: &IndexedPlan| {
            plan.tables()
                .iter()
                .map(|table| table.name.clone())
                .collect::<Vec<_>>()
        };
        assert_eq!(names(&forward), vec![String::from("a"), String::from("b")]);
        assert_eq!(names(&backward), names(&forward));
    }

    #[test]
    fn two_airs_providing_one_table_is_rejected() {
        // Which of the two a reader pulls from would be ambiguous.
        let reader = Declaring::bare(2).reading("t", 0, vec![0]);
        let table = Declaring::bare(1).providing("t", vec![0]);

        assert_eq!(
            plan(&[&reader, &table, &table], &[2, 1, 1]),
            Err(IndexedLookupError::TableProvidedTwice {
                table: String::from("t"),
                first: 1,
                second: 2,
            })
        );
    }

    #[test]
    fn a_table_nobody_provides_is_rejected() {
        // The reader names a table the batch never defines, so there is nothing to read.
        let reader = Declaring::bare(2).reading("missing", 0, vec![0]);
        let table = Declaring::bare(1).providing("t", vec![0]);
        let other = Declaring::bare(2).reading("t", 0, vec![0]);

        assert_eq!(
            plan(&[&reader, &table, &other], &[2, 1, 2]),
            Err(IndexedLookupError::TableNotProvided {
                table: String::from("missing"),
                air: 0,
            })
        );
    }

    #[test]
    fn a_table_nobody_reads_is_rejected() {
        // The reduction would carry a table and prove nothing about it.
        let table = Declaring::bare(1).providing("t", vec![0]);

        assert_eq!(
            plan(&[&table], &[1]),
            Err(IndexedLookupError::TableNotRead {
                table: String::from("t"),
                air: 0,
            })
        );
    }

    #[test]
    fn a_reader_pulling_the_wrong_number_of_columns_is_rejected() {
        // The table carries two columns per entry and the reader pulls one.
        //
        // One claim cannot fix two values, so the statement is not well formed.
        let reader = Declaring::bare(2).reading("t", 0, vec![1]);
        let table = Declaring::bare(2).providing("t", vec![0, 1]);

        assert_eq!(
            plan(&[&reader, &table], &[2, 1]),
            Err(IndexedLookupError::WidthMismatch {
                table: String::from("t"),
                air: 0,
                pulled: 1,
                carried: 2,
            })
        );
    }

    #[test]
    fn a_table_the_field_cannot_index_is_rejected() {
        // BabyBear enumerates 31 bits, so 2^31 entries stop embedding injectively.
        //
        // Two entries sharing an embedding share a pole, and weight moves between them unseen.
        let reader = Declaring::bare(2).reading("t", 0, vec![0]);
        let table = Declaring::bare(1).providing("t", vec![0]);

        assert_eq!(
            plan(&[&reader, &table], &[2, 31]),
            Err(IndexedLookupError::TableTooLarge {
                table: String::from("t"),
                num_variables: 31,
                bits: 31,
            })
        );
    }

    #[test]
    fn a_table_with_one_entry_is_rejected() {
        // The reduction splits on a variable, and a single-entry table has none.
        let reader = Declaring::bare(2).reading("t", 0, vec![0]);
        let table = Declaring::bare(1).providing("t", vec![0]);

        assert_eq!(
            plan(&[&reader, &table], &[2, 0]),
            Err(IndexedLookupError::TableWithoutEntries {
                table: String::from("t"),
            })
        );
    }

    #[test]
    fn a_reader_claim_is_read_off_its_own_air() {
        // Fixture state: air 0 reads "t" pulling its own columns 1 and 2.
        //
        //     bound point        (r_0, r_1, r_2)
        //     air 0 height       2^2, so its own point is the suffix (r_1, r_2)
        //     air 0 opened       [10, 11, 12, 13]
        //
        // The reader's claims are the opened values of the columns it pulls, in pull order.
        let reader = Declaring::bare(4).reading("t", 3, vec![1, 2]);
        let table = Declaring::bare(2).providing("t", vec![0, 1]);
        let plan = plan(&[&reader, &table], &[2, 1]).unwrap().unwrap();

        let point = Point::<EF>::new(vec![EF::from_u8(7), EF::from_u8(8), EF::from_u8(9)]);
        let reader_values = [10, 11, 12, 13].map(EF::from_u8);
        let table_values = [20, 21].map(EF::from_u8);
        let openings = [
            TableOpening::new(&reader_values, &[], &[]),
            TableOpening::new(&table_values, &[], &[]),
        ];

        let statement = plan.statement(&point, &openings);
        let readers = statement.readers();
        let lookups = statement.lookups(&readers);

        assert_eq!(lookups.len(), 1);
        assert_eq!(lookups[0].num_variables, 1);
        assert_eq!(lookups[0].readers.len(), 1);

        // The claim point is the trailing two coordinates, matching a 2^2-row trace.
        assert_eq!(
            readers[0].point.as_slice(),
            &[EF::from_u8(8), EF::from_u8(9)]
        );
        // Columns 1 and 2 of the reading AIR, in the order the reader pulls them.
        assert_eq!(readers[0].claims, &[EF::from_u8(11), EF::from_u8(12)]);
    }

    #[test]
    fn the_soundness_shape_counts_rows_and_entries_rather_than_readers() {
        // The entry challenge is charged one pole per leaf carrying a fraction.
        //
        // Counting readers instead of rows would understate it by orders of magnitude.
        //
        // The arithmetic is worth pinning on a lopsided fixture.
        //
        // Fixture state: one table of 2^2 entries and 2 columns, read by two AIRs.
        //
        //     table  "t"   2^2 = 4 entries
        //     reader air 0 2^5 = 32 rows
        //     reader air 2 2^3 = 8 rows
        //                  ------------------
        //     leaves       44, so a 2^6 padded table
        let reader = Declaring::bare(4).reading("t", 3, vec![1, 2]);
        let table = Declaring::bare(2).providing("t", vec![0, 1]);
        let second = Declaring::bare(4).reading("t", 3, vec![1, 2]);

        let plan = plan(&[&reader, &table, &second], &[5, 2, 3])
            .expect("the batch describes a reduction")
            .expect("some AIR declares an indexed lookup");
        let params = plan.security_params();

        assert_eq!(params.num_leaves, 4 + 32 + 8);
        assert_eq!(params.num_variables, 6);
        assert_eq!(params.max_readers_per_table, 2);
        assert_eq!(params.max_reader_variables, 5);
        assert_eq!(params.max_table_variables, 2);
        assert_eq!(params.num_column_claims, 2);
    }

    #[test]
    fn the_soundness_shape_sums_over_tables_where_it_says_it_does() {
        // Three of these numbers add across tables and two take an extreme.
        //
        // One table makes every pair of those agree, so the arithmetic needs two.
        //
        // Fixture state:
        //
        //     table "a"   2^2 = 4 entries, 2 columns, read by one 2^4 AIR
        //     table "b"   2^3 = 8 entries, 1 column,  read by two 2^3 AIRs
        //
        //     leaves           4 + 16 + 8 + 8 + 8 = 44      summed
        //     column claims    2 + 1 = 3                    summed
        //     readers/table    max(1, 2) = 2                extreme
        //     reader variables max(4, 3, 3) = 4             extreme
        //     table variables  max(2, 3) = 3                extreme
        let first = Declaring::bare(4).reading("a", 3, vec![1, 2]);
        let table_a = Declaring::bare(2).providing("a", vec![0, 1]);
        let second = Declaring::bare(2).reading("b", 1, vec![0]);
        let third = Declaring::bare(2).reading("b", 1, vec![0]);
        let table_b = Declaring::bare(1).providing("b", vec![0]);

        let plan = plan(
            &[&first, &table_a, &second, &third, &table_b],
            &[4, 2, 3, 3, 3],
        )
        .expect("the batch describes a reduction")
        .expect("some AIR declares an indexed lookup");
        let params = plan.security_params();

        assert_eq!(params.num_leaves, 4 + 16 + 8 + 8 + 8);
        assert_eq!(params.num_column_claims, 3);
        assert_eq!(params.max_readers_per_table, 2);
        assert_eq!(params.max_reader_variables, 4);
        assert_eq!(params.max_table_variables, 3);
    }
}
