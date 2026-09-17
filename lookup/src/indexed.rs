//! Declaration of indexed lookups, where a reader names the entry it reads.

use alloc::string::String;
use alloc::vec::Vec;

use p3_air::Air;
use p3_air::symbolic::AirLayout;
use p3_field::{ExtensionField, Field};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::symbolic::InteractionSymbolicBuilder;

/// Which committed window of an AIR a column index addresses.
///
/// Both windows are committed and both can be opened, so either can hold a table.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TraceWindow {
    /// The AIR's execution trace, committed once per proof.
    #[default]
    Main,
    /// The AIR's fixed trace, committed once at setup and reused by every proof.
    Preprocessed,
}

/// One reader's declaration that it pulls entries out of a named table.
///
/// ```text
///     payload[j][i] = table_column[j][position[i]]     for every row i
/// ```
///
/// # Soundness
///
/// The position column holds the field's embedding of the entry index.
///
/// Over a prime field that embedding is the index itself.
///
/// Over a binary tower it is the tower element whose bit pattern is the index.
///
/// An AIR writing anything else into that column describes a different statement.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexedRead {
    /// Name of the table this reader pulls from.
    pub table: String,
    /// Main-trace column holding the entry each row names.
    ///
    /// The position is witness data rather than a fixed column, so it lives in the main trace.
    pub position: usize,
    /// Main-trace columns holding the values pulled, in table-column order.
    pub payload: Vec<usize>,
}

/// One table's declaration of the columns an entry carries.
///
/// A table is provided once and read by any number of readers.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexedTable {
    /// Name every reader of this table agrees on.
    pub name: String,
    /// Which committed window the columns below address.
    pub window: TraceWindow,
    /// Columns an entry carries, in the order readers pull them.
    pub columns: Vec<usize>,
}

/// Everything one AIR declares about indexed lookups.
///
/// The declarations name committed columns rather than expressions.
///
/// The reduction opens them against a commitment, and an expression has nothing to open.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexedLookups {
    /// Tables this AIR reads from, in emission order.
    reads: Vec<IndexedRead>,
    /// Tables this AIR provides, in emission order.
    tables: Vec<IndexedTable>,
}

impl IndexedLookups {
    /// Build from raw declarations, checking each against the AIR's own widths.
    ///
    /// # Errors
    ///
    /// Returns an error when a declaration names a column the AIR does not have.
    ///
    /// Returns an error when a reader pulls no column, or pulls one twice.
    ///
    /// Returns an error when a table carries no column, or carries one twice.
    pub fn new(
        reads: Vec<IndexedRead>,
        tables: Vec<IndexedTable>,
        layout: &AirLayout,
    ) -> Result<Self, IndexedLookupError> {
        for read in &reads {
            // The position and the pulled values are witness data, so both sit in the main trace.
            let width = layout.main_width;
            if read.position >= width {
                return Err(IndexedLookupError::ColumnOutOfRange {
                    table: read.table.clone(),
                    column: read.position,
                    width,
                });
            }
            if read.payload.is_empty() {
                return Err(IndexedLookupError::EmptyPayload {
                    table: read.table.clone(),
                });
            }
            Self::check_columns(&read.payload, width, &read.table)?;
        }

        for table in &tables {
            let width = match table.window {
                TraceWindow::Main => layout.main_width,
                TraceWindow::Preprocessed => layout.preprocessed_width,
            };
            if table.columns.is_empty() {
                return Err(IndexedLookupError::EmptyTable {
                    table: table.name.clone(),
                });
            }
            Self::check_columns(&table.columns, width, &table.name)?;
        }

        Ok(Self { reads, tables })
    }

    /// Extract every indexed declaration from one AIR by running symbolic evaluation.
    ///
    /// # Errors
    ///
    /// Returns an error when a declaration names a column the AIR does not have.
    pub fn from_air<F, EF, A>(air: &A) -> Result<Self, IndexedLookupError>
    where
        F: Field,
        EF: ExtensionField<F>,
        A: Air<InteractionSymbolicBuilder<F, EF>>,
    {
        let layout = AirLayout::from_air(air);
        let builder = InteractionSymbolicBuilder::<F, EF>::from_air(air, layout);
        Self::new(
            builder.indexed_reads().to_vec(),
            builder.indexed_tables().to_vec(),
            &layout,
        )
    }

    /// Tables this AIR reads from, in emission order.
    pub fn reads(&self) -> &[IndexedRead] {
        &self.reads
    }

    /// Tables this AIR provides, in emission order.
    pub fn tables(&self) -> &[IndexedTable] {
        &self.tables
    }

    /// Whether this AIR takes part in any indexed lookup.
    pub const fn is_empty(&self) -> bool {
        self.reads.is_empty() && self.tables.is_empty()
    }

    /// Reject a column list that leaves the trace or names one column twice.
    ///
    /// A repeat would buy a second claim on one column and open nothing new.
    fn check_columns(
        columns: &[usize],
        width: usize,
        table: &str,
    ) -> Result<(), IndexedLookupError> {
        for (position, &column) in columns.iter().enumerate() {
            if column >= width {
                return Err(IndexedLookupError::ColumnOutOfRange {
                    table: String::from(table),
                    column,
                    width,
                });
            }
            if columns[..position].contains(&column) {
                return Err(IndexedLookupError::RepeatedColumn {
                    table: String::from(table),
                    column,
                });
            }
        }
        Ok(())
    }
}

/// What makes a set of indexed declarations fail to describe a reduction.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum IndexedLookupError {
    /// A declaration names a column past the end of the window it addresses.
    #[error("table {table}: column {column} is out of range for a window of width {width}")]
    ColumnOutOfRange {
        /// Name of the table the declaration belongs to.
        table: String,
        /// Column the declaration named.
        column: usize,
        /// Width of the window that column addresses.
        width: usize,
    },
    /// One declaration names the same column twice.
    #[error("table {table}: column {column} is named twice by one declaration")]
    RepeatedColumn {
        /// Name of the table the declaration belongs to.
        table: String,
        /// Column named more than once.
        column: usize,
    },
    /// A reader pulls nothing, so it claims nothing about the table.
    #[error("table {table}: a reader must pull at least one column")]
    EmptyPayload {
        /// Name of the table the reader claims to read.
        table: String,
    },
    /// A table carries nothing, so there is no value to pull out of it.
    #[error("table {table}: a table entry must carry at least one column")]
    EmptyTable {
        /// Name of the table.
        table: String,
    },
    /// Two AIRs both claim to provide one table.
    ///
    /// Which one a reader pulls from would then be ambiguous.
    #[error("table {table}: provided by air {first} and again by air {second}")]
    TableProvidedTwice {
        /// Name of the table.
        table: String,
        /// Position of the first providing AIR in caller order.
        first: usize,
        /// Position of the second providing AIR in caller order.
        second: usize,
    },
    /// A reader names a table no AIR provides.
    #[error("table {table}: read by air {air} but provided by none")]
    TableNotProvided {
        /// Name of the table.
        table: String,
        /// Position of the reading AIR in caller order.
        air: usize,
    },
    /// A table is provided but nobody reads it.
    ///
    /// The reduction would then prove nothing about it.
    #[error("table {table}: provided by air {air} but read by none")]
    TableNotRead {
        /// Name of the table.
        table: String,
        /// Position of the providing AIR in caller order.
        air: usize,
    },
    /// A table holds a single entry, leaving the reduction no variable to split on.
    #[error("table {table}: a table needs at least two entries for the reduction to split on")]
    TableWithoutEntries {
        /// Name of the table.
        table: String,
    },
    /// A table holds more entries than the base field embeds injectively.
    ///
    /// Two entries sharing an embedding share a pole, which lets weight move between them.
    #[error(
        "table {table}: 2^{num_variables} entries do not embed injectively in a field of {bits} bits"
    )]
    TableTooLarge {
        /// Name of the table.
        table: String,
        /// Base-two logarithm of the table's entry count.
        num_variables: usize,
        /// Bit width of the base field.
        bits: usize,
    },
    /// A reader pulls a different number of columns than the table carries.
    #[error("table {table}: air {air} pulls {pulled} columns from a table carrying {carried}")]
    WidthMismatch {
        /// Name of the table.
        table: String,
        /// Position of the reading AIR in caller order.
        air: usize,
        /// Columns the reader pulls.
        pulled: usize,
        /// Columns the table carries.
        carried: usize,
    },
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;

    // A layout with a main trace of the given width and no preprocessed columns.
    fn main_only(main_width: usize) -> AirLayout {
        AirLayout {
            main_width,
            ..Default::default()
        }
    }

    // One reader of `table` naming position column 0 and pulling the listed columns.
    fn read(payload: Vec<usize>) -> IndexedRead {
        IndexedRead {
            table: String::from("t"),
            position: 0,
            payload,
        }
    }

    #[test]
    fn a_declaration_naming_every_real_column_is_accepted() {
        // Fixture state: main trace of 4 columns.
        //
        //     position -> column 0
        //     payload  -> columns 1, 2
        //     table    -> columns 1, 3
        let lookups = IndexedLookups::new(
            vec![read(vec![1, 2])],
            vec![IndexedTable {
                name: String::from("t"),
                window: TraceWindow::Main,
                columns: vec![1, 3],
            }],
            &main_only(4),
        )
        .expect("every column named is a real column");

        assert_eq!(lookups.reads().len(), 1);
        assert_eq!(lookups.tables().len(), 1);
        assert!(!lookups.is_empty());
    }

    #[test]
    fn a_position_column_past_the_trace_is_rejected() {
        // Column 4 is one past the last column of a four-column trace.
        let error = IndexedLookups::new(
            vec![IndexedRead {
                table: String::from("t"),
                position: 4,
                payload: vec![1],
            }],
            Vec::new(),
            &main_only(4),
        )
        .expect_err("a position column must address a real column");

        assert_eq!(
            error,
            IndexedLookupError::ColumnOutOfRange {
                table: String::from("t"),
                column: 4,
                width: 4,
            }
        );
    }

    #[test]
    fn a_payload_naming_one_column_twice_is_rejected() {
        // Pulling column 1 twice would claim the same value under two challenge powers.
        let error = IndexedLookups::new(vec![read(vec![1, 1])], Vec::new(), &main_only(4))
            .expect_err("a reader must not pull one column twice");

        assert_eq!(
            error,
            IndexedLookupError::RepeatedColumn {
                table: String::from("t"),
                column: 1,
            }
        );
    }

    #[test]
    fn a_reader_pulling_nothing_is_rejected() {
        // A reader with no payload makes no claim about the table it names.
        let error = IndexedLookups::new(vec![read(Vec::new())], Vec::new(), &main_only(4))
            .expect_err("a reader must pull at least one column");

        assert_eq!(
            error,
            IndexedLookupError::EmptyPayload {
                table: String::from("t"),
            }
        );
    }

    #[test]
    fn a_preprocessed_table_is_measured_against_the_preprocessed_width() {
        // Fixture state: main width 4, preprocessed width 2.
        //
        //     main column 3         exists
        //     preprocessed column 3 does not
        //
        // So the same index is accepted in one window and rejected in the other.
        let layout = AirLayout {
            main_width: 4,
            preprocessed_width: 2,
            ..Default::default()
        };

        let table = |window| IndexedTable {
            name: String::from("t"),
            window,
            columns: vec![3],
        };

        IndexedLookups::new(Vec::new(), vec![table(TraceWindow::Main)], &layout)
            .expect("the main trace has a column 3");

        let error =
            IndexedLookups::new(Vec::new(), vec![table(TraceWindow::Preprocessed)], &layout)
                .expect_err("the preprocessed trace stops at column 1");

        assert_eq!(
            error,
            IndexedLookupError::ColumnOutOfRange {
                table: String::from("t"),
                column: 3,
                width: 2,
            }
        );
    }

    #[test]
    fn a_table_carrying_nothing_is_rejected() {
        // An entry with no column holds no value a reader could pull.
        let error = IndexedLookups::new(
            Vec::new(),
            vec![IndexedTable {
                name: String::from("t"),
                window: TraceWindow::Main,
                columns: Vec::new(),
            }],
            &main_only(4),
        )
        .expect_err("a table entry must carry at least one column");

        assert_eq!(
            error,
            IndexedLookupError::EmptyTable {
                table: String::from("t"),
            }
        );
    }

    #[test]
    fn an_air_declaring_nothing_reports_itself_empty() {
        // Most AIRs in a batch take no part in an indexed lookup.
        let lookups = IndexedLookups::new(Vec::new(), Vec::new(), &main_only(4))
            .expect("declaring nothing is valid");

        assert!(lookups.is_empty());
    }
}
