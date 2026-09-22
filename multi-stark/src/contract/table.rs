//! What one table fixes before a proof exists.

use alloc::string::String;
use alloc::vec::Vec;
use core::fmt::{Debug, Formatter, Result as FmtResult};

use p3_air::symbolic::AirLayout;
use p3_air::{Air, BaseAir};
use p3_bus::BusSymbolicBuilder;
use p3_field::{ExtensionField, Field};
use p3_lookup::InteractionSymbolicBuilder;

use crate::contract::constraints;
use crate::contract::digest::Preimage;
use crate::contract::error::DeclarationError;
use crate::folder::boundary_io_pins;

/// Largest number of columns of any one kind a table may declare.
pub const MAX_COLUMNS: usize = 1 << 20;

/// Smallest base-two logarithm of a table height the backend can prove.
pub const MIN_LOG_HEIGHT: u32 = 1;

/// Largest base-two logarithm of a table height.
pub const MAX_LOG_HEIGHT: u32 = 40;

/// Largest per-variable constraint degree a table may declare.
pub const MAX_CONSTRAINT_DEGREE: usize = 1 << 8;

/// Largest number of constraints a table may declare.
pub const MAX_CONSTRAINTS: usize = 1 << 20;

/// Largest number of channel flushes a table may declare.
pub const MAX_FLUSHES: usize = 1 << 10;

/// Largest tuple width a flush may declare.
pub const MAX_TUPLE_WIDTH: usize = 1 << 12;

/// Largest number of indexed reads or indexed tables a table may declare.
pub const MAX_INDEXED: usize = 1 << 16;

/// One family of equal-width tuples a table moves across a named channel.
///
/// The sign of the per-row multiplicity decides which side a row lands on, so no side is fixed.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FlushDeclaration {
    /// Name of the channel this family belongs to.
    pub channel: String,
    /// Number of field elements in one tuple.
    pub tuple_width: usize,
    /// Per-row upper bound on the magnitude of the multiplicity.
    pub max_multiplicity: u32,
}

/// How many columns of each kind a table has.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ColumnCounts {
    /// Columns committed once per proof.
    pub committed: usize,
    /// Columns fixed by the statement and committed once at setup.
    pub preprocessed: usize,
    /// Values supplied in the clear alongside the proof.
    pub public: usize,
}

/// What a table asserts about one row and its successor.
///
/// A cell the backend pins to a public value is one of these, though no evaluation writes it.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct LocalConstraints {
    /// Number of expressions the table asserts vanish on every row.
    pub count: usize,
    /// Largest per-variable degree among those expressions.
    pub degree: usize,
}

/// The heights a table is allowed to run at.
///
/// Both ends are base-two logarithms, and both are inclusive.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HeightRange {
    /// Smallest allowed exponent.
    pub min: u32,
    /// Largest allowed exponent.
    pub max: u32,
}

impl HeightRange {
    /// A range that admits exactly one height.
    #[must_use]
    pub const fn exactly(log_height: u32) -> Self {
        Self {
            min: log_height,
            max: log_height,
        }
    }

    /// A range between two inclusive exponents.
    #[must_use]
    pub const fn new(min: u32, max: u32) -> Self {
        Self { min, max }
    }
}

/// Everything one table fixes before a proof exists.
///
/// The only way to build one is to read it off a constraint system.
///
/// Every part but the height range is therefore what the system itself says.
#[derive(Clone, Eq, PartialEq)]
pub struct TableDeclaration {
    columns: ColumnCounts,
    constraints: LocalConstraints,
    heights: HeightRange,
    flushes: Vec<FlushDeclaration>,
    local_lookups: usize,
    buses: usize,
    indexed_reads: usize,
    indexed_tables: usize,
    system: Vec<u8>,
}

impl Debug for TableDeclaration {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.debug_struct("TableDeclaration")
            .field("columns", &self.columns)
            .field("constraints", &self.constraints)
            .field("heights", &self.heights)
            .field("flushes", &self.flushes)
            .field("local_lookups", &self.local_lookups)
            .field("buses", &self.buses)
            .field("indexed_reads", &self.indexed_reads)
            .field("indexed_tables", &self.indexed_tables)
            .field("system_bytes", &self.system.len())
            .finish()
    }
}

impl TableDeclaration {
    /// Read a table's declaration off the constraint system it is built from.
    ///
    /// One symbolic pass supplies the columns, the constraints, and the channel traffic.
    ///
    /// The cells the backend pins to public values are counted and scored alongside them.
    ///
    /// A lookup that carries no tuple is dropped, because the reduction drops it too.
    ///
    /// The contents of a fixed trace stay outside.
    ///
    /// The verifying key commits to those, and an opening against it is what binds them.
    ///
    /// # Panics
    ///
    /// Panics if a width the constraint system determines disagrees with the table.
    #[must_use]
    pub fn from_constraints<F, EF, A>(table: &A, heights: HeightRange) -> Self
    where
        F: Field,
        EF: ExtensionField<F>,
        A: BaseAir<F> + Air<InteractionSymbolicBuilder<F, EF>> + Air<BusSymbolicBuilder<F, EF>>,
    {
        let layout = AirLayout::from_air::<F>(table);
        let builder = InteractionSymbolicBuilder::<F, EF>::from_air(table, layout);
        // The interaction pass drops every bus declaration, so the bus pass runs too.
        let buses = BusSymbolicBuilder::<F, EF>::from_air(table, layout);

        let base = builder.base_constraints();
        let extension = builder.extension_constraints();
        let symbolic = base
            .iter()
            .map(|expression| expression.poly_degree(2, &[]))
            .chain(
                extension
                    .iter()
                    .map(|expression| expression.poly_degree(2, &[])),
            )
            .max()
            .unwrap_or(0);

        // Scored the way the reduction scores it, hint and pins included.
        let written = base.len() + extension.len();
        let own = if written == 0 {
            0
        } else {
            table
                .max_constraint_degree()
                .unwrap_or(symbolic)
                .max(symbolic)
        };
        let pins = boundary_io_pins(table.public_boundary_io());
        let degree = own.max(pins.degree);

        let flushes = builder
            .global_interactions()
            .iter()
            .map(|interaction| FlushDeclaration {
                channel: interaction.bus_name.clone(),
                tuple_width: interaction.fields.len(),
                max_multiplicity: interaction.count_weight,
            })
            .collect();

        let local_lookups = builder
            .local_interactions()
            .iter()
            .filter(|interaction| !interaction.tuples.is_empty())
            .count();

        Self {
            columns: ColumnCounts {
                committed: table.width(),
                preprocessed: table.preprocessed_width(),
                public: table.num_public_values(),
            },
            constraints: LocalConstraints {
                count: written + pins.count,
                degree,
            },
            heights,
            flushes,
            local_lookups,
            buses: buses.interactions().len(),
            indexed_reads: builder.indexed_reads().len(),
            indexed_tables: builder.indexed_tables().len(),
            system: constraints::encode(table, &builder, &buses),
        }
    }

    /// How many columns of each kind this table has.
    #[must_use]
    pub const fn columns(&self) -> ColumnCounts {
        self.columns
    }

    /// What this table asserts about one row and its successor.
    #[must_use]
    pub const fn constraints(&self) -> LocalConstraints {
        self.constraints
    }

    /// The heights this table is allowed to run at.
    #[must_use]
    pub const fn heights(&self) -> HeightRange {
        self.heights
    }

    /// The channel traffic this table declares.
    #[must_use]
    pub fn flushes(&self) -> &[FlushDeclaration] {
        &self.flushes
    }

    /// How many indexed reads this table performs.
    #[must_use]
    pub const fn indexed_reads(&self) -> usize {
        self.indexed_reads
    }

    /// Whether this table takes part in any lookup argument.
    #[must_use]
    pub const fn has_lookups(&self) -> bool {
        !self.flushes.is_empty() || self.local_lookups > 0
    }

    /// Whether this table moves any tuple across a bus.
    #[must_use]
    pub const fn has_buses(&self) -> bool {
        self.buses > 0
    }

    /// How many bus declarations this table makes.
    pub(super) const fn bus_declarations(&self) -> usize {
        self.buses
    }

    /// Name the first part on which this table and a constraint system disagree.
    ///
    /// The height range is left out, because no constraint system fixes it.
    pub(super) fn disagreement<F, EF, A>(&self, table: &A) -> Option<&'static str>
    where
        F: Field,
        EF: ExtensionField<F>,
        A: BaseAir<F> + Air<InteractionSymbolicBuilder<F, EF>> + Air<BusSymbolicBuilder<F, EF>>,
    {
        let read = Self::from_constraints::<F, EF, A>(table, self.heights);
        [
            (self.columns.committed != read.columns.committed).then_some("the committed columns"),
            (self.columns.preprocessed != read.columns.preprocessed)
                .then_some("the preprocessed columns"),
            (self.columns.public != read.columns.public).then_some("the public values"),
            (self.constraints.count != read.constraints.count).then_some("the constraint count"),
            (self.constraints.degree != read.constraints.degree).then_some("the constraint degree"),
            (self.flushes != read.flushes).then_some("the channel traffic"),
            (self.local_lookups != read.local_lookups).then_some("the local lookups"),
            (self.buses != read.buses).then_some("the bus declarations"),
            (self.indexed_reads != read.indexed_reads).then_some("the indexed reads"),
            (self.indexed_tables != read.indexed_tables).then_some("the indexed tables"),
            (self.system != read.system).then_some("the constraints themselves"),
        ]
        .into_iter()
        .flatten()
        .next()
    }

    /// Refuse a table whose numbers leave the reader without a bound.
    pub(super) fn validate(&self, table: usize) -> Result<(), DeclarationError> {
        let limit = |what, found, limit| DeclarationError::AboveLimit {
            table,
            what,
            found,
            limit,
        };
        let check = |what, found, ceiling| {
            (found <= ceiling)
                .then_some(())
                .ok_or_else(|| limit(what, found, ceiling))
        };

        check(
            "committed column count",
            self.columns.committed,
            MAX_COLUMNS,
        )?;
        check(
            "preprocessed column count",
            self.columns.preprocessed,
            MAX_COLUMNS,
        )?;
        check("public value count", self.columns.public, MAX_COLUMNS)?;
        check("constraint count", self.constraints.count, MAX_CONSTRAINTS)?;
        check(
            "constraint degree",
            self.constraints.degree,
            MAX_CONSTRAINT_DEGREE,
        )?;
        check("flush count", self.flushes.len(), MAX_FLUSHES)?;
        check("local lookup count", self.local_lookups, MAX_INDEXED)?;
        check("bus declaration count", self.buses, MAX_FLUSHES)?;
        check("indexed read count", self.indexed_reads, MAX_INDEXED)?;
        check("indexed table count", self.indexed_tables, MAX_INDEXED)?;
        check(
            "height exponent",
            self.heights.max as usize,
            MAX_LOG_HEIGHT as usize,
        )?;

        for flush in &self.flushes {
            check("flush tuple width", flush.tuple_width, MAX_TUPLE_WIDTH)?;
        }

        if self.heights.min < MIN_LOG_HEIGHT {
            return Err(DeclarationError::HeightBelowFloor {
                table,
                min: self.heights.min,
                floor: MIN_LOG_HEIGHT,
            });
        }

        (self.heights.min <= self.heights.max).then_some(()).ok_or(
            DeclarationError::EmptyHeightRange {
                table,
                min: self.heights.min,
                max: self.heights.max,
            },
        )
    }

    /// Absorb everything this table fixes into a statement fingerprint.
    pub(super) fn absorb(&self, preimage: &mut Preimage) {
        preimage.usize(self.columns.committed);
        preimage.usize(self.columns.preprocessed);
        preimage.usize(self.columns.public);
        preimage.usize(self.constraints.count);
        preimage.usize(self.constraints.degree);
        preimage.u32(self.heights.min);
        preimage.u32(self.heights.max);
        preimage.usize(self.local_lookups);
        preimage.usize(self.buses);
        preimage.usize(self.indexed_reads);
        preimage.usize(self.indexed_tables);
        preimage.usize(self.flushes.len());
        for flush in &self.flushes {
            preimage.bytes(flush.channel.as_bytes());
            preimage.usize(flush.tuple_width);
            preimage.u32(flush.max_multiplicity);
        }
        preimage.bytes(&self.system);
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_keccak::Keccak256Hash;

    use super::*;

    fn declared(heights: HeightRange) -> TableDeclaration {
        TableDeclaration {
            columns: ColumnCounts::default(),
            constraints: LocalConstraints::default(),
            heights,
            flushes: Vec::new(),
            local_lookups: 0,
            buses: 0,
            indexed_reads: 0,
            indexed_tables: 0,
            system: Vec::new(),
        }
    }

    #[test]
    fn an_unreachable_height_range_is_refused() {
        assert_eq!(
            declared(HeightRange::new(9, 8)).validate(0).unwrap_err(),
            DeclarationError::EmptyHeightRange {
                table: 0,
                min: 9,
                max: 8,
            }
        );
    }

    #[test]
    fn a_range_reaching_a_single_row_is_refused() {
        // A one-row trace leaves the reduction nothing to fold, so the floor is two rows.
        assert_eq!(
            declared(HeightRange::new(0, 20)).validate(2).unwrap_err(),
            DeclarationError::HeightBelowFloor {
                table: 2,
                min: 0,
                floor: 1,
            }
        );
        assert!(declared(HeightRange::new(1, 20)).validate(2).is_ok());
    }

    #[test]
    fn a_height_above_the_ceiling_is_refused() {
        // Forty is the ceiling, so forty-one is the smallest exponent that is refused.
        assert_eq!(
            declared(HeightRange::new(1, 41)).validate(0).unwrap_err(),
            DeclarationError::AboveLimit {
                table: 0,
                what: "height exponent",
                found: 41,
                limit: 40,
            }
        );
    }

    #[test]
    fn a_column_count_above_the_ceiling_is_refused() {
        // The ceiling is two to the twentieth, so one more than that is refused.
        let mut wide = declared(HeightRange::exactly(4));
        wide.columns.committed = 1_048_577;
        assert_eq!(
            wide.validate(3).unwrap_err(),
            DeclarationError::AboveLimit {
                table: 3,
                what: "committed column count",
                found: 1_048_577,
                limit: 1_048_576,
            }
        );
    }

    #[test]
    fn the_channel_a_flush_names_reaches_the_fingerprint() {
        // Two tables alike but for the channel name must not absorb the same bytes.
        let flush = |channel: &str| {
            let mut table = declared(HeightRange::exactly(4));
            table.flushes = vec![FlushDeclaration {
                channel: channel.into(),
                tuple_width: 2,
                max_multiplicity: 1,
            }];
            table
        };
        let absorb = |table: &TableDeclaration| {
            let mut preimage = Preimage::new(b"test");
            table.absorb(&mut preimage);
            preimage.finish(&Keccak256Hash)
        };
        assert_ne!(absorb(&flush("left")), absorb(&flush("right")));
    }

    #[test]
    fn what_a_table_asserts_reaches_the_fingerprint() {
        // Two tables alike in every count must not absorb the same bytes.
        let pinned = |system: &[u8]| {
            let mut table = declared(HeightRange::exactly(4));
            table.system = system.to_vec();
            table
        };
        let absorb = |table: &TableDeclaration| {
            let mut preimage = Preimage::new(b"test");
            table.absorb(&mut preimage);
            preimage.finish(&Keccak256Hash)
        };
        assert_ne!(absorb(&pinned(b"one")), absorb(&pinned(b"two")));
    }
}
