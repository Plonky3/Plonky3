//! What one table fixes before a proof exists.

use alloc::string::String;
use alloc::vec::Vec;

use p3_air::symbolic::AirLayout;
use p3_air::{Air, BaseAir};
use p3_field::{ExtensionField, Field};
use p3_lookup::InteractionSymbolicBuilder;

use crate::contract::digest::Preimage;
use crate::contract::error::DeclarationError;

/// Largest number of columns of any one kind a table may declare.
pub const MAX_COLUMNS: usize = 1 << 20;

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

/// Which side of a channel a table's rows land on.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FlushDirection {
    /// Every active row adds its tuple to the produced multiset.
    Push,
    /// Every active row adds its tuple to the consumed multiset.
    Pull,
    /// The sign of the per-row multiplicity decides, so no side is fixed here.
    RowSigned,
}

impl FlushDirection {
    /// The byte this direction contributes to a statement fingerprint.
    const fn tag(self) -> u8 {
        match self {
            Self::Push => 0,
            Self::Pull => 1,
            Self::RowSigned => 2,
        }
    }
}

/// One family of equal-width tuples a table moves across a named channel.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FlushDeclaration {
    /// Name of the channel this family belongs to.
    pub channel: String,
    /// Which multiset side the rows land on.
    pub direction: FlushDirection,
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
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TableDeclaration {
    columns: ColumnCounts,
    constraints: LocalConstraints,
    heights: HeightRange,
    flushes: Vec<FlushDeclaration>,
    local_lookups: usize,
    indexed_reads: usize,
    indexed_tables: usize,
}

impl TableDeclaration {
    /// Declare a table with no channel traffic and no indexed access.
    #[must_use]
    pub const fn new(
        columns: ColumnCounts,
        constraints: LocalConstraints,
        heights: HeightRange,
    ) -> Self {
        Self {
            columns,
            constraints,
            heights,
            flushes: Vec::new(),
            local_lookups: 0,
            indexed_reads: 0,
            indexed_tables: 0,
        }
    }

    /// Add the channel traffic this table declares.
    #[must_use]
    pub fn with_flushes(mut self, flushes: Vec<FlushDeclaration>) -> Self {
        self.flushes = flushes;
        self
    }

    /// Add the lookups this table keeps to itself.
    #[must_use]
    pub const fn with_local_lookups(mut self, count: usize) -> Self {
        self.local_lookups = count;
        self
    }

    /// Add the indexed reads this table performs and the indexed tables it offers.
    #[must_use]
    pub const fn with_indexed(mut self, reads: usize, tables: usize) -> Self {
        self.indexed_reads = reads;
        self.indexed_tables = tables;
        self
    }

    /// Read a table's declaration off the constraint system it is built from.
    ///
    /// One symbolic pass supplies the columns, the constraints, and the channel traffic.
    ///
    /// The direction of each flush is decided per row, so none is fixed here.
    ///
    /// # Panics
    ///
    /// Panics if a width the constraint system determines disagrees with the table.
    #[must_use]
    pub fn from_constraints<F, EF, A>(table: &A, heights: HeightRange) -> Self
    where
        F: Field,
        EF: ExtensionField<F>,
        A: BaseAir<F> + Air<InteractionSymbolicBuilder<F, EF>>,
    {
        let layout = AirLayout::from_air::<F>(table);
        let builder = InteractionSymbolicBuilder::<F, EF>::from_air(table, layout);

        let base = builder.base_constraints();
        let extension = builder.extension_constraints();
        let degree = base
            .iter()
            .map(|expression| expression.poly_degree(2, &[]))
            .chain(
                extension
                    .iter()
                    .map(|expression| expression.poly_degree(2, &[])),
            )
            .max()
            .unwrap_or(0);

        let flushes = builder
            .global_interactions()
            .iter()
            .map(|interaction| FlushDeclaration {
                channel: interaction.bus_name.clone(),
                direction: FlushDirection::RowSigned,
                tuple_width: interaction.fields.len(),
                max_multiplicity: interaction.count_weight,
            })
            .collect();

        Self {
            columns: ColumnCounts {
                committed: table.width(),
                preprocessed: table.preprocessed_width(),
                public: table.num_public_values(),
            },
            constraints: LocalConstraints {
                count: base.len() + extension.len(),
                degree,
            },
            heights,
            flushes,
            local_lookups: builder.local_interactions().len(),
            indexed_reads: builder.indexed_reads().len(),
            indexed_tables: builder.indexed_tables().len(),
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
        preimage.usize(self.indexed_reads);
        preimage.usize(self.indexed_tables);
        preimage.usize(self.flushes.len());
        for flush in &self.flushes {
            preimage.bytes(flush.channel.as_bytes());
            preimage.byte(flush.direction.tag());
            preimage.usize(flush.tuple_width);
            preimage.u32(flush.max_multiplicity);
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_keccak::Keccak256Hash;

    use super::*;

    fn declared(heights: HeightRange) -> TableDeclaration {
        TableDeclaration::new(
            ColumnCounts::default(),
            LocalConstraints::default(),
            heights,
        )
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
    fn a_height_above_the_ceiling_is_refused() {
        assert!(matches!(
            declared(HeightRange::new(0, MAX_LOG_HEIGHT + 1))
                .validate(0)
                .unwrap_err(),
            DeclarationError::AboveLimit { .. }
        ));
    }

    #[test]
    fn a_column_count_above_the_ceiling_is_refused() {
        let wide = TableDeclaration::new(
            ColumnCounts {
                committed: MAX_COLUMNS + 1,
                preprocessed: 0,
                public: 0,
            },
            LocalConstraints::default(),
            HeightRange::exactly(4),
        );
        assert!(matches!(
            wide.validate(3).unwrap_err(),
            DeclarationError::AboveLimit { table: 3, .. }
        ));
    }

    #[test]
    fn the_channel_a_flush_names_reaches_the_fingerprint() {
        // Two tables alike but for the channel name must not absorb the same bytes.
        let flush = |channel: &str| {
            declared(HeightRange::exactly(4)).with_flushes(vec![FlushDeclaration {
                channel: channel.into(),
                direction: FlushDirection::Push,
                tuple_width: 2,
                max_multiplicity: 1,
            }])
        };
        let absorb = |table: &TableDeclaration| {
            let mut preimage = Preimage::new(b"test");
            table.absorb(&mut preimage);
            preimage.finish(&Keccak256Hash)
        };
        assert_ne!(absorb(&flush("left")), absorb(&flush("right")));
    }
}
