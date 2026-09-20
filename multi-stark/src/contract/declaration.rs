//! The shape a caller declares before a proof exists.
//!
//! A declaration is the only description of a statement the verifier trusts.
//!
//! Every bound the proof reader applies is read from here, never from the encoded proof.

use alloc::string::String;
use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_air::symbolic::AirLayout;
use p3_air::{Air, BaseAir};
use p3_field::{ExtensionField, Field};
use p3_lookup::InteractionSymbolicBuilder;
use thiserror::Error;

use crate::contract::digest::Digest;
use crate::contract::secrecy::SecrecyLevel;

/// Largest number of tables one statement may declare.
pub const MAX_TABLES: usize = 1 << 12;

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

/// Largest grinding difficulty a run may request.
pub const MAX_POW_BITS: u32 = 64;

/// Hard ceiling on any declared proof-size budget, in bytes.
pub const MAX_PROOF_BYTES: usize = 1 << 30;

/// Why a declaration or a run of it is not usable.
#[derive(Clone, Debug, Eq, PartialEq, Error)]
pub enum DeclarationError {
    /// A statement with no tables proves nothing.
    #[error("a statement must declare at least one table")]
    NoTables,
    /// A count exceeds the ceiling this module fixes for it.
    #[error("table {table}: {what} is {found}, above the limit of {limit}")]
    AboveLimit {
        /// Position of the offending table in declaration order.
        table: usize,
        /// Which count is out of range.
        what: &'static str,
        /// The declared value.
        found: usize,
        /// The largest accepted value.
        limit: usize,
    },
    /// A height range excludes every height.
    #[error("table {table}: height range {min}..={max} is empty")]
    EmptyHeightRange {
        /// Position of the offending table in declaration order.
        table: usize,
        /// Smallest declared exponent.
        min: u32,
        /// Largest declared exponent.
        max: u32,
    },
    /// The declared proof-size budget is zero or above the hard ceiling.
    #[error("proof-size budget {found} is outside 1..={limit} bytes")]
    BudgetOutOfRange {
        /// The declared budget.
        found: usize,
        /// The largest accepted budget.
        limit: usize,
    },
    /// A run supplies a different number of heights than the statement has tables.
    #[error("the run supplies {found} heights for {expected} tables")]
    HeightCountMismatch {
        /// Number of declared tables.
        expected: usize,
        /// Number of heights the run supplied.
        found: usize,
    },
    /// A run picks a height the table never declared.
    #[error("table {table}: height exponent {found} is outside the declared {min}..={max}")]
    HeightNotDeclared {
        /// Position of the offending table in declaration order.
        table: usize,
        /// The exponent the run picked.
        found: u32,
        /// Smallest declared exponent.
        min: u32,
        /// Largest declared exponent.
        max: u32,
    },
    /// A run requests more grinding than this module accepts.
    #[error("grinding difficulty {found} is above the limit of {limit}")]
    PowBitsAboveLimit {
        /// The requested difficulty.
        found: u32,
        /// The largest accepted difficulty.
        limit: u32,
    },
    /// A run belongs to a different statement than the one it is used with.
    #[error("the run describes a different statement")]
    ForeignRun,
}

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
    /// The byte this direction contributes to a statement digest.
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

    fn validate(&self, table: usize) -> Result<(), DeclarationError> {
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

    fn absorb(&self, digest: &mut Digest) {
        digest.usize(self.columns.committed);
        digest.usize(self.columns.preprocessed);
        digest.usize(self.columns.public);
        digest.usize(self.constraints.count);
        digest.usize(self.constraints.degree);
        digest.u32(self.heights.min);
        digest.u32(self.heights.max);
        digest.usize(self.local_lookups);
        digest.usize(self.indexed_reads);
        digest.usize(self.indexed_tables);
        digest.usize(self.flushes.len());
        for flush in &self.flushes {
            digest.bytes(flush.channel.as_bytes());
            digest.byte(flush.direction.tag());
            digest.usize(flush.tuple_width);
            digest.u32(flush.max_multiplicity);
        }
    }
}

/// Every table of one statement, the commitment promise, and the proof-size budget.
///
/// The promise is a type parameter.
///
/// A statement proved under the weaker one is not the statement proved under the stronger.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MachineDeclaration<S> {
    tables: Vec<TableDeclaration>,
    max_proof_bytes: usize,
    secrecy: PhantomData<fn() -> S>,
}

impl<S: SecrecyLevel> MachineDeclaration<S> {
    /// Fix the tables of a statement and the byte budget its proofs must fit in.
    ///
    /// The budget is a public parameter, and the proof reader rejects anything longer.
    ///
    /// # Errors
    ///
    /// Returns an error when a declared count is outside the range this module accepts.
    pub fn new(
        tables: Vec<TableDeclaration>,
        max_proof_bytes: usize,
    ) -> Result<Self, DeclarationError> {
        if tables.is_empty() {
            return Err(DeclarationError::NoTables);
        }
        if tables.len() > MAX_TABLES {
            return Err(DeclarationError::AboveLimit {
                table: 0,
                what: "table count",
                found: tables.len(),
                limit: MAX_TABLES,
            });
        }
        if max_proof_bytes == 0 || max_proof_bytes > MAX_PROOF_BYTES {
            return Err(DeclarationError::BudgetOutOfRange {
                found: max_proof_bytes,
                limit: MAX_PROOF_BYTES,
            });
        }
        for (index, table) in tables.iter().enumerate() {
            table.validate(index)?;
        }

        Ok(Self {
            tables,
            max_proof_bytes,
            secrecy: PhantomData,
        })
    }

    /// The tables of this statement, in the order proofs list them.
    #[must_use]
    pub fn tables(&self) -> &[TableDeclaration] {
        &self.tables
    }

    /// The largest encoded proof this statement accepts, in bytes.
    #[must_use]
    pub const fn max_proof_bytes(&self) -> usize {
        self.max_proof_bytes
    }

    /// Whether any table commits columns fixed at setup.
    #[must_use]
    pub fn has_preprocessed(&self) -> bool {
        self.tables
            .iter()
            .any(|table| table.columns.preprocessed > 0)
    }

    /// Whether any table takes part in a lookup argument.
    #[must_use]
    pub fn has_lookups(&self) -> bool {
        self.tables.iter().any(TableDeclaration::has_lookups)
    }

    /// Total number of indexed reads across every table.
    #[must_use]
    pub fn num_indexed_reads(&self) -> usize {
        self.tables.iter().map(|table| table.indexed_reads).sum()
    }

    /// Pick the height of every table and the grinding difficulty for one proof.
    ///
    /// # Errors
    ///
    /// Returns an error when a height is outside its table's declared range.
    ///
    /// Returns an error when the heights and the tables disagree in number.
    pub fn run(&self, log_heights: &[usize], pow_bits: usize) -> Result<Run, DeclarationError> {
        if log_heights.len() != self.tables.len() {
            return Err(DeclarationError::HeightCountMismatch {
                expected: self.tables.len(),
                found: log_heights.len(),
            });
        }
        if pow_bits > MAX_POW_BITS as usize {
            return Err(DeclarationError::PowBitsAboveLimit {
                found: u32::try_from(pow_bits).unwrap_or(u32::MAX),
                limit: MAX_POW_BITS,
            });
        }

        let mut heights = Vec::with_capacity(log_heights.len());
        for (index, (&log_height, table)) in log_heights.iter().zip(&self.tables).enumerate() {
            let found = u32::try_from(log_height).unwrap_or(u32::MAX);
            if found < table.heights.min || found > table.heights.max {
                return Err(DeclarationError::HeightNotDeclared {
                    table: index,
                    found,
                    min: table.heights.min,
                    max: table.heights.max,
                });
            }
            heights.push(found);
        }

        Ok(Run {
            statement: self.statement_digest(),
            log_heights: heights,
            pow_bits: pow_bits as u32,
        })
    }

    /// Fingerprint of everything fixed before a proof exists.
    fn statement_digest(&self) -> [u8; 32] {
        let mut digest = Digest::new(b"p3-backend-contract/statement/v1");
        digest.byte(S::SECRECY.tag());
        digest.usize(self.max_proof_bytes);
        digest.usize(self.tables.len());
        for table in &self.tables {
            table.absorb(&mut digest);
        }
        digest.finish()
    }

    /// Fingerprint of the statement together with the choices one proof makes.
    pub(crate) fn run_digest(&self, run: &Run) -> Result<[u8; 32], DeclarationError> {
        let statement = self.statement_digest();
        if run.statement != statement {
            return Err(DeclarationError::ForeignRun);
        }

        let mut digest = Digest::new(b"p3-backend-contract/run/v1");
        digest.bytes(&statement);
        digest.u32(run.pow_bits);
        digest.usize(run.log_heights.len());
        for &log_height in &run.log_heights {
            digest.u32(log_height);
        }
        Ok(digest.finish())
    }
}

/// The choices one proof makes within a statement it cannot change.
///
/// The only way to build one is to pick heights a statement already allows.
///
/// A height outside the declared range therefore never reaches the prover or the verifier.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Run {
    statement: [u8; 32],
    log_heights: Vec<u32>,
    pow_bits: u32,
}

impl Run {
    /// Base-two logarithm of each table's height, in declaration order.
    #[must_use]
    pub fn log_heights(&self) -> &[u32] {
        &self.log_heights
    }

    /// Grinding difficulty every delegated round runs at.
    #[must_use]
    pub const fn pow_bits(&self) -> usize {
        self.pow_bits as usize
    }
}
