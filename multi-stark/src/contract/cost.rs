//! What one run of a statement costs, table by table, read off the shape alone.
//!
//! Every figure comes from the declaration and the run, never from a witness or a proof.
//!
//! A verifier can therefore compute the same report as the prover.
//!
//! The figures a proof also carries are pinned against real proofs by tests.

use alloc::vec::Vec;

use p3_field::Field;

use crate::contract::run::Run;
use crate::contract::table::TableDeclaration;

/// What one table costs in one run.
///
/// Every figure is exact for the shape except the peak storage, which is an upper bound.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TableCost {
    /// Base-two logarithm of the table height this run picked.
    pub log_height: u32,
    /// Committed cells: committed columns times height.
    pub committed_cells: usize,
    /// Cells fixed at setup: preprocessed columns times height.
    pub preprocessed_cells: usize,
    /// Tuple slots the table offers its channels: declared flushes times height.
    ///
    /// Lookup flushes and bus declarations both count.
    ///
    /// A row whose multiplicity is zero still occupies its slot.
    pub bus_flushes: usize,
    /// Zerocheck rounds in which this table's columns are still live.
    ///
    /// The batched zerocheck binds one variable per round, so a table is live for its height.
    pub sumcheck_rounds: usize,
    /// Bytes of column values opened for this table at the zerocheck point.
    ///
    /// One challenge-field element per committed or preprocessed column.
    ///
    /// The commitment scheme's own query phase is shared by the batch and not split here.
    pub opening_bytes: usize,
    /// Upper bound on the prover's scratch storage for this table, in bytes.
    ///
    /// ```text
    ///     folded columns : (committed + preprocessed) * height / 2 * |EF|
    ///     lookup leaves  : 2 * lookup flushes * height * |EF|
    ///     bus leaves     : bus declarations * height * |EF|
    /// ```
    ///
    /// The first fold turns every column into a challenge-field column of half the height.
    ///
    /// A lookup leaf is a fraction, so it holds two elements; a bus leaf is a product factor.
    ///
    /// The three are summed, since nothing guarantees one is freed before the next exists.
    pub peak_temporary_bytes: usize,
}

/// What every table of one run costs, in declaration order.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CostReport {
    /// One entry per table.
    tables: Vec<TableCost>,
}

impl CostReport {
    /// Price every table of a run from its declaration.
    ///
    /// `EF` is the challenge field, whose encoded width prices each opened or folded value.
    pub(super) fn new<EF: Field>(tables: &[TableDeclaration], run: &Run) -> Self {
        // Fixed-width encoding of one challenge-field element.
        let element = EF::order().bits().div_ceil(8) as usize;

        let tables = tables
            .iter()
            .zip(run.log_heights())
            .map(|(table, &log_height)| {
                let height = 1usize << log_height;
                let columns = table.columns();
                let opened = columns.committed + columns.preprocessed;
                let lookups = table.flushes().len();
                let buses = table.bus_declarations();

                let folded = opened * (height / 2);
                let leaves = 2 * lookups * height + buses * height;

                TableCost {
                    log_height,
                    committed_cells: columns.committed * height,
                    preprocessed_cells: columns.preprocessed * height,
                    bus_flushes: (lookups + buses) * height,
                    sumcheck_rounds: log_height as usize,
                    opening_bytes: opened * element,
                    peak_temporary_bytes: (folded + leaves) * element,
                }
            })
            .collect();

        Self { tables }
    }

    /// Cost of every table, in declaration order.
    #[must_use]
    pub fn tables(&self) -> &[TableCost] {
        &self.tables
    }

    /// Cost of the whole run.
    ///
    /// Counts and bytes add up across tables.
    ///
    /// The rounds are those of the batched zerocheck, which the tallest table fixes.
    ///
    /// The height is the tallest table's.
    #[must_use]
    pub fn total(&self) -> TableCost {
        self.tables
            .iter()
            .fold(TableCost::default(), |total, table| TableCost {
                log_height: total.log_height.max(table.log_height),
                committed_cells: total.committed_cells + table.committed_cells,
                preprocessed_cells: total.preprocessed_cells + table.preprocessed_cells,
                bus_flushes: total.bus_flushes + table.bus_flushes,
                sumcheck_rounds: total.sumcheck_rounds.max(table.sumcheck_rounds),
                opening_bytes: total.opening_bytes + table.opening_bytes,
                peak_temporary_bytes: total.peak_temporary_bytes + table.peak_temporary_bytes,
            })
    }
}
