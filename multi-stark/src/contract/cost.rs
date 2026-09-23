//! What one run of a statement costs, table by table, read off the shape alone.
//!
//! Every figure comes from the declaration and the run, never from a witness or a proof.
//!
//! A verifier can therefore compute the same report as the prover.
//!
//! The counts a proof also carries are pinned against real proofs by tests.
//!
//! A figure too large for a `usize` saturates at `usize::MAX` instead of wrapping.

use alloc::vec::Vec;

use p3_field::Field;

use crate::contract::run::Run;
use crate::contract::table::TableDeclaration;

/// What one table costs in one run.
///
/// Every count is exact for the shape.
///
/// The byte figures are not: the opened bytes use the element's width, and the peak is an estimate.
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
    /// Leaves the table adds to the one lookup tree every table shares.
    ///
    /// One leaf per lookup tuple per row.
    ///
    /// The whole run's figure is the tree's leaf layer, padded to a power of two.
    pub lookup_leaves: usize,
    /// Zerocheck rounds in which this table's columns are still live.
    ///
    /// The batched zerocheck binds one variable per round, so a table is live for its height.
    ///
    /// The whole run's figure is the round count of the proof:
    ///
    /// ```text
    ///     rounds = max(tallest log height, log2(padded lookup leaves))
    /// ```
    pub sumcheck_rounds: usize,
    /// Challenge-field values the commitment scheme opens for this table.
    ///
    /// Every batch the opening schedule gives the table counts:
    ///
    /// ```text
    ///     zerocheck point   every column, plus its next-row view where the table reads one
    ///     indexed points    one position per indexed read, the columns of an indexed table
    ///     bus point         the columns the bus declarations read
    /// ```
    ///
    /// Both committed windows count, main and preprocessed.
    pub opened_values: usize,
    /// Opened values times the width of one challenge-field element, in bytes.
    ///
    /// The width is the element's bit length rounded up to bytes.
    ///
    /// A serializer may write a small element in fewer bytes.
    ///
    /// The commitment scheme's own query phase is shared by the batch and not split here.
    pub opening_bytes: usize,
    /// Estimate of the prover's scratch storage for this table, in bytes.
    ///
    /// ```text
    ///     folded columns : (committed + preprocessed) * height / 2 * |EF|
    ///     lookup leaves  : 2 * lookup leaves * |EF|
    ///     bus leaves     : bus declarations * height * |EF|
    /// ```
    ///
    /// The first fold turns every column into a challenge-field column of half the height.
    ///
    /// A lookup leaf is a fraction, so it holds two elements; a bus leaf is a product factor.
    ///
    /// The three are summed, since nothing guarantees one is freed before the next exists.
    ///
    /// The whole run's figure adds the rest of the lookup tree:
    ///
    /// ```text
    ///     padding leaves : 2 * (padded leaves - leaves) * |EF|
    ///     reduced layers : 2 * (padded leaves - 1) * |EF|
    /// ```
    ///
    /// The prover's own sumcheck buffers come on top, so this is no upper bound.
    pub estimated_peak_bytes: usize,
}

/// What every table of one run costs, in declaration order.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CostReport {
    /// One entry per table.
    tables: Vec<TableCost>,
    /// Width of one challenge-field element, in bytes.
    element_bytes: usize,
}

impl CostReport {
    /// Price every table of a run from its declaration.
    ///
    /// `EF` is the challenge field, whose width prices each opened or folded value.
    pub(super) fn new<EF: Field>(tables: &[TableDeclaration], run: &Run) -> Self {
        // `bits` rather than the order's bit length: a binary field of 2^128 elements is 16 bytes.
        let element_bytes = EF::bits().div_ceil(8);

        let tables = tables
            .iter()
            .zip(run.log_heights())
            .map(|(table, &log_height)| {
                // A height past the pointer width saturates, as every figure it scales does.
                let height = 1usize.checked_shl(log_height).unwrap_or(usize::MAX);
                let rows = |count: usize| count.saturating_mul(height);

                let columns = table.columns();
                let opened = table.opened_values();
                let opened_values = opened.main.saturating_add(opened.preprocessed);
                let buses = table.bus_declarations();
                let lookup_leaves = rows(table.lookup_tuples());

                let folded = columns
                    .committed
                    .saturating_add(columns.preprocessed)
                    .saturating_mul(height / 2);
                let scratch = folded
                    .saturating_add(lookup_leaves.saturating_mul(2))
                    .saturating_add(rows(buses));

                TableCost {
                    log_height,
                    committed_cells: rows(columns.committed),
                    preprocessed_cells: rows(columns.preprocessed),
                    bus_flushes: rows(table.flushes().len().saturating_add(buses)),
                    lookup_leaves,
                    sumcheck_rounds: log_height as usize,
                    opened_values,
                    opening_bytes: opened_values.saturating_mul(element_bytes),
                    estimated_peak_bytes: scratch.saturating_mul(element_bytes),
                }
            })
            .collect();

        Self {
            tables,
            element_bytes,
        }
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
    /// The height is the tallest table's.
    ///
    /// The lookup tree is shared, so its padding and its reduced layers are counted here once.
    ///
    /// The rounds are those of the batched zerocheck.
    ///
    /// It runs over the taller of the tallest table and the lookup tree.
    #[must_use]
    pub fn total(&self) -> TableCost {
        let summed = self
            .tables
            .iter()
            .fold(TableCost::default(), |total, table| TableCost {
                log_height: total.log_height.max(table.log_height),
                committed_cells: total.committed_cells.saturating_add(table.committed_cells),
                preprocessed_cells: total
                    .preprocessed_cells
                    .saturating_add(table.preprocessed_cells),
                bus_flushes: total.bus_flushes.saturating_add(table.bus_flushes),
                lookup_leaves: total.lookup_leaves.saturating_add(table.lookup_leaves),
                sumcheck_rounds: total.sumcheck_rounds.max(table.sumcheck_rounds),
                opened_values: total.opened_values.saturating_add(table.opened_values),
                opening_bytes: total.opening_bytes.saturating_add(table.opening_bytes),
                estimated_peak_bytes: total
                    .estimated_peak_bytes
                    .saturating_add(table.estimated_peak_bytes),
            });

        // No lookup means no tree, and no round beyond the tallest table.
        if summed.lookup_leaves == 0 {
            return summed;
        }

        // The prover pads the leaf layer to a power of two and keeps every layer above it.
        let depth = ceil_log2(summed.lookup_leaves);
        let padded = 1usize.checked_shl(depth).unwrap_or(usize::MAX);
        let padding = padded.saturating_sub(summed.lookup_leaves);
        let reduced = padded - 1;
        let tree = padding
            .saturating_add(reduced)
            .saturating_mul(2)
            .saturating_mul(self.element_bytes);

        TableCost {
            lookup_leaves: padded,
            sumcheck_rounds: summed.sumcheck_rounds.max(depth as usize),
            estimated_peak_bytes: summed.estimated_peak_bytes.saturating_add(tree),
            ..summed
        }
    }
}

/// Smallest `k` with `2^k >= n`, for a positive `n`.
const fn ceil_log2(n: usize) -> u32 {
    usize::BITS - (n - 1).leading_zeros()
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_baby_bear::BabyBear;
    use p3_binary_field::BinaryField128;
    use p3_field::extension::BinomialExtensionField;

    use super::*;
    use crate::contract::table::{ColumnCounts, HeightRange, MAX_COLUMNS, MAX_LOG_HEIGHT};

    type EF = BinomialExtensionField<BabyBear, 4>;

    #[test]
    #[cfg(target_pointer_width = "64")]
    fn a_figure_too_large_to_count_saturates() {
        // Two of the widest tables at the tallest height, each 2^60 cells.
        let widest = TableDeclaration::shaped(
            ColumnCounts {
                committed: MAX_COLUMNS,
                preprocessed: 0,
                public: 0,
            },
            HeightRange::exactly(MAX_LOG_HEIGHT),
        );
        let run = Run::new([0; 32], vec![MAX_LOG_HEIGHT; 2], 0);
        let report = CostReport::new::<EF>(&[widest.clone(), widest], &run);

        // Each table's scratch is 2^59 elements of sixteen bytes, which is 2^63 and still fits.
        assert_eq!(report.tables()[0].committed_cells, 1 << 60);
        assert_eq!(report.tables()[0].estimated_peak_bytes, 1 << 63);

        // Two of them reach 2^64, which saturates instead of wrapping to zero.
        let total = report.total();
        assert_eq!(total.committed_cells, 1 << 61);
        assert_eq!(total.estimated_peak_bytes, usize::MAX);
    }

    #[test]
    fn an_element_is_priced_at_its_width() {
        // One committed column, opened once, in two challenge fields of 128 bits.
        let tables = [TableDeclaration::shaped(
            ColumnCounts {
                committed: 1,
                preprocessed: 0,
                public: 0,
            },
            HeightRange::exactly(1),
        )];
        let run = Run::new([0; 32], vec![1], 0);
        let bytes = |report: CostReport| report.tables()[0].opening_bytes;

        // The binary field's order has 129 bits, yet each element is 16 bytes wide.
        assert_eq!(bytes(CostReport::new::<BinaryField128>(&tables, &run)), 16);
        assert_eq!(bytes(CostReport::new::<EF>(&tables, &run)), 16);
    }

    #[test]
    fn the_rounds_ceil_the_lookup_tree() {
        // Exact powers keep their exponent, and anything above rounds up.
        assert_eq!(ceil_log2(1), 0);
        assert_eq!(ceil_log2(2), 1);
        assert_eq!(ceil_log2(24), 5);
        assert_eq!(ceil_log2(32), 5);
        assert_eq!(ceil_log2(33), 6);
        assert_eq!(ceil_log2(usize::MAX), usize::BITS);
    }
}
