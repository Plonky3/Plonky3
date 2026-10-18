//! Weighting Boolean columns by their rows, dense or packed.

use alloc::vec::Vec;

use p3_binary_field::{BitCoordinates, TowerLevel};
use p3_challenger::fs::TranscriptField;
use p3_field::{ExtensionField, Field, PrimeCharacteristicRing};
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use p3_multilinear_util::split_eq::SplitEq;
use p3_sumcheck::layout::{ColumnView, Table};

use super::{BooleanTraceCommitment, WORD_BITS};
use crate::boolean::BooleanBackend;

/// Rows one task sums when a dense column is weighted.
///
/// Large enough that dispatching a task is amortized over many rows.
///
/// Small enough to leave real parallelism at the trace heights this crate commits.
const ROW_GRAIN: usize = 1 << 12;

/// Bytes one packed word holds, each indexing one subset-sum table.
const BYTES_PER_WORD: usize = WORD_BITS / 8;

/// Row blocks of 64 rows one task sums with one set of subset-sum tables.
///
/// Large enough that allocating the task's tables is amortised.
/// Small enough to keep every core fed at the heights a wide trace reaches.
pub(super) const BLOCKS_PER_TASK: usize = 16;

/// Minimum selected columns that amortize the shared subset-sum tables.
const MIN_SHARED_COLUMNS: usize = 127;

impl<EF, B> BooleanTraceCommitment<EF, B>
where
    EF: BitCoordinates + ExtensionField<B::Val>,
    B: BooleanBackend<EF>,
    B::Val: TranscriptField + TowerLevel,
{
    /// Evaluate every column at one row point, and, when `next` holds, one row ahead of it.
    ///
    /// Both views weight the same table, so one equality table over the batch's point
    /// serves them both: the successor weights are that table shifted up by one row.
    ///
    /// The successor vector is empty unless it is asked for.
    #[cfg(test)]
    pub(super) fn evaluate_views(
        table: &Table<B::Val>,
        point: &Point<EF>,
        next: bool,
    ) -> (Vec<EF>, Vec<EF>) {
        let mut weights = SplitEq::<EF, EF>::new_packed(point, EF::ONE).materialize();
        let current = Self::weighted_columns(table, &weights);
        if !next {
            return (current, Vec::new());
        }

        // Row x reads row x + 1, so row z carries the weight of the row before it.
        //
        //     W[0] = 0, W[z] = eq[z - 1]
        //
        // The last row reads itself, so it also carries its own weight.
        let rows = weights.as_mut_slice();
        let last = rows.len() - 1;
        let repeated = rows[last];
        rows.copy_within(..last, 1);
        rows[0] = EF::ZERO;
        rows[last] += repeated;

        let successor = Self::weighted_columns(table, &weights);
        (current, successor)
    }

    /// Evaluate only the requested views, sharing packed-column work for dense selections and scanning sparse selections separately.
    pub(super) fn evaluate_requested(
        table: &Table<B::Val>,
        point: &Point<EF>,
        current: &[usize],
        next: &[usize],
    ) -> (Vec<EF>, Vec<EF>) {
        let mut weights = SplitEq::<EF, EF>::new_packed(point, EF::ONE).materialize();
        let current = Self::weighted_selected(table, &weights, current);
        if next.is_empty() {
            return (current, Vec::new());
        }
        let rows = weights.as_mut_slice();
        let last = rows.len() - 1;
        let repeated = rows[last];
        rows.copy_within(..last, 1);
        rows[0] = EF::ZERO;
        rows[last] += repeated;
        let successor = Self::weighted_selected(table, &weights, next);
        (current, successor)
    }

    fn weighted_selected(table: &Table<B::Val>, weights: &Poly<EF>, columns: &[usize]) -> Vec<EF> {
        if columns.iter().copied().eq(0..table.num_polys()) {
            return Self::weighted_columns(table, weights);
        }
        // Shared tables amortize their setup on wide, dense packed selections of full row words.
        // Smaller selections retain the direct scan, which is faster below this crossover.
        if table.packed_bits().is_some()
            && table.num_variables() >= WORD_BITS.ilog2() as usize
            && columns.len() >= table.num_polys().div_ceil(2).max(MIN_SHARED_COLUMNS)
        {
            let all = Self::weighted_columns(table, weights);
            return columns.iter().map(|&column| all[column]).collect();
        }
        columns
            .par_iter()
            .map(|&index| {
                let column = table.column(index);
                if column.as_dense().is_some() {
                    return Self::weighted_column(column, weights.as_slice());
                }
                weights
                    .as_slice()
                    .chunks(WORD_BITS)
                    .enumerate()
                    .map(|(block, weights)| {
                        let mut word = column
                            .boolean_word(block)
                            .expect("a packed column has words");
                        let mut sum = EF::ZERO;
                        while word != 0 {
                            sum += weights[word.trailing_zeros() as usize];
                            word &= word - 1;
                        }
                        sum
                    })
                    .sum::<EF>()
            })
            .collect()
    }

    /// Sum every column of a table against one weight per row.
    fn weighted_columns(table: &Table<B::Val>, weights: &Poly<EF>) -> Vec<EF> {
        let rows = weights.as_slice();
        // A packed table is summed word by word, so no cell is decoded.
        if let Some(words) = table.packed_bits() {
            return packed_column_sums(words, rows);
        }
        table
            .par_columns()
            .map(|column| Self::weighted_column(column, rows))
            .collect()
    }

    /// Sum one dense column against one weight per row.
    ///
    /// A cell is zero or one, so a value adds up the weights of the rows holding a one.
    ///
    /// # Panics
    ///
    /// Panics on a packed column, which [`packed_column_sums`] sums a whole table at a time.
    fn weighted_column(column: ColumnView<'_, B::Val>, rows: &[EF]) -> EF {
        let cells = column
            .as_dense()
            .expect("a packed table is summed word by word");

        // A dense column splits into runs of rows, each task summing the weights it selects.
        // Addition is order independent, so the split the pool chooses does not reach the sum.
        cells
            .par_chunks(ROW_GRAIN)
            .zip(rows.par_chunks(ROW_GRAIN))
            .map(|(cells, rows)| {
                cells
                    .iter()
                    .zip(rows)
                    .filter(|&(&cell, _)| cell == <B::Val>::ONE)
                    .map(|(_, &weight)| weight)
                    .sum::<EF>()
            })
            .sum()
    }
}

/// Sum each packed column's row weights over the rows where it holds one.
///
/// # Algorithm
///
/// Row `64 * b + j` of column `c` is bit `j` of `words[b][c]`.
///
/// A row block's 64 weights fall into eight bytes of eight rows each.
///
/// Every subset of one byte's rows gets its sum tabulated once per block:
///
/// ```text
///     table[g][s] = sum of weight(64 * b + 8 * g + j) over the set bits j of s
///     column c   += table[0][byte 0 of the word] + ... + table[7][byte 7 of the word]
/// ```
///
/// So a word costs eight table reads, however many of its bits are set.
///
/// The tables are shared by every column of the block, whose words are contiguous.
///
/// # Arguments
///
/// - `words`: one row per block of 64 rows, one entry per column.
/// - `row_weights`: one weight per row, at least as many as the column's rows.
pub(super) fn packed_column_sums<EF: Field>(
    words: &RowMajorMatrix<u64>,
    row_weights: &[EF],
) -> Vec<EF> {
    let width = words.width;
    debug_assert!(
        row_weights.len() > (words.values.len() / width).saturating_sub(1) * WORD_BITS,
        "every row block needs at least one weight"
    );
    words
        .values
        .par_chunks(width * BLOCKS_PER_TASK)
        .enumerate()
        .par_fold_reduce(
            || EF::zero_vec(width),
            |mut sums, (task, blocks)| {
                let mut tables = EF::zero_vec(BYTES_PER_WORD << 8);
                let (tables, _) = tables.as_chunks_mut::<256>();
                for (offset, block) in blocks.chunks_exact(width).enumerate() {
                    let first_row = (task * BLOCKS_PER_TASK + offset) * WORD_BITS;
                    for (group, table) in tables.iter_mut().enumerate() {
                        // Rows past the column's end carry no weight; their bits are zero.
                        let weight = |row: usize| {
                            row_weights
                                .get(first_row + 8 * group + row)
                                .copied()
                                .unwrap_or(EF::ZERO)
                        };
                        // Each subset extends the one without its lowest row by that row.
                        for subset in 1..256usize {
                            table[subset] = table[subset & (subset - 1)]
                                + weight(subset.trailing_zeros() as usize);
                        }
                    }
                    for (sum, &word) in sums.iter_mut().zip(block) {
                        *sum += tables
                            .iter()
                            .zip(word.to_le_bytes())
                            .map(|(table, byte)| table[usize::from(byte)])
                            .sum::<EF>();
                    }
                }
                sums
            },
            |mut left, right| {
                left.iter_mut()
                    .zip(right)
                    .for_each(|(left, right)| *left += right);
                left
            },
        )
}
