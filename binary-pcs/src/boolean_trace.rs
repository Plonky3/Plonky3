//! A batched trace of Boolean columns, committed as bits and opened as columns.
//!
//! ```text
//!     in    one table per AIR instance, cells in {0, 1}
//!     out   one bit witness, committed packed, opened column by column
//! ```
//!
//! # From a column claim to a bit claim
//!
//! Every column occupies one contiguous run of the bit witness.
//!
//! A run starts at a multiple of its own length, so its address is a bit string.
//!
//! ```text
//!     column of slot s, 2^a rows  ->  bits [s * 2^a, (s + 1) * 2^a)
//!     its value at a point r      ->  the witness at (bits of s, r)
//! ```
//!
//! The slot address therefore prefixes the row point, and nothing is re-indexed.
//!
//! The planner that assigns the slots is the one the commitment schemes use.
//!
//! # Work stays inside the column
//!
//! The slot prefix is fixed before the ring-switch sumcheck starts. For a complete opening
//! of one table, every column at the current row and either no column or every column at
//! the next row, all its columns are combined at one fresh random column point after their
//! claimed values are bound, so each batch uses one ring switch.
//! Other valid protocols retain one reduction per opened column.
//!
//! If one element holds `2^d_log` bits, a column folds `2^max(a - d_log, 0)` elements.
//!
//! Opening `W` equal-height columns shares the row equality weights across columns. The
//! optimized complete-table route scans those weights once per batch and opens one padded
//! stacked point; subset, reordered, mixed-height and mixed-view protocols use the
//! per-column route.
//!
//! # Booleanity is still free
//!
//! The packing is a bijection between bit strings and the elements a commitment holds.
//!
//! No commitment exists to a function the hypercube sends outside `{0, 1}`.
//!
//! A cell outside `{0, 1}` is refused at commitment rather than range-checked.
//!
//! The opened values are the bits, so a proof whose own trace disagrees with them fails.
//!
//! # What a batch opens
//!
//! A batch names columns read at the current row, and columns read one row ahead.
//!
//! The successor view is repeat-last: row `x` reads row `x + 1`, and the last row itself.
//!
//! It weights the column by the equality table shifted up by one row, which the
//! reduction underneath answers beside the reading at the point.
//!
//! The step stays inside the table's own row variables, so it never enters a slot address.
//!
//! ```text
//!     one table, every batch reading all columns    ->  one reduction per batch
//!     of both views, or of the current alone            over one shared column point
//!     anything else                                 ->  one reduction per column read
//! ```
//!
//! A column named by both views of a batch is one reduction answering both readings.
//!
//! The values of a batch are its current ones, in the order the batch names them,
//! followed by its next ones in the order the batch names those.
//!
//! An AIR whose constraints read only the current row names no successor column.

use alloc::vec::Vec;

use p3_binary_dft::EncodableLevel;
use p3_binary_field::{PackedGf2, PackedGf2x64, TowerLevel};
use p3_challenger::fs::TranscriptField;
use p3_challenger::{CanObserve, CanSampleUniformBits, FieldChallenger, GrindingChallenger};
use p3_commit::{Mmcs, MultilinearPcs};
use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use p3_multilinear_util::split_eq::SplitEq;
use p3_sumcheck::layout::{ColumnView, Table, TablePlacement, plan_stacked_layout};
use p3_sumcheck::ring_switch::bits::BitRingSwitch;
use p3_sumcheck::{
    OpeningEvals, OpeningProtocol, PrescribedOpeningSecurity, PrescribedPointPcs, TableShape,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::boolean::{
    BitOpening, BitReadings, BooleanMultilinearPcs, BooleanPcs, BooleanPcsError, BooleanProof,
};
use crate::boolean_trace_transcript::{
    ColumnBatchProverTranscript, ColumnBatchShape, ColumnBatchVerifierTranscript,
};
use crate::fold::{ChallengeField, FoldAlphabet};
use crate::packing::Coordinates;
use crate::params::BinaryPcsConfig;
use crate::prover::BinaryPcsProverData;

/// Bits one word of the staging buffer holds.
const WORD_BITS: usize = 64;

/// Rows one task sums when a dense column is weighted.
///
/// Large enough that dispatching a task is amortized over many rows.
///
/// Small enough to leave real parallelism at the trace heights this crate commits.
const ROW_GRAIN: usize = 1 << 12;

/// Bytes one packed word holds, each indexing one subset-sum table.
const BYTES_PER_WORD: usize = WORD_BITS / 8;

/// Adjacent packed columns one gather task copies, one source line's worth of words.
const GATHER_GROUP: usize = 8;

/// Row blocks of 64 rows one task sums with one set of subset-sum tables.
///
/// Large enough that allocating the task's tables is amortised.
/// Small enough to keep every core fed at the heights a wide trace reaches.
const BLOCKS_PER_TASK: usize = 16;

/// A commitment to a batch of Boolean trace tables, packed into one bit witness.
///
/// The committed object is the bits, so the codeword is as short as the alphabet allows.
///
/// The tables themselves are retained, for a prover that evaluates constraints over them.
pub struct BooleanTracePcs<EF: EncodableLevel, MT, MX> {
    /// The bit commitment every column claim is discharged against.
    inner: BooleanPcs<EF, MT, MX>,
}

impl<EF, MT, MX> BooleanTracePcs<EF, MT, MX>
where
    EF: ChallengeField<EF>
        + EncodableLevel
        + TranscriptField
        + TowerLevel
        + FoldAlphabet<EF>
        + Coordinates,
    MT: Mmcs<EF>,
    MX: Mmcs<EF, Error = MT::Error>,
{
    /// Build a commitment over a batch of tables stacking to `num_variables` bits.
    ///
    /// The arity is the one the layout planner derives from the table shapes.
    ///
    /// # Errors
    ///
    /// Returns an error unless the schedule was derived for `(EF, EF)`.
    /// Returns an error unless the schedule commits exactly the elements the packing holds.
    pub fn new(
        config: BinaryPcsConfig,
        mmcs: MT,
        round_mmcs: MX,
        num_variables: usize,
    ) -> Result<Self, BooleanTraceError<EF, MT::Error>> {
        BooleanPcs::new(config, mmcs, round_mmcs, num_variables)
            .map(|inner| Self { inner })
            .map_err(BooleanTraceError::Boolean)
    }

    /// Variables the stacked bit witness has, so `2^n` bits in all.
    #[must_use]
    pub const fn num_variables(&self) -> usize {
        self.inner.num_variables()
    }

    /// Where each table's columns land in the bit witness, planned from the shapes.
    ///
    /// # Errors
    ///
    /// Returns an error unless the shapes stack to the committed arity.
    fn placements(
        &self,
        shapes: &[TableShape],
    ) -> Result<Vec<TablePlacement>, BooleanTraceError<EF, MT::Error>> {
        // Prover and verifier both plan from the public shapes, so neither picks its own.
        let (arity, placements) = plan_stacked_layout(shapes);
        if arity != self.num_variables() {
            return Err(BooleanTraceError::StackedArity {
                expected: self.num_variables(),
                actual: arity,
            });
        }
        Ok(placements)
    }

    /// The bit claims one opening protocol raises, in transcript order.
    ///
    /// One claim per column a batch reads, that batch's point prefixed by the slot address,
    /// asking for whichever readings the claim's entry in the plan names.
    fn opening_claims(
        protocol: &OpeningProtocol,
        points: &[Point<EF>],
        placements: &[TablePlacement],
    ) -> (Vec<ColumnClaim>, Vec<BitOpening<EF>>) {
        let shapes = protocol.table_shapes();

        // Placements arrive largest table first, so index them by the table each one owns.
        let mut by_table = alloc::vec![None; shapes.len()];
        for placement in placements {
            by_table[placement.idx()] = Some(placement);
        }

        let claims = column_claims(protocol);
        let openings = claims
            .iter()
            .map(|claim| {
                let placement =
                    by_table[claim.table].expect("the planner places every supplied shape");
                BitOpening {
                    // Slot address as the leading coordinates, the row point as the trailing ones.
                    point: placement.selectors()[claim.column].lift_prefix(&points[claim.opening]),
                    // The successor view steps within the rows, never into the slot address.
                    row_variables: shapes[claim.table].num_variables(),
                    current: claim.current_at.is_some(),
                    next: claim.next_at.is_some(),
                }
            })
            .collect();
        (claims, openings)
    }

    /// Validate all public opening metadata without constructing per-column claims.
    fn validate_opening(
        &self,
        protocol: &OpeningProtocol,
        points: &[Point<EF>],
    ) -> Result<Vec<TablePlacement>, BooleanTraceError<EF, MT::Error>> {
        let shapes = protocol.table_shapes();
        let placements = self.placements(&shapes)?;
        if points.len() != protocol.num_openings() {
            return Err(BooleanTraceError::PointCount {
                expected: protocol.num_openings(),
                actual: points.len(),
            });
        }

        for ((table, _), point) in protocol.iter_openings().zip(points) {
            if point.num_variables() != shapes[table].num_variables() {
                return Err(BooleanTraceError::PointArity {
                    table,
                    expected: shapes[table].num_variables(),
                    actual: point.num_variables(),
                });
            }
        }
        Ok(placements)
    }

    /// Validate retained source shapes before any sampled point or opening transcript is used.
    fn validate_source_shapes(
        tables: &[Table<EF>],
        protocol: &OpeningProtocol,
    ) -> Result<(), BooleanTraceError<EF, MT::Error>> {
        let expected = protocol.table_shapes();
        if tables.len() != expected.len() {
            return Err(BooleanTraceError::TableCountMismatch {
                expected: expected.len(),
                actual: tables.len(),
            });
        }
        for (table, expected) in expected.iter().copied().enumerate() {
            let actual = tables[table].shape();
            if actual != expected {
                return Err(BooleanTraceError::TableShapeMismatch {
                    table,
                    expected,
                    actual,
                });
            }
        }
        Ok(())
    }

    /// The width and views of the complete single-table shape the optimized route handles.
    ///
    /// Every batch reads the whole width at the current row, and either none of it or all
    /// of it one row ahead, the same way in every batch.
    fn batched_shape(&self, protocol: &OpeningProtocol) -> Option<(usize, bool)> {
        let shapes = protocol.table_shapes();
        if shapes.len() != 1 || protocol.num_openings() == 0 {
            return None;
        }
        let width = shapes[0].width();
        let columns = (0..width).collect::<Vec<_>>();
        // The first batch fixes the views, and every other batch has to agree with it.
        let next = protocol
            .iter_openings()
            .next()
            .is_some_and(|(_, batch)| !batch.next().is_empty());
        let complete = |read: &[usize], asked: bool| {
            if asked {
                read == columns.as_slice()
            } else {
                read.is_empty()
            }
        };
        protocol
            .iter_openings()
            .all(|(table, batch)| {
                table == 0 && complete(batch.current(), true) && complete(batch.next(), next)
            })
            .then_some((width, next))
    }

    /// Evaluate every column at one row point, and, when `next` holds, one row ahead of it.
    ///
    /// Both views weight the same table, so one equality table over the batch's point
    /// serves them both: the successor weights are that table shifted up by one row.
    ///
    /// The successor vector is empty unless it is asked for.
    fn evaluate_views(table: &Table<EF>, point: &Point<EF>, next: bool) -> (Vec<EF>, Vec<EF>) {
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

    /// Sum every column of a table against one weight per row.
    fn weighted_columns(table: &Table<EF>, weights: &Poly<EF>) -> Vec<EF> {
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
    fn weighted_column(column: ColumnView<'_, EF>, rows: &[EF]) -> EF {
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
                    .filter(|&(&cell, _)| cell == EF::ONE)
                    .map(|(_, &weight)| weight)
                    .sum::<EF>()
            })
            .sum()
    }

    /// Evaluate the zero-padded column-value vector at its sampled column point.
    fn combine_columns(values: &[EF], column_point: &Point<EF>) -> EF {
        let padded_len = 1usize << column_point.num_variables();
        let mut padded = Vec::with_capacity(padded_len);
        padded.extend_from_slice(values);
        padded.resize(padded_len, EF::ZERO);
        SplitEq::<EF, EF>::new_packed(column_point, EF::ONE).eval_ext(Poly::new(padded).as_view())
    }

    /// Gather the Boolean cells of every table into one bit witness.
    ///
    /// # Errors
    ///
    /// - The shapes do not stack to the committed arity.
    /// - A cell holds neither zero nor one, so it addresses no bit.
    fn gather_bits(
        &self,
        tables: &[Table<EF>],
    ) -> Result<Vec<PackedGf2x64>, BooleanTraceError<EF, MT::Error>> {
        let shapes = tables.iter().map(Table::shape).collect::<Vec<_>>();
        let placements = self.placements(&shapes)?;

        // One word per sixty-four bits of the padded witness, the tail left at zero.
        let mut words = alloc::vec![0u64; 1 << (self.num_variables() - 6)];

        // A column of at least one word owns a run of whole words; a shorter one shares a word.
        //
        // Each entry is `(first word or cell, position in placement order, column)`.
        let mut runs = Vec::new();
        let mut short = Vec::new();
        for (position, placement) in placements.iter().enumerate() {
            let column_len = 1usize << tables[placement.idx()].num_variables();
            for (column, selector) in placement.selectors().iter().enumerate() {
                // A slot starts at a multiple of its own length, counted in cells.
                let offset = selector.index() * column_len;
                if column_len >= WORD_BITS {
                    runs.push((offset / WORD_BITS, position, column));
                } else {
                    short.push((offset, position, column));
                }
            }
        }

        // The runs are disjoint, so they are carved out of the witness and filled in parallel.
        //
        // Packed columns of one table go in groups of adjacent columns: one block row holds
        // their words side by side, so a group reads each source line once.
        runs.sort_unstable_by_key(|&(offset, ..)| offset);
        let mut rest = words.as_mut_slice();
        let mut consumed = 0;
        let mut carved = Vec::with_capacity(runs.len());
        for (offset, position, column) in runs {
            let len = (1usize << tables[placements[position].idx()].num_variables()) / WORD_BITS;
            let (_, tail) = core::mem::take(&mut rest).split_at_mut(offset - consumed);
            let (run, tail) = tail.split_at_mut(len);
            carved.push((position, column, run));
            rest = tail;
            consumed = offset + len;
        }
        carved.sort_unstable_by_key(|&(position, column, _)| (position, column));
        let mut groups: Vec<Vec<(usize, usize, &mut [u64])>> = Vec::new();
        for entry in carved {
            let packed = tables[placements[entry.0].idx()].packed_bits().is_some();
            match groups.last_mut() {
                Some(group)
                    if packed
                        && group.len() < GATHER_GROUP
                        && group[0].0 == entry.0
                        && group[0].1 / GATHER_GROUP == entry.1 / GATHER_GROUP =>
                {
                    group.push(entry);
                }
                _ => groups.push(alloc::vec![entry]),
            }
        }

        // A refusal names the first offending column in placement order, as a serial walk would.
        let long_refusal = groups
            .into_par_iter()
            .filter_map(|group| gather_runs(tables, &placements, group))
            .min();

        // A short column's bits are set in place, inside a word other columns may share.
        let mut short_refusal = None;
        for (offset, position, column) in short {
            let table = &tables[placements[position].idx()];
            let view = table.column(column);
            let bits = if let Some(cells) = view.as_dense() {
                let Some(bits) = pack_word(cells) else {
                    short_refusal = Some((position, column));
                    break;
                };
                bits
            } else {
                view.boolean_word(0)
                    .expect("packed short column must expose its source word")
            };
            for cell in 0..view.len() {
                let index = offset + cell;
                words[index / WORD_BITS] |= ((bits >> cell) & 1) << (index % WORD_BITS);
            }
        }

        if let Some((position, column)) = long_refusal.into_iter().chain(short_refusal).min() {
            return Err(BooleanTraceError::NonBooleanCell {
                table: placements[position].idx(),
                column,
            });
        }

        // Lane `j` of a block is bit `j` of its word, which is the packing's own convention.
        Ok(words.into_iter().map(PackedGf2::new).collect())
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
fn packed_column_sums<EF: Field>(words: &RowMajorMatrix<u64>, row_weights: &[EF]) -> Vec<EF> {
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

/// Fill the word runs of one group of columns, returning the first column refused.
///
/// A group is one dense column, or up to [`GATHER_GROUP`] adjacent packed columns of one table.
///
/// Each entry is `(position in placement order, column, the column's word run)`.
fn gather_runs<EF: Field>(
    tables: &[Table<EF>],
    placements: &[TablePlacement],
    mut group: Vec<(usize, usize, &mut [u64])>,
) -> Option<(usize, usize)> {
    let table = &tables[placements[group[0].0].idx()];
    if let Some(source) = table.packed_bits() {
        // Block row `b` holds word `b` of every column, so one row serves the whole group.
        for (block, row) in source.values.chunks_exact(source.width).enumerate() {
            for (_, column, run) in &mut group {
                run[block] = row[*column];
            }
        }
        return None;
    }
    let (position, column, run) = &mut group[0];
    let cells = table
        .column(*column)
        .as_dense()
        .expect("an unpacked table holds dense columns");
    // The slot is word aligned and fills whole words, so none is read back.
    let (chunks, rest) = cells.as_chunks::<WORD_BITS>();
    debug_assert!(rest.is_empty(), "a power-of-two column fills whole words");
    for (word, chunk) in run.iter_mut().zip(chunks) {
        let Some(bits) = pack_word(chunk) else {
            return Some((*position, *column));
        };
        *word = bits;
    }
    None
}

/// The bit of a cell holding zero or one, and nothing for any other value.
#[inline]
fn bit_of<EF: Field>(value: EF) -> Option<u64> {
    // A Boolean trace holds field zero and field one, and no third value.
    if value == EF::ZERO {
        Some(0)
    } else if value == EF::ONE {
        Some(1)
    } else {
        None
    }
}

/// Sixty-four Boolean cells as one word, the lowest cell in the lowest bit.
#[inline]
fn pack_word<EF: Field>(cells: &[EF]) -> Option<u64> {
    // Fold the run into a word; one non-Boolean cell leaves the whole word undefined.
    cells
        .iter()
        .enumerate()
        .try_fold(0u64, |word, (lane, &value)| {
            Some(word | (bit_of(value)? << lane))
        })
}

/// One bit claim of the per-column route: the column it reads, and where its values sit.
#[derive(Clone, Copy, Debug)]
struct ColumnClaim {
    /// Table the claim's batch opens.
    table: usize,
    /// Column of that table the claim reads.
    column: usize,
    /// Batch the claim belongs to, whose point it is lifted by.
    opening: usize,
    /// Where the reading at the point sits in the value run, when it is asked for.
    current_at: Option<usize>,
    /// Where the reading one row ahead sits in the value run, when it is asked for.
    next_at: Option<usize>,
}

/// One claim per column a batch reads, batches in protocol order.
///
/// A batch contributes one claim per entry of its current list, then one per successor
/// entry no current claim already answers. Each claim carries the positions of its own
/// values in the run, which is every batch's current values followed by its next ones.
fn column_claims(protocol: &OpeningProtocol) -> Vec<ColumnClaim> {
    let mut claims = Vec::new();
    let mut cursor = 0;
    for (opening, (table, batch)) in protocol.iter_openings().enumerate() {
        let next_cursor = cursor + batch.current().len();
        // Every successor entry is answered exactly once, so no claimed value goes unchecked.
        let mut answered = alloc::vec![false; batch.next().len()];
        for (offset, &column) in batch.current().iter().enumerate() {
            let at = batch
                .next()
                .iter()
                .zip(&answered)
                .position(|(&other, &taken)| other == column && !taken);
            if let Some(at) = at {
                answered[at] = true;
            }
            claims.push(ColumnClaim {
                table,
                column,
                opening,
                current_at: Some(cursor + offset),
                next_at: at.map(|at| next_cursor + at),
            });
        }
        for (at, &column) in batch.next().iter().enumerate() {
            if !answered[at] {
                claims.push(ColumnClaim {
                    table,
                    column,
                    opening,
                    current_at: None,
                    next_at: Some(next_cursor + at),
                });
            }
        }
        cursor = next_cursor + batch.next().len();
    }
    claims
}

/// Values one protocol opens: per batch, its current readings then its successor ones.
fn value_count(protocol: &OpeningProtocol) -> usize {
    protocol.iter_openings().map(|(_, batch)| batch.len()).sum()
}

/// Whether a claim plan writes each of the `len` value positions exactly once.
///
/// A position two claims write is one reading the plan answers twice, and a position no
/// claim writes is one the run leaves at whatever it was filled with.
fn covers_every_value(claims: &[ColumnClaim], len: usize) -> bool {
    let mut written = alloc::vec![false; len];
    for claim in claims {
        for at in [claim.current_at, claim.next_at].into_iter().flatten() {
            if at >= len || core::mem::replace(&mut written[at], true) {
                return false;
            }
        }
    }
    written.into_iter().all(|written| written)
}

/// Lay the readings every claim came back with out in the protocol's value order.
fn claim_values<EF: Field>(
    claims: &[ColumnClaim],
    readings: &[BitReadings<EF>],
    len: usize,
) -> Vec<EF> {
    // The zero fill stands only until the plan writes over it, which it does everywhere.
    debug_assert!(
        covers_every_value(claims, len),
        "the claim plan writes every value position exactly once",
    );
    let mut values = alloc::vec![EF::ZERO; len];
    for (claim, reading) in claims.iter().zip(readings) {
        if let Some(at) = claim.current_at {
            values[at] = reading
                .current
                .expect("a claim asking for the reading at the point carries it");
        }
        if let Some(at) = claim.next_at {
            values[at] = reading
                .next
                .expect("a claim asking for the reading one row ahead carries it");
        }
    }
    values
}

/// The readings every claim asks for, read back out of the protocol's value order.
fn claim_readings<EF: Field>(claims: &[ColumnClaim], values: &[EF]) -> Vec<BitReadings<EF>> {
    claims
        .iter()
        .map(|claim| BitReadings {
            current: claim.current_at.map(|at| values[at]),
            next: claim.next_at.map(|at| values[at]),
        })
        .collect()
}

/// Split the flat value run back into each batch's current and successor values.
fn opening_evals<EF: Field>(protocol: &OpeningProtocol, values: &[EF]) -> Vec<OpeningEvals<EF>> {
    let mut evals = Vec::with_capacity(protocol.num_openings());
    let mut cursor = 0;
    for (_, batch) in protocol.iter_openings() {
        let next_cursor = cursor + batch.current().len();
        let end = next_cursor + batch.next().len();
        evals.push(OpeningEvals::new(
            values[cursor..next_cursor].to_vec(),
            values[next_cursor..end].to_vec(),
        ));
        cursor = end;
    }
    evals
}

/// One point per opening batch, drawn from the transcript in batch order.
fn sample_points<EF, Challenger>(
    protocol: &OpeningProtocol,
    challenger: &mut Challenger,
) -> Vec<Point<EF>>
where
    EF: Field,
    Challenger: FieldChallenger<EF>,
{
    // Coordinates are drawn in the order the batches stream, one batch's point at a time.
    let shapes = protocol.table_shapes();
    protocol
        .iter_openings()
        .map(|(table, _)| {
            let num_variables = shapes[table].num_variables();
            Point::new((0..num_variables).map(|_| challenger.sample()).collect())
        })
        .collect()
}

/// The committed trace tables, held until the commitment is opened.
pub struct BooleanTraceData<EF: Field, MT: Mmcs<EF>> {
    /// Prover data of the bit commitment underneath.
    inner: BinaryPcsProverData<EF, EF, MT>,
    /// Source tables, lent back to whatever evaluates constraints over them.
    tables: Vec<Table<EF>>,
}

impl<EF: Field, MT: Mmcs<EF>> Clone for BooleanTraceData<EF, MT>
where
    BinaryPcsProverData<EF, EF, MT>: Clone,
{
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            tables: self.tables.clone(),
        }
    }
}

impl<EF: Field, MT: Mmcs<EF>> BooleanTraceData<EF, MT> {
    /// One committed table, in the order the tables were supplied.
    ///
    /// # Panics
    ///
    /// Panics on a table index this commitment does not hold.
    #[must_use]
    pub fn table(&self, index: usize) -> &Table<EF> {
        &self.tables[index]
    }
}

/// One opening of a Boolean trace: the column values, and the bit proof behind them.
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "EF: TowerLevel, MT::Commitment: Serialize, MT::MultiProof: Serialize, MX::Commitment: Serialize, MX::MultiProof: Serialize",
    deserialize = "EF: TowerLevel, MT::Commitment: Deserialize<'de>, MT::MultiProof: Deserialize<'de>, MX::Commitment: Deserialize<'de>, MX::MultiProof: Deserialize<'de>"
))]
pub struct BooleanTraceProof<EF: Field, MT: Mmcs<EF>, MX: Mmcs<EF>> {
    /// One value per column a batch reads, its current ones first, batches in transcript order.
    pub values: Vec<EF>,
    /// The bit commitment's own proof, answering for every value at once.
    pub opening: BooleanProof<EF, MT, MX>,
}

/// Why a Boolean trace could not be committed or opened.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum BooleanTraceError<EF, MmcsError> {
    /// The bit commitment underneath refused the witness or the opening.
    #[error(transparent)]
    Boolean(BooleanPcsError<EF, MmcsError>),

    /// The table shapes do not stack to the arity this commitment holds.
    #[error("the shapes stack to {actual} variables, the commitment holds {expected}")]
    StackedArity {
        /// Variables the commitment holds.
        expected: usize,
        /// Variables the shapes stack to.
        actual: usize,
    },

    /// The retained source tables do not have the exact shapes promised by the opening protocol.
    #[error("table {table} has shape {actual:?}, protocol requires {expected:?}")]
    TableShapeMismatch {
        /// Table whose retained shape disagreed with the protocol.
        table: usize,
        /// Shape the protocol describes.
        expected: TableShape,
        /// Shape the committed prover data retains.
        actual: TableShape,
    },

    /// The protocol and retained source table lists have different lengths.
    #[error("the prover retains {actual} tables, while the opening protocol describes {expected}")]
    TableCountMismatch {
        /// Number of tables described by the protocol.
        expected: usize,
        /// Number of tables retained by the prover.
        actual: usize,
    },

    /// A trace cell holds neither zero nor one, so it addresses no bit.
    #[error("column {column} of table {table} holds a cell outside the two Boolean values")]
    NonBooleanCell {
        /// Table holding the cell.
        table: usize,
        /// Column holding the cell.
        column: usize,
    },

    /// One point per opening batch is required.
    #[error("{actual} points against {expected} opening batches")]
    PointCount {
        /// Batches the protocol schedules.
        expected: usize,
        /// Points supplied.
        actual: usize,
    },

    /// A point does not name the rows of the table it opens.
    #[error("the point for table {table} names {actual} variables, expected {expected}")]
    PointArity {
        /// Table the point opens.
        table: usize,
        /// Variables that table's rows need.
        expected: usize,
        /// Variables the point names.
        actual: usize,
    },

    /// The proof carries a different number of values than the protocol opens.
    ///
    /// A column read at both rows carries one reading per row, so the count is readings,
    /// one per value position.
    #[error("the proof carries {actual} values against {expected} opened readings")]
    ValueCount {
        /// Readings the protocol opens, one per value position.
        expected: usize,
        /// Values the proof carries.
        actual: usize,
    },

    /// A typed column-batching transcript could not replay its proof shape.
    #[error(transparent)]
    ColumnBatchTranscript(#[from] p3_challenger::fs::TranscriptError),

    /// The inner opening disagreed with the prover's independently computed batch value.
    #[error("column batch {batch} returned an aggregate value different from its claimed columns")]
    ColumnBatchValueMismatch {
        /// Batch whose aggregate value disagreed.
        batch: usize,
    },
}

impl<EF, MT, MX, Challenger> MultilinearPcs<EF, Challenger> for BooleanTracePcs<EF, MT, MX>
where
    EF: ChallengeField<EF>
        + EncodableLevel
        + TranscriptField
        + TowerLevel
        + FoldAlphabet<EF>
        + Coordinates,
    MT: Mmcs<EF>,
    MX: Mmcs<EF, Error = MT::Error>,
    Challenger: FieldChallenger<EF>
        + GrindingChallenger<Witness = EF>
        + CanSampleUniformBits<EF>
        + CanObserve<MT::Commitment>
        + CanObserve<MX::Commitment>,
{
    type Val = EF;
    type Commitment = MT::Commitment;
    type ProverData = BooleanTraceData<EF, MT>;
    type Proof = BooleanTraceProof<EF, MT, MX>;
    type Error = BooleanTraceError<EF, MT::Error>;
    type ProverError = BooleanTraceError<EF, MT::Error>;
    type Witness = Vec<Table<EF>>;
    type OpeningProtocol = OpeningProtocol;

    fn num_vars(&self) -> usize {
        self.num_variables()
    }

    fn commit(
        &self,
        witness: Self::Witness,
        challenger: &mut Challenger,
    ) -> Result<(Self::Commitment, Self::ProverData), Self::ProverError> {
        // Gathering runs before the transcript is touched, so a refusal leaves it alone.
        let bits =
            tracing::info_span!("gather boolean bits").in_scope(|| self.gather_bits(&witness))?;
        let (commitment, inner) = self
            .inner
            .commit_bits(&bits, challenger)
            .map_err(BooleanTraceError::Boolean)?;
        Ok((
            commitment,
            BooleanTraceData {
                inner,
                tables: witness,
            },
        ))
    }

    fn observe_commitment(&self, commitment: &Self::Commitment, challenger: &mut Challenger) {
        self.inner.observe_commitment(commitment, challenger);
    }

    fn open(
        &self,
        prover_data: Self::ProverData,
        protocol: Self::OpeningProtocol,
        challenger: &mut Challenger,
    ) -> Result<Self::Proof, Self::ProverError> {
        Self::validate_source_shapes(&prover_data.tables, &protocol)?;
        // The sampled convention draws every batch's point from the transcript.
        let points = sample_points(&protocol, challenger);
        self.open_at(prover_data, &protocol, &points, challenger)
    }

    fn verify(
        &self,
        commitment: &Self::Commitment,
        proof: &Self::Proof,
        challenger: &mut Challenger,
        protocol: Self::OpeningProtocol,
    ) -> Result<(), Self::Error> {
        // Committing bound the root on the prover's sponge, so this binds it here.
        //
        // The points are then sampled from that sponge, as the prover sampled them.
        //
        // The prescribed-point entry point is the one that leaves the binding to its caller.
        self.observe_commitment(commitment, challenger);

        // The same draws the prover made, in the same order, before anything is checked.
        let points = sample_points(&protocol, challenger);
        self.verify_at(commitment, proof, &protocol, &points, challenger)
            .map(|_| ())
    }
}

impl<EF, MT, MX, Challenger> PrescribedPointPcs<EF, Challenger> for BooleanTracePcs<EF, MT, MX>
where
    EF: ChallengeField<EF>
        + EncodableLevel
        + TranscriptField
        + TowerLevel
        + FoldAlphabet<EF>
        + Coordinates,
    MT: Mmcs<EF>,
    MX: Mmcs<EF, Error = MT::Error>,
    Challenger: FieldChallenger<EF>
        + GrindingChallenger<Witness = EF>
        + CanSampleUniformBits<EF>
        + CanObserve<MT::Commitment>
        + CanObserve<MX::Commitment>,
{
    fn prescribed_security(&self, protocol: &OpeningProtocol) -> Option<PrescribedOpeningSecurity> {
        // A protocol this scheme would refuse gets no assessment, so a caller fails closed.
        let shapes = protocol.table_shapes();
        if plan_stacked_layout(&shapes).0 != self.num_variables() {
            return None;
        }
        // A reduction sends carry and last once its successor view outruns one element.
        let absorbed = BitRingSwitch::<EF>::ABSORBED;
        self.batched_shape(protocol).map_or_else(
            || {
                let claims = column_claims(protocol);
                let successor_tensors = claims.iter().any(|claim| {
                    claim.next_at.is_some() && shapes[claim.table].num_variables() > absorbed
                });
                Some(
                    self.inner
                        .readings_security(claims.len(), successor_tensors),
                )
            },
            |(width, next)| {
                let batches = protocol.num_openings();
                // Both combined claims of a batch read one column point, so each is charged.
                let views = 1 + usize::from(next);
                let k = width.next_power_of_two().trailing_zeros() as usize;
                let successor_tensors = next && shapes[0].num_variables() > absorbed;
                let mut security = self.inner.readings_security(batches, successor_tensors);
                security
                    .terms
                    .push(p3_security::multilinear::column_batch_term(
                        batches * views,
                        k,
                        EF::bits(),
                    ));
                Some(security)
            },
        )
    }

    fn open_at(
        &self,
        prover_data: Self::ProverData,
        protocol: &OpeningProtocol,
        points: &[Point<EF>],
        challenger: &mut Challenger,
    ) -> Result<Self::Proof, Self::ProverError> {
        // Every shape and every point is checked before the transcript moves.
        Self::validate_source_shapes(&prover_data.tables, protocol)?;
        let placements = self.validate_opening(protocol, points)?;
        let Some((width, next)) = self.batched_shape(protocol) else {
            let (claims, openings) = Self::opening_claims(protocol, points, &placements);
            let (readings, opening) = self
                .inner
                .open_readings(prover_data.inner, &openings, challenger)
                .map_err(BooleanTraceError::Boolean)?;
            let values = claim_values(&claims, &readings, value_count(protocol));
            return Ok(BooleanTraceProof { values, opening });
        };

        let BooleanTraceData { inner, tables } = prover_data;
        let shape = ColumnBatchShape {
            table_variables: protocol.table_shapes()[0].num_variables(),
            width,
            num_batches: protocol.num_openings(),
            next,
        };
        let run = width * (1 + usize::from(next));
        let mut values = Vec::with_capacity(run * shape.num_batches);
        tracing::info_span!("evaluate boolean columns", width, next).in_scope(|| {
            for point in points {
                let (current, successor) = Self::evaluate_views(&tables[0], point, next);
                values.extend(current);
                values.extend(successor);
            }
        });

        let mut transcript = ColumnBatchProverTranscript::new(challenger, shape);
        let mut openings = Vec::with_capacity(shape.num_batches);
        let mut expected = Vec::with_capacity(shape.num_batches);
        for (point, batch_values) in points.iter().zip(values.chunks_exact(run)) {
            let (current, successor) = batch_values.split_at(width);
            // Both value runs are bound before the point that combines either of them.
            let column_point = transcript.batch(point, current, successor);
            expected.push(BitReadings {
                current: Some(Self::combine_columns(current, &column_point)),
                next: next.then(|| Self::combine_columns(successor, &column_point)),
            });
            let mut lifted_point = column_point;
            lifted_point.extend(point);
            openings.push(BitOpening {
                point: lifted_point,
                row_variables: shape.table_variables,
                current: true,
                next,
            });
        }
        transcript.finish();

        let (readings, opening) = self
            .inner
            .open_readings(inner, &openings, challenger)
            .map_err(BooleanTraceError::Boolean)?;
        if readings.len() != expected.len() {
            return Err(BooleanTraceError::ColumnBatchValueMismatch {
                batch: expected.len(),
            });
        }
        for (batch, (actual, expected)) in readings.iter().zip(&expected).enumerate() {
            if actual != expected {
                return Err(BooleanTraceError::ColumnBatchValueMismatch { batch });
            }
        }
        Ok(BooleanTraceProof { values, opening })
    }

    fn verify_at(
        &self,
        commitment: &Self::Commitment,
        proof: &Self::Proof,
        protocol: &OpeningProtocol,
        points: &[Point<EF>],
        challenger: &mut Challenger,
    ) -> Result<Vec<OpeningEvals<EF>>, Self::Error> {
        // Every shape, point and value count is checked before the transcript moves.
        let placements = self.validate_opening(protocol, points)?;
        let expected_values = value_count(protocol);
        if proof.values.len() != expected_values {
            return Err(BooleanTraceError::ValueCount {
                expected: expected_values,
                actual: proof.values.len(),
            });
        }

        let Some((width, next)) = self.batched_shape(protocol) else {
            let (claims, openings) = Self::opening_claims(protocol, points, &placements);

            // One bit proof answers for every column of every batch at once.
            self.inner
                .verify_readings(
                    commitment,
                    &openings,
                    &claim_readings(&claims, &proof.values),
                    &proof.opening,
                    challenger,
                )
                .map_err(BooleanTraceError::Boolean)?;
            return Ok(opening_evals(protocol, &proof.values));
        };

        let shape = ColumnBatchShape {
            table_variables: protocol.table_shapes()[0].num_variables(),
            width,
            num_batches: protocol.num_openings(),
            next,
        };
        let run = width * (1 + usize::from(next));
        let mut transcript = ColumnBatchVerifierTranscript::new(challenger, shape);
        let mut openings = Vec::with_capacity(shape.num_batches);
        let mut readings = Vec::with_capacity(shape.num_batches);
        for (point, batch_values) in points.iter().zip(proof.values.chunks_exact(run)) {
            let (current, successor) = batch_values.split_at(width);
            // Both value runs are bound before the point that combines either of them.
            let column_point = transcript.batch(point, current, successor)?;
            readings.push(BitReadings {
                current: Some(Self::combine_columns(current, &column_point)),
                next: next.then(|| Self::combine_columns(successor, &column_point)),
            });
            let mut lifted_point = column_point;
            lifted_point.extend(point);
            openings.push(BitOpening {
                point: lifted_point,
                row_variables: shape.table_variables,
                current: true,
                next,
            });
        }
        transcript.finish();

        // One bit proof answers for every batched claim at once.
        self.inner
            .verify_readings(commitment, &openings, &readings, &proof.opening, challenger)
            .map_err(BooleanTraceError::Boolean)?;

        Ok(opening_evals(protocol, &proof.values))
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_binary_field::BinaryField128;
    use p3_field::PrimeCharacteristicRing;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_multilinear_util::poly::Poly;
    use p3_sumcheck::{OpeningBatch, PrescribedPointPcs, TableSpec};
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;
    use crate::params::BinaryPcsParams;
    use crate::test_util::{MyChallenger, MyMmcs, challenger, mmcs};

    type EF = BinaryField128;

    /// Columns every fixture table holds.
    const FIXTURE_WIDTH: usize = 2;

    /// A Boolean table of `2^log_height` rows and two columns.
    fn table(seed: u64, log_height: usize) -> Table<EF> {
        let rows = 1usize << log_height;
        let mut rng = SmallRng::seed_from_u64(seed);
        let cells = (0..FIXTURE_WIDTH * rows)
            .map(|_| EF::from_bool(rng.random::<bool>()))
            .collect();
        Table::new(RowMajorMatrix::new(cells, rows))
    }

    /// A Boolean table with an explicitly chosen width.
    fn table_with_width(seed: u64, log_height: usize, width: usize) -> Table<EF> {
        let rows = 1usize << log_height;
        let mut rng = SmallRng::seed_from_u64(seed);
        let cells = (0..width * rows)
            .map(|_| EF::from_bool(rng.random::<bool>()))
            .collect();
        Table::new(RowMajorMatrix::new(cells, rows))
    }

    fn packed_table(table: &Table<EF>) -> Table<EF> {
        let height = 1usize << table.num_variables();
        let words = (0..height.div_ceil(WORD_BITS))
            .flat_map(|block| {
                (0..table.num_polys()).map(move |column| {
                    (0..WORD_BITS).fold(0u64, |word, lane| {
                        let row = block * WORD_BITS + lane;
                        if row < height && table.column(column).value(row) == EF::ONE {
                            word | (1u64 << lane)
                        } else {
                            word
                        }
                    })
                })
            })
            .collect();
        Table::from_packed_bits(
            RowMajorMatrix::new(words, table.num_polys()),
            table.num_variables(),
        )
    }

    #[test]
    fn packed_column_sums_match_a_per_bit_sum() {
        // Part of one word, one block, one partial task, and several tasks with a partial last one.
        let blocks_past_tasks = (3 * BLOCKS_PER_TASK + 1) * WORD_BITS;
        for (height, width, seed) in [
            (1, 1, 1),
            (4, 1, 2),
            (32, 5, 3),
            (64, 3, 4),
            (5 * WORD_BITS, 1, 5),
            (blocks_past_tasks, 7, 6),
        ] {
            let mut rng = SmallRng::seed_from_u64(seed);
            let used = if height < WORD_BITS {
                (1u64 << height) - 1
            } else {
                u64::MAX
            };
            let words = (0..height.div_ceil(WORD_BITS) * width)
                .map(|_| rng.random::<u64>() & used)
                .collect::<Vec<_>>();
            let row_weights = (0..height).map(|_| rng.random()).collect::<Vec<EF>>();

            let expected = (0..width)
                .map(|column| {
                    (0..height)
                        .filter(|&row| {
                            (words[(row / WORD_BITS) * width + column] >> (row % WORD_BITS)) & 1
                                == 1
                        })
                        .map(|row| row_weights[row])
                        .sum::<EF>()
                })
                .collect::<Vec<_>>();
            let words = RowMajorMatrix::new(words, width);
            assert_eq!(
                packed_column_sums(&words, &row_weights),
                expected,
                "{height}x{width}"
            );
        }
    }

    #[test]
    fn gathered_bits_place_every_cell_at_its_slot() {
        // Packed and dense, long and short, and packed tables spanning several column groups.
        let shapes = [
            TableShape::new(7, 2 * GATHER_GROUP + 3),
            TableShape::new(6, 5),
            TableShape::new(4, GATHER_GROUP + 1),
            TableShape::new(3, 3),
        ];
        let dense = shapes
            .iter()
            .enumerate()
            .map(|(seed, shape)| {
                table_with_width(seed as u64, shape.num_variables(), shape.width())
            })
            .collect::<Vec<_>>();
        let tables = alloc::vec![
            packed_table(&dense[0]),
            dense[1].clone(),
            packed_table(&dense[2]),
            dense[3].clone(),
        ];
        let scheme = pcs(&shapes);

        let mut expected = alloc::vec![0u64; 1 << (scheme.num_variables() - 6)];
        for placement in scheme.placements(&shapes).unwrap() {
            let table = &dense[placement.idx()];
            let column_len = 1usize << table.num_variables();
            for (column, selector) in placement.selectors().iter().enumerate() {
                for row in 0..column_len {
                    if table.column(column).value(row) == EF::ONE {
                        let index = selector.index() * column_len + row;
                        expected[index / WORD_BITS] |= 1 << (index % WORD_BITS);
                    }
                }
            }
        }

        let gathered = scheme.gather_bits(&tables).unwrap();
        let gathered = gathered
            .iter()
            .map(|word| word.to_bits())
            .collect::<Vec<_>>();
        assert_eq!(gathered, expected);
    }

    #[test]
    fn a_gather_refuses_the_first_non_boolean_column_in_placement_order() {
        // The taller table is placed first, so its columns are reached before the short table's.
        let shapes = [TableShape::new(3, 2), TableShape::new(7, 4)];
        let scheme = pcs(&shapes);
        let refused = |bad: &[(usize, usize)]| {
            let tables = shapes
                .iter()
                .enumerate()
                .map(|(index, shape)| {
                    let rows = 1 << shape.num_variables();
                    let table =
                        table_with_width(index as u64, shape.num_variables(), shape.width());
                    let mut cells = table.iter_polys().flatten().copied().collect::<Vec<_>>();
                    for &(_, column) in bad.iter().filter(|&&(table, _)| table == index) {
                        cells[column * rows + rows / 2] = EF::GENERATOR;
                    }
                    Table::new(RowMajorMatrix::new(cells, rows))
                })
                .collect::<Vec<_>>();
            match scheme.gather_bits(&tables) {
                Err(BooleanTraceError::NonBooleanCell { table, column }) => (table, column),
                _ => panic!("a non-Boolean cell must be refused"),
            }
        };
        assert_eq!(refused(&[(0, 1), (1, 3)]), (1, 3));
        assert_eq!(refused(&[(1, 3), (1, 1)]), (1, 1));
        assert_eq!(refused(&[(0, 1), (0, 0)]), (0, 0));
    }

    /// A commitment over the batch these shapes describe.
    fn pcs(shapes: &[TableShape]) -> BooleanTracePcs<EF, MyMmcs, MyMmcs> {
        let params = BinaryPcsParams {
            log_inv_rate: 2,
            pow_bits: 0,
            security_level: 40,
        };
        let (arity, _) = plan_stacked_layout(shapes);
        // The packing absorbs seven variables, so the commitment holds the rest.
        let config = BinaryPcsConfig::try_new::<EF, EF>(arity - 7, params).unwrap();
        BooleanTracePcs::new(config, mmcs(), mmcs(), arity).unwrap()
    }

    /// Every column of every table, opened at one point per table.
    fn protocol(shapes: &[TableShape]) -> OpeningProtocol {
        successor_protocol(shapes, &[])
    }

    /// Every column of every table at the current row, and the `next` columns one row ahead.
    fn successor_protocol(shapes: &[TableShape], next: &[usize]) -> OpeningProtocol {
        OpeningProtocol::new(
            shapes
                .iter()
                .map(|shape| {
                    TableSpec::new(
                        *shape,
                        alloc::vec![OpeningBatch::new(
                            (0..shape.width()).collect(),
                            next.to_vec()
                        )],
                    )
                })
                .collect(),
        )
    }

    /// One table opened `num_batches` times, every column read at both rows each time.
    fn both_views_protocol(shape: TableShape, num_batches: usize) -> OpeningProtocol {
        OpeningProtocol::new(vec![TableSpec::new(
            shape,
            (0..num_batches)
                .map(|_| {
                    OpeningBatch::new((0..shape.width()).collect(), (0..shape.width()).collect())
                })
                .collect(),
        )])
    }

    /// One complete-batch proof of both views whose value at `tampered` is moved by one.
    ///
    /// The flow is the prover's own, so the moved value is bound before the column point is
    /// drawn, exactly as an honest value would be.
    fn forged_successor_proof(
        scheme: &BooleanTracePcs<EF, MyMmcs, MyMmcs>,
        data: BooleanTraceData<EF, MyMmcs>,
        shape: TableShape,
        points: &[Point<EF>],
        tampered: Option<usize>,
        challenger: &mut MyChallenger,
    ) -> BooleanTraceProof<EF, MyMmcs, MyMmcs> {
        type Scheme = BooleanTracePcs<EF, MyMmcs, MyMmcs>;
        let BooleanTraceData { inner, tables } = data;
        let width = shape.width();
        let mut values = Vec::new();
        for point in points {
            let (current, successor) = Scheme::evaluate_views(&tables[0], point, true);
            values.extend(current);
            values.extend(successor);
        }
        if let Some(at) = tampered {
            values[at] += EF::ONE;
        }

        let batch_shape = ColumnBatchShape {
            table_variables: shape.num_variables(),
            width,
            num_batches: points.len(),
            next: true,
        };
        let mut transcript = ColumnBatchProverTranscript::new(challenger, batch_shape);
        let mut openings = Vec::with_capacity(points.len());
        for (point, batch_values) in points.iter().zip(values.chunks_exact(2 * width)) {
            let (current, successor) = batch_values.split_at(width);
            let column_point = transcript.batch(point, current, successor);
            let mut lifted_point = column_point;
            lifted_point.extend(point);
            openings.push(BitOpening {
                point: lifted_point,
                row_variables: shape.num_variables(),
                current: true,
                next: true,
            });
        }
        transcript.finish();

        let (_, opening) = scheme
            .inner
            .open_readings(inner, &openings, challenger)
            .unwrap();
        BooleanTraceProof { values, opening }
    }

    /// A column read one row ahead: row `z` reads row `z + 1`, the last row itself.
    fn successor_reading(column: &[EF], point: &Point<EF>) -> EF {
        let rows = column.len();
        let eq = Poly::new_from_point(point.as_slice(), EF::ONE);
        (0..rows)
            .map(|row| eq.as_slice()[row] * column[(row + 1).min(rows - 1)])
            .sum()
    }

    #[test]
    fn an_opened_column_is_the_column_the_table_holds() {
        // Invariant: lifting a row point by a slot address reads that column and no other.
        //
        // Fixture state: three tables, so the slots are neither all one arity nor all aligned.
        //
        //     - table 0   2^10 rows, 2 columns
        //     - table 1   2^8  rows, 2 columns
        //     - table 2   2^4  rows, 2 columns
        //
        // The stack is log2_ceil(2048 + 512 + 32) = 12 variables wide.
        //
        // The third table is what puts a column shorter than one staging word in the batch.
        //
        //     - 2^10 and 2^8 rows  ->  whole words, written by the aligned run
        //     - 2^4 rows           ->  sixteen bits inside one word, set in place
        //
        // A wrong shift on that second path would place the column somewhere else.
        //
        // Every opened value is checked against the column's own multilinear.
        // That reference shares nothing with the packing or with the reduction.
        let shapes = [
            TableShape::new(10, FIXTURE_WIDTH),
            TableShape::new(8, FIXTURE_WIDTH),
            TableShape::new(4, FIXTURE_WIDTH),
        ];
        let tables = alloc::vec![table(0xB100, 10), table(0xB101, 8), table(0xB103, 4)];
        let scheme = pcs(&shapes);
        let protocol = protocol(&shapes);
        assert_eq!(scheme.num_variables(), 12);

        let mut rng = SmallRng::seed_from_u64(0xB102);
        let points = alloc::vec![
            Point::<EF>::rand(&mut rng, 10),
            Point::<EF>::rand(&mut rng, 8),
            Point::<EF>::rand(&mut rng, 4),
        ];

        let mut prover_chal = challenger();
        let (commitment, data) = scheme.commit(tables.clone(), &mut prover_chal).unwrap();
        let proof = scheme
            .open_at(data, &protocol, &points, &mut prover_chal)
            .unwrap();

        let mut verifier_chal = challenger();
        scheme.observe_commitment(&commitment, &mut verifier_chal);
        let evals = scheme
            .verify_at(&commitment, &proof, &protocol, &points, &mut verifier_chal)
            .unwrap();

        // One batch per table, each carrying that table's whole width and no successor value.
        assert_eq!(evals.len(), shapes.len());
        for (index, batch) in evals.iter().enumerate() {
            assert!(batch.next().is_empty());
            for (column, &value) in batch.current().iter().enumerate() {
                let reference = Poly::new(tables[index].poly(column).as_slice().to_vec());
                assert_eq!(
                    value,
                    reference.eval_base(&points[index]),
                    "{index}/{column}"
                );
            }
        }
    }

    #[test]
    fn packed_table_round_trip_matches_dense_openings() {
        for (log_height, width, seed) in [(5, 8, 0xB10A), (10, FIXTURE_WIDTH, 0xB10C)] {
            let shapes = [TableShape::new(log_height, width)];
            let dense = table_with_width(seed, log_height, width);
            let packed = packed_table(&dense);
            let scheme = pcs(&shapes);
            let protocol = protocol(&shapes);
            let point = Point::<EF>::rand(&mut SmallRng::seed_from_u64(seed + 1), log_height);

            let mut dense_challenger = challenger();
            let (dense_commitment, dense_data) =
                scheme.commit(vec![dense], &mut dense_challenger).unwrap();
            let dense_proof = scheme
                .open_at(
                    dense_data,
                    &protocol,
                    core::slice::from_ref(&point),
                    &mut dense_challenger,
                )
                .unwrap();
            let dense_after_open = p3_challenger::CanSample::<EF>::sample(&mut dense_challenger);
            let mut packed_challenger = challenger();
            let (packed_commitment, packed_data) =
                scheme.commit(vec![packed], &mut packed_challenger).unwrap();
            let packed_proof = scheme
                .open_at(
                    packed_data,
                    &protocol,
                    core::slice::from_ref(&point),
                    &mut packed_challenger,
                )
                .unwrap();
            let packed_after_open = p3_challenger::CanSample::<EF>::sample(&mut packed_challenger);

            assert_eq!(dense_commitment, packed_commitment);
            assert_eq!(
                postcard::to_allocvec(&dense_proof).unwrap(),
                postcard::to_allocvec(&packed_proof).unwrap()
            );
            assert_eq!(dense_after_open, packed_after_open);
            let mut verifier = challenger();
            scheme.observe_commitment(&packed_commitment, &mut verifier);
            scheme
                .verify_at(
                    &packed_commitment,
                    &packed_proof,
                    &protocol,
                    core::slice::from_ref(&point),
                    &mut verifier,
                )
                .unwrap();
        }
    }

    #[test]
    fn packed_mixed_heights_and_widths_match_dense_proof_bytes() {
        let shapes = [
            TableShape::new(6, 3),
            TableShape::new(4, 5),
            TableShape::new(7, 2),
        ];
        let dense = vec![
            table_with_width(0xB10C, 6, 3),
            table_with_width(0xB10D, 4, 5),
            table_with_width(0xB10E, 7, 2),
        ];
        let packed = dense.iter().map(packed_table).collect::<Vec<_>>();
        let scheme = pcs(&shapes);
        let protocol = protocol(&shapes);
        let mut rng = SmallRng::seed_from_u64(0xB10F);
        let points = shapes
            .iter()
            .map(|shape| Point::<EF>::rand(&mut rng, shape.num_variables()))
            .collect::<Vec<_>>();

        let mut dense_challenger = challenger();
        let (dense_commitment, dense_data) = scheme.commit(dense, &mut dense_challenger).unwrap();
        let dense_proof = scheme
            .open_at(dense_data, &protocol, &points, &mut dense_challenger)
            .unwrap();
        let dense_after_open = p3_challenger::CanSample::<EF>::sample(&mut dense_challenger);

        let mut packed_challenger = challenger();
        let (packed_commitment, packed_data) =
            scheme.commit(packed, &mut packed_challenger).unwrap();
        let packed_proof = scheme
            .open_at(packed_data, &protocol, &points, &mut packed_challenger)
            .unwrap();
        let packed_after_open = p3_challenger::CanSample::<EF>::sample(&mut packed_challenger);

        assert_eq!(dense_commitment, packed_commitment);
        assert_eq!(
            postcard::to_allocvec(&dense_proof).unwrap(),
            postcard::to_allocvec(&packed_proof).unwrap()
        );
        assert_eq!(dense_after_open, packed_after_open);
    }

    #[test]
    fn the_sampled_path_draws_the_same_points_on_both_sides() {
        // Invariant: the sampled convention needs no point to cross the wire.
        // Each side binds the commitment itself.
        //
        //     commit  ->  binds the root, then the caller draws nothing of its own
        //     verify  ->  binds the root, then samples the points the prover sampled
        //
        // A caller that bound the root by hand would bind it twice and sample elsewhere.
        // Both sides therefore run on a fresh sponge, as every other scheme here expects.
        //
        // Fixture state: one table of 2^10 rows and two columns, so the stack has arity 11.
        let shapes = [TableShape::new(10, FIXTURE_WIDTH)];
        let scheme = pcs(&shapes);
        let protocol = protocol(&shapes);

        let mut prover_chal = challenger();
        let (commitment, data) = scheme
            .commit(alloc::vec![table(0xB200, 10)], &mut prover_chal)
            .unwrap();
        let proof = scheme
            .open(data, protocol.clone(), &mut prover_chal)
            .unwrap();

        let mut verifier_chal = challenger();
        scheme
            .verify(&commitment, &proof, &mut verifier_chal, protocol.clone())
            .unwrap();

        // Binding it a second time moves every later draw, so the two sides split.
        let mut double_bound = challenger();
        scheme.observe_commitment(&commitment, &mut double_bound);
        assert!(
            scheme
                .verify(&commitment, &proof, &mut double_bound, protocol)
                .is_err()
        );
    }

    #[test]
    fn a_disagreeing_shape_or_point_count_is_refused() {
        // Invariant: the shape agreement is checked before the transcript moves at all.
        let shapes = [TableShape::new(10, FIXTURE_WIDTH)];
        let scheme = pcs(&shapes);

        // Mutation: one column fewer changes the retained table shape.
        let narrow = [TableShape::new(10, 1)];
        let mut chal = challenger();
        let (_, data) = scheme
            .commit(alloc::vec![table(0xB300, 10)], &mut chal)
            .unwrap();
        let point = Point::<EF>::rand(&mut SmallRng::seed_from_u64(0xB301), 10);
        let Err(error) = scheme.open_at(
            data,
            &protocol(&narrow),
            core::slice::from_ref(&point),
            &mut chal,
        ) else {
            panic!("a shape set stacking elsewhere describes another commitment")
        };
        let BooleanTraceError::TableShapeMismatch {
            table: table_index,
            expected,
            actual,
        } = error
        else {
            panic!("the retained source shape must be checked before stacking")
        };
        assert_eq!(table_index, 0);
        assert_eq!(expected, TableShape::new(10, 1));
        assert_eq!(actual, TableShape::new(10, FIXTURE_WIDTH));

        // Mutation: no point at all, against a protocol scheduling one batch.
        let (_, data) = scheme
            .commit(alloc::vec![table(0xB300, 10)], &mut chal)
            .unwrap();
        let Err(error) = scheme.open_at(data, &protocol(&shapes), &[], &mut chal) else {
            panic!("a batch with no point is opened nowhere")
        };
        assert!(matches!(
            error,
            BooleanTraceError::PointCount {
                expected: 1,
                actual: 0
            }
        ));
    }

    #[test]
    fn a_proof_short_of_one_value_is_refused() {
        // Invariant: the value run must cover every opened column, one value each.
        //
        // Fixture state: one table of two columns, so an honest proof carries two values.
        //
        // Mutation: drop the last value, leaving one against two opened readings.
        let shapes = [TableShape::new(10, FIXTURE_WIDTH)];
        let scheme = pcs(&shapes);
        let protocol = protocol(&shapes);

        let mut prover_chal = challenger();
        let (commitment, data) = scheme
            .commit(alloc::vec![table(0xB400, 10)], &mut prover_chal)
            .unwrap();
        let points = alloc::vec![Point::<EF>::rand(&mut SmallRng::seed_from_u64(0xB401), 10)];
        let mut proof = scheme
            .open_at(data, &protocol, &points, &mut prover_chal)
            .unwrap();
        proof.values.pop();

        let mut verifier_chal = challenger();
        scheme.observe_commitment(&commitment, &mut verifier_chal);
        let Err(error) =
            scheme.verify_at(&commitment, &proof, &protocol, &points, &mut verifier_chal)
        else {
            panic!("a proof answering for one column cannot answer for two")
        };
        assert!(matches!(
            error,
            BooleanTraceError::ValueCount {
                expected: 2,
                actual: 1
            }
        ));
    }

    #[test]
    fn the_claim_plan_writes_every_value_position_exactly_once() {
        // Invariant: the value run is laid out by the claim plan alone, so no position is
        // answered twice and none is left at the fill the run starts from.
        //
        //     current [0, 1, 2], next []        the whole width at the current row
        //     current [2, 0, 1], next [1, 2]    both sides, out of table order
        //     current [0, 1],    next [1, 0]    both sides, each reordering the other
        //     current [],        next [2, 0]    a successor view read nowhere else
        //     current [3, 1],    next [1, 3, 0] a successor side wider than the current one
        for (width, current, next) in [
            (3, vec![0, 1, 2], vec![]),
            (3, vec![2, 0, 1], vec![1, 2]),
            (2, vec![0, 1], vec![1, 0]),
            (3, vec![], vec![2, 0]),
            (4, vec![3, 1], vec![1, 3, 0]),
        ] {
            let protocol = OpeningProtocol::new(vec![TableSpec::new(
                TableShape::new(8, width),
                vec![OpeningBatch::new(current.clone(), next.clone())],
            )]);
            let len = value_count(&protocol);
            assert_eq!(len, current.len() + next.len());
            assert!(
                covers_every_value(&column_claims(&protocol), len),
                "{current:?} / {next:?}"
            );
        }

        // Several batches over several tables share one run, so the cursor has to advance
        // past every batch's own two sides before the next batch writes.
        let protocol = OpeningProtocol::new(vec![
            TableSpec::new(
                TableShape::new(8, 3),
                vec![
                    OpeningBatch::new(vec![2, 0, 1], vec![1, 2]),
                    OpeningBatch::new(vec![0], vec![2, 0]),
                ],
            ),
            TableSpec::new(
                TableShape::new(6, 2),
                vec![OpeningBatch::new(vec![1, 0], vec![0])],
            ),
        ]);
        let len = value_count(&protocol);
        assert_eq!(len, 5 + 3 + 3);
        assert!(covers_every_value(&column_claims(&protocol), len));

        // Both ways a plan can miss are refused, so the checks above are not vacuous.
        //
        //     one position written twice   a claim answering both views at one position
        //     one position never written   no claim at all against a run of one value
        let collided = ColumnClaim {
            table: 0,
            column: 0,
            opening: 0,
            current_at: Some(0),
            next_at: Some(0),
        };
        assert!(!covers_every_value(&[collided], 1));
        assert!(!covers_every_value(&[], 1));
    }

    #[test]
    fn a_single_table_batch_uses_one_reduction_for_all_columns() {
        // Invariant: one complete current-row batch for one table is discharged by one
        // column-point ring switch, regardless of the table width.
        let shape = TableShape::new(8, 3);
        let scheme = pcs(&[shape]);
        let protocol = OpeningProtocol::new(vec![TableSpec::new(
            shape,
            vec![OpeningBatch::new(vec![0, 1, 2], Vec::new())],
        )]);
        let table = table_with_width(0xB500, 8, 3);
        let point = Point::<EF>::rand(&mut SmallRng::seed_from_u64(0xB501), 8);

        let mut prover_chal = challenger();
        let (_, data) = scheme.commit(vec![table], &mut prover_chal).unwrap();
        let proof = scheme
            .open_at(data, &protocol, &[point], &mut prover_chal)
            .unwrap();

        assert_eq!(proof.values.len(), 3);
        assert_eq!(proof.opening.reductions.len(), 1);
    }

    #[test]
    fn batched_values_are_bound_before_the_column_point() {
        // Changing a claimed column value changes the verifier's column point and therefore
        // cannot be repaired by reusing the original one-reduction proof.
        let shape = TableShape::new(8, 3);
        let scheme = pcs(&[shape]);
        let protocol = OpeningProtocol::new(vec![TableSpec::new(
            shape,
            vec![OpeningBatch::new(vec![0, 1, 2], Vec::new())],
        )]);
        let table = table_with_width(0xB502, 8, 3);
        let point = Point::<EF>::rand(&mut SmallRng::seed_from_u64(0xB503), 8);

        let mut prover_chal = challenger();
        let (commitment, data) = scheme.commit(vec![table], &mut prover_chal).unwrap();
        let mut proof = scheme
            .open_at(
                data,
                &protocol,
                core::slice::from_ref(&point),
                &mut prover_chal,
            )
            .unwrap();
        proof.values[0] += EF::ONE;

        let mut verifier_chal = challenger();
        scheme.observe_commitment(&commitment, &mut verifier_chal);
        assert!(
            scheme
                .verify_at(&commitment, &proof, &protocol, &[point], &mut verifier_chal)
                .is_err()
        );
    }

    #[test]
    fn complete_batches_support_multiple_points_and_width_one() {
        // Width one has no column-point coordinates; two opening batches still use two
        // independent inner reductions and preserve the ordinary value return format.
        let shape = TableShape::new(8, 1);
        let scheme = pcs(&[shape]);
        let protocol = OpeningProtocol::new(vec![TableSpec::new(
            shape,
            vec![
                OpeningBatch::new(vec![0], Vec::new()),
                OpeningBatch::new(vec![0], Vec::new()),
            ],
        )]);
        let table = table_with_width(0xB504, 8, 1);
        let mut rng = SmallRng::seed_from_u64(0xB505);
        let points = vec![Point::<EF>::rand(&mut rng, 8), Point::rand(&mut rng, 8)];

        let mut prover_chal = challenger();
        let (commitment, data) = scheme
            .commit(vec![table.clone()], &mut prover_chal)
            .unwrap();
        let proof = scheme
            .open_at(data, &protocol, &points, &mut prover_chal)
            .unwrap();
        assert_eq!(proof.values.len(), 2);
        assert_eq!(proof.opening.reductions.len(), 2);

        let mut verifier_chal = challenger();
        scheme.observe_commitment(&commitment, &mut verifier_chal);
        let evals = scheme
            .verify_at(&commitment, &proof, &protocol, &points, &mut verifier_chal)
            .unwrap();
        for (point, eval) in points.iter().zip(evals) {
            let reference = Poly::new(table.poly(0).as_slice().to_vec());
            assert_eq!(eval.current()[0], reference.eval_base(point));
        }
    }

    #[test]
    fn optimized_security_charges_batches_and_column_coordinates() {
        let shape = TableShape::new(8, 3);
        let scheme = pcs(&[shape]);
        let protocol = OpeningProtocol::new(vec![TableSpec::new(
            shape,
            vec![
                OpeningBatch::new(vec![0, 1, 2], Vec::new()),
                OpeningBatch::new(vec![0, 1, 2], Vec::new()),
            ],
        )]);

        let security = <BooleanTracePcs<EF, MyMmcs, MyMmcs> as PrescribedPointPcs<
            EF,
            MyChallenger,
        >>::prescribed_security(&scheme, &protocol)
        .unwrap();
        let batching = security
            .terms
            .iter()
            .find(|term| term.label == "column-batching")
            .unwrap();
        assert!((batching.bits.bits() - 126.0).abs() < 1e-9);
        let ring_switch = security
            .terms
            .iter()
            .find(|term| term.label == "bit-ring-switch")
            .unwrap();
        assert!(ring_switch.bits.bits().is_finite());
    }

    #[test]
    fn optimized_security_rejects_a_protocol_with_the_wrong_stacked_arity() {
        let committed_shape = TableShape::new(8, 3);
        let scheme = pcs(&[committed_shape]);
        let protocol_shape = TableShape::new(8, 2);
        let protocol = OpeningProtocol::new(vec![TableSpec::new(
            protocol_shape,
            vec![OpeningBatch::new(vec![0, 1], Vec::new())],
        )]);

        let security = <BooleanTracePcs<EF, MyMmcs, MyMmcs> as PrescribedPointPcs<
            EF,
            MyChallenger,
        >>::prescribed_security(&scheme, &protocol);
        assert!(security.is_none());
    }

    #[test]
    fn non_power_two_complete_batches_round_trip_below_and_above_packing_width() {
        // Widths three and five exercise zero padding, while heights below and above 128
        // rows cover the short-column-in-word and whole-word packing layouts.
        for (width, log_height, num_batches, seed) in [
            (3, 6, 2, 0xB506),
            (3, 8, 2, 0xB507),
            (5, 6, 1, 0xB508),
            (5, 8, 1, 0xB509),
        ] {
            let shape = TableShape::new(log_height, width);
            let scheme = pcs(&[shape]);
            let protocol = OpeningProtocol::new(vec![TableSpec::new(
                shape,
                (0..num_batches)
                    .map(|_| OpeningBatch::new((0..width).collect(), Vec::new()))
                    .collect(),
            )]);
            let table = table_with_width(seed, log_height, width);
            let mut rng = SmallRng::seed_from_u64(seed + 1);
            let points = (0..num_batches)
                .map(|_| Point::<EF>::rand(&mut rng, log_height))
                .collect::<Vec<_>>();

            let mut prover_chal = challenger();
            let (commitment, data) = scheme
                .commit(vec![table.clone()], &mut prover_chal)
                .unwrap();
            let proof = scheme
                .open_at(data, &protocol, &points, &mut prover_chal)
                .unwrap();
            assert_eq!(proof.opening.reductions.len(), num_batches);

            let mut verifier_chal = challenger();
            scheme.observe_commitment(&commitment, &mut verifier_chal);
            let evals = scheme
                .verify_at(&commitment, &proof, &protocol, &points, &mut verifier_chal)
                .unwrap();
            for (batch, point) in points.iter().enumerate() {
                for column in 0..width {
                    let reference = Poly::new(table.poly(column).as_slice().to_vec());
                    assert_eq!(evals[batch].current()[column], reference.eval_base(point));
                }
            }
        }
    }

    #[test]
    fn optimized_proofs_reject_each_column_tampering_reduction_tampering_and_point_reordering() {
        let shape = TableShape::new(8, 3);
        let scheme = pcs(&[shape]);
        let protocol = OpeningProtocol::new(vec![TableSpec::new(
            shape,
            vec![
                OpeningBatch::new(vec![0, 1, 2], Vec::new()),
                OpeningBatch::new(vec![0, 1, 2], Vec::new()),
            ],
        )]);
        let table = table_with_width(0xB50A, 8, 3);
        let mut rng = SmallRng::seed_from_u64(0xB50B);
        let points = vec![Point::<EF>::rand(&mut rng, 8), Point::rand(&mut rng, 8)];
        let mut prover_chal = challenger();
        let (commitment, data) = scheme.commit(vec![table], &mut prover_chal).unwrap();
        let proof = scheme
            .open_at(data, &protocol, &points, &mut prover_chal)
            .unwrap();

        // Untampered control.
        let mut verifier_chal = challenger();
        scheme.observe_commitment(&commitment, &mut verifier_chal);
        scheme
            .verify_at(&commitment, &proof, &protocol, &points, &mut verifier_chal)
            .unwrap();

        for index in 0..proof.values.len() {
            let mut tampered = proof.clone();
            tampered.values[index] += EF::ONE;
            let mut verifier_chal = challenger();
            scheme.observe_commitment(&commitment, &mut verifier_chal);
            assert!(
                scheme
                    .verify_at(
                        &commitment,
                        &tampered,
                        &protocol,
                        &points,
                        &mut verifier_chal
                    )
                    .is_err()
            );
        }

        let mut tampered = proof.clone();
        tampered.opening.reductions.pop();
        let mut verifier_chal = challenger();
        scheme.observe_commitment(&commitment, &mut verifier_chal);
        assert!(
            scheme
                .verify_at(
                    &commitment,
                    &tampered,
                    &protocol,
                    &points,
                    &mut verifier_chal
                )
                .is_err()
        );

        let mut verifier_chal = challenger();
        scheme.observe_commitment(&commitment, &mut verifier_chal);
        let reordered = vec![points[1].clone(), points[0].clone()];
        assert!(
            scheme
                .verify_at(
                    &commitment,
                    &proof,
                    &protocol,
                    &reordered,
                    &mut verifier_chal
                )
                .is_err()
        );
    }

    #[test]
    fn optimized_shape_errors_leave_the_prover_transcript_untouched() {
        let shape = TableShape::new(8, 3);
        let scheme = pcs(&[shape]);
        // A batch naming the two views differently takes the per-column route, whose own
        // shape checks run before it binds anything either.
        let protocol = OpeningProtocol::new(vec![TableSpec::new(
            shape,
            vec![OpeningBatch::new(vec![0, 1, 2], vec![0])],
        )]);
        let mut prover_chal = challenger();
        let (_, data) = scheme
            .commit(vec![table_with_width(0xB50C, 8, 3)], &mut prover_chal)
            .unwrap();
        let mut expected = prover_chal.clone();
        let short_point = Point::<EF>::rand(&mut SmallRng::seed_from_u64(0xB50D), 7);
        let error = match scheme.open_at(data, &protocol, &[short_point], &mut prover_chal) {
            Ok(_) => panic!("a point with the wrong arity must be rejected"),
            Err(error) => error,
        };
        assert!(matches!(
            error,
            BooleanTraceError::PointArity { table: 0, .. }
        ));
        assert_eq!(
            p3_challenger::CanSample::<EF>::sample(&mut prover_chal),
            p3_challenger::CanSample::<EF>::sample(&mut expected),
        );

        let valid_protocol = OpeningProtocol::new(vec![TableSpec::new(
            shape,
            vec![OpeningBatch::new(vec![0, 1, 2], Vec::new())],
        )]);
        let mut prover_chal = challenger();
        let (_, data) = scheme
            .commit(vec![table_with_width(0xB510, 8, 3)], &mut prover_chal)
            .unwrap();
        let mut expected = prover_chal.clone();
        let bad_point = Point::<EF>::rand(&mut SmallRng::seed_from_u64(0xB511), 7);
        let error = match scheme.open_at(data, &valid_protocol, &[bad_point], &mut prover_chal) {
            Ok(_) => panic!("a point with the wrong arity must be rejected"),
            Err(error) => error,
        };
        assert!(matches!(
            error,
            BooleanTraceError::PointArity { table: 0, .. }
        ));
        assert_eq!(
            p3_challenger::CanSample::<EF>::sample(&mut prover_chal),
            p3_challenger::CanSample::<EF>::sample(&mut expected),
        );

        let narrow_shape = TableShape::new(8, 2);
        let narrow_protocol = OpeningProtocol::new(vec![TableSpec::new(
            narrow_shape,
            vec![OpeningBatch::new(vec![0, 1], Vec::new())],
        )]);
        let mut prover_chal = challenger();
        let (_, data) = scheme
            .commit(vec![table_with_width(0xB512, 8, 3)], &mut prover_chal)
            .unwrap();
        let mut expected = prover_chal.clone();
        let point = Point::<EF>::rand(&mut SmallRng::seed_from_u64(0xB513), 8);
        let error = match scheme.open_at(data, &narrow_protocol, &[point], &mut prover_chal) {
            Ok(_) => panic!("a protocol with the wrong stacked arity must be rejected"),
            Err(error) => error,
        };
        assert!(matches!(
            error,
            BooleanTraceError::TableShapeMismatch { .. }
        ));
        assert_eq!(
            p3_challenger::CanSample::<EF>::sample(&mut prover_chal),
            p3_challenger::CanSample::<EF>::sample(&mut expected),
        );
    }

    #[test]
    fn source_shape_errors_precede_sampling_for_open_and_open_at() {
        for (committed_shapes, protocol_shapes, expected) in [
            (
                vec![TableShape::new(8, 3)],
                vec![TableShape::new(8, 4)],
                "shape",
            ),
            (
                vec![TableShape::new(8, 3)],
                vec![TableShape::new(9, 2)],
                "shape",
            ),
            (
                vec![TableShape::new(8, 2), TableShape::new(8, 2)],
                vec![TableShape::new(8, 4)],
                "count",
            ),
        ] {
            let scheme = pcs(&committed_shapes);
            let tables = committed_shapes
                .iter()
                .enumerate()
                .map(|(index, shape)| {
                    table_with_width(0xB520 + index as u64, shape.num_variables(), shape.width())
                })
                .collect();
            let sampled_protocol = protocol(&protocol_shapes);
            let mut sampled_challenger = challenger();
            let (_, data) = scheme.commit(tables, &mut sampled_challenger).unwrap();
            let mut expected_challenger = sampled_challenger.clone();
            let error = match scheme.open(data, sampled_protocol, &mut sampled_challenger) {
                Ok(_) => panic!("a source shape mismatch must be rejected before sampling"),
                Err(error) => error,
            };
            match expected {
                "shape" => assert!(matches!(
                    error,
                    BooleanTraceError::TableShapeMismatch { .. }
                )),
                "count" => assert!(matches!(
                    error,
                    BooleanTraceError::TableCountMismatch { .. }
                )),
                _ => unreachable!(),
            }
            assert_eq!(
                p3_challenger::CanSample::<EF>::sample(&mut sampled_challenger),
                p3_challenger::CanSample::<EF>::sample(&mut expected_challenger),
            );

            let protocol = protocol(&protocol_shapes);
            let points = protocol_shapes
                .iter()
                .map(|shape| {
                    Point::<EF>::rand(&mut SmallRng::seed_from_u64(0xB530), shape.num_variables())
                })
                .collect::<Vec<_>>();
            let mut challenger = challenger();
            let (_, data) = scheme
                .commit(
                    committed_shapes
                        .iter()
                        .enumerate()
                        .map(|(index, shape)| {
                            table_with_width(
                                0xB540 + index as u64,
                                shape.num_variables(),
                                shape.width(),
                            )
                        })
                        .collect(),
                    &mut challenger,
                )
                .unwrap();
            let mut expected_challenger = challenger.clone();
            let error = match scheme.open_at(data, &protocol, &points, &mut challenger) {
                Ok(_) => panic!("a source shape mismatch must be rejected before opening"),
                Err(error) => error,
            };
            assert!(matches!(
                (expected, error),
                ("shape", BooleanTraceError::TableShapeMismatch { .. })
                    | ("count", BooleanTraceError::TableCountMismatch { .. })
            ));
            assert_eq!(
                p3_challenger::CanSample::<EF>::sample(&mut challenger),
                p3_challenger::CanSample::<EF>::sample(&mut expected_challenger),
            );
        }
    }

    #[test]
    fn reordered_subset_batches_use_the_fallback_column_route() {
        let shape = TableShape::new(8, 3);
        let scheme = pcs(&[shape]);
        let protocol = OpeningProtocol::new(vec![TableSpec::new(
            shape,
            vec![OpeningBatch::new(vec![2, 0], Vec::new())],
        )]);
        let table = table_with_width(0xB50E, 8, 3);
        let point = Point::<EF>::rand(&mut SmallRng::seed_from_u64(0xB50F), 8);

        let mut prover_chal = challenger();
        let (commitment, data) = scheme
            .commit(vec![table.clone()], &mut prover_chal)
            .unwrap();
        let proof = scheme
            .open_at(
                data,
                &protocol,
                core::slice::from_ref(&point),
                &mut prover_chal,
            )
            .unwrap();
        assert_eq!(proof.values.len(), 2);
        assert_eq!(proof.opening.reductions.len(), 2);

        let mut verifier_chal = challenger();
        scheme.observe_commitment(&commitment, &mut verifier_chal);
        let evals = scheme
            .verify_at(
                &commitment,
                &proof,
                &protocol,
                core::slice::from_ref(&point),
                &mut verifier_chal,
            )
            .unwrap();
        assert_eq!(
            evals[0].current(),
            &[
                Poly::new(table.poly(2).as_slice().to_vec()).eval_base(&point),
                Poly::new(table.poly(0).as_slice().to_vec()).eval_base(&point),
            ]
        );
    }

    #[test]
    fn complete_successor_batches_round_trip_around_the_packing_width() {
        // Invariant: a batch claiming every column at both rows is one reduction, and each
        // claimed value is that column's own reading at the batch's point.
        //
        // Fixture state: table arities below, at and above the seven one element absorbs.
        //
        //     2^3 rows   the whole successor view sits inside one element
        //     2^7 rows   exactly one element
        //     2^9 rows   the view outruns one element, so the reduction sends carry and last
        //
        // Width one draws no column coordinate, and width three pads the column point.
        //
        // Every value is checked against a reference built from the source table alone.
        for (log_height, width, num_batches, seed) in [
            (3, 32, 2, 0xB600),
            (7, 8, 1, 0xB602),
            (9, 4, 3, 0xB604),
            (9, 1, 2, 0xB606),
            (8, 3, 2, 0xB608),
        ] {
            let shape = TableShape::new(log_height, width);
            let scheme = pcs(&[shape]);
            let protocol = both_views_protocol(shape, num_batches);
            let table = table_with_width(seed, log_height, width);
            let mut rng = SmallRng::seed_from_u64(seed + 1);
            let points = (0..num_batches)
                .map(|_| Point::<EF>::rand(&mut rng, log_height))
                .collect::<Vec<_>>();

            let mut prover_chal = challenger();
            let (commitment, data) = scheme
                .commit(vec![table.clone()], &mut prover_chal)
                .unwrap();
            let proof = scheme
                .open_at(data, &protocol, &points, &mut prover_chal)
                .unwrap();
            assert_eq!(proof.values.len(), 2 * width * num_batches);
            assert_eq!(proof.opening.reductions.len(), num_batches);

            let mut verifier_chal = challenger();
            scheme.observe_commitment(&commitment, &mut verifier_chal);
            let evals = scheme
                .verify_at(&commitment, &proof, &protocol, &points, &mut verifier_chal)
                .unwrap();
            for (batch, point) in points.iter().enumerate() {
                for column in 0..width {
                    let cells = table.poly(column);
                    let reference = Poly::new(cells.as_slice().to_vec());
                    assert_eq!(
                        evals[batch].current()[column],
                        reference.eval_base(point),
                        "{batch}/{column}"
                    );
                    assert_eq!(
                        evals[batch].next()[column],
                        successor_reading(cells.as_slice(), point),
                        "{batch}/{column} next"
                    );
                }
            }
        }
    }

    #[test]
    fn packed_successor_tables_match_dense_proof_bytes() {
        // Invariant: a table held as words and the same table held as cells claim the same
        // readings at both rows, so their proofs are one byte run and one sponge state.
        for (log_height, width, seed) in [(5, 8, 0xB60A), (10, FIXTURE_WIDTH, 0xB60C)] {
            let shape = TableShape::new(log_height, width);
            let dense = table_with_width(seed, log_height, width);
            let packed = packed_table(&dense);
            let scheme = pcs(&[shape]);
            let protocol = both_views_protocol(shape, 1);
            let point = Point::<EF>::rand(&mut SmallRng::seed_from_u64(seed + 1), log_height);

            let mut dense_challenger = challenger();
            let (dense_commitment, dense_data) =
                scheme.commit(vec![dense], &mut dense_challenger).unwrap();
            let dense_proof = scheme
                .open_at(
                    dense_data,
                    &protocol,
                    core::slice::from_ref(&point),
                    &mut dense_challenger,
                )
                .unwrap();
            let dense_after_open = p3_challenger::CanSample::<EF>::sample(&mut dense_challenger);

            let mut packed_challenger = challenger();
            let (packed_commitment, packed_data) =
                scheme.commit(vec![packed], &mut packed_challenger).unwrap();
            let packed_proof = scheme
                .open_at(
                    packed_data,
                    &protocol,
                    core::slice::from_ref(&point),
                    &mut packed_challenger,
                )
                .unwrap();
            let packed_after_open = p3_challenger::CanSample::<EF>::sample(&mut packed_challenger);

            assert_eq!(dense_commitment, packed_commitment);
            assert_eq!(
                postcard::to_allocvec(&dense_proof).unwrap(),
                postcard::to_allocvec(&packed_proof).unwrap()
            );
            assert_eq!(dense_after_open, packed_after_open);

            let mut verifier = challenger();
            scheme.observe_commitment(&packed_commitment, &mut verifier);
            scheme
                .verify_at(
                    &packed_commitment,
                    &packed_proof,
                    &protocol,
                    core::slice::from_ref(&point),
                    &mut verifier,
                )
                .unwrap();
        }
    }

    #[test]
    fn partial_successor_views_use_the_fallback_column_route() {
        // Invariant: a batch naming the two views differently is answered column by column,
        // one claim per column either view names, a column in both sharing one claim.
        //
        //     current [0, 1, 2], next [1]   a subset read one row ahead
        //     current [0],       next [1]   a column read one row ahead and nowhere else
        //     current [0, 1],    next [1]   a column in both sets
        //     current [2, 0],    next [0]   columns out of their table order
        let shape = TableShape::new(8, 3);
        let scheme = pcs(&[shape]);
        let table = table_with_width(0xB60E, 8, 3);
        let point = Point::<EF>::rand(&mut SmallRng::seed_from_u64(0xB60F), 8);

        for (current, next, claims) in [
            (vec![0, 1, 2], vec![1], 3),
            (vec![0], vec![1], 2),
            (vec![0, 1], vec![1], 2),
            (vec![2, 0], vec![0], 2),
        ] {
            let protocol = OpeningProtocol::new(vec![TableSpec::new(
                shape,
                vec![OpeningBatch::new(current.clone(), next.clone())],
            )]);

            let mut prover_chal = challenger();
            let (commitment, data) = scheme
                .commit(vec![table.clone()], &mut prover_chal)
                .unwrap();
            let proof = scheme
                .open_at(
                    data,
                    &protocol,
                    core::slice::from_ref(&point),
                    &mut prover_chal,
                )
                .unwrap();
            assert_eq!(proof.values.len(), current.len() + next.len());
            assert_eq!(proof.opening.reductions.len(), claims);

            let mut verifier_chal = challenger();
            scheme.observe_commitment(&commitment, &mut verifier_chal);
            let evals = scheme
                .verify_at(
                    &commitment,
                    &proof,
                    &protocol,
                    core::slice::from_ref(&point),
                    &mut verifier_chal,
                )
                .unwrap();

            for (&column, &value) in current.iter().zip(evals[0].current()) {
                let reference = Poly::new(table.poly(column).as_slice().to_vec());
                assert_eq!(value, reference.eval_base(&point), "current {column}");
            }
            for (&column, &value) in next.iter().zip(evals[0].next()) {
                let cells = table.poly(column);
                assert_eq!(
                    value,
                    successor_reading(cells.as_slice(), &point),
                    "next {column}"
                );
            }
        }
    }

    #[test]
    fn reordered_successor_views_keep_each_side_in_its_own_order() {
        // Invariant: the two sides of a batch are laid out independently, so a column named
        // by both lands at the position its own side gives it.
        //
        //     current [0, 1],    next [1, 0]   each side reverses the other
        //     current [2, 0, 1], next [1, 2]   three columns, neither side in table order
        //
        // Both shapes take the per-column route, one claim per column either side names.
        //
        // Every returned value is checked against a reference built from the table alone.
        let shape = TableShape::new(8, 3);
        let scheme = pcs(&[shape]);
        let table = table_with_width(0xB618, 8, 3);
        let point = Point::<EF>::rand(&mut SmallRng::seed_from_u64(0xB619), 8);

        for (current, next, claims) in [(vec![0, 1], vec![1, 0], 2), (vec![2, 0, 1], vec![1, 2], 3)]
        {
            let protocol = OpeningProtocol::new(vec![TableSpec::new(
                shape,
                vec![OpeningBatch::new(current.clone(), next.clone())],
            )]);

            let mut prover_chal = challenger();
            let (commitment, data) = scheme
                .commit(vec![table.clone()], &mut prover_chal)
                .unwrap();
            let proof = scheme
                .open_at(
                    data,
                    &protocol,
                    core::slice::from_ref(&point),
                    &mut prover_chal,
                )
                .unwrap();
            assert_eq!(proof.values.len(), current.len() + next.len());
            assert_eq!(proof.opening.reductions.len(), claims);

            let mut verifier_chal = challenger();
            scheme.observe_commitment(&commitment, &mut verifier_chal);
            let evals = scheme
                .verify_at(
                    &commitment,
                    &proof,
                    &protocol,
                    core::slice::from_ref(&point),
                    &mut verifier_chal,
                )
                .unwrap();

            let current_reference = current
                .iter()
                .map(|&column| Poly::new(table.poly(column).as_slice().to_vec()).eval_base(&point))
                .collect::<Vec<_>>();
            let next_reference = next
                .iter()
                .map(|&column| successor_reading(table.poly(column).as_slice(), &point))
                .collect::<Vec<_>>();
            assert_eq!(evals[0].current(), current_reference, "current {current:?}");
            assert_eq!(evals[0].next(), next_reference, "next {next:?}");
        }
    }

    #[test]
    fn a_second_table_is_read_one_row_ahead_at_its_own_point() {
        // Invariant: two tables in one commitment each read their own rows, so the successor
        // view of one never steps into the slot of the other.
        //
        // Fixture state: 2^10 and 2^8 rows of two columns each, stacking to arity 12.
        let shapes = [
            TableShape::new(10, FIXTURE_WIDTH),
            TableShape::new(8, FIXTURE_WIDTH),
        ];
        let tables = vec![table(0xB610, 10), table(0xB611, 8)];
        let scheme = pcs(&shapes);
        let protocol = successor_protocol(&shapes, &[1]);
        let mut rng = SmallRng::seed_from_u64(0xB612);
        let points = vec![
            Point::<EF>::rand(&mut rng, 10),
            Point::<EF>::rand(&mut rng, 8),
        ];

        let mut prover_chal = challenger();
        let (commitment, data) = scheme.commit(tables.clone(), &mut prover_chal).unwrap();
        let proof = scheme
            .open_at(data, &protocol, &points, &mut prover_chal)
            .unwrap();

        let mut verifier_chal = challenger();
        scheme.observe_commitment(&commitment, &mut verifier_chal);
        let evals = scheme
            .verify_at(&commitment, &proof, &protocol, &points, &mut verifier_chal)
            .unwrap();

        for (index, batch) in evals.iter().enumerate() {
            for (column, &value) in batch.current().iter().enumerate() {
                let reference = Poly::new(tables[index].poly(column).as_slice().to_vec());
                assert_eq!(
                    value,
                    reference.eval_base(&points[index]),
                    "{index}/{column}"
                );
            }
            let cells = tables[index].poly(1);
            assert_eq!(
                batch.next(),
                &[successor_reading(cells.as_slice(), &points[index])],
                "{index} next"
            );
        }
    }

    #[test]
    fn a_tampered_successor_value_is_rejected_on_either_route() {
        // Invariant: every value of the run is bound before the challenges that combine it,
        // so moving any one of them, on either route, breaks the proof.
        let shape = TableShape::new(8, 3);
        let scheme = pcs(&[shape]);
        let table = table_with_width(0xB614, 8, 3);
        let point = Point::<EF>::rand(&mut SmallRng::seed_from_u64(0xB615), 8);

        for protocol in [
            both_views_protocol(shape, 1),
            OpeningProtocol::new(vec![TableSpec::new(
                shape,
                vec![OpeningBatch::new(vec![0, 1, 2], vec![1])],
            )]),
        ] {
            let mut prover_chal = challenger();
            let (commitment, data) = scheme
                .commit(vec![table.clone()], &mut prover_chal)
                .unwrap();
            let proof = scheme
                .open_at(
                    data,
                    &protocol,
                    core::slice::from_ref(&point),
                    &mut prover_chal,
                )
                .unwrap();

            // Untampered control.
            let mut verifier_chal = challenger();
            scheme.observe_commitment(&commitment, &mut verifier_chal);
            scheme
                .verify_at(
                    &commitment,
                    &proof,
                    &protocol,
                    core::slice::from_ref(&point),
                    &mut verifier_chal,
                )
                .unwrap();

            for index in 0..proof.values.len() {
                let mut tampered = proof.clone();
                tampered.values[index] += EF::ONE;
                let mut verifier_chal = challenger();
                scheme.observe_commitment(&commitment, &mut verifier_chal);
                assert!(
                    scheme
                        .verify_at(
                            &commitment,
                            &tampered,
                            &protocol,
                            core::slice::from_ref(&point),
                            &mut verifier_chal,
                        )
                        .is_err(),
                    "value {index}"
                );
            }
        }
    }

    #[test]
    fn successor_security_charges_both_views_of_every_batch() {
        // Invariant: the two combined claims of a batch share one column point, so the
        // batching term charges each of them.
        //
        //     current only   two batches over two coordinates   ->  128 - log2(4) = 126
        //     both views     four claims over two coordinates   ->  128 - log2(8) = 125
        let shape = TableShape::new(8, 3);
        let scheme = pcs(&[shape]);
        let security = <BooleanTracePcs<EF, MyMmcs, MyMmcs> as PrescribedPointPcs<
            EF,
            MyChallenger,
        >>::prescribed_security(&scheme, &both_views_protocol(shape, 2))
        .expect("a successor view is answered, so it is priced");

        let bits = |label: &str| {
            security
                .terms
                .iter()
                .find(|term| term.label == label)
                .unwrap_or_else(|| panic!("missing {label}"))
                .bits
                .bits()
        };
        assert!((bits("column-batching") - 125.0).abs() < 1e-9);
        assert!(bits("bit-ring-switch").is_finite());

        // The per-column route prices a successor protocol too, with no batching term.
        let fallback = OpeningProtocol::new(vec![TableSpec::new(
            shape,
            vec![OpeningBatch::new(vec![0, 1, 2], vec![1])],
        )]);
        let security = <BooleanTracePcs<EF, MyMmcs, MyMmcs> as PrescribedPointPcs<
            EF,
            MyChallenger,
        >>::prescribed_security(&scheme, &fallback)
        .expect("a per-column successor protocol is priced");
        assert!(
            security
                .terms
                .iter()
                .all(|term| term.label != "column-batching")
        );
    }

    #[test]
    fn a_forged_next_value_is_rejected_at_every_position() {
        // Invariant: every next value is combined into the claim the reduction answers, so
        // a prover that binds a moved one and then opens honestly is still refused.
        //
        // Moving a value in a finished proof cannot show this. The column point moves with
        // it, so such a rejection would not say whether the combination reads that position
        // at all. This flow binds the moved value first, exactly as an honest run does.
        let shape = TableShape::new(8, 3);
        let scheme = pcs(&[shape]);
        let protocol = both_views_protocol(shape, 2);
        let mut rng = SmallRng::seed_from_u64(0xB617);
        let points = vec![Point::<EF>::rand(&mut rng, 8), Point::rand(&mut rng, 8)];

        let forge = |tampered: Option<usize>| {
            let mut prover_chal = challenger();
            let (commitment, data) = scheme
                .commit(vec![table_with_width(0xB616, 8, 3)], &mut prover_chal)
                .unwrap();
            let proof =
                forged_successor_proof(&scheme, data, shape, &points, tampered, &mut prover_chal);
            let mut verifier_chal = challenger();
            scheme.observe_commitment(&commitment, &mut verifier_chal);
            scheme.verify_at(&commitment, &proof, &protocol, &points, &mut verifier_chal)
        };

        // Untampered control: this flow is the one the prover plays, so it verifies.
        forge(None).unwrap();

        // Every next value of every batch, which trails that batch's current ones.
        for batch in 0..points.len() {
            for column in 0..shape.width() {
                let at = batch * 2 * shape.width() + shape.width() + column;
                assert!(forge(Some(at)).is_err(), "next value {at}");
            }
        }
    }
}
