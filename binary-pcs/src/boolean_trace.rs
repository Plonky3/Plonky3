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
use core::marker::PhantomData;

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
    OpeningEvals, OpeningPointMismatch, OpeningProtocol, PrescribedOpeningSecurity,
    PrescribedPointPcs, TableShape,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::boolean::{
    BitOpening, BitReadings, BooleanBackend, BooleanMultilinearPcs, BooleanPcs, BooleanPcsError,
    BooleanProof,
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
pub struct BooleanTraceCommitment<EF, B> {
    /// The bit commitment every column claim is discharged against.
    inner: B,
    /// Marker tying the commitment to its challenge field; carries no runtime state.
    _marker: PhantomData<EF>,
}

/// The trace commitment discharged through the folding-only bit commitment.
pub type BooleanTracePcs<EF, MT, MX> = BooleanTraceCommitment<EF, BooleanPcs<EF, MT, MX>>;

impl<EF, MT, MX> BooleanTraceCommitment<EF, BooleanPcs<EF, MT, MX>>
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
    /// Returns an error unless the schedule was derived for the committed alphabet.
    ///
    /// Returns an error unless the schedule commits exactly the elements the packing holds.
    pub fn new(
        config: BinaryPcsConfig,
        mmcs: MT,
        round_mmcs: MX,
        num_variables: usize,
    ) -> Result<Self, BooleanTraceError<EF, MT::Error>> {
        BooleanPcs::new(config, mmcs, round_mmcs, num_variables)
            .map(Self::from_commitment)
            .map_err(BooleanTraceCommitmentError::Boolean)
    }
}

impl<EF, B> BooleanTraceCommitment<EF, B>
where
    EF: ChallengeField<EF>
        + EncodableLevel
        + TranscriptField
        + TowerLevel
        + FoldAlphabet<EF>
        + Coordinates,
    B: BooleanBackend<EF>,
{
    /// Stack trace tables into an already-built bit commitment.
    pub const fn from_commitment(inner: B) -> Self {
        Self {
            inner,
            _marker: PhantomData,
        }
    }

    /// Variables the stacked bit witness has, so `2^n` bits in all.
    #[must_use]
    pub fn num_variables(&self) -> usize {
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
    ) -> Result<Vec<TablePlacement>, BooleanTraceCommitmentError<B::Error>> {
        // Prover and verifier both plan from the public shapes, so neither picks its own.
        let (arity, placements) = plan_stacked_layout(shapes);
        if arity != self.num_variables() {
            return Err(BooleanTraceCommitmentError::StackedArity {
                expected: self.num_variables(),
                actual: arity,
            });
        }
        Ok(placements)
    }

    /// The bit claims the per-column route raises, in transcript order.
    ///
    /// One claim per column a batch reads, that batch's point prefixed by the slot address,
    /// asking for whichever readings the claim's entry in the plan names.
    fn bit_openings(
        protocol: &OpeningProtocol,
        claims: &[ColumnClaim],
        points: &[Point<EF>],
        placements: &[TablePlacement],
    ) -> Vec<BitOpening<EF>> {
        let shapes = protocol.table_shapes();

        // Placements arrive largest table first, so index them by the table each one owns.
        let mut by_table = alloc::vec![None; shapes.len()];
        for placement in placements {
            by_table[placement.idx()] = Some(placement);
        }

        claims
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
            .collect()
    }

    /// Validate all public opening metadata without constructing per-column claims.
    fn validate_opening(
        &self,
        protocol: &OpeningProtocol,
        points: &[Point<EF>],
    ) -> Result<Vec<TablePlacement>, BooleanTraceCommitmentError<B::Error>> {
        let placements = self.placements(&protocol.table_shapes())?;
        protocol
            .check_points(points)
            .map_err(|mismatch| match mismatch {
                OpeningPointMismatch::Count { expected, actual } => {
                    BooleanTraceCommitmentError::PointCount { expected, actual }
                }
                OpeningPointMismatch::Arity {
                    table,
                    expected,
                    actual,
                } => BooleanTraceCommitmentError::PointArity {
                    table,
                    expected,
                    actual,
                },
            })?;
        Ok(placements)
    }

    /// Validate retained source shapes before any sampled point or opening transcript is used.
    fn validate_source_shapes(
        tables: &[Table<EF>],
        protocol: &OpeningProtocol,
    ) -> Result<(), BooleanTraceCommitmentError<B::Error>> {
        let expected = protocol.table_shapes();
        if tables.len() != expected.len() {
            return Err(BooleanTraceCommitmentError::TableCountMismatch {
                expected: expected.len(),
                actual: tables.len(),
            });
        }
        for (table, expected) in expected.iter().copied().enumerate() {
            let actual = tables[table].shape();
            if actual != expected {
                return Err(BooleanTraceCommitmentError::TableShapeMismatch {
                    table,
                    expected,
                    actual,
                });
            }
        }
        Ok(())
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
    ) -> Result<Vec<PackedGf2x64>, BooleanTraceCommitmentError<B::Error>> {
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

        // The smallest (position, column) over all refusals is the first in placement order.
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
            return Err(BooleanTraceCommitmentError::NonBooleanCell {
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

/// Up to sixty-four Boolean cells as one word, the lowest cell in the lowest bit.
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

/// How one opening protocol is discharged against the bit commitment.
///
/// A function of the protocol alone: the security assessment, the prover and the verifier
/// read the same resolution.
#[derive(Clone, Debug)]
enum OpeningRoute {
    /// One table, every batch reading all of it at the current row and either none or all of
    /// it one row ahead: one reduction per batch, over one shared column point.
    Batched(ColumnBatchShape),
    /// Any other protocol: one reduction per column read, in transcript order.
    PerColumn(Vec<ColumnClaim>),
}

impl OpeningRoute {
    /// Resolve the route a protocol takes.
    fn new(protocol: &OpeningProtocol) -> Self {
        batched_shape(protocol)
            .map_or_else(|| Self::PerColumn(column_claims(protocol)), Self::Batched)
    }

    /// Reductions the bit commitment answers on this route.
    const fn num_reductions(&self) -> usize {
        match self {
            Self::Batched(shape) => shape.num_batches,
            Self::PerColumn(claims) => claims.len(),
        }
    }

    /// Whether some reduction reads a successor view over more rows than one element absorbs.
    fn successor_tensors(&self, shapes: &[TableShape], absorbed: usize) -> bool {
        match self {
            Self::Batched(shape) => shape.next && shape.table_variables > absorbed,
            Self::PerColumn(claims) => claims.iter().any(|claim| {
                claim.next_at.is_some() && shapes[claim.table].num_variables() > absorbed
            }),
        }
    }
}

/// The complete single-table shape the batched route handles, if the protocol has one.
///
/// Every batch reads the whole width at the current row, and either none of it or all
/// of it one row ahead, the same way in every batch.
fn batched_shape(protocol: &OpeningProtocol) -> Option<ColumnBatchShape> {
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
        .then(|| ColumnBatchShape {
            table_variables: shapes[0].num_variables(),
            width,
            num_batches: protocol.num_openings(),
            next,
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
pub struct BooleanTraceCommitmentData<EF: Field, D> {
    /// Prover data of the bit commitment underneath.
    inner: D,
    /// Source tables, lent back to whatever evaluates constraints over them.
    tables: Vec<Table<EF>>,
}

/// The retained data of a trace behind the folding-only bit commitment.
pub type BooleanTraceData<EF, MT> = BooleanTraceCommitmentData<EF, BinaryPcsProverData<EF, EF, MT>>;

impl<EF: Field, D> Clone for BooleanTraceCommitmentData<EF, D>
where
    D: Clone,
{
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            tables: self.tables.clone(),
        }
    }
}

impl<EF: Field, D> BooleanTraceCommitmentData<EF, D> {
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
    serialize = "EF: TowerLevel, P: Serialize",
    deserialize = "EF: TowerLevel, P: Deserialize<'de>"
))]
pub struct BooleanTraceCommitmentProof<EF: Field, P> {
    /// One value per column a batch reads, its current ones first, batches in transcript order.
    pub values: Vec<EF>,
    /// The bit commitment's own proof, answering for every value at once.
    pub opening: P,
}

/// One opening of a trace behind the folding-only bit commitment.
pub type BooleanTraceProof<EF, MT, MX> = BooleanTraceCommitmentProof<EF, BooleanProof<EF, MT, MX>>;

/// Why a trace behind the folding-only bit commitment could not be committed or opened.
pub type BooleanTraceError<EF, MmcsError> =
    BooleanTraceCommitmentError<BooleanPcsError<EF, MmcsError>>;

/// Why a Boolean trace could not be committed or opened.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum BooleanTraceCommitmentError<E> {
    /// The bit commitment underneath refused the witness or the opening.
    #[error(transparent)]
    Boolean(E),

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

impl<EF, B, Challenger> MultilinearPcs<EF, Challenger> for BooleanTraceCommitment<EF, B>
where
    EF: ChallengeField<EF>
        + EncodableLevel
        + TranscriptField
        + TowerLevel
        + FoldAlphabet<EF>
        + Coordinates,
    B: BooleanMultilinearPcs<EF, Challenger>,
    Challenger: FieldChallenger<EF>
        + GrindingChallenger<Witness = EF>
        + CanSampleUniformBits<EF>
        + CanObserve<B::Commitment>,
{
    type Val = EF;
    type Commitment = B::Commitment;
    type ProverData = BooleanTraceCommitmentData<EF, B::ProverData>;
    type Proof = BooleanTraceCommitmentProof<EF, B::Proof>;
    type Error = BooleanTraceCommitmentError<B::Error>;
    type ProverError = BooleanTraceCommitmentError<B::Error>;
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
            .map_err(BooleanTraceCommitmentError::Boolean)?;
        Ok((
            commitment,
            BooleanTraceCommitmentData {
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

impl<EF, B, Challenger> PrescribedPointPcs<EF, Challenger> for BooleanTraceCommitment<EF, B>
where
    EF: ChallengeField<EF>
        + EncodableLevel
        + TranscriptField
        + TowerLevel
        + FoldAlphabet<EF>
        + Coordinates,
    B: BooleanMultilinearPcs<EF, Challenger>,
    Challenger: FieldChallenger<EF>
        + GrindingChallenger<Witness = EF>
        + CanSampleUniformBits<EF>
        + CanObserve<B::Commitment>,
{
    fn prescribed_security(&self, protocol: &OpeningProtocol) -> Option<PrescribedOpeningSecurity> {
        // A protocol this scheme would refuse gets no assessment, so a caller fails closed.
        let shapes = protocol.table_shapes();
        if plan_stacked_layout(&shapes).0 != self.num_variables() {
            return None;
        }
        // A reduction sends carry and last once its successor view outruns one element.
        let absorbed = BitRingSwitch::<EF>::ABSORBED;
        let route = OpeningRoute::new(protocol);
        let mut security = self.inner.readings_security(
            route.num_reductions(),
            route.successor_tensors(&shapes, absorbed),
        )?;
        if let OpeningRoute::Batched(shape) = route {
            // Both combined claims of a batch read one column point, so each is charged.
            //
            // The batching challenge is drawn before any candidate has been named.
            security.charge_reduction(p3_security::multilinear::column_batch_term(
                shape.num_batches * shape.num_views(),
                shape.column_variables(),
                EF::bits(),
            ));
        }
        Some(security)
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
        let shape = match OpeningRoute::new(protocol) {
            OpeningRoute::PerColumn(claims) => {
                let openings = Self::bit_openings(protocol, &claims, points, &placements);
                let (readings, opening) = self
                    .inner
                    .open_readings(prover_data.inner, &openings, challenger)
                    .map_err(BooleanTraceCommitmentError::Boolean)?;
                let values = claim_values(&claims, &readings, value_count(protocol));
                return Ok(BooleanTraceCommitmentProof { values, opening });
            }
            OpeningRoute::Batched(shape) => shape,
        };

        let BooleanTraceCommitmentData { inner, tables } = prover_data;
        let ColumnBatchShape { width, next, .. } = shape;
        let run = shape.values_per_batch();
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
            .map_err(BooleanTraceCommitmentError::Boolean)?;
        if readings.len() != expected.len() {
            return Err(BooleanTraceCommitmentError::ColumnBatchValueMismatch {
                batch: expected.len(),
            });
        }
        for (batch, (actual, expected)) in readings.iter().zip(&expected).enumerate() {
            if actual != expected {
                return Err(BooleanTraceCommitmentError::ColumnBatchValueMismatch { batch });
            }
        }
        Ok(BooleanTraceCommitmentProof { values, opening })
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
            return Err(BooleanTraceCommitmentError::ValueCount {
                expected: expected_values,
                actual: proof.values.len(),
            });
        }

        let shape = match OpeningRoute::new(protocol) {
            OpeningRoute::PerColumn(claims) => {
                let openings = Self::bit_openings(protocol, &claims, points, &placements);

                // One bit proof answers for every column of every batch at once.
                self.inner
                    .verify_readings(
                        commitment,
                        &openings,
                        &claim_readings(&claims, &proof.values),
                        &proof.opening,
                        challenger,
                    )
                    .map_err(BooleanTraceCommitmentError::Boolean)?;
                return Ok(opening_evals(protocol, &proof.values));
            }
            OpeningRoute::Batched(shape) => shape,
        };

        let ColumnBatchShape { width, next, .. } = shape;
        let run = shape.values_per_batch();
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
            .map_err(BooleanTraceCommitmentError::Boolean)?;

        Ok(opening_evals(protocol, &proof.values))
    }
}

#[cfg(test)]
mod tests;
