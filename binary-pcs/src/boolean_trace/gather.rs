//! Gathering every table's Boolean cells into one bit witness.

use alloc::vec::Vec;

use p3_binary_field::{BitCoordinates, PackedGf2, PackedGf2x64, TowerLevel};
use p3_challenger::fs::TranscriptField;
use p3_field::{ExtensionField, Field};
use p3_maybe_rayon::prelude::*;
use p3_sumcheck::layout::{Table, TablePlacement};

use super::{BooleanTraceCommitment, BooleanTraceCommitmentError, WORD_BITS};
use crate::boolean::BooleanBackend;

/// Adjacent packed columns one gather task copies, one source line's worth of words.
pub(super) const GATHER_GROUP: usize = 8;

impl<EF, B> BooleanTraceCommitment<EF, B>
where
    EF: BitCoordinates + ExtensionField<B::Val>,
    B: BooleanBackend<EF>,
    B::Val: TranscriptField + TowerLevel,
{
    /// Gather the Boolean cells of every table into one bit witness.
    ///
    /// # Errors
    ///
    /// - The shapes do not stack to the committed arity.
    /// - A cell holds neither zero nor one, so it addresses no bit.
    pub(super) fn gather_bits(
        &self,
        tables: &[Table<B::Val>],
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
