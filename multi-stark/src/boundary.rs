//! Borgeaud boundary-IO handling of public inputs.
//!
//! A public boundary cell is committed as zero.
//! The verifier adds the public value back from its own copy:
//!
//! ```text
//!     committed = true column with its cell set to 0
//!     restored  = committed(point) + eq_cell(point) * public
//! ```
//!
//! The commitment therefore never carries data the verifier already holds.
//!
//! The prover folds the true cell value, not the committed zero.
//! A pin asserted alongside the AIR constraints equates the two sides:
//!
//! ```text
//!     folded   = committed + eq_cell * cell_value
//!     restored = committed + eq_cell * public
//!                            ^ equal only when cell_value == public
//! ```
//!
//! See <https://solvable.group/posts/super-air/> ("Handling public inputs").

use alloc::borrow::Cow;
use alloc::vec::Vec;

use p3_air::{BaseAir, BoundaryEnd, BoundaryPublic};
use p3_field::{ExtensionField, Field};
use p3_sumcheck::layout::Table;
use thiserror::Error;

use crate::selectors::BoundaryEvals;

/// Reasons a public boundary declaration cannot be applied to an AIR.
///
/// Each variant is a statement about the declaration alone.
/// No trace is involved.
#[derive(Copy, Clone, Debug, Error, PartialEq, Eq)]
pub enum BoundaryIoError {
    /// A declared cell names a column the AIR does not have.
    #[error("boundary-IO column {column} is out of range for main width {width}")]
    ColumnOutOfRange {
        /// Column the declaration names.
        column: usize,
        /// Number of main columns the AIR declares.
        width: usize,
    },
    /// A declared cell names a public value the AIR does not have.
    #[error(
        "boundary-IO public value {index} is out of range for {num_public_values} public values"
    )]
    PublicValueOutOfRange {
        /// Public-value index the declaration names.
        index: usize,
        /// Number of public values the AIR declares.
        num_public_values: usize,
    },
    /// Two declared cells name the same trace cell.
    ///
    /// The cell is blanked once but reconstructed twice.
    /// Even an honest proof then fails to verify.
    #[error("boundary-IO declares column {column} twice on the {end:?} row")]
    DuplicateCell {
        /// Column named by both declarations.
        column: usize,
        /// Trace end both declarations name.
        end: BoundaryEnd,
    },
}

/// Check that a public boundary declaration addresses only cells and values the AIR has.
///
/// A declaration is fixed by the AIR, never by the witness.
/// Checking it once at setup therefore costs nothing per proof.
///
/// # Errors
///
/// - A cell names a column outside the main trace.
/// - A cell names a public value the AIR does not declare.
/// - Two cells name the same trace cell.
pub fn validate<F, A>(air: &A) -> Result<(), BoundaryIoError>
where
    F: Field,
    A: BaseAir<F>,
{
    let cells = air.public_boundary_io();
    let width = air.width();
    let num_public_values = air.num_public_values();

    for (index, cell) in cells.iter().enumerate() {
        // Blanking and folding index columns by this number with no further check.
        if cell.column >= width {
            return Err(BoundaryIoError::ColumnOutOfRange {
                column: cell.column,
                width,
            });
        }

        // Reconstruction indexes the public values by this number with no further check.
        if cell.public_value >= num_public_values {
            return Err(BoundaryIoError::PublicValueOutOfRange {
                index: cell.public_value,
                num_public_values,
            });
        }

        // A repeat of an earlier cell would be blanked once and corrected twice.
        // A pairwise scan over the accepted prefix avoids allocating a set.
        if cells[..index]
            .iter()
            .any(|earlier| earlier.column == cell.column && earlier.end == cell.end)
        {
            return Err(BoundaryIoError::DuplicateCell {
                column: cell.column,
                end: cell.end,
            });
        }
    }

    Ok(())
}

/// The public boundary cells of one AIR, plus the two halves of their handling.
///
/// ```text
///     prover  : blanks each cell before commitment, folds the true values
///     verifier: adds the public values back to the opened committed values
/// ```
pub(crate) struct BoundaryIo<'a>(&'a [BoundaryPublic]);

impl<'a> BoundaryIo<'a> {
    /// Wrap an AIR's declared public boundary cells.
    pub(crate) const fn new(cells: &'a [BoundaryPublic]) -> Self {
        Self(cells)
    }

    /// Whether the AIR declares no public boundary cells.
    ///
    /// Every other operation on this type is a no-op in that case.
    pub(crate) const fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Blank every declared cell in place, returning the values removed.
    ///
    /// ```text
    ///     before: column[cell row] = v
    ///     after : column[cell row] = 0, and v handed back
    /// ```
    ///
    /// # Arguments
    ///
    /// - `table`: the true, transposed trace table, one row per column.
    ///
    /// # Returns
    ///
    /// The removed cell values, in declaration order.
    ///
    /// # Panics
    ///
    /// Panics when a declared cell names a column the table does not have.
    /// Setup rejects such a declaration before either party reaches this point.
    pub(crate) fn take_cells<F: Field>(&self, table: &mut Table<F>) -> Vec<F> {
        // Each column holds one evaluation per trace row.
        let height = 1 << table.num_variables();

        // Swap a zero into each declared cell and keep what was there.
        self.0
            .iter()
            .map(|cell| {
                core::mem::replace(&mut table.poly_mut(cell.column)[cell.row(height)], F::ZERO)
            })
            .collect()
    }

    /// Write blanked cell values back into a table.
    ///
    /// Each target cell must still be zero.
    /// That zero is what lines the verifier's reconstruction up with the fold:
    ///
    /// ```text
    ///     folded   = table + sum_cells eq_cell * value
    ///     restored = table + sum_cells eq_cell * public
    /// ```
    ///
    /// A commit path that moved or rewrote a blanked cell breaks that alignment.
    /// Asserting the zero catches it here rather than in an unverifiable proof.
    ///
    /// # Arguments
    ///
    /// - `table`: blanked table to edit, with the shape the values were taken from.
    /// - `values`: cell values in declaration order.
    ///
    /// # Panics
    ///
    /// Panics when `values` does not carry one entry per declared cell.
    /// Panics when a target cell is not zero.
    pub(crate) fn restore_cells<F: Field>(&self, table: &mut Table<F>, values: &[F]) {
        // Values and cells are paired positionally.
        assert_eq!(
            values.len(),
            self.0.len(),
            "boundary-IO cell values must match the declaration"
        );

        // Each column holds one evaluation per trace row.
        let height = 1 << table.num_variables();

        for (cell, &value) in self.0.iter().zip(values) {
            let slot = &mut table.poly_mut(cell.column)[cell.row(height)];

            // A nonzero slot means the commit path did not leave the cell blank.
            assert!(
                slot.is_zero(),
                "boundary-IO cell at column {} is not blank",
                cell.column
            );
            *slot = value;
        }
    }

    /// Add each public value back to the openings of a blanked commitment.
    ///
    /// A blanked column and the true column differ in one cell.
    /// Their multilinear extensions therefore differ by one Lagrange term:
    ///
    /// ```text
    ///     true(point) = committed(point) + eq_cell(point) * public
    /// ```
    ///
    /// Each cell is opened in two views, and each view needs its own weight:
    ///
    /// - The current-row view takes the cell's own weight.
    /// - The successor view takes the cell's successor weight.
    /// - A first-row cell has successor weight zero.
    /// - Only a last-row cell therefore corrects the successor view.
    ///
    /// An AIR that declares no cells borrows its openings straight through.
    ///
    /// # Arguments
    ///
    /// - `current`: opened current-row value of each column, in column order.
    /// - `next`: opened successor value of each next-row column, aligned with `next_columns`.
    /// - `next_columns`: column indices whose successor the AIR reads.
    /// - `point`: the per-instance bound point, a suffix of the shared sumcheck point.
    /// - `public_values`: the AIR's public inputs.
    ///
    /// # Returns
    ///
    /// The two corrected views, owned when a correction applied and borrowed otherwise.
    ///
    /// # Panics
    ///
    /// Panics when a declared cell names a column or public value that is out of range.
    /// Setup rejects such a declaration before either party reaches this point.
    pub(crate) fn reconstruct<'b, F, EF>(
        &self,
        current: &'b [EF],
        next: &'b [EF],
        next_columns: &[usize],
        point: &[EF],
        public_values: &[F],
    ) -> (Cow<'b, [EF]>, Cow<'b, [EF]>)
    where
        F: Field,
        EF: ExtensionField<F>,
    {
        // Nothing declared means nothing was blanked.
        // The openings are the true values already.
        if self.is_empty() {
            return (Cow::Borrowed(current), Cow::Borrowed(next));
        }

        // Start from the committed openings.
        // Each declared cell adds one correction below.
        let mut current = current.to_vec();
        let mut next = next.to_vec();

        // Lagrange weight of each trace end at this point.
        //
        //     first = prod_j (1 - r_j)
        //     last  = prod_j r_j
        let boundary = BoundaryEvals::at(point);

        // Weight the last row carries under the repeat-last successor map.
        let last_successor = BoundaryEvals::last_row_successor_weight(point);

        for cell in self.0 {
            // Openings live in the extension field.
            // Public inputs are base-field and lift into it.
            let public = EF::from(public_values[cell.public_value]);

            // Pick the pair of weights the cell's end contributes.
            //
            //     first end -> current view only
            //     last  end -> both views
            let (current_weight, successor_weight) = match cell.end {
                BoundaryEnd::First => (boundary.first, EF::ZERO),
                BoundaryEnd::Last => (boundary.last, last_successor),
            };

            // Correct the current-row claim for this cell's column.
            current[cell.column] += current_weight * public;

            // The successor slice only covers columns the AIR reads ahead.
            // A cell on any other column has no successor claim to correct.
            if let Some(slot) = next_columns.iter().position(|&c| c == cell.column) {
                next[slot] += successor_weight * public;
            }
        }

        (Cow::Owned(current), Cow::Owned(next))
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_multilinear_util::point::Point;
    use p3_multilinear_util::poly::Poly;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;

    /// AIR shell carrying exactly the shape and cells a test hands it.
    ///
    /// It asserts no constraints.
    /// Only the declaration check ever reads it.
    struct ProbeAir<'a> {
        /// Main-trace width the declaration is checked against.
        width: usize,
        /// Public-value count the declaration is checked against.
        num_public_values: usize,
        /// Declared public boundary cells.
        cells: &'a [BoundaryPublic],
    }

    impl BaseAir<F> for ProbeAir<'_> {
        fn width(&self) -> usize {
            self.width
        }

        fn num_public_values(&self) -> usize {
            self.num_public_values
        }

        fn public_boundary_io(&self) -> &[BoundaryPublic] {
            self.cells
        }
    }

    /// Width-2 height-4 table with distinct nonzero entries.
    ///
    /// ```text
    ///     column 0: [1, 2, 3, 4]
    ///     column 1: [5, 6, 7, 8]
    /// ```
    fn probe_table() -> Table<F> {
        // One row per column, four evaluations each.
        let values = (1u64..=8).map(F::from_u64).collect();
        Table::new(RowMajorMatrix::new(values, 4))
    }

    #[test]
    fn take_and_restore_cells_round_trip() {
        // Invariant: blanking and writing back are exact inverses.
        //
        // Fixture state: two cells on opposite ends of different columns.
        //
        //     column 0, first row: 1 -> 0
        //     column 1, last  row: 8 -> 0
        let cells = [
            BoundaryPublic::new(0, BoundaryEnd::First, 0),
            BoundaryPublic::new(1, BoundaryEnd::Last, 1),
        ];
        let boundary = BoundaryIo::new(&cells);
        let mut table = probe_table();

        let taken = boundary.take_cells(&mut table);

        // The removed values come back in declaration order.
        assert_eq!(taken, vec![F::ONE, F::from_u64(8)]);

        // Column 0 lost only its first entry.
        assert_eq!(
            table.poly(0).as_slice(),
            &[F::ZERO, F::from_u64(2), F::from_u64(3), F::from_u64(4)]
        );

        // Column 1 lost only its last entry.
        assert_eq!(
            table.poly(1).as_slice(),
            &[F::from_u64(5), F::from_u64(6), F::from_u64(7), F::ZERO]
        );

        // Writing the values back rebuilds the original table entry for entry.
        boundary.restore_cells(&mut table, &taken);
        assert_eq!(table.poly(0).as_slice(), probe_table().poly(0).as_slice());
        assert_eq!(table.poly(1).as_slice(), probe_table().poly(1).as_slice());
    }

    #[test]
    fn reconstruct_recovers_the_true_column() {
        // Invariant: the corrected openings of a blanked column equal the true column's.
        let mut rng = SmallRng::seed_from_u64(0x5A);
        let k = 5usize;
        let height = 1 << k;

        // Fixture state: a random column of height 32 with both ends public.
        //
        //     row 0      : a = 11
        //     row 31     : b = 23
        //     rows 1..31 : random
        let a = F::from_u64(11);
        let b = F::from_u64(23);
        let mut column: Vec<F> = (0..height).map(|_| rng.random()).collect();
        column[0] = a;
        column[height - 1] = b;

        // What the prover commits: the same column with both ends blanked.
        let mut committed = column.clone();
        committed[0] = F::ZERO;
        committed[height - 1] = F::ZERO;

        // Open both columns at one random point, in both views.
        let point = Point::<EF>::rand(&mut rng, k);
        let true_current = Poly::new(column.clone()).eval_base(&point);
        let true_next = Poly::new(column).eval_next_base(&point);
        let committed_current = [Poly::new(committed.clone()).eval_base(&point)];
        let committed_next = [Poly::new(committed).eval_next_base(&point)];

        // Both cells sit on column 0, which the AIR reads ahead.
        // Both views therefore take a correction.
        let cells = [
            BoundaryPublic::new(0, BoundaryEnd::First, 0),
            BoundaryPublic::new(0, BoundaryEnd::Last, 1),
        ];
        let (current, next) = BoundaryIo::new(&cells).reconstruct(
            &committed_current,
            &committed_next,
            &[0],
            point.as_slice(),
            &[a, b],
        );

        // Both corrected views match the true column's own evaluations.
        assert_eq!(*current, [true_current]);
        assert_eq!(*next, [true_next]);
    }

    #[test]
    fn reconstruct_shifts_a_nonzero_committed_cell() {
        // Invariant: reconstruction adds the public value to whatever sits at the cell.
        //   A nonzero committed cell therefore lands on a shifted column, not the true one.
        //
        //     committed cell u -> reconstruction lands on the column whose cell is u + public
        //
        // Nothing in this layer pins u to zero.
        // That is the gap the folder's pin closes.
        let mut rng = SmallRng::seed_from_u64(0x9E);
        let k = 4usize;
        let height = 1 << k;

        // Fixture state: a committed column whose last cell is 5 rather than 0.
        let u = F::from_u64(5);
        let public = F::from_u64(31);
        let mut committed: Vec<F> = (0..height).map(|_| rng.random()).collect();
        committed[height - 1] = u;

        // The column the reconstruction will actually land on.
        let mut shifted = committed.clone();
        shifted[height - 1] = u + public;

        // Open the committed column at one random point, in both views.
        let point = Point::<EF>::rand(&mut rng, k);
        let committed_current = [Poly::new(committed.clone()).eval_base(&point)];
        let committed_next = [Poly::new(committed).eval_next_base(&point)];

        let cells = [BoundaryPublic::new(0, BoundaryEnd::Last, 0)];
        let (current, next) = BoundaryIo::new(&cells).reconstruct(
            &committed_current,
            &committed_next,
            &[0],
            point.as_slice(),
            &[public],
        );

        // Both views match the shifted column rather than the committed one.
        assert_eq!(*current, [Poly::new(shifted.clone()).eval_base(&point)]);
        assert_eq!(*next, [Poly::new(shifted).eval_next_base(&point)]);
    }

    #[test]
    fn reconstruct_skips_successor_for_unread_columns() {
        // Invariant: a cell on a column the AIR never reads ahead corrects one view only.
        //
        // Fixture state: the cell sits on column 1, but only column 0 is read ahead.
        //
        //     current openings: [1, 1]   (columns 0 and 1)
        //     next    openings: [1]      (column 0 only)
        //                       → the cell has no successor slot to correct
        let point = Point::<EF>::rand(&mut SmallRng::seed_from_u64(7), 4);
        let cells = [BoundaryPublic::new(1, BoundaryEnd::Last, 0)];
        let (current, next) = BoundaryIo::new(&cells).reconstruct(
            &[EF::ONE, EF::ONE],
            &[EF::ONE],
            &[0],
            point.as_slice(),
            &[F::from_u64(9)],
        );

        // Column 1's current view gains the last-row weight times the public input.
        let boundary = BoundaryEvals::at(point.as_slice());
        assert_eq!(
            current[1],
            EF::ONE + boundary.last * EF::from(F::from_u64(9))
        );

        // The lone successor slot belongs to column 0 and is left alone.
        assert_eq!(*next, [EF::ONE]);
    }

    #[test]
    fn reconstruct_borrows_when_nothing_is_declared() {
        // Invariant: an AIR with no declared cells pays nothing here.
        //
        //     no declaration → both views borrowed → no copy, no weight computed
        let point = Point::<EF>::rand(&mut SmallRng::seed_from_u64(0x11), 3);
        let (current, next) = BoundaryIo::new(&[]).reconstruct::<F, EF>(
            &[EF::ONE, EF::TWO],
            &[EF::ONE],
            &[0],
            point.as_slice(),
            &[],
        );

        // An owned view would mean the empty case allocated.
        assert!(matches!(current, Cow::Borrowed(_)));
        assert!(matches!(next, Cow::Borrowed(_)));
    }

    #[test]
    fn validate_accepts_distinct_in_range_cells() {
        // Invariant: one column may carry a cell on each end.
        //
        //     (column 1, first) and (column 1, last) are two different trace cells
        let cells = [
            BoundaryPublic::new(1, BoundaryEnd::First, 0),
            BoundaryPublic::new(1, BoundaryEnd::Last, 1),
        ];
        let air = ProbeAir {
            width: 2,
            num_public_values: 2,
            cells: &cells,
        };

        assert_eq!(validate(&air), Ok(()));
    }

    #[test]
    fn validate_rejects_column_past_the_main_width() {
        // Mutation: name column 2 of a width-2 AIR.
        //
        //     columns present: 0, 1
        //     column named   : 2
        //                      → out of range
        let cells = [BoundaryPublic::new(2, BoundaryEnd::First, 0)];
        let air = ProbeAir {
            width: 2,
            num_public_values: 1,
            cells: &cells,
        };

        assert_eq!(
            validate(&air),
            Err(BoundaryIoError::ColumnOutOfRange {
                column: 2,
                width: 2
            })
        );
    }

    #[test]
    fn validate_rejects_public_value_past_the_declared_count() {
        // Mutation: name public value 3 of an AIR declaring one public value.
        //
        //     public values present: 0
        //     public value named   : 3
        //                            → out of range
        let cells = [BoundaryPublic::new(0, BoundaryEnd::First, 3)];
        let air = ProbeAir {
            width: 2,
            num_public_values: 1,
            cells: &cells,
        };

        assert_eq!(
            validate(&air),
            Err(BoundaryIoError::PublicValueOutOfRange {
                index: 3,
                num_public_values: 1
            })
        );
    }

    #[test]
    fn validate_rejects_two_cells_on_one_trace_cell() {
        // Mutation: point two cells at column 0's first row.
        //
        //     blanked   : once
        //     corrected : twice, by public value 0 and public value 1
        //                 → even an honest proof cannot verify
        let cells = [
            BoundaryPublic::new(0, BoundaryEnd::First, 0),
            BoundaryPublic::new(0, BoundaryEnd::First, 1),
        ];
        let air = ProbeAir {
            width: 2,
            num_public_values: 2,
            cells: &cells,
        };

        assert_eq!(
            validate(&air),
            Err(BoundaryIoError::DuplicateCell {
                column: 0,
                end: BoundaryEnd::First
            })
        );
    }
}
