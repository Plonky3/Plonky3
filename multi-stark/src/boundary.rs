//! Borgeaud boundary-IO handling of public inputs.
//!
//! A public boundary cell is committed as zero and restored at verification time
//! from the public input, so the commitment never carries the public data.
//!
//! ```text
//!     committed = true column with its corner cell set to 0
//!     restored  = committed(point) + eq_corner(point) * public
//! ```
//!
//! The prover folds the true table, so the sumcheck transcript reflects the restored values.
//! A corner-zero pin, asserted alongside the AIR constraints, forces the committed corner to zero.
//! The restored value is then exactly the public input.
//!
//! See <https://solvable.group/posts/super-air/> ("Handling public inputs").

use alloc::vec::Vec;

use p3_air::{BoundaryEnd, BoundaryPublic};
use p3_field::{ExtensionField, Field};
use p3_matrix::dense::RowMajorMatrix;
use p3_sumcheck::layout::Table;

use crate::selectors::BoundaryEvals;

/// The public boundary cells of one AIR, and the two halves of their handling.
///
/// One half runs on the prover and blanks each public cell before commitment.
/// The other runs on the verifier and restores the true opened values from the public inputs.
pub(crate) struct BoundaryIo<'a>(&'a [BoundaryPublic]);

impl<'a> BoundaryIo<'a> {
    /// Wrap an AIR's declared public boundary cells.
    pub(crate) const fn new(cells: &'a [BoundaryPublic]) -> Self {
        Self(cells)
    }

    /// Copy of a table with every public boundary cell set to zero.
    ///
    /// The prover commits this copy but folds the original true table.
    /// The verifier's reconstruction then lands exactly on the folded values.
    ///
    /// # Arguments
    ///
    /// - `table`: the true, transposed trace table, one row per column.
    ///
    /// # Returns
    ///
    /// A fresh table equal to the input with each declared corner cell replaced by zero.
    pub(crate) fn commit_zeroed<F: Field>(&self, table: &Table<F>) -> Table<F> {
        // Row count of the trace; each column is one contiguous row-major span of this length.
        let height = 1 << table.num_variables();

        // Copy every column into a fresh buffer: one row per column, `height` entries each.
        let mut values = Vec::with_capacity(table.num_polys() * height);
        for column in table.iter_polys() {
            values.extend_from_slice(column);
        }

        // Blank each declared corner cell in place.
        //   first end -> row 0
        //   last  end -> row height - 1
        for cell in self.0 {
            let row = match cell.end {
                BoundaryEnd::First => 0,
                BoundaryEnd::Last => height - 1,
            };
            values[cell.column * height + row] = F::ZERO;
        }

        Table::new(RowMajorMatrix::new(values, height))
    }

    /// Restore the true opened values from the committed corner-zeroed openings.
    ///
    /// The committed column is the true column with each public cell zeroed.
    /// Their multilinear extensions differ by one Lagrange term per cell:
    /// ```text
    ///     P(point) = committed(point) + eq_corner(point) * public
    /// ```
    ///
    /// - The current-row value adds the corner weight.
    /// - The successor value adds the corner's successor weight.
    /// - The first row has successor weight zero, so only the last row corrects the successor.
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
    /// The reconstructed current-row and successor value vectors.
    pub(crate) fn reconstruct<F, EF>(
        &self,
        current: &[EF],
        next: &[EF],
        next_columns: &[usize],
        point: &[EF],
        public_values: &[F],
    ) -> (Vec<EF>, Vec<EF>)
    where
        F: Field,
        EF: ExtensionField<F>,
    {
        // Start from the committed openings; each public cell adds one correction.
        let mut current = current.to_vec();
        let mut next = next.to_vec();

        // Corner Lagrange weights at this point: first = prod(1 - r), last = prod(r).
        let boundary = BoundaryEvals::at(point);
        // Successor-view weight of the last-row corner at this point.
        let last_successor = BoundaryEvals::last_row_successor_weight(point);

        for cell in self.0 {
            // Lift the public input into the extension field the openings live in.
            let public = EF::from(public_values[cell.public_value]);

            // First row corrects only the current view; last row corrects both.
            let (current_weight, successor_weight) = match cell.end {
                BoundaryEnd::First => (boundary.first, EF::ZERO),
                BoundaryEnd::Last => (boundary.last, last_successor),
            };

            // Current-row claim: add the corner weight times the public input.
            current[cell.column] += current_weight * public;

            // Successor claim: correct only a column whose successor the AIR reads.
            if let Some(slot) = next_columns.iter().position(|&c| c == cell.column) {
                next[slot] += successor_weight * public;
            }
        }

        (current, next)
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

    #[test]
    fn commit_zeroed_blanks_only_declared_corners() {
        // Fixture state: a width-2 trace of height 4 with distinct nonzero entries.
        //   column 0: [1, 2, 3, 4]
        //   column 1: [5, 6, 7, 8]
        let column0 = [1u64, 2, 3, 4].map(F::from_u64);
        let column1 = [5u64, 6, 7, 8].map(F::from_u64);
        let values = column0.iter().chain(&column1).copied().collect();
        let table = Table::new(RowMajorMatrix::new(values, 4));

        // Mutation: blank column 0's first cell and column 1's last cell.
        //   column 0 row 0 -> 0
        //   column 1 row 3 -> 0
        let cells = [
            BoundaryPublic::new(0, BoundaryEnd::First, 0),
            BoundaryPublic::new(1, BoundaryEnd::Last, 1),
        ];
        let zeroed = BoundaryIo::new(&cells).commit_zeroed(&table);

        // Only the two declared corners change; every other entry is untouched.
        assert_eq!(
            zeroed.poly(0).as_slice(),
            &[F::ZERO, F::from_u64(2), F::from_u64(3), F::from_u64(4)]
        );
        assert_eq!(
            zeroed.poly(1).as_slice(),
            &[F::from_u64(5), F::from_u64(6), F::from_u64(7), F::ZERO]
        );
    }

    #[test]
    fn reconstruct_inverts_corner_zeroing() {
        // Invariant: blanking two corners then reconstructing at a random point
        //   recovers the true column's current and successor evaluations.
        let mut rng = SmallRng::seed_from_u64(0x5A);
        let k = 5usize;
        let height = 1 << k;

        // Fixture state: a random column of height 2^5 with known public boundary cells.
        //   row 0        holds public a = 11
        //   row height-1 holds public b = 23
        let a = F::from_u64(11);
        let b = F::from_u64(23);
        let mut column: Vec<F> = (0..height).map(|_| rng.random()).collect();
        column[0] = a;
        column[height - 1] = b;

        // The committed column blanks both boundary corners.
        let mut committed = column.clone();
        committed[0] = F::ZERO;
        committed[height - 1] = F::ZERO;

        // Evaluate the true and committed columns at a random point, both views.
        let point = Point::<EF>::rand(&mut rng, k);
        let true_current = Poly::new(column.clone()).eval_base(&point);
        let true_next = Poly::new(column).eval_next_base(&point);
        let committed_current = Poly::new(committed.clone()).eval_base(&point);
        let committed_next = Poly::new(committed).eval_next_base(&point);

        // Reconstruct from the committed openings using the public values.
        //   column 0 is read on the next row, so both views are corrected.
        let cells = [
            BoundaryPublic::new(0, BoundaryEnd::First, 0),
            BoundaryPublic::new(0, BoundaryEnd::Last, 1),
        ];
        let (current, next) = BoundaryIo::new(&cells).reconstruct(
            &[committed_current],
            &[committed_next],
            &[0],
            point.as_slice(),
            &[a, b],
        );

        // Both reconstructed views equal the true column's evaluations.
        assert_eq!(current, vec![true_current]);
        assert_eq!(next, vec![true_next]);
    }

    #[test]
    fn reconstruct_skips_successor_for_unread_columns() {
        // Invariant: a public cell on a column the AIR never reads ahead corrects
        //   only the current-row claim, leaving the successor slice untouched.
        let point = Point::<EF>::rand(&mut SmallRng::seed_from_u64(7), 4);

        // Fixture state: the public cell sits on column 1, but only column 0 is
        //   read on the next row.
        //   current openings: [1, 1]   (columns 0 and 1)
        //   next    openings: [1]      (column 0 only)
        let cells = [BoundaryPublic::new(1, BoundaryEnd::Last, 0)];
        let (current, next) = BoundaryIo::new(&cells).reconstruct(
            &[EF::ONE, EF::ONE],
            &[EF::ONE],
            &[0],
            point.as_slice(),
            &[F::from_u64(9)],
        );

        // Column 1's current view gains the last-corner weight times the public input.
        let boundary = BoundaryEvals::at(point.as_slice());
        assert_eq!(
            current[1],
            EF::ONE + boundary.last * EF::from(F::from_u64(9))
        );
        // The lone successor slot (column 0) carries no public cell and is unchanged.
        assert_eq!(next, vec![EF::ONE]);
    }
}
