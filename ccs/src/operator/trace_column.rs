//! Trace column read a fixed number of rows ahead.

use alloc::vec::Vec;

use p3_challenger::fs::TranscriptField;
use p3_field::{ExtensionField, Field};
use p3_maybe_rayon::prelude::*;
use p3_multilinear_util::point::Point;

use super::{LinearOperator, axis_len, encode_usize};
use crate::mle::shift_indicator;

/// Encoding tag.
const TAG: u8 = 1;

/// Reads column c at row i + k of a column-major trace (cell c * H + i); the last k rows are zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TraceColumn {
    /// Trace row variables.
    log_height: usize,
    /// Trace column variables.
    log_width: usize,
    /// Column this matrix reads.
    column: usize,
    /// Rows ahead of the current row.
    offset: usize,
}

impl TraceColumn {
    /// Read a column at a row offset.
    ///
    /// # Panics
    ///
    /// - The column or offset is out of range, or the table overflows usize.
    #[must_use]
    pub fn new(log_height: usize, log_width: usize, column: usize, offset: usize) -> Self {
        let height = axis_len(log_height);
        let width = axis_len(log_width);
        assert!(
            column < width,
            "column {column} is outside a table of {width} columns"
        );
        assert!(
            offset < height,
            "offset {offset} skips past a table of {height} rows"
        );

        // The whole witness must be addressable.
        axis_len(log_height + log_width);

        Self {
            log_height,
            log_width,
            column,
            offset,
        }
    }

    /// Trace row variables.
    #[must_use]
    pub const fn log_height(&self) -> usize {
        self.log_height
    }

    /// Trace column variables.
    #[must_use]
    pub const fn log_width(&self) -> usize {
        self.log_width
    }

    /// Column this matrix reads.
    #[must_use]
    pub const fn column(&self) -> usize {
        self.column
    }

    /// Rows ahead of the current row.
    #[must_use]
    pub const fn offset(&self) -> usize {
        self.offset
    }

    /// Number of variables indexing a row: the trace row variables.
    #[must_use]
    pub const fn log_rows(&self) -> usize {
        self.log_height
    }

    /// Number of variables indexing a witness entry: column then row variables.
    #[must_use]
    pub const fn log_cols(&self) -> usize {
        self.log_height + self.log_width
    }

    /// Start and length of the witness range this matrix reads.
    const fn live_range(&self) -> (usize, usize) {
        let height = 1 << self.log_height;
        (self.column * height + self.offset, height - self.offset)
    }
}

impl<F: Field> LinearOperator<F> for TraceColumn {
    fn log_rows(&self) -> usize {
        Self::log_rows(self)
    }

    fn log_cols(&self) -> usize {
        Self::log_cols(self)
    }

    fn apply(&self, z: &[F], out: &mut [F]) {
        assert_eq!(
            z.len(),
            1 << (self.log_height + self.log_width),
            "witness length"
        );
        assert_eq!(out.len(), 1 << self.log_height, "output length");

        // Live rows copy a contiguous slice.
        let (start, live) = self.live_range();
        out[..live].copy_from_slice(&z[start..start + live]);

        // The last k rows are zero.
        out[live..].fill(F::ZERO);
    }

    fn add_transpose<EF: ExtensionField<F>>(&self, weights: &[EF], scale: EF, out: &mut [EF]) {
        assert_eq!(weights.len(), 1 << self.log_height, "weights length");
        assert_eq!(
            out.len(),
            1 << (self.log_height + self.log_width),
            "output length"
        );

        // out[c * H + k + i] += scale * w[i] for i < H - k, in parallel.
        let (start, live) = self.live_range();
        out[start..start + live]
            .par_iter_mut()
            .zip(&weights[..live])
            .with_min_task_bytes(2 * size_of::<EF>())
            .for_each(|(out, &w)| *out += w * scale);
    }

    fn evaluate<EF: ExtensionField<F>>(&self, row: &[EF], col: &[EF]) -> EF {
        assert_eq!(row.len(), self.log_height, "row point length");
        assert_eq!(
            col.len(),
            self.log_height + self.log_width,
            "column point length"
        );

        // Column bits come first.
        let (col_bits, cell_row) = col.split_at(self.log_width);

        // One iff the column matches and the cell row is row + offset; the column always fits.
        let column = Point::new(col_bits)
            .equality_at_vertex(self.column)
            .unwrap_or(EF::ZERO);
        column * shift_indicator(row, cell_row, self.offset)
    }

    fn encode(&self, out: &mut Vec<u8>)
    where
        F: TranscriptField,
    {
        // Tag, then fixed-width parameters.
        out.push(TAG);
        encode_usize(out, self.log_height);
        encode_usize(out, self.log_width);
        encode_usize(out, self.column);
        encode_usize(out, self.offset);
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;

    use super::*;
    use crate::operator::testing::assert_operator_contract;

    /// Base field.
    type F = BabyBear;
    /// Challenge field.
    type EF = BinomialExtensionField<F, 4>;

    #[test]
    fn fibonacci_next_row_reads_column_b_one_row_later() {
        // Fixture state: Fibonacci columns a = (1, 1, 2, 3) and b = (1, 2, 3, 5).
        //
        // Column b one row ahead is b shifted up with a trailing zero.
        let z: Vec<F> = [1, 1, 2, 3, 1, 2, 3, 5].map(F::from_u32).to_vec();
        let mut out = vec![F::ONE; 4];
        TraceColumn::new(2, 1, 1, 1).apply(&z, &mut out);
        assert_eq!(out, [2, 3, 5, 0].map(F::from_u32));
    }

    #[test]
    fn every_column_and_offset_satisfies_the_operator_contract() {
        // Invariant: apply, transpose and evaluation describe the same matrix.
        //
        // Fixture state: 8 x 4 table, every column and offset.
        for column in 0..4 {
            for offset in 0..8 {
                assert_operator_contract::<F, EF, _>(&TraceColumn::new(3, 2, column, offset), 7);
            }
        }
    }

    #[test]
    #[should_panic(expected = "skips past")]
    fn an_offset_of_the_full_height_is_rejected() {
        // Offset 4 on a 4-row table would be all zeros.
        let _ = TraceColumn::new(2, 1, 0, 4);
    }
}
