//! Matrix given by an explicit entry list.

use alloc::vec::Vec;

use p3_challenger::fs::TranscriptField;
use p3_field::{ExtensionField, Field};
use p3_multilinear_util::poly::Poly;

use super::{LinearOperator, axis_len, encode_usize};

/// Encoding tag.
const TAG: u8 = 3;

/// Matrix stored as its entries, where repeated positions add up.
///
/// The verifier pays linear in rows, columns and entries.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SparseMatrix<F> {
    /// Row variables.
    log_rows: usize,
    /// Column variables.
    log_cols: usize,
    /// Entries as row, column and coefficient.
    entries: Vec<(usize, usize, F)>,
}

impl<F: Field> SparseMatrix<F> {
    /// Build a matrix from its entries.
    ///
    /// # Panics
    ///
    /// - An entry lies outside the matrix, or an axis overflows usize.
    #[must_use]
    pub fn new(log_rows: usize, log_cols: usize, entries: Vec<(usize, usize, F)>) -> Self {
        // Validate positions once.
        let rows = axis_len(log_rows);
        let cols = axis_len(log_cols);
        for &(row, col, _) in &entries {
            assert!(
                row < rows && col < cols,
                "entry ({row}, {col}) is outside a {rows} x {cols} matrix"
            );
        }

        Self {
            log_rows,
            log_cols,
            entries,
        }
    }

    /// Entries as row, column and coefficient.
    #[must_use]
    pub fn entries(&self) -> &[(usize, usize, F)] {
        &self.entries
    }
}

impl<F: Field> LinearOperator<F> for SparseMatrix<F> {
    fn log_rows(&self) -> usize {
        self.log_rows
    }

    fn log_cols(&self) -> usize {
        self.log_cols
    }

    fn apply(&self, z: &[F], out: &mut [F]) {
        assert_eq!(z.len(), 1 << self.log_cols, "witness length");
        assert_eq!(out.len(), 1 << self.log_rows, "output length");

        // out[i] += c * z[j].
        out.fill(F::ZERO);
        for &(row, col, coeff) in &self.entries {
            out[row] += coeff * z[col];
        }
    }

    fn add_transpose<EF: ExtensionField<F>>(&self, weights: &[EF], scale: EF, out: &mut [EF]) {
        assert_eq!(weights.len(), 1 << self.log_rows, "weights length");
        assert_eq!(out.len(), 1 << self.log_cols, "output length");

        // out[j] += scale * w[i] * c, with the base coefficient last.
        for &(row, col, coeff) in &self.entries {
            out[col] += scale * weights[row] * coeff;
        }
    }

    fn evaluate<EF: ExtensionField<F>>(&self, row: &[EF], col: &[EF]) -> EF {
        assert_eq!(row.len(), self.log_rows, "row point length");
        assert_eq!(col.len(), self.log_cols, "column point length");

        // Equality tables on both axes.
        let row_eq = Poly::new_from_point(row, EF::ONE);
        let col_eq = Poly::new_from_point(col, EF::ONE);
        let (row_eq, col_eq) = (row_eq.as_slice(), col_eq.as_slice());

        // M~(row, col) = sum over entries of c * eq(row, i) * eq(col, j).
        self.entries
            .iter()
            .map(|&(i, j, coeff)| row_eq[i] * col_eq[j] * coeff)
            .sum()
    }

    fn encode(&self, out: &mut Vec<u8>)
    where
        F: TranscriptField,
    {
        // Tag, shape and entry count.
        out.push(TAG);
        encode_usize(out, self.log_rows);
        encode_usize(out, self.log_cols);
        encode_usize(out, self.entries.len());

        // Fixed-width entries.
        for &(row, col, coeff) in &self.entries {
            encode_usize(out, row);
            encode_usize(out, col);
            F::encode(&coeff, out);
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

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
    fn the_r1cs_example_matrix_computes_x_and_x_plus_y() {
        // Fixture state: z = (1, 3, 4, 0); row 0 picks x, row 1 adds x + y.
        let a = SparseMatrix::new(1, 2, vec![(0, 1, F::ONE), (1, 1, F::ONE), (1, 2, F::ONE)]);
        let z = [1, 3, 4, 0].map(F::from_u32);
        let mut out = [F::ZERO; 2];
        a.apply(&z, &mut out);
        assert_eq!(out, [3, 7].map(F::from_u32));
    }

    #[test]
    fn repeated_and_weighted_entries_satisfy_the_operator_contract() {
        // Invariant: duplicates add, and coefficients other than one scale.
        //
        // Fixture state: 4 x 8, one repeated position, one coefficient of 5.
        let m = SparseMatrix::new(
            2,
            3,
            vec![
                (0, 7, F::from_u32(5)),
                (0, 7, F::ONE),
                (3, 0, F::NEG_ONE),
                (2, 2, F::TWO),
            ],
        );
        assert_operator_contract::<F, EF, _>(&m, 11);
    }

    #[test]
    #[should_panic(expected = "outside")]
    fn an_entry_outside_the_matrix_is_rejected() {
        // A 2 x 2 matrix has no column 2.
        let _ = SparseMatrix::new(1, 1, vec![(0, 2, F::ONE)]);
    }
}
