//! Structured matrices given by a rule rather than a table of entries.

mod block_diagonal;
mod sparse;
mod trace_column;

use alloc::vec::Vec;

pub use block_diagonal::{BitShift, BlockDiagonal};
use p3_challenger::fs::TranscriptField;
use p3_field::{ExtensionField, Field};
pub use sparse::SparseMatrix;
pub use trace_column::TraceColumn;

/// Matrix given by a rule: apply, transpose and multilinear extension must describe the same matrix.
///
/// Indices are read most significant bit first.
///
/// The generic methods rule out trait objects by design; the closed sum is the operator enum.
pub trait LinearOperator<F: Field>: Sync {
    /// Number of variables indexing a row.
    fn log_rows(&self) -> usize;

    /// Number of variables indexing a column.
    fn log_cols(&self) -> usize;

    /// Overwrite the output with this matrix times the witness.
    ///
    /// # Panics
    ///
    /// - The witness or the output has the wrong length.
    fn apply(&self, z: &[F], out: &mut [F]);

    /// Add the scaled transpose applied to the weights into the output.
    ///
    /// # Panics
    ///
    /// - The weights or the output have the wrong length.
    fn add_transpose<EF: ExtensionField<F>>(&self, weights: &[EF], scale: EF, out: &mut [EF]);

    /// Multilinear extension at a row point and a column point.
    ///
    /// # Panics
    ///
    /// - A point has the wrong number of coordinates.
    fn evaluate<EF: ExtensionField<F>>(&self, row: &[EF], col: &[EF]) -> EF;

    /// Append an injective, self-delimiting encoding: no encoding is a prefix of another.
    fn encode(&self, out: &mut Vec<u8>)
    where
        F: TranscriptField;
}

/// Every matrix kind of this crate, dispatched once per whole-matrix operation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Operator<F> {
    /// Trace column read at a fixed row offset.
    TraceColumn(TraceColumn),
    /// Small block repeated down the diagonal.
    BlockDiagonal(BlockDiagonal<F>),
    /// Explicit entry list.
    Sparse(SparseMatrix<F>),
}

impl<F> From<TraceColumn> for Operator<F> {
    fn from(op: TraceColumn) -> Self {
        Self::TraceColumn(op)
    }
}

impl<F> From<BlockDiagonal<F>> for Operator<F> {
    fn from(op: BlockDiagonal<F>) -> Self {
        Self::BlockDiagonal(op)
    }
}

impl<F> From<SparseMatrix<F>> for Operator<F> {
    fn from(op: SparseMatrix<F>) -> Self {
        Self::Sparse(op)
    }
}

impl<F: Field> LinearOperator<F> for Operator<F> {
    fn log_rows(&self) -> usize {
        match self {
            Self::TraceColumn(op) => op.log_rows(),
            Self::BlockDiagonal(op) => op.log_rows(),
            Self::Sparse(op) => op.log_rows(),
        }
    }

    fn log_cols(&self) -> usize {
        match self {
            Self::TraceColumn(op) => op.log_cols(),
            Self::BlockDiagonal(op) => op.log_cols(),
            Self::Sparse(op) => op.log_cols(),
        }
    }

    fn apply(&self, z: &[F], out: &mut [F]) {
        match self {
            Self::TraceColumn(op) => op.apply(z, out),
            Self::BlockDiagonal(op) => op.apply(z, out),
            Self::Sparse(op) => op.apply(z, out),
        }
    }

    fn add_transpose<EF: ExtensionField<F>>(&self, weights: &[EF], scale: EF, out: &mut [EF]) {
        match self {
            Self::TraceColumn(op) => op.add_transpose(weights, scale, out),
            Self::BlockDiagonal(op) => op.add_transpose(weights, scale, out),
            Self::Sparse(op) => op.add_transpose(weights, scale, out),
        }
    }

    fn evaluate<EF: ExtensionField<F>>(&self, row: &[EF], col: &[EF]) -> EF {
        match self {
            Self::TraceColumn(op) => LinearOperator::<F>::evaluate(op, row, col),
            Self::BlockDiagonal(op) => op.evaluate(row, col),
            Self::Sparse(op) => op.evaluate(row, col),
        }
    }

    fn encode(&self, out: &mut Vec<u8>)
    where
        F: TranscriptField,
    {
        // Distinct leading tags keep kinds apart.
        match self {
            Self::TraceColumn(op) => LinearOperator::<F>::encode(op, out),
            Self::BlockDiagonal(op) => op.encode(out),
            Self::Sparse(op) => op.encode(out),
        }
    }
}

/// Append a size or index as eight big-endian bytes.
pub(crate) fn encode_usize(out: &mut Vec<u8>, value: usize) {
    out.extend_from_slice(&(value as u64).to_be_bytes());
}

/// Number of entries along an axis of the given bit width.
///
/// # Panics
///
/// - The length overflows usize.
pub(crate) fn axis_len(log_len: usize) -> usize {
    u32::try_from(log_len)
        .ok()
        .and_then(|shift| 1usize.checked_shl(shift))
        .expect("axis length overflows usize")
}

/// Checks that an operator's three views describe one matrix.
#[cfg(test)]
pub(crate) mod testing {
    use alloc::vec;
    use alloc::vec::Vec;

    use p3_field::{ExtensionField, Field};
    use p3_multilinear_util::point::Point;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::LinearOperator;

    /// Dense matrix, from the operator applied to every unit vector.
    pub(crate) fn dense<F: Field, O: LinearOperator<F>>(op: &O) -> Vec<Vec<F>> {
        let (rows, cols) = (1 << op.log_rows(), 1 << op.log_cols());
        let mut matrix = vec![vec![F::ZERO; cols]; rows];
        let mut unit = vec![F::ZERO; cols];
        let mut column = vec![F::ZERO; rows];
        for j in 0..cols {
            unit[j] = F::ONE;
            op.apply(&unit, &mut column);
            unit[j] = F::ZERO;
            for (i, &value) in column.iter().enumerate() {
                matrix[i][j] = value;
            }
        }
        matrix
    }

    /// Assert that apply, transpose and evaluation agree on one matrix.
    pub(crate) fn assert_operator_contract<F, EF, O>(op: &O, seed: u64)
    where
        F: Field,
        EF: ExtensionField<F>,
        O: LinearOperator<F>,
        rand::distr::StandardUniform: rand::distr::Distribution<F> + rand::distr::Distribution<EF>,
    {
        let mut rng = SmallRng::seed_from_u64(seed);
        let matrix = dense(op);
        let (rows, cols) = (matrix.len(), matrix[0].len());

        // Apply matches the dense product.
        let z: Vec<F> = (0..cols).map(|_| rng.random()).collect();
        let mut out = vec![F::ZERO; rows];
        op.apply(&z, &mut out);
        for i in 0..rows {
            let expected: F = (0..cols).map(|j| matrix[i][j] * z[j]).sum();
            assert_eq!(out[i], expected, "apply, row {i}");
        }

        // Transpose adds into a random accumulator.
        let weights: Vec<EF> = (0..rows).map(|_| rng.random()).collect();
        let scale: EF = rng.random();
        let start: Vec<EF> = (0..cols).map(|_| rng.random()).collect();
        let mut acc = start.clone();
        op.add_transpose(&weights, scale, &mut acc);
        for j in 0..cols {
            let moved: EF = (0..rows).map(|i| weights[i] * matrix[i][j]).sum();
            assert_eq!(acc[j], start[j] + scale * moved, "transpose, column {j}");
        }

        // Evaluate matches the dense extension.
        let row: Vec<EF> = (0..op.log_rows()).map(|_| rng.random()).collect();
        let col: Vec<EF> = (0..op.log_cols()).map(|_| rng.random()).collect();
        let expected: EF = matrix
            .iter()
            .enumerate()
            .flat_map(|(i, entries)| entries.iter().enumerate().map(move |(j, &m)| (i, j, m)))
            .map(|(i, j, m)| {
                let eq =
                    |point: &[EF], vertex| Point::new(point).equality_at_vertex(vertex).unwrap();
                eq(&row, i) * eq(&col, j) * m
            })
            .sum();
        assert_eq!(op.evaluate(&row, &col), expected, "evaluate");
    }
}
