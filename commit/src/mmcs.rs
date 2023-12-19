use alloc::vec;
use alloc::vec::Vec;
use core::fmt::Debug;
use serde::de::DeserializeOwned;
use serde::Serialize;

use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::{Dimensions, Matrix, MatrixRows};

/// A "Mixed Matrix Commitment Scheme" (MMCS) is a generalization of a vector commitment scheme; it
/// supports committing to matrices and then opening rows. It is also batch-oriented; one can commit
/// to a batch of matrices at once even if their widths and heights differ.
///
/// When a particular row index is opened, it is interpreted directly as a row index for matrices
/// with the largest height. For matrices with smaller heights, some bits of the row index are
/// removed (from the least-significant side) to get the effective row index. These semantics are
/// useful in the FRI protocol. See the documentation for `open_batch` for more details.
///
/// The `DirectMmcs` sub-trait represents an MMS which can be directly constructed from a set of
/// matrices. Other MMCSs may be virtual combinations of child MMCSs, or may be constructed in a
/// streaming manner.
pub trait Mmcs<T>: Clone {
    type ProverData;
    type Commitment: Clone + Serialize + DeserializeOwned + IntoIterator;
    type Proof: Serialize + DeserializeOwned;
    type Error: Debug;
    type Mat<'a>: MatrixRows<T> + Sync
    where
        Self: 'a;

    /// Opens a batch of rows from committed matrices
    /// returns `(openings, proof)`
    /// where `openings` is a vector whose `i`th element is the `j`th row of the ith matrix `M[i]`,
    /// and `j = index >> (log2_ceil(max_height) - log2_ceil(M[i].height))`.
    fn open_batch(
        &self,
        index: usize,
        prover_data: &Self::ProverData,
    ) -> (Vec<Vec<T>>, Self::Proof);

    /// Get the matrices that were committed to.
    fn get_matrices<'a>(&'a self, prover_data: &'a Self::ProverData) -> Vec<Self::Mat<'a>>;

    fn get_matrix_heights(&self, prover_data: &Self::ProverData) -> Vec<usize> {
        self.get_matrices(prover_data)
            .iter()
            .map(|matrix| matrix.height())
            .collect()
    }

    /// Get the largest height of any committed matrix.
    fn get_max_height(&self, prover_data: &Self::ProverData) -> usize {
        self.get_matrix_heights(prover_data)
            .into_iter()
            .max()
            .unwrap_or_else(|| panic!("No committed matrices?"))
    }

    /// Verify a batch opening.
    /// `index` is the row index we're opening for each matrix, following the same
    /// semantics as `open_batch`.
    /// `dimensions` is a slice whose ith element is the dimensions of the the matrix being opened
    /// in the ith opening
    fn verify_batch(
        &self,
        commit: &Self::Commitment,
        dimensions: &[Dimensions],
        index: usize,
        opened_values: &[Vec<T>],
        proof: &Self::Proof,
    ) -> Result<(), Self::Error>;
}

/// An MMCS over explicit inputs which are supplied upfront.
pub trait DirectMmcs<T>: Mmcs<T> {
    fn commit(&self, inputs: Vec<RowMajorMatrix<T>>) -> (Self::Commitment, Self::ProverData);

    fn commit_matrix(&self, input: RowMajorMatrix<T>) -> (Self::Commitment, Self::ProverData) {
        self.commit(vec![input])
    }

    fn commit_vec(&self, input: Vec<T>) -> (Self::Commitment, Self::ProverData) {
        self.commit_matrix(RowMajorMatrix::new_col(input))
    }
}
