use p3_util::{log2_strict_usize, reverse_bits_len};

use crate::dense::{DenseMatrix, DenseStorage};
use crate::permuted::{PermutedMatrix, RowPermutation};
use crate::Matrix;

/// A matrix that is possibly bit-reversed, and can easily switch
/// between orderings. Pretty much just either `RowMajorMatrix` or
/// `BitReversedMatrixView<RowMajorMatrix>`.
pub trait BitReversableMatrix<T: Send + Sync>: Matrix<T> {
    type BitRev: BitReversableMatrix<T>;
    fn bit_reverse_rows(self) -> Self::BitRev;
}

#[derive(Debug)]
pub struct BitrevPerm {
    log_height: usize,
}

impl RowPermutation for BitrevPerm {
    fn new<T: Send + Sync, Inner: Matrix<T>>(inner: &Inner) -> Self {
        Self {
            log_height: log2_strict_usize(inner.height()),
        }
    }
    fn height(&self) -> usize {
        1 << self.log_height
    }
    fn permute_row_index(&self, r: usize) -> usize {
        reverse_bits_len(r, self.log_height)
    }
}

pub type BitReversedMatrixView<Inner> = PermutedMatrix<BitrevPerm, Inner>;

impl<T: Clone + Send + Sync, S: DenseStorage<T>> BitReversableMatrix<T>
    for BitReversedMatrixView<DenseMatrix<T, S>>
{
    type BitRev = DenseMatrix<T, S>;
    fn bit_reverse_rows(self) -> Self::BitRev {
        self.inner()
    }
}

impl<T: Clone + Send + Sync, S: DenseStorage<T>> BitReversableMatrix<T> for DenseMatrix<T, S> {
    type BitRev = BitReversedMatrixView<DenseMatrix<T, S>>;
    fn bit_reverse_rows(self) -> Self::BitRev {
        BitReversedMatrixView::new(self)
    }
}
