use core::fmt::Debug;

use p3_util::{log2_strict_usize, reverse_bits_len};

use crate::dense::RowMajorMatrix;
use crate::util::reverse_matrix_index_bits;
use crate::{Matrix, MatrixGet, MatrixRowSlices, MatrixRowSlicesMut, MatrixRows};

/// A matrix that is possibly bit-reversed, and can easily switch
/// between orderings. Pretty much just either `RowMajorMatrix` or
/// `BitReversedMatrixView<RowMajorMatrix>`.
pub trait BitReversableMatrix<T: core::fmt::Debug>: MatrixRowSlicesMut<T> {
    type BitRev: BitReversableMatrix<T>;
    fn bit_reverse_rows(self) -> Self::BitRev;
}

#[derive(Debug)]
pub struct BitReversedMatrixView<Inner> {
    inner: Inner,
    log_height: usize,
}

impl<Inner> BitReversedMatrixView<Inner> {
    pub fn new<T>(inner: Inner) -> Self
    where
        Inner: Matrix<T>,
    {
        let log_height = log2_strict_usize(inner.height());
        Self { inner, log_height }
    }
}

impl<T, Inner: Matrix<T>> Matrix<T> for BitReversedMatrixView<Inner> {
    fn width(&self) -> usize {
        self.inner.width()
    }

    fn height(&self) -> usize {
        self.inner.height()
    }
}

impl<T, Inner: MatrixGet<T>> MatrixGet<T> for BitReversedMatrixView<Inner> {
    fn get(&self, r: usize, c: usize) -> T {
        self.inner.get(reverse_bits_len(r, self.log_height), c)
    }
}

impl<T: core::fmt::Debug, Inner: MatrixRows<T>> MatrixRows<T> for BitReversedMatrixView<Inner> {
    type Row<'a> = Inner::Row<'a> where Self: 'a;

    fn row(&self, r: usize) -> Self::Row<'_> {
        self.inner.row(reverse_bits_len(r, self.log_height))
    }

    fn to_row_major_matrix(self) -> RowMajorMatrix<T>
    where
        Self: Sized,
        T: Clone,
    {
        let mut mat = self.inner.to_row_major_matrix();
        reverse_matrix_index_bits(&mut mat);
        mat
    }
}

impl<T: core::fmt::Debug, Inner: MatrixRowSlices<T>> MatrixRowSlices<T> for BitReversedMatrixView<Inner> {
    fn row_slice(&self, r: usize) -> &[T] {
        self.inner.row_slice(reverse_bits_len(r, self.log_height))
    }
}

impl<T: core::fmt::Debug, Inner: MatrixRowSlicesMut<T>> MatrixRowSlicesMut<T> for BitReversedMatrixView<Inner> {
    fn row_slice_mut(&mut self, r: usize) -> &mut [T] {
        self.inner
            .row_slice_mut(reverse_bits_len(r, self.log_height))
    }
}

impl<T: Clone + Debug> BitReversableMatrix<T> for BitReversedMatrixView<RowMajorMatrix<T>> {
    type BitRev = RowMajorMatrix<T>;
    fn bit_reverse_rows(self) -> Self::BitRev {
        self.inner
    }
}

impl<T: Clone + Debug> BitReversableMatrix<T> for RowMajorMatrix<T> {
    type BitRev = BitReversedMatrixView<RowMajorMatrix<T>>;
    fn bit_reverse_rows(self) -> Self::BitRev {
        BitReversedMatrixView::new(self)
    }
}
