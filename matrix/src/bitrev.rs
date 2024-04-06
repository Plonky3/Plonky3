use core::fmt::Debug;
use core::ops::Deref;

use p3_field::PackedValue;
use p3_util::{log2_strict_usize, reverse_bits_len};

use crate::dense::{DenseMatrix, DenseStorage, RowMajorMatrix};
use crate::util::reverse_matrix_index_bits;
use crate::Matrix;

/// A matrix that is possibly bit-reversed, and can easily switch
/// between orderings. Pretty much just either `RowMajorMatrix` or
/// `BitReversedMatrixView<RowMajorMatrix>`.
pub trait BitReversableMatrix<T: Send + Sync>: Matrix<T> {
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
        T: Send + Sync,
        Inner: Matrix<T>,
    {
        let log_height = log2_strict_usize(inner.height());
        Self { inner, log_height }
    }
}

impl<T: Send + Sync, Inner: Matrix<T>> Matrix<T> for BitReversedMatrixView<Inner> {
    fn width(&self) -> usize {
        self.inner.width()
    }

    fn height(&self) -> usize {
        self.inner.height()
    }

    fn get(&self, r: usize, c: usize) -> T {
        self.inner.get(reverse_bits_len(r, self.log_height), c)
    }

    type Row<'a> = Inner::Row<'a> where Self: 'a;

    fn row(&self, r: usize) -> Self::Row<'_> {
        self.inner.row(reverse_bits_len(r, self.log_height))
    }

    fn row_slice(&self, r: usize) -> impl Deref<Target = [T]> {
        self.inner.row_slice(reverse_bits_len(r, self.log_height))
    }

    fn horizontally_packed_row<'a, P>(
        &'a self,
        r: usize,
    ) -> (impl Iterator<Item = P>, impl Iterator<Item = T>)
    where
        P: PackedValue<Value = T>,
        T: Clone + 'a,
    {
        self.inner
            .horizontally_packed_row(reverse_bits_len(r, self.log_height))
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

impl<T: Clone + Send + Sync, S: DenseStorage<T>> BitReversableMatrix<T>
    for BitReversedMatrixView<DenseMatrix<T, S>>
{
    type BitRev = DenseMatrix<T, S>;
    fn bit_reverse_rows(self) -> Self::BitRev {
        self.inner
    }
}

impl<T: Clone + Send + Sync, S: DenseStorage<T>> BitReversableMatrix<T> for DenseMatrix<T, S> {
    type BitRev = BitReversedMatrixView<DenseMatrix<T, S>>;
    fn bit_reverse_rows(self) -> Self::BitRev {
        BitReversedMatrixView::new(self)
    }
}
