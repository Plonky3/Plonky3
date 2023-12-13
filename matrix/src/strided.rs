use crate::{
    view::{MatrixView, RowPermutation},
    Matrix, MatrixRows,
};

pub struct VerticallyStridedMatrixView<Inner> {
    pub(crate) inner: Inner,
    pub(crate) stride: usize,
    pub(crate) offset: usize,
}

impl<T, Inner: Matrix<T>> Matrix<T> for VerticallyStridedMatrixView<Inner> {
    fn width(&self) -> usize {
        self.inner.width()
    }

    fn height(&self) -> usize {
        let h = self.inner.height();
        let full_strides = h / self.stride;
        let remainder = h % self.stride;
        let final_stride = self.offset < remainder;
        full_strides + final_stride as usize
    }
}

impl<T, Inner: MatrixRows<T>> MatrixRows<T> for VerticallyStridedMatrixView<Inner> {
    type Row<'a> = Inner::Row<'a> where Self: 'a;

    fn row(&self, r: usize) -> Self::Row<'_> {
        self.inner.row(r * self.stride + self.offset)
    }

    type Permuted = MatrixView<T, Self>;
    fn permute_rows(self, perm: RowPermutation) -> Self::Permuted
    where
        Self: Sized,
    {
        MatrixView::new(self, perm)
    }
}
