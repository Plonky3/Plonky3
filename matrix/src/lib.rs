//! Matrix library.

#![no_std]

extern crate alloc;

use crate::dense::RowMajorMatrix;
use crate::strided::VerticallyStridedMatrixView;

pub mod dense;
pub mod mul;
pub mod sparse;
pub mod stack;
pub mod strided;

pub trait Matrix<T> {
    fn width(&self) -> usize;

    fn height(&self) -> usize;

    fn dimensions(&self) -> Dimensions {
        Dimensions {
            width: self.width(),
            height: self.height(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Dimensions {
    pub width: usize,
    pub height: usize,
}

/// A `Matrix` that supports randomly accessing particular coefficients.
pub trait MatrixGet<T> {
    fn get(&self, r: usize, c: usize) -> T;
}

/// A `Matrix` that supports randomly accessing particular rows.
pub trait MatrixRows<T>: Matrix<T> {
    type Row<'a>: IntoIterator<Item = T>
    where
        Self: 'a;

    fn row(&self, r: usize) -> Self::Row<'_>;

    fn first_row(&self) -> Self::Row<'_> {
        self.row(0)
    }

    fn last_row(&self) -> Self::Row<'_> {
        self.row(self.height() - 1)
    }

    fn to_row_major_matrix(self) -> RowMajorMatrix<T>
    where
        Self: Sized,
        T: Clone,
    {
        todo!()
    }

    fn vertically_strided(self, stride: usize, offset: usize) -> VerticallyStridedMatrixView<Self>
    where
        Self: Sized,
    {
        VerticallyStridedMatrixView {
            inner: self,
            stride,
            offset,
        }
    }
}

/// A `Matrix` which supports access its rows as slices.
pub trait MatrixRowSlices<T>: MatrixRows<T> {
    fn row_slice(&self, r: usize) -> &[T];
}

/// A `Matrix` which supports access its rows as mutable slices.
pub trait MatrixRowSlicesMut<T>: MatrixRowSlices<T> {
    fn row_slice_mut(&mut self, r: usize) -> &mut [T];
}
