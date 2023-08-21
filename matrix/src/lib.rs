//! Matrix library.

#![no_std]

extern crate alloc;

use crate::dense::RowMajorMatrix;

pub mod dense;
pub mod mul;
pub mod sparse;
pub mod stack;

pub trait Matrix<T> {
    fn width(&self) -> usize;

    fn height(&self) -> usize;
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
}

/// A `Matrix` which supports access its rows as slices.
pub trait MatrixRowSlices<T>: MatrixRows<T> {
    fn row_slice(&self, r: usize) -> &[T];
}
