//! Matrix library.

#![no_std]

extern crate alloc;

use alloc::vec::Vec;
use core::fmt::{Debug, Display, Formatter};

use p3_maybe_rayon::prelude::*;

use crate::dense::RowMajorMatrix;
use crate::strided::VerticallyStridedMatrixView;

pub mod bitrev;
pub mod dense;
pub mod mul;
pub mod routines;
pub mod sparse;
pub mod stack;
pub mod strided;
pub mod util;

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

#[derive(Clone, Copy)]
pub struct Dimensions {
    pub width: usize,
    pub height: usize,
}

impl Debug for Dimensions {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}x{}", self.width, self.height)
    }
}

impl Display for Dimensions {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}x{}", self.width, self.height)
    }
}

/// A `Matrix` that supports randomly accessing particular coefficients.
pub trait MatrixGet<T>: Matrix<T> {
    fn get(&self, r: usize, c: usize) -> T;
}

/// A `Matrix` that supports randomly accessing particular rows.
pub trait MatrixRows<T>: Matrix<T> {
    type Row<'a>: IntoIterator<Item = T>
    where
        Self: 'a;

    fn row(&self, r: usize) -> Self::Row<'_>;

    fn rows(&self) -> impl Iterator<Item = Self::Row<'_>> {
        (0..self.height()).map(|r| self.row(r))
    }

    fn row_vec(&self, r: usize) -> Vec<T> {
        self.row(r).into_iter().collect()
    }

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
        RowMajorMatrix::new(
            (0..self.height()).flat_map(|r| self.row(r)).collect(),
            self.width(),
        )
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

    fn row_slices<'a>(&'a self) -> impl Iterator<Item = &'a [T]>
    where
        T: 'a,
    {
        (0..self.height()).map(|r| self.row_slice(r))
    }
}

/// A `Matrix` which supports access its rows as mutable slices.
pub trait MatrixRowSlicesMut<T>: MatrixRowSlices<T> {
    fn row_slice_mut(&mut self, r: usize) -> &mut [T];

    // BEWARE: if we add a matrix type that has several rows in the same memory location,
    // these default implementations will be invalid
    // For example, a "tiling" matrix view that repeats its rows

    /// # Safety
    /// Each row index in `rs` must be unique.
    unsafe fn disjoint_row_slices_mut<const N: usize>(&mut self, rs: [usize; N]) -> [&mut [T]; N] {
        rs.map(|r| {
            let s = self.row_slice_mut(r);
            // launder the lifetime to 'a instead of being bound to self
            unsafe { core::slice::from_raw_parts_mut(s.as_mut_ptr(), s.len()) }
        })
    }
    fn row_pair_slices_mut(&mut self, r0: usize, r1: usize) -> (&mut [T], &mut [T]) {
        // make it safe by ensuring rs unique
        assert_ne!(r0, r1);
        let [s0, s1] = unsafe { self.disjoint_row_slices_mut([r0, r1]) };
        (s0, s1)
    }
}

pub trait MatrixRowChunksMut<T: Send>: MatrixRowSlicesMut<T> {
    type RowChunkMut<'a>: MatrixRowSlicesMut<T> + Send
    where
        Self: 'a;
    fn par_row_chunks_mut(
        &mut self,
        chunk_rows: usize,
    ) -> impl IndexedParallelIterator<Item = Self::RowChunkMut<'_>>;
}

/// A `TransposeMatrix` which supports transpose logic for matrices
pub trait MatrixTranspose<T>: MatrixRows<T> {
    fn transpose(self) -> Self;
}
