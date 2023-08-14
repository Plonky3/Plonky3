use alloc::vec::Vec;

use p3_field::{ExtensionField, Field};
use p3_maybe_rayon::{MaybeParChunksMut, ParallelIterator};
use rand::distributions::{Distribution, Standard};
use rand::Rng;

use crate::{Matrix, MatrixGet, MatrixRowSlices, MatrixRows};

/// A dense matrix stored in row-major form.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RowMajorMatrix<T> {
    /// All values, stored in row-major order.
    pub values: Vec<T>,
    pub width: usize,
}

impl<T> RowMajorMatrix<T> {
    #[must_use]
    pub fn new(values: Vec<T>, width: usize) -> Self {
        debug_assert_eq!(values.len() % width, 0);
        Self { values, width }
    }

    #[must_use]
    pub fn new_row(values: Vec<T>) -> Self {
        let width = values.len();
        Self { values, width }
    }

    #[must_use]
    pub fn new_col(values: Vec<T>) -> Self {
        Self { values, width: 1 }
    }

    pub fn row_mut(&mut self, r: usize) -> &mut [T] {
        debug_assert!(r < self.height());
        &mut self.values[r * self.width..(r + 1) * self.width]
    }

    pub fn rows(&self) -> impl Iterator<Item = &[T]> {
        self.values.chunks_exact(self.width)
    }

    pub fn rows_mut(&mut self) -> impl Iterator<Item = &mut [T]> {
        self.values.chunks_exact_mut(self.width)
    }

    #[must_use]
    pub fn as_view(&self) -> RowMajorMatrixView<T> {
        RowMajorMatrixView {
            values: &self.values,
            width: self.width,
        }
    }

    pub fn as_view_mut(&mut self) -> RowMajorMatrixViewMut<T> {
        RowMajorMatrixViewMut {
            values: &mut self.values,
            width: self.width,
        }
    }

    /// Expand this matrix, if necessary, to a minimum of `height` rows.
    pub fn expand_to_height(&mut self, height: usize)
    where
        T: Default + Clone,
    {
        if self.height() < height {
            self.values.resize(self.width * height, T::default());
        }
    }

    pub fn map<U, F: Fn(T) -> U>(&self, f: F) -> RowMajorMatrix<U>
    where
        T: Clone,
    {
        RowMajorMatrix {
            values: self.values.iter().map(|v| f(v.clone())).collect(),
            width: self.width,
        }
    }

    pub fn to_ext<EF: ExtensionField<T>>(&self) -> RowMajorMatrix<EF>
    where
        T: Field,
    {
        self.map(EF::from_base)
    }

    pub fn rand<R: Rng>(rng: &mut R, rows: usize, cols: usize) -> Self
    where
        Standard: Distribution<T>,
    {
        let values = rng.sample_iter(Standard).take(rows * cols).collect();
        Self {
            values,
            width: cols,
        }
    }

    pub fn rand_nonzero<R: Rng>(rng: &mut R, rows: usize, cols: usize) -> Self
    where
        T: Field,
        Standard: Distribution<T>,
    {
        let values = rng
            .sample_iter(Standard)
            .filter(|x| !x.is_zero())
            .take(rows * cols)
            .collect();
        Self {
            values,
            width: cols,
        }
    }
}

impl<T> Matrix<T> for RowMajorMatrix<T> {
    fn width(&self) -> usize {
        self.width
    }

    fn height(&self) -> usize {
        self.values.len() / self.width
    }
}

impl<T: Clone> MatrixGet<T> for RowMajorMatrix<T> {
    fn get(&self, r: usize, c: usize) -> T {
        self.values[r * self.width + c].clone()
    }
}

impl<T: Clone> MatrixRows<T> for RowMajorMatrix<T> {
    type Row = Vec<T>;

    fn row(&self, r: usize) -> Self::Row {
        // TODO: The copying here is unfortunate, should Row be a GAT to we can return a slice?
        self.row_slice(r).to_vec()
    }

    fn to_row_major_matrix(self) -> RowMajorMatrix<T>
    where
        Self: Sized,
    {
        self
    }
}

impl<T: Clone> MatrixRowSlices<T> for RowMajorMatrix<T> {
    fn row_slice(&self, r: usize) -> &[T] {
        debug_assert!(r < self.height());
        &self.values[r * self.width..(r + 1) * self.width]
    }
}

#[derive(Copy, Clone)]
pub struct RowMajorMatrixView<'a, T> {
    pub values: &'a [T],
    width: usize,
}

impl<'a, T> RowMajorMatrixView<'a, T> {
    pub fn rows(&self) -> impl Iterator<Item = &[T]> {
        self.values.chunks_exact(self.width)
    }
}

impl<T> Matrix<T> for RowMajorMatrixView<'_, T> {
    fn width(&self) -> usize {
        self.width
    }

    fn height(&self) -> usize {
        self.values.len() / self.width
    }
}

impl<T: Clone> MatrixRows<T> for RowMajorMatrixView<'_, T> {
    type Row = Vec<T>;

    fn row(&self, r: usize) -> Self::Row {
        // TODO: The copying here is unfortunate, should Row be a GAT to we can return a slice?
        self.row_slice(r).to_vec()
    }

    fn to_row_major_matrix(self) -> RowMajorMatrix<T>
    where
        Self: Sized,
        T: Clone,
    {
        RowMajorMatrix::new(self.values.to_vec(), self.width)
    }
}

impl<'a, T: Clone> MatrixRowSlices<T> for RowMajorMatrixView<'a, T> {
    fn row_slice(&self, r: usize) -> &[T] {
        debug_assert!(r < self.height());
        &self.values[r * self.width..(r + 1) * self.width]
    }
}

pub struct RowMajorMatrixViewMut<'a, T> {
    pub values: &'a mut [T],
    width: usize,
}

impl<'a, T> RowMajorMatrixViewMut<'a, T> {
    pub fn row_mut(&mut self, r: usize) -> &mut [T] {
        debug_assert!(r < self.height());
        &mut self.values[r * self.width..(r + 1) * self.width]
    }

    pub fn rows_mut(&mut self) -> impl Iterator<Item = &mut [T]> {
        self.values.chunks_exact_mut(self.width)
    }

    pub fn par_rows_mut(&mut self) -> impl ParallelIterator<Item = &mut [T]>
    where
        T: Send,
    {
        self.values.par_chunks_exact_mut(self.width)
    }

    pub fn rows(&self) -> impl Iterator<Item = &[T]> {
        self.values.chunks_exact(self.width)
    }

    #[must_use]
    pub fn as_view(&self) -> RowMajorMatrixView<T> {
        RowMajorMatrixView {
            values: self.values,
            width: self.width,
        }
    }

    pub fn split_rows(&mut self, r: usize) -> (RowMajorMatrixViewMut<T>, RowMajorMatrixViewMut<T>) {
        let (upper_values, lower_values) = self.values.split_at_mut(r * self.width);
        let upper = RowMajorMatrixViewMut {
            values: upper_values,
            width: self.width,
        };
        let lower = RowMajorMatrixViewMut {
            values: lower_values,
            width: self.width,
        };
        (upper, lower)
    }
}

impl<T> Matrix<T> for RowMajorMatrixViewMut<'_, T> {
    fn width(&self) -> usize {
        self.width
    }

    fn height(&self) -> usize {
        self.values.len() / self.width
    }
}

impl<T: Clone> MatrixRows<T> for RowMajorMatrixViewMut<'_, T> {
    type Row = Vec<T>;

    fn row(&self, r: usize) -> Self::Row {
        // TODO: The copying here is unfortunate, should Row be a GAT to we can return a slice?
        self.row_slice(r).to_vec()
    }

    fn to_row_major_matrix(self) -> RowMajorMatrix<T>
    where
        Self: Sized,
        T: Clone,
    {
        RowMajorMatrix::new(self.values.to_vec(), self.width)
    }
}

impl<'a, T: Clone> MatrixRowSlices<T> for RowMajorMatrixViewMut<'a, T> {
    fn row_slice(&self, r: usize) -> &[T] {
        debug_assert!(r < self.height());
        &self.values[r * self.width..(r + 1) * self.width]
    }
}
