use crate::Matrix;
use alloc::vec::Vec;
use rand::distributions::{Distribution, Standard};
use rand::Rng;

/// A dense matrix stored in row-major form.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DenseMatrix<T> {
    /// All values, stored in row-major order.
    pub values: Vec<T>,
    width: usize,
}

impl<T> DenseMatrix<T> {
    pub fn row(&self, r: usize) -> &[T] {
        debug_assert!(r < self.height());
        &self.values[r * self.width..(r + 1) * self.width]
    }

    pub fn row_mut(&mut self, r: usize) -> &mut [T] {
        debug_assert!(r < self.height());
        &mut self.values[r * self.width..(r + 1) * self.width]
    }

    pub fn as_view(&self) -> DenseMatrixView<T> {
        DenseMatrixView {
            values: &self.values,
            width: self.width,
        }
    }

    pub fn as_view_mut(&mut self) -> DenseMatrixViewMut<T> {
        DenseMatrixViewMut {
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
}

impl<T> Matrix<T> for DenseMatrix<T> {
    fn width(&self) -> usize {
        self.width
    }

    fn height(&self) -> usize {
        self.values.len() / self.width
    }
}

#[derive(Copy, Clone)]
pub struct DenseMatrixView<'a, T> {
    pub values: &'a [T],
    width: usize,
}

impl<'a, T> DenseMatrixView<'a, T> {
    pub fn row(&self, r: usize) -> &[T] {
        debug_assert!(r < self.height());
        &self.values[r * self.width..(r + 1) * self.width]
    }
}

impl<T> Matrix<T> for DenseMatrixView<'_, T> {
    fn width(&self) -> usize {
        self.width
    }

    fn height(&self) -> usize {
        self.values.len() / self.width
    }
}

pub struct DenseMatrixViewMut<'a, T> {
    pub values: &'a mut [T],
    width: usize,
}

impl<'a, T> DenseMatrixViewMut<'a, T> {
    pub fn row(&self, r: usize) -> &[T] {
        debug_assert!(r < self.height());
        &self.values[r * self.width..(r + 1) * self.width]
    }

    pub fn row_mut(&mut self, r: usize) -> &mut [T] {
        debug_assert!(r < self.height());
        &mut self.values[r * self.width..(r + 1) * self.width]
    }

    pub fn as_view(&self) -> DenseMatrixView<T> {
        DenseMatrixView {
            values: self.values,
            width: self.width,
        }
    }

    pub fn split_rows(&mut self, r: usize) -> (DenseMatrixViewMut<T>, DenseMatrixViewMut<T>) {
        let (upper_values, lower_values) = self.values.split_at_mut(r * self.width);
        let upper = DenseMatrixViewMut {
            values: upper_values,
            width: self.width,
        };
        let lower = DenseMatrixViewMut {
            values: lower_values,
            width: self.width,
        };
        (upper, lower)
    }
}

impl<T> Matrix<T> for DenseMatrixViewMut<'_, T> {
    fn width(&self) -> usize {
        self.width
    }

    fn height(&self) -> usize {
        self.values.len() / self.width
    }
}
