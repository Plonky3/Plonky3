use crate::Matrix;
use alloc::vec::Vec;
use p3_field::Field;
use rand::distributions::{Distribution, Standard};
use rand::Rng;

/// A dense matrix stored in row-major form.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RowMajorMatrix<T> {
    /// All values, stored in row-major order.
    pub values: Vec<T>,
    width: usize,
}

impl<T> RowMajorMatrix<T> {
    #[must_use]
    pub fn new(values: Vec<T>, width: usize) -> Self {
        debug_assert_eq!(values.len() % width, 0);
        Self { values, width }
    }

    pub fn row_mut(&mut self, r: usize) -> &mut [T] {
        debug_assert!(r < self.height());
        &mut self.values[r * self.width..(r + 1) * self.width]
    }

    pub fn rows(&self) -> impl Iterator<Item = &[T]> {
        self.values.chunks_exact(self.width)
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

    fn row(&self, r: usize) -> &[T] {
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
    #[must_use]
    pub fn row(&self, r: usize) -> &[T] {
        debug_assert!(r < self.height());
        &self.values[r * self.width..(r + 1) * self.width]
    }

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

    fn row(&self, r: usize) -> &[T] {
        debug_assert!(r < self.height());
        &self.values[r * self.width..(r + 1) * self.width]
    }
}

pub struct RowMajorMatrixViewMut<'a, T> {
    pub values: &'a mut [T],
    width: usize,
}

impl<'a, T> RowMajorMatrixViewMut<'a, T> {
    #[must_use]
    pub fn row(&self, r: usize) -> &[T] {
        debug_assert!(r < self.height());
        &self.values[r * self.width..(r + 1) * self.width]
    }

    pub fn row_mut(&mut self, r: usize) -> &mut [T] {
        debug_assert!(r < self.height());
        &mut self.values[r * self.width..(r + 1) * self.width]
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

    fn row(&self, r: usize) -> &[T] {
        debug_assert!(r < self.height());
        &self.values[r * self.width..(r + 1) * self.width]
    }
}
