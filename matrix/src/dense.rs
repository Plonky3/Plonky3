use alloc::vec::Vec;
use core::iter::Cloned;
use core::slice;

use p3_field::{ExtensionField, Field};
use p3_maybe_rayon::{IndexedParallelIterator, MaybeParChunksMut, ParallelIterator};
use rand::distributions::{Distribution, Standard};
use rand::Rng;

use crate::{Matrix, MatrixGet, MatrixRowSlices, MatrixRowSlicesMut, MatrixRows, MatrixTranspose};

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
        debug_assert!(width >= 1);
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

    pub fn row_chunks_mut(
        &mut self,
        chunk_rows: usize,
    ) -> impl IndexedParallelIterator<Item = RowMajorMatrixViewMut<T>>
    where
        T: Send,
    {
        self.values
            .par_chunks_exact_mut(self.width * chunk_rows)
            .map(|slice| RowMajorMatrixViewMut::new(slice, self.width))
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

    pub fn scale_row(&mut self, r: usize, scale: T)
    where
        T: Field,
    {
        let row = self.row_slice_mut(r);
        let (prefix, shorts, suffix) = unsafe { row.align_to_mut::<T::Packing>() };
        prefix.iter_mut().for_each(|x| *x *= scale);
        shorts.iter_mut().for_each(|x| *x *= scale);
        suffix.iter_mut().for_each(|x| *x *= scale);
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
    type Row<'a> = Cloned<slice::Iter<'a, T>> where T: 'a;

    fn row(&self, r: usize) -> Self::Row<'_> {
        self.row_slice(r).iter().cloned()
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

impl<T: Clone> MatrixRowSlicesMut<T> for RowMajorMatrix<T> {
    fn row_slice_mut(&mut self, r: usize) -> &mut [T] {
        debug_assert!(r < self.height());
        &mut self.values[r * self.width..(r + 1) * self.width]
    }
}

#[derive(Copy, Clone)]
pub struct RowMajorMatrixView<'a, T> {
    pub values: &'a [T],
    pub width: usize,
}

impl<'a, T> RowMajorMatrixView<'a, T> {
    #[must_use]
    pub fn new(values: &'a mut [T], width: usize) -> Self {
        debug_assert_eq!(values.len() % width, 0);
        Self { values, width }
    }

    pub fn rows(&self) -> impl Iterator<Item = &[T]> {
        self.values.chunks_exact(self.width)
    }

    pub fn split_rows(&self, r: usize) -> (RowMajorMatrixView<T>, RowMajorMatrixView<T>) {
        let (upper_values, lower_values) = self.values.split_at(r * self.width);
        let upper = RowMajorMatrixView {
            values: upper_values,
            width: self.width,
        };
        let lower = RowMajorMatrixView {
            values: lower_values,
            width: self.width,
        };
        (upper, lower)
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

impl<T: Clone> MatrixGet<T> for RowMajorMatrixView<'_, T> {
    fn get(&self, r: usize, c: usize) -> T {
        self.values[r * self.width + c].clone()
    }
}

impl<T: Clone> MatrixRows<T> for RowMajorMatrixView<'_, T> {
    type Row<'a> = Cloned<slice::Iter<'a, T>> where Self: 'a, T: 'a;

    fn row(&self, r: usize) -> Self::Row<'_> {
        self.row_slice(r).iter().cloned()
    }

    fn to_row_major_matrix(self) -> RowMajorMatrix<T>
    where
        Self: Sized,
        T: Clone,
    {
        RowMajorMatrix::new(self.values.to_vec(), self.width)
    }
}

impl<T: Clone> MatrixRowSlices<T> for RowMajorMatrixView<'_, T> {
    fn row_slice(&self, r: usize) -> &[T] {
        debug_assert!(r < self.height());
        &self.values[r * self.width..(r + 1) * self.width]
    }
}

pub struct RowMajorMatrixViewMut<'a, T> {
    pub values: &'a mut [T],
    pub width: usize,
}

impl<'a, T> RowMajorMatrixViewMut<'a, T> {
    #[must_use]
    pub fn new(values: &'a mut [T], width: usize) -> Self {
        debug_assert_eq!(values.len() % width, 0);
        Self { values, width }
    }

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

    /// Return a pair of rows, each in the form (prefix, shorts, suffix), as they would be returned
    /// by the `align_to_mut` method.
    #[allow(clippy::type_complexity)]
    pub fn packing_aligned_rows(
        &mut self,
        row_1: usize,
        row_2: usize,
    ) -> (
        (&mut [T], &mut [T::Packing], &mut [T]),
        (&mut [T], &mut [T::Packing], &mut [T]),
    )
    where
        T: Field,
    {
        let RowMajorMatrixViewMut { values, width } = self;
        let start_1 = row_1 * *width;
        let start_2 = row_2 * *width;
        let (hi_part, lo_part) = values.split_at_mut(start_2);
        let slice_1 = &mut hi_part[start_1..][..*width];
        let slice_2 = &mut lo_part[..*width];
        unsafe {
            (
                slice_1.align_to_mut::<T::Packing>(),
                slice_2.align_to_mut::<T::Packing>(),
            )
        }
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

impl<T: Clone> MatrixGet<T> for RowMajorMatrixViewMut<'_, T> {
    fn get(&self, r: usize, c: usize) -> T {
        self.values[r * self.width + c].clone()
    }
}

impl<T: Clone> MatrixRows<T> for RowMajorMatrixViewMut<'_, T> {
    type Row<'a> = Cloned<slice::Iter<'a, T>> where Self: 'a, T: 'a;

    fn row(&self, r: usize) -> Self::Row<'_> {
        self.row_slice(r).iter().cloned()
    }

    fn to_row_major_matrix(self) -> RowMajorMatrix<T>
    where
        Self: Sized,
        T: Clone,
    {
        RowMajorMatrix::new(self.values.to_vec(), self.width)
    }
}

impl<T: Clone> MatrixRowSlices<T> for RowMajorMatrixViewMut<'_, T> {
    fn row_slice(&self, r: usize) -> &[T] {
        debug_assert!(r < self.height());
        &self.values[r * self.width..(r + 1) * self.width]
    }
}

impl<T: Clone> MatrixRowSlicesMut<T> for RowMajorMatrixViewMut<'_, T> {
    fn row_slice_mut(&mut self, r: usize) -> &mut [T] {
        debug_assert!(r < self.height());
        &mut self.values[r * self.width..(r + 1) * self.width]
    }
}

impl<T> MatrixTranspose<T> for RowMajorMatrix<T>
where
    T: Clone + Send + Sync,
{
    const BLOCK_SIZE: usize = 16;

    fn transpose(self) -> Self {
        let block_size = Self::BLOCK_SIZE;
        let height = self.height();
        let width = self.width();

        let transposed_values = {
            let mut v = Vec::with_capacity(width * height);
            for _ in 0..(width * height) {
                v.push(self.values[0].clone());
            }
            v
        };

        let mut transposed = Self::new(transposed_values, height);

        transposed
            .values
            .par_chunks_mut(height)
            .enumerate()
            .for_each(|(row_ind, row)| {
                row.par_chunks_mut(block_size)
                    .enumerate()
                    .for_each(|(block_num, row_block)| {
                        let row_block_len = row_block.len();
                        (0..row_block_len).for_each(|col_ind| {
                            let src_row_ind = block_size * block_num + col_ind;
                            let src_col_ind = row_ind;
                            let src_index = src_row_ind * width + src_col_ind;

                            row_block[col_ind] = self.values[src_index].clone();
                        });
                    });
            });

        transposed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transpose_square_matrix() {
        const START_INDEX: usize = 1;
        const VALUE_LEN: usize = 9;
        const WIDTH: usize = 3;
        const HEIGHT: usize = 3;

        let matrix_values = (START_INDEX..=VALUE_LEN).collect::<Vec<_>>();
        let matrix = RowMajorMatrix::new(matrix_values, WIDTH);
        let transposed = matrix.transpose();
        let should_be_transposed_values = [1, 4, 7, 2, 5, 8, 3, 6, 9].to_vec();
        let should_be_transposed = RowMajorMatrix::new(should_be_transposed_values, HEIGHT);
        assert_eq!(transposed, should_be_transposed);
    }

    #[test]
    fn test_transpose_row_matrix() {
        const START_INDEX: usize = 1;
        const VALUE_LEN: usize = 30;
        const WIDTH: usize = 1;
        const HEIGHT: usize = 30;

        let matrix_values = (START_INDEX..=VALUE_LEN).collect::<Vec<_>>();
        let matrix = RowMajorMatrix::new(matrix_values.clone(), WIDTH);
        let transposed = matrix.transpose();
        let should_be_transposed = RowMajorMatrix::new(matrix_values.clone(), HEIGHT);
        assert_eq!(transposed, should_be_transposed);
    }

    #[test]
    fn test_transpose_rectangular_matrix() {
        const START_INDEX: usize = 1;
        const VALUE_LEN: usize = 30;
        const WIDTH: usize = 5;
        const HEIGHT: usize = 6;

        let matrix_values = (START_INDEX..=VALUE_LEN).collect::<Vec<_>>();
        let matrix = RowMajorMatrix::new(matrix_values, WIDTH);
        let transposed = matrix.transpose();
        let should_be_transposed_values = [
            1, 6, 11, 16, 21, 26, 2, 7, 12, 17, 22, 27, 3, 8, 13, 18, 23, 28, 4, 9, 14, 19, 24, 29,
            5, 10, 15, 20, 25, 30,
        ]
        .to_vec();
        let should_be_transposed = RowMajorMatrix::new(should_be_transposed_values, HEIGHT);
        assert_eq!(transposed, should_be_transposed);
    }

    #[test]
    fn test_transpose_larger_rectangular_matrix() {
        const START_INDEX: usize = 1;
        const VALUE_LEN: usize = 131072; // 512 * 256
        const WIDTH: usize = 256;
        const HEIGHT: usize = 512;

        let matrix_values = (START_INDEX..=VALUE_LEN).collect::<Vec<_>>();
        let matrix = RowMajorMatrix::new(matrix_values, WIDTH);
        let transposed = matrix.clone().transpose();

        assert_eq!(transposed.width(), HEIGHT);
        assert_eq!(transposed.height(), WIDTH);

        for col_index in 0..WIDTH {
            for row_index in 0..HEIGHT {
                assert_eq!(
                    matrix.values[row_index * WIDTH + col_index],
                    transposed.values[col_index * HEIGHT + row_index]
                );
            }
        }
    }
}
