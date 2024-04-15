use alloc::vec;
use alloc::vec::Vec;
use core::borrow::{Borrow, BorrowMut};
use core::marker::PhantomData;
use core::ops::Deref;
use core::{iter, slice};

use p3_field::{ExtensionField, Field, PackedValue};
use p3_maybe_rayon::prelude::*;
use rand::distributions::{Distribution, Standard};
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::Matrix;

/// A default constant for block size matrix transposition. The value was chosen with 32-byte type, in mind.
const TRANSPOSE_BLOCK_SIZE: usize = 64;

/// A dense matrix stored in row-major form.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DenseMatrix<T, V = Vec<T>> {
    pub values: V,
    pub width: usize,
    _phantom: PhantomData<T>,
}

pub type RowMajorMatrix<T> = DenseMatrix<T, Vec<T>>;
pub type RowMajorMatrixView<'a, T> = DenseMatrix<T, &'a [T]>;
pub type RowMajorMatrixViewMut<'a, T> = DenseMatrix<T, &'a mut [T]>;

pub trait DenseStorage<T>: Borrow<[T]> + Into<Vec<T>> + Send + Sync {}
impl<T, S: Borrow<[T]> + Into<Vec<T>> + Send + Sync> DenseStorage<T> for S {}

impl<T: Clone + Send + Sync + Default> DenseMatrix<T> {
    /// Create a new dense matrix of the given dimensions, backed by a `Vec`, and filled with
    /// default values.
    #[must_use]
    pub fn default(width: usize, height: usize) -> Self {
        Self::new(vec![T::default(); width * height], width)
    }
}

impl<T: Clone + Send + Sync, S: DenseStorage<T>> DenseMatrix<T, S> {
    #[must_use]
    pub fn new(values: S, width: usize) -> Self {
        debug_assert!(width >= 1);
        debug_assert_eq!(values.borrow().len() % width, 0);
        Self {
            values,
            width,
            _phantom: PhantomData,
        }
    }

    #[must_use]
    pub fn new_row(values: S) -> Self {
        let width = values.borrow().len();
        Self::new(values, width)
    }

    #[must_use]
    pub fn new_col(values: S) -> Self {
        Self::new(values, 1)
    }

    pub fn as_view(&self) -> RowMajorMatrixView<'_, T> {
        RowMajorMatrixView::new(self.values.borrow(), self.width)
    }

    pub fn as_view_mut(&mut self) -> RowMajorMatrixViewMut<'_, T>
    where
        S: BorrowMut<[T]>,
    {
        RowMajorMatrixViewMut::new(self.values.borrow_mut(), self.width)
    }

    pub fn flatten_to_base<F: Field>(&self) -> RowMajorMatrix<F>
    where
        T: ExtensionField<F>,
    {
        let width = self.width * T::D;
        let values = self
            .values
            .borrow()
            .iter()
            .flat_map(|x| x.as_base_slice().iter().copied())
            .collect();
        RowMajorMatrix::new(values, width)
    }

    pub fn par_row_slices(&self) -> impl IndexedParallelIterator<Item = &[T]>
    where
        T: Sync,
    {
        self.values.borrow().par_chunks_exact(self.width)
    }

    pub fn row_mut(&mut self, r: usize) -> &mut [T]
    where
        S: BorrowMut<[T]>,
    {
        &mut self.values.borrow_mut()[r * self.width..(r + 1) * self.width]
    }

    pub fn rows_mut(&mut self) -> impl Iterator<Item = &mut [T]>
    where
        S: BorrowMut<[T]>,
    {
        self.values.borrow_mut().chunks_exact_mut(self.width)
    }

    pub fn par_rows_mut<'a>(&'a mut self) -> impl IndexedParallelIterator<Item = &'a mut [T]>
    where
        T: 'a + Send,
        S: BorrowMut<[T]>,
    {
        self.values.borrow_mut().par_chunks_exact_mut(self.width)
    }

    pub fn horizontally_packed_row_mut<P>(&mut self, r: usize) -> (&mut [P], &mut [T])
    where
        P: PackedValue<Value = T>,
        S: BorrowMut<[T]>,
    {
        P::pack_slice_with_suffix_mut(self.row_mut(r))
    }

    pub fn scale_row(&mut self, r: usize, scale: T)
    where
        T: Field,
        S: BorrowMut<[T]>,
    {
        let (packed, sfx) = self.horizontally_packed_row_mut::<T::Packing>(r);
        let packed_scale: T::Packing = scale.into();
        packed.iter_mut().for_each(|x| *x *= packed_scale);
        sfx.iter_mut().for_each(|x| *x *= scale);
    }

    pub fn scale(&mut self, scale: T)
    where
        T: Field,
        S: BorrowMut<[T]>,
    {
        let (packed, sfx) = T::Packing::pack_slice_with_suffix_mut(self.values.borrow_mut());
        let packed_scale: T::Packing = scale.into();
        packed.iter_mut().for_each(|x| *x *= packed_scale);
        sfx.iter_mut().for_each(|x| *x *= scale);
    }

    pub fn split_rows(&self, r: usize) -> (RowMajorMatrixView<T>, RowMajorMatrixView<T>) {
        let (lo, hi) = self.values.borrow().split_at(r * self.width);
        (
            DenseMatrix::new(lo, self.width),
            DenseMatrix::new(hi, self.width),
        )
    }

    pub fn split_rows_mut(
        &mut self,
        r: usize,
    ) -> (RowMajorMatrixViewMut<T>, RowMajorMatrixViewMut<T>)
    where
        S: BorrowMut<[T]>,
    {
        let (lo, hi) = self.values.borrow_mut().split_at_mut(r * self.width);
        (
            DenseMatrix::new(lo, self.width),
            DenseMatrix::new(hi, self.width),
        )
    }

    pub fn par_row_chunks_mut(
        &mut self,
        chunk_rows: usize,
    ) -> impl IndexedParallelIterator<Item = RowMajorMatrixViewMut<T>>
    where
        T: Send,
        S: BorrowMut<[T]>,
    {
        self.values
            .borrow_mut()
            .par_chunks_mut(self.width * chunk_rows)
            .map(|slice| RowMajorMatrixViewMut::new(slice, self.width))
    }

    pub fn par_row_chunks_exact_mut(
        &mut self,
        chunk_rows: usize,
    ) -> impl IndexedParallelIterator<Item = RowMajorMatrixViewMut<T>>
    where
        T: Send,
        S: BorrowMut<[T]>,
    {
        self.values
            .borrow_mut()
            .par_chunks_exact_mut(self.width * chunk_rows)
            .map(|slice| RowMajorMatrixViewMut::new(slice, self.width))
    }

    pub fn row_pair_mut(&mut self, row_1: usize, row_2: usize) -> (&mut [T], &mut [T])
    where
        S: BorrowMut<[T]>,
    {
        debug_assert_ne!(row_1, row_2);
        let start_1 = row_1 * self.width;
        let start_2 = row_2 * self.width;
        let (lo, hi) = self.values.borrow_mut().split_at_mut(start_2);
        (&mut lo[start_1..][..self.width], &mut hi[..self.width])
    }

    #[allow(clippy::type_complexity)]
    pub fn packed_row_pair_mut<P>(
        &mut self,
        row_1: usize,
        row_2: usize,
    ) -> ((&mut [P], &mut [T]), (&mut [P], &mut [T]))
    where
        S: BorrowMut<[T]>,
        P: PackedValue<Value = T>,
    {
        let (slice_1, slice_2) = self.row_pair_mut(row_1, row_2);
        (
            P::pack_slice_with_suffix_mut(slice_1),
            P::pack_slice_with_suffix_mut(slice_2),
        )
    }

    pub fn bit_reversed_zero_pad(self, added_bits: usize) -> RowMajorMatrix<T>
    where
        T: Copy + Default + Send + Sync,
    {
        if added_bits == 0 {
            return self.to_row_major_matrix();
        }

        // This is equivalent to:
        //     reverse_matrix_index_bits(mat);
        //     mat
        //         .values
        //         .resize(mat.values.len() << added_bits, F::zero());
        //     reverse_matrix_index_bits(mat);
        // But rather than implement it with bit reversals, we directly construct the resulting matrix,
        // whose rows are zero except for rows whose low `added_bits` bits are zero.

        let w = self.width;
        let mut padded = RowMajorMatrix::new(
            vec![T::default(); self.values.borrow().len() << added_bits],
            w,
        );
        padded
            .par_row_chunks_exact_mut(1 << added_bits)
            .zip(self.par_row_slices())
            .for_each(|(mut ch, r)| ch.row_mut(0).copy_from_slice(r));

        padded
    }
}

impl<T: Clone + Send + Sync, S: DenseStorage<T>> Matrix<T> for DenseMatrix<T, S> {
    fn width(&self) -> usize {
        self.width
    }
    fn height(&self) -> usize {
        self.values.borrow().len() / self.width
    }
    fn get(&self, r: usize, c: usize) -> T {
        self.values.borrow()[r * self.width + c].clone()
    }
    type Row<'a> = iter::Cloned<slice::Iter<'a, T>> where Self: 'a;
    fn row(&self, r: usize) -> Self::Row<'_> {
        self.values.borrow()[r * self.width..(r + 1) * self.width]
            .iter()
            .cloned()
    }
    fn row_slice(&self, r: usize) -> impl Deref<Target = [T]> {
        &self.values.borrow()[r * self.width..(r + 1) * self.width]
    }
    fn to_row_major_matrix(self) -> RowMajorMatrix<T>
    where
        Self: Sized,
        T: Clone,
    {
        RowMajorMatrix::new(self.values.into(), self.width)
    }
    fn horizontally_packed_row<'a, P>(
        &'a self,
        r: usize,
    ) -> (impl Iterator<Item = P>, impl Iterator<Item = T>)
    where
        P: PackedValue<Value = T>,
        T: Clone + 'a,
    {
        let buf = &self.values.borrow()[r * self.width..(r + 1) * self.width];
        let (packed, sfx) = P::pack_slice_with_suffix(buf);
        (packed.iter().cloned(), sfx.iter().cloned())
    }
}

impl<T: Clone + Default + Send + Sync> DenseMatrix<T, Vec<T>> {
    pub fn rand<R: Rng>(rng: &mut R, rows: usize, cols: usize) -> Self
    where
        Standard: Distribution<T>,
    {
        let values = rng.sample_iter(Standard).take(rows * cols).collect();
        Self::new(values, cols)
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
        Self::new(values, cols)
    }

    pub fn transpose(self) -> Self {
        let block_size = TRANSPOSE_BLOCK_SIZE;
        let height = self.height();
        let width = self.width();

        let transposed_values: Vec<T> = vec![T::default(); width * height];
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
                            let original_mat_row_ind = block_size * block_num + col_ind;
                            let original_mat_col_ind = row_ind;
                            let original_values_index =
                                original_mat_row_ind * width + original_mat_col_ind;

                            row_block[col_ind] = self.values[original_values_index].clone();
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
        let should_be_transposed_values = vec![1, 4, 7, 2, 5, 8, 3, 6, 9];
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
        let should_be_transposed = RowMajorMatrix::new(matrix_values, HEIGHT);
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
        let should_be_transposed_values = vec![
            1, 6, 11, 16, 21, 26, 2, 7, 12, 17, 22, 27, 3, 8, 13, 18, 23, 28, 4, 9, 14, 19, 24, 29,
            5, 10, 15, 20, 25, 30,
        ];
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

    #[test]
    fn test_transpose_very_large_rectangular_matrix() {
        const START_INDEX: usize = 1;
        const VALUE_LEN: usize = 1048576; // 512 * 256
        const WIDTH: usize = 1024;
        const HEIGHT: usize = 1024;

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
