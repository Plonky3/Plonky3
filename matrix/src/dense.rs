use alloc::borrow::Cow;
use alloc::vec;
use alloc::vec::Vec;
use core::borrow::{Borrow, BorrowMut};
use core::iter;
use core::marker::PhantomData;
use core::ops::Deref;

use p3_field::{
    ExtensionField, Field, PackedValue, par_scale_slice_in_place, scale_slice_in_place_single_core,
};
use p3_maybe_rayon::prelude::*;
use rand::Rng;
use rand::distr::{Distribution, StandardUniform};
use serde::{Deserialize, Serialize};
use tracing::instrument;

use crate::Matrix;

/// A dense matrix in row-major format, with customizable backing storage.
///
/// The data is stored as a flat buffer, where rows are laid out consecutively.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DenseMatrix<T, V = Vec<T>> {
    /// Flat buffer of matrix values in row-major order.
    pub values: V,
    /// Number of columns in the matrix.
    ///
    /// The number of rows is implicitly determined as `values.len() / width`.
    pub width: usize,
    /// Marker for the element type `T`, unused directly.
    ///
    /// Required to retain type information when `V` does not own or contain `T`.
    _phantom: PhantomData<T>,
}

pub type RowMajorMatrix<T> = DenseMatrix<T>;
pub type RowMajorMatrixView<'a, T> = DenseMatrix<T, &'a [T]>;
pub type RowMajorMatrixViewMut<'a, T> = DenseMatrix<T, &'a mut [T]>;
pub type RowMajorMatrixCow<'a, T> = DenseMatrix<T, Cow<'a, [T]>>;

pub trait DenseStorage<T>: Borrow<[T]> + Send + Sync {
    fn to_vec(self) -> Vec<T>;
}

// Cow doesn't impl IntoOwned so we can't blanket it
impl<T: Clone + Send + Sync> DenseStorage<T> for Vec<T> {
    fn to_vec(self) -> Self {
        self
    }
}

impl<T: Clone + Send + Sync> DenseStorage<T> for &[T] {
    fn to_vec(self) -> Vec<T> {
        <[T]>::to_vec(self)
    }
}

impl<T: Clone + Send + Sync> DenseStorage<T> for &mut [T] {
    fn to_vec(self) -> Vec<T> {
        <[T]>::to_vec(self)
    }
}

impl<T: Clone + Send + Sync> DenseStorage<T> for Cow<'_, [T]> {
    fn to_vec(self) -> Vec<T> {
        self.into_owned()
    }
}

impl<T: Clone + Send + Sync + Default> DenseMatrix<T> {
    /// Create a new dense matrix of the given dimensions, backed by a `Vec`, and filled with
    /// default values.
    #[must_use]
    pub fn default(width: usize, height: usize) -> Self {
        Self::new(vec![T::default(); width * height], width)
    }
}

impl<T: Clone + Send + Sync, S: DenseStorage<T>> DenseMatrix<T, S> {
    /// Create a new dense matrix of the given dimensions, backed by the given storage.
    ///
    /// Note that it is undefined behavior to create a matrix such that
    /// `values.len() % width != 0`.
    #[must_use]
    pub fn new(values: S, width: usize) -> Self {
        debug_assert!(values.borrow().len().is_multiple_of(width));
        Self {
            values,
            width,
            _phantom: PhantomData,
        }
    }

    /// Create a new RowMajorMatrix containing a single row.
    #[must_use]
    pub fn new_row(values: S) -> Self {
        let width = values.borrow().len();
        Self::new(values, width)
    }

    /// Create a new RowMajorMatrix containing a single column.
    #[must_use]
    pub fn new_col(values: S) -> Self {
        Self::new(values, 1)
    }

    /// Get a view of the matrix, i.e. a reference to the underlying data.
    pub fn as_view(&self) -> RowMajorMatrixView<'_, T> {
        RowMajorMatrixView::new(self.values.borrow(), self.width)
    }

    /// Get a mutable view of the matrix, i.e. a mutable reference to the underlying data.
    pub fn as_view_mut(&mut self) -> RowMajorMatrixViewMut<'_, T>
    where
        S: BorrowMut<[T]>,
    {
        RowMajorMatrixViewMut::new(self.values.borrow_mut(), self.width)
    }

    /// Copy the values from the given matrix into this matrix.
    pub fn copy_from<S2>(&mut self, source: &DenseMatrix<T, S2>)
    where
        T: Copy,
        S: BorrowMut<[T]>,
        S2: DenseStorage<T>,
    {
        assert_eq!(self.dimensions(), source.dimensions());
        // Equivalent to:
        // self.values.borrow_mut().copy_from_slice(source.values.borrow());
        self.par_rows_mut()
            .zip(source.par_row_slices())
            .for_each(|(dst, src)| {
                dst.copy_from_slice(src);
            });
    }

    /// Flatten an extension field matrix to a base field matrix.
    pub fn flatten_to_base<F: Field>(self) -> RowMajorMatrix<F>
    where
        T: ExtensionField<F>,
    {
        let width = self.width * T::DIMENSION;
        let values = T::flatten_to_base(self.values.to_vec());
        RowMajorMatrix::new(values, width)
    }

    /// Get an iterator over the rows of the matrix.
    pub fn row_slices(&self) -> impl Iterator<Item = &[T]> {
        self.values.borrow().chunks_exact(self.width)
    }

    /// Get a parallel iterator over the rows of the matrix.
    pub fn par_row_slices(&self) -> impl IndexedParallelIterator<Item = &[T]>
    where
        T: Sync,
    {
        self.values.borrow().par_chunks_exact(self.width)
    }

    /// Returns a slice of the given row.
    ///
    /// # Panics
    /// Panics if `r` larger than self.height().
    pub fn row_mut(&mut self, r: usize) -> &mut [T]
    where
        S: BorrowMut<[T]>,
    {
        &mut self.values.borrow_mut()[r * self.width..(r + 1) * self.width]
    }

    /// Get a mutable iterator over the rows of the matrix.
    pub fn rows_mut(&mut self) -> impl Iterator<Item = &mut [T]>
    where
        S: BorrowMut<[T]>,
    {
        self.values.borrow_mut().chunks_exact_mut(self.width)
    }

    /// Get a mutable parallel iterator over the rows of the matrix.
    pub fn par_rows_mut<'a>(&'a mut self) -> impl IndexedParallelIterator<Item = &'a mut [T]>
    where
        T: 'a + Send,
        S: BorrowMut<[T]>,
    {
        self.values.borrow_mut().par_chunks_exact_mut(self.width)
    }

    /// Get a mutable iterator over the rows of the matrix which packs the rows into packed values.
    ///
    /// If `P::WIDTH` does not divide `self.width`, the remainder of the row will be returned as a
    /// base slice.
    pub fn horizontally_packed_row_mut<P>(&mut self, r: usize) -> (&mut [P], &mut [T])
    where
        P: PackedValue<Value = T>,
        S: BorrowMut<[T]>,
    {
        P::pack_slice_with_suffix_mut(self.row_mut(r))
    }

    /// Scale the given row by the given value.
    ///
    /// # Panics
    /// Panics if `r` larger than `self.height()`.
    pub fn scale_row(&mut self, r: usize, scale: T)
    where
        T: Field,
        S: BorrowMut<[T]>,
    {
        scale_slice_in_place_single_core(self.row_mut(r), scale);
    }

    /// Scale the given row by the given value.
    ///
    /// # Performance
    /// This function is parallelized, which may introduce some overhead compared to
    /// [`Self::scale_row`] when the width is small.
    ///
    /// # Panics
    /// Panics if `r` larger than `self.height()`.
    pub fn par_scale_row(&mut self, r: usize, scale: T)
    where
        T: Field,
        S: BorrowMut<[T]>,
    {
        par_scale_slice_in_place(self.row_mut(r), scale);
    }

    /// Scale the entire matrix by the given value.
    pub fn scale(&mut self, scale: T)
    where
        T: Field,
        S: BorrowMut<[T]>,
    {
        par_scale_slice_in_place(self.values.borrow_mut(), scale);
    }

    /// Split the matrix into two matrix views, one with the first `r` rows and one with the remaining rows.
    ///
    /// # Panics
    /// Panics if `r` larger than `self.height()`.
    pub fn split_rows(&self, r: usize) -> (RowMajorMatrixView<'_, T>, RowMajorMatrixView<'_, T>) {
        let (lo, hi) = self.values.borrow().split_at(r * self.width);
        (
            DenseMatrix::new(lo, self.width),
            DenseMatrix::new(hi, self.width),
        )
    }

    /// Split the matrix into two mutable matrix views, one with the first `r` rows and one with the remaining rows.
    ///
    /// # Panics
    /// Panics if `r` larger than `self.height()`.
    pub fn split_rows_mut(
        &mut self,
        r: usize,
    ) -> (RowMajorMatrixViewMut<'_, T>, RowMajorMatrixViewMut<'_, T>)
    where
        S: BorrowMut<[T]>,
    {
        let (lo, hi) = self.values.borrow_mut().split_at_mut(r * self.width);
        (
            DenseMatrix::new(lo, self.width),
            DenseMatrix::new(hi, self.width),
        )
    }

    /// Get an iterator over the rows of the matrix which takes `chunk_rows` rows at a time.
    ///
    /// If `chunk_rows` does not divide the height of the matrix, the last chunk will be smaller.
    pub fn par_row_chunks(
        &self,
        chunk_rows: usize,
    ) -> impl IndexedParallelIterator<Item = RowMajorMatrixView<'_, T>>
    where
        T: Send,
    {
        self.values
            .borrow()
            .par_chunks(self.width * chunk_rows)
            .map(|slice| RowMajorMatrixView::new(slice, self.width))
    }

    /// Get a parallel iterator over the rows of the matrix which takes `chunk_rows` rows at a time.
    ///
    /// If `chunk_rows` does not divide the height of the matrix, the last chunk will be smaller.
    pub fn par_row_chunks_exact(
        &self,
        chunk_rows: usize,
    ) -> impl IndexedParallelIterator<Item = RowMajorMatrixView<'_, T>>
    where
        T: Send,
    {
        self.values
            .borrow()
            .par_chunks_exact(self.width * chunk_rows)
            .map(|slice| RowMajorMatrixView::new(slice, self.width))
    }

    /// Get a mutable iterator over the rows of the matrix which takes `chunk_rows` rows at a time.
    ///
    /// If `chunk_rows` does not divide the height of the matrix, the last chunk will be smaller.
    pub fn par_row_chunks_mut(
        &mut self,
        chunk_rows: usize,
    ) -> impl IndexedParallelIterator<Item = RowMajorMatrixViewMut<'_, T>>
    where
        T: Send,
        S: BorrowMut<[T]>,
    {
        self.values
            .borrow_mut()
            .par_chunks_mut(self.width * chunk_rows)
            .map(|slice| RowMajorMatrixViewMut::new(slice, self.width))
    }

    /// Get a mutable iterator over the rows of the matrix which takes `chunk_rows` rows at a time.
    ///
    /// If `chunk_rows` does not divide the height of the matrix, the last up to `chunk_rows - 1` rows
    /// of the matrix will be omitted.
    pub fn row_chunks_exact_mut(
        &mut self,
        chunk_rows: usize,
    ) -> impl Iterator<Item = RowMajorMatrixViewMut<'_, T>>
    where
        T: Send,
        S: BorrowMut<[T]>,
    {
        self.values
            .borrow_mut()
            .chunks_exact_mut(self.width * chunk_rows)
            .map(|slice| RowMajorMatrixViewMut::new(slice, self.width))
    }

    /// Get a parallel mutable iterator over the rows of the matrix which takes `chunk_rows` rows at a time.
    ///
    /// If `chunk_rows` does not divide the height of the matrix, the last up to `chunk_rows - 1` rows
    /// of the matrix will be omitted.
    pub fn par_row_chunks_exact_mut(
        &mut self,
        chunk_rows: usize,
    ) -> impl IndexedParallelIterator<Item = RowMajorMatrixViewMut<'_, T>>
    where
        T: Send,
        S: BorrowMut<[T]>,
    {
        self.values
            .borrow_mut()
            .par_chunks_exact_mut(self.width * chunk_rows)
            .map(|slice| RowMajorMatrixViewMut::new(slice, self.width))
    }

    /// Get a pair of mutable slices of the given rows.
    ///
    /// # Panics
    /// Panics if `row_1` or `row_2` are out of bounds or if `row_1 >= row_2`.
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

    /// Get a pair of mutable slices of the given rows, both packed into packed field elements.
    ///
    /// If `P:WIDTH` does not divide `self.width`, the remainder of the row will be returned as a base slice.
    ///
    /// # Panics
    /// Panics if `row_1` or `row_2` are out of bounds or if `row_1 >= row_2`.
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

    /// Append zeros to the "end" of the given matrix, except that the matrix is in bit-reversed order,
    /// so in actuality we're interleaving zero rows.
    #[instrument(level = "debug", skip_all)]
    pub fn bit_reversed_zero_pad(self, added_bits: usize) -> RowMajorMatrix<T>
    where
        T: Field,
    {
        if added_bits == 0 {
            return self.to_row_major_matrix();
        }

        // This is equivalent to:
        //     reverse_matrix_index_bits(mat);
        //     mat
        //         .values
        //         .resize(mat.values.len() << added_bits, F::ZERO);
        //     reverse_matrix_index_bits(mat);
        // But rather than implement it with bit reversals, we directly construct the resulting matrix,
        // whose rows are zero except for rows whose low `added_bits` bits are zero.

        let w = self.width;
        let mut padded =
            RowMajorMatrix::new(T::zero_vec(self.values.borrow().len() << added_bits), w);
        padded
            .par_row_chunks_exact_mut(1 << added_bits)
            .zip(self.par_row_slices())
            .for_each(|(mut ch, r)| ch.row_mut(0).copy_from_slice(r));

        padded
    }
}

impl<T: Clone + Send + Sync, S: DenseStorage<T>> Matrix<T> for DenseMatrix<T, S> {
    #[inline]
    fn width(&self) -> usize {
        self.width
    }

    #[inline]
    fn height(&self) -> usize {
        if self.width == 0 {
            0
        } else {
            self.values.borrow().len() / self.width
        }
    }

    #[inline]
    unsafe fn get_unchecked(&self, r: usize, c: usize) -> T {
        unsafe {
            // Safety: The caller must ensure that r < self.height() and c < self.width().
            self.values
                .borrow()
                .get_unchecked(r * self.width + c)
                .clone()
        }
    }

    #[inline]
    unsafe fn row_subseq_unchecked(
        &self,
        r: usize,
        start: usize,
        end: usize,
    ) -> impl IntoIterator<Item = T, IntoIter = impl Iterator<Item = T> + Send + Sync> {
        unsafe {
            // Safety: The caller must ensure that r < self.height() and start <= end <= self.width().
            self.values
                .borrow()
                .get_unchecked(r * self.width + start..r * self.width + end)
                .iter()
                .cloned()
        }
    }

    #[inline]
    unsafe fn row_subslice_unchecked(
        &self,
        r: usize,
        start: usize,
        end: usize,
    ) -> impl Deref<Target = [T]> {
        unsafe {
            // Safety: The caller must ensure that r < self.height()
            self.values
                .borrow()
                .get_unchecked(r * self.width + start..r * self.width + end)
        }
    }

    fn to_row_major_matrix(self) -> RowMajorMatrix<T>
    where
        Self: Sized,
        T: Clone,
    {
        RowMajorMatrix::new(self.values.to_vec(), self.width)
    }

    #[inline]
    fn horizontally_packed_row<'a, P>(
        &'a self,
        r: usize,
    ) -> (
        impl Iterator<Item = P> + Send + Sync,
        impl Iterator<Item = T> + Send + Sync,
    )
    where
        P: PackedValue<Value = T>,
        T: Clone + 'a,
    {
        let buf = &self.values.borrow()[r * self.width..(r + 1) * self.width];
        let (packed, sfx) = P::pack_slice_with_suffix(buf);
        (packed.iter().copied(), sfx.iter().cloned())
    }

    #[inline]
    fn padded_horizontally_packed_row<'a, P>(
        &'a self,
        r: usize,
    ) -> impl Iterator<Item = P> + Send + Sync
    where
        P: PackedValue<Value = T>,
        T: Clone + Default + 'a,
    {
        let buf = &self.values.borrow()[r * self.width..(r + 1) * self.width];
        let (packed, sfx) = P::pack_slice_with_suffix(buf);
        packed.iter().copied().chain(iter::once(P::from_fn(|i| {
            sfx.get(i).cloned().unwrap_or_default()
        })))
    }
}

impl<T: Clone + Default + Send + Sync> DenseMatrix<T> {
    pub fn as_cow<'a>(self) -> RowMajorMatrixCow<'a, T> {
        RowMajorMatrixCow::new(Cow::Owned(self.values), self.width)
    }

    pub fn rand<R: Rng>(rng: &mut R, rows: usize, cols: usize) -> Self
    where
        StandardUniform: Distribution<T>,
    {
        let values = rng.sample_iter(StandardUniform).take(rows * cols).collect();
        Self::new(values, cols)
    }

    pub fn rand_nonzero<R: Rng>(rng: &mut R, rows: usize, cols: usize) -> Self
    where
        T: Field,
        StandardUniform: Distribution<T>,
    {
        let values = rng
            .sample_iter(StandardUniform)
            .filter(|x| !x.is_zero())
            .take(rows * cols)
            .collect();
        Self::new(values, cols)
    }

    pub fn pad_to_height(&mut self, new_height: usize, fill: T) {
        assert!(new_height >= self.height());
        self.values.resize(self.width * new_height, fill);
    }
}

impl<T: Copy + Default + Send + Sync, V: DenseStorage<T>> DenseMatrix<T, V> {
    /// Return the transpose of this matrix.
    pub fn transpose(&self) -> RowMajorMatrix<T> {
        let nelts = self.height() * self.width();
        let mut values = vec![T::default(); nelts];
        transpose::transpose(
            self.values.borrow(),
            &mut values,
            self.width(),
            self.height(),
        );
        RowMajorMatrix::new(values, self.height())
    }

    /// Transpose the matrix returning the result in `other` without intermediate allocation.
    pub fn transpose_into<W: DenseStorage<T> + BorrowMut<[T]>>(
        &self,
        other: &mut DenseMatrix<T, W>,
    ) {
        assert_eq!(self.height(), other.width());
        assert_eq!(other.height(), self.width());
        transpose::transpose(
            self.values.borrow(),
            other.values.borrow_mut(),
            self.width(),
            self.height(),
        );
    }
}

impl<'a, T: Clone + Default + Send + Sync> RowMajorMatrixView<'a, T> {
    pub fn as_cow(self) -> RowMajorMatrixCow<'a, T> {
        RowMajorMatrixCow::new(Cow::Borrowed(self.values), self.width)
    }
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_field::FieldArray;

    use super::*;

    #[test]
    fn test_new() {
        let matrix = RowMajorMatrix::new(vec![1, 2, 3, 4, 5, 6], 2);
        assert_eq!(matrix.width, 2);
        assert_eq!(matrix.height(), 3);
        assert_eq!(matrix.values, vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_new_row() {
        let matrix = RowMajorMatrix::new_row(vec![1, 2, 3]);
        assert_eq!(matrix.width, 3);
        assert_eq!(matrix.height(), 1);
    }

    #[test]
    fn test_new_col() {
        let matrix = RowMajorMatrix::new_col(vec![1, 2, 3]);
        assert_eq!(matrix.width, 1);
        assert_eq!(matrix.height(), 3);
    }

    #[test]
    fn test_height_with_zero_width() {
        let matrix: DenseMatrix<i32> = RowMajorMatrix::new(vec![], 0);
        assert_eq!(matrix.height(), 0);
    }

    #[test]
    fn test_get_methods() {
        let matrix = RowMajorMatrix::new(vec![1, 2, 3, 4, 5, 6], 2); // Height = 3, Width = 2
        assert_eq!(matrix.get(0, 0), Some(1));
        assert_eq!(matrix.get(1, 1), Some(4));
        assert_eq!(matrix.get(2, 0), Some(5));
        unsafe {
            assert_eq!(matrix.get_unchecked(0, 1), 2);
            assert_eq!(matrix.get_unchecked(1, 0), 3);
            assert_eq!(matrix.get_unchecked(2, 1), 6);
        }
        assert_eq!(matrix.get(3, 0), None); // Height out of bounds
        assert_eq!(matrix.get(0, 2), None); // Width out of bounds
    }

    #[test]
    fn test_row_methods() {
        let matrix = RowMajorMatrix::new(vec![1, 2, 3, 4, 5, 6, 7, 8], 4); // Height = 2, Width = 4
        let row: Vec<_> = matrix.row(1).unwrap().into_iter().collect();
        assert_eq!(row, vec![5, 6, 7, 8]);
        unsafe {
            let row: Vec<_> = matrix.row_unchecked(0).into_iter().collect();
            assert_eq!(row, vec![1, 2, 3, 4]);
            let row: Vec<_> = matrix.row_subseq_unchecked(0, 0, 3).into_iter().collect();
            assert_eq!(row, vec![1, 2, 3]);
            let row: Vec<_> = matrix.row_subseq_unchecked(0, 1, 3).into_iter().collect();
            assert_eq!(row, vec![2, 3]);
            let row: Vec<_> = matrix.row_subseq_unchecked(0, 2, 4).into_iter().collect();
            assert_eq!(row, vec![3, 4]);
        }
        assert!(matrix.row(2).is_none()); // Height out of bounds
    }

    #[test]
    fn test_row_slice_methods() {
        let matrix = RowMajorMatrix::new(vec![1, 2, 3, 4, 5, 6, 7, 8, 9], 3); // Height = 3, Width = 3
        let slice0 = matrix.row_slice(0);
        let slice2 = matrix.row_slice(2);
        assert_eq!(slice0.unwrap().deref(), &[1, 2, 3]);
        assert_eq!(slice2.unwrap().deref(), &[7, 8, 9]);
        unsafe {
            assert_eq!(&[1, 2, 3], matrix.row_slice_unchecked(0).deref());
            assert_eq!(&[7, 8, 9], matrix.row_slice_unchecked(2).deref());

            assert_eq!(&[1, 2, 3], matrix.row_subslice_unchecked(0, 0, 3).deref());
            assert_eq!(&[8], matrix.row_subslice_unchecked(2, 1, 2).deref());
        }
        assert!(matrix.row_slice(3).is_none()); // Height out of bounds
    }

    #[test]
    fn test_as_view() {
        let matrix = RowMajorMatrix::new(vec![1, 2, 3, 4], 2);
        let view = matrix.as_view();
        assert_eq!(view.values, &[1, 2, 3, 4]);
        assert_eq!(view.width, 2);
    }

    #[test]
    fn test_as_view_mut() {
        let mut matrix = RowMajorMatrix::new(vec![1, 2, 3, 4], 2);
        let view = matrix.as_view_mut();
        view.values[0] = 10;
        assert_eq!(matrix.values, vec![10, 2, 3, 4]);
    }

    #[test]
    fn test_copy_from() {
        let mut matrix1 = RowMajorMatrix::new(vec![0, 0, 0, 0], 2);
        let matrix2 = RowMajorMatrix::new(vec![1, 2, 3, 4], 2);
        matrix1.copy_from(&matrix2);
        assert_eq!(matrix1.values, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_split_rows() {
        let matrix = RowMajorMatrix::new(vec![1, 2, 3, 4, 5, 6], 2);
        let (top, bottom) = matrix.split_rows(1);
        assert_eq!(top.values, vec![1, 2]);
        assert_eq!(bottom.values, vec![3, 4, 5, 6]);
    }

    #[test]
    fn test_split_rows_mut() {
        let mut matrix = RowMajorMatrix::new(vec![1, 2, 3, 4, 5, 6], 2);
        let (top, bottom) = matrix.split_rows_mut(1);
        assert_eq!(top.values, vec![1, 2]);
        assert_eq!(bottom.values, vec![3, 4, 5, 6]);
    }

    #[test]
    fn test_row_mut() {
        let mut matrix = RowMajorMatrix::new(vec![1, 2, 3, 4, 5, 6], 2);
        matrix.row_mut(1)[0] = 10;
        assert_eq!(matrix.values, vec![1, 2, 10, 4, 5, 6]);
    }

    #[test]
    fn test_bit_reversed_zero_pad() {
        let matrix = RowMajorMatrix::new(
            vec![
                BabyBear::new(1),
                BabyBear::new(2),
                BabyBear::new(3),
                BabyBear::new(4),
            ],
            2,
        );
        let padded = matrix.bit_reversed_zero_pad(1);
        assert_eq!(padded.width, 2);
        assert_eq!(
            padded.values,
            vec![
                BabyBear::new(1),
                BabyBear::new(2),
                BabyBear::new(0),
                BabyBear::new(0),
                BabyBear::new(3),
                BabyBear::new(4),
                BabyBear::new(0),
                BabyBear::new(0)
            ]
        );
    }

    #[test]
    fn test_bit_reversed_zero_pad_no_change() {
        let matrix = RowMajorMatrix::new(
            vec![
                BabyBear::new(1),
                BabyBear::new(2),
                BabyBear::new(3),
                BabyBear::new(4),
            ],
            2,
        );
        let padded = matrix.bit_reversed_zero_pad(0);

        assert_eq!(padded.width, 2);
        assert_eq!(
            padded.values,
            vec![
                BabyBear::new(1),
                BabyBear::new(2),
                BabyBear::new(3),
                BabyBear::new(4),
            ]
        );
    }

    #[test]
    fn test_scale() {
        let mut matrix = RowMajorMatrix::new(
            vec![
                BabyBear::new(1),
                BabyBear::new(2),
                BabyBear::new(3),
                BabyBear::new(4),
                BabyBear::new(5),
                BabyBear::new(6),
            ],
            2,
        );
        matrix.scale(BabyBear::new(2));
        assert_eq!(
            matrix.values,
            vec![
                BabyBear::new(2),
                BabyBear::new(4),
                BabyBear::new(6),
                BabyBear::new(8),
                BabyBear::new(10),
                BabyBear::new(12)
            ]
        );
    }

    #[test]
    fn test_scale_row() {
        let mut matrix = RowMajorMatrix::new(
            vec![
                BabyBear::new(1),
                BabyBear::new(2),
                BabyBear::new(3),
                BabyBear::new(4),
                BabyBear::new(5),
                BabyBear::new(6),
            ],
            2,
        );
        matrix.scale_row(1, BabyBear::new(3));
        assert_eq!(
            matrix.values,
            vec![
                BabyBear::new(1),
                BabyBear::new(2),
                BabyBear::new(9),
                BabyBear::new(12),
                BabyBear::new(5),
                BabyBear::new(6),
            ]
        );
    }

    #[test]
    fn test_to_row_major_matrix() {
        let matrix = RowMajorMatrix::new(vec![1, 2, 3, 4, 5, 6], 2);
        let converted = matrix.to_row_major_matrix();

        // The converted matrix should have the same values and width
        assert_eq!(converted.width, 2);
        assert_eq!(converted.height(), 3);
        assert_eq!(converted.values, vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_horizontally_packed_row() {
        type Packed = FieldArray<BabyBear, 2>;

        let matrix = RowMajorMatrix::new(
            vec![
                BabyBear::new(1),
                BabyBear::new(2),
                BabyBear::new(3),
                BabyBear::new(4),
                BabyBear::new(5),
                BabyBear::new(6),
            ],
            3,
        );

        let (packed_iter, suffix_iter) = matrix.horizontally_packed_row::<Packed>(1);

        let packed: Vec<_> = packed_iter.collect();
        let suffix: Vec<_> = suffix_iter.collect();

        assert_eq!(
            packed,
            vec![Packed::from([BabyBear::new(4), BabyBear::new(5)])]
        );
        assert_eq!(suffix, vec![BabyBear::new(6)]);
    }

    #[test]
    fn test_padded_horizontally_packed_row() {
        use p3_baby_bear::BabyBear;

        type Packed = FieldArray<BabyBear, 2>;

        let matrix = RowMajorMatrix::new(
            vec![
                BabyBear::new(1),
                BabyBear::new(2),
                BabyBear::new(3),
                BabyBear::new(4),
                BabyBear::new(5),
                BabyBear::new(6),
            ],
            3,
        );

        let packed_iter = matrix.padded_horizontally_packed_row::<Packed>(1);
        let packed: Vec<_> = packed_iter.collect();

        assert_eq!(
            packed,
            vec![
                Packed::from([BabyBear::new(4), BabyBear::new(5)]),
                Packed::from([BabyBear::new(6), BabyBear::new(0)])
            ]
        );
    }

    #[test]
    fn test_pad_to_height() {
        let mut matrix = RowMajorMatrix::new(vec![1, 2, 3, 4, 5, 6], 3);

        // Original matrix:
        // [ 1  2  3 ]
        // [ 4  5  6 ] (height = 2)

        matrix.pad_to_height(4, 9);

        // Expected matrix after padding:
        // [ 1  2  3 ]
        // [ 4  5  6 ]
        // [ 9  9  9 ]  <-- Newly added row
        // [ 9  9  9 ]  <-- Newly added row

        assert_eq!(matrix.height(), 4);
        assert_eq!(matrix.values, vec![1, 2, 3, 4, 5, 6, 9, 9, 9, 9, 9, 9]);
    }

    #[test]
    fn test_transpose_into() {
        let matrix = RowMajorMatrix::new(vec![1, 2, 3, 4, 5, 6], 3);

        // Original matrix:
        // [ 1  2  3 ]
        // [ 4  5  6 ]

        let mut transposed = RowMajorMatrix::new(vec![0; 6], 2);

        matrix.transpose_into(&mut transposed);

        // Expected transposed matrix:
        // [ 1  4 ]
        // [ 2  5 ]
        // [ 3  6 ]

        assert_eq!(transposed.width, 2);
        assert_eq!(transposed.height(), 3);
        assert_eq!(transposed.values, vec![1, 4, 2, 5, 3, 6]);
    }

    #[test]
    fn test_flatten_to_base() {
        let matrix = RowMajorMatrix::new(
            vec![
                BabyBear::new(2),
                BabyBear::new(3),
                BabyBear::new(4),
                BabyBear::new(5),
            ],
            2,
        );

        let flattened: RowMajorMatrix<BabyBear> = matrix.flatten_to_base();

        assert_eq!(flattened.width, 2);
        assert_eq!(
            flattened.values,
            vec![
                BabyBear::new(2),
                BabyBear::new(3),
                BabyBear::new(4),
                BabyBear::new(5),
            ]
        );
    }

    #[test]
    fn test_horizontally_packed_row_mut() {
        type Packed = FieldArray<BabyBear, 2>;

        let mut matrix = RowMajorMatrix::new(
            vec![
                BabyBear::new(1),
                BabyBear::new(2),
                BabyBear::new(3),
                BabyBear::new(4),
                BabyBear::new(5),
                BabyBear::new(6),
            ],
            3,
        );

        let (packed, suffix) = matrix.horizontally_packed_row_mut::<Packed>(1);
        packed[0] = Packed::from([BabyBear::new(9), BabyBear::new(10)]);
        suffix[0] = BabyBear::new(11);

        assert_eq!(
            matrix.values,
            vec![
                BabyBear::new(1),
                BabyBear::new(2),
                BabyBear::new(3),
                BabyBear::new(9),
                BabyBear::new(10),
                BabyBear::new(11),
            ]
        );
    }

    #[test]
    fn test_par_row_chunks() {
        let matrix = RowMajorMatrix::new(vec![1, 2, 3, 4, 5, 6, 7, 8], 2);

        let chunks: Vec<_> = matrix.par_row_chunks(2).collect();

        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].values, vec![1, 2, 3, 4]);
        assert_eq!(chunks[1].values, vec![5, 6, 7, 8]);
    }

    #[test]
    fn test_par_row_chunks_exact() {
        let matrix = RowMajorMatrix::new(vec![1, 2, 3, 4, 5, 6], 2);

        let chunks: Vec<_> = matrix.par_row_chunks_exact(1).collect();

        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].values, vec![1, 2]);
        assert_eq!(chunks[1].values, vec![3, 4]);
        assert_eq!(chunks[2].values, vec![5, 6]);
    }

    #[test]
    fn test_par_row_chunks_mut() {
        let mut matrix = RowMajorMatrix::new(vec![1, 2, 3, 4, 5, 6, 7, 8], 2);

        matrix
            .par_row_chunks_mut(2)
            .for_each(|chunk| chunk.values.iter_mut().for_each(|x| *x += 10));

        assert_eq!(matrix.values, vec![11, 12, 13, 14, 15, 16, 17, 18]);
    }

    #[test]
    fn test_row_chunks_exact_mut() {
        let mut matrix = RowMajorMatrix::new(vec![1, 2, 3, 4, 5, 6], 2);

        for chunk in matrix.row_chunks_exact_mut(1) {
            chunk.values.iter_mut().for_each(|x| *x *= 2);
        }

        assert_eq!(matrix.values, vec![2, 4, 6, 8, 10, 12]);
    }

    #[test]
    fn test_par_row_chunks_exact_mut() {
        let mut matrix = RowMajorMatrix::new(vec![1, 2, 3, 4, 5, 6], 2);

        matrix
            .par_row_chunks_exact_mut(1)
            .for_each(|chunk| chunk.values.iter_mut().for_each(|x| *x += 5));

        assert_eq!(matrix.values, vec![6, 7, 8, 9, 10, 11]);
    }

    #[test]
    fn test_row_pair_mut() {
        let mut matrix = RowMajorMatrix::new(vec![1, 2, 3, 4, 5, 6], 2);

        let (row1, row2) = matrix.row_pair_mut(0, 2);
        row1[0] = 9;
        row2[1] = 10;

        assert_eq!(matrix.values, vec![9, 2, 3, 4, 5, 10]);
    }

    #[test]
    fn test_packed_row_pair_mut() {
        type Packed = FieldArray<BabyBear, 2>;

        let mut matrix = RowMajorMatrix::new(
            vec![
                BabyBear::new(1),
                BabyBear::new(2),
                BabyBear::new(3),
                BabyBear::new(4),
                BabyBear::new(5),
                BabyBear::new(6),
            ],
            3,
        );

        let ((packed1, sfx1), (packed2, sfx2)) = matrix.packed_row_pair_mut::<Packed>(0, 1);
        packed1[0] = Packed::from([BabyBear::new(7), BabyBear::new(8)]);
        packed2[0] = Packed::from([BabyBear::new(33), BabyBear::new(44)]);
        sfx1[0] = BabyBear::new(99);
        sfx2[0] = BabyBear::new(9);

        assert_eq!(
            matrix.values,
            vec![
                BabyBear::new(7),
                BabyBear::new(8),
                BabyBear::new(99),
                BabyBear::new(33),
                BabyBear::new(44),
                BabyBear::new(9),
            ]
        );
    }

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
        let transposed = matrix.transpose();

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
        let transposed = matrix.transpose();

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
    fn test_vertically_packed_row_pair() {
        type Packed = FieldArray<BabyBear, 2>;

        let matrix = RowMajorMatrix::new((1..17).map(BabyBear::new).collect::<Vec<_>>(), 4);

        // Calling the function with r = 0 and step = 2
        let packed = matrix.vertically_packed_row_pair::<Packed>(0, 2);

        // Matrix visualization:
        //
        // [  1   2   3   4  ]  <-- Row 0
        // [  5   6   7   8  ]  <-- Row 1
        // [  9  10  11  12  ]  <-- Row 2
        // [ 13  14  15  16  ]  <-- Row 3
        //
        // Packing rows 0-1 together, then rows 2-3 together:
        //
        // Packed result:
        // [
        //   (1, 5), (2, 6), (3, 7), (4, 8),   // First packed row (Row 0 & Row 1)
        //   (9, 13), (10, 14), (11, 15), (12, 16),   // Second packed row (Row 2 & Row 3)
        // ]

        assert_eq!(
            packed,
            (1..5)
                .chain(9..13)
                .map(|i| [BabyBear::new(i), BabyBear::new(i + 4)].into())
                .collect::<Vec<_>>(),
        );
    }

    #[test]
    fn test_vertically_packed_row_pair_overlap() {
        type Packed = FieldArray<BabyBear, 2>;

        let matrix = RowMajorMatrix::new((1..17).map(BabyBear::new).collect::<Vec<_>>(), 4);

        // Original matrix visualization:
        //
        // [  1   2   3   4  ]  <-- Row 0
        // [  5   6   7   8  ]  <-- Row 1
        // [  9  10  11  12  ]  <-- Row 2
        // [ 13  14  15  16  ]  <-- Row 3
        //
        // Packing rows 0-1 together, then rows 1-2 together:
        //
        // Expected packed result:
        // [
        //   (1, 5), (2, 6), (3, 7), (4, 8),   // First packed row (Row 0 & Row 1)
        //   (5, 9), (6, 10), (7, 11), (8, 12) // Second packed row (Row 1 & Row 2)
        // ]

        // Calling the function with overlapping rows (r = 0 and step = 1)
        let packed = matrix.vertically_packed_row_pair::<Packed>(0, 1);

        assert_eq!(
            packed,
            (1..5)
                .chain(5..9)
                .map(|i| [BabyBear::new(i), BabyBear::new(i + 4)].into())
                .collect::<Vec<_>>(),
        );
    }

    #[test]
    fn test_vertically_packed_row_pair_wraparound_start_1() {
        use p3_baby_bear::BabyBear;
        use p3_field::FieldArray;

        type Packed = FieldArray<BabyBear, 2>;

        let matrix = RowMajorMatrix::new((1..17).map(BabyBear::new).collect::<Vec<_>>(), 4);

        // Original matrix visualization:
        //
        // [  1   2   3   4  ]  <-- Row 0
        // [  5   6   7   8  ]  <-- Row 1
        // [  9  10  11  12  ]  <-- Row 2
        // [ 13  14  15  16  ]  <-- Row 3
        //
        // Packing starts from row 1, skipping 2 rows (step = 2):
        // - The first packed row should contain row 1 & row 2.
        // - The second packed row should contain row 3 & row 1 (wraparound case).
        //
        // Expected packed result:
        // [
        //   (5, 9), (6, 10), (7, 11), (8, 12),   // Packed row (Row 1 & Row 2)
        //   (13, 1), (14, 2), (15, 3), (16, 4)    // Packed row (Row 3 & Row 1)
        // ]

        // Calling the function with wraparound scenario (starting at r = 1 with step = 2)
        let packed = matrix.vertically_packed_row_pair::<Packed>(1, 2);

        assert_eq!(
            packed,
            vec![
                Packed::from([BabyBear::new(5), BabyBear::new(9)]),
                Packed::from([BabyBear::new(6), BabyBear::new(10)]),
                Packed::from([BabyBear::new(7), BabyBear::new(11)]),
                Packed::from([BabyBear::new(8), BabyBear::new(12)]),
                Packed::from([BabyBear::new(13), BabyBear::new(1)]),
                Packed::from([BabyBear::new(14), BabyBear::new(2)]),
                Packed::from([BabyBear::new(15), BabyBear::new(3)]),
                Packed::from([BabyBear::new(16), BabyBear::new(4)]),
            ]
        );
    }
}
