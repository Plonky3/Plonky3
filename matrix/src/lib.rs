//! Matrix library.

#![no_std]

extern crate alloc;

use alloc::vec::Vec;
use core::fmt::{Debug, Display, Formatter};
use core::ops::Deref;

use itertools::{Itertools, izip};
use p3_field::{
    BasedVectorSpace, ExtensionField, Field, PackedValue, PrimeCharacteristicRing, dot_product,
};
use p3_maybe_rayon::prelude::*;
use strided::{VerticallyStridedMatrixView, VerticallyStridedRowIndexMap};
use tracing::instrument;
use p3_util::log2_strict_usize;

use crate::dense::RowMajorMatrix;

pub mod bitrev;
pub mod dense;
pub mod extension;
pub mod horizontally_truncated;
pub mod row_index_mapped;
pub mod stack;
pub mod strided;
pub mod util;

/// A simple struct representing the shape of a matrix.
///
/// The `Dimensions` type stores the number of columns (`width`) and rows (`height`)
/// of a matrix. It is commonly used for querying and displaying matrix shapes.
#[derive(Copy, Clone, PartialEq, Eq)]
pub struct Dimensions {
    /// Number of columns in the matrix.
    pub width: usize,
    /// Number of rows in the matrix.
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

/// A generic trait for two-dimensional matrix-like data structures.
///
/// The `Matrix` trait provides a uniform interface for accessing rows, elements,
/// and computing with matrices in both sequential and parallel contexts. It supports
/// packing strategies for SIMD optimizations and interaction with extension fields.
pub trait Matrix<T: Send + Sync + Clone>: Send + Sync {
    /// Returns the number of columns in the matrix.
    fn width(&self) -> usize;

    /// Returns the number of rows in the matrix.
    fn height(&self) -> usize;

    /// Returns the dimensions (width, height) of the matrix.
    fn dimensions(&self) -> Dimensions {
        Dimensions {
            width: self.width(),
            height: self.height(),
        }
    }

    // The methods:
    // get, get_unchecked, row, row_unchecked, row_subseq_unchecked, row_slice, row_slice_unchecked, row_subslice_unchecked
    // are all defined in a circular manner so you only need to implement a subset of them.
    // In particular is is enough to implement just one of: row_unchecked, row_subseq_unchecked
    //
    // That being said, most implementations will want to implement several methods for performance reasons.

    /// Returns the element at the given row and column.
    ///
    /// Returns `None` if either `r >= height()` or `c >= width()`.
    #[inline]
    fn get(&self, r: usize, c: usize) -> Option<T> {
        (r < self.height() && c < self.width()).then(|| unsafe {
            // Safety: Clearly `r < self.height()` and `c < self.width()`.
            self.get_unchecked(r, c)
        })
    }

    /// Returns the element at the given row and column.
    ///
    /// For a safe alternative, see [`get`].
    ///
    /// # Safety
    /// The caller must ensure that `r < self.height()` and `c < self.width()`.
    /// Breaking any of these assumptions is considered undefined behaviour.
    #[inline]
    unsafe fn get_unchecked(&self, r: usize, c: usize) -> T {
        unsafe { self.row_slice_unchecked(r)[c].clone() }
    }

    /// Returns an iterator over the elements of the `r`-th row.
    ///
    /// The iterator will have `self.width()` elements.
    ///
    /// Returns `None` if `r >= height()`.
    #[inline]
    fn row(
        &self,
        r: usize,
    ) -> Option<impl IntoIterator<Item = T, IntoIter = impl Iterator<Item = T> + Send + Sync>> {
        (r < self.height()).then(|| unsafe {
            // Safety: Clearly `r < self.height()`.
            self.row_unchecked(r)
        })
    }

    /// Returns an iterator over the elements of the `r`-th row.
    ///
    /// The iterator will have `self.width()` elements.
    ///
    /// For a safe alternative, see [`row`].
    ///
    /// # Safety
    /// The caller must ensure that `r < self.height()`.
    /// Breaking this assumption is considered undefined behaviour.
    #[inline]
    unsafe fn row_unchecked(
        &self,
        r: usize,
    ) -> impl IntoIterator<Item = T, IntoIter = impl Iterator<Item = T> + Send + Sync> {
        unsafe { self.row_subseq_unchecked(r, 0, self.width()) }
    }

    /// Returns an iterator over the elements of the `r`-th row from position `start` to `end`.
    ///
    /// When `start = 0` and `end = width()`, this is equivalent to [`row_unchecked`].
    ///
    /// For a safe alternative, use [`row`], along with the `skip` and `take` iterator methods.
    ///
    /// # Safety
    /// The caller must ensure that `r < self.height()` and `start <= end <= self.width()`.
    /// Breaking any of these assumptions is considered undefined behaviour.
    #[inline]
    unsafe fn row_subseq_unchecked(
        &self,
        r: usize,
        start: usize,
        end: usize,
    ) -> impl IntoIterator<Item = T, IntoIter = impl Iterator<Item = T> + Send + Sync> {
        unsafe {
            self.row_unchecked(r)
                .into_iter()
                .skip(start)
                .take(end - start)
        }
    }

    /// Returns the elements of the `r`-th row as something which can be coerced to a slice.
    ///
    /// Returns `None` if `r >= height()`.
    #[inline]
    fn row_slice(&self, r: usize) -> Option<impl Deref<Target = [T]>> {
        (r < self.height()).then(|| unsafe {
            // Safety: Clearly `r < self.height()`.
            self.row_slice_unchecked(r)
        })
    }

    /// Returns the elements of the `r`-th row as something which can be coerced to a slice.
    ///
    /// For a safe alternative, see [`row_slice`].
    ///
    /// # Safety
    /// The caller must ensure that `r < self.height()`.
    /// Breaking this assumption is considered undefined behaviour.
    #[inline]
    unsafe fn row_slice_unchecked(&self, r: usize) -> impl Deref<Target = [T]> {
        unsafe { self.row_subslice_unchecked(r, 0, self.width()) }
    }

    /// Returns a subset of elements of the `r`-th row as something which can be coerced to a slice.
    ///
    /// When `start = 0` and `end = width()`, this is equivalent to [`row_slice_unchecked`].
    ///
    /// For a safe alternative, see [`row_slice`].
    ///
    /// # Safety
    /// The caller must ensure that `r < self.height()` and `start <= end <= self.width()`.
    /// Breaking any of these assumptions is considered undefined behaviour.
    #[inline]
    unsafe fn row_subslice_unchecked(
        &self,
        r: usize,
        start: usize,
        end: usize,
    ) -> impl Deref<Target = [T]> {
        unsafe {
            self.row_subseq_unchecked(r, start, end)
                .into_iter()
                .collect_vec()
        }
    }

    /// Returns an iterator over all rows in the matrix.
    #[inline]
    fn rows(&self) -> impl Iterator<Item = impl Iterator<Item = T>> + Send + Sync {
        unsafe {
            // Safety: `r` always satisfies `r < self.height()`.
            (0..self.height()).map(move |r| self.row_unchecked(r).into_iter())
        }
    }

    /// Returns a parallel iterator over all rows in the matrix.
    #[inline]
    fn par_rows(
        &self,
    ) -> impl IndexedParallelIterator<Item = impl Iterator<Item = T>> + Send + Sync {
        unsafe {
            // Safety: `r` always satisfies `r < self.height()`.
            (0..self.height())
                .into_par_iter()
                .map(move |r| self.row_unchecked(r).into_iter())
        }
    }

    /// Collect the elements of the rows `r` through `r + c`. If anything is larger than `self.height()`
    /// simply wrap around to the beginning of the matrix.
    fn wrapping_row_slices(&self, r: usize, c: usize) -> Vec<impl Deref<Target = [T]>> {
        unsafe {
            // Safety: Thank to the `%`, the rows index is always less than `self.height()`.
            (0..c)
                .map(|i| self.row_slice_unchecked((r + i) % self.height()))
                .collect_vec()
        }
    }

    /// Returns an iterator over the first row of the matrix.
    ///
    /// Returns None if `height() == 0`.
    #[inline]
    fn first_row(
        &self,
    ) -> Option<impl IntoIterator<Item = T, IntoIter = impl Iterator<Item = T> + Send + Sync>> {
        self.row(0)
    }

    /// Returns an iterator over the last row of the matrix.
    ///
    /// Returns None if `height() == 0`.
    #[inline]
    fn last_row(
        &self,
    ) -> Option<impl IntoIterator<Item = T, IntoIter = impl Iterator<Item = T> + Send + Sync>> {
        if self.height() == 0 {
            None
        } else {
            // Safety: Clearly `self.height() - 1 < self.height()`.
            unsafe { Some(self.row_unchecked(self.height() - 1)) }
        }
    }

    /// Converts the matrix into a `RowMajorMatrix` by collecting all rows into a single vector.
    fn to_row_major_matrix(self) -> RowMajorMatrix<T>
    where
        Self: Sized,
        T: Clone,
    {
        RowMajorMatrix::new(self.rows().flatten().collect(), self.width())
    }

    /// Get a packed iterator over the `r`-th row.
    ///
    /// If the row length is not divisible by the packing width, the final elements
    /// are returned as a base iterator with length `<= P::WIDTH - 1`.
    ///
    /// # Panics
    /// Panics if `r >= height()`.
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
        assert!(r < self.height(), "Row index out of bounds.");
        let num_packed = self.width() / P::WIDTH;
        unsafe {
            // Safety: We have already checked that `r < height()`.
            let mut iter = self
                .row_subseq_unchecked(r, 0, num_packed * P::WIDTH)
                .into_iter();

            // array::from_fn is guaranteed to always call in order.
            let packed =
                (0..num_packed).map(move |_| P::from_fn(|_| iter.next().unwrap_unchecked()));

            let sfx = self
                .row_subseq_unchecked(r, num_packed * P::WIDTH, self.width())
                .into_iter();
            (packed, sfx)
        }
    }

    /// Get a packed iterator over the `r`-th row.
    ///
    /// If the row length is not divisible by the packing width, the final entry will be zero-padded.
    ///
    /// # Panics
    /// Panics if `r >= height()`.
    fn padded_horizontally_packed_row<'a, P>(
        &'a self,
        r: usize,
    ) -> impl Iterator<Item = P> + Send + Sync
    where
        P: PackedValue<Value = T>,
        T: Clone + Default + 'a,
    {
        let mut row_iter = self.row(r).expect("Row index out of bounds.").into_iter();
        let num_elems = self.width().div_ceil(P::WIDTH);
        // array::from_fn is guaranteed to always call in order.
        (0..num_elems).map(move |_| P::from_fn(|_| row_iter.next().unwrap_or_default()))
    }

    /// Get a parallel iterator over all packed rows of the matrix.
    ///
    /// If the matrix width is not divisible by the packing width, the final elements
    /// of each row are returned as a base iterator with length `<= P::WIDTH - 1`.
    fn par_horizontally_packed_rows<'a, P>(
        &'a self,
    ) -> impl IndexedParallelIterator<
        Item = (
            impl Iterator<Item = P> + Send + Sync,
            impl Iterator<Item = T> + Send + Sync,
        ),
    >
    where
        P: PackedValue<Value = T>,
        T: Clone + 'a,
    {
        (0..self.height())
            .into_par_iter()
            .map(|r| self.horizontally_packed_row(r))
    }

    /// Get a parallel iterator over all packed rows of the matrix.
    ///
    /// If the matrix width is not divisible by the packing width, the final entry of each row will be zero-padded.
    fn par_padded_horizontally_packed_rows<'a, P>(
        &'a self,
    ) -> impl IndexedParallelIterator<Item = impl Iterator<Item = P> + Send + Sync>
    where
        P: PackedValue<Value = T>,
        T: Clone + Default + 'a,
    {
        (0..self.height())
            .into_par_iter()
            .map(|r| self.padded_horizontally_packed_row(r))
    }

    /// Pack together a collection of adjacent rows from the matrix.
    ///
    /// Returns an iterator whose i'th element is packing of the i'th element of the
    /// rows r through r + P::WIDTH - 1. If we exceed the height of the matrix,
    /// wrap around and include initial rows.
    #[inline]
    fn vertically_packed_row<P>(&self, r: usize) -> impl Iterator<Item = P>
    where
        T: Copy,
        P: PackedValue<Value = T>,
    {
        // Precompute row slices once to minimize redundant calls and improve performance.
        let rows = self.wrapping_row_slices(r, P::WIDTH);

        // Using precomputed rows avoids repeatedly calling `row_slice`, which is costly.
        (0..self.width()).map(move |c| P::from_fn(|i| rows[i][c]))
    }

    /// Pack together a collection of rows and "next" rows from the matrix.
    ///
    /// Returns a vector corresponding to 2 packed rows. The i'th element of the first
    /// row contains the packing of the i'th element of the rows r through r + P::WIDTH - 1.
    /// The i'th element of the second row contains the packing of the i'th element of the
    /// rows r + step through r + step + P::WIDTH - 1. If at some point we exceed the
    /// height of the matrix, wrap around and include initial rows.
    #[inline]
    fn vertically_packed_row_pair<P>(&self, r: usize, step: usize) -> Vec<P>
    where
        T: Copy,
        P: PackedValue<Value = T>,
    {
        // Whilst it would appear that this can be replaced by two calls to vertically_packed_row
        // tests seem to indicate that combining them in the same function is slightly faster.
        // It's probably allowing the compiler to make some optimizations on the fly.

        let rows = self.wrapping_row_slices(r, P::WIDTH);
        let next_rows = self.wrapping_row_slices(r + step, P::WIDTH);

        (0..self.width())
            .map(|c| P::from_fn(|i| rows[i][c]))
            .chain((0..self.width()).map(|c| P::from_fn(|i| next_rows[i][c])))
            .collect_vec()
    }

    /// Returns a view over a vertically strided submatrix.
    ///
    /// The view selects rows using `r = offset + i * stride` for each `i`.
    fn vertically_strided(self, stride: usize, offset: usize) -> VerticallyStridedMatrixView<Self>
    where
        Self: Sized,
    {
        VerticallyStridedRowIndexMap::new_view(self, stride, offset)
    }

    /// Compute Mᵀv, aka premultiply this matrix by the given vector,
    /// aka scale each row by the corresponding entry in `v` and take the sum across rows.
    /// `v` can be a vector of extension elements.
    #[instrument(level = "debug", skip_all, fields(dims = %self.dimensions()))]
    fn columnwise_dot_product<EF>(&self, v: &[EF]) -> Vec<EF>
    where
        T: Field,
        EF: ExtensionField<T>,
    {
        let packed_width = self.width().div_ceil(T::Packing::WIDTH);

        let packed_result = self
            .par_padded_horizontally_packed_rows::<T::Packing>()
            .zip(v)
            .par_fold_reduce(
                || EF::ExtensionPacking::zero_vec(packed_width),
                |mut acc, (row, &scale)| {
                    let scale = EF::ExtensionPacking::from_basis_coefficients_fn(|i| {
                        T::Packing::from(scale.as_basis_coefficients_slice()[i])
                    });
                    izip!(&mut acc, row).for_each(|(l, r)| *l += scale * r);
                    acc
                },
                |mut acc_l, acc_r| {
                    izip!(&mut acc_l, acc_r).for_each(|(l, r)| *l += r);
                    acc_l
                },
            );

        packed_result
            .into_iter()
            .flat_map(|p| {
                (0..T::Packing::WIDTH).map(move |i| {
                    EF::from_basis_coefficients_fn(|j| {
                        p.as_basis_coefficients_slice()[j].as_slice()[i]
                    })
                })
            })
            .take(self.width())
            .collect()
    }

    /// Compute the matrix vector product `M . vec`, aka take the dot product of each
    /// row of `M` by `vec`. If the length of `vec` is longer than the width of `M`,
    /// `vec` is truncated to the first `width()` elements.
    ///
    /// We make use of `PackedFieldExtension` to speed up computations. Thus `vec` is passed in as
    /// a slice of `PackedFieldExtension` elements.
    ///
    /// # Panics
    /// This function panics if the length of `vec` is less than `self.width().div_ceil(T::Packing::WIDTH)`.
    fn rowwise_packed_dot_product<EF>(
        &self,
        vec: &[EF::ExtensionPacking],
    ) -> impl IndexedParallelIterator<Item = EF>
    where
        T: Field,
        EF: ExtensionField<T>,
    {
        // The length of a `padded_horizontally_packed_row` is `self.width().div_ceil(T::Packing::WIDTH)`.
        assert!(vec.len() >= self.width().div_ceil(T::Packing::WIDTH));

        // TODO: This is a base - extension dot product and so it should
        // be possible to speed this up using ideas in `packed_linear_combination`.
        // TODO: Perhaps we should be packing rows vertically not horizontally.
        self.par_padded_horizontally_packed_rows::<T::Packing>()
            .map(move |row_packed| {
                let packed_sum_of_packed: EF::ExtensionPacking =
                    dot_product(vec.iter().copied(), row_packed);
                let sum_of_packed: EF = EF::from_basis_coefficients_fn(|i| {
                    packed_sum_of_packed.as_basis_coefficients_slice()[i]
                        .as_slice()
                        .iter()
                        .copied()
                        .sum()
                });
                sum_of_packed
            })
    }

    /// Returns the base-2 logarithm of the matrix height.
    ///
    /// # Panics
    /// Panics if the height is not a power of two.
    #[inline]
    fn log_height(&self) -> usize {
        log2_strict_usize(self.height())
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;
    use alloc::{format, vec};

    use itertools::izip;
    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use super::*;

    #[test]
    fn test_columnwise_dot_product() {
        type F = BabyBear;
        type EF = BinomialExtensionField<BabyBear, 4>;

        let mut rng = SmallRng::seed_from_u64(1);
        let m = RowMajorMatrix::<F>::rand(&mut rng, 1 << 8, 1 << 4);
        let v = RowMajorMatrix::<EF>::rand(&mut rng, 1 << 8, 1).values;

        let mut expected = vec![EF::ZERO; m.width()];
        for (row, &scale) in izip!(m.rows(), &v) {
            for (l, r) in izip!(&mut expected, row) {
                *l += scale * r;
            }
        }

        assert_eq!(m.columnwise_dot_product(&v), expected);
    }

    // Mock implementation for testing purposes
    struct MockMatrix {
        data: Vec<Vec<u32>>,
        width: usize,
        height: usize,
    }

    impl Matrix<u32> for MockMatrix {
        fn width(&self) -> usize {
            self.width
        }

        fn height(&self) -> usize {
            self.height
        }

        unsafe fn row_unchecked(
            &self,
            r: usize,
        ) -> impl IntoIterator<Item = u32, IntoIter = impl Iterator<Item = u32> + Send + Sync>
        {
            // Just a mock implementation so we just do the easy safe thing.
            self.data[r].clone()
        }
    }

    #[test]
    fn test_dimensions() {
        let dims = Dimensions {
            width: 3,
            height: 5,
        };
        assert_eq!(dims.width, 3);
        assert_eq!(dims.height, 5);
        assert_eq!(format!("{:?}", dims), "3x5");
        assert_eq!(format!("{}", dims), "3x5");
    }

    #[test]
    fn test_mock_matrix_dimensions() {
        let matrix = MockMatrix {
            data: vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]],
            width: 3,
            height: 3,
        };
        assert_eq!(matrix.width(), 3);
        assert_eq!(matrix.height(), 3);
        assert_eq!(
            matrix.dimensions(),
            Dimensions {
                width: 3,
                height: 3
            }
        );
    }

    #[test]
    fn test_first_row() {
        let matrix = MockMatrix {
            data: vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]],
            width: 3,
            height: 3,
        };
        let mut first_row = matrix.first_row().unwrap().into_iter();
        assert_eq!(first_row.next(), Some(1));
        assert_eq!(first_row.next(), Some(2));
        assert_eq!(first_row.next(), Some(3));
    }

    #[test]
    fn test_last_row() {
        let matrix = MockMatrix {
            data: vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]],
            width: 3,
            height: 3,
        };
        let mut last_row = matrix.last_row().unwrap().into_iter();
        assert_eq!(last_row.next(), Some(7));
        assert_eq!(last_row.next(), Some(8));
        assert_eq!(last_row.next(), Some(9));
    }

    #[test]
    fn test_first_last_row_empty_matrix() {
        let matrix = MockMatrix {
            data: vec![],
            width: 3,
            height: 0,
        };
        let first_row = matrix.first_row();
        let last_row = matrix.last_row();
        assert!(first_row.is_none());
        assert!(last_row.is_none());
    }

    #[test]
    fn test_to_row_major_matrix() {
        let matrix = MockMatrix {
            data: vec![vec![1, 2], vec![3, 4]],
            width: 2,
            height: 2,
        };
        let row_major = matrix.to_row_major_matrix();
        assert_eq!(row_major.values, vec![1, 2, 3, 4]);
        assert_eq!(row_major.width, 2);
    }

    #[test]
    fn test_matrix_get_methods() {
        let matrix = MockMatrix {
            data: vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]],
            width: 3,
            height: 3,
        };
        assert_eq!(matrix.get(0, 0), Some(1));
        assert_eq!(matrix.get(1, 2), Some(6));
        assert_eq!(matrix.get(2, 1), Some(8));

        unsafe {
            assert_eq!(matrix.get_unchecked(0, 1), 2);
            assert_eq!(matrix.get_unchecked(1, 0), 4);
            assert_eq!(matrix.get_unchecked(2, 2), 9);
        }

        assert_eq!(matrix.get(3, 0), None); // Height out of bounds
        assert_eq!(matrix.get(0, 3), None); // Width out of bounds
    }

    #[test]
    fn test_matrix_row_methods_iteration() {
        let matrix = MockMatrix {
            data: vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]],
            width: 3,
            height: 3,
        };

        let mut row_iter = matrix.row(1).unwrap().into_iter();
        assert_eq!(row_iter.next(), Some(4));
        assert_eq!(row_iter.next(), Some(5));
        assert_eq!(row_iter.next(), Some(6));
        assert_eq!(row_iter.next(), None);

        unsafe {
            let mut row_iter_unchecked = matrix.row_unchecked(2).into_iter();
            assert_eq!(row_iter_unchecked.next(), Some(7));
            assert_eq!(row_iter_unchecked.next(), Some(8));
            assert_eq!(row_iter_unchecked.next(), Some(9));
            assert_eq!(row_iter_unchecked.next(), None);

            let mut row_iter_subset = matrix.row_subseq_unchecked(0, 1, 3).into_iter();
            assert_eq!(row_iter_subset.next(), Some(2));
            assert_eq!(row_iter_subset.next(), Some(3));
            assert_eq!(row_iter_subset.next(), None);
        }

        assert!(matrix.row(3).is_none()); // Height out of bounds
    }

    #[test]
    fn test_row_slice_methods() {
        let matrix = MockMatrix {
            data: vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]],
            width: 3,
            height: 3,
        };
        let row_slice = matrix.row_slice(1).unwrap();
        assert_eq!(*row_slice, [4, 5, 6]);
        unsafe {
            let row_slice_unchecked = matrix.row_slice_unchecked(2);
            assert_eq!(*row_slice_unchecked, [7, 8, 9]);

            let row_subslice = matrix.row_subslice_unchecked(0, 1, 2);
            assert_eq!(*row_subslice, [2]);
        }

        assert!(matrix.row_slice(3).is_none()); // Height out of bounds
    }

    #[test]
    fn test_matrix_rows() {
        let matrix = MockMatrix {
            data: vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]],
            width: 3,
            height: 3,
        };

        let all_rows: Vec<Vec<u32>> = matrix.rows().map(|row| row.collect()).collect();
        assert_eq!(all_rows, vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]]);
    }

    #[test]
    fn test_log_height() {
        let mat = MockMatrix {
            data: vec![vec![1, 2], vec![3, 4], vec![5, 6], vec![7, 8]],
            width: 2,
            height: 4,
        };
        assert_eq!(mat.log_height(), 2); // log2(4) == 2

        let mat2 = MockMatrix {
            data: vec![vec![1]],
            width: 1,
            height: 1,
        };
        assert_eq!(mat2.log_height(), 0); // log2(1) == 0
    }
}
