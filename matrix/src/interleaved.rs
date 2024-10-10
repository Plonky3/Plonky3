use alloc::vec;
use alloc::vec::Vec;
use core::ops::Deref;

use p3_util::reverse_slice_index_bits;

use crate::{Dimensions, Matrix};
use crate::repeated::VerticallyRepeated;

/// A sequence of matrices, each of the same dimensions, stacked together vertically.
#[derive(Clone)]
pub struct VerticallyInterleaved<Inner> {
    inner: Vec<Inner>,
    width: usize,
    inner_height: usize,
    outer_height: usize,
}

impl<Inner> VerticallyInterleaved<Inner> {
    pub fn new<T>(inner: Vec<Inner>) -> Self
    where
        T: Send + Sync,
        Inner: Matrix<T>,
    {
        let inner_dims = inner.first().expect("No inner matrices?").dimensions();
        for mat in &inner {
            assert_eq!(
                inner_dims,
                mat.dimensions(),
                "Matrices have nonuniform dimensions"
            );
        }
        let width = inner_dims.width;
        let inner_height = inner_dims.height;
        let outer_height = inner.len() * inner_height;
        Self {
            inner,
            width,
            inner_height,
            outer_height,
        }
    }

    pub fn single<T>(inner: Inner) -> Self
    where
        T: Send + Sync,
        Inner: Matrix<T>,
    {
        Self::new(vec![inner])
    }
}

impl<T: Clone + Send + Sync, Inner: Matrix<T>> Matrix<T> for VerticallyInterleaved<Inner> {
    fn width(&self) -> usize {
        self.width
    }

    fn height(&self) -> usize {
        self.outer_height
    }

    fn dimensions(&self) -> Dimensions {
        Dimensions {
            width: self.width,
            height: self.outer_height,
        }
    }

    #[inline]
    fn get(&self, r: usize, c: usize) -> T {
        self.row_slice(r).deref()[c].clone()
    }

    type Row<'a>
        = Inner::Row<'a>
    where
        Self: 'a;

    #[inline]
    fn row(&self, r: usize) -> Self::Row<'_> {
        let mat_idx = r % self.inner.len();
        let r = r / self.inner.len();
        self.inner[mat_idx].row(r)
    }

    #[inline]
    fn row_slice(&self, r: usize) -> impl Deref<Target = [T]> {
        let mat_idx = r % self.inner.len();
        let r = r / self.inner.len();
        self.inner[mat_idx].row_slice(r)
    }

    // #[inline]
    // fn truncate_rows_power_of_two(&self, log_rows: usize) -> impl Matrix<T>
    // where
    //     T: Clone,
    // {
    //     let log_rows_per_mat = log_rows - log2_strict_usize(self.inner.len());
    //     let inner = self
    //         .inner
    //         .iter()
    //         .map(|mat| mat.truncate_rows_power_of_two(log_rows_per_mat))
    //         .collect();
    //     VerticallyInterleaved::new(inner)
    // }

    type BitRev = VerticallyRepeated<Inner::BitRev>;
    fn bit_reverse_rows(self) -> Self::BitRev {
        let mut mats: Vec<Inner::BitRev> = self
            .inner
            .into_iter()
            .map(Matrix::bit_reverse_rows)
            .collect();
        reverse_slice_index_bits(&mut mats);
        VerticallyRepeated::new(mats)
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use crate::dense::DenseMatrix;
    use crate::Matrix;

    use super::*;

    #[test]
    fn to_row_major_matrix() {
        let mats: Vec<DenseMatrix<u32>> = vec![
            DenseMatrix::new_col(vec![1, 2, 3]),
            DenseMatrix::new_col(vec![4, 5, 6]),
            DenseMatrix::new_col(vec![7, 8, 9]),
            DenseMatrix::new_col(vec![10, 11, 12]),
        ];
        let interleaved = VerticallyInterleaved::new(mats);
        let row_major = interleaved.to_row_major_matrix();
        assert_eq!(
            row_major.values,
            vec![1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12]
        );
    }
}
