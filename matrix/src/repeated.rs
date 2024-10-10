use alloc::vec;
use alloc::vec::Vec;
use core::ops::Deref;

use p3_util::reverse_slice_index_bits;

use crate::interleaved::VerticallyInterleaved;
use crate::{Dimensions, Matrix};

/// A sequence of matrices, each of the same dimensions, stacked together vertically.
#[derive(Clone)]
pub struct VerticallyRepeated<Inner> {
    inner: Vec<Inner>,
    width: usize,
    inner_height: usize,
    outer_height: usize,
}

impl<Inner> VerticallyRepeated<Inner> {
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

impl<T: Clone + Send + Sync, Inner: Matrix<T>> Matrix<T> for VerticallyRepeated<Inner> {
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
        let mat_idx = r / self.inner_height;
        let r = r % self.inner_height;
        self.inner[mat_idx].row(r)
    }

    #[inline]
    fn row_slice(&self, r: usize) -> impl Deref<Target = [T]> {
        let mat_idx = r / self.inner_height;
        let r = r % self.inner_height;
        self.inner[mat_idx].row_slice(r)
    }

    type BitRev = VerticallyInterleaved<Inner::BitRev>;
    fn bit_reverse_rows(self) -> Self::BitRev {
        let mut mats: Vec<Inner::BitRev> = self
            .inner
            .into_iter()
            .map(Matrix::bit_reverse_rows)
            .collect();
        reverse_slice_index_bits(&mut mats);
        VerticallyInterleaved::new(mats)
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use crate::dense::DenseMatrix;
    use crate::repeated::VerticallyRepeated;
    use crate::util::reverse_matrix_index_bits;
    use crate::Matrix;

    #[test]
    fn test_to_row_major_matrix() {
        let mut rng = rand::thread_rng();
        let combined: DenseMatrix<u32> = DenseMatrix::rand(&mut rng, 256, 3);
        let matrices: Vec<DenseMatrix<u32>> = combined
            .par_row_chunks_exact(64)
            .map(|mat| mat.to_row_major_matrix())
            .collect();
        let repeated = VerticallyRepeated::new(matrices.clone());
        assert_eq!(combined, repeated.to_row_major_matrix())
    }

    #[test]
    fn test_bit_reverse_rows() {
        let mut rng = rand::thread_rng();
        let combined: DenseMatrix<u32> = DenseMatrix::rand(&mut rng, 256, 3);
        let matrices: Vec<DenseMatrix<u32>> = combined
            .par_row_chunks_exact(64)
            .map(|mat| mat.to_row_major_matrix())
            .collect();
        let repeated = VerticallyRepeated::new(matrices.clone());
        let interleaved = repeated.clone().bit_reverse_rows();

        let mut repeated = repeated.to_row_major_matrix();
        let interleaved = interleaved.to_row_major_matrix();
        reverse_matrix_index_bits(&mut repeated);
        assert_eq!(repeated, interleaved);
    }
}
