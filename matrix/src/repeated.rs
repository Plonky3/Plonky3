use alloc::vec;
use alloc::vec::Vec;
use core::ops::Deref;

use p3_field::PackedValue;
use p3_maybe_rayon::prelude::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use p3_util::{log2_strict_usize, reverse_slice_index_bits};

use crate::interleaved::VerticallyInterleaved;
use crate::{Dimensions, Matrix};

/// A sequence of matrices, each of the same dimensions, stacked together vertically.
#[derive(Clone)]
pub struct VerticallyRepeated<Inner> {
    inner: Vec<Inner>,
    width: usize,
    log_num_mats: usize,
    log_inner_height: usize,
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
        let log_num_mats = log2_strict_usize(inner.len());
        let log_inner_height = log2_strict_usize(inner_dims.height);
        let outer_height = inner.len() << log_inner_height;
        Self {
            inner,
            width,
            log_num_mats,
            log_inner_height,
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
    #[inline(always)]
    fn width(&self) -> usize {
        self.width
    }

    #[inline(always)]
    fn height(&self) -> usize {
        self.outer_height
    }

    #[inline(always)]
    fn dimensions(&self) -> Dimensions {
        Dimensions {
            width: self.width,
            height: self.outer_height,
        }
    }

    #[inline(always)]
    fn get(&self, r: usize, c: usize) -> T {
        let mat_idx = r >> self.log_inner_height;
        let r = r % (1 << self.log_inner_height);
        self.inner[mat_idx].get(r, c)
    }

    type Row<'a>
        = Inner::Row<'a>
    where
        Self: 'a;

    #[inline(always)]
    fn row(&self, r: usize) -> Self::Row<'_> {
        let mat_idx = r >> self.log_inner_height;
        let r = r % (1 << self.log_inner_height);
        self.inner[mat_idx].row(r)
    }

    #[inline(always)]
    fn row_slice(&self, r: usize) -> impl Deref<Target = [T]> {
        let mat_idx = r >> self.log_inner_height;
        let r = r % (1 << self.log_inner_height);
        self.inner[mat_idx].row_slice(r)
    }

    fn truncate_rows_power_of_two(&self, log_rows: usize) -> impl Matrix<T>
    where
        T: Clone,
        Self: Sized,
    {
        let inner: Vec<_> = if log_rows >= self.log_inner_height {
            self.inner[..1 << (log_rows - self.log_inner_height)]
                .iter()
                // This shouldn't actually truncate, we just need the same type in the two branches.
                .map(|mat| mat.truncate_rows_power_of_two(self.log_inner_height))
                .collect()
        } else {
            vec![self.inner[0].truncate_rows_power_of_two(log_rows)]
        };
        VerticallyRepeated::new(inner)
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

    fn par_vertically_packed_pairs_wrapping<P>(
        &self,
        distance: usize,
    ) -> impl ParallelIterator<Item = (usize, [impl Iterator<Item = P>; 2])>
    where
        P: PackedValue<Value = T>,
    {
        self.inner.par_iter().flat_map(move |mat| mat.par_vertically_packed_pairs_wrapping(distance))
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
