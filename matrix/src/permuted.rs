use core::ops::Deref;

use p3_field::PackedValue;
use p3_maybe_rayon::prelude::*;

use crate::dense::RowMajorMatrix;
use crate::Matrix;

/// A RowPermutation remaps row indices, and can change the height.
pub trait RowPermutation: Send + Sync {
    fn height(&self) -> usize;
    fn permute_row_index(&self, r: usize) -> usize;

    /// Permutations can optionally provide an optimized method to
    /// convert to dense form.
    fn to_row_major_matrix<T: Clone + Send + Sync, Inner: Matrix<T>>(
        &self,
        inner: Inner,
    ) -> RowMajorMatrix<T> {
        RowMajorMatrix::new(
            (0..inner.height())
                .flat_map(|r| inner.row(self.permute_row_index(r)))
                .collect(),
            inner.width(),
        )
    }
}

#[derive(Debug)]
pub struct PermutedMatrix<Perm, Inner> {
    pub perm: Perm,
    pub inner: Inner,
}

impl<T: Send + Sync, Perm: RowPermutation, Inner: Matrix<T>> Matrix<T>
    for PermutedMatrix<Perm, Inner>
{
    fn width(&self) -> usize {
        self.inner.width()
    }
    fn height(&self) -> usize {
        self.perm.height()
    }

    type Row<'a> = Inner::Row<'a>
    where
        Self: 'a;

    fn row(&self, r: usize) -> Self::Row<'_> {
        self.inner.row(self.perm.permute_row_index(r))
    }

    // Override these methods so we use the potentially optimized inner methods instead of defaults.

    fn row_slice(&self, r: usize) -> impl Deref<Target = [T]> {
        self.inner.row_slice(self.perm.permute_row_index(r))
    }

    fn horizontally_packed_row<'a, P>(
        &'a self,
        r: usize,
    ) -> (impl Iterator<Item = P>, impl Iterator<Item = T>)
    where
        P: PackedValue<Value = T>,
        T: Clone + 'a,
    {
        self.inner
            .horizontally_packed_row(self.perm.permute_row_index(r))
    }

    fn to_row_major_matrix(self) -> RowMajorMatrix<T>
    where
        Self: Sized,
        T: Clone,
    {
        // Use Perm's optimized permutation routine, if it has one.
        self.perm.to_row_major_matrix(self.inner)
    }
}
