use core::marker::PhantomData;

use alloc::vec;
use alloc::{boxed::Box, vec::Vec};
use p3_maybe_rayon::{MaybeIntoParIter, ParallelIterator};
use p3_util::{log2_strict_usize, reverse_bits, reverse_bits_len};

use crate::dense::RowMajorMatrix;
use crate::{Matrix, MatrixRowSlices, MatrixRowSlicesMut, MatrixRows};

#[derive(Debug, PartialEq, Eq)]
pub enum RowPermutation {
    BitReversed,
    // Opaque(Box<dyn Fn(/*height*/ usize, /*index*/ usize) -> usize>),
}

impl RowPermutation {
    fn apply(&self, height: usize, index: usize) -> usize {
        match self {
            RowPermutation::BitReversed => reverse_bits(index, height),
            // RowPermutation::Opaque(f) => f(height, index),
        }
    }
}

#[derive(Debug)]
pub struct MatrixView<T, Inner> {
    inner: Inner,
    perms: Vec<RowPermutation>,
    _phantom: PhantomData<T>,
}

impl<T: PartialEq, Inner: MatrixRows<T>> PartialEq for MatrixView<T, Inner> {
    fn eq(&self, other: &Self) -> bool {
        (0..self.height()).all(|r| {
            self.row(r)
                .into_iter()
                .zip(other.row(r))
                .all(|(x, y)| x == y)
        })
    }
}
impl<T: Eq, Inner: MatrixRows<T>> Eq for MatrixView<T, Inner> {}

impl<T, Inner> MatrixView<T, Inner> {
    pub fn identity(inner: Inner) -> Self {
        Self {
            inner,
            perms: vec![],
            _phantom: PhantomData,
        }
    }
    pub fn new(inner: Inner, perm: RowPermutation) -> Self {
        Self {
            inner,
            perms: vec![perm],
            _phantom: PhantomData,
        }
    }

    fn permute_row_index(&self, index: usize) -> usize
    where
        Inner: Matrix<T>,
    {
        if self.perms.is_empty() {
            index
        } else {
            // Apply permutations right-to-left.
            self.perms
                .iter()
                .rfold(index, |index, perm| perm.apply(self.inner.height(), index))
        }
    }
}

impl<T, Inner: Matrix<T>> Matrix<T> for MatrixView<T, Inner> {
    fn width(&self) -> usize {
        self.inner.width()
    }
    fn height(&self) -> usize {
        self.inner.height()
    }
}

impl<T, Inner: MatrixRows<T>> MatrixRows<T> for MatrixView<T, Inner> {
    type Row<'a> = Inner::Row<'a> where Self: 'a;

    fn row(&self, r: usize) -> Self::Row<'_> {
        self.inner.row(self.permute_row_index(r))
    }

    fn to_row_major_matrix(self) -> RowMajorMatrix<T>
    where
        Self: Sized,
        T: Clone,
    {
        if self.perms.is_empty() {
            // Fast path for realizing the identity transform
            self.inner.to_row_major_matrix()
        } else if &self.perms == &[RowPermutation::BitReversed] {
            // Fast path for a single bit-reversal
            let mut mat = self.inner.to_row_major_matrix();
            reverse_matrix_index_bits(&mut mat);
            mat
        } else {
            RowMajorMatrix::new(
                (0..self.height()).flat_map(|r| self.row(r)).collect(),
                self.width(),
            )
        }
    }

    type Permuted = Self;
    fn permute_rows(mut self, perm: RowPermutation) -> Self::Permuted
    where
        Self: Sized,
    {
        use RowPermutation::*;
        match (perm, self.perms.last()) {
            (BitReversed, Some(BitReversed)) => {
                self.perms.pop();
            }
            (perm, _) => self.perms.push(perm),
        }
        self
    }
}

impl<T, Inner: MatrixRowSlices<T>> MatrixRowSlices<T> for MatrixView<T, Inner> {
    type PermutedSlices = Self;
    fn row_slice(&self, r: usize) -> &[T] {
        self.inner.row_slice(self.permute_row_index(r))
    }
}

impl<T, Inner: MatrixRowSlicesMut<T>> MatrixRowSlicesMut<T> for MatrixView<T, Inner> {
    type PermutedSlicesMut = Self;
    fn row_slice_mut(&mut self, r: usize) -> &mut [T] {
        self.inner.row_slice_mut(self.permute_row_index(r))
    }
}

fn reverse_matrix_index_bits<F>(mat: &mut RowMajorMatrix<F>) {
    let w = mat.width();
    let h = mat.height();
    let log_h = log2_strict_usize(h);
    let values = mat.values.as_mut_ptr() as usize;

    (0..h).into_par_iter().for_each(|i| {
        let values = values as *mut F;
        let j = reverse_bits_len(i, log_h);
        if i < j {
            unsafe { swap_rows_raw(values, w, i, j) };
        }
    });
}

unsafe fn swap_rows_raw<F>(mat: *mut F, w: usize, i: usize, j: usize) {
    let row_i = core::slice::from_raw_parts_mut(mat.add(i * w), w);
    let row_j = core::slice::from_raw_parts_mut(mat.add(j * w), w);
    row_i.swap_with_slice(row_j);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dense::RowMajorMatrix;

    #[test]
    fn test_bit_reversal() {
        let x = RowMajorMatrix::<usize>::new((0..8).collect(), 2);

        let x_br = x.permute_rows(RowPermutation::BitReversed);
        assert_eq!(x_br.row_slice(0), &[0, 1]);
        assert_eq!(x_br.row_slice(1), &[4, 5]);
        assert_eq!(x_br.row_slice(2), &[2, 3]);
        assert_eq!(x_br.row_slice(3), &[6, 7]);

        let x_br_br = x_br.permute_rows(RowPermutation::BitReversed);
        assert_eq!(x_br_br.row_slice(0), &[0, 1]);
        assert_eq!(x_br_br.row_slice(1), &[2, 3]);
        assert_eq!(x_br_br.row_slice(2), &[4, 5]);
        assert_eq!(x_br_br.row_slice(3), &[6, 7]);

        assert!(x_br_br.perms.is_empty(), "bit-reversing twice is a no-op");
    }
}
