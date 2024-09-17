use core::iter::Cloned;
use core::slice;

use p3_matrix::{Matrix, MatrixRowSlices, MatrixRows};

#[derive(Copy, Clone, Debug)]
pub struct TwoRowMatrixView<'a, T> {
    pub local: &'a [T],
    pub next: &'a [T],
}

impl<'a, T> TwoRowMatrixView<'a, T> {
    pub fn new(local: &'a [T], next: &'a [T]) -> Self {
        Self { local, next }
    }
}

impl<'a, T> Matrix<T> for TwoRowMatrixView<'a, T> {
    fn width(&self) -> usize {
        self.local.len()
    }

    fn height(&self) -> usize {
        2
    }
}

impl<T: Clone + core::fmt::Debug> MatrixRows<T> for TwoRowMatrixView<'_, T> {
    type Row<'a> = Cloned<slice::Iter<'a, T>> where Self: 'a, T: 'a;

    fn row(&self, r: usize) -> Self::Row<'_> {
        let slice = match r {
            0 => self.local,
            1 => self.next,
            _ => panic!("Only two rows available"),
        };
        slice.iter().cloned()
    }
}

impl<T: Clone + core::fmt::Debug> MatrixRowSlices<T> for TwoRowMatrixView<'_, T> {
    fn row_slice(&self, r: usize) -> &[T] {
        match r {
            0 => self.local,
            1 => self.next,
            _ => panic!("Only two rows available"),
        }
    }
}
