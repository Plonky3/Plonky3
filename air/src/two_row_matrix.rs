use core::{iter, slice};

use p3_matrix::Matrix;

#[derive(Copy, Clone, Debug)]
pub struct TwoRowMatrixView<'a, T> {
    pub local: &'a [T],
    pub next: &'a [T],
}

impl<'a, T> TwoRowMatrixView<'a, T> {
    pub const fn new(local: &'a [T], next: &'a [T]) -> Self {
        Self { local, next }
    }
}

impl<'a, T: Clone + Send + Sync> Matrix<T> for TwoRowMatrixView<'a, T> {
    fn width(&self) -> usize {
        self.local.len()
    }

    fn height(&self) -> usize {
        2
    }

    fn get(&self, r: usize, c: usize) -> T {
        match r {
            0 => self.local[c].clone(),
            1 => self.next[c].clone(),
            _ => panic!("Only two rows available"),
        }
    }

    type Row<'b> = iter::Cloned<slice::Iter<'b, T>> where Self: 'b;

    fn row(&self, r: usize) -> Self::Row<'_> {
        let slice = match r {
            0 => self.local,
            1 => self.next,
            _ => panic!("Only two rows available"),
        };
        slice.iter().cloned()
    }
}
