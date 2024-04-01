use alloc::vec::Vec;
use core::iter;
use core::ops::Range;

use rand::distributions::{Distribution, Standard};
use rand::Rng;

use crate::Matrix;

/// A sparse matrix stored in the compressed sparse row format.
#[derive(Debug)]
pub struct CsrMatrix<T> {
    width: usize,

    /// A list of `(col, coefficient)` pairs.
    nonzero_values: Vec<(usize, T)>,

    /// Indices of `nonzero_values`. The `i`th index here indicates the first index belonging to the
    /// `i`th row.
    row_indices: Vec<usize>,
}

impl<T> CsrMatrix<T> {
    fn row_index_range(&self, r: usize) -> Range<usize> {
        debug_assert!(r < self.height());
        self.row_indices[r]..self.row_indices[r + 1]
    }

    #[must_use]
    pub fn row(&self, r: usize) -> &[(usize, T)] {
        &self.nonzero_values[self.row_index_range(r)]
    }

    pub fn row_mut(&mut self, r: usize) -> &mut [(usize, T)] {
        let range = self.row_index_range(r);
        &mut self.nonzero_values[range]
    }

    pub fn rand_fixed_row_weight<R: Rng>(
        rng: &mut R,
        rows: usize,
        cols: usize,
        row_weight: usize,
    ) -> Self
    where
        Standard: Distribution<T>,
    {
        let nonzero_values = iter::repeat_with(|| (rng.gen_range(0..cols), rng.gen()))
            .take(rows * row_weight)
            .collect();
        let row_indices = (0..=rows).map(|r| r * row_weight).collect();
        Self {
            width: cols,
            nonzero_values,
            row_indices,
        }
    }
}

impl<T> Matrix<T> for CsrMatrix<T> {
    fn width(&self) -> usize {
        self.width
    }

    fn height(&self) -> usize {
        self.row_indices.len() - 1
    }
}
