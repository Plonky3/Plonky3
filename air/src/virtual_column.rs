use alloc::vec;
use alloc::vec::Vec;
use p3_field::field::{Field, FieldLike};

/// An affine function over columns.
pub struct VirtualColumn<F: Field> {
    column_weights: Vec<(usize, F)>,
    constant: F,
}

impl<F: Field> VirtualColumn<F> {
    pub fn single(column: usize) -> Self {
        Self {
            column_weights: vec![(column, F::ONE)],
            constant: F::ZERO,
        }
    }

    pub fn apply<FL: FieldLike<F>, V>(&self, row: &[V]) -> FL
    where
        V: Into<FL> + Copy,
    {
        let mut result = FL::from(self.constant);
        for (column, weight) in self.column_weights.iter() {
            result += row[*column].into() * *weight;
        }
        result
    }
}
