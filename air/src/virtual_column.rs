use alloc::vec;
use alloc::vec::Vec;
use p3_field::field::{AbstractField, Field};

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

    pub fn apply<Exp: AbstractField<F>, Var>(&self, row: &[Var]) -> Exp
    where
        Var: Into<Exp> + Copy,
    {
        let mut result = Exp::from(self.constant);
        for (column, weight) in self.column_weights.iter() {
            result += row[*column].into() * *weight;
        }
        result
    }
}
