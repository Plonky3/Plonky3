use alloc::vec;
use alloc::vec::Vec;
use core::ops::Mul;
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

    pub fn apply<Exp, Var>(&self, row: &[Var]) -> Exp
    where
        F: Into<Exp>,
        Exp: AbstractField + Mul<F, Output = Exp>,
        Var: Into<Exp> + Copy,
    {
        let mut result = self.constant.into();
        for (column, weight) in self.column_weights.iter() {
            result += row[*column].into() * *weight;
        }
        result
    }
}
