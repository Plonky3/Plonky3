use alloc::vec;
use alloc::vec::Vec;
use core::ops::Mul;
use p3_field::{AbstractField, Field};

/// An affine function over columns in a PAIR.
pub struct VirtualPairCol<F: AbstractField> {
    column_weights: Vec<(PairCol, F)>,
    constant: F,
}

/// A column in a PAIR, i.e. either a preprocessed column or a main trace column.
pub enum PairCol {
    Preprocessed(usize),
    Main(usize),
}

impl PairCol {
    fn get<T: Copy>(&self, preprocessed: &[T], main: &[T]) -> T {
        match self {
            PairCol::Preprocessed(i) => preprocessed[*i],
            PairCol::Main(i) => main[*i],
        }
    }
}

impl<F: AbstractField> VirtualPairCol<F> {
    #[must_use]
    pub fn one() -> Self {
        Self::constant(F::ONE)
    }

    #[must_use]
    pub fn constant(x: F) -> Self {
        Self {
            column_weights: vec![],
            constant: x,
        }
    }

    #[must_use]
    pub fn single(column: PairCol) -> Self {
        Self {
            column_weights: vec![(column, F::ONE)],
            constant: F::ZERO,
        }
    }

    #[must_use]
    pub fn single_preprocessed(column: usize) -> Self {
        Self::single(PairCol::Preprocessed(column))
    }

    #[must_use]
    pub fn single_main(column: usize) -> Self {
        Self::single(PairCol::Main(column))
    }

    pub fn apply<Expr, Var>(&self, preprocessed: &[Var], main: &[Var]) -> Expr
    where
        F: Into<Expr>,
        Expr: AbstractField + Mul<F, Output = Expr>,
        Var: Into<Expr> + Copy,
    {
        let mut result = self.constant.clone().into();
        for (column, weight) in &self.column_weights {
            result += column.get(preprocessed, main).into() * weight.clone();
        }
        result
    }
}
