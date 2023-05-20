use alloc::vec;
use alloc::vec::Vec;
use core::ops::Mul;
use p3_field::{AbstractField, Field};

/// An affine function over columns in a PAIR.
pub struct VirtualPairCol<F: Field> {
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

impl<F: Field> VirtualPairCol<F> {
    pub fn single(column: PairCol) -> Self {
        Self {
            column_weights: vec![(column, F::ONE)],
            constant: F::ZERO,
        }
    }

    pub fn single_preprocessed(column: usize) -> Self {
        Self::single(PairCol::Preprocessed(column))
    }

    pub fn single_main(column: usize) -> Self {
        Self::single(PairCol::Main(column))
    }

    pub fn apply<Exp, Var>(&self, preprocessed: &[Var], main: &[Var]) -> Exp
    where
        F: Into<Exp>,
        Exp: AbstractField + Mul<F, Output = Exp>,
        Var: Into<Exp> + Copy,
    {
        let mut result = self.constant.into();
        for (column, weight) in self.column_weights.iter() {
            result += column.get(preprocessed, main).into() * *weight;
        }
        result
    }
}
