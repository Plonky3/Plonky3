//! APIs for AIRs, and generalizations like PAIRs.

#![no_std]

extern crate alloc;

pub mod cumulative_product;
pub mod two_row_matrix;
pub mod virtual_column;

use core::ops::{Add, Mul, Sub};
use p3_field::field::{AbstractField, Field};
use p3_matrix::Matrix;

pub trait Air<AB: AirBuilder>: Sync {
    fn eval(&self, builder: &mut AB);
}

pub trait AirBuilder: Sized {
    type F: Field;

    type Exp: AbstractField<Self::F>
        + Add<Self::Var, Output = Self::Exp>
        + Sub<Self::Var, Output = Self::Exp>
        + Mul<Self::Var, Output = Self::Exp>;

    type Var: Into<Self::Exp>
        + Copy
        + Add<Self::Var, Output = Self::Exp>
        + Add<Self::F, Output = Self::Exp>
        + Add<Self::Exp, Output = Self::Exp>
        + Sub<Self::Var, Output = Self::Exp>
        + Sub<Self::F, Output = Self::Exp>
        + Sub<Self::Exp, Output = Self::Exp>
        + Mul<Self::Var, Output = Self::Exp>
        + Mul<Self::F, Output = Self::Exp>
        + Mul<Self::Exp, Output = Self::Exp>;

    type M: Matrix<Self::Var>;

    fn main(&self) -> Self::M;

    fn is_first_row(&self) -> Self::Exp;
    fn is_last_row(&self) -> Self::Exp;
    fn is_transition(&self) -> Self::Exp {
        self.is_transition_window(2)
    }
    fn is_transition_window(&self, size: usize) -> Self::Exp;

    fn when<I: Into<Self::Exp>>(&mut self, condition: I) -> FilteredAirBuilder<Self> {
        FilteredAirBuilder {
            inner: self,
            condition: condition.into(),
        }
    }

    fn when_first_row(&mut self) -> FilteredAirBuilder<Self> {
        self.when(self.is_first_row())
    }

    fn when_last_row(&mut self) -> FilteredAirBuilder<Self> {
        self.when(self.is_last_row())
    }

    fn when_transition(&mut self) -> FilteredAirBuilder<Self> {
        self.when(self.is_transition())
    }

    fn assert_zero<I: Into<Self::Exp>>(&mut self, x: I);

    fn assert_one<I: Into<Self::Exp>>(&mut self, x: I) {
        self.assert_zero(x.into() - Self::Exp::ONE);
    }

    fn assert_eq<I1: Into<Self::Exp>, I2: Into<Self::Exp>>(&mut self, x: I1, y: I2) {
        self.assert_zero(x.into() - y.into());
    }

    /// Assert that `x` is a boolean, i.e. either 0 or 1.
    fn assert_bool<I: Into<Self::Exp>>(&mut self, x: I) {
        let x = x.into();
        self.assert_zero(x.clone() * (x - Self::Exp::ONE));
    }
}

pub trait PairBuilder: AirBuilder {
    fn preprocessed(&self) -> Self::M;
}

pub trait PermutationAirBuilder: AirBuilder {
    fn permutation(&self) -> Self::M;

    fn permutation_randomness(&self) -> Self::Exp;
}

pub struct FilteredAirBuilder<'a, AB: AirBuilder> {
    inner: &'a mut AB,
    condition: AB::Exp,
}

impl<'a, AB: AirBuilder> AirBuilder for FilteredAirBuilder<'a, AB> {
    type F = AB::F;
    type Exp = AB::Exp;
    type Var = AB::Var;
    type M = AB::M;

    fn main(&self) -> Self::M {
        self.inner.main()
    }

    fn is_first_row(&self) -> Self::Exp {
        self.inner.is_first_row()
    }

    fn is_last_row(&self) -> Self::Exp {
        self.inner.is_last_row()
    }

    fn is_transition_window(&self, size: usize) -> Self::Exp {
        self.inner.is_transition_window(size)
    }

    fn assert_zero<I: Into<Self::Exp>>(&mut self, x: I) {
        self.inner.assert_zero(self.condition.clone() * x.into())
    }
}

#[cfg(test)]
mod tests {
    use crate::{Air, AirBuilder};
    use p3_matrix::Matrix;

    struct FibonacciAir;

    impl<AB: AirBuilder> Air<AB> for FibonacciAir {
        fn eval(&self, builder: &mut AB) {
            let main = builder.main();

            let x_0 = main.row(0)[0];
            let x_1 = main.row(1)[0];
            let x_2 = main.row(2)[0];

            builder.when_first_row().assert_zero(x_0);
            builder.when_first_row().assert_one(x_1);
            builder.when_transition().assert_eq(x_0 + x_1, x_2);
        }
    }
}
