//! APIs for AIRs, and generalizations like PAIRs.

#![no_std]
#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

extern crate alloc;

pub mod cumulative_product;
pub mod two_row_matrix;
pub mod virtual_column;

use core::ops::{Add, Mul, Sub};
use p3_field::field::{Field, FieldLike};
use p3_matrix::Matrix;

pub trait Air<AB: AirBuilder>: Sync {
    fn eval(&self, builder: &mut AB);
}

pub trait AirBuilder: Sized
where
    Self::FL: Add<Self::Var, Output = Self::FL>
        + Sub<Self::Var, Output = Self::FL>
        + Mul<Self::Var, Output = Self::FL>,
{
    type F: Field;
    type FL: FieldLike<Self::F>;

    type Var: Into<Self::FL>
        + Copy
        + Add<Self::Var, Output = Self::FL>
        + Add<Self::F, Output = Self::FL>
        + Add<Self::FL, Output = Self::FL>
        + Sub<Self::Var, Output = Self::FL>
        + Sub<Self::F, Output = Self::FL>
        + Sub<Self::FL, Output = Self::FL>
        + Mul<Self::Var, Output = Self::FL>
        + Mul<Self::F, Output = Self::FL>
        + Mul<Self::FL, Output = Self::FL>;

    type M: Matrix<Self::Var>;

    fn main(&self) -> Self::M;

    fn is_first_row(&self) -> Self::FL;
    fn is_last_row(&self) -> Self::FL;
    fn is_transition(&self) -> Self::FL {
        self.is_transition_window(2)
    }
    fn is_transition_window(&self, size: usize) -> Self::FL;

    fn when<I: Into<Self::FL>>(&mut self, condition: I) -> FilteredAirBuilder<Self> {
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

    fn assert_zero<I: Into<Self::FL>>(&mut self, x: I);

    fn assert_one<I: Into<Self::FL>>(&mut self, x: I) {
        self.assert_zero(x.into() - Self::FL::ONE);
    }

    fn assert_eq<I1: Into<Self::FL>, I2: Into<Self::FL>>(&mut self, x: I1, y: I2) {
        self.assert_zero(x.into() - y.into());
    }

    /// Assert that `x` is a boolean, i.e. either 0 or 1.
    fn assert_bool<I: Into<Self::FL>>(&mut self, x: I) {
        let x = x.into();
        self.assert_zero(x.clone() * (x - Self::FL::ONE));
    }
}

pub trait PairBuilder: AirBuilder {
    fn preprocessed(&self) -> Self::M;
}

pub trait PermutationAirBuilder: AirBuilder {
    fn permutation(&self) -> Self::M;

    fn permutation_randomness(&self) -> Self::FL;
}

pub struct FilteredAirBuilder<'a, AB: AirBuilder> {
    inner: &'a mut AB,
    condition: AB::FL,
}

impl<'a, AB: AirBuilder> AirBuilder for FilteredAirBuilder<'a, AB> {
    type F = AB::F;
    type FL = AB::FL;
    type Var = AB::Var;
    type M = AB::M;

    fn main(&self) -> Self::M {
        self.inner.main()
    }

    fn is_first_row(&self) -> Self::FL {
        self.inner.is_first_row()
    }

    fn is_last_row(&self) -> Self::FL {
        self.inner.is_last_row()
    }

    fn is_transition_window(&self, size: usize) -> Self::FL {
        self.inner.is_transition_window(size)
    }

    fn assert_zero<I: Into<Self::FL>>(&mut self, x: I) {
        self.inner.assert_zero(self.condition.clone() * x.into())
    }
}

#[cfg(test)]
mod tests {
    use crate::{Air, AirBuilder, PairBuilder};
    use p3_matrix::Matrix;
    use p3_mersenne_31::Mersenne31;

    struct FibonacciAir;

    impl<AB: AirBuilder> Air<AB> for FibonacciAir
    where
        AB: PairBuilder<F = Mersenne31>,
    {
        fn eval(&self, builder: &mut AB) {
            builder.preprocessed();
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
