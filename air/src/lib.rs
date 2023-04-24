//! APIs for AIRs, and generalizations like PAIRs.

#![no_std]
#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

extern crate alloc;

pub mod constraint_consumer;
pub mod cumulative_product;
pub mod symbolic;
pub mod types;
pub mod virtual_column;
pub mod window;

use crate::constraint_consumer::ConstraintConsumer;
use crate::types::AirTypes;
use crate::window::AirWindow;

/// An AIR.
pub trait Air<T, W>
where
    T: AirTypes,
    W: AirWindow<T>,
{
    fn eval<CC>(&self, constraints: &mut CC)
    where
        CC: ConstraintConsumer<T, W>;
}

#[cfg(test)]
mod tests {
    use crate::window::AirWindow;
    use crate::{Air, AirTypes, ConstraintConsumer};
    use p3_field::field::Field;
    use p3_matrix::Matrix;
    use p3_mersenne_31::Mersenne31;

    struct FibonacciAir;

    impl<T, W> Air<T, W> for FibonacciAir
    where
        T: AirTypes<F = Mersenne31>,
        W: AirWindow<T>,
    {
        fn eval<CC>(&self, constraints: &mut CC)
        where
            CC: ConstraintConsumer<T, W>,
        {
            let main = constraints.window().main();
            let x_0 = main.row(0)[0];
            let x_1 = main.row(1)[0];
            let x_2 = main.row(2)[0];

            let second_row = T::F::ZERO; // TODO
            constraints.when_first_row().assert_zero(x_0);
            constraints.when(second_row).assert_one(x_1);
            constraints.when_transition().assert_eq(x_0 + x_1, x_2);
        }
    }
}
