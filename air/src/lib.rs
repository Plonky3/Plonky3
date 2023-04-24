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
pub trait Air<T, W, CC>
where
    T: AirTypes,
    W: AirWindow<T::Var>,
    CC: ConstraintConsumer<T>,
{
    fn eval(&self, window: &W, constraints: &mut CC);
}

#[cfg(test)]
mod tests {
    use crate::window::PairWindow;
    use crate::{Air, AirTypes, ConstraintConsumer};
    use p3_matrix::Matrix;
    use p3_mersenne_31::Mersenne31;

    struct MulAir;

    impl<T, W, CC> Air<T, W, CC> for MulAir
    where
        T: AirTypes<F = Mersenne31>,
        W: PairWindow<T::Var>,
        CC: ConstraintConsumer<T>,
    {
        fn eval(&self, window: &W, constraints: &mut CC) {
            let preprocessed = window.preprocessed();
            let main = window.main();
            let preprocessed_local = preprocessed.row(0);
            let main_local = main.row(0);

            let selector = preprocessed_local[0];
            let diff = main_local[0] * main_local[1] - main_local[2];
            constraints.enforce(selector * diff);
        }
    }
}
