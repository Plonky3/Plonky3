//! A minimal STARK framework.

#![no_std]

use p3_air::constraint_consumer::{ConstraintCollector, ConstraintConsumer};
use p3_air::symbolic::{Symbolic, SymbolicVar};
use p3_air::window::AirWindow;
use p3_air::Air;
// use p3_commit::pcs::PCS;
use p3_field::field::Field;
use p3_matrix::dense::DenseMatrixView;

pub trait StarkConfig {
    type F: Field;
    // type PCS: PCS<Self::F>;
}

pub struct BasicAirWindow<'a, T> {
    main: DenseMatrixView<'a, T>,
}

impl<'a, T> AirWindow<T> for BasicAirWindow<'a, T> {
    type M = DenseMatrixView<'a, T>;

    fn main(&self) -> &'a Self::M {
        todo!()
    }
}

pub struct FoldingConstraintConsumer;

impl<F: Field> ConstraintConsumer<F> for FoldingConstraintConsumer {
    fn global(&mut self, value: F) {
        todo!()
    }
}

pub fn prove<'a, SC, F, A>(air: &A)
where
    SC: StarkConfig,
    F: Field,
    A: Air<F, BasicAirWindow<'a, F>, FoldingConstraintConsumer>
        + Air<Symbolic<F>, BasicAirWindow<'a, SymbolicVar<F>>, ConstraintCollector<Symbolic<F>>>,
{
    let main = todo!();
    let window = BasicAirWindow { main };
    let mut consumer = FoldingConstraintConsumer;
    air.eval(window, &mut consumer);
}

#[cfg(test)]
mod tests {
    use crate::{prove, StarkConfig};
    use p3_air::constraint_consumer::ConstraintConsumer;
    use p3_air::types::AirTypes;
    use p3_air::window::AirWindow;
    use p3_air::Air;
    use p3_matrix::Matrix;
    use p3_mersenne_31::Mersenne31;

    struct MyConfig;

    impl StarkConfig for MyConfig {
        type F = Mersenne31;
        // type PCS = todo!();
    }

    struct MulAir;

    impl<'a, T, W, CC> Air<T, W, CC> for MulAir
    where
        T: AirTypes<F = Mersenne31>,
        W: AirWindow<T::Var>,
        CC: ConstraintConsumer<T>,
    {
        fn eval(&self, window: W, constraints: &mut CC) {
            let main_local = window.main().row(0);
            let diff = main_local[0] * main_local[1] - main_local[2];
            constraints.global(diff);
        }
    }

    #[test]
    fn test_prove() {
        prove::<MyConfig, Mersenne31, MulAir>(&MulAir)
    }
}
