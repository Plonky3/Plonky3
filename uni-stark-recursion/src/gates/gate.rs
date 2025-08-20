use p3_field::Field;

use crate::circuit_builder::{CircuitBuilder, CircuitError};

pub trait Gate<F: Field> {
    fn n_inputs(&self) -> usize;
    fn n_outputs(&self) -> usize;

    fn generate(&self, builder: &mut CircuitBuilder<F>) -> Result<(), CircuitError>;

    fn check_shape(&self, n_inputs: usize, n_outputs: usize) {
        assert_eq!(n_inputs, self.n_inputs());
        assert_eq!(n_outputs, self.n_outputs());
    }
}
