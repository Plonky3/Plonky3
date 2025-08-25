use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;

use crate::air::{AddEvent, MulEvent, SubEvent};

#[derive(Default)]
pub struct AllEvents<F: Field> {
    pub add_events: Vec<AddEvent<F>>,
    pub sub_events: Vec<SubEvent<F>>,
    pub mul_events: Vec<MulEvent<F>>,
}

pub trait Table<F: Field> {
    fn generate_trace(&self, all_events: &AllEvents<F>) -> RowMajorMatrix<F>;
}
