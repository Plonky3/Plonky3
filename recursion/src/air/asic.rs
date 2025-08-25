use p3_field::Field;

use crate::circuit_builder::gates::event::Table;

pub struct Asic<F> {
    pub asic: Vec<Box<dyn Table<F>>>,
}

impl<F: Field> Asic<F> {
    pub fn generate_trace(
        &self,
        all_events: &crate::circuit_builder::gates::event::AllEvents<F>,
    ) -> Vec<p3_matrix::dense::RowMajorMatrix<F>> {
        self.asic
            .iter()
            .map(|table| table.generate_trace(all_events))
            .collect()
    }
}
