use p3_field::Field;

use crate::circuit_builder::gates::event::AllEvents;

pub trait FillableCollumns {
    type Event;
    /// Fills the columns with the provided events.
    fn fill<F: Field>(&mut self, events: AllEvents<F>);

    /// Returns the number of columns that this trait can fill.
    fn num_columns(&self) -> usize;
}
