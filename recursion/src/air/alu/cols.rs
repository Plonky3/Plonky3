use std::borrow::{Borrow, BorrowMut};
use std::marker::PhantomData;

use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;

use crate::air::AluAir;
use crate::circuit_builder::gates::event::{AllEvents, Table};

pub struct AddEvent<F, const REPETITIONS: usize = 1>(pub FieldOpEvent<F, REPETITIONS>);
pub struct SubEvent<F, const REPETITIONS: usize = 1>(pub FieldOpEvent<F>);
pub struct MulEvent<F, const REPETITIONS: usize = 1>(pub FieldOpEvent<F>);

/// Represents an event in the field operation trace.
pub struct FieldOpEvent<T, const REPETITIONS: usize = 1> {
    pub left_addr: [usize; REPETITIONS],
    pub left_val: [T; REPETITIONS],
    pub right_addr: [usize; REPETITIONS],
    pub right_val: [T; REPETITIONS],
    pub res_addr: [usize; REPETITIONS],
    pub res_val: [T; REPETITIONS],
}

impl<F: Field, const REPETITIONS: usize> FieldOpEvent<F, REPETITIONS> {
    fn generate_trace<'a, I: Iterator<Item = &'a FieldOpEvent<F, REPETITIONS>>>(
        events: I,
        events_len: usize,
    ) -> RowMajorMatrix<F> {
        let n_padded = events_len.next_power_of_two();
        let mut trace = RowMajorMatrix::new(
            F::zero_vec(n_padded * AluAir::<REPETITIONS>::TRACE_WIDTH),
            AluAir::<REPETITIONS>::TRACE_WIDTH,
        );

        let (prefix, rows, suffix) =
            unsafe { trace.values.align_to_mut::<AluCols<F, REPETITIONS>>() };
        assert!(prefix.is_empty(), "Alignment should match");
        assert!(suffix.is_empty(), "Alignment should match");
        assert_eq!(rows.len(), events_len);

        for event in events {
            for i in 0..REPETITIONS {
                let row = &mut rows[i];
                row.left_addr[i] = F::from_usize(event.left_addr[i]);
                row.left_val[i] = event.left_val[i];
                row.right_addr[i] = F::from_usize(event.right_addr[i]);
                row.right_val[i] = event.right_val[i];
                row.res_addr[i] = F::from_usize(event.res_addr[i]);
                row.res_val[i] = event.res_val[i];
            }
        }
        trace
    }
}

pub struct AddTable;
impl<F: Field> Table<F> for AddTable {
    fn generate_trace(&self, all_events: &AllEvents<F>) -> RowMajorMatrix<F> {
        FieldOpEvent::generate_trace(
            all_events.add_events.iter().map(|x| &x.0),
            all_events.add_events.len(),
        )
    }
}

#[derive(PartialEq, Eq, Hash)]
pub struct SubTable;
impl<F: Field> Table<F> for SubTable {
    fn generate_trace(&self, all_events: &AllEvents<F>) -> RowMajorMatrix<F> {
        FieldOpEvent::generate_trace(
            all_events.sub_events.iter().map(|x| &x.0),
            all_events.sub_events.len(),
        )
    }
}

#[derive(PartialEq, Eq, Hash)]
pub struct MulTable;
impl<F: Field> Table<F> for MulTable {
    fn generate_trace(&self, all_events: &AllEvents<F>) -> RowMajorMatrix<F> {
        FieldOpEvent::generate_trace(
            all_events.mul_events.iter().map(|x| &x.0),
            all_events.mul_events.len(),
        )
    }
}

#[repr(C)]
/// Represents the columns in the ALU trace.
/// REPETITIONS counts how many `a * b = c` operations to do per row in the AIR
pub struct AluCols<F, const REPETITIONS: usize = 1> {
    pub left_addr: [F; REPETITIONS],
    pub left_val: [F; REPETITIONS],
    pub right_addr: [F; REPETITIONS],
    pub right_val: [F; REPETITIONS],
    pub res_addr: [F; REPETITIONS],
    pub res_val: [F; REPETITIONS],
    _phantom_data: PhantomData<F>,
}

impl<F, const REPETITIONS: usize> AluCols<F, REPETITIONS> {
    pub const TRACE_WIDTH: usize = 6 * REPETITIONS;
}

impl<F, const REPETITIONS: usize> Borrow<AluCols<F, REPETITIONS>> for [F] {
    fn borrow(&self) -> &AluCols<F, REPETITIONS> {
        debug_assert_eq!(self.len(), AluCols::<F, REPETITIONS>::TRACE_WIDTH);
        let (prefix, shorts, _suffix) = unsafe { self.align_to::<AluCols<F, REPETITIONS>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &shorts[0]
    }
}

impl<F: Field, const REPETITIONS: usize> BorrowMut<AluCols<F, REPETITIONS>> for [F] {
    fn borrow_mut(&mut self) -> &mut AluCols<F, REPETITIONS> {
        debug_assert_eq!(self.len(), AluCols::<F, REPETITIONS>::TRACE_WIDTH);
        let (prefix, shorts, _suffix) = unsafe { self.align_to_mut::<AluCols<F, REPETITIONS>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &mut shorts[0]
    }
}
