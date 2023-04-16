use crate::AirTypes;
use alloc::vec::Vec;

pub trait ConstraintConsumer<T: AirTypes> {
    fn when(&mut self, filter: T::Exp) -> FilteredConstraintConsumer<T, Self> {
        FilteredConstraintConsumer {
            inner: self,
            filter,
        }
    }

    fn global(&mut self, value: T::Exp);
    // fn first_row(&mut self, value: T);
    // fn last_row(&mut self, value: T);
    // fn transition(&mut self, value: T);
}

pub struct CollectingConstraintConsumer<T: AirTypes> {
    pub constraints: Vec<T::Exp>,
}

impl<T: AirTypes> ConstraintConsumer<T> for CollectingConstraintConsumer<T> {
    fn global(&mut self, value: T::Exp) {
        self.constraints.push(value);
    }
}

pub struct FilteredConstraintConsumer<'a, T, CC>
where
    T: AirTypes,
    CC: ConstraintConsumer<T> + ?Sized,
{
    inner: &'a mut CC,
    filter: T::Exp,
}

impl<'a, T, CC> ConstraintConsumer<T> for FilteredConstraintConsumer<'a, T, CC>
where
    T: AirTypes,
    CC: ConstraintConsumer<T> + ?Sized,
{
    fn global(&mut self, value: T::Exp) {
        self.inner.global(self.filter.clone() * value);
    }
}
