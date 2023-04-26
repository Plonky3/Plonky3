use crate::AirTypes;
use alloc::vec::Vec;
use p3_field::field::Field;

pub trait ConstraintConsumer<T: AirTypes> {
    fn when<I: Into<T::Exp>>(&mut self, filter: I) -> FilteredConstraintConsumer<T, Self> {
        FilteredConstraintConsumer {
            inner: self,
            filter: filter.into(),
        }
    }

    fn assert_zero<I: Into<T::Exp>>(&mut self, constraint: I);

    fn assert_one<I: Into<T::Exp>>(&mut self, constraint: I) {
        let constraint: T::Exp = constraint.into();
        self.assert_zero::<T::Exp>(constraint - T::F::ONE);
    }

    fn assert_eq<I1: Into<T::Exp>, I2: Into<T::Exp>>(&mut self, a: I1, b: I2) {
        self.assert_zero(a.into() - b.into());
    }
}

pub struct ConstraintCollector<T: AirTypes> {
    pub constraints: Vec<T::Exp>,
}

impl<T: AirTypes> ConstraintConsumer<T> for ConstraintCollector<T> {
    fn assert_zero<I: Into<T::Exp>>(&mut self, constraint: I) {
        self.constraints.push(constraint.into());
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
    fn assert_zero<I: Into<T::Exp>>(&mut self, constraint: I) {
        self.inner
            .assert_zero(self.filter.clone() * constraint.into());
    }
}
