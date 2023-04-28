use crate::{AirTypes, AirWindow};
use alloc::vec::Vec;
use core::marker::PhantomData;
use p3_field::field::Field;

pub trait ConstraintConsumer<T, W>
where
    T: AirTypes,
    W: AirWindow<T>,
{
    fn window(&self) -> &W;

    /// Returns a constraint consumer whose constraints are enforced iff `filter` is nonzero.
    fn when<I: Into<T::Exp>>(&mut self, filter: I) -> FilteredConstraintConsumer<T, W, Self> {
        FilteredConstraintConsumer {
            inner: self,
            filter: filter.into(),
            _phantom_w: Default::default(),
        }
    }

    /// Returns a constraint consumer whose constraints are enforced iff `x != y`.
    fn when_ne<I: Into<T::Exp>>(&mut self, x: I, y: I) -> FilteredConstraintConsumer<T, W, Self> {
        self.when(x.into() - y.into())
    }

    /// Returns a constraint consumer whose constraints are enforced only on the first row.
    fn when_first_row(&mut self) -> FilteredConstraintConsumer<T, W, Self> {
        self.when(self.window().is_first_row())
    }

    /// Returns a constraint consumer whose constraints are enforced only on the last row.
    fn when_last_row(&mut self) -> FilteredConstraintConsumer<T, W, Self> {
        self.when(self.window().is_last_row())
    }

    /// Returns a constraint consumer whose constraints are enforced on all rows except the last.
    /// This corresponds to transition constraints when using two-row windows.
    fn when_transition(&mut self) -> FilteredConstraintConsumer<T, W, Self> {
        self.when(self.window().is_transition())
    }

    fn assert_zero<I: Into<T::Exp>>(&mut self, constraint: I);

    fn assert_one<I: Into<T::Exp>>(&mut self, constraint: I) {
        let constraint: T::Exp = constraint.into();
        self.assert_zero::<T::Exp>(constraint - T::F::ONE);
    }

    fn assert_eq<I1: Into<T::Exp>, I2: Into<T::Exp>>(&mut self, x: I1, y: I2) {
        self.assert_zero(x.into() - y.into());
    }

    /// Assert that `x` is a boolean value, i.e. 0 or 1.
    fn assert_bool<I: Into<T::Exp>>(&mut self, x: I) {
        let constraint: T::Exp = constraint.into();
        self.assert_zero::<T::Exp>(constraint.clone() * (constraint - T::F::ONE));
    }
}

// TODO: Remove? Meant as an example.
pub trait OddEvenConstraintConsumer<T, W>: ConstraintConsumer<T, W>
where
    T: AirTypes,
    W: AirWindow<T>,
{
    fn when_even_row(&mut self) -> FilteredConstraintConsumer<T, W, Self>;
    fn when_odd_row(&mut self) -> FilteredConstraintConsumer<T, W, Self>;
}

pub struct ConstraintCollector<T: AirTypes, W: AirWindow<T>> {
    pub window: W,
    pub constraints: Vec<T::Exp>,
}

impl<T, W> ConstraintConsumer<T, W> for ConstraintCollector<T, W>
where
    T: AirTypes,
    W: AirWindow<T>,
{
    fn window(&self) -> &W {
        &self.window
    }

    fn assert_zero<I: Into<T::Exp>>(&mut self, constraint: I) {
        self.constraints.push(constraint.into());
    }
}

pub struct FilteredConstraintConsumer<'a, T, W, CC>
where
    T: AirTypes,
    W: AirWindow<T>,
    CC: ConstraintConsumer<T, W> + ?Sized,
{
    inner: &'a mut CC,
    filter: T::Exp,
    _phantom_w: PhantomData<W>,
}

impl<'a, T, W, CC> ConstraintConsumer<T, W> for FilteredConstraintConsumer<'a, T, W, CC>
where
    T: AirTypes,
    W: AirWindow<T>,
    CC: ConstraintConsumer<T, W> + ?Sized,
{
    fn window(&self) -> &W {
        self.inner.window()
    }

    fn assert_zero<I: Into<T::Exp>>(&mut self, constraint: I) {
        self.inner
            .assert_zero(self.filter.clone() * constraint.into());
    }
}
