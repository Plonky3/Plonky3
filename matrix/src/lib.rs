//! Matrix library.

#![no_std]

extern crate alloc;

use alloc::boxed::Box;

pub mod dense;
pub mod mul;
pub mod sparse;
pub mod stack;

pub trait Matrix<T> {
    fn width(&self) -> usize;
    fn height(&self) -> usize;

    fn row(&self, r: usize) -> &[T];

    fn get(&self, r: usize, c: usize) -> T
    where
        T: Copy,
    {
        self.row(r)[c]
    }
}

impl<T> Matrix<T> for Box<dyn Matrix<T>> {
    fn width(&self) -> usize {
        self.as_ref().width()
    }

    fn height(&self) -> usize {
        self.as_ref().height()
    }

    fn row(&self, r: usize) -> &[T] {
        self.as_ref().row(r)
    }
}
