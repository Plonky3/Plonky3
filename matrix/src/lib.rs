//! Matrix library.

#![no_std]

extern crate alloc;

pub mod dense;
pub mod mul;
pub mod sparse;

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
