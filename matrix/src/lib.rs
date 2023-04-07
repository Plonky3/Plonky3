//! Matrix library.

#![no_std]

extern crate alloc;

pub mod dense;
pub mod mul;
pub mod sparse;

pub trait Matrix<T> {
    fn width(&self) -> usize;
    fn height(&self) -> usize;
}
