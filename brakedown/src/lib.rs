//! This crate contains an implementation of the Spielman-based code described in the Brakedown paper.

#![no_std]

extern crate alloc;

mod brakedown_code;
mod macros;
mod standard_fast;

pub use brakedown_code::*;
pub use standard_fast::*;
