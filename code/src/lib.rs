//! A framework for codes (in the coding theory sense).

#![no_std]

extern crate alloc;

mod code;
mod identity;
mod registry;
mod systematic;

pub use code::*;
pub use identity::*;
pub use registry::*;
pub use systematic::*;
