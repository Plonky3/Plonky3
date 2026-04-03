//! A framework for various (not necessarily hiding) cryptographic commitment schemes.

#![no_std]

extern crate alloc;

mod adapters;
mod domain;
mod mmcs;
mod pcs;
mod periodic;
pub mod soundness;

#[cfg(any(test, feature = "test-utils"))]
pub mod testing;

pub use adapters::*;
pub use domain::*;
pub use mmcs::*;
pub use pcs::*;
pub use periodic::*;
pub use soundness::SecurityAssumption;
