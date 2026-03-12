//! APIs for AIRs, and generalizations like PAIRs.

#![no_std]

extern crate alloc;

mod air;
mod check_constraints;
mod named;
pub mod symbolic;
pub mod utils;
mod virtual_column;

pub use air::*;
pub use check_constraints::*;
pub use named::*;
pub use symbolic::*;
pub use virtual_column::*;
