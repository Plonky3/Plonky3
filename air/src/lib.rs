//! APIs for AIRs, and generalizations like PAIRs.

#![no_std]

extern crate alloc;

mod air;
mod air_claims;
mod check_constraints;
pub mod symbolic;
pub mod utils;
mod virtual_column;

pub use air::*;
pub use air_claims::*;
pub use check_constraints::*;
pub use symbolic::*;
pub use virtual_column::*;
