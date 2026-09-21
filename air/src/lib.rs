//! APIs for AIRs, and generalizations like PAIRs.

#![no_std]

extern crate alloc;

mod air;
pub mod boundary;
mod builder;
mod check_constraints;
mod filtered;
mod named;
mod periodic;
pub mod symbolic;
pub mod utils;
mod virtual_column;
mod window;

pub use air::*;
pub use boundary::{BoundaryEnd, BoundaryIoError, BoundaryPublic};
pub use builder::*;
pub use check_constraints::*;
pub use filtered::*;
pub use named::*;
pub use periodic::*;
pub use symbolic::*;
pub use virtual_column::*;
pub use window::*;
