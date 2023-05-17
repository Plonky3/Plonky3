//! A minimal STARK framework.

#![no_std]
#![feature(associated_type_bounds)]

extern crate alloc;

mod builder;
mod config;
mod prover;
mod sym_var;

pub use builder::*;
pub use config::*;
pub use prover::*;
pub use sym_var::*;
