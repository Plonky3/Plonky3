//! A minimal STARK framework.

#![no_std]

extern crate alloc;

mod config;
mod folder;
mod prover;
mod sym_var;

pub use folder::*;
pub use config::*;
pub use prover::*;
pub use sym_var::*;
