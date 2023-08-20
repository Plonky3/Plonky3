//! A minimal univariate STARK framework.

#![no_std]

extern crate alloc;

mod config;
mod folder;
mod prover;
mod sym_var;
mod zerofier_coset;

pub use config::*;
pub use folder::*;
pub use prover::*;
pub use sym_var::*;
pub use zerofier_coset::*;
