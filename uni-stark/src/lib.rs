//! A minimal univariate STARK framework.

#![no_std]

extern crate alloc;

mod config;
mod decompose;
mod folder;
mod proof;
mod prover;
mod sym_var;
mod verifier;
mod zerofier_coset;

pub use config::*;
pub use decompose::*;
pub use folder::*;
pub use proof::*;
pub use prover::*;
pub use sym_var::*;
pub use verifier::*;
pub use zerofier_coset::*;
