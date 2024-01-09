//! A minimal univariate STARK framework.

#![no_std]

extern crate alloc;

mod config;
mod decompose;
mod folder;
mod proof;
mod prover;
mod symbolic_builder;
mod symbolic_expression;
mod symbolic_variable;
mod verifier;
mod zerofier_coset;

#[cfg(debug_assertions)]
mod check_constraints;

pub use config::*;
pub use decompose::*;
pub use folder::*;
pub use proof::*;
pub use prover::*;
pub use symbolic_variable::*;
pub use verifier::*;
pub use zerofier_coset::*;
