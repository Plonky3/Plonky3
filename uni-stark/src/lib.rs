//! A minimal univariate STARK framework.

#![no_std]

extern crate alloc;

mod check_constraints;
mod config;
mod folder;
mod preprocessed;
mod proof;
mod prover;
mod sub_builder;
mod symbolic_builder;
mod verifier;

pub use check_constraints::*;
pub use config::*;
pub use folder::*;
// Public re-exports from p3-air.
pub use p3_air::symbolic::*;
pub use preprocessed::*;
pub use proof::*;
pub use prover::*;
pub use sub_builder::*;
pub use symbolic_builder::*;
pub use verifier::*;
