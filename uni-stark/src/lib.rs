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
mod symbolic;
mod verifier;

pub use check_constraints::*;
pub use config::*;
pub use folder::*;
pub use preprocessed::*;
pub use proof::*;
pub use prover::*;
pub use sub_builder::*;
pub use symbolic::*;
pub use verifier::*;
