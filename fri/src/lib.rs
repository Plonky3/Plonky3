//! An implementation of the FRI low-degree test (LDT).

#![no_std]

extern crate alloc;

mod config;
mod fold_even_odd;
mod proof;
pub mod prover;
mod two_adic_pcs;
pub mod verifier;

pub use config::*;
pub use fold_even_odd::*;
pub use proof::*;
pub use two_adic_pcs::*;
