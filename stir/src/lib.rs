//! An implementation of the STIR low-degree test (LDT).
//! https://eprint.iacr.org/2024/390.

// NP TODO re-introduce no_std
// #![no_std]

// NP TODOs
// - Credit Giacomo and link to his code
// - Think about MMCS

extern crate alloc;

mod config;

// mod fold_even_odd;
mod proof;
// pub mod prover;
// pub mod verifier;

pub use config::*;
// pub use fold_even_odd::*;
// pub use proof::*;
// pub use two_adic_pcs::*;
