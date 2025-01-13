//! An implementation of the STIR low-degree test (LDT).
//! https://eprint.iacr.org/2024/390.

// NP TODO re-introduce no_std
// #![no_std]

// NP TODOs
// - Credit Giacomo and link to his code
// - Think about MMCS
// - Batching (fold multiple words)
// - Protocol builder

extern crate alloc;

mod config;
// mod fold_even_odd;
mod coset;
mod polynomial;
mod proof;
pub mod prover;
mod proximity_gaps;
mod utils;
// NP pub mod verifier;

pub use config::{StirConfig, StirParameters};
pub use proof::StirProof;
pub use proximity_gaps::*;

// NP pub use fold_even_odd::*;
// NP pub use proof::*;
// NP pub use two_adic_pcs::*;
