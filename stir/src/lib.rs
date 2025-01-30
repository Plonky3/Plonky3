//! An implementation of the STIR low-degree test (LDT).
//! https://eprint.iacr.org/2024/390.

// NP TODO re-introduce no_std
// #![no_std]

// NP TODOs
// - Credit Giacomo and link to his code
// - Think about MMCS
// - Batching (fold multiple words)
// - Protocol builder

// NP TODO profusely add documentation

extern crate alloc;

mod config;
pub mod coset;
mod polynomial;
mod proof;
pub mod prover;
mod proximity_gaps;
mod utils;
pub mod verifier;

#[cfg(test)]
pub mod test_utils;

pub use config::{StirConfig, StirParameters};
pub use proof::StirProof;
pub use proximity_gaps::*;
