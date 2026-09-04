#![doc = include_str!("../README.md")]
#![no_std]

extern crate alloc;

mod error;
mod fold;
mod params;
mod pcs;
mod proof;
mod prover;
#[cfg(test)]
pub(crate) mod test_util;
mod verifier;

pub use error::BinaryPcsError;
pub use fold::{fold_codeword, fold_pair};
pub use params::{BinaryPcsConfig, BinaryPcsConfigError, BinaryPcsParams};
pub use pcs::BinaryPcs;
pub use proof::{BinaryPcsProof, RoundProof};
pub use prover::BinaryPcsProverData;
