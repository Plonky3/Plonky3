//! WHIR polynomial commitment scheme: commit, prove, and verify.

mod adapter;
pub(crate) mod committer;
pub mod proof;
pub mod prover;
pub(crate) mod utils;
pub mod verifier;
pub mod zk;

pub use adapter::WhirProverData;

#[cfg(test)]
mod tests;
