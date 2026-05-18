//! WHIR polynomial commitment scheme: commit, prove, and verify.

mod adapter;
pub mod code_switch_zk;
pub mod committer;
mod hiding_adapter;
pub mod proof;
pub mod prover;
pub mod utils;
pub mod verifier;

pub use adapter::*;
pub use hiding_adapter::*;

#[cfg(test)]
mod tests;
