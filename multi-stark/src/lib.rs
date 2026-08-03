//! Multilinear SuperSpartan-flavored STARK prover for AIRs.
//!
//! # References
//!
//! - Setty, Thaler, Wahby. Customizable Constraint Systems for succinct arguments. <https://eprint.iacr.org/2023/552.pdf>
//! - Borgeaud, W. AIR-specific optimizations on top of SuperSpartan. <https://solvable.group/posts/super-air/>

#![no_std]

extern crate alloc;

pub mod config;
pub mod folder;
pub mod instance;
pub mod keys;
pub mod metadata;
pub mod opening;
pub mod packed_ext;
pub mod proof;
pub mod prover;
pub mod rounds;
pub mod selectors;
pub mod verifier;
pub mod zerocheck;

pub use instance::{ProverInstance, ProverInstances, VerifierInstance, VerifierInstances};
pub use keys::{ProvingKey, VerifyingKey, setup};
pub use proof::MultiStarkProof;
pub use prover::prove;
pub use verifier::{VerificationError, verify};
