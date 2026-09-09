//! Multilinear SuperSpartan-flavored STARK prover for AIRs.
//!
//! [`prove_with_security`] and [`verify_with_security`] enforce a requested bound
//! for the complete statement. They compose AIR, lookup, and commitment-opening
//! errors and require explicit hash-security evidence from the configuration.
//! [`prove`] and [`verify`] enforce no minimum security level. A PCS's own target
//! and the `ExtensionField` trait bound alone do not establish the full proof's
//! security; see [`security_report`] for the assumptions and labeled contributions.
//!
//! # References
//!
//! - Setty, Thaler, Wahby. Customizable Constraint Systems for succinct arguments. <https://eprint.iacr.org/2023/552.pdf>
//! - Borgeaud, W. AIR-specific optimizations on top of SuperSpartan. <https://solvable.group/posts/super-air/>

#![no_std]

extern crate alloc;

pub mod config;
pub mod folder;
pub mod fractional_gkr;
pub mod instance;
pub mod keys;
pub mod lookup;
pub mod opening;
pub mod packed_ext;
pub mod proof;
pub mod prover;
pub mod rounds;
pub mod security;
pub mod selectors;
pub mod transcript;
pub mod verifier;
pub mod zerocheck;

pub use instance::{ProverInstance, ProverInstances, VerifierInstance, VerifierInstances};
pub use keys::{ProvingKey, VerifyingKey, setup};
pub use proof::MultiStarkProof;
pub use prover::{prove, prove_with_security};
pub use security::{MultiStarkSecurityReport, SecurityError, security_report};
pub use verifier::{VerificationError, verify, verify_with_security};
